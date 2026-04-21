"""AutoGluon wrapper around the local Internal-TabDPT repo.

Must live in its own module because `run_experiments_new` pickles the model class.
Expects a symlink at ``tabarena/third_party/Internal-TabDPT`` pointing to the
Internal-TabDPT checkout; see ``README.md`` in this folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

if TYPE_CHECKING:
    import pandas as pd


INTERNAL_TABDPT_REPO = (
    Path(__file__).resolve().parents[3] / "third_party" / "Internal-TabDPT"
)


def _resolve_checkpoint(repo_path: Path, run_path: str) -> Path:
    ckpt = repo_path / "runs" / run_path / "latest.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _ensure_on_syspath(repo_path: Path) -> None:
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


class CustomInternalTabDPTModel(AbstractTorchModel):
    """Wraps Internal-TabDPT's TabDPTClassifier / TabDPTRegressor as a TabArena model."""

    ag_key = "INT-TABDPT"
    ag_name = "Internal-TabDPT"
    seed_name = "seed"
    default_random_seed = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator: LabelEncoderFeatureGenerator | None = None
        self._predict_hps: dict | None = None

    def _set_default_params(self):
        # Mirrors Internal-TabDPT/eval_full.py::FullEval with use_retrieval=False:
        # context_size >= max n_train in TabArena-Lite, so the whole training set
        # is used as a single shared context on every task.
        # `run_path` is intentionally left unset; callers must supply it.
        defaults = {
            "internal_repo_path": str(INTERNAL_TABDPT_REPO),
            "context_size": 262144,
            "use_retrieval": False,
        }
        for k, v in defaults.items():
            self._set_default_param_value(k, v)

    def _preprocess(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.to_numpy(dtype=np.float32, copy=False)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int = 1,
        **kwargs,
    ):
        from torch.cuda import is_available as cuda_is_available

        hps = self._get_model_params()
        run_path = hps.get("run_path")
        if not run_path:
            raise ValueError(
                "CustomInternalTabDPTModel requires `run_path` (folder under "
                "Internal-TabDPT/runs/ containing `latest.ckpt`)."
            )
        repo_path = Path(hps["internal_repo_path"]).resolve()
        _ensure_on_syspath(repo_path)
        checkpoint = _resolve_checkpoint(repo_path, run_path)

        if num_gpus >= 1 and not cuda_is_available():
            raise AssertionError("Fit requested a GPU but CUDA is not available.")
        device = "cuda:0" if num_gpus >= 1 else "cpu"

        import tabdpt
        from tabdpt import TabDPTClassifier, TabDPTRegressor

        print(f"[internal-tabdpt] tabdpt module : {tabdpt.__file__}")
        print(f"[internal-tabdpt] checkpoint    : {checkpoint}")
        print(f"[internal-tabdpt] device        : {device}")

        is_cls = self.problem_type in (BINARY, MULTICLASS)
        model_cls = TabDPTClassifier if is_cls else TabDPTRegressor
        supported = (
            ("context_size", "temperature", "use_retrieval")
            if is_cls
            else ("context_size", "use_retrieval")
        )
        self._predict_hps = {k: hps[k] for k in supported if k in hps}

        X_np = self.preprocess(X, y=y)
        self.model = model_cls(path=str(checkpoint), device=device)
        self.model.fit(X_np, y.to_numpy())

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X_np = self.preprocess(X, **kwargs)
        if self.problem_type == REGRESSION:
            y_pred = self.model.predict(X_np, **(self._predict_hps or {}))
            return np.asarray(y_pred).reshape(-1)
        proba = self.model.predict_proba(X_np, **(self._predict_hps or {}))
        return self._convert_proba_to_unified_form(proba)

    def get_device(self) -> str:
        if getattr(self, "model", None) is None:
            return "cpu"
        return getattr(self.model, "device", "cpu")

    def _set_device(self, device: str):
        if getattr(self, "model", None) is None:
            return
        self.model.device = device
        if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
            self.model.model.to(device)

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        num_gpus = min(1, ResourceManager.get_gpu_count_torch(cuda_only=True))
        return num_cpus, num_gpus

    def get_minimum_resources(self, is_gpu_available: bool = False) -> dict[str, int | float]:
        return {"num_cpus": 1, "num_gpus": 0.5 if is_gpu_available else 0}

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        out = super()._get_default_ag_args_ensemble(**kwargs)
        out.update({"refit_folds": True})
        return out


def get_configs_for_custom_internal_tabdpt(config_overrides: dict | None = None):
    """Default-only holdout experiments (no HPO, no bagging).

    ``config_overrides`` MUST include ``run_path`` (folder under
    ``Internal-TabDPT/runs/`` containing ``latest.ckpt``).
    """
    from tabarena.utils.config_utils import ConfigGenerator

    if not config_overrides or "run_path" not in config_overrides:
        raise ValueError("config_overrides must include a `run_path` entry.")

    gen = ConfigGenerator(
        model_cls=CustomInternalTabDPTModel,
        manual_configs=[dict(config_overrides)],
        search_space={},
    )
    return gen.generate_all_holdout_experiments(num_random_configs=0)
