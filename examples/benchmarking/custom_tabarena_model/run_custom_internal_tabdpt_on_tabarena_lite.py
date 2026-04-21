"""Run TabArena-Lite with a local Internal-TabDPT checkpoint.

Example:
    python run_custom_internal_tabdpt_on_tabarena_lite.py \\
        --run-path NoRetrieval/Small_latest \\
        --run-name small_latest

Point ``--run-path`` at a folder inside ``Internal-TabDPT/runs/`` that contains a
``latest.ckpt`` file. Results go to ``./tabarena_out/<run-name>/``; pass the same
``--run-name`` to ``run_evaluate_internal_tabdpt.py`` afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openml
from custom_internal_tabdpt_model import get_configs_for_custom_internal_tabdpt

from tabarena.benchmark.experiment import run_experiments_new

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Internal-TabDPT on TabArena-Lite.")
    p.add_argument(
        "--run-path",
        required=True,
        help="Checkpoint folder under Internal-TabDPT/runs/ (e.g. NoRetrieval/Small_latest).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run name. Used as the subfolder under ./tabarena_out/ and must be passed "
            "(identically) to run_evaluate_internal_tabdpt.py --run-name. "
            "Defaults to a slug of --run-path."
        ),
    )
    p.add_argument(
        "--use-retrieval",
        action="store_true",
        help="Enable per-sample KNN retrieval (sets context_size=1024 unless overridden).",
    )
    p.add_argument("--context-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {"run_path": args.run_path}
    if args.use_retrieval:
        overrides["use_retrieval"] = True
        overrides["context_size"] = args.context_size or 1024
    elif args.context_size is not None:
        overrides["context_size"] = args.context_size

    run_name = args.run_name or args.run_path.replace("/", "_").lower()
    output_dir = HERE / "tabarena_out" / run_name

    print(f"[run] run_path   = {args.run_path}")
    print(f"[run] run_name   = {run_name}")
    print(f"[run] output_dir = {output_dir}")
    print(f"[run] overrides  = {overrides}")

    run_experiments_new(
        output_dir=str(output_dir),
        model_experiments=get_configs_for_custom_internal_tabdpt(config_overrides=overrides),
        tasks=openml.study.get_suite("tabarena-v0.1").tasks,
        repetitions_mode="TabArena-Lite",
    )


if __name__ == "__main__":
    main()
