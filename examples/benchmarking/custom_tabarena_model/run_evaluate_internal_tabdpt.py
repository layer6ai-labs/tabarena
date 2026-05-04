"""Evaluate an Internal-TabDPT TabArena(-Lite) run against the TabArena leaderboard.

Requires that you first ran ``run_custom_internal_tabdpt_on_tabarena_lite.py`` so that
raw results exist under ``./tabarena_out/<run-name>/``. Pass the same ``--run-name`` here.

Writes full figures under ``./evals/<run-name>/`` and an additional
``./evals/<run-name>/plots_default_only/`` folder. That pass uses
``plot_tune_types=["default"]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from custom_internal_tabdpt_model import CustomInternalTabDPTModel
from tabarena.nips2025_utils.compare import compare_on_tabarena
from tabarena.nips2025_utils.end_to_end_single import EndToEndResultsSingle, EndToEndSingle
from tabarena.website.website_format import format_leaderboard

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-name",
        required=True,
        help=(
            "Run name. Must match the --run-name passed to "
            "run_custom_internal_tabdpt_on_tabarena_lite.py; used as the subfolder "
            "under ./tabarena_out/ and ./evals/."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    method = CustomInternalTabDPTModel.ag_name
    path_raw = HERE / "tabarena_out" / args.run_name
    fig_output_dir = HERE / "evals" / args.run_name

    print(f"[evaluate] run_name        = {args.run_name}")
    print(f"[evaluate] method          = {method}")
    print(f"[evaluate] path_raw        = {path_raw}")
    print(f"[evaluate] fig_output_dir  = {fig_output_dir}")

    EndToEndSingle.from_path_raw(path_raw=path_raw).to_results()
    results = EndToEndResultsSingle.from_cache(method=method)
    only_valid_tasks = True
    leaderboard = results.compare_on_tabarena(
        only_valid_tasks=only_valid_tasks,
        output_dir=fig_output_dir,
    )
    print(format_leaderboard(leaderboard).to_markdown(index=False))

    # Same leaderboard math as main run; plot_tune_types also trims times / CDD / Pareto.
    default_only_dir = fig_output_dir / "plots_default_only"
    default_only_dir.mkdir(parents=True, exist_ok=True)
    results_df = results.get_results(fillna=not only_valid_tasks)
    compare_on_tabarena(
        output_dir=default_only_dir,
        new_results=results_df,
        only_valid_tasks=only_valid_tasks,
        plot_tune_types=["default"],
        verbose=False,
    )
    print(f"[evaluate] default-only plots -> {default_only_dir}")


if __name__ == "__main__":
    main()
