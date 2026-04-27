# Internal-TabDPT on TabArena-Lite

Benchmarks a locally-trained Internal-TabDPT checkpoint against the TabArena Lite leaderboard,
using the same flow as `custom_random_forest_model.py`.

## Layout

```
custom_tabarena_model/
├── custom_internal_tabdpt_model.py                   AutoGluon model wrapper
├── run_custom_internal_tabdpt_on_tabarena_lite.py    Runs TabArena-Lite with the wrapper
├── run_evaluate_internal_tabdpt.py                   Builds the leaderboard + figures
├── custom_random_forest_model.py                     Default reference
├── run_custom_model_on_tabarena_lite.py              Default reference
└── run_evaluate_model.py                             Default reference
```

## Setup

### 1. Clone side-by-side and check out the right branches

```
Repo/
├── Internal-TabDPT/      # https://github.com/layer6ai-labs/Internal-TabDPT; check out the branch
│                         # whose `runs/` folder contains the checkpoint you want to test
└── tabarena/             # https://github.com/layer6ai-labs/tabarena (layer6 fork);
                          # check out the `custom_tabdpt` branch
```

```bash
# Internal-TabDPT on the branch that carries your checkpoint
git clone https://github.com/layer6ai-labs/Internal-TabDPT.git
cd Internal-TabDPT
git checkout <ckpt-branch>     # e.g. main, or a feature branch with runs/.../latest.ckpt
cd ..

# TabArena (layer6 fork) on the custom_tabdpt branch where this integration lives
git clone https://github.com/layer6ai-labs/tabarena.git
cd tabarena
git checkout custom_tabdpt
cd ..
```

### 2. Install the TabArena environment

```bash
# Dedicated venv
uv venv --seed --python 3.12 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate

# TabArena (benchmark extra)
cd tabarena
uv pip install --prerelease=allow -e "./tabarena[benchmark]"
```

Optional smoke test that the env is healthy (uses the upstream default CustomRF example; does  
not require Internal-TabDPT yet):

```bash
cd examples/benchmarking/custom_tabarena_model
python run_custom_model_on_tabarena_lite.py
python run_evaluate_model.py
```

### 3. Symlink Internal-TabDPT into TabArena

The wrapper imports `tabdpt` from `tabarena/third_party/Internal-TabDPT` so that local
edits to Internal-TabDPT are picked up without reinstalling:

```bash
cd <tabarena-root>
mkdir -p third_party
ln -s ../../Internal-TabDPT third_party/Internal-TabDPT
```

## Usage

### 1. Drop in a checkpoint

Train a checkpoint under `Internal-TabDPT/runs/<run_path>/latest.ckpt`. Any  
folder name works; you will pass `<run_path>` on the command line in later steps.

### 2. Run Internal-TabDPT on TabArena-Lite Benchmarking Datasets

```bash
cd tabarena/examples/benchmarking/custom_tabarena_model
python run_custom_internal_tabdpt_on_tabarena_lite.py \
    --run-path <run_path> --run-name <your chosen run name>

# Example: command if the checkpoint locates in runs/NoRetrieval/Small_latest
python run_custom_internal_tabdpt_on_tabarena_lite.py \
    --run-path NoRetrieval/Small_latest --run-name small_latest

# Example: with retrieval-mode inference
python run_custom_internal_tabdpt_on_tabarena_lite.py \
    --run-path NoRetrieval/Small_latest --run-name small_latest_retr --use-retrieval
```

Flags:


| Flag              | Required                  | Notes                                                                                                                         |
| ----------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--run-path`      | yes                       | Folder under `Internal-TabDPT/runs/` with `latest.ckpt`.                                                                      |
| `--run-name`      | no (slug of `--run-path`) | Run name. Used as the subfolder under `./tabarena_out/`; pass the same value to `run_evaluate_internal_tabdpt.py --run-name`. |
| `--use-retrieval` | no                        | Off by default; when on, sets `context_size=1024` unless overridden.                                                          |
| `--context-size`  | no                        | Override inference context size. Default: `262144` (no retrieval) or `1024` (retrieval).                                      |


Raw per-task predictions and `results.pkl` files land in
`./tabarena_out/<run-name>/data/Internal-TabDPT_c1/<task_id>/`.

### 3. Evaluate against the leaderboard

```bash
python run_evaluate_internal_tabdpt.py --run-name <your chosen run name>

# Example:
python run_evaluate_internal_tabdpt.py --run-name small_latest
```

- `--run-name` must match the `--run-name` passed to
`run_custom_internal_tabdpt_on_tabarena_lite.py` (it points at the same
`./tabarena_out/<run-name>/` folder and writes figures to `./evals/<run-name>/`).

Outputs will be printed in the console output as well as saved in `./evals/<run-name>/`:

- `tabarena_leaderboard.csv` — full leaderboard
- `leaderboard.tex`          — LaTeX version
- `results_per_split.csv`    — per-split raw numbers
- `tuning-impact-elo.pdf`    — **main Elo rank plot** (this is the headline figure to read off a run)
- `*.pdf`                    — other plots: Pareto fronts, winrate matrix, time plots

