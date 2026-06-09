# Repo Conventions

## Installation

```bash
uv pip install -e ".[dev]"
```

## Code structure

- **Package**: All model code lives in `rhaister/`. Install with `uv pip install -e .` for editable development.
- **Single entry point**: `rhaister.train.train_and_evaluate()` is the public interface. Supports fewshot (default) and zeroshot (`zeroshot=True`) modes.
- **Fewshot model**: ALS decomposition + unweighted drug ridge regression + neural calibration network (calnet) for p-values.
- **Zeroshot model (Rhaister-O)**: Per-(drug, gene) diagonal model `y = β + γ·x` using HVG centroid baseline expression. Activated via `HP_ZEROSHOT=1` or `zeroshot=True`. Matches paper equations 8-9.

## File layout

| Path | Purpose |
|---|---|
| `rhaister/` | Python package (install with `uv pip install -e .`) |
| `rhaister/train.py` | Model code. Fewshot by default; `HP_ZEROSHOT=1` for zeroshot. |
| `rhaister/prepare_combined.py` | Data loading (pdex + cell_eval), evaluation (6 State metrics), logging |
| `rhaister/state_metrics.py` | Six metric functions from the State paper |
| `rhaister/eval_splits.py` | Multi-split evaluation for leaderboard |
| `splits/` | Per-dataset configs (`dataset.toml`) and split TOMLs |
| `tests/` | Unit tests (pytest) |
| `scripts/` | Helper scripts (baselines, sweeps, evaluations) |
| `figures/` | Paper figure generation scripts |
| `docs/` | Architecture docs, data inventory, design decisions |

## Running the model

```bash
# Fewshot model (uses tahoe_5_holdout by default)
python -m rhaister.train full_v12

# Zeroshot model (Rhaister-O, all 5 Tahoe holdouts)
HP_ZEROSHOT=1 python -m rhaister.train zs_test
HP_ZEROSHOT=1 python -m rhaister.train zs_h5 --split tahoe/5_holdout

# Other datasets
python -m rhaister.train parse_test --split parse/split_0
python -m rhaister.train replogle_test --split replogle_nadig/split_0

# Multi-split evaluation
python -m rhaister.eval_splits full_v12
```

## Reproducing paper figures

Figures require result JSON files at the repo root, produced by running baselines + model on all splits. The result files are gitignored — they must be regenerated locally.

### Step 1: Generate result JSONs

```bash
# Tahoe baselines (5 splits, ~30 min)
python scripts/baselines.py

# Tahoe fewshot model results (5 splits, ~15 min)
WANDB_MODE=disabled python -c "
import os, json; os.environ['WANDB_MODE']='disabled'
from rhaister.train import train_and_evaluate
results = {}
for i in range(5, 10):
    split = f'tahoe_{i}_holdout'
    m = train_and_evaluate(split_name=split, log=False, compute_discrimination=True)
    results[split] = {k: float(v) if hasattr(v,'__float__') else v for k,v in m.items()}
json.dump(results, open('rhaister_results.json','w'), indent=2)
"

# Parse + Replogle sweeps (baselines + model)
python scripts/run_parse_sweep.py
python scripts/run_replogle_nadig_sweep.py

# Sensitivity: Tahoe + PRISM (baselines + model, 5 splits each)
for i in 0 1 2 3 4; do
  python scripts/baseline_sensitivity.py --split EmeraldBay/split_\$i
  python -m rhaister.train_sensitivity rhaister_v1 --split EmeraldBay/split_\$i
  HP_FEATURES="pdex,pdex_pv,pdex_fdr,cell_eval" python -m rhaister.train_sensitivity rhaister_v1_feat_all --split EmeraldBay/split_\$i
done
python scripts/baseline_sensitivity.py --split prism/split_0
python -m rhaister.train_sensitivity rhaister_v1 --split prism/split_0

# Zeroshot (Rhaister-O, 5 Tahoe holdouts, ~2 min each)
for i in 5 6 7 8 9; do
  WANDB_MODE=disabled HP_ZEROSHOT=1 python -m rhaister.train zs_h\$i --split tahoe/\${i}_holdout
done
# Zeroshot baselines
WANDB_MODE=disabled python scripts/zeroshot_baselines.py

# STATE model results + A/B half-sample reference (from WandB)
python scripts/fetch_state.py
python scripts/fetch_wandb.py
```

### Expected paper metrics (mean across splits)

| Dataset | Mode | pearson_delta | Notes |
|---|---|---|---|
| Tahoe-100M | Fewshot | 0.87 | 5 holdouts |
| Parse PBMC | Fewshot | 0.46 | fewshot_donor_split_0 |
| Replogle-Nadig | Fewshot | 0.41 | 4 splits |
| Tahoe sensitivity | Sensitivity | R²=0.26 (0.31 with features) | 5 splits |
| PRISM sensitivity | Sensitivity | R²=0.87 | split_0 |
| Tahoe-100M | Zeroshot | 0.63 | 5 holdouts |

### Step 2: Generate figures

```bash
# Fig 1 top: Tahoe main metrics
python figures/main_metrics.py

# Fig 1 middle: Parse main metrics
python figures/main_metrics_parse_fewshot_donor.py

# Fig 1 bottom: Replogle main metrics
python figures/main_metrics_replogle_nadig.py

# Fig 2: Sensitivity (Tahoe + PRISM)
python figures/sensitivity_r2.py --dataset EmeraldBay
python figures/sensitivity_r2.py --dataset prism

# Fig 4: Zeroshot
python figures/main_metrics_zeroshot.py
```

Figures are rendered to `figures/*.pdf` and `figures/*.png` (gitignored).

### Required result files per figure

| Figure | Script | Required JSONs |
|---|---|---|
| Fig 1 top (Tahoe) | `figures/main_metrics.py` | `baseline_results.json`, `rhaister_results.json`, `a_vs_b_results.tahoe_only.json`, `state_results.json` |
| Fig 1 mid (Parse) | `figures/main_metrics_parse_fewshot_donor.py` | `baseline_results.json`, `model_results.json`, `a_vs_b_results.json` |
| Fig 1 bot (Replogle) | `figures/main_metrics_replogle_nadig.py` | `baseline_results.json`, `model_results.json`, `state_results.json` |
| Fig 2 (Sensitivity) | `figures/sensitivity_r2.py` | `results_sensitivity.jsonl` |
| Fig 4 (Zeroshot) | `figures/main_metrics_zeroshot.py` | `titration_cells_zeroshot_results.json`, `zeroshot_baselines_results.json`, `rhaister_results.json`, `a_vs_b_results.tahoe_only.json`, `state_results.json` |

## Unit tests

```bash
python -m pytest tests/ -v    # 57 tests, ~5s
```

## Data

Data paths in `splits/<dataset>/dataset.toml` are relative (e.g. `pdex/all_plates_pdex.parquet`). They are resolved via:

1. **`RHAISTER_DATA_ROOT`** env var — if `$RHAISTER_DATA_ROOT/<dataset>/` exists, paths resolve there. On the cluster: `export RHAISTER_DATA_ROOT=/nvme-shared/shreshth/rhaister_data`
2. **HuggingFace** — auto-downloads from `tahoebio/*-de-rhaister` repos if no local data found. Requires `HF_TOKEN` for private repos.

## Datasets

| Dataset | HuggingFace | Splits |
|---|---|---|
| Tahoe-100M | `tahoebio/tahoe-de-rhaister` | `tahoe/5_holdout` .. `tahoe/9_holdout` |
| Parse PBMC | `tahoebio/parse-de-rhaister` | `parse/split_0` .. `parse/split_4` |
| Replogle-Nadig | `tahoebio/replogle-nadig-de-rhaister` | `replogle_nadig/split_0` .. `replogle_nadig/split_3` |
