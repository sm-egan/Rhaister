# Rhaister Architecture Guide

## Repo structure (after refactor)

```
Rhaister/
├── train.py                    # Model code — unified entry point (--mode fewshot|zeroshot)
├── train_zeroshot.py           # Thin wrapper (deprecated, calls train.py --mode zeroshot)
├── prepare_combined.py         # Data loading (pdex + cell_eval), evaluation, logging
├── prepare.py                  # Legacy pdex-only loader (still used by eval_per_group.py)
├── prepare_cell_eval.py        # Legacy cell_eval-only loader
├── state_metrics.py            # 6 State paper metric functions (pure, stateless)
├── eval_splits.py              # Multi-split evaluation harness
├── splits/                     # Per-dataset configs + split TOMLs
│   ├── tahoe/dataset.toml      #   Tahoe-100M: 50 cells x 1137 drugs x 14 plates
│   ├── parse/dataset.toml      #   Parse PBMC: 18 cell types x 12 donors x ~100 cytokines
│   └── perturbai/dataset.toml  #   PerturbAI: whole-brain CRISPR atlas
├── scripts/
│   ├── data_prep/              # Generate input parquets from raw h5ad files
│   │   ├── configs/tahoe.yaml  #   YAML config per dataset
│   │   ├── configs/parse.yaml
│   │   ├── compute_pdex.py     #   Mann-Whitney DE via pdex library
│   │   └── compute_celleval_deltas.py  # Pseudobulk deltas
│   ├── sensitivity/            # PerturbHD hit-discovery eval pipeline
│   │   ├── perturbhd_eval.py   #   Core: PCA + ranker + recall@budget
│   │   ├── build_eval_table.py #   Pivot predictions + join labels
│   │   ├── run_all_holdouts.py #   Full pipeline orchestrator
│   │   └── plot_*.py           #   Visualization scripts
│   ├── baselines.py            # Naive baselines (global/cell/treatment mean, additive)
│   ├── eval_a_vs_b.py          # A/B noise-ceiling calibration
│   └── run_titrations_drugs.py # Drug titration experiments
├── tests/                      # pytest suite (57 tests, 3.3s)
├── phdish/                     # Original sensitivity eval (untracked, being integrated)
├── AGENT.md                    # Autoresearch agent instructions
├── PROGRAM.md                  # Full problem definition
├── EXPERIMENTS.md              # Autoresearch experiment log
├── results.jsonl               # Machine-readable results
└── pyproject.toml              # Dependencies + tool config
```

## Data flow

```
Raw h5ad files (single-cell)
    │
    ├── scripts/data_prep/compute_pdex.py ──────► pdex parquets (FC, p-val, FDR, ref_mean)
    │                                              [long format, per plate x cell_line]
    │
    └── scripts/data_prep/compute_celleval_deltas.py ──► cell_eval parquets (expression deltas)
                                                          [wide format, per plate x cell_line]
    │
    ▼
prepare_combined.py:prepare_all()
    │
    ├── load_pdex_data()  ──► pivot, aggregate across plates (mean FC, min p-val)
    ├── load_cell_eval_data() ──► concat plate files, aggregate replicates
    ├── align gene sets (intersection of pdex + cell_eval)
    ├── train/test split (from TOML config)
    └── cache to /tmp/tahoe_cache_combined/
    │
    ▼
train.py:train_and_evaluate(mode="fewshot"|"zeroshot")
    │
    ├── [fewshot] FC: ALS → SVD → similarity → neural → ridge regression
    ├── [fewshot] P-value: ALS on NLP → regression → calibration network → BH FDR
    ├── [fewshot] Delta: ALS → regression blend
    │
    ├── [zeroshot] Delta: SVD-denoised treat means + EB-shrunk diagonal γ + subspace projection
    ├── [zeroshot] FC: treatment mean + D→FC per-gene scale mapping
    └── [zeroshot] P-value: z-score calibration + gene DE prior → erfc → BH FDR
    │
    See docs/zeroshot_architecture.md for full RhaisterO model details.
    │
    ▼
evaluate_test(Y_pred, D_pred, F_pred, P_pred)  [one-shot, raises on 2nd call]
    │
    ├── Legacy: pdex_static/pearson_delta_mean, pdex_static/auprc_p05
    ├── State metric 1: pearson_delta (expression deltas)
    ├── State metric 2: discrimination_score (optional, slow)
    ├── State metric 3: spearman_lfc_sig (FC for significant genes)
    ├── State metric 4: pr_auc (DE gene recovery)
    ├── State metric 5: de_overlap (top DE gene set overlap)
    └── State metric 6: de_spearman_sig (effect size ranking)
```

## Sensitivity prediction pipeline (PerturbHD)

```
Rhaister predictions (per cell x treatment x gene)
    │
    ▼
build_eval_table.py: pivot to wide (cell, treatment, split, growth_rate, gene_0..gene_N)
    │                 join with PRISM or seqrun growth_rate labels
    ▼
perturbhd_eval.py:
    1. StandardScaler + PCA(100) on ALL rows (labelled + unlabelled)
    2. For each hit definition (bottom 5/10/20%, growth_rate < 0):
       a. Binary labels: hit / non-hit
       b. Train classifier (TabPFN/LogReg/HGB) on PCA features → predict_proba
       c. Rank test pairs by score
       d. Compute recall@budget and recall@FDR
    3. Two schemes: OOD (honest) and internal_cv (inflated)
```

## Autoresearch agent loop

The autoresearch agent (Claude Code) follows AGENT.md:
- Only modifies `train.py`
- All other files are immutable
- One-shot evaluator prevents test-set leakage
- Experiments logged to EXPERIMENTS.md and results.jsonl
- HP_* env vars gate experimental code paths (~40 params)

## Dataset configs

Each dataset has a `splits/<dataset>/dataset.toml`:

```toml
[data]
pdex_path = "/path/to/all_plates_pdex.parquet"
cell_eval_dir = "/path/to/cell_eval_dir"
cell_eval_glob = "plate_plate*.parquet"

[data.columns]
cell_line = "cell_line"  # or ["donor", "cell_type"] for composite
cell_eval_treatment = "treatment"  # or "cytokine" for Parse

[data.variants.A]  # for noise-ceiling calibration
pdex_path = "/path/to/variant_A_pdex.parquet"

[genes]
static_list = "static_2k_genes.json"

[splits]
default_config = "generalization_converted_cell_lines_3b.toml"
```

## Key numbers

| Dataset | Cells | Treatments | Genes (static) | Genes (full) | Pairs |
|---------|-------|------------|----------------|--------------|-------|
| Tahoe | 50 cell lines | 1137 drugs | 1995 | 62,710 | 56,827 |
| Parse | 12 donors x 18 types | ~100 cytokines | 2000 | 40,353 | 19,187 |
| PerturbAI | variable | CRISPR targets | 2000 | variable | variable |

## Gene coverage

- pdex: 62,710 genes (all data types: FC, p-value, FDR, ref_mean)
- cell_eval 2K: 2,000 HVG genes (expression deltas)
- cell_eval full: 62,710 genes (expression deltas, all genes)
- Per-cell-line top-3K list: 28,661 union genes, all present in both pdex and cell_eval_full
  Path: `/nvme-shared/drive_3/ml/pod_diffusion/Data/top3k_genes_per_cellline_remapped_v2.json`
