# Tahoe Gene Expression Prediction — Research Program

## Objective

**Predict per-gene differential expression** for unseen (cell_line, treatment) combinations, evaluated on all six metrics from the State paper (Adduri et al. 2025). A single model must produce three outputs — expression deltas, log2 fold changes, and FDR-adjusted significance — from which the six metrics are computed.

### Three prediction targets

| Target | Source | Array key | Description |
|--------|--------|-----------|-------------|
| **Expression deltas** (D) | cell_eval | `D_train/D_test` | Pseudobulk mean(treated) − mean(DMSO), linear scale |
| **Log2 fold change** (Y) | pdex | `Y_train/Y_test` | `log2(target_mean / ref_mean)`, Mann-Whitney |
| **FDR** (F) | pdex | `F_train/F_test` | BH-adjusted p-values, min across plates |

Deltas and fold changes do **not** correlate (r ≈ 0.1) despite measuring related quantities. They differ in scale (linear vs log), normalization (DMSO subtraction vs ratio), and gene weighting (high-expression genes dominate deltas). The model must learn both spaces.

### Six evaluation metrics

| # | Metric | Input | State Fig | Description |
|---|--------|-------|-----------|-------------|
| 1 | `state/pearson_delta_mean` | D | 2E | Pearson of expression deltas, per obs |
| 2 | `state/discrimination_mean` | D | 2D | L1 perturbation identity ranking (optional, slow) |
| 3 | `state/spearman_lfc_sig_mean` | Y, F | 2H | Spearman of FC for FDR-significant genes |
| 4 | `state/pr_auc_mean` | F | 2F | AUPRC for DE gene recovery |
| 5 | `state/de_overlap_mean` | Y, F | 2I | Top DE gene set overlap |
| 6 | `state/de_spearman_sig` | F | 2J | Spearman of #significant genes across obs |

Legacy metrics (`pdex_static/pearson_delta_mean`, `pdex_static/auprc_p05`) are also reported for continuity.

## Data

**Two data sources**, loaded jointly by `prepare_combined.py`:

- **pdex**: `/nvme-shared/Data/100m_h5ad/pdex_results/all_plates_pdex.parquet` (4B rows long format). Provides fold_change, p_value, fdr, ref_mean per (cell_line, treatment, gene, plate). Pivoted to wide: FC mean across plates, p-value/FDR min across plates, ref_mean mean.
- **cell_eval**: `/nvme-shared/Data/tahoe_100m_de_cell_eval/plate_plate*.parquet` (14 plate files, wide format). Provides expression deltas per (cell_line, treatment, gene). Mean-aggregated across plates.

Both sources share 1995 genes (intersection of static 2K set with pdex coverage), 50 cell lines, 1137 treatments, and 56,827 unique (cell_line, treatment) pairs after aggregation. Split: **`tahoe_5_holdout` only** — 53,154 train, 3,673 test.

### What `prepare_combined.py` provides
```python
data = prepare_all("tahoe_5_holdout")
data["Y_train"]       # (53154, 1995) log2 fold changes (pdex)
data["P_train"]       # (53154, 1995) p-values (pdex)
data["F_train"]       # (53154, 1995) FDR (pdex)
data["D_train"]       # (53154, 1995) expression deltas (cell_eval)
data["R_train"]       # (53154, 1995) ref_mean baseline expression (pdex)
data["R_test"]        # (3673, 1995) ref_mean for test
data["F_test"]        # (3673, 1995) FDR for test
data["evaluate_test"] # one-shot: evaluate_test(Y_pred, D_pred=, F_pred=, P_pred=)
# Other: train_cells, train_treatments, test_cells, test_treatments,
#        gene_cols, cell_to_idx, treat_to_idx, n_cells, n_treatments, n_test
```

### Understanding the data types
- **ref_mean**: Baseline expression per (cell_line, gene). Encodes cell identity, determines statistical power. `log1p` recommended. Available at prediction time.
- **Deltas vs FC**: Deltas are linear-scale differences dominated by high-expression genes. FC is log-scale ratio — all genes weighted more equally. A model good at FC is not automatically good at deltas.
- **FDR vs p-values**: FDR is per-observation BH-corrected. The model can predict p-values and derive FDR via `pvalues_to_fdr_bh()`, or predict FDR directly.

## Current architecture (train.py)

**Single entry point: `train_and_evaluate(experiment_name, split_name, log, data)`**

Currently runs largely independent pipelines for each target:

### FC prediction (pipeline + regression)
1. **`_pipeline()`**: NaN-aware ALS → SVD embeddings → similarity correction → neural interaction
2. **`_compute_regression()`**: Drug-as-linear-combination ridge, cell-weighted, blend at alpha=0.85
3. Global MLP on residuals (w=0.12), cell regression (alpha=0.05)

### P-value / FDR prediction
4. ALS on −log10(p), per-treatment similarity correction
5. Per-gene standardized `_compute_regression()` for NLP
6. NLP cell regression (alpha=0.15)
7. Neural calibration network: (|FC|, NLP, ref_mean, gene_stats) → significance
8. FDR derived from P_pred via BH correction

### Delta prediction
9. ALS on D_train → mu_d + treat_eff_d + cell_eff_d
10. `_compute_regression()` on deltas, blend at 0.85

### Interaction between targets
Currently minimal — the calnet uses |FC| as an input feature for significance prediction, and NLP-guided shrinkage modulates FC. But the three pipelines largely share no structure. **This is the main area for improvement**: building a cohesive model where all targets share cell and treatment representations.

## Evaluation
- Evaluation computes all six State metrics + legacy metrics in a single call
- `evaluate_test(Y_pred, D_pred=D_pred, F_pred=F_pred, P_pred=P_pred)`
- Discrimination score (metric 2) is off by default (`compute_discrimination=True` to enable)
- Results logged to `results.jsonl` and WandB

## Current results

| Model | fc_pearson | auprc | delta_pearson | spearman_lfc | pr_auc | de_overlap | spearman_sig |
|-------|-----------|-------|---------------|-------------|--------|------------|-------------|
| global_mean | 0.161 | 0.495 | 0.204 | 0.292 | 0.382 | 0.307 | — |
| cell_mean | 0.265 | 0.619 | 0.393 | 0.504 | 0.553 | 0.398 | 0.457 |
| treatment_mean | 0.283 | 0.581 | 0.483 | 0.508 | 0.440 | 0.336 | 0.735 |
| additive | 0.354 | 0.671 | 0.574 | 0.631 | 0.594 | 0.487 | 0.856 |
| **full model** | **0.650** | **0.800** | **0.909** | **0.842** | **0.774** | **0.655** | **0.958** |

## How to run
```bash
# Development (always uses tahoe_5_holdout)
python train.py full_v12

# Baselines
python scripts/baselines.py
```
