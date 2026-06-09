# Data Inventory

## Tahoe-100M

### Core data (Tier 1 — ~42 GB)
| File | Path | Size | What |
|---|---|---|---|
| pdex | `/nvme-shared/Data/100m_h5ad/pdex_results/all_plates_pdex.parquet` | 40.6 GB | FC, p-value, FDR, ref_mean (long format) |
| cell_eval 2K | `/nvme-shared/Data/tahoe_100m_de_cell_eval/plate_plate*.parquet` (14 files) | 1.0 GB | Expression deltas, 2000 HVG genes |

### Full gene set (Tier 2 — ~15 GB additional)
| File | Path | Size |
|---|---|---|
| cell_eval full | `/nvme-shared/Data/tahoe_100m_de_cell_eval_full/plate*_full_filtered.parquet` (14 files) | 15.3 GB |
| Per-cell-line gene list | `/nvme-shared/drive_3/ml/pod_diffusion/Data/top3k_genes_per_cellline_remapped_v2.json` | ~3 MB |

### A/B variants (Tier 3 — ~88 GB additional)
| File | Path | Size |
|---|---|---|
| pdex group A | `/nvme-shared/Data/100m_h5ad_A/pdex_results/all_plates_pdex.parquet` | 43.4 GB |
| pdex group B | `/nvme-shared/Data/100m_h5ad_B/pdex_results/all_plates_pdex.parquet` | 43.5 GB |
| cell_eval group A | `/nvme-shared/drive_3/ANALYSIS/analysis_255/Scripts/sc_aggregate/results_tahoe100m_celleval/groupA/group_A_celleval_plate*.parquet` (14 files) | 0.7 GB |
| cell_eval group B | `/nvme-shared/drive_3/ANALYSIS/analysis_255/Scripts/sc_aggregate/results_tahoe100m_celleval/groupB/group_B_celleval_plate*.parquet` (14 files) | 0.7 GB |

### Sensitivity labels
| File | Path | Size | Coverage |
|---|---|---|---|
| seqrun growth_rate | `/nvme-shared/Data/seqrun_26_27_merged/5/qc/generic/growth_rate_long.parquet` | small | 282 (cell, treatment) pairs |
| PRISM dose-matched | `/nvme-shared/Data/prism/dose_matched_pdex_prism.parquet` | small | 9,240 (cell, drug, dose) rows, 35 cells x 88 drugs x 3 doses |

### Raw h5ad files
| File | Path | Size |
|---|---|---|
| Plate h5ads | `/nvme-shared/Data/100m_h5ad/plate{1..14}_full_filtered.h5ad.gz` | ~40 GB each |

## Parse PBMC

### Core data
| File | Path | Size |
|---|---|---|
| pdex | `/nvme-shared/Data/parse_pbmc/by_cell_type/pdex_results/all_celltypes_pdex.parquet` | 11.6 GB |
| cell_eval | `/nvme-shared/Data/parse_pbmc/by_cell_type/delta_results/all_celltypes_delta.parquet` | 2.1 GB |

### A/B variants
| File | Path | Size |
|---|---|---|
| pdex A | `/nvme-shared/Data/parse_pbmc/by_cell_type_A/pdex_results/all_celltypes_pdex.parquet` | 10.1 GB |
| cell_eval A | `/nvme-shared/Data/parse_pbmc/by_cell_type_A/delta_results/all_celltypes_delta.parquet` | 1.8 GB |
| pdex B | `/nvme-shared/Data/parse_pbmc/by_cell_type_B/pdex_results/all_celltypes_pdex.parquet` | 10.2 GB |
| cell_eval B | `/nvme-shared/Data/parse_pbmc/by_cell_type_B/delta_results/all_celltypes_delta.parquet` | 1.8 GB |

## PerturbAI

### Core data
| File | Path | Size |
|---|---|---|
| pdex (pooled) | `/nvme-shared/Data/perturbai/wholebrain_crispr_atlas/h5ads/pdex_results_pooled/all_pdex.parquet` | 23.7 GB |
| cell_eval (pooled) | `/nvme-shared/Data/perturbai/wholebrain_crispr_atlas/h5ads/delta_results_pooled/all_delta.parquet` | 4.0 GB |

## Upstream scripts (generate parquets from h5ad)

| Script | Location | What |
|---|---|---|
| Tahoe pdex | `/nvme-shared/Data/100m_h5ad/run_pdex.py` (original) / `scripts/data_prep/compute_pdex.py` (in-repo) | Mann-Whitney via pdex library |
| Tahoe cell_eval | `Barotaxis2/vector_diffusion/scripts/data_prep/compute_celleval_lfc.py` (original) / `scripts/data_prep/compute_celleval_deltas.py` (in-repo) | Pseudobulk mean - DMSO |
| Parse pdex | `/nvme-shared/Data/parse_pbmc/by_cell_type/run_pdex.py` | Per-donor parallelized |
| Parse cell_eval | `/nvme-shared/Data/parse_pbmc/by_cell_type/compute_delta.py` | normalize_total(3320) + log1p |

## Gene lists

| List | Path | Count | Notes |
|---|---|---|---|
| Tahoe static 2K | `splits/tahoe/static_2k_genes.json` | 2,000 | Current model gene set |
| Parse static 2K | `splits/parse/static_2k_genes.json` | 2,000 | 530 overlap with Tahoe |
| PerturbAI static 2K | `splits/perturbai/static_2k_genes.json` | 2,000 | Mouse genes |
| Per-cell-line top 3K | `/nvme-shared/drive_3/ml/pod_diffusion/Data/top3k_genes_per_cellline_remapped_v2.json` | 28,661 union | 1,261-18,956 per cell |
| pdex total | (in parquet) | 62,710 | All available |
| cell_eval full | (in parquet columns) | 62,710 | Matches pdex |

## S3 upload tiers (from issue #3)

| Tier | Cumulative | What you can do |
|---|---|---|
| Tier 1 | ~42 GB | Reproduce headline model + metrics |
| +Tier 2 | ~57 GB | Broader gene lists (up to 62K) |
| +Tier 3 | ~145 GB | Noise-ceiling A/B calibration |
| +Tier 4 | ~370 GB | Cross-dataset (Parse, PerturbAI) |
