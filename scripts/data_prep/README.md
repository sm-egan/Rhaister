# Data Preparation Scripts

These scripts generate the input parquets Rhaister consumes from raw single-cell
h5ad files. Both scripts are config-driven: a YAML file specifies column names,
control conditions, normalization, and paths. See `configs/` for dataset configs.

## Two output types

| Output | Format | Method | Script |
|---|---|---|---|
| **pdex** | Long: (target, feature, fold_change, p_value, fdr, ...) | Mann-Whitney U via `pdex` library | `compute_pdex.py` |
| **cell_eval deltas** | Wide: (group_col, treatment_col, gene1, ..., geneN) | Pseudobulk mean(treated) - mean(control) | `compute_celleval_deltas.py` |

These are kept as separate parquets because they may cover different gene subsets
(e.g. cell_eval HVG mode produces 2K genes while pdex covers 62K).

Both are computed **within batch** using that batch's own control cells:

| Dataset | Batch unit | Cell identity | Treatment | Control |
|---|---|---|---|---|
| **Tahoe** | plate (14) | cell_line (50 CVCL IDs) | drug+dose | DMSO |
| **Parse** | cell_type (18) | donor (12) | cytokine | PBS |

## Usage

```bash
# --- Tahoe ---

# pdex (fold change, p-value, FDR)
python scripts/data_prep/compute_pdex.py scripts/data_prep/configs/tahoe.yaml

# cell_eval deltas (2K HVG genes)
python scripts/data_prep/compute_celleval_deltas.py scripts/data_prep/configs/tahoe.yaml

# cell_eval deltas (full ~63K genes)
python scripts/data_prep/compute_celleval_deltas.py scripts/data_prep/configs/tahoe.yaml --profile celleval_full

# --- Parse ---

# pdex
python scripts/data_prep/compute_pdex.py scripts/data_prep/configs/parse.yaml

# cell_eval deltas
python scripts/data_prep/compute_celleval_deltas.py scripts/data_prep/configs/parse.yaml

# --- Common options ---

# Process specific files only
python scripts/data_prep/compute_pdex.py configs/tahoe.yaml \
    --files plate1_full_filtered.h5ad.gz plate2_full_filtered.h5ad.gz

# Override output directory
python scripts/data_prep/compute_celleval_deltas.py configs/tahoe.yaml \
    --output_dir /tmp/test_output
```

## Config structure

```yaml
dataset: tahoe                          # dataset name
h5ad_dir: /path/to/h5ad/files          # directory with h5ad files
h5ad_pattern: "plate*.h5ad.gz"          # glob pattern for h5ad files

treatment_col: drugname_drugconc        # obs column for treatment condition
control: "[('DMSO_TF', 0.0, 'uM')]"    # control condition value
group_col: cell_line                    # stratification column within each file
file_id_col: null                       # optional: carry filename as a column

pdex:
  output_dir: /path/to/pdex_output
  threads: 32

celleval:                               # profile name (--profile celleval)
  output_dir: /path/to/delta_output
  gene_mode: hvg                        # "hvg" or "all_genes"
  hvg_key: X_hvg                        # obsm key for HVG mode
  gene_name_col: gene_symbol            # var column for gene names
  target_sum: 1872                      # for normalize_total in all_genes mode

celleval_full:                          # alternate profile (--profile celleval_full)
  output_dir: /path/to/delta_full_output
  gene_mode: all_genes
  gene_name_col: gene_name
  target_sum: 1872
```

## Dependencies

- `anndata` — h5ad I/O
- `pdex` — Mann-Whitney differential expression (`compute_pdex.py` only)
- `scipy` — sparse matrix operations
- `polars` — parquet I/O (`compute_pdex.py`)
- `pandas` — parquet I/O (`compute_celleval_deltas.py`)
- `pyyaml` — config loading
- `numpy`

## Adding a new dataset

1. Create `configs/<dataset>.yaml` specifying your column names, control, and paths
2. Run `compute_pdex.py` and `compute_celleval_deltas.py` with your config
3. Add a `splits/<dataset>/dataset.toml` pointing to the output parquets
