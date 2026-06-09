#!/usr/bin/env python
"""Compute cell-eval pseudobulk expression deltas from h5ad files.

For each h5ad file in the input directory:
  1. Load expression matrix (HVG from obsm, or full .X with normalization)
  2. Pseudobulk by (group_col, treatment_col) via sparse matrix multiplication
  3. Delta = mean(treated) - mean(control) per group
  4. Save as wide-format parquet: [group_col, treatment_col, gene1, ..., geneN]

All dataset-specific parameters (column names, control condition, normalization)
are read from a YAML config file. See configs/ for examples.

Usage:
    # Tahoe HVG (2K genes)
    python scripts/data_prep/compute_celleval_deltas.py configs/tahoe.yaml

    # Tahoe full genome (63K genes)
    python scripts/data_prep/compute_celleval_deltas.py configs/tahoe.yaml --profile celleval_full

    # Parse
    python scripts/data_prep/compute_celleval_deltas.py configs/parse.yaml

    # Override output directory
    python scripts/data_prep/compute_celleval_deltas.py configs/tahoe.yaml --output_dir /tmp/test
"""

import argparse
import gc
import glob
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def sparse_normalize_total_log1p(X, target_sum):
    """In-place normalize_total + log1p on a sparse CSR matrix."""
    X = X.tocsr()
    if X.data.dtype.kind != "f":
        X = X.astype(np.float32, copy=False)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    safe = np.where(row_sums > 0, row_sums, 1.0).astype(X.data.dtype)
    row_idx = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr))
    X.data *= (target_sum / safe[row_idx]).astype(X.data.dtype, copy=False)
    np.log1p(X.data, out=X.data)
    return X


def extract_expression(adata, cfg):
    """Extract expression matrix and gene names based on config."""
    gene_mode = cfg.get("gene_mode", "all_genes")

    if gene_mode == "hvg":
        hvg_key = cfg.get("hvg_key", "X_hvg")
        X = np.log1p(adata.obsm[hvg_key])
        if hasattr(X, "toarray"):
            X = X.toarray()
        gene_name_col = cfg.get("gene_name_col", "gene_symbol")
        gene_names = adata.var[gene_name_col].values
    else:
        target_sum = cfg.get("target_sum", 1872)
        if sp.issparse(adata.X):
            X = sparse_normalize_total_log1p(adata.X.copy(), target_sum)
        else:
            row_sums = adata.X.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            X = np.log1p(adata.X / row_sums * target_sum)

        gene_name_col = cfg.get("gene_name_col")
        if gene_name_col and gene_name_col in adata.var.columns:
            gene_names = adata.var[gene_name_col].values
        else:
            gene_names = adata.var.index.to_numpy()

    return X, gene_names


# ---------------------------------------------------------------------------
# Pseudobulk
# ---------------------------------------------------------------------------


def pseudobulk_means(X, group_labels):
    """Compute per-group mean expression via sparse group matrix.

    Args:
        X: (n_cells, n_genes) expression matrix (sparse or dense)
        group_labels: (n_cells,) string array of group identifiers

    Returns:
        means: (n_groups, n_genes) ndarray
        group_to_idx: dict mapping group label -> row index in means
    """
    unique_groups, inverse = np.unique(group_labels, return_inverse=True)
    n_groups = len(unique_groups)
    n_cells = len(group_labels)

    counts = np.bincount(inverse, minlength=n_groups)
    weights = (1.0 / counts[inverse]).astype(np.float32)

    group_matrix = csr_matrix(
        (weights, (inverse, np.arange(n_cells))),
        shape=(n_groups, n_cells),
    )

    means = group_matrix @ X
    if sp.issparse(means):
        means = means.toarray()

    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    return np.asarray(means), group_to_idx


def compute_deltas(means, group_to_idx, gene_names, control, group_col, treatment_col, file_id=None, file_id_col=None):
    """Compute treatment - control deltas.

    Groups are encoded as "group_val||treatment_val" in group_to_idx keys.
    Returns a DataFrame with columns [file_id_col?, group_col, treatment_col, gene1..geneN].
    """
    groups_by_batch = {}
    for key in group_to_idx:
        batch_val, treat_val = key.split("||", 1)
        groups_by_batch.setdefault(batch_val, []).append(treat_val)

    meta_rows = []
    delta_blocks = []
    n_skipped = 0

    for batch_val, treatments in groups_by_batch.items():
        ctrl_key = f"{batch_val}||{control}"
        if ctrl_key not in group_to_idx:
            n_skipped += len([t for t in treatments if t != control])
            continue

        ctrl_mean = means[group_to_idx[ctrl_key]]

        for treat_val in treatments:
            if treat_val == control:
                continue
            treat_key = f"{batch_val}||{treat_val}"
            delta = means[group_to_idx[treat_key]] - ctrl_mean

            row = {}
            if file_id_col and file_id is not None:
                row[file_id_col] = file_id
            row[group_col] = batch_val
            row[treatment_col] = treat_val
            meta_rows.append(row)
            delta_blocks.append(delta)

    if n_skipped > 0:
        print(f"  skipped {n_skipped} condition(s) (no control)")

    if not delta_blocks:
        return pd.DataFrame()

    delta_arr = np.asarray(delta_blocks, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    gene_df = pd.DataFrame(delta_arr, columns=list(gene_names))
    return pd.concat([meta_df, gene_df], axis=1)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process_file(h5ad_path, cfg, global_cfg):
    """Process one h5ad file and return a delta DataFrame."""
    file_stem = Path(h5ad_path).name.split(".h5ad")[0]

    print(f"\n{'=' * 60}")
    print(f"Processing {file_stem}")
    print(f"{'=' * 60}")

    print(f"  Loading {h5ad_path}...")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Reset var index if needed for gene_name_col access
    if cfg.get("gene_mode") == "all_genes":
        adata.var = adata.var.reset_index()

    X, gene_names = extract_expression(adata, cfg)
    print(f"  Expression: {X.shape[1]:,} genes, mode={cfg.get('gene_mode', 'all_genes')}")

    group_col = global_cfg["group_col"]
    treatment_col = global_cfg["treatment_col"]

    # Build compound group labels: "group_val||treatment_val"
    group_vals = adata.obs[group_col].astype(str).values
    treat_vals = adata.obs[treatment_col].astype(str).values
    labels = np.array([f"{g}||{t}" for g, t in zip(group_vals, treat_vals)])

    means, group_to_idx = pseudobulk_means(X, labels)
    print(f"  Pseudobulk: {len(group_to_idx)} groups")

    file_id_col = global_cfg.get("file_id_col")
    file_id = file_stem if file_id_col else None

    delta_df = compute_deltas(
        means,
        group_to_idx,
        gene_names,
        global_cfg["control"],
        group_col,
        treatment_col,
        file_id=file_id,
        file_id_col=file_id_col,
    )
    print(f"  Output: {delta_df.shape}")

    del adata, X, means
    gc.collect()
    return delta_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="YAML config file (e.g. configs/tahoe.yaml)")
    parser.add_argument(
        "--profile",
        type=str,
        default="celleval",
        help="Config profile: 'celleval' or 'celleval_full' (default: celleval)",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory from config")
    parser.add_argument("--files", nargs="+", default=None, help="Process only these h5ad files (basenames)")
    args = parser.parse_args()

    with open(args.config) as f:
        global_cfg = yaml.safe_load(f)

    profile = global_cfg.get(args.profile, {})
    if not profile:
        raise ValueError(
            f"Profile '{args.profile}' not found in {args.config}. "
            f"Available: {[k for k in global_cfg if isinstance(global_cfg[k], dict)]}"
        )

    output_dir = Path(args.output_dir or profile["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_dir = Path(global_cfg["h5ad_dir"])
    pattern = global_cfg.get("h5ad_pattern", "*.h5ad*")
    h5ad_files = sorted(glob.glob(str(h5ad_dir / pattern)))

    if args.files:
        requested = set(args.files)
        h5ad_files = [f for f in h5ad_files if Path(f).name in requested]

    print(f"Dataset: {global_cfg['dataset']}")
    print(f"Profile: {args.profile}")
    print(f"Gene mode: {profile.get('gene_mode', 'all_genes')}")
    print(f"h5ad files: {len(h5ad_files)} in {h5ad_dir}")
    print(f"Output: {output_dir}")

    all_dfs = []
    for h5ad_path in h5ad_files:
        file_stem = Path(h5ad_path).name.split(".h5ad")[0]
        out_path = output_dir / f"{file_stem}.parquet"

        if out_path.exists():
            print(f"\n  {file_stem}: already done, skipping")
            all_dfs.append(pd.read_parquet(out_path))
            continue

        df = process_file(h5ad_path, profile, global_cfg)
        if not df.empty:
            df.to_parquet(out_path, index=False)
            print(f"  Saved {out_path}")
            all_dfs.append(df)

    # Combine all files
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_name = f"all_{global_cfg['dataset']}_deltas.parquet"
        combined_path = output_dir / combined_name
        combined.to_parquet(combined_path, index=False)
        print(f"\nCombined: {combined.shape} -> {combined_path}")

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    total = sum(len(df) for df in all_dfs)
    n_genes = (
        all_dfs[0].shape[1]
        - len(
            [
                c
                for c in all_dfs[0].columns
                if c in (global_cfg["group_col"], global_cfg["treatment_col"], global_cfg.get("file_id_col", ""))
            ]
        )
        if all_dfs
        else 0
    )
    print(f"  {total:,} conditions x {n_genes:,} genes across {len(all_dfs)} files")


if __name__ == "__main__":
    main()
