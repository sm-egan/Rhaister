#!/usr/bin/env python
"""Run pdex differential expression from h5ad files.

For each h5ad file, computes per-gene fold change, p-value, and FDR by
comparing each treatment condition to a control using the pdex library
(Mann-Whitney U test). Results are stratified by a grouping column
(e.g. cell_line for Tahoe, donor for Parse) so that each comparison uses
only cells from the same batch/group.

All dataset-specific parameters are read from a YAML config file.

Usage:
    # Tahoe: per-plate, stratified by cell_line, DMSO control
    python scripts/data_prep/compute_pdex.py configs/tahoe.yaml

    # Parse: per-cell_type, stratified by donor, PBS control
    python scripts/data_prep/compute_pdex.py configs/parse.yaml

    # Process specific files only
    python scripts/data_prep/compute_pdex.py configs/tahoe.yaml \
        --files plate1_full_filtered.h5ad.gz plate2_full_filtered.h5ad.gz

Dependencies: anndata, polars, pdex
"""

import argparse
import gc
import glob
import time
from pathlib import Path

import anndata as ad
import polars as pl
import yaml
from pdex import pdex


def process_group(adata_subset, groupby, reference, threads, group_label, file_label):
    """Run pdex on a subset of cells (one group within one file)."""
    n_groups = adata_subset.obs[groupby].nunique()

    if n_groups < 2 or reference not in adata_subset.obs[groupby].values:
        return None, f"skip ({adata_subset.n_obs} cells, {n_groups} groups, no control)"

    t0 = time.time()
    result = pdex(
        adata_subset,
        groupby=groupby,
        mode="ref",
        reference=reference,
        is_log1p=False,
        threads=threads,
    )
    dt = time.time() - t0
    return result, f"ok ({len(result)} rows, {dt:.0f}s)"


def process_file(h5ad_path, cfg, global_cfg, output_dir):
    """Process one h5ad file: stratify by group_col, run pdex per group."""
    file_stem = Path(h5ad_path).name.split(".h5ad")[0]
    file_out = output_dir / f"{file_stem}_pdex.parquet"

    if file_out.exists():
        print(f"  {file_stem}: already done, skipping")
        return file_out

    print(f"\n{'=' * 60}")
    print(f"Processing {file_stem}")
    print(f"{'=' * 60}")

    t0 = time.time()
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Loaded {adata.n_obs:,} cells in {time.time() - t0:.0f}s")

    group_col = global_cfg["group_col"]
    groupby = global_cfg["treatment_col"]
    reference = global_cfg["control"]
    threads = cfg.get("threads", 32)
    file_id_col = global_cfg.get("file_id_col")

    groups = sorted(adata.obs[group_col].unique())
    print(f"  {len(groups)} groups in {group_col}")

    all_results = []
    for i, grp in enumerate(groups):
        grp_out = output_dir / f"{file_stem}_{group_col}_{grp}.parquet"
        if grp_out.exists():
            print(f"    [{i + 1}/{len(groups)}] {grp}: cached")
            all_results.append(pl.read_parquet(grp_out))
            continue

        subset = adata[adata.obs[group_col] == grp].copy()
        result, status = process_group(subset, groupby, reference, threads, grp, file_stem)
        print(f"    [{i + 1}/{len(groups)}] {grp}: {status}")

        if result is not None:
            # Tag with metadata
            extra_cols = [pl.lit(grp).alias(group_col)]
            if file_id_col:
                extra_cols.append(pl.lit(file_stem).alias(file_id_col))
            result = result.with_columns(extra_cols)
            result.write_parquet(grp_out)
            all_results.append(result)

        del subset
        gc.collect()

    del adata
    gc.collect()

    if all_results:
        combined = pl.concat(all_results)
        combined.write_parquet(file_out)
        print(f"  {file_stem}: {len(combined):,} rows -> {file_out.name} ({time.time() - t0:.0f}s)")
    else:
        print(f"  {file_stem}: no results")

    return file_out if all_results else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=str, help="YAML config file (e.g. configs/tahoe.yaml)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory from config")
    parser.add_argument("--files", nargs="+", default=None, help="Process only these h5ad files (basenames)")
    args = parser.parse_args()

    with open(args.config) as f:
        global_cfg = yaml.safe_load(f)

    cfg = global_cfg.get("pdex", {})
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_dir = Path(global_cfg["h5ad_dir"])
    pattern = global_cfg.get("h5ad_pattern", "*.h5ad*")
    h5ad_files = sorted(glob.glob(str(h5ad_dir / pattern)))

    if args.files:
        requested = set(args.files)
        h5ad_files = [f for f in h5ad_files if Path(f).name in requested]

    print(f"Dataset: {global_cfg['dataset']}")
    print(f"h5ad files: {len(h5ad_files)} in {h5ad_dir}")
    print(f"Group col: {global_cfg['group_col']}")
    print(f"Treatment col: {global_cfg['treatment_col']}")
    print(f"Control: {global_cfg['control']}")
    print(f"Output: {output_dir}")

    grand_t0 = time.time()
    result_files = []
    for h5ad_path in h5ad_files:
        result = process_file(h5ad_path, cfg, global_cfg, output_dir)
        if result:
            result_files.append(result)

    # Combine all per-file parquets
    if result_files:
        all_data = pl.concat([pl.read_parquet(f) for f in result_files])
        combined_name = f"all_{global_cfg['dataset']}_pdex.parquet"
        combined_path = output_dir / combined_name
        all_data.write_parquet(combined_path)
        print(f"\nAll files combined: {len(all_data):,} rows -> {combined_path}")

    print(f"\nTotal time: {time.time() - grand_t0:.0f}s")


if __name__ == "__main__":
    main()
