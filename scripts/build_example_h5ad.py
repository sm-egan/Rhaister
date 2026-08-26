#!/usr/bin/env python
"""Build the walkthrough's example h5ad from the public Tahoe-100M dataset.

tutorials/rhaister_walkthrough.ipynb needs a small single-cell AnnData with
several cell lines, several drugs and their DMSO controls, so it can show the
`AnnData -> DE matrices` bridge on data the model then trains on. The full
per-plate h5ads are cluster-internal (docs/data_inventory.md, ~40 GB each), but
the same cells are public in `tahoebio/Tahoe-100M` as sparse parquet shards.

This script pulls just the rows it needs:

  * `metadata/obs_metadata.parquet` is 2.3 GB, but the four columns needed to
    recover treatment labels are dictionary-encoded and total ~1.5 MB.
  * Each of the 3,388 data shards holds ~48 pooled cell lines. Shard footers
    carry min/max statistics for `sample`, so the row groups covering the wanted
    samples are selected without reading any cell data.

Typical run downloads a few hundred MB and takes ~10 minutes.

Usage
-----
    python scripts/build_example_h5ad.py --out sample_tahoe.h5ad

    python scripts/build_example_h5ad.py --plate plate1 \\
        --cell-lines CVCL_0218 CVCL_0366 CVCL_0459 \\
        --drugs Bortezomib Afatinib --cells-per-group 200
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rhaister.tutorial_data import (
    WALKTHROUGH_CELL_LINES,
    WALKTHROUGH_DRUGS,
    load_gene_panel,
    treatment_label,
)

HF_REPO = "tahoebio/Tahoe-100M"
N_SHARDS = 3388
SHARD_FMT = "data/train-{i:05d}-of-03388.parquet"
CONTROL_DRUG = "DMSO_TF"


# ---------------------------------------------------------------------------
# Metadata (cheap: dictionary-encoded columns)
# ---------------------------------------------------------------------------


def load_sample_metadata(plate, verbose=True):
    """sample -> treatment label, for one plate.

    Reads four dictionary-encoded columns of obs_metadata.parquet (~1.5 MB of
    the file's 2.3 GB) rather than the whole thing.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    if verbose:
        print(f"Reading sample metadata for {plate}...")
    tbl = pq.read_table(
        fs.open(f"datasets/{HF_REPO}/metadata/obs_metadata.parquet", "rb"),
        columns=["plate", "sample", "drug", "drugname_drugconc"],
    )
    df = tbl.to_pandas().drop_duplicates()
    df = df[df["plate"].astype(str) == plate]
    if df.empty:
        raise ValueError(f"No samples found for plate {plate!r}.")
    sample_to_treatment = dict(zip(df["sample"], df["drugname_drugconc"]))
    sample_to_drug = dict(zip(df["sample"], df["drug"]))
    if verbose:
        print(f"  {len(sample_to_treatment)} samples, {df['drug'].nunique()} drugs on {plate}")
    return sample_to_treatment, sample_to_drug


def load_gene_index(gene_panel, verbose=True):
    """token_id -> position in `gene_panel`, for the genes present in both."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, "metadata/gene_metadata.parquet", repo_type="dataset")
    meta = pd.read_parquet(path, columns=["gene_symbol", "token_id"])
    panel_pos = {g: i for i, g in enumerate(gene_panel)}
    token_to_col = {}
    for symbol, token in zip(meta["gene_symbol"], meta["token_id"]):
        pos = panel_pos.get(symbol)
        if pos is not None:
            token_to_col[int(token)] = pos
    if verbose:
        print(f"  {len(token_to_col)}/{len(gene_panel)} panel genes found in the Tahoe gene vocabulary")
    return token_to_col


def load_cell_names(verbose=True):
    """CVCL id -> readable cell line name."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, "metadata/cell_line_metadata.parquet", repo_type="dataset")
    meta = pd.read_parquet(path, columns=["cell_name", "Cell_ID_Cellosaur"]).drop_duplicates(
        subset="Cell_ID_Cellosaur"
    )
    return dict(zip(meta["Cell_ID_Cellosaur"], meta["cell_name"]))


# ---------------------------------------------------------------------------
# Locating the shards and row groups worth reading
# ---------------------------------------------------------------------------


def _shard_footer(fs, i):
    return pq.ParquetFile(fs.open(f"datasets/{HF_REPO}/{SHARD_FMT.format(i=i)}", "rb")).metadata


def _shard_plates(fs, i):
    md = _shard_footer(fs, i)
    names = [md.schema.column(k).name for k in range(md.num_columns)]
    pi = names.index("plate")
    return {md.row_group(g).column(pi).statistics.min for g in range(md.num_row_groups)}


def find_plate_shards(fs, plate, stride=8, threads=16, verbose=True):
    """Shard indices holding `plate`.

    Shards are grouped by plate in contiguous blocks, so a strided probe finds
    the block and the edges are then walked outwards.
    """
    if verbose:
        print(f"Locating {plate} shards (strided footer probe)...")
    probes = list(range(0, N_SHARDS, stride))
    with ThreadPoolExecutor(threads) as ex:
        hits = [i for i, plates in zip(probes, ex.map(lambda i: _shard_plates(fs, i), probes)) if plate in plates]
    if not hits:
        raise ValueError(f"Plate {plate!r} not found in {HF_REPO}.")

    lo, hi = min(hits), max(hits)
    while lo > 0 and plate in _shard_plates(fs, lo - 1):
        lo -= 1
    while hi < N_SHARDS - 1 and plate in _shard_plates(fs, hi + 1):
        hi += 1
    if verbose:
        print(f"  shards {lo}..{hi} ({hi - lo + 1} shards)")
    return list(range(lo, hi + 1))


def select_row_groups(fs, shards, samples, threads=16, verbose=True):
    """(shard, row_group) pairs whose `sample` statistics overlap `samples`."""
    lo, hi = min(samples), max(samples)

    def scan(i):
        md = _shard_footer(fs, i)
        names = [md.schema.column(k).name for k in range(md.num_columns)]
        si = names.index("sample")
        out = []
        for g in range(md.num_row_groups):
            st = md.row_group(g).column(si).statistics
            if st is None or not (st.max < lo or st.min > hi):
                out.append((i, g))
        return out

    if verbose:
        print(f"Scanning {len(shards)} shard footers for matching row groups...")
    with ThreadPoolExecutor(threads) as ex:
        selected = [pair for part in ex.map(scan, shards) for pair in part]
    if verbose:
        print(f"  {len(selected)} candidate row groups")
    return selected


# ---------------------------------------------------------------------------
# Reading cells
# ---------------------------------------------------------------------------


def collect_cells(
    fs, selected, sample_to_treatment, sample_to_drug, cell_lines, treatments,
    cells_per_group, token_to_col, n_panel, verbose=True,
):
    """Read selected row groups, keeping up to `cells_per_group` cells per
    (cell_line, treatment). Returns (rows, obs_records) where rows are CSR
    pieces already projected onto the panel gene axis."""
    cell_lines, treatments = set(cell_lines), set(treatments)
    counts = {}
    rows, obs = [], []
    t0 = time.time()

    for n, (shard, group) in enumerate(selected, 1):
        need = sum(
            max(0, cells_per_group - counts.get((c, t), 0))
            for c in cell_lines
            for t in treatments
        )
        if need == 0:
            if verbose:
                print("  all groups filled — stopping early")
            break

        pf = pq.ParquetFile(fs.open(f"datasets/{HF_REPO}/{SHARD_FMT.format(i=shard)}", "rb"))
        tbl = pf.read_row_group(group, columns=["genes", "expressions", "sample", "cell_line_id"])
        chunk = tbl.to_pydict()

        for genes, exprs, sample, cell in zip(
            chunk["genes"], chunk["expressions"], chunk["sample"], chunk["cell_line_id"]
        ):
            if cell not in cell_lines:
                continue
            treatment = sample_to_treatment.get(sample)
            if treatment is None or treatment not in treatments:
                continue
            key = (cell, treatment)
            if counts.get(key, 0) >= cells_per_group:
                continue

            cols, vals = [], []
            for token, value in zip(genes, exprs):
                pos = token_to_col.get(token)
                if pos is not None and value:
                    cols.append(pos)
                    vals.append(value)
            if not cols:
                continue

            rows.append(sp.csr_matrix(
                (np.asarray(vals, dtype=np.float32), np.asarray(cols, dtype=np.int32),
                 np.array([0, len(cols)], dtype=np.int32)),
                shape=(1, n_panel),
            ))
            obs.append({
                "cell_line": cell,
                "drug": sample_to_drug.get(sample, ""),
                "treatment": treatment,
                "tscp_count": float(sum(exprs)),
            })
            counts[key] = counts.get(key, 0) + 1

        if verbose:
            filled = sum(1 for v in counts.values() if v >= cells_per_group)
            print(
                f"  [{n}/{len(selected)}] {len(obs)} cells, "
                f"{filled}/{len(cell_lines) * len(treatments)} groups full "
                f"({time.time() - t0:.0f}s)"
            )

    return rows, obs, counts


def build(args):
    import anndata as ad
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    gene_panel = load_gene_panel(args.dataset)
    token_to_col = load_gene_index(gene_panel)

    sample_to_treatment, sample_to_drug = load_sample_metadata(args.plate)
    treatments = [treatment_label(d, args.dose) for d in args.drugs]

    # Controls: whichever DMSO label this plate actually uses.
    control_labels = sorted(
        {t for s, t in sample_to_treatment.items() if sample_to_drug.get(s) == CONTROL_DRUG}
    )
    if not control_labels:
        raise ValueError(f"No {CONTROL_DRUG} control samples found on {args.plate}.")
    print(f"  control label(s): {control_labels}")
    treatments += control_labels

    wanted_samples = {s for s, t in sample_to_treatment.items() if t in set(treatments)}
    if not wanted_samples:
        raise ValueError("None of the requested drugs are present on this plate.")
    print(f"  {len(wanted_samples)} samples cover {len(treatments)} treatments")

    shards = find_plate_shards(fs, args.plate)
    selected = select_row_groups(fs, shards, wanted_samples)

    rows, obs, counts = collect_cells(
        fs, selected, sample_to_treatment, sample_to_drug,
        args.cell_lines, treatments, args.cells_per_group,
        token_to_col, len(gene_panel),
    )
    if not rows:
        raise ValueError("No cells matched the request.")

    thin = [k for k, v in sorted(counts.items()) if v < args.min_cells_per_group]
    if thin:
        print(f"  warning: {len(thin)} (cell_line, treatment) groups have < {args.min_cells_per_group} cells")

    X = sp.vstack(rows, format="csr")
    obs_df = pd.DataFrame(obs)
    cell_names = load_cell_names()
    obs_df["cell_name"] = obs_df["cell_line"].map(cell_names).fillna(obs_df["cell_line"])
    obs_df = obs_df[["cell_line", "cell_name", "drug", "treatment", "tscp_count"]]
    obs_df.index = [f"cell_{i}" for i in range(len(obs_df))]

    adata = ad.AnnData(X=X, obs=obs_df)
    adata.var_names = gene_panel
    adata.uns["build_example_h5ad"] = json.dumps({
        "source": HF_REPO,
        "plate": args.plate,
        "dose": args.dose,
        "cells_per_group": args.cells_per_group,
        "gene_panel": f"splits/{args.dataset}/static_2k_genes.json",
    })

    adata.write_h5ad(args.out)
    print(
        f"\nWrote {args.out}: {adata.n_obs} cells x {adata.n_vars} genes, "
        f"{obs_df['cell_line'].nunique()} cell lines, {obs_df['treatment'].nunique()} treatments"
    )
    return adata


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="sample_tahoe.h5ad", help="output h5ad path")
    p.add_argument("--plate", default="plate1", help="Tahoe plate (plate encodes dose; plate1 is 0.05 uM)")
    p.add_argument("--dose", type=float, default=0.05, help="dose in uM, must match the plate")
    p.add_argument("--dataset", default="tahoe", help="gene panel to project onto (splits/<dataset>/)")
    p.add_argument("--cell-lines", nargs="+", default=WALKTHROUGH_CELL_LINES, help="CVCL ids to keep")
    p.add_argument("--drugs", nargs="+", default=WALKTHROUGH_DRUGS, help="drug names to keep")
    p.add_argument("--cells-per-group", type=int, default=200, help="cap per (cell_line, treatment)")
    p.add_argument("--min-cells-per-group", type=int, default=25, help="warn below this many cells")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
