"""Noise-ceiling calibration: score sub-sample A against sub-sample B.

Groups A and B are independent sub-samples of the same experiments, so A is an
optimal predictor of B (up to sampling noise) and vice versa. Running each as
"prediction" against the other as "truth" gives a calibration ceiling for the
eight metrics logged by eval_splits.py.

Each dataset declares its A/B paths in splits/<dataset>/dataset.toml under
[data.variants.{A,B}]. Only datasets that declare both variants are considered.
Tahoe's legacy long-format cell_eval is handled via the `cell_eval_format =
"tahoe_long"` flag on its variants; other datasets use the shared wide loader.

Usage:
    # all datasets with [data.variants], all non-titration splits
    uv run python scripts/eval_a_vs_b.py

    # only specific datasets (all their splits)
    uv run python scripts/eval_a_vs_b.py --datasets parse

    # specific "<dataset>/<split>" names (overrides --datasets)
    uv run python scripts/eval_a_vs_b.py --splits parse/donor_split_0 parse/donor_split_1

    # merge into an existing output file instead of overwriting (preserves
    # per-split entries from previous runs that aren't in this invocation)
    uv run python scripts/eval_a_vs_b.py --splits parse/fewshot_donor_split_0 --merge

    # override the gene list (applies to all selected datasets)
    uv run python scripts/eval_a_vs_b.py --gene-list path/to/alt_genes.json

    # + wandb
    uv run python scripts/eval_a_vs_b.py --wandb
"""
import argparse
import glob
import json
import os
import sys
import tomllib
from collections import defaultdict

import numpy as np
import pandas as pd

from rhaister import prepare_combined as pc

METRIC_KEYS = [
    "pdex_static/pearson_delta_mean",
    "pdex_static/auprc_p05",
    "state/pearson_delta_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
    "state/pr_auc_mean",
    "state/spearman_lfc_sig_mean",
    "state/discrimination_mean",
]


# === variant cell_eval loaders ===

def _format_tahoe_treatment(drug: str, dose) -> str:
    # pdex target column: "[('5-Azacytidine', 0.05, 'uM')]" — drug trimmed, dose as float repr.
    return f"[('{drug.strip()}', {float(dose)}, 'uM')]"


def _load_cell_eval_tahoe_long(variant_cfg: dict, static_genes: list[str]) -> pd.DataFrame:
    """Load legacy long-format Tahoe cell_eval (group_{A,B}_celleval_plate*.parquet)
    and pivot to the wide (cell_line, treatment, <gene_cols>) shape. Treatment is
    composed from drug+dose to match the pdex `target` formatting.
    """
    pattern = os.path.join(variant_cfg["cell_eval_dir"], variant_cfg["cell_eval_glob"])
    files = sorted(glob.glob(pattern))
    assert files, f"No cell_eval parquet files at {pattern}"
    print(f"  found {len(files)} cell_eval files")

    gene_set = set(static_genes)
    parts = []
    for f in files:
        df = pd.read_parquet(
            f, columns=["gene_name", "log2FoldChange", "cell_line", "drug", "dose", "plate"],
        )
        df = df[df["gene_name"].isin(gene_set)]
        parts.append(df)
    long_df = pd.concat(parts, ignore_index=True)
    del parts

    long_df["treatment"] = [
        _format_tahoe_treatment(d, s) for d, s in zip(long_df["drug"], long_df["dose"])
    ]
    wide = long_df.pivot_table(
        index=["cell_line", "treatment"],
        columns="gene_name",
        values="log2FoldChange",
        aggfunc="mean",
    )
    wide.reset_index(inplace=True)
    wide.columns.name = None
    gene_cols = [c for c in wide.columns if c not in ("cell_line", "treatment")]
    wide[gene_cols] = wide[gene_cols].fillna(0.0)
    print(f"  {len(wide)} (cell_line, treatment) pairs, {len(gene_cols)} genes")
    return wide


def _load_cell_eval_for_variant(variant_cfg: dict, static_genes: list[str]) -> pd.DataFrame:
    fmt = variant_cfg.get("cell_eval_format", "wide")
    if fmt == "wide":
        return pc.load_cell_eval_data(variant_cfg)
    if fmt == "tahoe_long":
        return _load_cell_eval_tahoe_long(variant_cfg, static_genes)
    raise ValueError(f"unknown cell_eval_format: {fmt!r}")


# === per-dataset load + align ===

def load_variant(dataset: str, variant: str, gene_list_override: str | None = None) -> dict:
    """Load + align pdex and cell_eval for one dataset variant.

    If `gene_list_override` is set, it replaces `cfg["gene_list"]` so that both
    the pdex and cell_eval loaders read the alternate list.
    """
    cfg = pc.load_variant_config(dataset, variant)
    if gene_list_override is not None:
        cfg["gene_list"] = gene_list_override
    with open(cfg["gene_list"]) as f:
        static_genes = json.load(f)

    print(f"  pdex path:      {cfg['pdex_path']}")
    print(f"  cell_eval src:  {cfg.get('cell_eval_path') or cfg.get('cell_eval_dir')}")
    print(f"  cell_eval fmt:  {cfg.get('cell_eval_format', 'wide')}")

    fc_df, pv_df, fdr_df, _ = pc.load_pdex_data(cfg)
    pdex_genes = pc.get_gene_columns(fc_df)
    print(f"  pdex: {len(fc_df)} rows, {len(pdex_genes)} genes")

    ce_df = _load_cell_eval_for_variant(cfg, static_genes)
    ce_genes = pc.get_gene_columns(ce_df)

    fc_agg = pc.aggregate_replicates(fc_df)
    pv_agg = pc.aggregate_replicates(pv_df)
    fdr_agg = pc.aggregate_replicates(fdr_df)
    ce_agg = pc.aggregate_replicates(ce_df)
    del fc_df, pv_df, fdr_df, ce_df
    print(f"  aggregated: pdex={len(fc_agg)}, cell_eval={len(ce_agg)}")

    shared_genes = [g for g in static_genes if g in set(pdex_genes) and g in set(ce_genes)]
    print(f"  pdex×cell_eval genes: {len(shared_genes)}/{len(static_genes)}")
    fc_agg = fc_agg[["cell_line", "treatment"] + shared_genes]
    pv_agg = pv_agg[["cell_line", "treatment"] + shared_genes]
    fdr_agg = fdr_agg[["cell_line", "treatment"] + shared_genes]
    ce_agg = ce_agg[["cell_line", "treatment"] + shared_genes]

    key_pdex = fc_agg[["cell_line", "treatment"]].copy()
    key_pdex["_in_pdex"] = True
    ce_agg = ce_agg.merge(key_pdex, on=["cell_line", "treatment"], how="inner").drop(columns="_in_pdex")

    key_ce = ce_agg[["cell_line", "treatment"]].copy()
    key_ce["_in_ce"] = True
    fc_agg = fc_agg.merge(key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")
    pv_agg = pv_agg.merge(key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")
    fdr_agg = fdr_agg.merge(key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")

    for df in (fc_agg, pv_agg, fdr_agg, ce_agg):
        df.sort_values(["cell_line", "treatment"], inplace=True)
        df.reset_index(drop=True, inplace=True)
    print(f"  aligned: {len(fc_agg)} shared (cell_line, treatment) pairs")

    return {"fc": fc_agg, "pv": pv_agg, "fdr": fdr_agg, "ce": ce_agg, "gene_cols": shared_genes}


def reconcile_groups(a: dict, b: dict) -> tuple[dict, dict]:
    """Restrict two loaded groups to the intersection of rows AND gene columns."""
    key_a = a["fc"][["cell_line", "treatment"]].copy()
    key_a["_ka"] = True
    key_b = b["fc"][["cell_line", "treatment"]].copy()
    key_b["_kb"] = True
    shared_keys = (
        key_a.merge(key_b, on=["cell_line", "treatment"], how="inner")
        [["cell_line", "treatment"]]
        .sort_values(["cell_line", "treatment"])
        .reset_index(drop=True)
    )

    shared_genes = [g for g in a["gene_cols"] if g in set(b["gene_cols"])]
    if len(shared_genes) != len(a["gene_cols"]) or len(shared_genes) != len(b["gene_cols"]):
        print(f"  A×B genes: {len(shared_genes)} (A={len(a['gene_cols'])}, B={len(b['gene_cols'])})")
    if len(shared_keys) != len(a["fc"]) or len(shared_keys) != len(b["fc"]):
        print(f"  A×B rows: {len(shared_keys)} (A={len(a['fc'])}, B={len(b['fc'])})")

    def align(group):
        out = {}
        for k in ("fc", "pv", "fdr", "ce"):
            merged = shared_keys.merge(group[k], on=["cell_line", "treatment"], how="left")
            out[k] = merged[["cell_line", "treatment"] + shared_genes].reset_index(drop=True)
        out["gene_cols"] = shared_genes
        return out

    return align(a), align(b)


def split_to_matrices(group: dict, split_info: dict) -> dict:
    _, test_fc = pc.make_splits(group["fc"], split_info)
    _, test_pv = pc.make_splits(group["pv"], split_info)
    _, test_fdr = pc.make_splits(group["fdr"], split_info)
    _, test_ce = pc.make_splits(group["ce"], split_info)

    gene_cols = group["gene_cols"]
    test_cells, test_tr, Y_test = pc.to_matrices(test_fc, gene_cols)
    _, _, P_test = pc.to_matrices(test_pv, gene_cols)
    _, _, F_test = pc.to_matrices(test_fdr, gene_cols)
    _, _, D_test = pc.to_matrices(test_ce, gene_cols)
    D_test = np.nan_to_num(D_test, nan=0.0)

    return {
        "Y": Y_test, "P": P_test, "F": F_test, "D": D_test,
        "cells": test_cells, "treatments": test_tr,
        "gene_cols": gene_cols,
    }


def score_pair(a_test: dict, b_test: dict) -> tuple[dict, dict]:
    assert np.array_equal(a_test["cells"], b_test["cells"]), "row order mismatch"
    assert np.array_equal(a_test["treatments"], b_test["treatments"]), "treatment order mismatch"
    assert a_test["Y"].shape == b_test["Y"].shape, (
        f"shape mismatch: A={a_test['Y'].shape}, B={b_test['Y'].shape}"
    )

    print("  scoring A-as-pred vs B-as-truth...")
    ab = pc.evaluate(
        Y_true=b_test["Y"], Y_pred=a_test["Y"],
        P_true=b_test["P"], P_pred=a_test["P"],
        D_true=b_test["D"], D_pred=a_test["D"],
        F_true=b_test["F"], F_pred=a_test["F"],
        test_cells=b_test["cells"], test_treatments=b_test["treatments"],
        gene_cols=b_test["gene_cols"], compute_discrimination=True,
    )
    print("  scoring B-as-pred vs A-as-truth...")
    ba = pc.evaluate(
        Y_true=a_test["Y"], Y_pred=b_test["Y"],
        P_true=a_test["P"], P_pred=b_test["P"],
        D_true=a_test["D"], D_pred=b_test["D"],
        F_true=a_test["F"], F_pred=b_test["F"],
        test_cells=a_test["cells"], test_treatments=a_test["treatments"],
        gene_cols=a_test["gene_cols"], compute_discrimination=True,
    )
    return ab, ba


# === discovery ===

def _has_variants(dataset: str) -> bool:
    cfg_path = os.path.join(pc.SPLITS_DIR, dataset, "dataset.toml")
    if not os.path.isfile(cfg_path):
        return False
    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)
    variants = raw.get("data", {}).get("variants", {})
    return "A" in variants and "B" in variants


def discover_datasets() -> list[str]:
    """Datasets under splits/ that declare both A and B variants."""
    out = []
    for name in sorted(os.listdir(pc.SPLITS_DIR)):
        if not os.path.isdir(os.path.join(pc.SPLITS_DIR, name)):
            continue
        if _has_variants(name):
            out.append(name)
    return out


def discover_splits(dataset: str) -> list[str]:
    """Subdirs of splits/<dataset>/ that contain the default_split_config file
    and aren't titration variants (titrations share a ceiling with their parent)."""
    cfg = pc.load_dataset_config(dataset)
    dataset_dir = cfg["dataset_dir"]
    default_name = cfg["default_split_config"]
    out = []
    for name in sorted(os.listdir(dataset_dir)):
        if "titration" in name:
            continue
        sub = os.path.join(dataset_dir, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, default_name)):
            out.append(f"{dataset}/{name}")
    return out


# === output assembly ===

def build_output(per_dataset: dict, *, print_summary: bool = True) -> dict:
    """Derive metadata/summary/asymmetry blocks from a per_dataset mapping.

    `per_dataset[dataset]` must have shape
    `{n_genes_shared, n_aligned_pairs, per_split: {split: {a_vs_b, b_vs_a, mean, n_samples}}}`.
    The summary/asymmetry blocks are pure derivations, so this function is the
    single place that knows how to roll them up — used both for fresh runs and
    for `--merge` runs that load prior per_split entries from disk.
    """
    if print_summary:
        print(f"\n{'='*70}\nSUMMARY (mean of A-vs-B and B-vs-A)\n{'='*70}")
    summary = {}
    for dataset, res in per_dataset.items():
        if print_summary:
            print(f"\n{dataset}:")
        split_names = list(res["per_split"].keys())
        dsum = {}
        for key in METRIC_KEYS:
            vals = [res["per_split"][s]["mean"][key] for s in split_names if key in res["per_split"][s]["mean"]]
            if not vals:
                continue
            dsum[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_split": vals}
            if print_summary:
                per = "  ".join(f"{v:.4f}" for v in vals)
                print(f"  {key:<42} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   {per}")
        summary[dataset] = dsum

    asymmetry = {}
    for dataset, res in per_dataset.items():
        das = {}
        for key in METRIC_KEYS:
            diffs = [
                abs(res["per_split"][s]["a_vs_b"][key] - res["per_split"][s]["b_vs_a"][key])
                for s in res["per_split"]
                if key in res["per_split"][s]["a_vs_b"] and key in res["per_split"][s]["b_vs_a"]
            ]
            if diffs:
                das[key] = {"max": max(diffs), "per_split": diffs}
        asymmetry[dataset] = das

    return {
        "metadata": {
            "datasets": list(per_dataset.keys()),
            "splits": [s for d in per_dataset.values() for s in d["per_split"]],
        },
        "per_dataset": per_dataset,
        "summary": summary,
        "asymmetry": asymmetry,
    }


def merge_per_dataset(existing: dict, fresh: dict) -> dict:
    """Merge fresh per_dataset into existing; fresh per_split entries override."""
    merged = {ds: {**v, "per_split": dict(v["per_split"])} for ds, v in existing.items()}
    for dataset, res in fresh.items():
        if dataset in merged:
            merged[dataset]["n_genes_shared"] = res["n_genes_shared"]
            merged[dataset]["n_aligned_pairs"] = res["n_aligned_pairs"]
            merged[dataset]["per_split"].update(res["per_split"])
        else:
            merged[dataset] = {**res, "per_split": dict(res["per_split"])}
    return merged


# === main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to evaluate (default: all with [data.variants])")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="'<dataset>/<split>' names; overrides --datasets")
    parser.add_argument("--wandb", action="store_true", help="Log results to WandB")
    parser.add_argument(
        "--gene-list",
        default=None,
        help="Override the gene-list JSON used for all selected datasets "
             "(default: each dataset's static_2k_genes.json from dataset.toml). "
             "Use this for exploring alternative gene sets.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results into an existing --output file (per_split entries "
             "from this run override matching keys; other splits/datasets are "
             "preserved). Default behavior overwrites the file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a_vs_b_results.json"),
        help="Path to write the structured JSON result file",
    )
    args = parser.parse_args()

    by_dataset: dict[str, list[str]] = defaultdict(list)
    if args.splits is not None:
        for s in args.splits:
            ds, sp = pc.parse_split_name(s)
            by_dataset[ds].append(f"{ds}/{sp}")
        datasets = sorted(by_dataset.keys())
    else:
        datasets = args.datasets or discover_datasets()
        for d in datasets:
            by_dataset[d] = discover_splits(d)

    print(f"Datasets to evaluate: {datasets}")
    for d in datasets:
        print(f"  {d}: {by_dataset[d]}")

    results = {}
    for dataset in datasets:
        splits = by_dataset[dataset]
        if not splits:
            print(f"[skip] {dataset}: no splits")
            continue
        if not _has_variants(dataset):
            print(f"[skip] {dataset}: no [data.variants.{{A,B}}] in dataset.toml")
            continue

        print(f"\n{'='*70}\n{dataset.upper()}: LOADING GROUP A\n{'='*70}")
        a = load_variant(dataset, "A", gene_list_override=args.gene_list)
        print(f"\n{'='*70}\n{dataset.upper()}: LOADING GROUP B\n{'='*70}")
        b = load_variant(dataset, "B", gene_list_override=args.gene_list)
        print(f"\n{'='*70}\n{dataset.upper()}: RECONCILING\n{'='*70}")
        a, b = reconcile_groups(a, b)

        per_split = {}
        for split in splits:
            print(f"\n{'='*70}\n{split}\n{'='*70}")
            split_info = pc.load_split(split)
            a_test = split_to_matrices(a, split_info)
            b_test = split_to_matrices(b, split_info)
            print(f"  test pairs: {len(a_test['cells'])}")
            if len(a_test["cells"]) == 0:
                print("  [skip] no test samples after applying split to A∩B")
                continue
            ab, ba = score_pair(a_test, b_test)
            mean = {k: 0.5 * (ab[k] + ba[k]) for k in METRIC_KEYS if k in ab and k in ba}
            per_split[split] = {
                "a_vs_b": ab, "b_vs_a": ba, "mean": mean,
                "n_samples": ab.get("n_samples", 0),
            }

        results[dataset] = {
            "n_genes_shared": len(a["gene_cols"]),
            "n_aligned_pairs": len(a["fc"]),
            "per_split": per_split,
        }

        del a, b

    if args.wandb:
        import wandb
        for dataset, res in results.items():
            for split, r in res["per_split"].items():
                wandb.init(
                    project="perturbation-eval", job_type="a_vs_b",
                    name=f"a_vs_b_{split.replace('/', '_')}",
                    config={"dataset": dataset, "split": split, "n_samples": r["n_samples"]},
                    reinit=True,
                )
                wandb.log({k: r["mean"][k] for k in METRIC_KEYS if k in r["mean"]})
                wandb.finish()

    if args.merge and os.path.isfile(args.output):
        with open(args.output) as f:
            prior = json.load(f)
        prior_per_dataset = prior.get("per_dataset", {})
        fresh_splits = [s for d in results.values() for s in d["per_split"]]
        print(f"\nMerging {len(fresh_splits)} fresh split(s) into {args.output} "
              f"(prior had {sum(len(v.get('per_split', {})) for v in prior_per_dataset.values())} splits)")
        per_dataset = merge_per_dataset(prior_per_dataset, results)
    else:
        per_dataset = results

    output = build_output(per_dataset)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nStructured results written to {args.output}")


if __name__ == "__main__":
    main()
