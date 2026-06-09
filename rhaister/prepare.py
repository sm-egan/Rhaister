"""
prepare.py — IMMUTABLE. Data loading, split parsing, evaluation, caching.
Do NOT modify this file. Only train.py should be modified by the agent.
"""

import datetime
import json
import os
import tomllib

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score

from rhaister.prepare_combined import load_dataset_config, parse_split_name

# === CONSTANTS ===
SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")
CACHE_DIR = "/tmp/tahoe_cache"
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results.tsv")

# Back-compat: default to tahoe dataset for module-level constants that other
# code may reference. Dataset-specific loaders below resolve these per-split.
_DEFAULT_DATASET_CFG = load_dataset_config("tahoe")
DATA_PATH = _DEFAULT_DATASET_CFG["pdex_path"]
GENE_LIST = _DEFAULT_DATASET_CFG["gene_list"]


def load_data(dataset_cfg=None):
    """Load pdex parquet (long format), filter to static gene set, pivot to wide."""
    if dataset_cfg is None:
        dataset_cfg = _DEFAULT_DATASET_CFG
    with open(dataset_cfg["gene_list"]) as f:
        static_genes = json.load(f)

    print("  Reading and filtering parquet...")
    table = pq.read_table(
        dataset_cfg["pdex_path"],
        columns=["target", "feature", "fold_change", "p_value", "cell_line"],
        filters=[("feature", "in", static_genes)],
    )
    long_df = table.to_pandas()
    del table
    print(f"  Filtered to {len(long_df)} rows ({long_df['feature'].nunique()} genes)")

    # fold_change is log2(target_mean / ref_mean); handle inf from log2(0)
    long_df["fold_change"] = long_df["fold_change"].replace([np.inf, -np.inf], np.nan)

    # Pivot to wide format: mean fold_change, min p_value across plates
    print("  Pivoting to wide format...")
    fc_wide = long_df.pivot_table(
        index=["cell_line", "target"],
        columns="feature",
        values="fold_change",
        aggfunc="mean",
    )
    pv_wide = long_df.pivot_table(
        index=["cell_line", "target"],
        columns="feature",
        values="p_value",
        aggfunc="min",
    )
    del long_df

    # Align indices and gene columns
    fc_wide = fc_wide.reset_index()
    pv_wide = pv_wide.reset_index()
    fc_wide.rename(columns={"target": "treatment"}, inplace=True)
    pv_wide.rename(columns={"target": "treatment"}, inplace=True)
    fc_wide.columns.name = None
    pv_wide.columns.name = None

    # Fill NaN: fold_change → 0.0 (no effect), p_value → 1.0 (not significant)
    gene_cols = [c for c in fc_wide.columns if c not in ("cell_line", "treatment")]
    fc_wide[gene_cols] = fc_wide[gene_cols].fillna(0.0)
    pv_wide[gene_cols] = pv_wide[gene_cols].fillna(1.0)

    return fc_wide, pv_wide


def get_gene_columns(df):
    """Return list of gene column names (everything except cell_line, treatment)."""
    return [c for c in df.columns if c not in ("cell_line", "treatment")]


def aggregate_replicates(df):
    """Mean-aggregate over replicates: group by (cell_line, treatment).
    For p-value DataFrames, this is a no-op since pivoting already aggregated."""
    gene_cols = get_gene_columns(df)
    return df.groupby(["cell_line", "treatment"], as_index=False)[gene_cols].mean()


def load_split(split_name="tahoe/5_holdout"):
    """Parse the TOML split file. Accepts "<dataset>/<split>" or legacy "<dataset>_<split>".
    Returns dict with:
       - holdout_cells: list of cell_line IDs
       - test_treatments: dict mapping cell_line -> set of test treatment strings
    """
    dataset, split = parse_split_name(split_name)
    dataset_cfg = load_dataset_config(dataset)
    toml_path = os.path.join(SPLITS_DIR, dataset, split, dataset_cfg["default_split_config"])
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    fewshot = config["fewshot"]
    holdout_cells = []
    test_treatments = {}
    for key, v in fewshot.items():
        cell = key.split(".")[-1]
        holdout_cells.append(cell)
        test_treatments[cell] = set(v["test"])

    return {
        "holdout_cells": holdout_cells,
        "test_treatments": test_treatments,
    }


def make_splits(df_agg, split_info):
    """Split aggregated DataFrame into train and test.
    - test: held-out cells' test treatments
    - train: everything else
    """
    holdout = set(split_info["holdout_cells"])
    test_tr = split_info["test_treatments"]

    test_mask = df_agg.apply(
        lambda r: r["cell_line"] in holdout and r["treatment"] in test_tr.get(r["cell_line"], set()),
        axis=1,
    )
    train_mask = ~test_mask

    return df_agg[train_mask].reset_index(drop=True), df_agg[test_mask].reset_index(drop=True)


def to_matrices(df, gene_cols):
    """Convert DataFrame to (cell_lines, treatments, Y) numpy arrays."""
    return (
        df["cell_line"].values,
        df["treatment"].values,
        df[gene_cols].values.astype(np.float64),
    )


def make_evaluator(Y_test, P_test):
    """Returns a one-shot evaluation function. Calling it more than once raises an error,
    preventing hyperparameter tuning on test data."""
    called = [False]

    def evaluate_test(Y_pred, P_pred=None):
        if called[0]:
            raise RuntimeError(
                "evaluate_test() was already called. "
                "Test data can only be evaluated once — no hyperparameter tuning on test data."
            )
        called[0] = True
        return evaluate(Y_test, Y_pred, P_test, P_pred)

    return evaluate_test


def evaluate(Y_true, Y_pred, P_true=None, P_pred=None):
    """Evaluate fold change predictions (Pearson per row) and optionally
    p-value predictions (AUPRC with p < 0.05 as positive class)."""
    n_samples = Y_true.shape[0]
    per_obs = []
    for i in range(n_samples):
        r, _ = pearsonr(Y_true[i], Y_pred[i])
        per_obs.append(r if np.isfinite(r) else 0.0)
    per_obs = np.array(per_obs)
    results = {
        "pdex_static/pearson_delta_mean": float(np.mean(per_obs)),
        "pdex_static/pearson_delta_median": float(np.median(per_obs)),
        "pdex_static/pearson_delta_std": float(np.std(per_obs)),
        "n_samples": n_samples,
    }

    if P_true is not None and P_pred is not None:
        P_true_arr = np.asarray(P_true)
        P_pred_arr = np.asarray(P_pred)
        # Binary labels: 1 if true p < 0.05 (significant), 0 otherwise
        labels = (P_true_arr < 0.05).astype(np.int32).ravel()
        # Score: predicted probability of significance = 1 - predicted p-value
        scores = (1.0 - P_pred_arr).ravel()
        results["pdex_static/auprc_p05"] = float(average_precision_score(labels, scores))

    return results


def log_result(experiment_name, metrics, notes=""):
    """Append result to results.tsv and log to WandB."""
    # TSV logging
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("timestamp\texperiment\tmean_pearson\tmedian_pearson\tstd_pearson\tn_samples\tnotes\n")
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{datetime.datetime.now().isoformat()}\t{experiment_name}\t"
            f"{metrics['pdex_static/pearson_delta_mean']:.6f}\t{metrics['pdex_static/pearson_delta_median']:.6f}\t"
            f"{metrics['pdex_static/pearson_delta_std']:.6f}\t{metrics['n_samples']}\t{notes}\n"
        )

    # WandB logging
    try:
        import wandb

        wandb.init(
            project="rhaister-autoresearch",
            name=experiment_name,
            config={"notes": notes},
            reinit=True,
        )
        wandb.log(metrics)
        wandb.finish()
    except Exception as e:
        print(f"WandB logging failed: {e}")


def _cache_path(dataset, split):
    return os.path.join(CACHE_DIR, dataset, split)


def _newest_parquet_mtime(dataset_cfg):
    return os.path.getmtime(dataset_cfg["pdex_path"])


def _cache_is_valid(dataset, split, dataset_cfg):
    cp = _cache_path(dataset, split)
    meta_path = os.path.join(cp, "meta.json")
    if not os.path.exists(meta_path):
        return False
    cache_mtime = os.path.getmtime(meta_path)
    return cache_mtime > _newest_parquet_mtime(dataset_cfg)


def _save_cache(dataset, split, data):
    cp = _cache_path(dataset, split)
    os.makedirs(cp, exist_ok=True)

    np.save(os.path.join(cp, "Y_train.npy"), data["Y_train"])
    np.save(os.path.join(cp, "Y_test.npy"), data["Y_test"])
    np.save(os.path.join(cp, "P_train.npy"), data["P_train"])
    np.save(os.path.join(cp, "P_test.npy"), data["P_test"])

    meta = {
        "train_cells": data["train_cells"].tolist(),
        "train_treatments": data["train_treatments"].tolist(),
        "test_cells": data["test_cells"].tolist(),
        "test_treatments": data["test_treatments"].tolist(),
        "gene_cols": data["gene_cols"],
        "cell_to_idx": data["cell_to_idx"],
        "treat_to_idx": data["treat_to_idx"],
        "n_cells": data["n_cells"],
        "n_treatments": data["n_treatments"],
    }
    with open(os.path.join(cp, "meta.json"), "w") as f:
        json.dump(meta, f)


def _load_cache(dataset, split):
    cp = _cache_path(dataset, split)

    Y_train = np.load(os.path.join(cp, "Y_train.npy"), mmap_mode="r")
    Y_test = np.load(os.path.join(cp, "Y_test.npy"), mmap_mode="r")
    P_train = np.load(os.path.join(cp, "P_train.npy"), mmap_mode="r")
    P_test = np.load(os.path.join(cp, "P_test.npy"), mmap_mode="r")

    with open(os.path.join(cp, "meta.json")) as f:
        meta = json.load(f)

    result = {
        "Y_train": Y_train,
        "P_train": P_train,
        "evaluate_test": make_evaluator(Y_test, P_test),
        "P_test": P_test,
        "n_test": Y_test.shape[0],
        "train_cells": np.array(meta["train_cells"]),
        "train_treatments": np.array(meta["train_treatments"]),
        "test_cells": np.array(meta["test_cells"]),
        "test_treatments": np.array(meta["test_treatments"]),
        "gene_cols": meta["gene_cols"],
        "cell_to_idx": meta["cell_to_idx"],
        "treat_to_idx": meta["treat_to_idx"],
        "n_cells": meta["n_cells"],
        "n_treatments": meta["n_treatments"],
    }

    # Load ref_mean if cached
    r_train_path = os.path.join(cp, "R_train.npy")
    r_test_path = os.path.join(cp, "R_test.npy")
    if os.path.exists(r_train_path) and os.path.exists(r_test_path):
        result["R_train"] = np.load(r_train_path, mmap_mode="r")
        result["R_test"] = np.load(r_test_path, mmap_mode="r")

    return result


def prepare_all(split_name="tahoe/5_holdout"):
    """One-call function: load, aggregate, split, cache, return matrices + metadata."""
    dataset, split = parse_split_name(split_name)
    dataset_cfg = load_dataset_config(dataset)

    if _cache_is_valid(dataset, split, dataset_cfg):
        print(f"Loading from cache ({_cache_path(dataset, split)})...")
        return _load_cache(dataset, split)

    print("Loading parquet files...")
    fc_df, pv_df = load_data(dataset_cfg)
    gene_cols = get_gene_columns(fc_df)
    print(f"Loaded {len(fc_df)} rows, {len(gene_cols)} genes")

    print("Aggregating replicates...")
    fc_agg = aggregate_replicates(fc_df)
    pv_agg = aggregate_replicates(pv_df)
    print(f"Aggregated to {len(fc_agg)} unique (cell_line, treatment) pairs")

    print("Parsing split...")
    split_info = load_split(split_name)
    train_fc, test_fc = make_splits(fc_agg, split_info)
    train_pv, test_pv = make_splits(pv_agg, split_info)
    print(f"Train: {len(train_fc)}, Test: {len(test_fc)}")

    train_cells, train_tr, Y_train = to_matrices(train_fc, gene_cols)
    test_cells, test_tr, Y_test = to_matrices(test_fc, gene_cols)
    _, _, P_train = to_matrices(train_pv, gene_cols)
    _, _, P_test = to_matrices(test_pv, gene_cols)

    # Build index maps from ALL unique values in full dataset
    all_cells = sorted(fc_agg["cell_line"].unique())
    all_treatments = sorted(fc_agg["treatment"].unique())
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    treat_to_idx = {t: i for i, t in enumerate(all_treatments)}

    data = {
        "train_cells": train_cells,
        "train_treatments": train_tr,
        "Y_train": Y_train,
        "P_train": P_train,
        "test_cells": test_cells,
        "test_treatments": test_tr,
        "Y_test": Y_test,
        "P_test": P_test,
        "gene_cols": gene_cols,
        "cell_to_idx": cell_to_idx,
        "treat_to_idx": treat_to_idx,
        "n_cells": len(all_cells),
        "n_treatments": len(all_treatments),
    }

    print("Saving cache...")
    _save_cache(dataset, split, data)

    # Reload with mmap, wrapping Y_test in one-shot evaluator
    return _load_cache(dataset, split)
