"""
prepare_cell_eval.py — IMMUTABLE. Data loading, split parsing, evaluation, caching
for cell_eval DE data (wide-format parquet, no p-values).

Same interface as prepare.py but loads from cell_eval parquet files.
Since cell_eval has no p-values, P_train/P_test are filled with NaN.
"""

import datetime
import glob
import json
import os
import tomllib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# === CONSTANTS ===
DATA_DIR = "/nvme-shared/Data/tahoe_100m_de_cell_eval"
SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")
CACHE_DIR = "/tmp/tahoe_cache_cell_eval"
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_cell_eval.tsv")


def load_data():
    """Load all parquet files, return concatenated DataFrame."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "plate_plate*.parquet")))
    assert len(files) > 0, f"No parquet files found in {DATA_DIR}"
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def get_gene_columns(df):
    """Return list of gene column names (everything except cell_line, treatment)."""
    return [c for c in df.columns if c not in ("cell_line", "treatment")]


def aggregate_replicates(df):
    """Mean-aggregate over replicates: group by (cell_line, treatment)."""
    gene_cols = get_gene_columns(df)
    return df.groupby(["cell_line", "treatment"], as_index=False)[gene_cols].mean()


def load_split(split_name="tahoe/5_holdout"):
    """Parse the TOML split file. Accepts "<dataset>/<split>" or legacy "<dataset>_<split>"."""
    from rhaister.prepare_combined import load_dataset_config, parse_split_name

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
    """Split aggregated DataFrame into train and test."""
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
    """Returns a one-shot evaluation function."""
    called = [False]

    def evaluate_test(Y_pred, P_pred=None):
        if called[0]:
            raise RuntimeError(
                "evaluate_test() was already called. "
                "Test data can only be evaluated once — no hyperparameter tuning on test data."
            )
        called[0] = True
        return evaluate(Y_test, Y_pred)

    return evaluate_test


def evaluate(Y_true, Y_pred, P_true=None, P_pred=None):
    """Pearson per row. P_true/P_pred accepted for interface compat but ignored."""
    n_samples = Y_true.shape[0]
    per_obs = []
    for i in range(n_samples):
        r, _ = pearsonr(Y_true[i], Y_pred[i])
        per_obs.append(r if np.isfinite(r) else 0.0)
    per_obs = np.array(per_obs)
    return {
        "cell_eval_static/pearson_delta_mean": float(np.mean(per_obs)),
        "cell_eval_static/pearson_delta_median": float(np.median(per_obs)),
        "cell_eval_static/pearson_delta_std": float(np.std(per_obs)),
        "n_samples": n_samples,
    }


def log_result(experiment_name, metrics, notes=""):
    """Append result to results_cell_eval.tsv."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("timestamp\texperiment\tmean_pearson\tmedian_pearson\tstd_pearson\tn_samples\tnotes\n")
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{datetime.datetime.now().isoformat()}\t{experiment_name}\t"
            f"{metrics['cell_eval_static/pearson_delta_mean']:.6f}\t{metrics['cell_eval_static/pearson_delta_median']:.6f}\t"
            f"{metrics['cell_eval_static/pearson_delta_std']:.6f}\t{metrics['n_samples']}\t{notes}\n"
        )


def _cache_path(split_name):
    return os.path.join(CACHE_DIR, split_name)


def _newest_parquet_mtime():
    files = glob.glob(os.path.join(DATA_DIR, "plate_plate*.parquet"))
    return max(os.path.getmtime(f) for f in files)


def _cache_is_valid(split_name):
    cp = _cache_path(split_name)
    meta_path = os.path.join(cp, "meta.json")
    if not os.path.exists(meta_path):
        return False
    cache_mtime = os.path.getmtime(meta_path)
    return cache_mtime > _newest_parquet_mtime()


def _save_cache(split_name, data):
    cp = _cache_path(split_name)
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


def _load_cache(split_name):
    cp = _cache_path(split_name)

    Y_train = np.load(os.path.join(cp, "Y_train.npy"), mmap_mode="r")
    Y_test = np.load(os.path.join(cp, "Y_test.npy"), mmap_mode="r")
    P_train = np.load(os.path.join(cp, "P_train.npy"), mmap_mode="r")
    P_test = np.load(os.path.join(cp, "P_test.npy"), mmap_mode="r")

    with open(os.path.join(cp, "meta.json")) as f:
        meta = json.load(f)

    return {
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


def prepare_all(split_name="tahoe_5_holdout"):
    """One-call function: load, aggregate, split, cache, return matrices + metadata."""
    if _cache_is_valid(split_name):
        print(f"Loading from cache ({_cache_path(split_name)})...")
        return _load_cache(split_name)

    print("Loading parquet files...")
    df = load_data()
    gene_cols = get_gene_columns(df)
    print(f"Loaded {len(df)} rows, {len(gene_cols)} genes")

    print("Aggregating replicates...")
    df_agg = aggregate_replicates(df)
    print(f"Aggregated to {len(df_agg)} unique (cell_line, treatment) pairs")

    print("Parsing split...")
    split_info = load_split(split_name)
    train_df, test_df = make_splits(df_agg, split_info)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_cells, train_tr, Y_train = to_matrices(train_df, gene_cols)
    test_cells, test_tr, Y_test = to_matrices(test_df, gene_cols)

    # No p-values in cell_eval — fill with 1.0 (not significant)
    # This is neutral for p-value weighting: -log10(1.0) = 0, so weight = 1
    P_train = np.ones_like(Y_train)
    P_test = np.ones_like(Y_test)

    # Build index maps from ALL unique values in full dataset
    all_cells = sorted(df_agg["cell_line"].unique())
    all_treatments = sorted(df_agg["treatment"].unique())
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
    _save_cache(split_name, data)

    # Reload with mmap, wrapping Y_test in one-shot evaluator
    return _load_cache(split_name)
