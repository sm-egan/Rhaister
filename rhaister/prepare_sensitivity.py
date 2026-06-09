"""
prepare_sensitivity.py — Loader and one-shot evaluator for scalar growth-rate
datasets (RIFIVDU, PRISM, and future drug-sensitivity datasets of the same
modality).

Each (cell_line, treatment) observation is a single scalar `growth_rate`.
Treatment IDs are the source `condition` column verbatim — stringified Python
list-of-tuples, same encoding as tahoe targets.

Per-dataset behavior is driven entirely by `splits/<dataset>/dataset.toml`:

  [data]
  parquet_path = "..."

  [data.filters]                 # optional; defaults all-true / empty
  drop_null_growth = true
  drop_multi_drug  = true
  drop_drugs       = ["DMSO_T0"]

  [metrics]
  applicable = ["mse", "mae"]

  [splits]
  default_config = "split.toml"

Source data has occasional drug names containing apostrophes (e.g.
"3'-Fluorobenzylspiperone") which break `ast.literal_eval`. The condition
parser uses a regex that handles those cases.
"""

import datetime
import json
import os
import re
import tomllib

import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
CACHE_ROOT = "/tmp/rhaister_cache_sensitivity"
RESULTS_FILE = os.path.join(REPO_ROOT, "results_sensitivity.jsonl")

# Regex matches "[('drug', dose, 'unit')]" where drug may contain apostrophes
# (the leading "[('" and trailing ", N, 'UNIT')]" anchor it unambiguously).
_SINGLE_RE = re.compile(r"^\[\('(.+)',\s*([\-\d\.eE\+]+),\s*'([^']+)'\)\]$")
_MULTI_PAT = re.compile(r"\),\s*\(")


def _is_multi_drug(s):
    return bool(_MULTI_PAT.search(s))


def _parse_drug(s):
    """Return drug name from a single-drug condition string, or None on miss."""
    m = _SINGLE_RE.match(s)
    return m.group(1) if m else None


def _dataset_dir(dataset):
    return os.path.join(SPLITS_DIR, dataset)


def _resolve_data_root(dataset):
    """Resolve data root for a sensitivity dataset (same logic as prepare_combined)."""
    env_root = os.environ.get("RHAISTER_DATA_ROOT")
    if env_root:
        per_dataset = os.path.join(env_root, dataset)
        if os.path.isdir(per_dataset):
            return per_dataset
        return env_root
    return None


def _resolve_path(path, data_root):
    """Resolve a data path: absolute paths pass through, relative paths are
    joined onto data_root (if set), otherwise returned as-is."""
    if not path or os.path.isabs(path):
        return path
    if data_root is not None:
        resolved = os.path.join(data_root, path)
        if os.path.exists(resolved):
            return resolved
    return path


def _load_dataset_config(dataset):
    with open(os.path.join(_dataset_dir(dataset), "dataset.toml"), "rb") as f:
        return tomllib.load(f)


def _filters(cfg):
    f = cfg.get("data", {}).get("filters", {})
    return {
        "drop_null_growth": bool(f.get("drop_null_growth", True)),
        "drop_multi_drug": bool(f.get("drop_multi_drug", True)),
        "drop_drugs": set(f.get("drop_drugs", []) or []),
    }


def parse_split_name(split_name):
    """Accept 'EmeraldBay/split_0', 'EmeraldBay_split_0', or bare 'split_0' (no default)."""
    if "/" in split_name:
        return tuple(split_name.split("/", 1))
    for d in sorted(os.listdir(SPLITS_DIR)):
        if not os.path.isdir(_dataset_dir(d)):
            continue
        prefix = d + "_"
        if split_name.startswith(prefix):
            return d, split_name[len(prefix):]
    raise ValueError(f"Cannot resolve dataset for split '{split_name}'")


def dataset_metrics(dataset):
    """Metric names declared as applicable in dataset.toml."""
    return list(_load_dataset_config(dataset)["metrics"]["applicable"])


def _apply_filters_and_aggregate(df, filters):
    if filters["drop_null_growth"]:
        df = df[df["growth_rate"].notna()]
    if filters["drop_multi_drug"]:
        is_multi = df["condition"].map(_is_multi_drug)
        df = df.loc[~is_multi]
    if filters["drop_drugs"]:
        drug = df["condition"].map(_parse_drug)
        df = df.loc[~drug.isin(filters["drop_drugs"])]
    return (
        df[["cell_line", "condition", "growth_rate"]]
        .groupby(["cell_line", "condition"], as_index=False)["growth_rate"]
        .mean()
    )


def load_data(dataset):
    """Load source parquet, apply configured filters, aggregate replicates."""
    cfg = _load_dataset_config(dataset)
    data_root = _resolve_data_root(dataset)
    df = pd.read_parquet(_resolve_path(cfg["data"]["parquet_path"], data_root))
    return _apply_filters_and_aggregate(df, _filters(cfg))


def has_variants(dataset):
    """True if dataset.toml declares both A and B variants."""
    cfg = _load_dataset_config(dataset)
    variants = cfg.get("data", {}).get("variants", {})
    return "A" in variants and "B" in variants


def has_primary_screen(dataset):
    """True if dataset.toml declares an external primary-screen parquet."""
    cfg = _load_dataset_config(dataset)
    return "primary_screen_parquet" in cfg.get("data", {}).get("external", {})


def load_primary_screen(dataset):
    """Return a dict {(cell_line, drug, dose_secondary): primary_growth_rate}.

    The primary screen tested each (cell, drug) at essentially one dose
    (~2.5 μM); the parquet matches each primary measurement to one or more
    nearby secondary doses (121 unique dose_secondary values, all clustered
    around 2.5 μM). When multiple primary measurements share a (cell, drug,
    dose_secondary) key, keep the row with smallest abs_log_diff (closest dose
    match). Rows with null primary_growth_rate are dropped. Test rows whose
    secondary dose isn't in this 121-dose subset will have no match and be
    excluded from the baseline.
    """
    cfg = _load_dataset_config(dataset)
    data_root = _resolve_data_root(dataset)
    path = _resolve_path(cfg["data"]["external"]["primary_screen_parquet"], data_root)
    df = pd.read_parquet(path, columns=[
        "cell_line", "drug", "dose_secondary",
        "primary_growth_rate", "abs_log_diff",
    ])
    df = df[df["primary_growth_rate"].notna()]
    df = (
        df.sort_values("abs_log_diff")
          .drop_duplicates(["cell_line", "drug", "dose_secondary"], keep="first")
    )
    return {
        (c, d, float(s)): float(p)
        for c, d, s, p in zip(
            df["cell_line"].to_numpy(),
            df["drug"].to_numpy(),
            df["dose_secondary"].to_numpy(),
            df["primary_growth_rate"].to_numpy(),
        )
    }


def load_variant_data(dataset, variant):
    """Load a single A/B half through the same filter pipeline, aggregate per (cell, cond).

    Returns a DataFrame indexed implicitly by (cell_line, condition); use
    .set_index([...]) at the call site if you need fast lookup.
    """
    cfg = _load_dataset_config(dataset)
    variants = cfg.get("data", {}).get("variants", {})
    if variant not in variants:
        raise KeyError(f"Variant '{variant}' not declared in {dataset}/dataset.toml")
    data_root = _resolve_data_root(dataset)
    df = pd.read_parquet(_resolve_path(variants[variant]["parquet_path"], data_root))
    return _apply_filters_and_aggregate(df, _filters(cfg))


def load_split(split_name):
    dataset, split = parse_split_name(split_name)
    cfg = _load_dataset_config(dataset)
    split_file = cfg["splits"]["default_config"]
    toml_path = os.path.join(_dataset_dir(dataset), split, split_file)
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    fewshot = config["fewshot"]
    holdout_cells = []
    test_treatments = {}
    for key, v in fewshot.items():
        cell = key.split(".")[-1]
        holdout_cells.append(cell)
        test_treatments[cell] = set(v["test"])
    return {"holdout_cells": holdout_cells, "test_treatments": test_treatments}


def make_splits(df, split_info):
    holdout = set(split_info["holdout_cells"])
    test_tr = split_info["test_treatments"]
    cell_arr = df["cell_line"].to_numpy()
    cond_arr = df["condition"].to_numpy()
    test_mask = np.array([
        (c in holdout) and (t in test_tr.get(c, ()))
        for c, t in zip(cell_arr, cond_arr)
    ])
    return (
        df.loc[~test_mask].reset_index(drop=True),
        df.loc[test_mask].reset_index(drop=True),
    )


def to_arrays(df):
    return (
        df["cell_line"].to_numpy(),
        df["condition"].to_numpy(),
        df["growth_rate"].to_numpy(dtype=np.float64),
    )


def compute_metrics(y_pred, y_true):
    """Compute MSE, MAE, R^2, and Pearson r between two 1-D arrays.

    R^2 = 1 - MSE / Var(y_true), with Var as the population variance (ddof=0).
    Equals 1 at perfect fit, 0 when y_pred = mean(y_true), negative when worse.
    Pearson r is undefined if either array has zero variance — return NaN then.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    diff = y_pred - y_true
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    var_true = float(np.var(y_true))
    r2 = float(1.0 - mse / var_true) if var_true > 0 else float("nan")
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        pearson = float(np.corrcoef(y_pred, y_true)[0, 1])
    else:
        pearson = float("nan")
    return {
        "sensitivity/mse": mse,
        "sensitivity/mae": mae,
        "sensitivity/r2": r2,
        "sensitivity/pearson": pearson,
    }


def make_evaluator(y_test):
    y_test = np.asarray(y_test, dtype=np.float64)
    called = [False]

    def evaluate_test(y_pred):
        if called[0]:
            raise RuntimeError(
                "evaluate_test() was already called. Test data can only be evaluated once."
            )
        called[0] = True
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if y_pred.shape != y_test.shape:
            raise ValueError(
                f"y_pred shape {y_pred.shape} != y_test shape {y_test.shape}"
            )
        out = compute_metrics(y_pred, y_test)
        out["n_test"] = int(y_test.size)
        return out

    return evaluate_test


def _cache_path(split_name):
    return os.path.join(CACHE_ROOT, split_name.replace("/", "_"))


def _cache_is_valid(split_name):
    cp = _cache_path(split_name)
    meta_path = os.path.join(cp, "meta.json")
    if not os.path.exists(meta_path):
        return False
    dataset, _ = parse_split_name(split_name)
    data_root = _resolve_data_root(dataset)
    parquet = _resolve_path(_load_dataset_config(dataset)["data"]["parquet_path"], data_root)
    return os.path.getmtime(meta_path) > os.path.getmtime(parquet)


def _save_cache(split_name, data):
    cp = _cache_path(split_name)
    os.makedirs(cp, exist_ok=True)
    np.save(os.path.join(cp, "y_train.npy"), data["y_train"])
    np.save(os.path.join(cp, "y_test.npy"), data["y_test"])
    meta = {
        "train_cells": data["train_cells"].tolist(),
        "train_treatments": data["train_treatments"].tolist(),
        "test_cells": data["test_cells"].tolist(),
        "test_treatments": data["test_treatments"].tolist(),
        "cell_to_idx": data["cell_to_idx"],
        "treat_to_idx": data["treat_to_idx"],
        "n_cells": data["n_cells"],
        "n_treatments": data["n_treatments"],
    }
    with open(os.path.join(cp, "meta.json"), "w") as f:
        json.dump(meta, f)


def _load_cache(split_name):
    cp = _cache_path(split_name)
    y_train = np.load(os.path.join(cp, "y_train.npy"))
    y_test = np.load(os.path.join(cp, "y_test.npy"))
    with open(os.path.join(cp, "meta.json")) as f:
        meta = json.load(f)
    return {
        "train_cells": np.array(meta["train_cells"]),
        "train_treatments": np.array(meta["train_treatments"]),
        "y_train": y_train,
        "test_cells": np.array(meta["test_cells"]),
        "test_treatments": np.array(meta["test_treatments"]),
        "y_test": y_test,
        "evaluate_test": make_evaluator(y_test),
        "cell_to_idx": meta["cell_to_idx"],
        "treat_to_idx": meta["treat_to_idx"],
        "n_cells": meta["n_cells"],
        "n_treatments": meta["n_treatments"],
    }


def prepare_all(split_name, with_features=None):
    """Load, split, cache, and return arrays + a one-shot evaluator.

    `with_features` (None | "cell_eval") attaches X_train_features /
    X_test_features (per-gene z-scored using train stats), aligned with the
    train/test rows. The disk cache stores only the no-features payload;
    features are loaded fresh each call when requested.
    """
    if _cache_is_valid(split_name):
        print(f"Loading from cache ({_cache_path(split_name)})...")
        data = _load_cache(split_name)
        if with_features:
            dataset, _ = parse_split_name(split_name)
            data = attach_features(data, dataset, source=with_features)
        return data

    dataset, _ = parse_split_name(split_name)
    print(f"Loading {dataset} parquet (filters from dataset.toml)...")
    df = load_data(dataset)
    print(f"  {len(df)} (cell_line, condition) rows after filter+aggregate")
    print(f"  {df['cell_line'].nunique()} cell lines, {df['condition'].nunique()} conditions")

    print("Parsing split...")
    split_info = load_split(split_name)
    train_df, test_df = make_splits(df, split_info)
    print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")

    train_cells, train_treatments, y_train = to_arrays(train_df)
    test_cells, test_treatments, y_test = to_arrays(test_df)

    all_cells = sorted(df["cell_line"].unique())
    all_treatments = sorted(df["condition"].unique())
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    treat_to_idx = {t: i for i, t in enumerate(all_treatments)}

    data = {
        "train_cells": train_cells,
        "train_treatments": train_treatments,
        "y_train": y_train,
        "test_cells": test_cells,
        "test_treatments": test_treatments,
        "y_test": y_test,
        "cell_to_idx": cell_to_idx,
        "treat_to_idx": treat_to_idx,
        "n_cells": len(all_cells),
        "n_treatments": len(all_treatments),
    }
    print(f"Saving cache to {_cache_path(split_name)}...")
    _save_cache(split_name, data)
    data = _load_cache(split_name)
    if with_features:
        data = attach_features(data, dataset, source=with_features)
    return data


def _load_cell_eval_features(dataset):
    """Load cell_eval as (cell, treatment) -> (gene_cols, feature_vector).

    Returns (feature_map, gene_cols). Uses prepare_combined.load_cell_eval_data,
    which already honors [data.filters] (drop_multi_drug, drop_drugs) and the
    [data.columns] cell_eval_treatment override.
    """
    import prepare_combined
    cfg = prepare_combined.load_dataset_config(dataset)
    df = prepare_combined.load_cell_eval_data(cfg)
    gene_cols = [c for c in df.columns if c not in ("cell_line", "treatment")]
    feats = df[gene_cols].to_numpy(dtype=np.float32)
    feature_map = {
        (c, t): feats[i]
        for i, (c, t) in enumerate(zip(df["cell_line"], df["treatment"]))
    }
    return feature_map, gene_cols


def _load_pdex_features(dataset, column="fold_change", transform=None):
    """Load one pdex statistic per (cell, treatment) as a 2K-gene vector.

    rifivdu's pdex parquet stores the gene identity as an integer index in the
    `feature` column (values 0..58394); those indices line up with the
    cell_eval parquet's gene-column order. We build that map, translate the
    static gene list (symbols) to integer IDs, read pdex with a pyarrow filter
    on the translated IDs, then pivot to (cell_line, target, gene_symbol).
    For rifivdu the fold_change column already holds log2 values.

    `column` selects which pdex statistic to use (fold_change, p_value, fdr).
    `transform="neg_log10"` applies -log10 to the values (used for p_value /
    fdr to make their tails approximately linear).
    """
    import prepare_combined
    import pyarrow.parquet as pq
    cfg = prepare_combined.load_dataset_config(dataset)

    # Build integer-ID -> gene-symbol map from cell_eval parquet column order.
    ce_path = cfg["cell_eval_path"] or (
        os.path.join(cfg["cell_eval_dir"], cfg["cell_eval_glob"])
        if cfg.get("cell_eval_dir") else None
    )
    if ce_path is None:
        raise RuntimeError(f"cannot resolve cell_eval path for {dataset!r} pdex translation")
    ce_cols = pq.ParquetFile(ce_path).schema_arrow.names
    meta = {"cell_line", "treatment", cfg["cell_eval_treatment_column"]}
    gene_symbols_ordered = [c for c in ce_cols if c not in meta]
    symbol_to_id = {g: str(i) for i, g in enumerate(gene_symbols_ordered)}

    with open(cfg["gene_list"]) as f:
        static_genes = json.load(f)
    static_ids = [symbol_to_id[g] for g in static_genes if g in symbol_to_id]
    id_to_symbol = {symbol_to_id[g]: g for g in static_genes if g in symbol_to_id}
    if len(static_ids) < len(static_genes):
        print(f"  pdex: {len(static_genes) - len(static_ids)} of {len(static_genes)} static genes "
              f"absent from cell_eval columns, dropping")

    print(f"  Reading rifivdu pdex parquet ({column}, transform={transform})...")
    table = pq.read_table(
        cfg["pdex_path"],
        columns=["cell_line", "target", "feature", column],
        filters=[("feature", "in", static_ids)],
    )
    long_df = table.to_pandas()
    del table

    # Translate integer IDs back to gene symbols for downstream column names.
    long_df["feature"] = long_df["feature"].map(id_to_symbol)
    long_df[column] = long_df[column].replace([np.inf, -np.inf], np.nan)
    if transform == "neg_log10":
        # Floor at 1e-300 to avoid -inf from log10(0); p_values and fdrs lower
        # than that are effectively zero anyway.
        long_df[column] = -np.log10(long_df[column].clip(lower=1e-300))

    # Reuse the same treatment filters as the rest of the pipeline (drop
    # multi-drug, drop_drugs). The shared helper lives in prepare_combined.
    long_df = prepare_combined._apply_treatment_filters(
        long_df, "target", cfg.get("filters", {}), source="pdex",
    )
    print(f"  {len(long_df)} rows after filter")

    wide = long_df.pivot_table(
        index=["cell_line", "target"],
        columns="feature",
        values=column,
        aggfunc="mean",
    )
    wide.reset_index(inplace=True)
    wide.rename(columns={"target": "treatment"}, inplace=True)
    wide.columns.name = None
    gene_cols = [c for c in wide.columns if c not in ("cell_line", "treatment")]
    wide[gene_cols] = wide[gene_cols].fillna(0.0)

    feats = wide[gene_cols].to_numpy(dtype=np.float32)
    feature_map = {
        (c, t): feats[i]
        for i, (c, t) in enumerate(zip(wide["cell_line"], wide["treatment"]))
    }
    return feature_map, gene_cols


def _align_features(cells, treatments, feature_map, n_features):
    """Lookup feature vectors per (cell, treatment); zero-fill missing rows.

    Returns (X, missing_count)."""
    X = np.zeros((len(cells), n_features), dtype=np.float32)
    missing = 0
    for i, (c, t) in enumerate(zip(cells, treatments)):
        vec = feature_map.get((c, t))
        if vec is None:
            missing += 1
        else:
            X[i] = vec
    return X, missing


# source-name → (loader_kwargs) dispatch. cell_eval is special-cased; pdex
# accepts column + transform.
def _resolve_source(name, dataset):
    if name == "cell_eval":
        return _load_cell_eval_features(dataset)
    if name == "pdex":
        return _load_pdex_features(dataset, column="fold_change")
    if name == "pdex_pv":
        return _load_pdex_features(dataset, column="p_value", transform="neg_log10")
    if name == "pdex_fdr":
        return _load_pdex_features(dataset, column="fdr", transform="neg_log10")
    raise ValueError(f"unknown feature source: {name!r}")


def _build_block(data, feature_map, gene_cols, block_name):
    """Build one feature block: row-aligned train/test matrices, z-scored on
    observed train rows, plus a dense (n_cells × n_treatments × n_features)
    tensor. Returns a dict carrying all three plus per-block metadata."""
    n_features = len(gene_cols)
    X_train, n_miss_tr = _align_features(
        data["train_cells"], data["train_treatments"], feature_map, n_features,
    )
    X_test, n_miss_te = _align_features(
        data["test_cells"], data["test_treatments"], feature_map, n_features,
    )
    train_mask = np.array([
        (c, t) in feature_map
        for c, t in zip(data["train_cells"], data["train_treatments"])
    ])
    if int(train_mask.sum()) == 0:
        raise RuntimeError(f"No training rows have {block_name!r} features; cannot z-score")
    mu = X_train[train_mask].mean(axis=0)
    sigma = X_train[train_mask].std(axis=0)
    sigma = np.where(sigma > 1e-6, sigma, 1.0)
    X_train = ((X_train - mu) / sigma).astype(np.float32)
    X_test = ((X_test - mu) / sigma).astype(np.float32)

    cell_to_idx = data["cell_to_idx"]
    treat_to_idx = data["treat_to_idx"]
    feat_mat = np.zeros(
        (data["n_cells"], data["n_treatments"], n_features), dtype=np.float32,
    )
    n_filled = 0
    for (c, t), vec in feature_map.items():
        ci = cell_to_idx.get(c)
        ti = treat_to_idx.get(t)
        if ci is None or ti is None:
            continue
        feat_mat[ci, ti] = (vec - mu) / sigma
        n_filled += 1
    print(f"  block {block_name!r}: dim={n_features}, missing "
          f"{n_miss_tr}/{len(X_train)} train, {n_miss_te}/{len(X_test)} test rows; "
          f"feat_mat populated {n_filled}/{data['n_cells'] * data['n_treatments']}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "feat_mat": feat_mat,
        "gene_cols": [f"{block_name}__{g}" for g in gene_cols],
        "n_features": n_features,
    }


def attach_features(data, dataset, source="cell_eval"):
    """Attach gene-vector features to a data dict.

    `source` is a comma-separated list of source names. Supported sources:
      - "cell_eval"  : cell_eval delta vector (2K HVG)
      - "pdex"       : pdex log2 fold-change (2K HVG)
      - "pdex_pv"    : pdex p_value, transformed with -log10
      - "pdex_fdr"   : pdex fdr, transformed with -log10

    Each block is z-scored per gene using observed-training-row statistics,
    then concatenated along the feature axis. The dispatch on a single name
    behaves exactly like the previous single-source code path.

    Outputs on `data`:
      - X_train_features / X_test_features : row-aligned (n_rows × Σ n_genes_i)
      - feat_mat                            : dense (n_cells × n_treats × Σ n_genes_i)
      - feature_dim, feature_source, feature_gene_cols metadata"""
    sources = [s.strip() for s in source.split(",") if s.strip()]
    if not sources:
        raise ValueError(f"empty feature source: {source!r}")
    blocks = []
    for src in sources:
        feature_map, gene_cols = _resolve_source(src, dataset)
        blocks.append(_build_block(data, feature_map, gene_cols, src))

    X_train = np.concatenate([b["X_train"] for b in blocks], axis=1)
    X_test = np.concatenate([b["X_test"] for b in blocks], axis=1)
    feat_mat = np.concatenate([b["feat_mat"] for b in blocks], axis=-1)
    gene_cols = [g for b in blocks for g in b["gene_cols"]]

    data = dict(data)
    data["X_train_features"] = X_train
    data["X_test_features"] = X_test
    data["feat_mat"] = feat_mat
    data["feature_dim"] = X_train.shape[1]
    data["feature_source"] = source
    data["feature_gene_cols"] = gene_cols
    return data


def log_result(record):
    record = {"timestamp": datetime.datetime.now().isoformat(), **record}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
