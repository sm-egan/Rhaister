"""
prepare_combined.py — Data loading, split parsing, evaluation, caching.
Loads both pdex (fold_change, p_value, fdr) and cell_eval (expression deltas)
for the static 2K gene set, evaluates all six State paper metrics.
"""

import glob
import json
import os
import datetime
import re
import tomllib

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from rhaister.state_metrics import (
    pearson_delta,
    discrimination_score,
    de_spearman_lfc_sig,
    pr_auc,
    de_overlap,
    de_spearman_sig,
)

# === CONSTANTS ===
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
SPLITS_DIR = os.path.join(_REPO_ROOT, "splits")
CACHE_DIR = "/tmp/tahoe_cache_combined"
RESULTS_JSONL = os.path.join(_REPO_ROOT, "autoresearch", "results.jsonl")

# Default dataset — used when a caller passes an unqualified legacy split name
# (e.g., "tahoe_5_holdout") without a "<dataset>/" prefix.
DEFAULT_DATASET = "tahoe"


# Stringified Python list-of-tuples treatment encoding, shared with
# prepare_sensitivity. Single-drug: "[('Drug', 0.5, 'uM')]"; multi-drug rows
# contain "), (" between tuples.
_TREATMENT_SINGLE_RE = re.compile(r"^\[\('(.+)',\s*([\-\d\.eE\+]+),\s*'([^']+)'\)\]$")
_TREATMENT_MULTI_RE = re.compile(r"\),\s*\(")


def _is_multi_drug(s):
    return bool(_TREATMENT_MULTI_RE.search(s))


def _parse_drug(s):
    m = _TREATMENT_SINGLE_RE.match(s)
    return m.group(1) if m else None


def _apply_treatment_filters(df, treatment_col, filters, source):
    """Drop multi-drug rows and rows in drop_drugs, on `treatment_col`.

    `filters` is the dict surfaced by load_dataset_config — empty / all-false
    means no-op (tahoe today). `source` is a short label used in the log line."""
    if not filters:
        return df
    n_in = len(df)
    if filters.get("drop_multi_drug"):
        df = df.loc[~df[treatment_col].map(_is_multi_drug)]
    drop_drugs = filters.get("drop_drugs") or ()
    if drop_drugs:
        drugs_to_drop = set(drop_drugs)
        df = df.loc[~df[treatment_col].map(_parse_drug).isin(drugs_to_drop)]
    dropped = n_in - len(df)
    if dropped:
        print(f"  filters ({source}): dropped {dropped}/{n_in} rows on '{treatment_col}'")
    return df


def parse_split_name(split_name):
    """Parse a split name into (dataset, split).

    Accepts two forms:
      - Qualified:   "tahoe/5_holdout"        -> ("tahoe", "5_holdout")
      - Legacy flat: "tahoe_5_holdout"        -> ("tahoe", "5_holdout")
      - Bare:        "5_holdout"              -> (DEFAULT_DATASET, "5_holdout")
    """
    if "/" in split_name:
        dataset, split = split_name.split("/", 1)
        return dataset, split

    # Legacy: try "<dataset>_<split>" if <dataset> is a known dataset dir.
    for candidate in os.listdir(SPLITS_DIR):
        if not os.path.isdir(os.path.join(SPLITS_DIR, candidate)):
            continue
        prefix = candidate + "_"
        if split_name.startswith(prefix):
            return candidate, split_name[len(prefix):]

    # Fallback: treat as a split under the default dataset.
    return DEFAULT_DATASET, split_name


# Map dataset names to HuggingFace repos for automatic download.
HF_DATASET_REPOS = {
    "tahoe": "tahoebio/tahoe-de-rhaister",
    "parse": "tahoebio/parse-de-rhaister",
    "replogle_nadig": "tahoebio/replogle-nadig-de-rhaister",
}


def _resolve_data_root(dataset):
    """Resolve the data root directory for a dataset.

    Priority:
    1. RHAISTER_DATA_ROOT env var — if $RHAISTER_DATA_ROOT/<dataset>/ exists,
       use that (per-dataset subdirectories); otherwise use RHAISTER_DATA_ROOT
       directly (single-dataset root).
    2. HuggingFace snapshot_download cache — downloads if not cached
    3. None — paths in dataset.toml must be absolute
    """
    env_root = os.environ.get("RHAISTER_DATA_ROOT")
    if env_root:
        per_dataset = os.path.join(env_root, dataset)
        if os.path.isdir(per_dataset):
            return per_dataset
        return env_root

    hf_repo = HF_DATASET_REPOS.get(dataset)
    if hf_repo:
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(hf_repo, repo_type="dataset")
        except Exception:
            pass

    return None


def _resolve_path(path, data_root, dataset_dir):
    """Resolve a data path: absolute paths pass through, relative paths are
    resolved against data_root (if set). Returns None for None inputs."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if data_root is not None:
        resolved = os.path.join(data_root, path)
        if os.path.exists(resolved):
            return resolved
    # Fallback: resolve relative to the dataset_dir (splits/<dataset>/)
    return os.path.join(dataset_dir, path)


def load_dataset_config(dataset):
    """Load splits/<dataset>/dataset.toml.

    Data paths in [data] can be absolute (cluster use) or relative (portable /
    HuggingFace use). Relative paths are resolved against RHAISTER_DATA_ROOT
    or a HuggingFace snapshot cache. Definition files (gene list, split TOMLs,
    feature_names) are always resolved relative to splits/<dataset>/.

    Only [splits] is required. [data] and [genes] may be absent; unresolved
    fields are None.
    """
    cfg_path = os.path.join(SPLITS_DIR, dataset, "dataset.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    dataset_dir = os.path.join(SPLITS_DIR, dataset)

    data = cfg.get("data", {})
    genes = cfg.get("genes", {})
    columns = data.get("columns", {})
    raw_filters = data.get("filters", {})
    gene_list_name = genes.get("static_list")
    gene_list = os.path.join(dataset_dir, gene_list_name) if gene_list_name else None

    # Treatment filters applied during pdex/cell_eval loading. Defaults are
    # off / empty so datasets without a [data.filters] block (e.g. tahoe today)
    # see no behavior change. `drop_null_growth` is target-side only and is
    # consumed by prepare_sensitivity, not here.
    filters = {
        "drop_multi_drug": bool(raw_filters.get("drop_multi_drug", False)),
        "drop_drugs": list(raw_filters.get("drop_drugs", []) or []),
    }

    # Column mapping: lets a dataset whose source parquet uses different column
    # names plug into the loader. The loader always produces `cell_line` and
    # `treatment` after parsing. `cell_line` may be a scalar (rename) or a list
    # (compose as "col1__col2__..."). Defaults match tahoe.
    cell_line_cols = columns.get("cell_line", "cell_line")
    if isinstance(cell_line_cols, str):
        cell_line_cols = [cell_line_cols]

    # Optional: pdex `feature` column holds integer indices ("0".."N-1") into
    # this gene-name list rather than gene symbols. Used by replogle_nadig.
    feature_names_name = data.get("feature_names_list")
    feature_names_path = (
        os.path.join(dataset_dir, feature_names_name) if feature_names_name else None
    )

    # Resolve data paths: absolute paths pass through (cluster), relative paths
    # resolve against RHAISTER_DATA_ROOT or HuggingFace cache.
    data_root = _resolve_data_root(dataset)
    _rp = lambda p: _resolve_path(p, data_root, dataset_dir)

    # Also resolve definition files from HF data root if present there
    # (HF repos store definitions under definition/ alongside data).
    if data_root and gene_list and not os.path.exists(gene_list):
        hf_gene_list = os.path.join(data_root, "definition", gene_list_name)
        if os.path.exists(hf_gene_list):
            gene_list = hf_gene_list
    if data_root and feature_names_path and not os.path.exists(feature_names_path):
        hf_fn = os.path.join(data_root, "definition", feature_names_name)
        if os.path.exists(hf_fn):
            feature_names_path = hf_fn

    return {
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "pdex_path": _rp(data.get("pdex_path")),
        "cell_eval_path": _rp(data.get("cell_eval_path")),
        "cell_eval_dir": _rp(data.get("cell_eval_dir")),
        "cell_eval_glob": data.get("cell_eval_glob", "plate_plate*.parquet"),
        "control_expression_path": _rp(data.get("control_expression_path")),
        "cell_centroid_path": _rp(data.get("cell_centroid_path")),
        "cell_centroid_hvg_path": _rp(data.get("cell_centroid_hvg_path")),
        "cell_line_columns": list(cell_line_cols),
        "cell_eval_treatment_column": columns.get("cell_eval_treatment", "treatment"),
        "gene_list": gene_list,
        "feature_names_list": feature_names_path,
        "filters": filters,
        "default_split_config": cfg["splits"]["default_config"],
    }


def load_variant_config(dataset, variant):
    """Return a dataset_cfg with [data.variants.<variant>] paths overlaid.

    Variants are independent sub-samples (groups A/B) of the same experiments,
    used by scripts/eval_a_vs_b.py for noise-ceiling calibration. The cell_line,
    treatment, and gene_list metadata are inherited from the base config; only
    data paths and (optionally) `cell_eval_format` are overridden.
    """
    cfg_path = os.path.join(SPLITS_DIR, dataset, "dataset.toml")
    with open(cfg_path, "rb") as f:
        raw = tomllib.load(f)
    variants = raw.get("data", {}).get("variants", {})
    if variant not in variants:
        raise KeyError(f"dataset '{dataset}' has no [data.variants.{variant}] in dataset.toml")
    v = variants[variant]

    cfg = load_dataset_config(dataset)
    cfg["pdex_path"] = v["pdex_path"]
    # cell_eval source is either a single file or a directory+glob — not both;
    # the variant fully re-specifies it rather than partially overriding the base.
    cfg["cell_eval_path"] = v.get("cell_eval_path")
    cfg["cell_eval_dir"] = v.get("cell_eval_dir")
    if "cell_eval_glob" in v:
        cfg["cell_eval_glob"] = v["cell_eval_glob"]
    cfg["cell_eval_format"] = v.get("cell_eval_format", "wide")
    cfg["variant"] = variant
    return cfg


def _compose_cell_line(df, cell_line_cols):
    """Mutate df in-place: ensure a `cell_line` column exists by composing
    `{col1}__{col2}__...` from cell_line_cols, then dropping the sources if
    they aren't literally 'cell_line'. No-op when df already has the final
    column and cell_line_cols == ['cell_line']."""
    if cell_line_cols == ["cell_line"]:
        return
    if len(cell_line_cols) == 1:
        df.rename(columns={cell_line_cols[0]: "cell_line"}, inplace=True)
        return
    df["cell_line"] = df[cell_line_cols[0]].astype(str)
    for c in cell_line_cols[1:]:
        df["cell_line"] = df["cell_line"] + "__" + df[c].astype(str)
    drop = [c for c in cell_line_cols if c != "cell_line"]
    df.drop(columns=drop, inplace=True)


def load_pdex_data(dataset_cfg):
    """Load pdex parquet (long format), filter to static gene set, pivot to wide.
    Returns (fc_wide, pv_wide, fdr_wide, ref_wide) DataFrames."""
    with open(dataset_cfg["gene_list"]) as f:
        static_genes = json.load(f)

    # Some datasets (replogle_nadig) store `feature` as an integer index ("0"..
    # "N-1") into a fixed gene-name list rather than as a gene symbol. Translate
    # to int-string indices for the parquet filter, then map back after read.
    feature_names_path = dataset_cfg.get("feature_names_list")
    feature_idx_to_name = None
    if feature_names_path:
        with open(feature_names_path) as f:
            feature_names = json.load(f)
        name_to_idx = {g: str(i) for i, g in enumerate(feature_names)}
        idx_filter = [name_to_idx[g] for g in static_genes if g in name_to_idx]
        feature_idx_to_name = {str(i): g for i, g in enumerate(feature_names)}
        feature_filter_values = idx_filter
    else:
        feature_filter_values = static_genes

    cell_line_cols = dataset_cfg["cell_line_columns"]
    read_cols = list(cell_line_cols) + [
        "target", "feature", "fold_change", "p_value", "fdr", "ref_mean",
    ]

    print("  Reading and filtering pdex parquet...")
    table = pq.read_table(
        dataset_cfg["pdex_path"],
        columns=read_cols,
        filters=[("feature", "in", feature_filter_values)],
    )
    long_df = table.to_pandas()
    del table
    _compose_cell_line(long_df, cell_line_cols)
    if feature_idx_to_name is not None:
        long_df["feature"] = long_df["feature"].map(feature_idx_to_name)
    long_df = _apply_treatment_filters(
        long_df, "target", dataset_cfg.get("filters", {}), source="pdex"
    )
    print(f"  Filtered to {len(long_df)} rows ({long_df['feature'].nunique()} genes)")

    return _pivot_pdex_long(long_df)


def _pivot_pdex_long(long_df, verbose=True):
    """Pivot long pdex rows (cell_line, target, feature, ...) into the four wide
    (observations x genes) frames Rhaister consumes: fold change, p-value, FDR,
    reference mean. Shared by load_pdex_data and tutorial_data.fetch_de_subset."""
    # fold_change is log2(target_mean / ref_mean); handle inf from log2(0)
    long_df["fold_change"] = long_df["fold_change"].replace([np.inf, -np.inf], np.nan)

    # Pivot to wide format: mean fold_change, min p_value, min fdr, mean ref_mean across plates
    if verbose:
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
    fdr_wide = long_df.pivot_table(
        index=["cell_line", "target"],
        columns="feature",
        values="fdr",
        aggfunc="min",
    )
    ref_wide = long_df.pivot_table(
        index=["cell_line", "target"],
        columns="feature",
        values="ref_mean",
        aggfunc="mean",
    )
    del long_df

    for df in (fc_wide, pv_wide, fdr_wide, ref_wide):
        df.reset_index(inplace=True)
        df.rename(columns={"target": "treatment"}, inplace=True)
        df.columns.name = None

    gene_cols = [c for c in fc_wide.columns if c not in ("cell_line", "treatment")]
    fc_wide[gene_cols] = fc_wide[gene_cols].fillna(0.0)
    pv_wide[gene_cols] = pv_wide[gene_cols].fillna(1.0)
    fdr_wide[gene_cols] = fdr_wide[gene_cols].fillna(1.0)
    ref_wide[gene_cols] = ref_wide[gene_cols].fillna(0.0)

    return fc_wide, pv_wide, fdr_wide, ref_wide


def load_cell_eval_data(dataset_cfg):
    """Load cell_eval parquet(s), filter to static gene set, return a DataFrame
    with columns [cell_line, treatment, <gene_cols>]. Supports either a single
    file (cell_eval_path) or a directory + glob (cell_eval_dir + cell_eval_glob).
    """
    if dataset_cfg.get("cell_eval_path"):
        files = [dataset_cfg["cell_eval_path"]]
    else:
        pattern = os.path.join(dataset_cfg["cell_eval_dir"], dataset_cfg["cell_eval_glob"])
        files = sorted(glob.glob(pattern))
        assert len(files) > 0, f"No parquet files found matching {pattern}"

    cell_line_cols = dataset_cfg["cell_line_columns"]
    treatment_col = dataset_cfg["cell_eval_treatment_column"]
    meta_cols = list(cell_line_cols) + [treatment_col]

    # Subset to metadata + HVG gene columns present in the file, to avoid
    # loading huge gene matrices when only a static subset is needed.
    with open(dataset_cfg["gene_list"]) as f:
        static_genes = json.load(f)
    file_cols = pq.ParquetFile(files[0]).schema_arrow.names
    static_set = set(static_genes)
    gene_cols = [c for c in file_cols if c in static_set]
    read_cols = meta_cols + gene_cols

    dfs = [pd.read_parquet(f, columns=read_cols) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    _compose_cell_line(df, cell_line_cols)
    if treatment_col != "treatment":
        df.rename(columns={treatment_col: "treatment"}, inplace=True)
    df = _apply_treatment_filters(
        df, "treatment", dataset_cfg.get("filters", {}), source="cell_eval"
    )
    return df


def get_gene_columns(df):
    """Return list of gene column names (everything except cell_line, treatment)."""
    return [c for c in df.columns if c not in ("cell_line", "treatment")]


def aggregate_replicates(df):
    """Mean-aggregate over replicates: group by (cell_line, treatment)."""
    gene_cols = get_gene_columns(df)
    return df.groupby(["cell_line", "treatment"], as_index=False)[gene_cols].mean()


def load_split(split_name="tahoe/5_holdout"):
    """Parse the TOML split file. split_name is "<dataset>/<split>" or legacy "<dataset>_<split>"."""
    dataset, split = parse_split_name(split_name)
    dataset_cfg = load_dataset_config(dataset)
    toml_path = os.path.join(
        SPLITS_DIR, dataset, split, dataset_cfg["default_split_config"]
    )
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)

    fewshot = config["fewshot"]
    holdout_cells = []
    test_treatments = {}
    for key, v in fewshot.items():
        cell = key.split(".")[-1]
        test = set(v["test"])
        # Optional `donors` field expands one (cell_type) entry into many
        # composite "<donor>__<cell_type>" cell_lines, all sharing the same
        # test partition. Used by datasets where the observation unit is
        # (donor, cell_type) but the split file records cell_type only.
        donors = v.get("donors")
        if donors:
            for donor in donors:
                composite = f"{donor}__{cell}"
                holdout_cells.append(composite)
                test_treatments[composite] = test
        else:
            holdout_cells.append(cell)
            test_treatments[cell] = test

    return {
        "holdout_cells": holdout_cells,
        "test_treatments": test_treatments,
    }


def make_splits(df_agg, split_info):
    """Split aggregated DataFrame into train and test."""
    holdout = set(split_info["holdout_cells"])
    test_tr = split_info["test_treatments"]

    test_mask = df_agg.apply(
        lambda r: r["cell_line"] in holdout
        and r["treatment"] in test_tr.get(r["cell_line"], set()),
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


def pvalues_to_fdr_bh(pvals):
    """Per-row Benjamini-Hochberg FDR correction. (n_obs, n_genes) -> (n_obs, n_genes)."""
    n_genes = pvals.shape[1]
    sorted_idx = np.argsort(pvals, axis=1)
    sorted_pvals = np.take_along_axis(pvals, sorted_idx, axis=1)
    ranks = np.arange(1, n_genes + 1).astype(np.float64)
    fdr = sorted_pvals * n_genes / ranks
    # Enforce monotonicity from right
    for j in range(n_genes - 2, -1, -1):
        fdr[:, j] = np.minimum(fdr[:, j], fdr[:, j + 1])
    fdr = np.clip(fdr, 0, 1)
    # Unsort back to original gene order
    result = np.empty_like(fdr)
    np.put_along_axis(result, sorted_idx, fdr, axis=1)
    return result


def _build_de_frame(fc, fdr, cells, treats, gene_cols):
    """Convert wide arrays to long Polars DataFrame for State DE metrics."""
    n_obs, n_genes = fc.shape
    target_ids = [f"{c}::{t}" for c, t in zip(cells, treats)]

    return pl.DataFrame({
        "target": np.repeat(target_ids, n_genes),
        "feature": np.tile(gene_cols, n_obs),
        "fold_change": fc.ravel().astype(np.float64),
        "fdr": fdr.ravel().astype(np.float64),
        "abs_log2_fold_change": np.abs(fc.ravel()).astype(np.float64),
    })


def make_evaluator(Y_test, P_test, D_test, F_test, test_cells, test_treatments, gene_cols):
    """Returns a one-shot evaluation function computing all six State metrics."""
    called = [False]

    def evaluate_test(Y_pred_fc, D_pred=None, F_pred=None, P_pred=None, compute_discrimination=False):
        if called[0]:
            raise RuntimeError(
                "evaluate_test() was already called. "
                "Test data can only be evaluated once — no hyperparameter tuning on test data."
            )
        called[0] = True
        return evaluate(
            Y_test, Y_pred_fc, P_test, P_pred,
            D_test, D_pred, F_test, F_pred,
            test_cells, test_treatments, gene_cols,
            compute_discrimination=compute_discrimination,
        )

    return evaluate_test


def evaluate(
    Y_true, Y_pred, P_true=None, P_pred=None,
    D_true=None, D_pred=None, F_true=None, F_pred=None,
    test_cells=None, test_treatments=None, gene_cols=None,
    compute_discrimination=False,
):
    """Evaluate all six State metrics plus legacy metrics."""
    n_samples = Y_true.shape[0]

    # --- Legacy: Pearson on FC ---
    from rhaister.state_metrics import _rowwise_pearson
    per_obs_fc = _rowwise_pearson(np.asarray(Y_true), np.asarray(Y_pred))
    results = {
        "pdex_static/pearson_delta_mean": float(np.mean(per_obs_fc)),
        "pdex_static/pearson_delta_median": float(np.median(per_obs_fc)),
        "pdex_static/pearson_delta_std": float(np.std(per_obs_fc)),
        "n_samples": n_samples,
    }

    # --- Legacy: AUPRC on p-values ---
    if P_true is not None and P_pred is not None:
        labels = (np.asarray(P_true) < 0.05).astype(np.int32).ravel()
        scores = (1.0 - np.asarray(P_pred)).ravel()
        results["pdex_static/auprc_p05"] = float(average_precision_score(labels, scores))

    # --- State metric 1: Pearson on expression deltas ---
    if D_true is not None and D_pred is not None:
        per_obs_delta = pearson_delta(np.asarray(D_true), np.asarray(D_pred))
        # Replace NaN with 0 for aggregation
        per_obs_delta = np.where(np.isfinite(per_obs_delta), per_obs_delta, 0.0)
        results["state/pearson_delta_mean"] = float(np.nanmean(per_obs_delta))
        results["state/pearson_delta_median"] = float(np.nanmedian(per_obs_delta))

    # --- State metric 2: Discrimination score (optional, slow) ---
    if compute_discrimination and D_true is not None and D_pred is not None:
        disc = discrimination_score(np.asarray(D_true), np.asarray(D_pred))
        results["state/discrimination_mean"] = float(np.mean(disc))
        results["state/discrimination_median"] = float(np.median(disc))

    # --- State metrics 3-6: DE metrics via Polars ---
    if F_true is not None and test_cells is not None and gene_cols is not None:
        # Resolve F_pred: use provided FDR, or derive from P_pred via BH
        f_pred = F_pred
        if f_pred is None and P_pred is not None:
            f_pred = pvalues_to_fdr_bh(np.asarray(P_pred))

        if f_pred is not None:
            de_real = _build_de_frame(
                np.asarray(Y_true), np.asarray(F_true),
                test_cells, test_treatments, gene_cols,
            )
            de_pred = _build_de_frame(
                np.asarray(Y_pred), np.asarray(f_pred),
                test_cells, test_treatments, gene_cols,
            )

            # Metric 3: Spearman LFC (significant genes)
            spearman_lfc = de_spearman_lfc_sig(de_real, de_pred)
            vals = [v for v in spearman_lfc.values() if np.isfinite(v)]
            if vals:
                results["state/spearman_lfc_sig_mean"] = float(np.mean(vals))
                results["state/spearman_lfc_sig_median"] = float(np.median(vals))

            # Metric 4: PR-AUC
            prauc = pr_auc(de_real, de_pred)
            vals = [v for v in prauc.values() if np.isfinite(v)]
            if vals:
                results["state/pr_auc_mean"] = float(np.mean(vals))
                results["state/pr_auc_median"] = float(np.median(vals))

            # Metric 5: DE overlap
            overlap = de_overlap(de_real, de_pred)
            vals = [v for v in overlap.values() if np.isfinite(v)]
            if vals:
                results["state/de_overlap_mean"] = float(np.mean(vals))
                results["state/de_overlap_median"] = float(np.median(vals))

            # Metric 6: Effect size Spearman
            results["state/de_spearman_sig"] = float(de_spearman_sig(de_real, de_pred))

    return results


def log_result(experiment_name, metrics, notes="", runtime_seconds=None):
    """Append result to autoresearch/results.jsonl and log to WandB."""
    ts = datetime.datetime.now().isoformat()

    record = {
        "timestamp": ts,
        "experiment": experiment_name,
        "metrics": dict(metrics),
        "notes": notes,
        "decision": None,
        "runtime_seconds": runtime_seconds,
    }
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")

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


def update_decision(experiment_name, decision):
    """Update the decision field on the most recent JSONL entry for an experiment.
    decision should be 'accepted' or 'rejected'.
    """
    if decision not in ("accepted", "rejected"):
        raise ValueError(f"decision must be 'accepted' or 'rejected', got {decision!r}")

    records = []
    with open(RESULTS_JSONL) as f:
        for line in f:
            records.append(json.loads(line))

    found = False
    for i in range(len(records) - 1, -1, -1):
        if records[i]["experiment"] == experiment_name:
            records[i]["decision"] = decision
            found = True
            break

    if not found:
        raise ValueError(f"No JSONL entry found for experiment '{experiment_name}'")

    with open(RESULTS_JSONL, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _build_zeroshot_R(dataset_cfg, train_cells, train_tr, test_cells, test_tr, gene_cols):
    """Build per-observation reference matrices R_train, R_test from the
    control-only expression parquet, replacing pdex-derived ref_mean.

    For each (cell_line, treatment) train/test observation, looks up the set of
    plates that pair was experimentally observed on (from pdex `plate` column)
    and averages the control_expression `ref_mean` across those (cell_line,
    plate) entries. Plate assignments are a property of the experimental design,
    not the perturbation result — so a held-out drug's plate is known a priori,
    and the resulting R requires no per-perturbation data."""
    ctrl_path = dataset_cfg.get("control_expression_path")
    if not ctrl_path:
        raise ValueError(
            f"Dataset '{dataset_cfg['dataset']}' has no control_expression_path "
            f"in dataset.toml; cannot build zeroshot reference."
        )

    cell_line_cols = dataset_cfg["cell_line_columns"]

    # 1. (cell_line, treatment) -> set of plates.
    # Pdex's parquet row groups are partitioned by (cell_line, plate): each row
    # group's plate and cell_line min == max. So we iterate row groups, read the
    # row group's cell_line/plate from the parquet stats, and pull only the
    # distinct target values from that row group — avoiding the 4B-row
    # materialization a naive `read_table().drop_duplicates()` would do.
    print("  zeroshot: reading plate mapping from pdex (per-row-group scan)...")
    pf = pq.ParquetFile(dataset_cfg["pdex_path"])
    cell_col_idx = pf.metadata.schema.names.index(cell_line_cols[0])
    plate_col_idx = pf.metadata.schema.names.index("plate")
    plate_groups = {}
    n_rg = pf.num_row_groups
    for i in range(n_rg):
        rg = pf.metadata.row_group(i)
        cell_raw = rg.column(cell_col_idx).statistics.min
        cell = cell_raw.decode() if isinstance(cell_raw, bytes) else cell_raw
        plate = rg.column(plate_col_idx).statistics.min
        rg_tbl = pf.read_row_group(i, columns=["target"])
        for t in rg_tbl.column("target").unique().to_pylist():
            plate_groups.setdefault((cell, t), set()).add(plate)
    plate_groups = {k: sorted(v) for k, v in plate_groups.items()}
    if len(cell_line_cols) > 1:
        raise NotImplementedError(
            f"Zeroshot plate scan supports single-column cell_line only (got {cell_line_cols})"
        )
    print(f"  zeroshot: plate mapping has {len(plate_groups)} (cell_line, treatment) pairs "
          f"from {n_rg} row groups")

    # 2. control_expression parquet: (cell_line, plate, feature) -> ref_mean,
    # plus the per-(cell_line, plate) ref_lib_mean (constant across the feature
    # axis — same value broadcast to every gene row in the parquet).
    print(f"  zeroshot: reading control_expression parquet ({ctrl_path})...")
    table = pq.read_table(
        ctrl_path,
        columns=list(cell_line_cols) + ["plate", "feature", "ref_mean", "ref_lib_mean"],
        filters=[("feature", "in", list(gene_cols))],
    )
    ctrl_df = table.to_pandas()
    del table
    _compose_cell_line(ctrl_df, cell_line_cols)
    print(f"  zeroshot: control_expression filtered to {len(ctrl_df)} rows")

    print("  zeroshot: pivoting control_expression to wide...")
    ctrl_wide = ctrl_df.pivot_table(
        index=["cell_line", "plate"],
        columns="feature",
        values="ref_mean",
        aggfunc="mean",
    )
    # ref_lib_mean is constant across genes within a (cell_line, plate) row, so
    # `first` and `mean` are equivalent — first is just cheaper.
    lib_per_cp = ctrl_df.groupby(["cell_line", "plate"])["ref_lib_mean"].first()
    del ctrl_df
    # Align columns to gene_cols order; pad missing genes with 0.0 (same fillna
    # convention as the standard pdex-derived ref).
    missing_genes = [g for g in gene_cols if g not in ctrl_wide.columns]
    for g in missing_genes:
        ctrl_wide[g] = 0.0
    ctrl_wide = ctrl_wide[gene_cols].fillna(0.0)
    if missing_genes:
        print(f"  zeroshot: {len(missing_genes)} static genes had no control_expression rows; zero-filled")
    ctrl_idx = {tuple(idx): i for i, idx in enumerate(ctrl_wide.index)}
    ctrl_arr = ctrl_wide.values.astype(np.float32)
    # Align lib_per_cp with ctrl_wide row order.
    lib_arr = lib_per_cp.reindex(ctrl_wide.index).fillna(0.0).values.astype(np.float32)
    n_genes = len(gene_cols)

    def _build_R(cells, treats, label):
        R = np.zeros((len(cells), n_genes), dtype=np.float32)
        L = np.zeros(len(cells), dtype=np.float32)
        miss_plate, miss_ctrl = 0, 0
        for i, (c, t) in enumerate(zip(cells, treats)):
            plates = plate_groups.get((c, t))
            if not plates:
                miss_plate += 1
                continue
            rows = [ctrl_idx[(c, p)] for p in plates if (c, p) in ctrl_idx]
            if not rows:
                miss_ctrl += 1
                continue
            R[i] = ctrl_arr[rows].mean(axis=0)
            L[i] = lib_arr[rows].mean()
        if miss_plate or miss_ctrl:
            print(f"  zeroshot {label}: {miss_plate}/{len(cells)} missing plate map, "
                  f"{miss_ctrl}/{len(cells)} missing control row; zero-filled")
        return R, L

    R_train, L_train = _build_R(train_cells, train_tr, "train")
    R_test, L_test = _build_R(test_cells, test_tr, "test")

    # 3. Optional per-cell-line centroid embedding (one vector per cell_line,
    # broadcast across all (cell_line, treatment) observations for that cell).
    # Provides a second control-only feature source alongside the per-gene
    # ref_mean and per-plate library size.
    centroid_path = dataset_cfg.get("cell_centroid_path")
    if centroid_path:
        print(f"  zeroshot: reading cell centroid parquet ({centroid_path})...")
        c_df = pd.read_parquet(centroid_path)
        # Expected columns: cell_line_id, centroid (list[float])
        centroid_dim = len(c_df["centroid"].iloc[0])
        centroid_arr = np.stack(c_df["centroid"].to_numpy()).astype(np.float32)
        centroid_by_cell = {
            row["cell_line_id"]: centroid_arr[i] for i, row in c_df.iterrows()
        }
        zero_vec = np.zeros(centroid_dim, dtype=np.float32)
        miss_train = sum(1 for c in train_cells if c not in centroid_by_cell)
        miss_test = sum(1 for c in test_cells if c not in centroid_by_cell)
        if miss_train or miss_test:
            print(f"  zeroshot centroid: {miss_train}/{len(train_cells)} train and "
                  f"{miss_test}/{len(test_cells)} test cells absent in centroid file; zero-filled")
        C_train = np.stack([centroid_by_cell.get(c, zero_vec) for c in train_cells])
        C_test = np.stack([centroid_by_cell.get(c, zero_vec) for c in test_cells])
    else:
        C_train = np.zeros((len(train_cells), 0), dtype=np.float32)
        C_test = np.zeros((len(test_cells), 0), dtype=np.float32)

    # 4. Optional per-(cell_line, plate) HVG centroid: ~2000 highly-variable
    # gene columns of DMSO-condition mean expression. Plate-resolved like the
    # control_expression reference, so we average across the (cell, drug)'s
    # plate set just like R/L.
    hvg_path = dataset_cfg.get("cell_centroid_hvg_path")
    if hvg_path:
        print(f"  zeroshot: reading HVG centroid parquet ({hvg_path})...")
        hvg_df = pd.read_parquet(hvg_path)
        # Plate is stored as "plate_plate{N}" — match to pdex's int plate column.
        hvg_df["plate"] = hvg_df["plate"].str.replace("plate_plate", "", regex=False).astype(int)
        # Align HVG columns to the target gene axis (gene_cols) so H can act as a
        # per-(cell, gene) baseline in the diagonal-model formulation. Missing
        # target genes are zero-filled.
        missing_hvg = [g for g in gene_cols if g not in hvg_df.columns]
        if missing_hvg:
            for g in missing_hvg:
                hvg_df[g] = 0.0
            print(f"  zeroshot HVG: {len(missing_hvg)} target genes absent in HVG file; zero-filled")
        hvg_arr = hvg_df[gene_cols].values.astype(np.float32)
        hvg_idx = {(row["cell_line_id"], row["plate"]): i
                   for i, row in hvg_df.reset_index(drop=True).iterrows()}
        hvg_dim = len(gene_cols)
        print(f"  zeroshot: HVG centroid has {len(hvg_df)} (cell_line, plate) rows, "
              f"aligned to {hvg_dim} target genes")

        def _build_H(cells, treats, label):
            H = np.zeros((len(cells), hvg_dim), dtype=np.float32)
            miss_plate, miss_hvg = 0, 0
            for i, (c, t) in enumerate(zip(cells, treats)):
                plates = plate_groups.get((c, t))
                if not plates:
                    miss_plate += 1
                    continue
                rows = [hvg_idx[(c, p)] for p in plates if (c, p) in hvg_idx]
                if not rows:
                    miss_hvg += 1
                    continue
                H[i] = hvg_arr[rows].mean(axis=0)
            if miss_plate or miss_hvg:
                print(f"  zeroshot HVG {label}: {miss_plate}/{len(cells)} missing plate map, "
                      f"{miss_hvg}/{len(cells)} missing HVG row; zero-filled")
            return H

        H_train = _build_H(train_cells, train_tr, "train")
        H_test = _build_H(test_cells, test_tr, "test")
    else:
        H_train = np.zeros((len(train_cells), 0), dtype=np.float32)
        H_test = np.zeros((len(test_cells), 0), dtype=np.float32)

    return R_train, L_train, C_train, H_train, R_test, L_test, C_test, H_test


def _cache_path(dataset, split):
    return os.path.join(CACHE_DIR, dataset, split)


def _zeroshot_R_paths(dataset, split):
    cp = _cache_path(dataset, split)
    return (
        os.path.join(cp, "R_zeroshot_train.npy"),
        os.path.join(cp, "R_zeroshot_test.npy"),
        os.path.join(cp, "L_zeroshot_train.npy"),
        os.path.join(cp, "L_zeroshot_test.npy"),
        os.path.join(cp, "C_zeroshot_train.npy"),
        os.path.join(cp, "C_zeroshot_test.npy"),
        os.path.join(cp, "H_zeroshot_train.npy"),
        os.path.join(cp, "H_zeroshot_test.npy"),
    )


def _zeroshot_R_cache_valid(dataset, split, dataset_cfg):
    paths = _zeroshot_R_paths(dataset, split)
    if not all(os.path.exists(p) for p in paths):
        return False
    ctrl_path = dataset_cfg.get("control_expression_path")
    if not ctrl_path or not os.path.exists(ctrl_path):
        return False
    cache_mtime = min(os.path.getmtime(p) for p in paths)
    src_mtimes = [os.path.getmtime(ctrl_path), os.path.getmtime(dataset_cfg["pdex_path"])]
    for key in ("cell_centroid_path", "cell_centroid_hvg_path"):
        p = dataset_cfg.get(key)
        if p and os.path.exists(p):
            src_mtimes.append(os.path.getmtime(p))
    return cache_mtime > max(src_mtimes)


def _newest_data_mtime(dataset_cfg):
    pdex_mtime = os.path.getmtime(dataset_cfg["pdex_path"])
    if dataset_cfg.get("cell_eval_path"):
        ce_files = [dataset_cfg["cell_eval_path"]]
    else:
        pattern = os.path.join(dataset_cfg["cell_eval_dir"], dataset_cfg["cell_eval_glob"])
        ce_files = glob.glob(pattern)
    ce_mtime = max(os.path.getmtime(f) for f in ce_files) if ce_files else 0
    return max(pdex_mtime, ce_mtime)


def _cache_is_valid(dataset, split, dataset_cfg):
    cp = _cache_path(dataset, split)
    meta_path = os.path.join(cp, "meta.json")
    if not os.path.exists(meta_path):
        return False
    cache_mtime = os.path.getmtime(meta_path)
    return cache_mtime > _newest_data_mtime(dataset_cfg)


def _save_cache(dataset, split, data):
    cp = _cache_path(dataset, split)
    os.makedirs(cp, exist_ok=True)

    for key in ("Y_train", "Y_test", "P_train", "P_test", "D_train", "D_test",
                 "F_train", "F_test", "R_train", "R_test"):
        np.save(os.path.join(cp, f"{key}.npy"), data[key])

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
    D_train = np.load(os.path.join(cp, "D_train.npy"), mmap_mode="r")
    D_test = np.load(os.path.join(cp, "D_test.npy"), mmap_mode="r")
    F_train = np.load(os.path.join(cp, "F_train.npy"), mmap_mode="r")
    F_test = np.load(os.path.join(cp, "F_test.npy"), mmap_mode="r")
    R_train = np.load(os.path.join(cp, "R_train.npy"), mmap_mode="r")
    R_test = np.load(os.path.join(cp, "R_test.npy"), mmap_mode="r")

    with open(os.path.join(cp, "meta.json")) as f:
        meta = json.load(f)

    test_cells = np.array(meta["test_cells"])
    test_treatments = np.array(meta["test_treatments"])

    return {
        "Y_train": Y_train,
        "P_train": P_train,
        "D_train": D_train,
        "F_train": F_train,
        "F_test": F_test,
        "R_train": R_train,
        "R_test": R_test,
        "evaluate_test": make_evaluator(
            Y_test, P_test, D_test, F_test,
            test_cells, test_treatments, meta["gene_cols"],
        ),
        "n_test": Y_test.shape[0],
        "train_cells": np.array(meta["train_cells"]),
        "train_treatments": np.array(meta["train_treatments"]),
        "test_cells": test_cells,
        "test_treatments": test_treatments,
        "gene_cols": meta["gene_cols"],
        "cell_to_idx": meta["cell_to_idx"],
        "treat_to_idx": meta["treat_to_idx"],
        "n_cells": meta["n_cells"],
        "n_treatments": meta["n_treatments"],
    }


def prepare_all(split_name="tahoe/5_holdout", zeroshot=False):
    """One-call function: load both pdex and cell_eval, aggregate, split, cache, return.
    split_name accepts "<dataset>/<split>" or legacy flat "<dataset>_<split>".

    zeroshot=True converts the data to a pure-zeroshot setup:
      (1) R_train/R_test are rebuilt from dataset_cfg["control_expression_path"]
          (per (cell_line, plate) mean reference counts, averaged across the
          plates each (cell_line, treatment) was observed on);
      (2) all training rows whose cell_line is in the split's holdout set are
          dropped — the held-out cell has no perturbation panel, only control
          expression. Test split (Y/P/F/D targets) is unchanged so metrics are
          directly comparable to the standard run."""
    dataset, split = parse_split_name(split_name)
    dataset_cfg = load_dataset_config(dataset)

    if _cache_is_valid(dataset, split, dataset_cfg):
        print(f"Loading from cache ({_cache_path(dataset, split)})...")
        data = _load_cache(dataset, split)
        if zeroshot:
            _apply_zeroshot_R(data, dataset, split, dataset_cfg)
            _drop_holdout_cells_from_train(data, split_name)
        return data

    # --- Load pdex ---
    print(f"Loading pdex data ({dataset})...")
    fc_df, pv_df, fdr_df, ref_df = load_pdex_data(dataset_cfg)
    gene_cols = get_gene_columns(fc_df)
    print(f"  pdex: {len(fc_df)} rows, {len(gene_cols)} genes")

    # --- Load cell_eval ---
    print("Loading cell_eval data...")
    ce_df = load_cell_eval_data(dataset_cfg)
    ce_gene_cols = get_gene_columns(ce_df)
    print(f"  cell_eval: {len(ce_df)} rows, {len(ce_gene_cols)} genes")

    # --- Aggregate replicates ---
    print("Aggregating replicates...")
    fc_agg = aggregate_replicates(fc_df)
    pv_agg = aggregate_replicates(pv_df)
    fdr_agg = aggregate_replicates(fdr_df)
    ref_agg = aggregate_replicates(ref_df)
    ce_agg = aggregate_replicates(ce_df)
    del fc_df, pv_df, fdr_df, ref_df, ce_df
    print(f"  pdex: {len(fc_agg)} pairs, cell_eval: {len(ce_agg)} pairs")

    # --- Align gene columns (intersection, preserving static gene order) ---
    with open(dataset_cfg["gene_list"]) as f:
        static_genes = json.load(f)
    shared_genes = [g for g in static_genes if g in set(gene_cols) and g in set(ce_gene_cols)]
    if len(shared_genes) < len(static_genes):
        print(f"  Gene alignment: {len(shared_genes)}/{len(static_genes)} shared "
              f"(pdex: {len(gene_cols)}, cell_eval: {len(ce_gene_cols)})")
    gene_cols = shared_genes
    fc_agg = fc_agg[["cell_line", "treatment"] + gene_cols]
    pv_agg = pv_agg[["cell_line", "treatment"] + gene_cols]
    fdr_agg = fdr_agg[["cell_line", "treatment"] + gene_cols]
    ref_agg = ref_agg[["cell_line", "treatment"] + gene_cols]
    ce_agg = ce_agg[["cell_line", "treatment"] + gene_cols]

    # --- Inner join on (cell_line, treatment) ---
    merge_key = fc_agg[["cell_line", "treatment"]].copy()
    merge_key["_in_pdex"] = True
    ce_agg = ce_agg.merge(merge_key, on=["cell_line", "treatment"], how="inner").drop(columns="_in_pdex")

    merge_key_ce = ce_agg[["cell_line", "treatment"]].copy()
    merge_key_ce["_in_ce"] = True
    fc_agg = fc_agg.merge(merge_key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")
    pv_agg = pv_agg.merge(merge_key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")
    fdr_agg = fdr_agg.merge(merge_key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")
    ref_agg = ref_agg.merge(merge_key_ce, on=["cell_line", "treatment"], how="inner").drop(columns="_in_ce")

    # Sort all by the same order
    for df in (fc_agg, pv_agg, fdr_agg, ref_agg, ce_agg):
        df.sort_values(["cell_line", "treatment"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"  Aligned: {len(fc_agg)} shared (cell_line, treatment) pairs")

    # --- Parse split ---
    print("Parsing split...")
    split_info = load_split(split_name)
    train_fc, test_fc = make_splits(fc_agg, split_info)
    train_pv, test_pv = make_splits(pv_agg, split_info)
    train_fdr, test_fdr = make_splits(fdr_agg, split_info)
    train_ref, test_ref = make_splits(ref_agg, split_info)
    train_ce, test_ce = make_splits(ce_agg, split_info)
    print(f"  Train: {len(train_fc)}, Test: {len(test_fc)}")

    # --- Convert to matrices ---
    train_cells, train_tr, Y_train = to_matrices(train_fc, gene_cols)
    test_cells, test_tr, Y_test = to_matrices(test_fc, gene_cols)
    _, _, P_train = to_matrices(train_pv, gene_cols)
    _, _, P_test = to_matrices(test_pv, gene_cols)
    _, _, F_train = to_matrices(train_fdr, gene_cols)
    _, _, F_test = to_matrices(test_fdr, gene_cols)
    _, _, R_train = to_matrices(train_ref, gene_cols)
    _, _, R_test = to_matrices(test_ref, gene_cols)
    _, _, D_train = to_matrices(train_ce, gene_cols)
    _, _, D_test = to_matrices(test_ce, gene_cols)

    # Fill NaN in deltas
    D_train = np.nan_to_num(D_train, nan=0.0)
    D_test = np.nan_to_num(D_test, nan=0.0)

    # --- Build index maps ---
    all_cells = sorted(set(fc_agg["cell_line"].unique()))
    all_treatments = sorted(set(fc_agg["treatment"].unique()))
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}
    treat_to_idx = {t: i for i, t in enumerate(all_treatments)}

    data = {
        "train_cells": train_cells,
        "train_treatments": train_tr,
        "Y_train": Y_train,
        "P_train": P_train,
        "D_train": D_train,
        "F_train": F_train,
        "R_train": R_train.astype(np.float32),
        "test_cells": test_cells,
        "test_treatments": test_tr,
        "Y_test": Y_test,
        "P_test": P_test,
        "D_test": D_test,
        "F_test": F_test,
        "R_test": R_test.astype(np.float32),
        "gene_cols": gene_cols,
        "cell_to_idx": cell_to_idx,
        "treat_to_idx": treat_to_idx,
        "n_cells": len(all_cells),
        "n_treatments": len(all_treatments),
    }

    print("Saving cache...")
    _save_cache(dataset, split, data)

    # Reload with mmap, wrapping test data in one-shot evaluator
    data = _load_cache(dataset, split)
    if zeroshot:
        _apply_zeroshot_R(data, dataset, split, dataset_cfg)
        _drop_holdout_cells_from_train(data, split_name)
    return data


def _drop_holdout_cells_from_train(data, split_name):
    """Mutate data in place: drop all train rows whose cell_line is in the
    split's holdout set. After this the held-out cells appear only in test."""
    split_info = load_split(split_name)
    holdout = set(split_info["holdout_cells"])
    train_cells = np.asarray(data["train_cells"])
    keep = np.array([c not in holdout for c in train_cells], dtype=bool)
    dropped = int((~keep).sum())
    if dropped == 0:
        return
    keys = ["Y_train", "P_train", "D_train", "F_train", "R_train",
            "train_cells", "train_treatments"]
    if "L_train" in data:
        keys.append("L_train")
    if "C_train" in data:
        keys.append("C_train")
    if "H_train" in data:
        keys.append("H_train")
    for key in keys:
        data[key] = np.asarray(data[key])[keep]
    print(f"  zeroshot: dropped {dropped} train rows for holdout cells "
          f"(train now {len(data['train_cells'])} rows)")


def _apply_zeroshot_R(data, dataset, split, dataset_cfg):
    """Replace data['R_train']/data['R_test'] with the control-only reference
    and attach the per-observation arrays L (library size), C (per-cell DMSO
    centroid embedding), and H (per-(cell, plate) HVG centroid). Cached on
    disk so subsequent runs skip the pdex/control/centroid reads."""
    paths = _zeroshot_R_paths(dataset, split)
    (train_p, test_p, ltrain_p, ltest_p,
     ctrain_p, ctest_p, htrain_p, htest_p) = paths
    if _zeroshot_R_cache_valid(dataset, split, dataset_cfg):
        print(f"Loading zeroshot R+L+C+H from cache ({train_p})...")
        R_train = np.load(train_p, mmap_mode="r")
        R_test = np.load(test_p, mmap_mode="r")
        L_train = np.load(ltrain_p, mmap_mode="r")
        L_test = np.load(ltest_p, mmap_mode="r")
        C_train = np.load(ctrain_p, mmap_mode="r")
        C_test = np.load(ctest_p, mmap_mode="r")
        H_train = np.load(htrain_p, mmap_mode="r")
        H_test = np.load(htest_p, mmap_mode="r")
    else:
        print("Building zeroshot R+L+C+H from control_expression / centroid parquets...")
        (R_train, L_train, C_train, H_train,
         R_test, L_test, C_test, H_test) = _build_zeroshot_R(
            dataset_cfg,
            data["train_cells"], data["train_treatments"],
            data["test_cells"], data["test_treatments"],
            data["gene_cols"],
        )
        os.makedirs(os.path.dirname(train_p), exist_ok=True)
        for arr, p in (
            (R_train, train_p), (R_test, test_p),
            (L_train, ltrain_p), (L_test, ltest_p),
            (C_train, ctrain_p), (C_test, ctest_p),
            (H_train, htrain_p), (H_test, htest_p),
        ):
            np.save(p, arr)
        print(f"Saved zeroshot R+L+C+H cache to {train_p}")
    data["R_train"] = R_train
    data["R_test"] = R_test
    data["L_train"] = L_train
    data["L_test"] = L_test
    data["C_train"] = C_train
    data["C_test"] = C_test
    data["H_train"] = H_train
    data["H_test"] = H_test
