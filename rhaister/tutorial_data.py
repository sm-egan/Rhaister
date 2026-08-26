"""
tutorial_data.py — data loading for tutorials/rhaister_walkthrough.ipynb.

The walkthrough needs two things the rest of the package does not provide:

1. The `AnnData -> matrices` bridge (step 3 of the notebook), computed the same
   way the production pipeline computes it. `de_from_anndata` reuses
   `scripts/data_prep/` so the tutorial exercises the real code path rather than
   a parallel reimplementation.

2. A *small* slice of published Tahoe DE data to train on. `prepare_all` resolves
   data via `snapshot_download`, which for tahoe means pulling 41.8 GB — most of
   it a single 40.6 GB pdex parquet. `fetch_de_subset` instead prunes row groups
   from that parquet's footer statistics and reads only the ones covering the
   requested (plate, cell_line, target) cells: a few hundred MB, cached locally
   afterwards as a few-MB parquet.

Both honour RHAISTER_DATA_ROOT first, matching prepare_combined._resolve_data_root.
"""

import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rhaister.prepare_combined import _pivot_pdex_long, make_evaluator

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

# HuggingFace dataset holding the published Tahoe DE summaries.
HF_TAHOE_DE_REPO = "tahoebio/tahoe-de-rhaister"
PDEX_REPO_PATH = "pdex/all_plates_pdex.parquet"
CELL_EVAL_REPO_PATH = "cell_eval/plate_plate{plate}.parquet"

# normalize_total target used for Tahoe throughout the pipeline
# (scripts/data_prep/configs/tahoe.yaml, celleval_full profile).
TAHOE_TARGET_SUM = 1872
TAHOE_CONTROL = "[('DMSO_TF', 0.0, 'uM')]"

# Committed subset that lets the default notebook path run with no network.
BUNDLED_SUBSET = os.path.join(_REPO_ROOT, "tutorials", "data", "tahoe_walkthrough_de_subset.parquet")

# --- The walkthrough's default slice -------------------------------------
# Plate 1 is the 0.05 uM dose (plate encodes dose in Tahoe). These eight cell
# lines have complete coverage of the plate's 92 treatments, and the drugs are a
# contiguous block in target-sort order, which keeps the pruned pdex read to a
# few dozen row groups instead of a few hundred.
WALKTHROUGH_PLATE = 1
WALKTHROUGH_DOSE = 0.05
WALKTHROUGH_CELL_LINES = [
    "CVCL_0218",
    "CVCL_0366",
    "CVCL_0459",
    "CVCL_0504",
    "CVCL_1239",
    "CVCL_1285",
    "CVCL_1517",
    "CVCL_C466",
]
WALKTHROUGH_DRUGS = [
    "Afatinib",
    "Alpelisib",
    "Anastrozole",
    "BAY1125976",
    "BI-3406",
    "BI-78D3",
    "Belumosudil (mesylate)",
    "Belzutifan",
    "Bentamapimod",
    "Bimiralisib",
    "Binimetinib",
    "Bortezomib",
    "Brivudine",
    "CP21R7",
    "Capivasertib",
    "Capmatinib",
    "Celecoxib",
    "DT-061",
    "Dabrafenib",
    "ERK5-IN-2",
    "ETC-206",
    "EX229",
    "Elimusertib hydrochloride",
    "Encorafenib",
]
# Four targeted-inhibitor treatments held out from two cell lines. Both cell
# lines keep their other 20 treatments in train, and all four drugs are still
# observed in the other six cell lines — compositional generalization, not a
# cold start.
WALKTHROUGH_HOLDOUT_DRUGS = ["Bortezomib", "Binimetinib", "Capivasertib", "Dabrafenib"]
WALKTHROUGH_HOLDOUT_CELL_LINES = ["CVCL_0218", "CVCL_1285"]


def walkthrough_treatments(drugs=None, dose=WALKTHROUGH_DOSE):
    """Treatment labels for the walkthrough's default drug panel."""
    return [treatment_label(d, dose) for d in (drugs or WALKTHROUGH_DRUGS)]


def walkthrough_holdout():
    """The default {cell_line: [treatment, ...]} holdout for compositional_split."""
    treats = walkthrough_treatments(WALKTHROUGH_HOLDOUT_DRUGS)
    return {c: list(treats) for c in WALKTHROUGH_HOLDOUT_CELL_LINES}


# ---------------------------------------------------------------------------
# Treatment labels
# ---------------------------------------------------------------------------


def treatment_label(drug, dose, unit="uM"):
    """Build Tahoe's stringified treatment label: "[('Drug', 0.05, 'uM')]".

    This is the exact encoding used in obs["drugname_drugconc"] and in the
    `target` column of the published pdex parquet, so labels built here join
    directly against both.
    """
    return f"[('{drug}', {dose}, '{unit}')]"


# ---------------------------------------------------------------------------
# Step 3: the AnnData -> matrices bridge
# ---------------------------------------------------------------------------


def _load_data_prep_module():
    """Import scripts/data_prep/compute_celleval_deltas.py as a module.

    It lives outside the installed package (it is a script, not a library), so
    it is loaded by path rather than imported. Requires a source checkout.
    """
    path = os.path.join(_REPO_ROOT, "scripts", "data_prep", "compute_celleval_deltas.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. The walkthrough's step 3 reuses the repo's own "
            "data-prep code, so it needs a source checkout of Rhaister "
            "(snapshot_download or git clone), not just an installed wheel."
        )
    spec = importlib.util.spec_from_file_location("_rhaister_compute_celleval_deltas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def de_from_anndata(
    adata,
    gene_panel=None,
    control_label=TAHOE_CONTROL,
    group_col="cell_line",
    treatment_col="treatment",
    target_sum=TAHOE_TARGET_SUM,
    pdex_threads=4,
    verbose=True,
):
    """Summarise single cells into the five matrices Rhaister consumes.

    One observation is a (group, treatment) pair — for Tahoe, (cell_line, drug+dose).
    Returns a dict with:

        gene_cols : list[str]                 the gene axis, in panel order
        Y, P, F, D, R : dict[(group, treatment)] -> np.ndarray(len(gene_cols))

    where Y is log2 fold change, P the Mann-Whitney p-value, F its BH FDR,
    D the expression delta and R the control (DMSO) reference mean.

    Y/P/F/R come from `pdex` exactly as scripts/data_prep/compute_pdex.py calls
    it; D from scripts/data_prep/compute_celleval_deltas.py. Both are the code
    paths that produced the published parquets, so the numbers here are
    comparable to them (up to subsampling).
    """
    import pdex as pdex_mod

    deltas_mod = _load_data_prep_module()

    if gene_panel is None:
        gene_cols = list(adata.var_names)
    else:
        # Preserve panel order; keep only genes actually measured here.
        measured = set(adata.var_names)
        gene_cols = [g for g in gene_panel if g in measured]
    if not gene_cols:
        raise ValueError("No genes in common between `gene_panel` and adata.var_names.")

    adata = adata[:, gene_cols].copy()
    gidx = {g: i for i, g in enumerate(gene_cols)}
    n_genes = len(gene_cols)

    groups = sorted(adata.obs[group_col].astype(str).unique())
    if verbose:
        print(f"  {adata.n_obs} cells x {n_genes} genes, {len(groups)} {group_col} value(s)")

    # --- D: normalize_total -> log1p -> pseudobulk -> treated minus control ---
    X, gene_names = deltas_mod.extract_expression(
        adata, {"gene_mode": "all_genes", "target_sum": target_sum}
    )
    labels = np.array(
        [
            f"{g}||{t}"
            for g, t in zip(
                adata.obs[group_col].astype(str).values,
                adata.obs[treatment_col].astype(str).values,
            )
        ]
    )
    means, group_to_idx = deltas_mod.pseudobulk_means(X, labels)
    delta_df = deltas_mod.compute_deltas(
        means, group_to_idx, gene_names, control_label, group_col, treatment_col
    )

    D = {}
    delta_gene_cols = [c for c in delta_df.columns if c not in (group_col, treatment_col)]
    delta_pos = {g: i for i, g in enumerate(delta_gene_cols)}
    delta_vals = delta_df[delta_gene_cols].to_numpy(dtype=np.float64)
    for row, (g, t) in enumerate(zip(delta_df[group_col], delta_df[treatment_col])):
        vec = np.zeros(n_genes)
        for gene, j in gidx.items():
            k = delta_pos.get(gene)
            if k is not None:
                vec[j] = delta_vals[row, k]
        D[(str(g), str(t))] = np.nan_to_num(vec, nan=0.0)

    # --- Y/P/F/R: pdex, once per group, with that group's control as reference ---
    Y, P, F, R = {}, {}, {}, {}
    for g in groups:
        sub = adata[adata.obs[group_col].astype(str) == g].copy()
        if control_label not in set(sub.obs[treatment_col].astype(str)):
            if verbose:
                print(f"  skip {g}: no control cells ({control_label})")
            continue
        res = pdex_mod.pdex(
            sub,
            groupby=treatment_col,
            mode="ref",
            reference=control_label,
            is_log1p=False,
            threads=pdex_threads,
            as_pandas=True,
        )

        # ref_mean is the control baseline: identical across a group's targets,
        # so it collapses to one vector per group.
        ref_vec = np.zeros(n_genes)
        for gene, v in res.groupby("feature")["ref_mean"].first().items():
            if gene in gidx:
                ref_vec[gidx[gene]] = v

        for lab, rows in res.groupby("target"):
            col = rows["feature"].map(gidx)
            keep = col.notna().values
            j = col[keep].astype(int).values
            y = np.zeros(n_genes)
            p = np.ones(n_genes)
            f = np.ones(n_genes)
            # pdex emits +/-inf log2FC when a gene is all-zero in one group; map
            # those (and NaN) to 0, mirroring prepare_combined.load_pdex_data.
            # p/f are in [0, 1], so only Y is ever non-finite.
            y[j] = np.nan_to_num(
                rows["log2_fold_change"].values[keep], nan=0.0, posinf=0.0, neginf=0.0
            )
            p[j] = np.nan_to_num(rows["p_value"].values[keep], nan=1.0)
            f[j] = np.nan_to_num(rows["fdr"].values[keep], nan=1.0)
            key = (str(g), str(lab))
            Y[key], P[key], F[key], R[key] = y, p, f, ref_vec

    if verbose:
        print(f"  computed {len(Y)} (group, treatment) observations")
    return {"gene_cols": gene_cols, "Y": Y, "P": P, "F": F, "D": D, "R": R}


# ---------------------------------------------------------------------------
# Steps 4-6: a small slice of the published DE data
# ---------------------------------------------------------------------------


def _local_data_root():
    """RHAISTER_DATA_ROOT resolution, without ever calling snapshot_download."""
    env_root = os.environ.get("RHAISTER_DATA_ROOT")
    if not env_root:
        return None
    per_dataset = os.path.join(env_root, "tahoe")
    return per_dataset if os.path.isdir(per_dataset) else env_root


def _open_pdex(verbose=True):
    """Return a ParquetFile for the pdex parquet — local if RHAISTER_DATA_ROOT
    points at it, otherwise streamed from HuggingFace without downloading."""
    root = _local_data_root()
    if root:
        local = os.path.join(root, PDEX_REPO_PATH)
        if os.path.exists(local):
            if verbose:
                print(f"  pdex: local ({local})")
            return pq.ParquetFile(local)

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    remote = f"datasets/{HF_TAHOE_DE_REPO}/{PDEX_REPO_PATH}"
    if verbose:
        print(f"  pdex: streaming row groups from {HF_TAHOE_DE_REPO}")
    return pq.ParquetFile(fs.open(remote, "rb"))


def _select_row_groups(metadata, plate, cell_lines, treatments):
    """Row groups whose statistics can possibly contain the requested rows.

    The parquet is clustered by (plate, cell_line, target), so min/max stats on
    those three columns prune ~13,000 row groups down to a few dozen.
    """
    names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
    pi, ci, ti = names.index("plate"), names.index("cell_line"), names.index("target")
    lo, hi = min(treatments), max(treatments)

    selected = []
    for g in range(metadata.num_row_groups):
        rg = metadata.row_group(g)
        ps = rg.column(pi).statistics
        if ps is None or not (ps.min <= plate <= ps.max):
            continue
        cs = rg.column(ci).statistics
        if cs is not None and not any(cs.min <= c <= cs.max for c in cell_lines):
            continue
        ts = rg.column(ti).statistics
        if ts is not None and (ts.max < lo or ts.min > hi):
            continue
        selected.append(g)
    return selected


def fetch_pdex_subset(plate, cell_lines, treatments, gene_panel, verbose=True):
    """Read just the requested (plate, cell_line, target) rows of the pdex parquet.

    Returns the long-format DataFrame (cell_line, target, feature, fold_change,
    p_value, fdr, ref_mean) restricted to `gene_panel`.
    """
    cell_lines, treatments = set(cell_lines), set(treatments)
    genes = set(gene_panel)

    pf = _open_pdex(verbose=verbose)
    groups = _select_row_groups(pf.metadata, plate, cell_lines, treatments)
    if not groups:
        raise ValueError(
            f"No pdex row groups match plate={plate}, cell_lines={sorted(cell_lines)[:3]}..."
        )
    mb = sum(pf.metadata.row_group(g).total_byte_size for g in groups) / 1e6
    if verbose:
        print(
            f"  {len(groups)}/{pf.metadata.num_row_groups} row groups selected (~{mb:.0f} MB) — reading..."
        )

    cols = ["cell_line", "target", "feature", "fold_change", "p_value", "fdr", "ref_mean"]
    parts = []
    for n, g in enumerate(groups, 1):
        tbl = pf.read_row_group(g, columns=cols)
        df = tbl.to_pandas()
        df = df[
            df["cell_line"].isin(cell_lines)
            & df["target"].isin(treatments)
            & df["feature"].isin(genes)
        ]
        if len(df):
            parts.append(df)
        if verbose:
            print(f"    row group {n}/{len(groups)} — {sum(len(p) for p in parts)} rows kept")
    if not parts:
        raise ValueError("Row groups matched but no rows survived filtering — check labels.")
    return pd.concat(parts, ignore_index=True)


def fetch_cell_eval_subset(plate, cell_lines, treatments, gene_panel, verbose=True):
    """Load the expression deltas D for one plate, restricted to the request.

    The per-plate cell_eval parquet is ~70 MB, small enough to fetch whole.
    """
    rel = CELL_EVAL_REPO_PATH.format(plate=plate)
    root = _local_data_root()
    path = None
    if root:
        candidate = os.path.join(root, rel)
        if os.path.exists(candidate):
            path = candidate
            if verbose:
                print(f"  cell_eval: local ({candidate})")
    if path is None:
        from huggingface_hub import hf_hub_download

        if verbose:
            print(f"  cell_eval: downloading {rel} from {HF_TAHOE_DE_REPO} (~70 MB, cached)")
        path = hf_hub_download(HF_TAHOE_DE_REPO, rel, repo_type="dataset")

    available = pq.ParquetFile(path).schema_arrow.names
    panel = [g for g in gene_panel if g in set(available)]
    df = pd.read_parquet(path, columns=["cell_line", "treatment", *panel])
    return df[df["cell_line"].isin(set(cell_lines)) & df["treatment"].isin(set(treatments))].copy()


def fetch_de_subset(
    plate,
    cell_lines,
    treatments,
    gene_panel,
    cache_path=None,
    use_bundled=True,
    verbose=True,
):
    """The five wide (observations x genes) frames for the requested slice.

    Returns (fc, pv, fdr, ref, delta), each a DataFrame with columns
    [cell_line, treatment, <genes>], sharing one gene axis and row order.

    Resolution order: a bundled subset committed with the tutorial, then
    `cache_path`, then a pruned read of the published parquets (which is then
    written to `cache_path`).
    """
    cell_lines, treatments = sorted(set(cell_lines)), sorted(set(treatments))

    for candidate in ([BUNDLED_SUBSET] if use_bundled else []) + ([cache_path] if cache_path else []):
        if candidate and os.path.exists(candidate):
            if verbose:
                print(f"Loading cached DE subset ({candidate})")
            frames = _unpack_subset(pd.read_parquet(candidate), cell_lines, treatments)
            if frames is not None:
                return frames
            if verbose:
                print("  cached subset does not cover this request — refetching")

    if verbose:
        print(f"Fetching DE subset: plate {plate}, {len(cell_lines)} cell lines, {len(treatments)} treatments")
    long_df = fetch_pdex_subset(plate, cell_lines, treatments, gene_panel, verbose=verbose)
    fc, pv, fdr, ref = _pivot_pdex_long(long_df, verbose=verbose)
    delta = fetch_cell_eval_subset(plate, cell_lines, treatments, gene_panel, verbose=verbose)

    frames = _align_frames(fc, pv, fdr, ref, delta, gene_panel)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        _pack_subset(*frames).to_parquet(cache_path, index=False)
        if verbose:
            print(f"  cached to {cache_path}")
    return frames


def _align_frames(fc, pv, fdr, ref, delta, gene_panel):
    """Put all five frames on one gene axis and one (cell_line, treatment) index."""
    keys = ["cell_line", "treatment"]
    shared_genes = set(fc.columns) & set(delta.columns)
    gene_cols = [g for g in gene_panel if g in shared_genes]
    if not gene_cols:
        raise ValueError("pdex and cell_eval share no genes from the panel.")

    index = fc[keys].merge(delta[keys], on=keys, how="inner").drop_duplicates()
    index = index.sort_values(keys).reset_index(drop=True)
    if index.empty:
        raise ValueError("No (cell_line, treatment) pairs present in both pdex and cell_eval.")

    out = []
    for df, fill in ((fc, 0.0), (pv, 1.0), (fdr, 1.0), (ref, 0.0), (delta, 0.0)):
        merged = index.merge(df[keys + gene_cols], on=keys, how="left")
        merged[gene_cols] = merged[gene_cols].fillna(fill)
        out.append(merged)
    return tuple(out)


_SUBSET_MATRICES = ("Y", "P", "F", "R", "D")


def _pack_subset(fc, pv, fdr, ref, delta):
    """Stack the five frames into one long-ish frame for compact caching.

    Cast to float32: the cache is a convenience copy of published values, and
    halving it keeps the bundled subset in the same size class as the existing
    test fixtures. Downstream stacking widens back to float64.
    """
    parts = []
    for name, df in zip(_SUBSET_MATRICES, (fc, pv, fdr, ref, delta)):
        part = df.copy()
        genes = [c for c in part.columns if c not in ("cell_line", "treatment")]
        part[genes] = part[genes].astype(np.float32)
        part.insert(0, "matrix", name)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _unpack_subset(packed, cell_lines, treatments):
    """Inverse of _pack_subset. Returns None if the cache misses the request."""
    if "matrix" not in packed.columns:
        return None
    have_cells = set(packed["cell_line"])
    have_treats = set(packed["treatment"])
    if not set(cell_lines) <= have_cells or not set(treatments) <= have_treats:
        return None

    keep = packed["cell_line"].isin(set(cell_lines)) & packed["treatment"].isin(set(treatments))
    packed = packed[keep]
    out = []
    for name in _SUBSET_MATRICES:
        df = packed[packed["matrix"] == name].drop(columns="matrix")
        out.append(df.sort_values(["cell_line", "treatment"]).reset_index(drop=True))
    return tuple(out)


# ---------------------------------------------------------------------------
# The data dict train_and_evaluate consumes
# ---------------------------------------------------------------------------


def stack_observations(keys, source, n_genes=None):
    """(observations x genes) matrix for `keys`, in order, from a dict."""
    if not keys:
        if n_genes is None:
            raise ValueError("n_genes is required to stack an empty key list.")
        return np.zeros((0, n_genes), dtype=np.float64)
    return np.array([source[k] for k in keys], dtype=np.float64)


def build_data_dict(Y, P, F, D, R, train_keys, test_keys, gene_cols):
    """Assemble the dict `train_and_evaluate(data=...)` consumes.

    Y/P/F/D/R are dicts keyed by (cell_line, treatment) — the layout
    `de_from_anndata` returns and `frames_to_observations` produces. Keys are
    (cell_line, treatment) tuples; train_keys and test_keys partition them.

    Mirrors the key set prepare_combined._load_cache returns, plus Y_test/D_test
    which the notebook's prediction plot reads. `evaluate_test` is the same
    one-shot evaluator prepare_all seals the test data behind: calling it twice
    raises, so rebuild the dict to score again.
    """
    train_keys, test_keys = list(train_keys), list(test_keys)
    if not test_keys:
        raise ValueError("test_keys is empty — nothing to evaluate.")
    overlap = set(train_keys) & set(test_keys)
    if overlap:
        raise ValueError(f"train_keys and test_keys overlap: {sorted(overlap)[:3]}")

    n_genes = len(gene_cols)
    cells = lambda keys: np.array([k[0] for k in keys])  # noqa: E731
    treats = lambda keys: np.array([k[1] for k in keys])  # noqa: E731

    Y_train = stack_observations(train_keys, Y, n_genes)
    Y_test = stack_observations(test_keys, Y, n_genes)
    P_train = stack_observations(train_keys, P, n_genes)
    P_test = stack_observations(test_keys, P, n_genes)
    F_train = stack_observations(train_keys, F, n_genes)
    F_test = stack_observations(test_keys, F, n_genes)
    D_train = np.nan_to_num(stack_observations(train_keys, D, n_genes), nan=0.0)
    D_test = np.nan_to_num(stack_observations(test_keys, D, n_genes), nan=0.0)
    R_train = stack_observations(train_keys, R, n_genes)
    R_test = stack_observations(test_keys, R, n_genes)

    test_cells, test_treatments = cells(test_keys), treats(test_keys)
    all_cells = sorted({k[0] for k in train_keys + test_keys})
    all_treatments = sorted({k[1] for k in train_keys + test_keys})

    return {
        "train_cells": cells(train_keys),
        "train_treatments": treats(train_keys),
        "Y_train": Y_train,
        "P_train": P_train,
        "D_train": D_train,
        "F_train": F_train,
        "R_train": R_train.astype(np.float32),
        "test_cells": test_cells,
        "test_treatments": test_treatments,
        "Y_test": Y_test,
        "P_test": P_test,
        "D_test": D_test,
        "F_test": F_test,
        "R_test": R_test.astype(np.float32),
        "evaluate_test": make_evaluator(
            Y_test, P_test, D_test, F_test, test_cells, test_treatments, list(gene_cols)
        ),
        "n_test": Y_test.shape[0],
        "gene_cols": list(gene_cols),
        "cell_to_idx": {c: i for i, c in enumerate(all_cells)},
        "treat_to_idx": {t: i for i, t in enumerate(all_treatments)},
        "n_cells": len(all_cells),
        "n_treatments": len(all_treatments),
    }


def frames_to_observations(fc, pv, fdr, ref, delta, gene_cols):
    """Turn the five wide frames from fetch_de_subset into per-observation dicts,
    the layout build_data_dict and de_from_anndata share."""
    out = []
    for df in (fc, pv, fdr, ref, delta):
        values = df[gene_cols].to_numpy(dtype=np.float64)
        out.append(
            {
                (c, t): values[i]
                for i, (c, t) in enumerate(zip(df["cell_line"], df["treatment"]))
            }
        )
    Y, P, F, R, D = out
    return {"Y": Y, "P": P, "F": F, "R": R, "D": D}


# ---------------------------------------------------------------------------
# Misc helpers the notebook uses
# ---------------------------------------------------------------------------


def load_gene_panel(dataset="tahoe"):
    """The static gene panel for a dataset, in its canonical order."""
    path = os.path.join(_REPO_ROOT, "splits", dataset, "static_2k_genes.json")
    with open(path) as f:
        return json.load(f)


def compositional_split(keys, holdout):
    """Split observations into (train_keys, test_keys) for compositional generalization.

    `holdout` maps cell_line -> list of treatments to hold out. Every held-out
    cell line and treatment must still appear elsewhere in train, otherwise the
    per-cell ridge has nothing to regress on and silently falls back to the
    additive baseline — so that is checked here rather than discovered later.
    """
    keys = sorted(keys)
    test_set = {(c, t) for c, treats in holdout.items() for t in treats}
    missing = test_set - set(keys)
    if missing:
        raise ValueError(f"Held-out pairs not present in the data: {sorted(missing)[:5]}")

    train_keys = [k for k in keys if k not in test_set]
    test_keys = [k for k in keys if k in test_set]

    train_cells = {c for c, _ in train_keys}
    train_treats = {t for _, t in train_keys}
    for c, t in test_keys:
        if c not in train_cells:
            raise ValueError(f"Cell line {c!r} is held out entirely — no cross-cell basis in train.")
        if t not in train_treats:
            raise ValueError(f"Treatment {t!r} is held out entirely — it never appears in train.")
    return train_keys, test_keys


__all__ = [
    "TAHOE_CONTROL",
    "TAHOE_TARGET_SUM",
    "WALKTHROUGH_CELL_LINES",
    "WALKTHROUGH_DRUGS",
    "WALKTHROUGH_PLATE",
    "build_data_dict",
    "compositional_split",
    "de_from_anndata",
    "fetch_cell_eval_subset",
    "fetch_de_subset",
    "fetch_pdex_subset",
    "frames_to_observations",
    "load_gene_panel",
    "stack_observations",
    "treatment_label",
    "walkthrough_holdout",
    "walkthrough_treatments",
]
