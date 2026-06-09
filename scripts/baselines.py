"""Compute naive baselines with all six State metrics.

Baselines (applied to FC, deltas, and NLP independently):
  global_mean      — predict global mean per gene
  cell_mean        — predict per-cell-line mean per gene
  treatment_mean   — predict per-treatment mean per gene
  additive         — predict mu + treat_effect + cell_effect (ALS)

Usage:
    python scripts/baselines.py
"""
import json
import sys
import os

import numpy as np
import torch
from rhaister.prepare_combined import prepare_all, evaluate, pvalues_to_fdr_bh, parse_split_name, CACHE_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(REPO_ROOT, "baseline_results.json")
DEFAULT_PREDICTIONS_DIR = os.path.join(REPO_ROOT, "predictions")


BASELINE_NAMES = ["global_mean", "cell_mean", "treatment_mean", "additive", "svd_residual"]
ALS_BASELINE_NAMES = ["global_mean", "cell_mean", "treatment_mean", "additive"]

SVD_RANK = int(os.environ.get("HP_SVD_RANK", "10"))
SVD_ITERS = int(os.environ.get("HP_SVD_ITERS", "20"))


def _als_decomposition(Y, c_idx, t_idx, n_cell, n_treat, n_iter=5):
    """ALS decomposition: mu + cell_effect + treatment_effect."""
    n_genes = Y.shape[1]
    mu = Y.mean(axis=0)
    cell_eff = np.zeros((n_cell, n_genes))
    treat_eff = np.zeros((n_treat, n_genes))
    for _ in range(n_iter):
        for ti in range(n_treat):
            mask = t_idx == ti
            if mask.sum() > 0:
                treat_eff[ti] = (Y[mask] - mu - cell_eff[c_idx[mask]]).mean(axis=0)
        for ci in range(n_cell):
            mask = c_idx == ci
            if mask.sum() > 0:
                cell_eff[ci] = (Y[mask] - mu - treat_eff[t_idx[mask]]).mean(axis=0)
    return mu, cell_eff, treat_eff


def _build_predictions(mu, cell_eff, treat_eff, test_c, test_t, n_test):
    """Build the four additive baseline predictions from ALS components."""
    return {
        "global_mean": np.tile(mu, (n_test, 1)),
        "cell_mean": mu + cell_eff[test_c],
        "treatment_mean": mu + treat_eff[test_t],
        "additive": mu + treat_eff[test_t] + cell_eff[test_c],
    }


def _svd_residual_predictions(Y_train, c_idx, t_idx, mu, cell_eff, treat_eff,
                               n_cell, n_treat, test_c, test_t,
                               rank=SVD_RANK, n_iter=SVD_ITERS):
    """SVD matrix-completion baseline on mean-subtracted residuals.

    Subtracts additive ALS prediction from observed Y_train to form residuals,
    lays them onto a (n_cell, n_treat, n_genes) grid with test positions masked
    as missing, and iteratively fills the missing entries with per-gene truncated
    SVD (rank k) of the (n_cell, n_treat) residual matrix. Final prediction for
    each test (c, t) is mu + cell_eff[c] + treat_eff[t] + completed_residual.
    This is a fair head-to-head with the full model's ridge-regression residual
    predictor — both operate in residual space after ALS mean effects.
    """
    n_train, n_genes = Y_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    residuals = (
        Y_train.astype(np.float32)
        - mu.astype(np.float32)
        - cell_eff[c_idx].astype(np.float32)
        - treat_eff[t_idx].astype(np.float32)
    )

    # Build (n_cell, n_treat, n_genes) residual tensor; observed = True where we have data.
    R = torch.zeros(n_cell, n_treat, n_genes, device=device, dtype=torch.float32)
    mask = torch.zeros(n_cell, n_treat, device=device, dtype=torch.bool)
    c_idx_t = torch.from_numpy(c_idx).long().to(device)
    t_idx_t = torch.from_numpy(t_idx).long().to(device)
    R[c_idx_t, t_idx_t] = torch.from_numpy(residuals).to(device)
    mask[c_idx_t, t_idx_t] = True

    # Iterative per-gene truncated SVD (batched across genes).
    R_gnt = R.permute(2, 0, 1).contiguous()          # (n_genes, n_cell, n_treat)
    R_obs = R_gnt.clone()
    mask_gnt = mask.unsqueeze(0).expand_as(R_gnt)
    k = min(rank, n_cell, n_treat)
    for _ in range(n_iter):
        U, S, Vh = torch.linalg.svd(R_gnt, full_matrices=False)
        S_trunc = S.clone()
        S_trunc[:, k:] = 0
        R_recon = U @ torch.diag_embed(S_trunc) @ Vh
        R_gnt = torch.where(mask_gnt, R_obs, R_recon)

    R_completed = R_gnt.permute(1, 2, 0).cpu().numpy()  # (n_cell, n_treat, n_genes)
    test_c_arr = np.asarray(test_c)
    test_t_arr = np.asarray(test_t)
    residual_test = R_completed[test_c_arr, test_t_arr]
    additive_test = mu + cell_eff[test_c_arr] + treat_eff[test_t_arr]
    return additive_test + residual_test


def _write_baseline_predictions_parquets(
    split_name, predictions_dir,
    fc_preds_test, d_preds_test, fc_preds_train, d_preds_train,
    Y_test, D_test, Y_train, D_train,
    test_cells, test_treatments, train_cells, train_treatments, gene_cols,
):
    """Write per-baseline (cell, treatment, gene) prediction parquets for one split.

    Output schema matches `train.py`'s full-model parquets minus FDR columns:
    cell_line, treatment, gene, split, y_true, y_pred, d_true, d_pred. Two files
    per baseline per split (`_train.parquet` and `_test.parquet`); train rows use
    train-side ground truth and the same ALS fit evaluated at training indices.
    """
    import pandas as pd

    out_dir = predictions_dir or DEFAULT_PREDICTIONS_DIR
    os.makedirs(out_dir, exist_ok=True)
    safe_split = split_name.replace("/", "_")

    genes = np.asarray(gene_cols)
    n_genes = genes.shape[0]

    Y_test_arr = np.asarray(Y_test, dtype=np.float32)
    D_test_arr = np.asarray(D_test, dtype=np.float32)
    Y_train_arr = np.asarray(Y_train, dtype=np.float32)
    D_train_arr = np.asarray(D_train, dtype=np.float32)
    test_cells_arr = np.asarray(test_cells)
    test_treatments_arr = np.asarray(test_treatments)
    train_cells_arr = np.asarray(train_cells)
    train_treatments_arr = np.asarray(train_treatments)

    n_test = Y_test_arr.shape[0]
    n_train = Y_train_arr.shape[0]
    assert Y_test_arr.shape == (n_test, n_genes), f"Y_test shape mismatch: {Y_test_arr.shape}"
    assert Y_train_arr.shape == (n_train, n_genes), f"Y_train shape mismatch: {Y_train_arr.shape}"

    passes = [
        ("test", test_cells_arr, test_treatments_arr, n_test, Y_test_arr, D_test_arr,
         fc_preds_test, d_preds_test),
        ("train", train_cells_arr, train_treatments_arr, n_train, Y_train_arr, D_train_arr,
         fc_preds_train, d_preds_train),
    ]

    written = []
    for name in ALS_BASELINE_NAMES:
        for pass_tag, cells, treats, n_obs, y_true, d_true, fc_preds, d_preds in passes:
            y_pred = np.asarray(fc_preds[name], dtype=np.float32)
            d_pred = np.asarray(d_preds[name], dtype=np.float32)
            assert y_pred.shape == (n_obs, n_genes), f"{name}/{pass_tag} y_pred shape: {y_pred.shape}"
            assert d_pred.shape == (n_obs, n_genes), f"{name}/{pass_tag} d_pred shape: {d_pred.shape}"

            df = pd.DataFrame({
                "cell_line": pd.Categorical(np.repeat(cells, n_genes)),
                "treatment": pd.Categorical(np.repeat(treats, n_genes)),
                "gene":      pd.Categorical(np.tile(genes, n_obs)),
                "split":     pd.Categorical([pass_tag] * (n_obs * n_genes)),
                "y_true":    y_true.ravel(),
                "y_pred":    y_pred.ravel(),
                "d_true":    d_true.ravel(),
                "d_pred":    d_pred.ravel(),
            })
            out = os.path.join(out_dir, f"baseline_{name}__{safe_split}_{pass_tag}.parquet")
            df.to_parquet(out, index=False)
            print(f"Saved baseline predictions: {out}  ({len(df)} rows)")
            written.append(out)
    return written


def compute_baselines(split_name="tahoe_5_holdout", save_predictions=False, predictions_dir=None):
    """Compute baselines for a single split, evaluating all six State metrics.

    When `save_predictions=True`, additionally write per-baseline (cell, treatment,
    gene) parquets for the four ALS baselines and skip the SVD-residual + NLP
    paths (only `d_pred` / `y_pred` are needed downstream). Metrics are not
    computed in that mode and the function returns None.
    """
    data = prepare_all(split_name)
    Y_train = np.array(data["Y_train"], dtype=np.float64)
    D_train = np.array(data["D_train"], dtype=np.float64)
    n_test = data["n_test"]
    test_cells = data["test_cells"]
    test_treatments = data["test_treatments"]
    gene_cols = data["gene_cols"]

    # Load ground truth directly from cache (evaluate_test is one-shot)
    dataset, split = parse_split_name(split_name)
    cache_dir = os.path.join(CACHE_DIR, dataset, split)
    Y_test = np.load(os.path.join(cache_dir, "Y_test.npy"), mmap_mode="r")
    D_test = np.load(os.path.join(cache_dir, "D_test.npy"), mmap_mode="r")

    # Build index maps
    unique_cells = sorted(set(data["train_cells"]))
    unique_treats = sorted(set(data["train_treatments"]))
    cell_map = {c: i for i, c in enumerate(unique_cells)}
    treat_map = {t: i for i, t in enumerate(unique_treats)}
    c_idx = np.array([cell_map[c] for c in data["train_cells"]])
    t_idx = np.array([treat_map[t] for t in data["train_treatments"]])
    test_c = np.array([cell_map[c] for c in test_cells])
    test_t = np.array([treat_map[t] for t in test_treatments])
    n_cell, n_treat = len(unique_cells), len(unique_treats)
    n_train = Y_train.shape[0]

    # ALS on FC and delta targets (always needed; cheap)
    mu_fc, cell_fc, treat_fc = _als_decomposition(Y_train, c_idx, t_idx, n_cell, n_treat)
    fc_preds = _build_predictions(mu_fc, cell_fc, treat_fc, test_c, test_t, n_test)

    mu_d, cell_d, treat_d = _als_decomposition(D_train, c_idx, t_idx, n_cell, n_treat)
    d_preds = _build_predictions(mu_d, cell_d, treat_d, test_c, test_t, n_test)

    if save_predictions:
        # Build train-side predictions from the same ALS fit so test predictions
        # remain genuinely OOD (mu/cell_eff/treat_eff fit on Y_train/D_train only).
        fc_preds_train = _build_predictions(mu_fc, cell_fc, treat_fc, c_idx, t_idx, n_train)
        d_preds_train = _build_predictions(mu_d, cell_d, treat_d, c_idx, t_idx, n_train)
        _write_baseline_predictions_parquets(
            split_name, predictions_dir,
            fc_preds, d_preds, fc_preds_train, d_preds_train,
            Y_test, D_test, Y_train, D_train,
            test_cells, test_treatments, data["train_cells"], data["train_treatments"],
            gene_cols,
        )
        return None

    # Full metric path: SVD residual + NLP + evaluation.
    P_train = np.array(data["P_train"], dtype=np.float64)
    P_test = np.load(os.path.join(cache_dir, "P_test.npy"), mmap_mode="r")
    F_test = np.load(os.path.join(cache_dir, "F_test.npy"), mmap_mode="r")

    fc_preds["svd_residual"] = _svd_residual_predictions(
        Y_train, c_idx, t_idx, mu_fc, cell_fc, treat_fc,
        n_cell, n_treat, test_c, test_t,
    )
    d_preds["svd_residual"] = _svd_residual_predictions(
        D_train, c_idx, t_idx, mu_d, cell_d, treat_d,
        n_cell, n_treat, test_c, test_t,
    )

    NLP_train = -np.log10(np.clip(P_train, 1e-30, 1.0))
    mu_nlp, cell_nlp, treat_nlp = _als_decomposition(NLP_train, c_idx, t_idx, n_cell, n_treat)
    nlp_preds = _build_predictions(mu_nlp, cell_nlp, treat_nlp, test_c, test_t, n_test)
    nlp_preds["svd_residual"] = _svd_residual_predictions(
        NLP_train, c_idx, t_idx, mu_nlp, cell_nlp, treat_nlp,
        n_cell, n_treat, test_c, test_t,
    )

    # Evaluate each baseline
    results = {}
    for name in BASELINE_NAMES:
        Y_pred = fc_preds[name]
        D_pred = d_preds[name]
        NLP_pred = np.clip(nlp_preds[name], 0, 30)
        P_pred = np.power(10.0, -NLP_pred)
        F_pred = pvalues_to_fdr_bh(P_pred)

        metrics = evaluate(
            Y_test, Y_pred, P_test, P_pred,
            D_test, D_pred, F_test, F_pred,
            test_cells, test_treatments, gene_cols,
            compute_discrimination=True,
        )
        results[name] = metrics

    return results


SPLITS = [f"tahoe_{i}_holdout" for i in range(5, 10)]

METRIC_KEYS = [
    "pdex_static/pearson_delta_mean",
    "pdex_static/auprc_p05",
    "state/pearson_delta_mean",
    "state/spearman_lfc_sig_mean",
    "state/pr_auc_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
    "state/discrimination_mean",
]
SHORT_NAMES = ["fc_pearson", "auprc_p05", "delta_pearson",
               "spearman_lfc", "pr_auc", "de_overlap", "spearman_sig", "discrim"]


def main():
    split_name = None
    log_wandb = "--wandb" in sys.argv
    save_predictions = "--save-predictions" in sys.argv
    predictions_dir = None
    args = list(sys.argv[1:])
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--wandb", "--save-predictions"):
            i += 1
            continue
        if arg == "--predictions-dir":
            predictions_dir = args[i + 1]
            i += 2
            continue
        # Accept any split name (qualified "<dataset>/<split>" or legacy form)
        split_name = arg
        i += 1

    if save_predictions:
        # Predictions-only mode: skip metrics + JSON write so other consumers'
        # `svd_residual` rows in baseline_results.json aren't clobbered.
        targets = [split_name] if split_name else SPLITS
        for split in targets:
            print(f"Saving baseline predictions for {split}...")
            compute_baselines(split, save_predictions=True, predictions_dir=predictions_dir)
        return

    if split_name:
        # Single split mode
        results = compute_baselines(split_name)
        _print_table(f"Baselines ({split_name})", results)
        _save_results({split_name: results})
        if log_wandb:
            _log_wandb(split_name, results)
    else:
        # Multi-split mode: run all splits, show per-split + summary
        all_results = {}
        for split in SPLITS:
            print(f"Computing baselines for {split}...")
            all_results[split] = compute_baselines(split)
            if log_wandb:
                _log_wandb(split, all_results[split])

        _save_results(all_results)

        for split in SPLITS:
            _print_table(split, all_results[split])

        # Summary: mean ± std across splits
        print(f"\n{'='*60}")
        print("SUMMARY (mean ± std across splits)")
        print(f"{'='*60}")
        print(f"\n{'Baseline':<18}" + "".join(f"{s:>14}" for s in SHORT_NAMES))
        print("-" * (18 + 14 * len(METRIC_KEYS)))
        for name in BASELINE_NAMES:
            parts = []
            for k in METRIC_KEYS:
                vals = [all_results[s][name].get(k, float('nan')) for s in SPLITS]
                parts.append(f"{np.mean(vals):8.4f}±{np.std(vals):.3f}")
            print(f"{name:<18}" + "".join(f"{p:>14}" for p in parts))
        print()


def _save_results(new_results):
    """Merge new results into baseline_results.json, preserving other splits."""
    existing = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
    for split, split_results in new_results.items():
        existing[split] = {
            name: {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
            for name, metrics in split_results.items()
        }
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    print(f"Saved results to {RESULTS_FILE}")


def _log_wandb(split, results):
    import wandb
    split_config = f"configs/{split}/generalization_converted_cell_lines_3b.toml"
    for name in BASELINE_NAMES:
        metrics = {k: results[name].get(k, float('nan')) for k in METRIC_KEYS}
        wandb.init(
            project="perturbation-eval",
            job_type="baseline",
            name=f"baseline_{name}",
            config={"split_config": split_config, "baseline": name},
            reinit=True,
        )
        wandb.log(metrics)
        wandb.finish()


def _print_table(title, results):
    print(f"\n{'Baseline':<18}" + "".join(f"{s:>14}" for s in SHORT_NAMES) + f"   [{title}]")
    print("-" * (18 + 14 * len(METRIC_KEYS)))
    for name in BASELINE_NAMES:
        m = results[name]
        vals = "".join(f"{m.get(k, float('nan')):14.4f}" for k in METRIC_KEYS)
        print(f"{name:<18}{vals}")


if __name__ == "__main__":
    main()
