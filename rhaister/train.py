"""
train.py — Full prediction model for gene expression fold changes, p-values,
and expression deltas.

Public interface:
    train_and_evaluate(experiment_name, split_name, log, data)
        — runs the full model: ALS additive baseline + unweighted
          drug-as-linear-combination ridge per view + small calibration MLP on
          the p-value path.

Internal components:
    _als_decompose(...)       — ALS additive decomposition: Y ≈ μ + treat_eff + cell_eff
    _compute_regression(...)  — drug-as-linear-combination ridge regression

Run: python train.py [experiment_name]
"""

import os
import sys
import time
import numpy as np
import torch
from rhaister.prepare_combined import (
    prepare_all, log_result, pvalues_to_fdr_bh, parse_split_name, _cache_path,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _save_predictions_parquet(experiment_name, split_name, Y_pred, D_pred, F_pred, data):
    """Write long-format parquet of predictions alongside truths.

    One row per (cell_line, treatment, gene). In the default "test" pass,
    truths are loaded from the combined cache (evaluator closures them). In
    the "train" pass (triggered by HP_SAVE_PREDICTIONS=full), train rows have
    been swapped into the test position, so truths come from data["*_train"].
    """
    import pandas as pd

    pass_tag = os.environ.get("_RHAISTER_SAVE_PASS", "test")
    if pass_tag == "train":
        Y_truth = np.asarray(data["Y_train"])
        D_truth = np.asarray(data["D_train"])
        F_truth = np.asarray(data["F_train"])
    else:
        dataset, split = parse_split_name(split_name)
        cp = _cache_path(dataset, split)
        Y_truth = np.load(os.path.join(cp, "Y_test.npy"), mmap_mode="r")
        D_truth = np.load(os.path.join(cp, "D_test.npy"), mmap_mode="r")
        F_truth = np.load(os.path.join(cp, "F_test.npy"), mmap_mode="r")

    cells = np.asarray(data["test_cells"])
    treatments = np.asarray(data["test_treatments"])
    genes = np.asarray(data["gene_cols"])
    n_obs, n_genes = Y_truth.shape

    df = pd.DataFrame({
        "cell_line":  pd.Categorical(np.repeat(cells, n_genes)),
        "treatment":  pd.Categorical(np.repeat(treatments, n_genes)),
        "gene":       pd.Categorical(np.tile(genes, n_obs)),
        "split":      pd.Categorical([pass_tag] * (n_obs * n_genes)),
        "y_true":     np.asarray(Y_truth).ravel(),
        "y_pred":     np.asarray(Y_pred).ravel(),
        "d_true":     np.asarray(D_truth).ravel(),
        "d_pred":     np.asarray(D_pred).ravel(),
        "fdr_true":   np.asarray(F_truth).ravel(),
        "fdr_pred":   np.asarray(F_pred).ravel(),
    })

    mode = os.environ.get("HP_SAVE_PREDICTIONS", "1")
    safe_split = split_name.replace("/", "_")
    out_dir = os.path.join(REPO_ROOT, "predictions")
    if mode == "full":
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{experiment_name}__{safe_split}_{pass_tag}.parquet")
    elif mode == "1":
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{experiment_name}__{safe_split}.parquet")
    else:
        # Explicit path. In a train pass (only reachable via full mode), suffix it.
        if pass_tag == "train":
            base, ext = os.path.splitext(mode)
            out = base + "_train" + (ext or ".parquet")
        else:
            out = mode
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved predictions ({pass_tag}): {out}  ({len(df)} rows)")
    return out

torch.set_float32_matmul_precision("high")  # TF32 on Ampere+ for ~2x matmul speedup


def _compute_regression_zeroshot(data, lam=1.0, center=True, device=None,
                                 quiet=False):
    """Zeroshot drug regression: per held-out cell hc, fit Y[c, dy] ~ X[c] @ W
    on non-holdout cells c, then predict Y[hc, dy] = X[hc] @ W.

    X[c] = log1p(R[c]) concatenated with log1p(L[c]) (last column) when L is
    in data — R is per-gene mean control expression, L is per-observation mean
    library size; log1p puts both on a comparable scale for the ridge.

    Uses the dual-form ridge (SVD of X_w, n_nc << n_features) so each hc costs
    one SVD + per-drug matmul. HP_RR_RANK applies as truncation of the leading
    singular components."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # HP_ZS_FEATURES selects which feature blocks to use (any subset of R/L/C/H).
    # R: per-gene control_expression mean (log1p); L: library size (log1p);
    # C: per-cell DMSO centroid embedding; H: per-(cell, plate) HVG centroid
    # (log1p). All blocks concatenated raw — no per-column or per-block scaling.
    feat_select = os.environ.get("HP_ZS_FEATURES", "RLCH").upper()
    use_R = "R" in feat_select
    use_L = "L" in feat_select and ("L_train" in data and "L_test" in data)
    use_C = (
        "C" in feat_select
        and "C_train" in data and "C_test" in data
        and np.asarray(data["C_train"]).shape[1] > 0
    )
    use_H = (
        "H" in feat_select
        and "H_train" in data and "H_test" in data
        and np.asarray(data["H_train"]).shape[1] > 0
    )
    if not (use_R or use_L or use_C or use_H):
        raise ValueError(f"HP_ZS_FEATURES={feat_select!r}: no usable feature blocks selected")

    Y_train_g = torch.from_numpy(np.array(data["Y_train"], dtype=np.float32)).to(device)
    if use_R:
        R_train = torch.from_numpy(np.array(data["R_train"], dtype=np.float32)).to(device)
        R_test_np = np.array(data["R_test"], dtype=np.float32)
    if use_L:
        L_train = torch.from_numpy(np.array(data["L_train"], dtype=np.float32)).to(device)
        L_test_np = np.array(data["L_test"], dtype=np.float32)
    if use_C:
        C_train = torch.from_numpy(np.array(data["C_train"], dtype=np.float32)).to(device)
        C_test_np = np.array(data["C_test"], dtype=np.float32)
    if use_H:
        H_train = torch.from_numpy(np.array(data["H_train"], dtype=np.float32)).to(device)
        H_test_np = np.array(data["H_test"], dtype=np.float32)

    train_cells = list(data["train_cells"])
    train_treats = list(data["train_treatments"])
    test_cells = list(data["test_cells"])
    test_treats = list(data["test_treatments"])
    n_test = data["n_test"]
    n_genes = Y_train_g.shape[1]

    # Per-cell averages.
    unique_cells = sorted(set(train_cells))
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    n_nc = len(unique_cells)
    c_idx = torch.tensor([cell_to_idx[c] for c in train_cells], device=device, dtype=torch.long)
    cell_count = torch.bincount(c_idx, minlength=n_nc).float().clamp(min=1)

    holdout_cells = sorted(set(test_cells))
    hc_to_idx = {c: i for i, c in enumerate(holdout_cells)}
    n_hc = len(holdout_cells)
    hc_idx_arr = torch.tensor([hc_to_idx[c] for c in test_cells], device=device, dtype=torch.long)
    hc_count = torch.bincount(hc_idx_arr, minlength=n_hc).float().clamp(min=1)

    def _avg_per_cell(arr_train, idx_train, n_groups, counts):
        """Mean of arr_train rows grouped by idx_train -> (n_groups, n_features)."""
        if arr_train.dim() == 1:
            s = torch.zeros(n_groups, device=device)
            s.scatter_add_(0, idx_train, arr_train)
            return s / counts
        n_f = arr_train.shape[1]
        s = torch.zeros(n_groups, n_f, device=device)
        s.scatter_add_(0, idx_train.unsqueeze(1).expand(-1, n_f), arr_train)
        return s / counts.unsqueeze(1)

    feat_train_blocks = []
    feat_test_blocks = []
    sources = []
    if use_R:
        R_cell = _avg_per_cell(R_train, c_idx, n_nc, cell_count)
        R_hc = _avg_per_cell(torch.from_numpy(R_test_np).to(device), hc_idx_arr, n_hc, hc_count)
        feat_train_blocks.append(torch.log1p(R_cell))
        feat_test_blocks.append(torch.log1p(R_hc))
        sources.append(f"log1p(R){R_cell.shape[1]}")
    if use_L:
        L_cell = _avg_per_cell(L_train, c_idx, n_nc, cell_count)
        L_hc = _avg_per_cell(torch.from_numpy(L_test_np).to(device), hc_idx_arr, n_hc, hc_count)
        feat_train_blocks.append(torch.log1p(L_cell).unsqueeze(1))
        feat_test_blocks.append(torch.log1p(L_hc).unsqueeze(1))
        sources.append("log1p(L)")
    if use_C:
        C_cell = _avg_per_cell(C_train, c_idx, n_nc, cell_count)
        C_hc = _avg_per_cell(torch.from_numpy(C_test_np).to(device), hc_idx_arr, n_hc, hc_count)
        feat_train_blocks.append(C_cell)
        feat_test_blocks.append(C_hc)
        sources.append(f"centroid({C_cell.shape[1]})")
    if use_H:
        H_cell = _avg_per_cell(H_train, c_idx, n_nc, cell_count)
        H_hc = _avg_per_cell(torch.from_numpy(H_test_np).to(device), hc_idx_arr, n_hc, hc_count)
        feat_train_blocks.append(torch.log1p(H_cell))
        feat_test_blocks.append(torch.log1p(H_hc))
        sources.append(f"log1p(H){H_cell.shape[1]}")

    X_cell = torch.cat(feat_train_blocks, dim=1)  # (n_nc, n_feat)
    X_hc = torch.cat(feat_test_blocks, dim=1)  # (n_hc, n_feat)
    n_feat = X_cell.shape[1]
    if not quiet:
        print(f"  zeroshot reg [HP_ZS_FEATURES={feat_select}]: feature stack = "
              f"{' + '.join(sources)} = {n_feat} dims (no scaling)")

    ct_to_row = {}
    for i, (c, t) in enumerate(zip(train_cells, train_treats)):
        ct_to_row[(c, t)] = i

    test_drugs_by_hc = {}
    test_idx_by_hc_drug = {}
    for i, (c, t) in enumerate(zip(test_cells, test_treats)):
        test_drugs_by_hc.setdefault(c, set()).add(t)
        test_idx_by_hc_drug[(c, t)] = i

    Y_pred = np.zeros((n_test, n_genes), dtype=np.float64)
    covered = np.zeros(n_test, dtype=bool)
    rr_rank = int(os.environ.get("HP_RR_RANK", "75"))

    for hc in holdout_cells:
        candidate_drugs = test_drugs_by_hc[hc]
        common_drugs = sorted(
            dy for dy in candidate_drugs
            if all((c, dy) in ct_to_row for c in unique_cells)
        )
        if not common_drugs:
            continue
        n_dy = len(common_drugs)
        if not quiet:
            print(f"  zeroshot reg {hc}: covered drugs {n_dy}/{len(candidate_drugs)}, cells {n_nc}")

        dy_row_idx = torch.tensor(
            [ct_to_row[(c, dy)] for c in unique_cells for dy in common_drugs],
            device=device, dtype=torch.long,
        )
        Y_all = Y_train_g[dy_row_idx].reshape(n_nc, n_dy, n_genes)

        if center:
            # Treat-mean as the explicit additive baseline (analogue of ALS
            # additive in the non-zeroshot model — there cell_eff[hc] absorbs
            # the cell-level offset; here hc has no panel obs, so we use the
            # per-drug mean across non-holdout cells as the starting point and
            # let the ridge learn only the residual cell-specific deviation).
            # W = 0 ⇒ Y_pred = treat_mean.
            mu_X = X_cell.mean(dim=0)        # (n_feat,)
            mu_Y = Y_all.mean(dim=0)         # (n_dy, n_genes) — per-drug treat-mean
            X_w = X_cell - mu_X.unsqueeze(0)
            Y_w = Y_all - mu_Y.unsqueeze(0)
        else:
            X_w = X_cell
            Y_w = Y_all

        # Dual-form ridge: X_w = U S V^T (thin SVD); W = V diag(S/(S^2+lam)) U^T Y_w.
        U, S, Vh = torch.linalg.svd(X_w.double(), full_matrices=False)
        # Apply rr_rank truncation by keeping only the top-k singular components.
        if rr_rank > 0 and rr_rank < len(S):
            S = S[:rr_rank]
            U = U[:, :rr_rank]
            Vh = Vh[:rr_rank, :]
        ridge_factor = S / (S * S + lam)  # (r,)

        Y_w_flat = Y_w.reshape(n_nc, -1).double()  # (n_nc, n_dy * n_genes)
        UTy = U.T @ Y_w_flat  # (r, n_dy * n_genes)

        X_hc_vec = X_hc[hc_to_idx[hc]].double()  # (n_feat,)
        if center:
            X_hc_centered = X_hc_vec - mu_X.double()
        else:
            X_hc_centered = X_hc_vec
        rhc_V = X_hc_centered @ Vh.T  # (r,)
        coef = rhc_V * ridge_factor  # (r,)
        pred_flat = coef @ UTy  # (n_dy * n_genes,)
        pred = pred_flat.reshape(n_dy, n_genes)
        if center:
            # Add the per-drug treat-mean back — the ridge produced the residual.
            pred = pred + mu_Y.double()
        pred_np = pred.cpu().numpy()

        for j, dy in enumerate(common_drugs):
            test_i = test_idx_by_hc_drug.get((hc, dy))
            if test_i is not None:
                Y_pred[test_i] = pred_np[j]
                covered[test_i] = True

    del Y_train_g
    torch.cuda.empty_cache()
    if not quiet:
        print(f"  zeroshot reg: {covered.sum()}/{n_test} test observations covered")
    return Y_pred, covered


def _compute_regression_zeroshot_diagonal(data, lam=1.0, device=None,
                                          quiet=False):
    """Diagonal zeroshot model:
        Y_hat(c, t, g) = treat_mean(t, g) + gamma_{t,g} * (z_{c,g} - z_mean_g)

    Per-(treatment, gene) scalar coefficient — no cross-gene mixing. Equivalent
    to a univariate weighted ridge per (drug, gene) where the regressor is the
    cell's own value at that gene. Picks z from HP_ZS_DIAG_Z (default "H", i.e.
    HVG centroid baseline expression; "R" uses log1p control_expression).

    The denominator only depends on (gene, hc) via the cell weights, so all
    drugs share it for a given hc — the loop solves once per hc and reuses."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    z_source = os.environ.get("HP_ZS_DIAG_Z", "H").upper()
    if z_source == "R":
        Z_train_t = torch.log1p(torch.from_numpy(np.array(data["R_train"], dtype=np.float32))).to(device)
        Z_test_t = torch.log1p(torch.from_numpy(np.array(data["R_test"], dtype=np.float32))).to(device)
        z_label = "log1p(R)"
    elif z_source == "H":
        if "H_train" not in data:
            raise ValueError("HP_ZS_DIAG_Z=H but H_train/H_test not in data (cell_centroid_hvg_path not configured?)")
        Z_train_t = torch.log1p(torch.from_numpy(np.array(data["H_train"], dtype=np.float32))).to(device)
        Z_test_t = torch.log1p(torch.from_numpy(np.array(data["H_test"], dtype=np.float32))).to(device)
        z_label = "log1p(H)"
    else:
        raise ValueError(f"HP_ZS_DIAG_Z must be 'R' or 'H', got {z_source!r}")

    # Imputation may have expanded Y_train / train_cells beyond the original
    # R/L/C/H arrays.  The diagonal model needs Z and Y to be row-aligned, so
    # truncate to the original (pre-imputation) length that matches Z_train_t.
    n_orig = Z_train_t.shape[0]
    Y_train_g = torch.from_numpy(np.array(data["Y_train"], dtype=np.float32)[:n_orig]).to(device)
    train_cells = list(data["train_cells"])[:n_orig]
    train_treats = list(data["train_treatments"])[:n_orig]
    test_cells = list(data["test_cells"])
    test_treats = list(data["test_treatments"])
    n_test = data["n_test"]
    n_genes = Y_train_g.shape[1]
    if Z_train_t.shape[1] != n_genes:
        raise ValueError(f"diagonal model needs z and Y on same gene axis "
                         f"(got z={Z_train_t.shape[1]}, Y={n_genes})")

    unique_cells = sorted(set(train_cells))
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    n_nc = len(unique_cells)
    c_idx = torch.tensor([cell_to_idx[c] for c in train_cells], device=device, dtype=torch.long)
    cell_count = torch.bincount(c_idx, minlength=n_nc).float().clamp(min=1)
    Z_cell_sum = torch.zeros(n_nc, n_genes, device=device)
    Z_cell_sum.scatter_add_(0, c_idx.unsqueeze(1).expand(-1, n_genes), Z_train_t)
    Z_cell = Z_cell_sum / cell_count.unsqueeze(1)

    holdout_cells = sorted(set(test_cells))
    hc_to_idx = {c: i for i, c in enumerate(holdout_cells)}
    n_hc = len(holdout_cells)
    hc_idx_arr = torch.tensor([hc_to_idx[c] for c in test_cells], device=device, dtype=torch.long)
    hc_count = torch.bincount(hc_idx_arr, minlength=n_hc).float().clamp(min=1)
    Z_hc_sum = torch.zeros(n_hc, n_genes, device=device)
    Z_hc_sum.scatter_add_(0, hc_idx_arr.unsqueeze(1).expand(-1, n_genes), Z_test_t)
    Z_hc = Z_hc_sum / hc_count.unsqueeze(1)

    # Mean-center z across non-holdout cells (the residual analog).
    z_mean = Z_cell.mean(dim=0)
    Z_cell_c = (Z_cell - z_mean.unsqueeze(0)).double()
    Z_hc_c = (Z_hc - z_mean.unsqueeze(0)).double()

    # Diagnostic: HP_ZS_DIAG_PERM lets us pair each output gene g with a
    # randomly-chosen input gene perm[g] instead of with g itself. Same
    # parameter count, same algorithm — tests whether the g→g pairing is
    # specifically informative or whether "any single-gene predictor per
    # output" is what matters (i.e., the model's gain over ridge is about
    # parameter count, not about the diagonal structure).
    perm_mode = os.environ.get("HP_ZS_DIAG_PERM", "")
    if perm_mode and perm_mode != "identity":
        seed = int(perm_mode) if perm_mode.lstrip("-").isdigit() else 0
        gen = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(n_genes, generator=gen).to(device)
        Z_cell_c = Z_cell_c[:, perm]
        Z_hc_c = Z_hc_c[:, perm]
        if not quiet:
            print(f"  zeroshot diagonal: PERMUTED input gene axis with seed={seed} "
                  f"(testing whether g→g pairing matters)")

    ct_to_row = {}
    for i, (c, t) in enumerate(zip(train_cells, train_treats)):
        ct_to_row[(c, t)] = i
    test_drugs_by_hc = {}
    test_idx_by_hc_drug = {}
    for i, (c, t) in enumerate(zip(test_cells, test_treats)):
        test_drugs_by_hc.setdefault(c, set()).add(t)
        test_idx_by_hc_drug[(c, t)] = i

    Y_pred = np.zeros((n_test, n_genes), dtype=np.float64)
    covered = np.zeros(n_test, dtype=bool)

    if not quiet:
        print(f"  zeroshot diagonal: z={z_label}, per-(drug, gene) scalar gamma, mean-centered z")

    for hc in holdout_cells:
        common_drugs = sorted(
            dy for dy in test_drugs_by_hc[hc]
            if all((c, dy) in ct_to_row for c in unique_cells)
        )
        if not common_drugs:
            continue
        n_dy = len(common_drugs)
        if not quiet:
            print(f"  zeroshot diag {hc}: covered drugs {n_dy}/{len(test_drugs_by_hc[hc])}, cells {n_nc}")

        dy_row_idx = torch.tensor(
            [ct_to_row[(c, dy)] for c in unique_cells for dy in common_drugs],
            device=device, dtype=torch.long,
        )
        Y_all = Y_train_g[dy_row_idx].reshape(n_nc, n_dy, n_genes).double()

        # Treat-mean as additive baseline (per (drug, gene))
        mu_Y = Y_all.mean(dim=0)               # (n_dy, n_genes)
        Y_resid = Y_all - mu_Y.unsqueeze(0)    # (n_nc, n_dy, n_genes)

        # Per-(drug, gene) or per-(gene-only-shared) univariate ridge.
        # HP_ZS_DIAG_SHARED=1 collapses the drug axis: one gamma_g per gene
        # shared across all drugs (the report's row-2 "shared cell-line centroid
        # effect"). Default is drug-specific gamma_{t,g} (row 3).
        # HP_ZS_DIAG_FORCE_ZERO=1 sets gamma=0 and predicts treat-mean only —
        # gives a "panel-free per-drug mean" baseline through the same pipeline
        # (calnet etc.) so it's apples-to-apples with the diagonal model.
        shared_gamma = os.environ.get("HP_ZS_DIAG_SHARED", "0") == "1"
        force_zero_gamma = os.environ.get("HP_ZS_DIAG_FORCE_ZERO", "0") == "1"
        z_hc_vec = Z_hc_c[hc_to_idx[hc]]       # (n_genes,)
        if force_zero_gamma:
            # γ = 0: prediction is just the per-(drug, gene) treat-mean baseline.
            pred = mu_Y.expand(n_dy, n_genes).clone()
        elif shared_gamma:
            # Sum residuals over drugs too: gamma_g pools across all (c, t).
            num_g = torch.einsum('cg,ctg->g', Z_cell_c, Y_resid)           # (n_genes,)
            den_g = n_dy * (Z_cell_c * Z_cell_c).sum(dim=0) + lam          # (n_genes,)
            gamma_g = num_g / den_g                                        # (n_genes,)
            pred = mu_Y + gamma_g.unsqueeze(0) * z_hc_vec.unsqueeze(0)     # (n_dy, n_genes)
        else:
            #   gamma_{t,g} = sum_c z_{c,g} * y_resid_{c,t,g} / (sum_c z_{c,g}^2 + lam)
            num = torch.einsum('cg,ctg->tg', Z_cell_c, Y_resid)            # (n_dy, n_genes)
            den = (Z_cell_c * Z_cell_c).sum(dim=0) + lam                   # (n_genes,)
            gamma = num / den.unsqueeze(0)                                 # (n_dy, n_genes)
            pred = mu_Y + gamma * z_hc_vec.unsqueeze(0)                    # (n_dy, n_genes)
        pred_np = pred.cpu().numpy()

        for j, dy in enumerate(common_drugs):
            test_i = test_idx_by_hc_drug.get((hc, dy))
            if test_i is not None:
                Y_pred[test_i] = pred_np[j]
                covered[test_i] = True

    del Y_train_g, Z_train_t, Z_test_t
    torch.cuda.empty_cache()
    if not quiet:
        print(f"  zeroshot diagonal: {covered.sum()}/{n_test} test observations covered")
    return Y_pred, covered


def _impute_view_into_data(data, Y_view, mu, cell_eff, treat_eff, cell_map, treat_map, label=""):
    """Return a data-dict copy suitable for _compute_regression, with Y_train set
    to Y_view. When HP_IMPUTE_MISSING=1, append synthesized rows for every
    (cell_line, treatment) combo in cell_map x treat_map that isn't already
    observed AND isn't in the test set, using the ALS additive prediction
    mu + cell_eff[c] + treat_eff[t]. This is NaN-aware ALS matrix completion —
    ALS (already fit on observed rows only) gives an unbiased estimator for
    missing entries, densifying the regression basis without dropping cells.
    The test-position exclusion is critical: without it, imputed (held_out,
    test_cyt) rows leak into the regression basis D_x."""
    out = {k: v for k, v in data.items()}
    out["Y_train"] = np.asarray(Y_view)

    if os.environ.get("HP_IMPUTE_MISSING", "1") != "1":
        return out

    train_cells = np.asarray(data["train_cells"])
    train_tr = np.asarray(data["train_treatments"])
    present = set(zip(train_cells.tolist(), train_tr.tolist()))
    test_positions = set(zip(
        np.asarray(data["test_cells"]).tolist(),
        np.asarray(data["test_treatments"]).tolist(),
    ))
    missing = [
        (c, t) for c in cell_map for t in treat_map
        if (c, t) not in present and (c, t) not in test_positions
    ]
    if not missing:
        return out

    n_genes = out["Y_train"].shape[1]
    syn = np.empty((len(missing), n_genes), dtype=out["Y_train"].dtype)
    for i, (c, t) in enumerate(missing):
        syn[i] = mu + cell_eff[cell_map[c]] + treat_eff[treat_map[t]]

    out["Y_train"] = np.concatenate([out["Y_train"], syn], axis=0)
    out["train_cells"] = np.concatenate([train_cells, np.array([c for c, _ in missing])])
    out["train_treatments"] = np.concatenate([train_tr, np.array([t for _, t in missing])])
    if label:
        print(f"  {label}: imputed +{len(missing)} missing (cell,treat) rows via ALS")
    return out


def _als_decompose(experiment_name="unified", split_name="tahoe_5_holdout", log=True, data=None, return_pred=False):
    """ALS decomposition: Y ≈ μ + treat_eff + cell_eff.
    When return_pred=True, returns a dict of intermediate results without evaluating."""
    if data is None:
        data = prepare_all(split_name)
    train_cells, train_tr = data["train_cells"], data["train_treatments"]
    evaluate_test = data["evaluate_test"]
    test_cells, test_tr = data["test_cells"], data["test_treatments"]
    n_genes, n_train, n_test = data["Y_train"].shape[1], data["Y_train"].shape[0], data["n_test"]

    unique_treatments = sorted(set(train_tr))
    local_treat_map = {t: i for i, t in enumerate(unique_treatments)}
    unique_cells = sorted(set(train_cells))
    local_cell_map = {c: i for i, c in enumerate(unique_cells)}
    train_t_idx = np.array([local_treat_map[t] for t in train_tr])
    train_c_idx = np.array([local_cell_map[c] for c in train_cells])
    test_t_local = np.array([local_treat_map[t] for t in test_tr])
    # Sentinel index for cells absent from train (zeroshot held-out cells): we
    # pad cell_effects with a zero row at this index so the additive prediction
    # silently drops the cell term for hc, matching "no cell_eff for hc".
    held_out_cell_idx = len(unique_cells)
    test_c_local = np.array([local_cell_map.get(c, held_out_cell_idx) for c in test_cells])

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_treat, n_cell = len(unique_treatments), len(unique_cells)

    Y_fc = torch.from_numpy(np.array(data["Y_train"], dtype=np.float32)).to(device)

    Y_train_g = Y_fc

    train_t_g = torch.from_numpy(train_t_idx).long().to(device)
    train_c_g = torch.from_numpy(train_c_idx).long().to(device)

    # P-value weighting for ALS: give more weight to confident measurements
    pval_weight_alpha = float(os.environ.get("HP_PW_ALPHA", "0.0"))
    if pval_weight_alpha > 0:
        P_train_g = torch.from_numpy(np.array(data["P_train"], dtype=np.float32)).to(device)
        # w = 1 + alpha * (-log10(p)), clamped to avoid inf at p=0
        pval_weights = 1.0 + pval_weight_alpha * (-torch.log10(P_train_g.clamp(min=1e-30)))
        del P_train_g
        print(f"[{time.time()-t0:.0f}s] P-value weighting: alpha={pval_weight_alpha}, mean_w={pval_weights.mean().item():.2f}, max_w={pval_weights.max().item():.1f}")
    else:
        pval_weights = None

    if pval_weights is not None:
        mu_g = (Y_train_g * pval_weights).sum(0) / pval_weights.sum(0).clamp(min=1)
    else:
        mu_g = Y_train_g.mean(0)

    n_cols = Y_train_g.shape[1]
    treat_eff = torch.zeros(n_treat, n_cols, device=device)
    cell_eff = torch.zeros(n_cell, n_cols, device=device)
    t_exp = train_t_g.unsqueeze(1).expand(n_train, n_cols)
    c_exp = train_c_g.unsqueeze(1).expand(n_train, n_cols)

    n_als = int(os.environ.get("HP_ALS", "5"))
    if pval_weights is not None:
        tc = torch.zeros(n_treat, n_cols, device=device); tc.scatter_add_(0, t_exp, pval_weights); tc.clamp_(min=1)
        cc = torch.zeros(n_cell, n_cols, device=device); cc.scatter_add_(0, c_exp, pval_weights); cc.clamp_(min=1)
        for _ in range(n_als):
            r = (Y_train_g - mu_g - cell_eff[train_c_g]) * pval_weights
            treat_eff.zero_(); treat_eff.scatter_add_(0, t_exp, r); treat_eff /= tc
            r = (Y_train_g - mu_g - treat_eff[train_t_g]) * pval_weights
            cell_eff.zero_(); cell_eff.scatter_add_(0, c_exp, r); cell_eff /= cc
        del tc, cc, pval_weights
    else:
        tc = torch.bincount(train_t_g, minlength=n_treat).float().clamp(min=1).unsqueeze(1)
        cc = torch.bincount(train_c_g, minlength=n_cell).float().clamp(min=1).unsqueeze(1)
        for _ in range(n_als):
            r = Y_train_g - mu_g - cell_eff[train_c_g]
            treat_eff.zero_(); treat_eff.scatter_add_(0, t_exp, r); treat_eff /= tc
            r = Y_train_g - mu_g - treat_eff[train_t_g]
            cell_eff.zero_(); cell_eff.scatter_add_(0, c_exp, r); cell_eff /= cc

    del Y_train_g

    mu = mu_g.cpu().numpy()
    treat_effects = treat_eff.cpu().numpy()
    cell_effects = cell_eff.cpu().numpy()
    del treat_eff, cell_eff, mu_g

    # Pad cell_effects with one zero row at the held-out-cell sentinel index so
    # the additive prediction silently drops the cell term for any test cell
    # that wasn't in train (zeroshot). cell_effects itself stays the original
    # shape for downstream code that indexes by local_cell_map.
    _ce_for_test = np.vstack([cell_effects, np.zeros((1, n_cols), dtype=cell_effects.dtype)])
    additive = mu + treat_effects[test_t_local] + _ce_for_test[test_c_local]
    Y_pred = additive.astype(np.float64)
    print(f"[{time.time()-t0:.0f}s] ALS done")

    if return_pred:
        return {
            "full": Y_pred,
            "mu": mu, "treat_effects": treat_effects, "cell_effects": cell_effects,
            "local_treat_map": local_treat_map, "local_cell_map": local_cell_map,
        }
    metrics = evaluate_test(Y_pred)
    for k, v in sorted(metrics.items()):
        print(f"RESULT {k}={v:.6f}" if isinstance(v, float) else f"RESULT {k}={v}")
    if log:
        log_result(experiment_name, metrics)
    return metrics


def run_both(name):
    """Run both PCL and static evaluations."""
    import prepare as prepare_static
    import prepare_pcl
    global prepare_all, log_result
    orig_prepare, orig_log = prepare_all, log_result

    prepare_all = prepare_pcl.prepare_all
    log_result = prepare_pcl.log_result
    print("=== PCL evaluation ===")
    pcl = train_and_evaluate(name)
    prepare_all = prepare_static.prepare_all
    log_result = prepare_static.log_result
    print("\n=== Static evaluation ===")
    static = train_and_evaluate(name + "_static")
    prepare_all, log_result = orig_prepare, orig_log
    return pcl, static


def _compute_regression(data, lam=1.0, pca_k=0, center=True, device=None, pval_weight=0.0, gene_weights=None, per_cell_gene_weights=None, return_W=False, W_prior=None, W_prior_mu=0.0, quiet=False):
    """Core regression: drug-as-linear-combination with gene-independent weights.
    For each test drug d_y, learns w in R^D_x via ridge regression across training cells,
    then predicts LFC(held_out_cell, d_y) = LFC(held_out_cell, D_x) @ w.
    center=True subtracts per-drug means (separates treatment effect from cell-specific signal).
    gene_weights: optional (n_genes,) tensor, weights genes in normal equations.
    per_cell_gene_weights: optional dict {cell_id: (n_genes,) tensor}, per-held-out-cell gene weights for XTX/XTY.
    return_W: if True, also return a dict {hc: W_matrix} of drug combination weight matrices.
    W_prior: optional dict {hc: W_matrix}, drug combination weights to use as regularization prior.
    W_prior_mu: regularization strength toward W_prior (0 = no prior)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Y_train to GPU once — all indexing happens on GPU
    Y_gpu = torch.from_numpy(np.array(data["Y_train"], dtype=np.float32)).to(device)
    X_gpu = Y_gpu  # features same as targets

    train_cells, train_treats = list(data["train_cells"]), list(data["train_treatments"])
    test_cells, test_treats = list(data["test_cells"]), list(data["test_treatments"])
    n_genes, n_test = Y_gpu.shape[1], data["n_test"]

    # Build lookups: cell -> {treat -> row_idx}, cell -> set of treats
    ct_row, ct_set = {}, {}
    for i, (c, t) in enumerate(zip(train_cells, train_treats)):
        ct_row.setdefault(c, {})[t] = i
        ct_set.setdefault(c, set()).add(t)

    test_map = {}
    for i, (c, t) in enumerate(zip(test_cells, test_treats)):
        test_map.setdefault(c, {})[t] = i

    holdout_cells = sorted(set(test_cells))
    holdout_set = set(holdout_cells)
    non_holdout = sorted(c for c in ct_set if c not in holdout_set)

    Y_pred = np.zeros((n_test, n_genes), dtype=np.float64)
    covered = 0
    W_matrices = {} if return_W else None

    for hc in holdout_cells:
        dx_set = ct_set[hc]  # training drugs for this held-out cell
        dy_set = set(test_map[hc])  # test drugs for this held-out cell

        # Common treatments across hc and all non-holdout cells
        common_dx = dx_set.copy()
        common_dy = dy_set.copy()
        for c in non_holdout:
            common_dx &= ct_set[c]
            common_dy &= ct_set[c]
        D_x, D_y = sorted(common_dx), sorted(common_dy)
        n_dx, n_dy = len(D_x), len(D_y)
        n_nc = len(non_holdout)

        if not quiet: print(f"  {hc}: D_x={n_dx}/{len(dx_set)}, D_y={n_dy}/{len(dy_set)}, cells={n_nc}")

        # GPU-resident indexing: all data stays on GPU
        dx_idx = torch.tensor([ct_row[c][t] for c in non_holdout for t in D_x], device=device, dtype=torch.long)
        dy_idx = torch.tensor([ct_row[c][t] for c in non_holdout for t in D_y], device=device, dtype=torch.long)
        n_feat = X_gpu.shape[1]
        X_all = X_gpu[dx_idx].reshape(n_nc, n_dx, n_feat)
        Y_all = Y_gpu[dy_idx].reshape(n_nc, n_dy, n_genes)

        # Vectorized normal equations (float32 matmul, cast to float64 for solve).
        # No per-row weighting across training cells — see paper §2 Note on
        # cell-similarity weighting.
        X_all_w = X_all
        Y_all_w = Y_all

        # Per-cell expression-aware gene weighting (held-out-cell-specific)
        # Weights only the drug-drug similarity kernel (XTX), not the prediction target (Y).
        if per_cell_gene_weights is not None and hc in per_cell_gene_weights:
            gw = per_cell_gene_weights[hc].to(device)  # (n_genes,)
            gw_sqrt = gw.sqrt().reshape(1, 1, n_genes)
            X_all_w = X_all_w * gw_sqrt
            # Y_all_w intentionally NOT weighted: predictions remain in original gene space

        # Optional: apply per-gene weights to normal equations
        if gene_weights is not None:
            gw_sqrt = gene_weights.sqrt().unsqueeze(0).unsqueeze(0)  # (1, 1, n_genes)
            X_gw = X_all_w[:, :, :n_genes] * gw_sqrt
            Y_gw = Y_all_w * gw_sqrt
            X_flat = X_gw.permute(1, 0, 2).reshape(n_dx, -1)
            XTX = (X_flat @ X_flat.T).double()
            X_fc_flat = X_gw.permute(1, 0, 2).reshape(n_dx, -1)
            Y_flat = Y_gw.permute(1, 0, 2).reshape(n_dy, -1)
            XTY = (X_fc_flat @ Y_flat.T).double()
            del X_gw, Y_gw
        else:
            X_flat = X_all_w.permute(1, 0, 2).reshape(n_dx, -1)
            XTX = (X_flat @ X_flat.T).double()
            X_fc_flat = X_all_w[:, :, :n_genes].permute(1, 0, 2).reshape(n_dx, -1)
            Y_flat = Y_all_w.permute(1, 0, 2).reshape(n_dy, -1)
            XTY = (X_fc_flat @ Y_flat.T).double()

        if center and n_feat == n_genes:
            mu_X = X_all.mean(dim=0).double()
            mu_Y = Y_all.mean(dim=0).double()
            XTX -= n_nc * (mu_X @ mu_X.T)
            XTY -= n_nc * (mu_X @ mu_Y.T)

        del X_all, Y_all, X_all_w, Y_all_w, X_flat, Y_flat, X_fc_flat

        # Optional PCA compression of D_x features
        if pca_k > 0 and pca_k < n_dx:
            evals, evecs = torch.linalg.eigh(XTX)
            V = evecs[:, -pca_k:].flip(1)
            XTX_p = V.T @ XTX @ V
            XTY_p = V.T @ XTY
            W_pca = torch.linalg.solve(XTX_p + lam * torch.eye(pca_k, device=device, dtype=torch.float64), XTY_p)
            W = V @ W_pca
        else:
            # Optional: regularize toward W_prior (FC drug combination weights as NLP prior)
            if W_prior is not None and hc in W_prior and W_prior_mu > 0:
                Wp = torch.from_numpy(W_prior[hc]).to(device).double()
                # W = (XTX + (lam + mu)*I)^{-1} (XTY + mu * Wp)
                W = torch.linalg.solve(
                    XTX + (lam + W_prior_mu) * torch.eye(n_dx, device=device, dtype=torch.float64),
                    XTY + W_prior_mu * Wp
                )
                del Wp
            else:
                W = torch.linalg.solve(XTX + lam * torch.eye(n_dx, device=device, dtype=torch.float64), XTY)

        # Optional reduced-rank regression: truncate W to top-r singular values
        rr_rank = int(os.environ.get("HP_RR_RANK", "75"))
        if rr_rank > 0 and rr_rank < min(W.shape):
            U_w, S_w, Vh_w = torch.linalg.svd(W, full_matrices=False)
            W = U_w[:, :rr_rank] @ torch.diag(S_w[:rr_rank]) @ Vh_w[:rr_rank, :]

        # Store W matrix if requested
        if return_W:
            W_matrices[hc] = W.cpu().numpy()

        # Predict
        hc_idx = torch.tensor([ct_row[hc][t] for t in D_x], device=device, dtype=torch.long)
        X_hc = Y_gpu[hc_idx].double()  # use FC-only features for prediction
        if center:
            Pred_t = mu_Y + W.T @ (X_hc - mu_X)
        else:
            Pred_t = W.T @ X_hc

        Pred = Pred_t.cpu().numpy()

        for j, dy in enumerate(D_y):
            idx = test_map[hc].get(dy)
            if idx is not None:
                Y_pred[idx] = Pred[j]
                covered += 1

        del XTX, XTY, W, X_hc

    del Y_gpu, X_gpu
    torch.cuda.empty_cache()
    print(f"  Covered: {covered}/{n_test} test observations")
    covered_mask = np.zeros(n_test, dtype=bool)
    covered_mask[:] = False
    # Re-mark covered
    for hc in holdout_cells:
        dy_set = set(test_map[hc])
        common_dy = dy_set.copy()
        for c in non_holdout:
            common_dy &= ct_set[c]
        for dy in sorted(common_dy):
            idx = test_map[hc].get(dy)
            if idx is not None:
                covered_mask[idx] = True
    if return_W:
        return Y_pred, covered_mask, W_matrices
    return Y_pred, covered_mask


def train_and_evaluate(experiment_name="full_v12", split_name="tahoe_5_holdout", log=True, data=None, compute_discrimination=False, zeroshot=None):
    """Full model: pipeline + regression blend for FC, ALS + regression + calnet for p-values.

    zeroshot=True swaps R_train/R_test for a control-only reference (see
    prepare_combined.prepare_all). Targets and the test split are unchanged.
    Default None reads HP_ZEROSHOT from the environment (1/true/yes → True)."""
    t_run_start = time.time()
    if zeroshot is None:
        zeroshot = os.environ.get("HP_ZEROSHOT", "0").lower() in ("1", "true", "yes")
    if data is None:
        data = prepare_all(split_name, zeroshot=zeroshot)
    P_train = np.array(data["P_train"], dtype=np.float64)
    Y_train = np.array(data["Y_train"], dtype=np.float64)
    n_test = data["n_test"]
    n_genes = data["Y_train"].shape[1]
    train_cells, train_tr = data["train_cells"], data["train_treatments"]
    test_cells, test_tr = data["test_cells"], data["test_treatments"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    # --- FC prediction: ALS additive + drug regression ---
    os.environ.setdefault("HP_PW_ALPHA", "0.02")
    als_result = _als_decompose(data=data, return_pred=True, log=False)
    als_pred = als_result["full"]
    print(f"[{time.time()-t0:.0f}s] ALS predictions computed")

    # Drug regression. Zeroshot path: ridge on the R/L/C/H feature stack (or a
    # per-(drug, gene) diagonal model). Standard path: drug-as-linear-combination
    # ridge over ALS-imputed Y, with column-mean centering. Both regressions are
    # unweighted across training cells (see paper_methods.md §2 "Note on
    # cell-similarity weighting" for the ablation that motivated removing the
    # earlier Borda rank-weighting machinery).
    if zeroshot:
        # HP_ZS_MODEL selects the regression form: "diagonal" (default —
        # per-(drug, gene) scalar gamma on HVG centroid baseline expression;
        # matches the paper's Rhaister-O equations 8-9) or "ridge" (multi-feature
        # ridge on R/L/C/H stack).
        zs_model = os.environ.get("HP_ZS_MODEL", "diagonal").lower()
        if zs_model == "diagonal":
            reg_pred, reg_mask = _compute_regression_zeroshot_diagonal(data, lam=1.0)
        else:
            reg_pred, reg_mask = _compute_regression_zeroshot(data, lam=1.0, center=True)
        Y_pred_fc = np.where(reg_mask[:, None], reg_pred, als_pred)
        print(f"[{time.time()-t0:.0f}s] FC zeroshot reg applied "
              f"({reg_mask.sum()}/{n_test} covered, {n_test - reg_mask.sum()} ALS fallback)")
    else:
        data_reg = _impute_view_into_data(
            data, data["Y_train"],
            als_result["mu"], als_result["cell_effects"], als_result["treat_effects"],
            als_result["local_cell_map"], als_result["local_treat_map"],
            label="FC regression",
        )
        reg_pred, reg_mask = _compute_regression(data_reg, lam=1.0, center=False)
        Y_pred_fc = np.where(reg_mask[:, None], reg_pred, als_pred)
        print(f"[{time.time()-t0:.0f}s] FC: drug reg applied ({reg_mask.sum()}/{n_test} covered, {n_test - reg_mask.sum()} ALS fallback)")

    # --- P-value ALS model ---
    neglogp_train = -np.log10(np.clip(P_train, 1e-30, 1.0))
    NLP_g = torch.from_numpy(neglogp_train.astype(np.float32)).to(device)

    unique_treatments = sorted(set(train_tr))
    local_treat_map = {t: i for i, t in enumerate(unique_treatments)}
    unique_cells = sorted(set(train_cells))
    local_cell_map = {c: i for i, c in enumerate(unique_cells)}
    train_t_idx = torch.tensor([local_treat_map[t] for t in train_tr], device=device, dtype=torch.long)
    train_c_idx = torch.tensor([local_cell_map[c] for c in train_cells], device=device, dtype=torch.long)
    n_treat, n_cell, n_train = len(unique_treatments), len(unique_cells), NLP_g.shape[0]
    t_exp = train_t_idx.unsqueeze(1).expand(n_train, n_genes)
    c_exp = train_c_idx.unsqueeze(1).expand(n_train, n_genes)

    mu_nlp = NLP_g.mean(0)
    treat_eff_p = torch.zeros(n_treat, n_genes, device=device)
    cell_eff_p = torch.zeros(n_cell, n_genes, device=device)
    tc = torch.bincount(train_t_idx, minlength=n_treat).float().clamp(min=1).unsqueeze(1)
    cc = torch.bincount(train_c_idx, minlength=n_cell).float().clamp(min=1).unsqueeze(1)
    for _ in range(5):
        r = NLP_g - mu_nlp - cell_eff_p[train_c_idx]
        treat_eff_p.zero_(); treat_eff_p.scatter_add_(0, t_exp, r); treat_eff_p /= tc
        r = NLP_g - mu_nlp - treat_eff_p[train_t_idx]
        cell_eff_p.zero_(); cell_eff_p.scatter_add_(0, c_exp, r); cell_eff_p /= cc

    test_t_local = np.array([local_treat_map[t] for t in test_tr])
    held_out_idx = len(unique_cells)  # sentinel for cells absent from train
    test_c_local = np.array([local_cell_map.get(c, held_out_idx) for c in test_cells])

    mu_np = mu_nlp.cpu().numpy()
    te_np = treat_eff_p.cpu().numpy()
    ce_np = cell_eff_p.cpu().numpy()
    ce_np_padded = np.vstack([ce_np, np.zeros((1, n_genes), dtype=ce_np.dtype)])
    NLP_pred = mu_np + te_np[test_t_local] + ce_np_padded[test_c_local]
    NLP_pred = np.clip(NLP_pred, 0, 30)
    P_als = np.power(10.0, -NLP_pred)

    # --- P-value regression (per-gene standardized) ---
    use_preg = os.environ.get("HP_PREG", "1") == "1"
    if use_preg:
        neglogp_train = -np.log10(np.clip(P_train, 1e-30, 1.0)).astype(np.float32)
        preg_std = os.environ.get("HP_PREG_STD", "1") == "1"

        data_preg = _impute_view_into_data(
            data, neglogp_train,
            mu_np, ce_np, te_np,
            local_cell_map, local_treat_map,
            label="P-value regression",
        )
        if preg_std:
            nlp_mean_g = neglogp_train.mean(axis=0)
            nlp_std_g = neglogp_train.std(axis=0).clip(1e-6)
            data_preg["Y_train"] = ((data_preg["Y_train"] - nlp_mean_g) / nlp_std_g).astype(np.float32)
            print(f"  P-value regression: per-gene standardized")
        preg_lam = float(os.environ.get("HP_PREG_LAM", "1.0"))
        orig_rr = os.environ.get("HP_RR_RANK", "75")
        os.environ["HP_RR_RANK"] = os.environ.get("HP_PREG_RR", "0")

        if zeroshot:
            if os.environ.get("HP_ZS_MODEL", "diagonal").lower() == "diagonal":
                preg_pred, preg_mask = _compute_regression_zeroshot_diagonal(
                    data_preg, lam=preg_lam, quiet=True
                )
            else:
                preg_pred, preg_mask = _compute_regression_zeroshot(
                    data_preg, lam=preg_lam, center=True, quiet=True
                )
        else:
            preg_pred, preg_mask = _compute_regression(data_preg, lam=preg_lam, center=True)
        os.environ["HP_RR_RANK"] = orig_rr
        if preg_std:
            preg_pred = preg_pred * nlp_std_g + nlp_mean_g  # de-standardize
        NLP_reg = np.clip(preg_pred, 0, 30)
        P_reg = np.power(10.0, -NLP_reg)
        print(f"[{time.time()-t0:.0f}s] P-value regression done")

        # Use regression where covered, ALS elsewhere
        P_pred = np.where(preg_mask[:, None], P_reg, P_als)
    else:
        P_pred = P_als


    # --- Delta ALS for calnet features (early computation) ---
    delta_als_train = None
    delta_als_test = None
    calnet_delta_feat = os.environ.get("HP_CALNET_DELTA", "1") == "1"
    if calnet_delta_feat and "D_train" in data:
        D_cal = torch.from_numpy(np.array(data["D_train"], dtype=np.float32)).to(device)
        n_train_dc = D_cal.shape[0]
        train_t_dc = torch.tensor([local_treat_map[t] for t in train_tr], device=device, dtype=torch.long)
        train_c_dc = torch.tensor([local_cell_map[c] for c in train_cells], device=device, dtype=torch.long)
        t_exp_dc = train_t_dc.unsqueeze(1).expand(n_train_dc, n_genes)
        c_exp_dc = train_c_dc.unsqueeze(1).expand(n_train_dc, n_genes)
        mu_dc = D_cal.mean(0)
        te_dc = torch.zeros(n_treat, n_genes, device=device)
        ce_dc = torch.zeros(n_cell, n_genes, device=device)
        tc_dc = torch.bincount(train_t_dc, minlength=n_treat).float().clamp(min=1).unsqueeze(1)
        cc_dc = torch.bincount(train_c_dc, minlength=n_cell).float().clamp(min=1).unsqueeze(1)
        for _ in range(5):
            r = D_cal - mu_dc - ce_dc[train_c_dc]
            te_dc.zero_(); te_dc.scatter_add_(0, t_exp_dc, r); te_dc /= tc_dc
            r = D_cal - mu_dc - te_dc[train_t_dc]
            ce_dc.zero_(); ce_dc.scatter_add_(0, c_exp_dc, r); ce_dc /= cc_dc
        delta_als_train = (mu_dc + te_dc[train_t_dc] + ce_dc[train_c_dc]).cpu().numpy().astype(np.float32)
        test_t_dc = torch.tensor([local_treat_map[t] for t in test_tr], device=device, dtype=torch.long)
        # Pad ce_dc with a zero row so held-out cells (sentinel index) get no cell term.
        ce_dc_padded = torch.cat([ce_dc, torch.zeros(1, n_genes, device=device)], dim=0)
        test_c_dc = torch.tensor(
            [local_cell_map.get(c, held_out_idx) for c in test_cells], device=device, dtype=torch.long
        )
        delta_als_test = (mu_dc + te_dc[test_t_dc] + ce_dc_padded[test_c_dc]).cpu().numpy().astype(np.float32)
        del D_cal, te_dc, ce_dc, ce_dc_padded, mu_dc
        torch.cuda.empty_cache()
        print(f"  Delta ALS for calnet: train {delta_als_train.shape}, test {delta_als_test.shape}")

    # --- Neural calibration network for p-values ---
    use_calnet = os.environ.get("HP_CALNET", "1") == "1"
    if use_calnet and "R_train" in data and "R_test" in data:
        R_train_np = np.array(data["R_train"], dtype=np.float32)
        R_test_np = np.array(data["R_test"], dtype=np.float32)

        nlp_actual = -np.log10(np.clip(P_train, 1e-30, 1.0)).astype(np.float32)

        # ALS additive predictions for training (consistent with test)
        train_t_cal = [local_treat_map[t] for t in train_tr]
        train_c_cal = [local_cell_map[c] for c in train_cells]
        NLP_als_tr = (mu_np + te_np[train_t_cal] + ce_np[train_c_cal]).astype(np.float32)

        # FC additive predictions for training
        mu_cal = als_result["mu"]
        te_cal = als_result["treat_effects"]
        ce_cal = als_result["cell_effects"]
        FC_als_train = (mu_cal + te_cal[train_t_cal] + ce_cal[train_c_cal]).astype(np.float32)

        # Per-gene statistics (same for train and test)
        gene_mean_nlp = nlp_actual.mean(axis=0)
        Y_train_f32 = np.array(data["Y_train"], dtype=np.float32)
        gene_fc_std = Y_train_f32.std(axis=0)

        del Y_train_f32

        # ALS predictions for test (held-out cells get no cell term via padded sentinel row).
        test_t_cal = [local_treat_map[t] for t in test_tr]
        test_c_cal = [local_cell_map.get(c, held_out_idx) for c in test_cells]
        ce_cal_padded = np.vstack([ce_cal, np.zeros((1, n_genes), dtype=ce_cal.dtype)])
        NLP_als_te = (mu_np + te_np[test_t_cal] + ce_np_padded[test_c_cal]).astype(np.float32)
        FC_als_test = (mu_cal + te_cal[test_t_cal] + ce_cal_padded[test_c_cal]).astype(np.float32)

        n_tr_cal = len(train_tr)
        cal_device = device
        calnet_h = int(os.environ.get("HP_CALNET_H", "128"))
        calnet_ep = int(os.environ.get("HP_CALNET_EP", "200"))
        calnet_w = float(os.environ.get("HP_CALNET_W", "0.5"))
        calnet_cellmean = os.environ.get("HP_CALNET_CELLMEAN", "1") == "1"
        n_feat_cal = 5
        if calnet_cellmean:
            n_feat_cal += 1
        if calnet_delta_feat and delta_als_train is not None:
            n_feat_cal += 1

        log_ref_train_t = torch.from_numpy(np.log1p(R_train_np)).to(cal_device)
        fc_train_t = torch.from_numpy(np.abs(FC_als_train)).to(cal_device)
        nlp_als_t = torch.from_numpy(NLP_als_tr).to(cal_device)
        nlp_actual_t = torch.from_numpy(nlp_actual).to(cal_device)
        gene_mean_nlp_t = torch.from_numpy(gene_mean_nlp).to(cal_device)
        gene_fc_std_t = torch.from_numpy(gene_fc_std).to(cal_device)
        if calnet_cellmean:
            cell_mean_ref_train = np.log1p(R_train_np).mean(axis=1).astype(np.float32)
            cell_mean_ref_test = np.log1p(R_test_np).mean(axis=1).astype(np.float32)
            cell_mean_ref_tr_t = torch.from_numpy(cell_mean_ref_train).to(cal_device)
            cell_mean_ref_te_t = torch.from_numpy(cell_mean_ref_test).to(cal_device)
        if calnet_delta_feat and delta_als_train is not None:
            delta_train_t = torch.from_numpy(np.abs(delta_als_train)).to(cal_device)
            delta_test_t = torch.from_numpy(np.abs(delta_als_test)).to(cal_device)

        # Pre-stack training features into a single (n_tr_cal * n_genes, n_feat) tensor.
        # Per-batch sampling becomes one fancy index instead of 7 + torch.stack.
        # Memory: 53154 * 1995 * 7 * 4B = ~3GB for the full config; negligible on B200.
        gene_mean_full = gene_mean_nlp_t.unsqueeze(0).expand(n_tr_cal, -1)
        gene_std_full = gene_fc_std_t.unsqueeze(0).expand(n_tr_cal, -1)
        train_feat_list = [fc_train_t, nlp_als_t, log_ref_train_t, gene_mean_full, gene_std_full]
        if calnet_cellmean:
            train_feat_list.append(cell_mean_ref_tr_t.unsqueeze(1).expand(-1, n_genes))
        if calnet_delta_feat and delta_als_train is not None:
            train_feat_list.append(delta_train_t)
        F_train_flat = torch.stack(train_feat_list, dim=2).reshape(n_tr_cal * n_genes, n_feat_cal).contiguous()
        y_train_flat = nlp_actual_t.reshape(-1)

        calnet_ens = int(os.environ.get("HP_CALNET_ENS", "1"))

        batch_size = 262144
        n_batches_per_ep = max(1, (n_tr_cal * n_genes) // batch_size // 10)

        # Build test features once.
        log_ref_test_t = torch.from_numpy(np.log1p(R_test_np)).to(cal_device)
        NLP_current = torch.from_numpy(-np.log10(np.clip(P_pred, 1e-30, 1.0)).astype(np.float32)).to(cal_device)
        fc_test_t = torch.from_numpy(np.abs(Y_pred_fc).astype(np.float32)).to(cal_device)
        te_feats = [
            fc_test_t.reshape(-1),
            NLP_current.reshape(-1),
            log_ref_test_t.reshape(-1),
            gene_mean_nlp_t.unsqueeze(0).expand(n_test, n_genes).reshape(-1),
            gene_fc_std_t.unsqueeze(0).expand(n_test, n_genes).reshape(-1),
        ]
        if calnet_cellmean:
            te_feats.append(cell_mean_ref_te_t.unsqueeze(1).expand(n_test, n_genes).reshape(-1))
        if calnet_delta_feat and delta_als_test is not None:
            te_feats.append(delta_test_t.reshape(-1))
        x_te = torch.stack(te_feats, dim=1)
        del te_feats

        NLP_cal_accum = torch.zeros(n_test, n_genes, device=cal_device)
        for ens_i in range(calnet_ens):
            torch.manual_seed(ens_i * 42 + 7)
            cal_net = torch.nn.Sequential(
                torch.nn.Linear(n_feat_cal, calnet_h), torch.nn.GELU(),
                torch.nn.Linear(calnet_h, calnet_h // 2), torch.nn.GELU(),
                torch.nn.Linear(calnet_h // 2, 1)
            ).to(cal_device)
            cal_net_fwd = torch.compile(cal_net, mode="reduce-overhead", dynamic=False)
            cal_opt = torch.optim.Adam(cal_net.parameters(), lr=1e-3, weight_decay=1e-5)
            cal_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cal_opt, T_max=calnet_ep)

            for ep in range(calnet_ep):
                cal_net.train()
                for _ in range(n_batches_per_ep):
                    obs_idx = torch.randint(0, n_tr_cal, (batch_size,), device=cal_device)
                    gene_idx = torch.randint(0, n_genes, (batch_size,), device=cal_device)
                    flat_idx = obs_idx * n_genes + gene_idx
                    x = F_train_flat[flat_idx]
                    y = y_train_flat[flat_idx].unsqueeze(1)
                    pred = cal_net_fwd(x)
                    loss = ((pred - y) ** 2).mean()
                    cal_opt.zero_grad(); loss.backward(); cal_opt.step()
                cal_sched.step()

            cal_net.eval()
            with torch.no_grad():
                NLP_cal_accum += cal_net(x_te).squeeze(1).reshape(n_test, n_genes)
            del cal_net, cal_net_fwd

        NLP_cal = NLP_cal_accum / calnet_ens
        del NLP_cal_accum

        if calnet_cellmean:
            del cell_mean_ref_tr_t, cell_mean_ref_te_t
        if calnet_delta_feat and delta_als_train is not None:
            del delta_train_t, delta_test_t
        del fc_train_t, nlp_als_t, log_ref_train_t, nlp_actual_t
        del F_train_flat, y_train_flat, gene_mean_full, gene_std_full

        NLP_cal = NLP_cal.clamp(0, 30)
        P_calnet = torch.pow(torch.tensor(10.0), -NLP_cal).cpu().numpy()

        # Blend calibration net output with original P_pred
        sig_orig = 1.0 - P_pred
        sig_cal_blend = 1.0 - P_calnet
        P_pred = 1.0 - ((1 - calnet_w) * sig_orig + calnet_w * sig_cal_blend)
        del NLP_current, fc_test_t, log_ref_test_t, NLP_cal
        del gene_mean_nlp_t, gene_fc_std_t
        torch.cuda.empty_cache()
        print(f"  Calibration net: h={calnet_h}, ep={calnet_ep}, w={calnet_w}, nfeat={n_feat_cal}, ens={calnet_ens}")

    del NLP_g, treat_eff_p, cell_eff_p
    torch.cuda.empty_cache()

    # --- Expression delta prediction (cell_eval deltas) ---
    D_pred = None
    if "D_train" in data:
        D_train = np.array(data["D_train"], dtype=np.float64)
        D_g = torch.from_numpy(D_train.astype(np.float32)).to(device)

        unique_treatments_d = sorted(set(train_tr))
        local_treat_map_d = {t: i for i, t in enumerate(unique_treatments_d)}
        unique_cells_d = sorted(set(train_cells))
        local_cell_map_d = {c: i for i, c in enumerate(unique_cells_d)}
        train_t_idx_d = torch.tensor([local_treat_map_d[t] for t in train_tr], device=device, dtype=torch.long)
        train_c_idx_d = torch.tensor([local_cell_map_d[c] for c in train_cells], device=device, dtype=torch.long)
        n_treat_d, n_cell_d, n_train_d = len(unique_treatments_d), len(unique_cells_d), D_g.shape[0]
        t_exp_d = train_t_idx_d.unsqueeze(1).expand(n_train_d, n_genes)
        c_exp_d = train_c_idx_d.unsqueeze(1).expand(n_train_d, n_genes)

        # ALS: mu + treat_eff + cell_eff
        mu_d = D_g.mean(0)
        treat_eff_d = torch.zeros(n_treat_d, n_genes, device=device)
        cell_eff_d = torch.zeros(n_cell_d, n_genes, device=device)
        tc_d = torch.bincount(train_t_idx_d, minlength=n_treat_d).float().clamp(min=1).unsqueeze(1)
        cc_d = torch.bincount(train_c_idx_d, minlength=n_cell_d).float().clamp(min=1).unsqueeze(1)
        for _ in range(5):
            r = D_g - mu_d - cell_eff_d[train_c_idx_d]
            treat_eff_d.zero_(); treat_eff_d.scatter_add_(0, t_exp_d, r); treat_eff_d /= tc_d
            r = D_g - mu_d - treat_eff_d[train_t_idx_d]
            cell_eff_d.zero_(); cell_eff_d.scatter_add_(0, c_exp_d, r); cell_eff_d /= cc_d

        # Additive prediction for deltas (held-out cells get no cell term).
        held_out_idx_d = len(unique_cells_d)
        test_t_local_d = np.array([local_treat_map_d[t] for t in test_tr])
        test_c_local_d = np.array([local_cell_map_d.get(c, held_out_idx_d) for c in test_cells])
        cell_eff_d_padded = torch.cat([cell_eff_d, torch.zeros(1, n_genes, device=device)], dim=0)
        D_pred_additive = (mu_d.cpu().numpy()
                           + treat_eff_d[test_t_local_d].cpu().numpy()
                           + cell_eff_d_padded[test_c_local_d].cpu().numpy())

        # Delta regression (drug-as-linear-combination)
        delta_blend = float(os.environ.get("HP_DELTA_BLEND", "0.95"))
        delta_center = os.environ.get("HP_DELTA_CENTER", "1") == "1"
        data_delta = _impute_view_into_data(
            data, D_train,
            mu_d.cpu().numpy(), cell_eff_d.cpu().numpy(), treat_eff_d.cpu().numpy(),
            local_cell_map_d, local_treat_map_d,
            label="Delta regression",
        )
        orig_rr = os.environ.get("HP_RR_RANK", "75")
        os.environ["HP_RR_RANK"] = os.environ.get("HP_DELTA_RR", "0")
        if zeroshot:
            if os.environ.get("HP_ZS_MODEL", "diagonal").lower() == "diagonal":
                delta_reg_pred, delta_reg_mask = _compute_regression_zeroshot_diagonal(
                    data_delta, lam=1.0, quiet=True)
            else:
                delta_reg_pred, delta_reg_mask = _compute_regression_zeroshot(
                    data_delta, lam=1.0, center=delta_center, quiet=True)
        else:
            delta_reg_pred, delta_reg_mask = _compute_regression(
                data_delta, lam=1.0, center=delta_center, quiet=True)
        os.environ["HP_RR_RANK"] = orig_rr
        D_pred = np.where(delta_reg_mask[:, None],
                          (1 - delta_blend) * D_pred_additive + delta_blend * delta_reg_pred,
                          D_pred_additive)

        del D_g, treat_eff_d, cell_eff_d
        torch.cuda.empty_cache()
        print(f"[{time.time()-t0:.0f}s] Delta prediction done (blend={delta_blend})")

    # --- Derive FDR from P_pred ---
    F_pred = pvalues_to_fdr_bh(P_pred)

    print(f"[{time.time()-t0:.0f}s] Full model done")

    if os.environ.get("HP_SAVE_PREDICTIONS"):
        _save_predictions_parquet(experiment_name, split_name, Y_pred_fc, D_pred, F_pred, data)

    metrics = data["evaluate_test"](Y_pred_fc, D_pred=D_pred, F_pred=F_pred, P_pred=P_pred, compute_discrimination=compute_discrimination)
    for k, v in sorted(metrics.items()):
        print(f"RESULT {k}={v:.6f}" if isinstance(v, float) else f"RESULT {k}={v}")
    if log:
        log_result(experiment_name, metrics, runtime_seconds=time.time() - t_run_start)

    # HP_SAVE_PREDICTIONS=full: also run in-sample inference on train rows and
    # save to a sibling "_train" parquet. Recurses into train_and_evaluate with
    # train rows swapped into the test position; train data is unchanged, so
    # the fitted model is identical — only the prediction targets differ.
    # Cell-regression blocks are disabled in the train pass because their
    # "non-holdout" cell set is empty when all train cells are in the test
    # position (they'd error trying to index an empty tensor list).
    if (os.environ.get("HP_SAVE_PREDICTIONS") == "full"
            and os.environ.get("_RHAISTER_SAVE_PASS") != "train"):
        print(f"[{time.time()-t_run_start:.0f}s] HP_SAVE_PREDICTIONS=full: starting train-pass inference")
        swapped = dict(data)
        swapped["test_cells"] = np.asarray(data["train_cells"])
        swapped["test_treatments"] = np.asarray(data["train_treatments"])
        swapped["n_test"] = len(swapped["test_cells"])
        swapped["R_test"] = data["R_train"]
        swapped["evaluate_test"] = lambda *a, **k: {}
        os.environ["_RHAISTER_SAVE_PASS"] = "train"
        # Disable calnet in train pass: it forwards one flat (n_test * n_genes,
        # n_feat) tensor, which OOMs at ~53k train rows × 1995 genes. Train-pass
        # fdr_pred falls back to BH on uncalibrated p-values.
        overrides = {"HP_CALNET": "0"}
        saved_env = {v: os.environ.get(v) for v in overrides}
        for v, val in overrides.items():
            os.environ[v] = val
        try:
            train_and_evaluate(experiment_name, split_name, log=False, data=swapped)
        finally:
            os.environ.pop("_RHAISTER_SAVE_PASS", None)
            for v, val in saved_env.items():
                if val is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = val

    return metrics


def _cli():
    """CLI entry point for rhaister-train."""
    name = sys.argv[1] if len(sys.argv) > 1 else "full_v12"
    split = "tahoe_5_holdout"
    for arg in sys.argv[2:]:
        if arg.startswith("--split"):
            split = sys.argv[sys.argv.index(arg) + 1] if "=" not in arg else arg.split("=")[1]
    zeroshot = "--zeroshot" in sys.argv

    if "--both" in sys.argv:
        run_both(name)
    else:
        train_and_evaluate(name, split, zeroshot=zeroshot if zeroshot else None)


if __name__ == "__main__":
    _cli()
