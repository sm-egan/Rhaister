"""train_sensitivity.py — Rhaister-style trained model for scalar growth-rate datasets.

Adapts the FC pipeline from `train.py` to scalar targets (n_genes=1):

    ALS additive  →  unweighted drug regression  →  ALS fallback

with an optional transcriptomic-feature drug ridge blended in when HP_FEATURES
is set (e.g. HP_FEATURES="pdex,pdex_pv,pdex_fdr,cell_eval").

Mirrors the diagram in `model_diagram.drawio`.

Usage:
    uv run python train_sensitivity.py rhaister_v1 --split EmeraldBay/split_0
    HP_FEATURES="pdex,pdex_pv,pdex_fdr,cell_eval" \\
        uv run python train_sensitivity.py rhaister_v1_feat_all --split EmeraldBay/split_0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prepare_sensitivity


# ---------- Stage 1: ALS additive ----------------------------------------------------

def _als_decompose_scalar(y, c_idx, t_idx, n_cell, n_treat, n_iter):
    """Vectorized scalar ALS: y ≈ mu + cell_eff[c] + treat_eff[t].

    Lifted from scripts/baseline_sensitivity._als_decomposition. O(n_train) per pass.
    """
    mu = float(np.mean(y))
    cell_eff = np.zeros(n_cell)
    treat_eff = np.zeros(n_treat)
    cell_count = np.bincount(c_idx, minlength=n_cell).astype(np.float64)
    treat_count = np.bincount(t_idx, minlength=n_treat).astype(np.float64)
    safe_cell = np.maximum(cell_count, 1.0)
    safe_treat = np.maximum(treat_count, 1.0)
    for _ in range(n_iter):
        resid = y - mu - cell_eff[c_idx]
        treat_sum = np.bincount(t_idx, weights=resid, minlength=n_treat)
        treat_eff = np.where(treat_count > 0, treat_sum / safe_treat, 0.0)
        resid = y - mu - treat_eff[t_idx]
        cell_sum = np.bincount(c_idx, weights=resid, minlength=n_cell)
        cell_eff = np.where(cell_count > 0, cell_sum / safe_cell, 0.0)
    return mu, cell_eff, treat_eff


# ---------- Stage 2: drug regression -------------------------------------------------

def _drug_regression(y_imp, obs_mask, test_pairs, lam, holdout_set):
    """Per-drug ridge over training cells (unweighted), n_genes=1.

    Uses dense (n_cell, n_treat) slicing on `y_imp` (real where observed,
    ALS-imputed otherwise) and the Woodbury form (n_nc × n_nc) so it scales to
    PRISM (n_dx ~ 8400, n_nc ~ 432). The strict "drug present in EVERY non-holdout
    cell" filter from train.py would drop almost everything at PRISM's 93%
    coverage; instead we let the ALS imputation back-fill missing entries.

    For each held-out cell hc:
      - D_x = treats hc has observed in train (no cross-cell intersection)
      - D_y = treats hc must predict (the test treats)
      - Stack X_all = y_imp[non_holdout, D_x] (n_nc, n_dx) and similarly Y_all
      - Center with column means
      - Solve B = (X̃ X̃ᵀ + λI)⁻¹ Ỹ, predict y = μ_Y + Bᵀ (X̃ (x_hc − μ_X))
    """
    n_cell, _ = y_imp.shape
    has_any_train = obs_mask.any(axis=1)
    non_holdout = np.array(
        [c for c in range(n_cell) if has_any_train[c] and c not in holdout_set],
        dtype=np.int64,
    )
    n_nc = len(non_holdout)
    if n_nc == 0:
        return np.zeros(len(test_pairs)), np.zeros(len(test_pairs), dtype=bool)

    test_by_hc: dict[int, list[tuple[int, int]]] = {}
    for i, (c, t) in enumerate(test_pairs):
        test_by_hc.setdefault(int(c), []).append((i, int(t)))

    n_test = len(test_pairs)
    y_pred = np.zeros(n_test, dtype=np.float64)
    covered = np.zeros(n_test, dtype=bool)

    for hc, items in test_by_hc.items():
        D_x = np.where(obs_mask[hc])[0]
        if D_x.size == 0:
            continue

        D_y = np.array(sorted({t for _, t in items}), dtype=np.int64)
        if D_y.size == 0:
            continue

        X_all = y_imp[np.ix_(non_holdout, D_x)]  # (n_nc, n_dx)
        Y_all = y_imp[np.ix_(non_holdout, D_y)]  # (n_nc, n_dy)

        mu_X = X_all.mean(axis=0)            # (n_dx,)
        mu_Y = Y_all.mean(axis=0)            # (n_dy,)
        X_c = X_all - mu_X                   # (n_nc, n_dx)
        Y_c = Y_all - mu_Y                   # (n_nc, n_dy)

        # Woodbury form: solve in n_nc × n_nc instead of n_dx × n_dx.
        A = X_c @ X_c.T                       # (n_nc, n_nc)
        B = np.linalg.solve(A + lam * np.eye(n_nc), Y_c)  # (n_nc, n_dy)

        x_hc = y_imp[hc, D_x] - mu_X         # (n_dx,)
        pred = mu_Y + B.T @ (X_c @ x_hc)     # (n_dy,)

        D_y_pos = {int(dy): j for j, dy in enumerate(D_y)}
        for test_i, t in items:
            j = D_y_pos.get(t)
            if j is not None:
                y_pred[test_i] = pred[j]
                covered[test_i] = True

    return y_pred, covered


# ---------- Stage 2b: feature drug regression ----------------------------------------

def _drug_regression_features(feat_mat, y_imp, obs_mask, test_pairs, lam, holdout_set):
    """Feature sibling of _drug_regression (unweighted): same per-drug ridge, but
    the predictors are per-drug transcriptomic feature vectors (pdex / cell_eval)
    instead of scalar growth_rates. The target stays scalar y_imp at D_y.

    For each held-out cell hc:
      - D_x   = treats hc has observed in train
      - D_y   = treats hc must predict
      - X_all = feat_mat[non_holdout, D_x, :] flattened to (n_nc, n_dx · n_feat)
      - Y_all = y_imp[non_holdout, D_y]                          (n_nc, n_dy)
      - Solve B = (X̃ X̃ᵀ + λI)⁻¹ Ỹ (Woodbury, n_nc × n_nc); predict from
        x_hc = feat_mat[hc, D_x] flattened. λ is larger than the scalar ridge's
        because the feature matrix is far wider (n_dx · n_feat vs n_dx).
    """
    n_cell, _ = y_imp.shape
    n_features = feat_mat.shape[-1]
    has_any_train = obs_mask.any(axis=1)
    non_holdout = np.array(
        [c for c in range(n_cell) if has_any_train[c] and c not in holdout_set],
        dtype=np.int64,
    )
    n_nc = len(non_holdout)
    if n_nc == 0:
        return np.zeros(len(test_pairs)), np.zeros(len(test_pairs), dtype=bool)

    test_by_hc: dict[int, list[tuple[int, int]]] = {}
    for i, (c, t) in enumerate(test_pairs):
        test_by_hc.setdefault(int(c), []).append((i, int(t)))

    n_test = len(test_pairs)
    y_pred = np.zeros(n_test, dtype=np.float64)
    covered = np.zeros(n_test, dtype=bool)

    for hc, items in test_by_hc.items():
        D_x = np.where(obs_mask[hc])[0]
        if D_x.size == 0:
            continue

        D_y = np.array(sorted({t for _, t in items}), dtype=np.int64)
        if D_y.size == 0:
            continue

        n_dx = D_x.size
        # Flatten the (n_nc, n_dx, n_feat) slice to (n_nc, n_dx · n_feat).
        X_all = feat_mat[np.ix_(non_holdout, D_x)].reshape(n_nc, n_dx * n_features)
        Y_all = y_imp[np.ix_(non_holdout, D_y)]  # (n_nc, n_dy)

        mu_X = X_all.mean(axis=0)
        mu_Y = Y_all.mean(axis=0)
        X_c = X_all - mu_X
        Y_c = Y_all - mu_Y

        # Woodbury form: solve in n_nc × n_nc instead of (n_dx · n_feat)².
        A = X_c @ X_c.T
        B = np.linalg.solve(A + lam * np.eye(n_nc), Y_c)

        x_hc = feat_mat[hc, D_x].reshape(-1) - mu_X
        pred = mu_Y + B.T @ (X_c @ x_hc)

        D_y_pos = {int(dy): j for j, dy in enumerate(D_y)}
        for test_i, t in items:
            j = D_y_pos.get(t)
            if j is not None:
                y_pred[test_i] = pred[j]
                covered[test_i] = True

    return y_pred, covered


# ---------- Stage 2c: MLP refinement (feature variant only) --------------------------

def _mlp_residual(y_resid_train, c_train, t_train, c_test, t_test,
                  n_cell, n_treat, emb_dim, hidden, epochs, lr, weight_decay,
                  batch_size, device, seed=0,
                  feats_train=None, feats_test=None):  # noqa: PLR0913
    """Train cell/treatment embeddings + MLP on the additive residual; return the
    MLP residual prediction on test.

    feats_train / feats_test (optional, both-or-neither) attach per-row feature
    vectors that the MLP projects to emb_dim and concatenates with the cell +
    treatment embeddings. Only used by the feature variant — torch is imported
    lazily here so the base model and the PRISM baselines never pay the cost.
    """
    import torch

    class _ResidualMLP(torch.nn.Module):
        def __init__(self, n_cell, n_treat, emb_dim, hidden, feat_dim=0):
            super().__init__()
            self.cell_emb = torch.nn.Embedding(n_cell, emb_dim)
            self.treat_emb = torch.nn.Embedding(n_treat, emb_dim)
            torch.nn.init.normal_(self.cell_emb.weight, std=0.1)
            torch.nn.init.normal_(self.treat_emb.weight, std=0.1)
            in_dim = emb_dim * 2
            if feat_dim > 0:
                self.feat_proj = torch.nn.Linear(feat_dim, emb_dim)
                in_dim += emb_dim
            else:
                self.feat_proj = None
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden), torch.nn.GELU(),
                torch.nn.BatchNorm1d(hidden), torch.nn.Dropout(0.1),
                torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
                torch.nn.Dropout(0.05),
                torch.nn.Linear(hidden // 2, 1),
            )

        def forward(self, c_idx, t_idx, feats=None):
            parts = [self.cell_emb(c_idx), self.treat_emb(t_idx)]
            if self.feat_proj is not None:
                assert feats is not None, "MLP built with feat_dim>0 but no feats provided"
                parts.append(self.feat_proj(feats))
            return self.net(torch.cat(parts, dim=1)).squeeze(-1)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    feat_dim = feats_train.shape[1] if feats_train is not None else 0
    net = _ResidualMLP(n_cell, n_treat, emb_dim, hidden, feat_dim=feat_dim).to(device)
    c_t = torch.from_numpy(c_train.astype(np.int64)).to(device)
    t_t = torch.from_numpy(t_train.astype(np.int64)).to(device)
    y_t = torch.from_numpy(y_resid_train.astype(np.float32)).to(device)
    f_t = torch.from_numpy(feats_train.astype(np.float32)).to(device) if feats_train is not None else None

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_tr = len(c_t)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            f_idx = f_t[idx] if f_t is not None else None
            pred = net(c_t[idx], t_t[idx], f_idx)
            loss = ((pred - y_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    with torch.no_grad():
        c_te = torch.from_numpy(c_test.astype(np.int64)).to(device)
        t_te = torch.from_numpy(t_test.astype(np.int64)).to(device)
        f_te = torch.from_numpy(feats_test.astype(np.float32)).to(device) if feats_test is not None else None
        y_pred = net(c_te, t_te, f_te).cpu().numpy().astype(np.float64)
    return y_pred


# ---------- Public entry point -------------------------------------------------------

def train_and_evaluate_sensitivity(experiment_name, split_name="EmeraldBay/split_0",
                                    log=True, data=None, device=None):
    """Rhaister-style scalar sensitivity model: ALS additive → unweighted drug
    ridge → ALS fallback. When HP_FEATURES is set (the 'with features' variant),
    a transcriptomic-feature drug ridge and an MLP refinement are blended in.
    Cell-similarity weighting stays retired (no-cellsim is canonical)."""
    t0 = time.time()
    HP_FEATURES = os.environ.get("HP_FEATURES", "") or None  # "" / unset → no features
    if data is None:
        data = prepare_sensitivity.prepare_all(split_name, with_features=HP_FEATURES)

    HP_ALS = int(os.environ.get("HP_ALS", "30"))
    HP_DREG_LAM = float(os.environ.get("HP_DREG_LAM", "1.0"))
    # Feature drug-ridge blend: 0 = scalar drug ridge only, 1 = feature ridge
    # only. LAM is larger than the scalar ridge's since the feature matrix is
    # much wider (n_dx · n_feat vs n_dx).
    HP_FEAT_RIDGE_W = float(os.environ.get("HP_FEAT_RIDGE_W", "0.25"))
    HP_FEAT_RIDGE_LAM = float(os.environ.get("HP_FEAT_RIDGE_LAM", "10.0"))
    # MLP refinement on the additive residual (feature variant only); blended at
    # HP_MLP_W. Defaults match the prior EmeraldBay tuning.
    HP_MLP_W = float(os.environ.get("HP_MLP_W", "0.12"))
    HP_MLP_EMB = int(os.environ.get("HP_MLP_EMB", "32"))
    HP_MLP_H = int(os.environ.get("HP_MLP_H", "128"))
    HP_MLP_EP = int(os.environ.get("HP_MLP_EP", "200"))
    HP_MLP_LR = float(os.environ.get("HP_MLP_LR", "1e-3"))
    HP_MLP_WD = float(os.environ.get("HP_MLP_WD", "5e-4"))
    HP_MLP_BS = int(os.environ.get("HP_MLP_BS", "512"))
    HP_SEED = int(os.environ.get("HP_SEED", "0"))

    cell_to_idx = data["cell_to_idx"]
    treat_to_idx = data["treat_to_idx"]
    n_cell = data["n_cells"]
    n_treat = data["n_treatments"]

    c_train = np.array([cell_to_idx[c] for c in data["train_cells"]], dtype=np.int64)
    t_train = np.array([treat_to_idx[t] for t in data["train_treatments"]], dtype=np.int64)
    c_test = np.array([cell_to_idx[c] for c in data["test_cells"]], dtype=np.int64)
    t_test = np.array([treat_to_idx[t] for t in data["test_treatments"]], dtype=np.int64)
    y_train = np.asarray(data["y_train"], dtype=np.float64)
    y_test = np.asarray(data["y_test"], dtype=np.float64)
    n_train, n_test = len(y_train), len(y_test)

    # Feature variant (HP_FEATURES set) adds a transcriptomic-feature drug ridge
    # and an MLP refinement on top of the 2-stage scalar model.
    feat_mat = data.get("feat_mat")
    has_features = feat_mat is not None
    use_feat_ridge = has_features and HP_FEAT_RIDGE_W > 0
    use_mlp = has_features and HP_MLP_W > 0
    n_stages = 2 + int(use_feat_ridge) + int(use_mlp)

    print(f"[1/{n_stages}] ALS additive ({HP_ALS} iters)")
    mu, cell_eff, treat_eff = _als_decompose_scalar(
        y_train, c_train, t_train, n_cell, n_treat, n_iter=HP_ALS,
    )
    print(f"      mu={mu:+.4f}  "
          f"cell_eff∈[{cell_eff.min():+.3f},{cell_eff.max():+.3f}]  "
          f"treat_eff∈[{treat_eff.min():+.3f},{treat_eff.max():+.3f}]")
    additive_train = mu + cell_eff[c_train] + treat_eff[t_train]
    additive_test = mu + cell_eff[c_test] + treat_eff[t_test]

    # ALS-imputed dense matrix used by drug regression to handle PRISM-style
    # sparse coverage.
    obs_mask = np.zeros((n_cell, n_treat), dtype=bool)
    obs_mask[c_train, t_train] = True
    y_imp = mu + cell_eff[:, None] + treat_eff[None, :]
    y_imp[c_train, t_train] = y_train
    holdout_set = {int(c) for c in c_test}

    print(f"[2/{n_stages}] Drug regression (lam={HP_DREG_LAM})")
    test_pairs = list(zip(c_test.tolist(), t_test.tolist()))
    y_drug_reg, covered = _drug_regression(
        y_imp, obs_mask, test_pairs, lam=HP_DREG_LAM,
        holdout_set=holdout_set,
    )
    n_covered = int(covered.sum())
    print(f"      covered {n_covered}/{n_test} test entries; falling back to ALS for the rest")
    y_pred = np.where(covered, y_drug_reg, additive_test)

    # Feature drug regression: a parallel unweighted ridge over the per-drug
    # transcriptomic feature vectors loaded by attach_features, blended into the
    # scalar drug-regression prediction with weight HP_FEAT_RIDGE_W. (Cell-
    # similarity weighting stays retired — no-cellsim is canonical.)
    if use_feat_ridge:
        print(f"[3/{n_stages}] Feature drug regression "
              f"(lam={HP_FEAT_RIDGE_LAM}, blend_w={HP_FEAT_RIDGE_W}, "
              f"source={data.get('feature_source')})")
        y_drug_reg_feat, covered_feat = _drug_regression_features(
            feat_mat, y_imp, obs_mask, test_pairs,
            lam=HP_FEAT_RIDGE_LAM, holdout_set=holdout_set,
        )
        y_pred_feat = np.where(covered_feat, y_drug_reg_feat, additive_test)
        print(f"      covered {int(covered_feat.sum())}/{n_test} test entries")
        y_pred = (1.0 - HP_FEAT_RIDGE_W) * y_pred + HP_FEAT_RIDGE_W * y_pred_feat

    # MLP refinement (feature variant only): learn the additive residual from
    # cell + treatment embeddings concatenated with projected feature vectors,
    # then blend at HP_MLP_W.
    if use_mlp:
        import torch
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feats_train = data.get("X_train_features")
        feats_test = data.get("X_test_features")
        feat_dim = feats_train.shape[1] if feats_train is not None else 0
        print(f"[{n_stages}/{n_stages}] MLP refinement "
              f"(emb={HP_MLP_EMB}, hidden={HP_MLP_H}, epochs={HP_MLP_EP}, "
              f"blend_w={HP_MLP_W}, feats={feat_dim}d, device={device})")
        resid_train = y_train - additive_train
        mlp_resid_test = _mlp_residual(
            resid_train, c_train, t_train, c_test, t_test,
            n_cell=n_cell, n_treat=n_treat,
            emb_dim=HP_MLP_EMB, hidden=HP_MLP_H, epochs=HP_MLP_EP,
            lr=HP_MLP_LR, weight_decay=HP_MLP_WD, batch_size=HP_MLP_BS,
            device=device, seed=HP_SEED,
            feats_train=feats_train, feats_test=feats_test,
        )
        y_mlp = additive_test + mlp_resid_test
        y_pred = (1.0 - HP_MLP_W) * y_pred + HP_MLP_W * y_mlp

    metrics = data["evaluate_test"](y_pred)
    runtime = time.time() - t0

    print()
    print(f"Split: {split_name}  |  n_train={n_train}  n_test={n_test}  runtime={runtime:.2f}s")
    print(f"y_test  mean={y_test.mean():+.4f}  std={y_test.std():.4f}")
    print()
    print(f"  MSE      = {metrics['sensitivity/mse']:.4f}")
    print(f"  MAE      = {metrics['sensitivity/mae']:.4f}")
    print(f"  R^2      = {metrics['sensitivity/r2']:.4f}")
    print(f"  Pearson  = {metrics['sensitivity/pearson']:.4f}")

    if log:
        record = {
            "baseline": experiment_name,
            "split": split_name,
            "mse": metrics["sensitivity/mse"],
            "mae": metrics["sensitivity/mae"],
            "r2": metrics["sensitivity/r2"],
            "pearson": metrics["sensitivity/pearson"],
            "n_train": int(n_train),
            "n_test": int(n_test),
            "runtime_seconds": round(runtime, 3),
        }
        prepare_sensitivity.log_result(record)
        print(f"\nLogged to {prepare_sensitivity.RESULTS_FILE}")

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", help="Experiment name (e.g. rhaister_v1)")
    p.add_argument("--split", default="EmeraldBay/split_0")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()
    train_and_evaluate_sensitivity(
        experiment_name=args.experiment,
        split_name=args.split,
        log=not args.no_log,
    )


if __name__ == "__main__":
    main()
