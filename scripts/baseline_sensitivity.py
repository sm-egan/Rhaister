"""Baselines for scalar-target sensitivity datasets (RIFIVDU and future
drug-sensitivity datasets of the same modality).

Baselines (all scalar targets):
  global_mean      — mu (overall train mean)
  cell_mean        — mu + cell_eff[c]
  treatment_mean   — mu + treat_eff[t]
  additive         — mu + cell_eff[c] + treat_eff[t]   (ALS)
  svd_residual     — additive + low-rank SVD completion of (cell × treatment)
                     residual matrix on the train mask

Metrics: MSE, MAE (declared in splits/<dataset>/dataset.toml [metrics].applicable).

Usage:
    uv run python scripts/baseline_sensitivity.py --split EmeraldBay/split_0
"""
import argparse
import os
import sys
import time

import numpy as np


from rhaister import prepare_sensitivity


BASELINE_NAMES = ["global_mean", "cell_mean", "treatment_mean", "additive", "svd_residual"]
A_VS_B_BASELINE = "a_vs_b"
PRIMARY_BASELINE = "primary_to_secondary"
ADJACENT_BASELINE = "adjacent_dose"


_CONDITION_RE = __import__("re").compile(
    r"^\[\('(.+)',\s*([\-\d\.eE\+]+),\s*'([^']+)'\)\]$"
)


def _parse_condition(s):
    """Return (drug, dose) for a single-drug condition string, else (None, None)."""
    m = _CONDITION_RE.match(s)
    if m is None:
        return None, None
    return m.group(1), float(m.group(2))


def _als_decomposition(y, c_idx, t_idx, n_cell, n_treat, n_iter=30):
    """Scalar ALS: y ~= mu + cell_eff[c] + treat_eff[t].

    Vectorized via np.bincount — O(n_train) per pass instead of O(n_cell+n_treat)
    Python-level loops, which matters at prism scale (n_treat ~ 12K).
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


def _svd_completion(R_obs, mask, rank, n_iter):
    """Iterative truncated-SVD completion of a 2-D matrix.

    R_obs: (n_cell, n_treat) residuals at observed positions, 0 elsewhere.
    mask:  bool (n_cell, n_treat), True at observed positions.

    Uses sklearn's randomized_svd for the truncated factorization — at prism
    scale (481 x 12392) this is far faster than np.linalg.svd's full SVD.
    """
    from sklearn.utils.extmath import randomized_svd
    R = R_obs.copy()
    k = min(rank, *R.shape)
    rng = np.random.default_rng(0)
    for _ in range(n_iter):
        U, S, Vt = randomized_svd(R, n_components=k, random_state=rng.integers(2**31))
        R_recon = (U * S) @ Vt
        R = np.where(mask, R_obs, R_recon)
    return R


def _a_vs_b_pairs(dataset, test_cells, test_treatments):
    """Look up A and B growth_rate at each test (cell, cond) pair.

    Returns (a_vals, b_vals, valid_mask) — float arrays plus a bool mask of
    pairs where BOTH A and B have a non-null aggregated value.
    """
    A = prepare_sensitivity.load_variant_data(dataset, "A").set_index(["cell_line", "condition"])["growth_rate"]
    B = prepare_sensitivity.load_variant_data(dataset, "B").set_index(["cell_line", "condition"])["growth_rate"]
    keys = list(zip(test_cells, test_treatments))
    a_vals = np.array([A.get(k, np.nan) for k in keys], dtype=np.float64)
    b_vals = np.array([B.get(k, np.nan) for k in keys], dtype=np.float64)
    valid = np.isfinite(a_vals) & np.isfinite(b_vals)
    return a_vals, b_vals, valid


def _primary_to_secondary_pred(dataset, test_cells, test_treatments, y_test):
    """Predict secondary growth_rate from the primary-screen measurement at the
    matched (cell, drug, dose). Only ~12% of test rows have a matching dose —
    primary parquet's dose_secondary covers 121 of the 939 secondary doses, all
    clustered near the primary's 2.5 μM dose. Returns (pred, y_truth,
    valid_mask) where valid_mask flags rows that have a match.
    """
    primary = prepare_sensitivity.load_primary_screen(dataset)
    pred = np.full(len(test_cells), np.nan, dtype=np.float64)
    for i, (cell, cond) in enumerate(zip(test_cells, test_treatments)):
        drug, dose = _parse_condition(cond)
        if drug is None:
            continue
        v = primary.get((cell, drug, dose))
        if v is not None:
            pred[i] = v
    valid = np.isfinite(pred)
    return pred, np.asarray(y_test, dtype=np.float64), valid


def _adjacent_dose_pred(test_cells, test_treatments, y_test):
    """Predict each test row's growth_rate from the nearest other dose of the
    SAME (cell, drug), measured in log-dose space, using that neighbour's
    growth_rate verbatim.

    Held-out splits hold out whole drugs within held-out cells, so the entire
    dose-response curve of a held-out (cell, drug) lands in the test set
    (median 8 doses, up to 16). This is therefore an oracle / titration noise
    ceiling — like the a_vs_b ceiling it reads held-out labels — answering "if
    you already knew the adjacent dose on this curve, how well would copying it
    do?". valid_mask flags rows that have at least one same-(cell, drug)
    neighbour (essentially all of them; singletons have none).

    Returns (pred, y_truth, valid_mask).
    """
    y_test = np.asarray(y_test, dtype=np.float64)
    pred = np.full(len(test_cells), np.nan, dtype=np.float64)

    # Group row indices by (cell, drug); doses on the same curve are neighbours.
    groups = {}
    for i, (cell, cond) in enumerate(zip(test_cells, test_treatments)):
        drug, dose = _parse_condition(cond)
        if drug is None or not np.isfinite(y_test[i]):
            continue
        groups.setdefault((cell, drug), []).append((dose, i))

    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda t: t[0])
        doses = np.array([d for d, _ in members], dtype=np.float64)
        idxs = [i for _, i in members]
        log_doses = np.log(doses)  # secondary doses are strictly positive
        for j in range(len(members)):
            # nearest neighbour in log-dose, excluding self
            gaps = np.abs(log_doses - log_doses[j])
            gaps[j] = np.inf
            nn = int(np.argmin(gaps))
            pred[idxs[j]] = y_test[idxs[nn]]

    valid = np.isfinite(pred)
    return pred, y_test, valid


def compute_baselines(split_name, rank, iters):
    data = prepare_sensitivity.prepare_all(split_name)
    cell_to_idx = data["cell_to_idx"]
    treat_to_idx = data["treat_to_idx"]
    n_cell = data["n_cells"]
    n_treat = data["n_treatments"]

    train_c = np.array([cell_to_idx[c] for c in data["train_cells"]])
    train_t = np.array([treat_to_idx[t] for t in data["train_treatments"]])
    test_c = np.array([cell_to_idx[c] for c in data["test_cells"]])
    test_t = np.array([treat_to_idx[t] for t in data["test_treatments"]])

    y_train = data["y_train"]
    y_test = data["y_test"]

    mu, cell_eff, treat_eff = _als_decomposition(
        y_train, train_c, train_t, n_cell, n_treat, n_iter=iters
    )

    additive_test = mu + cell_eff[test_c] + treat_eff[test_t]

    # Residual matrix on train mask
    R_obs = np.zeros((n_cell, n_treat), dtype=np.float64)
    mask = np.zeros((n_cell, n_treat), dtype=bool)
    additive_train = mu + cell_eff[train_c] + treat_eff[train_t]
    R_obs[train_c, train_t] = y_train - additive_train
    mask[train_c, train_t] = True
    R_completed = _svd_completion(R_obs, mask, rank=rank, n_iter=iters)
    svd_test = additive_test + R_completed[test_c, test_t]

    preds = {
        "global_mean": np.full_like(y_test, mu),
        "cell_mean": mu + cell_eff[test_c],
        "treatment_mean": mu + treat_eff[test_t],
        "additive": additive_test,
        "svd_residual": svd_test,
    }
    return preds, y_test, data


def _metrics(y_pred, y_true):
    return prepare_sensitivity.compute_metrics(y_pred, y_true)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="EmeraldBay/split_0")
    p.add_argument("--rank", type=int, default=int(os.environ.get("HP_SVD_RANK", "5")))
    p.add_argument("--iters", type=int, default=int(os.environ.get("HP_SVD_ITERS", "30")))
    p.add_argument("--wandb", action="store_true", help="Opt in to WandB logging")
    args = p.parse_args()

    t0 = time.time()
    preds, y_test, data = compute_baselines(args.split, rank=args.rank, iters=args.iters)
    n_train = data["y_train"].size
    n_test = y_test.size

    def _row(name, m, n_test_used):
        return {
            "baseline": name,
            "split": args.split,
            "mse": m["sensitivity/mse"],
            "mae": m["sensitivity/mae"],
            "r2": m["sensitivity/r2"],
            "pearson": m["sensitivity/pearson"],
            "n_train": int(n_train),
            "n_test": int(n_test_used),
            "rank": args.rank,
            "iters": args.iters,
        }

    print()
    print(f"Split: {args.split}  |  n_train={n_train}  n_test={n_test}  rank={args.rank}  iters={args.iters}")
    print(f"y_test  mean={float(y_test.mean()):+.4f}  std={float(y_test.std()):.4f}")
    print()
    print(f"{'baseline':<16}  {'MSE':>10}  {'MAE':>10}  {'R^2':>10}  {'Pearson':>10}")
    print("-" * 64)
    rows = []
    for name in BASELINE_NAMES:
        m = _metrics(preds[name], y_test)
        print(f"{name:<16}  {m['sensitivity/mse']:>10.4f}  {m['sensitivity/mae']:>10.4f}  "
              f"{m['sensitivity/r2']:>10.4f}  {m['sensitivity/pearson']:>10.4f}")
        rows.append(_row(name, m, n_test))

    # A-vs-B noise ceiling — only when the dataset declares A/B variants.
    dataset, _ = prepare_sensitivity.parse_split_name(args.split)
    if prepare_sensitivity.has_variants(dataset):
        a_vals, b_vals, valid = _a_vs_b_pairs(dataset, data["test_cells"], data["test_treatments"])
        n_pairs = int(valid.sum())
        if n_pairs == 0:
            print(f"{A_VS_B_BASELINE:<16}  (no valid A/B pairs on test set)")
        else:
            m = _metrics(a_vals[valid], b_vals[valid])
            print(f"{A_VS_B_BASELINE:<16}  {m['sensitivity/mse']:>10.4f}  {m['sensitivity/mae']:>10.4f}  "
                  f"{m['sensitivity/r2']:>10.4f}  {m['sensitivity/pearson']:>10.4f}  "
                  f"(n_pairs={n_pairs}/{n_test})")
            rows.append(_row(A_VS_B_BASELINE, m, n_pairs))

    # Primary-screen reference — only when the dataset declares an external primary screen.
    if prepare_sensitivity.has_primary_screen(dataset):
        p_pred, y_truth, valid = _primary_to_secondary_pred(
            dataset, data["test_cells"], data["test_treatments"], y_test,
        )
        n_pairs = int(valid.sum())
        if n_pairs == 0:
            print(f"{PRIMARY_BASELINE:<16}  (no valid primary matches on test set)")
        else:
            m = _metrics(p_pred[valid], y_truth[valid])
            print(f"{PRIMARY_BASELINE:<16}  {m['sensitivity/mse']:>10.4f}  {m['sensitivity/mae']:>10.4f}  "
                  f"{m['sensitivity/r2']:>10.4f}  {m['sensitivity/pearson']:>10.4f}  "
                  f"(n_pairs={n_pairs}/{n_test})")
            rows.append(_row(PRIMARY_BASELINE, m, n_pairs))

    # Adjacent-dose ceiling — copy the nearest dose on the same (cell, drug)
    # curve. Self-skips when no (cell, drug) has >=2 test doses.
    a_pred, y_truth, valid = _adjacent_dose_pred(
        data["test_cells"], data["test_treatments"], y_test,
    )
    n_pairs = int(valid.sum())
    if n_pairs == 0:
        print(f"{ADJACENT_BASELINE:<16}  (no multi-dose (cell, drug) groups on test set)")
    else:
        m = _metrics(a_pred[valid], y_truth[valid])
        print(f"{ADJACENT_BASELINE:<16}  {m['sensitivity/mse']:>10.4f}  {m['sensitivity/mae']:>10.4f}  "
              f"{m['sensitivity/r2']:>10.4f}  {m['sensitivity/pearson']:>10.4f}  "
              f"(n_pairs={n_pairs}/{n_test})")
        rows.append(_row(ADJACENT_BASELINE, m, n_pairs))

    runtime = time.time() - t0
    for r in rows:
        r["runtime_seconds"] = round(runtime, 3)
        prepare_sensitivity.log_result(r)
    print()
    print(f"Wrote {len(rows)} rows to {prepare_sensitivity.RESULTS_FILE}  (runtime {runtime:.2f}s)")

    if args.wandb:
        import wandb
        for r in rows:
            wandb.init(
                project="rhaister-sensitivity",
                name=f"{r['baseline']}_{args.split.replace('/', '_')}",
                config={k: v for k, v in r.items() if k != "baseline"},
                reinit=True,
            )
            wandb.log({
                "sensitivity/mse": r["mse"],
                "sensitivity/mae": r["mae"],
                "sensitivity/r2": r["r2"],
                "sensitivity/pearson": r["pearson"],
            })
            wandb.finish()


if __name__ == "__main__":
    main()
