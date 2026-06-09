"""Per-holdout zeroshot baselines for the main_metrics figure.

All baselines computed from zeroshot-eligible training data only (i.e., the
panel-free subset that drops held-out cells from train) so the figure can
honestly say "this is what's achievable without panel observations of the
test cell line."

Three baselines:
- Global Mean Zeroshot: per-gene mean of Y/D/P across all zeroshot train rows;
  broadcast to all test rows (no drug, no cell information used).
- Perturbation Mean Zeroshot: per-(drug, gene) mean over zeroshot train rows;
  for each test row, look up the mean of its drug.
- Rhaister Zeroshot Shared: shared-γ-per-gene diagonal model
  (HP_ZS_DIAG_SHARED=1) through the same pipeline as the headline zeroshot
  model (so it gets the same calnet calibration).

The headline drug-specific γ_{t,g} model values are already in
titration_cells_zeroshot_results.json at level L=45.

Saved to: zeroshot_baselines_results.json
"""
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np

HOLDOUTS = [5, 6, 7, 8, 9]
METRICS = [
    "pdex_static/pearson_delta_mean",
    "pdex_static/auprc_p05",
    "state/pearson_delta_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
    "state/pr_auc_mean",
    "state/spearman_lfc_sig_mean",
    "state/discrimination_mean",
]
OUTPUT = "zeroshot_baselines_results.json"


def _load_test_arrays(h):
    cp = f"/tmp/tahoe_cache_combined/tahoe/{h}_holdout"
    return (
        np.load(f"{cp}/Y_test.npy"),
        np.load(f"{cp}/P_test.npy"),
        np.load(f"{cp}/D_test.npy"),
        np.load(f"{cp}/F_test.npy"),
    )


def _direct_baseline(h, kind):
    """Direct (no model pipeline) baseline: kind in {'global_mean', 'pert_mean'}.
    Predicts Y/D/P from train statistics and derives F via Benjamini-Hochberg."""
    from rhaister.prepare_combined import prepare_all, evaluate, pvalues_to_fdr_bh
    data = prepare_all(f"tahoe_{h}_holdout", zeroshot=True)
    Y_test, P_test, D_test, F_test = _load_test_arrays(h)

    Y_train = np.asarray(data["Y_train"], dtype=np.float64)
    D_train = np.asarray(data["D_train"], dtype=np.float64)
    P_train = np.asarray(data["P_train"], dtype=np.float64)
    train_treats = np.asarray(data["train_treatments"])
    test_treats = np.asarray(data["test_treatments"])
    test_cells = np.asarray(data["test_cells"])
    n_test = data["n_test"]
    n_genes = Y_train.shape[1]

    if kind == "global_mean":
        Y_pred = np.broadcast_to(Y_train.mean(0), (n_test, n_genes)).copy()
        D_pred = np.broadcast_to(D_train.mean(0), (n_test, n_genes)).copy()
        P_pred = np.broadcast_to(P_train.mean(0), (n_test, n_genes)).copy()
    elif kind == "pert_mean":
        drugs = sorted(set(train_treats))
        d_to_idx = {d: i for i, d in enumerate(drugs)}
        n_drugs = len(drugs)
        Y_d = np.zeros((n_drugs, n_genes), dtype=np.float64)
        D_d = np.zeros((n_drugs, n_genes), dtype=np.float64)
        P_d = np.zeros((n_drugs, n_genes), dtype=np.float64)
        counts = np.zeros(n_drugs)
        for i, t in enumerate(train_treats):
            j = d_to_idx[t]
            Y_d[j] += Y_train[i]
            D_d[j] += D_train[i]
            P_d[j] += P_train[i]
            counts[j] += 1
        Y_d /= counts[:, None].clip(min=1)
        D_d /= counts[:, None].clip(min=1)
        P_d /= counts[:, None].clip(min=1)
        Y_global = Y_train.mean(0)
        D_global = D_train.mean(0)
        P_global = P_train.mean(0)
        Y_pred = np.zeros((n_test, n_genes))
        D_pred = np.zeros((n_test, n_genes))
        P_pred = np.zeros((n_test, n_genes))
        for i, t in enumerate(test_treats):
            j = d_to_idx.get(t)
            if j is not None:
                Y_pred[i] = Y_d[j]
                D_pred[i] = D_d[j]
                P_pred[i] = P_d[j]
            else:
                Y_pred[i] = Y_global
                D_pred[i] = D_global
                P_pred[i] = P_global
    else:
        raise ValueError(f"unknown direct baseline kind: {kind!r}")
    F_pred = pvalues_to_fdr_bh(P_pred)

    return evaluate(
        Y_test, Y_pred, P_test, P_pred, D_test, D_pred, F_test, F_pred,
        test_cells, test_treats, data["gene_cols"], compute_discrimination=False,
    )


def _pipeline_baseline(h, env_overrides):
    """Run the full zeroshot pipeline with given env overrides (e.g.
    HP_ZS_DIAG_SHARED=1)."""
    from rhaister import train
    from rhaister.prepare_combined import prepare_all
    os.environ["HP_ZS_MODEL"] = "diagonal"
    os.environ.setdefault("HP_ZS_DIAG_Z", "H")
    for k, v in env_overrides.items():
        os.environ[k] = v
    data = prepare_all(f"tahoe_{h}_holdout", zeroshot=True)
    return train.train_and_evaluate(
        experiment_name=f"zs_baseline_h{h}",
        split_name=f"tahoe_{h}_holdout",
        data=data,
        log=False,
        zeroshot=True,
        compute_discrimination=False,
    )


def run_one(task):
    h, name, mode, payload = task
    print(f"[h={h} {name}] starting", flush=True)
    if mode == "direct":
        metrics = _direct_baseline(h, payload)
    elif mode == "pipeline":
        metrics = _pipeline_baseline(h, payload)
    else:
        raise ValueError(mode)
    return h, name, {m: float(metrics[m]) for m in METRICS if m in metrics}


BASELINES = [
    # (display_name, mode, payload)
    ("Global Mean Zeroshot",       "direct",   "global_mean"),
    ("Perturbation Mean Zeroshot", "direct",   "pert_mean"),
    ("Rhaister Zeroshot Shared",   "pipeline", {"HP_ZS_DIAG_SHARED": "1"}),
]


def main():
    n_workers = 4
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            n_workers = int(sys.argv[i + 1])

    tasks = [(h, name, mode, payload)
             for h in HOLDOUTS for (name, mode, payload) in BASELINES]
    print(f"Dispatching {len(tasks)} (holdout, baseline) configs across {n_workers} workers")

    results = {b[0]: {} for b in BASELINES}

    ctx = mp.get_context("spawn")
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            h_req, name_req, *_ = futures[fut]
            try:
                h, name, row = fut.result()
            except Exception as exc:
                n_done += 1
                print(f"[FAILED h={h_req} {name_req}] {exc} ({n_done}/{len(tasks)})", flush=True)
                continue
            results[name][str(h)] = row
            n_done += 1
            short = "  ".join(f"{m.split('/')[-1]}={row.get(m, float('nan')):.3f}" for m in METRICS)
            print(f"[done h={h} {name:32s}] {short} ({n_done}/{len(tasks)})", flush=True)

    out = {
        "experiment": "zeroshot_baselines",
        "description": (
            "Per-holdout panel-free zeroshot baselines. Global Mean and "
            "Perturbation Mean are direct statistical baselines (no model "
            "pipeline) computed from zeroshot-eligible train data; the Shared "
            "diagonal goes through the model pipeline with HP_ZS_DIAG_SHARED=1 "
            "(γ_g shared across drugs). The headline drug-specific γ_{t,g} "
            "results live in titration_cells_zeroshot_results.json (L=45)."
        ),
        "holdouts": HOLDOUTS,
        "baselines": [b[0] for b in BASELINES],
        "metrics_per_baseline": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
