"""Zeroshot cell-line titration across tahoe holdouts (parallel).

Varies the number of non-holdout training cells while keeping the same fixed
test set per holdout. Uses the zeroshot diagonal model (HP_ZS_MODEL=diagonal,
HP_ZS_DIAG_Z=H). Holdout cells are dropped from train inside prepare_all
(zeroshot=True); titration just subsets non-holdout cells, with coverage
augmentation to keep every test treatment present.

`state/discrimination_mean` is computed (compute_discrimination=True), which
is the slow step — that's why this script uses a process pool to fan out
across multiple workers. Per-process CUDA contexts share the GPU; on a B200
4 workers fit easily, and CPU-bound discrimination scoring overlaps well.

Usage:
    uv run python scripts/run_titrations_cells_zeroshot.py [--wandb] [--workers N]
"""
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np


HOLDOUTS = [5, 6, 7, 8, 9]
CELL_LEVELS = [1, 2, 3, 5, 10, 15, 20, 30, 45]
SEED = 42
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
TRAIN_ARRAYS = (
    "Y_train", "P_train", "D_train", "F_train", "R_train",
    "L_train", "C_train", "H_train",
    "train_cells", "train_treatments",
)
OUTPUT = "titration_cells_zeroshot_results.json"


def _subsample_cells(data, selected_cells):
    """Restrict to selected_cells, augment with whole non-holdout cells so
    every test treatment is covered (avoids ALS local_treat_map KeyError)."""
    train_cells = np.asarray(data["train_cells"])
    train_tr = np.asarray(data["train_treatments"])
    test_tr_set = set(np.asarray(data["test_treatments"]))

    keep_cells = set(selected_cells)
    base_mask = np.isin(train_cells, list(keep_cells))
    covered = set(train_tr[base_mask])
    missing = test_tr_set - covered

    candidates = sorted(set(train_cells) - keep_cells)
    extra_cells = []
    for t in sorted(missing):
        if t in covered:
            continue
        for c in candidates:
            if c in extra_cells:
                continue
            c_trs = set(train_tr[train_cells == c])
            if t in c_trs:
                extra_cells.append(c)
                covered |= c_trs
                break
    final_keep = keep_cells | set(extra_cells)
    final_mask = np.isin(train_cells, list(final_keep))
    out = dict(data)
    for k in TRAIN_ARRAYS:
        if k in data:
            out[k] = np.asarray(data[k])[final_mask]
    return out, len(extra_cells)


def run_one(task):
    """Worker: run one (holdout, level) configuration end-to-end. Returns
    a (h, L, L_actual, augmented, metrics_dict) tuple."""
    h, L = task
    # Late imports so module load happens inside the worker process (avoids
    # CUDA context inheritance issues with fork).
    from rhaister import train
    from rhaister.prepare_combined import prepare_all

    os.environ["HP_ZS_MODEL"] = "diagonal"
    os.environ.setdefault("HP_ZS_DIAG_Z", "H")

    split_name = f"tahoe_{h}_holdout"
    data = prepare_all(split_name, zeroshot=True)
    non_holdout = sorted(set(data["train_cells"]))
    n_total = len(non_holdout)
    L_actual = min(L, n_total)
    shuffled = list(np.random.RandomState(SEED).permutation(non_holdout))
    selected = shuffled[:L_actual]
    sub_data, n_extra = _subsample_cells(data, selected)

    print(f"[h={h} L={L}] start: actual={L_actual}, +{n_extra} aug, "
          f"train_rows={len(sub_data['train_cells'])}", flush=True)
    metrics = train.train_and_evaluate(
        experiment_name=f"zs_titr_h{h}_L{L}",
        split_name=split_name,
        data=sub_data,
        log=False,
        zeroshot=True,
        compute_discrimination=False,
    )
    row = {m: float(metrics[m]) for m in METRICS if m in metrics}
    return h, L, L_actual, n_extra, row


def main():
    use_wandb = "--wandb" in sys.argv
    n_workers = 4
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv):
            n_workers = int(sys.argv[i + 1])

    if use_wandb:
        import wandb

    tasks = [(h, L) for h in HOLDOUTS for L in CELL_LEVELS]
    print(f"Dispatching {len(tasks)} (holdout, level) configs across {n_workers} workers")

    results = {m: {str(h): {} for h in HOLDOUTS} for m in METRICS}
    actual_levels = {str(h): {} for h in HOLDOUTS}

    # "spawn" so each worker gets a fresh CUDA context (fork + CUDA is unsafe).
    ctx = mp.get_context("spawn")
    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            h_req, L_req = futures[fut]
            try:
                h, L, L_actual, n_extra, row = fut.result()
            except Exception as exc:
                n_done += 1
                print(f"[FAILED h={h_req} L={L_req}] {exc}  ({n_done}/{len(tasks)})", flush=True)
                continue
            for m, v in row.items():
                results[m][str(h)][str(L)] = v
            actual_levels[str(h)][str(L)] = L_actual
            n_done += 1
            short = "  ".join(f"{m.split('/')[-1]}={row.get(m, float('nan')):.3f}" for m in METRICS)
            print(f"[done h={h} L={L:>3} +{n_extra}aug]  {short}  ({n_done}/{len(tasks)})", flush=True)
            if use_wandb:
                wandb.init(
                    project="perturbation-eval",
                    job_type="rhaister_titration_cells_zeroshot",
                    name=f"zs_titr_h{h}_L{L}",
                    config={"holdout": h, "level": L, "level_actual": L_actual,
                            "axis": "cells", "zeroshot": True, "model": "diagonal",
                            "z_source": "H"},
                    reinit=True,
                )
                wandb.log(row)
                wandb.finish()

    out = {
        "experiment": "cell_titration_zeroshot",
        "description": (
            "Zeroshot cell-line titration across tahoe holdouts (5, 6, 7, 8, 9). "
            "For each holdout split: subsample non-holdout training cells "
            "(single seed=42, fresh RandomState permutation), evaluate the "
            "same fixed test set per holdout. Different holdouts already "
            "supply the across-split randomness. Diagonal model with "
            "z=log1p(H) (HVG centroid). Coverage augmentation adds whole "
            "non-holdout cells to cover any otherwise-missing test treatment; "
            "actual_levels[h][L] records what was used after that augmentation. "
            "Includes state/discrimination_mean (compute_discrimination=True)."
        ),
        "holdouts": HOLDOUTS,
        "seed": SEED,
        "levels": CELL_LEVELS,
        "actual_levels": actual_levels,
        "metrics": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUTPUT}")

    print(f"\n{'='*60}\nSUMMARY: state/pearson_delta_mean\n{'='*60}")
    m = "state/pearson_delta_mean"
    header = "  h \\ L  " + "  ".join(f"{L:>8}" for L in CELL_LEVELS)
    print(header)
    for h in HOLDOUTS:
        vals = "  ".join(
            f"{results[m][str(h)].get(str(L), float('nan')):>8.4f}"
            for L in CELL_LEVELS
        )
        print(f"    {h}    {vals}")


if __name__ == "__main__":
    main()
