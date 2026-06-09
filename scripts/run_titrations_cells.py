"""Cell-line titration: vary the number of non-holdout training cell lines.

In-memory subsetting — no predefined split files are needed. For each holdout,
we permute the non-holdout cells with a fixed seed and take nested prefixes of
increasing size. Holdout-cell training rows (the fewshot rows) are always kept
so test-time cell lookup still resolves.

Usage:
    uv run python scripts/run_titrations_cells.py [--wandb]
"""
import json
import os
import sys


import numpy as np
from rhaister import train
from rhaister.prepare_combined import prepare_all

HOLDOUTS = [5, 6, 7, 8, 9]
CELL_LEVELS = [1, 5, 10, 20, 45]
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
    "train_cells", "train_treatments",
)
OUTPUT = "titration_cells_results.json"


def _subsample_cells(data, selected_cells):
    """Restrict training rows to selected_cells ∪ holdout_cells, adding whole
    cells if needed to cover every test treatment.

    Two invariants to respect:
    (a) local_treat_map in train.py:45 is built from training treatments, so
        every test treatment must appear in training → avoids KeyError.
    (b) _compute_regression intersects treatment sets across non-holdout cells
        (train.py:307-311); the intersection must be non-empty → each
        non-holdout cell must contribute its full treatment inventory.

    We enforce (a) by greedily adding non-selected cells (in full) whose
    treatment inventory contains missing test treatments. (b) then holds
    because every non-holdout cell in the final subset retains all its rows.
    Added cells count toward the effective training-cell total but not toward
    the requested level L.
    """
    train_cells = np.asarray(data["train_cells"])
    train_tr = np.asarray(data["train_treatments"])
    holdout_set = set(data["test_cells"])
    test_tr_set = set(np.asarray(data["test_treatments"]))

    keep_cells = set(selected_cells) | holdout_set
    base_mask = np.array([c in keep_cells for c in train_cells])
    covered = set(train_tr[base_mask])
    missing = test_tr_set - covered

    candidates = sorted(set(train_cells) - holdout_set - set(selected_cells))
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

    if extra_cells:
        print(f"  coverage augmentation: +{len(extra_cells)} whole cells for {len(missing)} missing test treatments")

    final_keep = keep_cells | set(extra_cells)
    final_mask = np.array([c in final_keep for c in train_cells])

    out = dict(data)
    for k in TRAIN_ARRAYS:
        out[k] = np.asarray(data[k])[final_mask]
    return out


def main():
    use_wandb = "--wandb" in sys.argv
    if use_wandb:
        import wandb

    results = {m: {str(h): {} for h in HOLDOUTS} for m in METRICS}
    actual_levels = {str(h): {} for h in HOLDOUTS}

    for h in HOLDOUTS:
        split_name = f"tahoe_{h}_holdout"
        print(f"\n### Holdout {h}: {split_name} ###")
        # Use a scratch load just to determine cell counts and the permutation.
        probe = prepare_all(split_name)
        holdout_cells = set(probe["test_cells"])
        non_holdout = sorted(set(probe["train_cells"]) - holdout_cells)
        n_total = len(non_holdout)
        shuffled = list(np.random.RandomState(SEED).permutation(non_holdout))
        del probe
        print(f"  {n_total} non-holdout cells available (+{len(holdout_cells)} holdout kept)")

        for L in CELL_LEVELS:
            L_actual = min(L, n_total)
            selected = shuffled[:L_actual]
            # Reload fresh each iteration — evaluate_test is a one-shot closure
            # that errors on the second call (train.py guard).
            data = prepare_all(split_name)
            sub_data = _subsample_cells(data, selected)
            n_train = len(sub_data["train_cells"])
            print(f"\n{'='*60}\nh={h}  L={L}  (actual={L_actual}, train_rows={n_train})\n{'='*60}")

            metrics = train.train_and_evaluate(
                data=sub_data, log=False, compute_discrimination=True,
            )

            row = {m: float(metrics[m]) for m in METRICS}
            for m, v in row.items():
                results[m][str(h)][str(L)] = v
            actual_levels[str(h)][str(L)] = L_actual
            print(f"  h={h} L={L:>3}  " + "  ".join(f"{m}={row[m]:.4f}" for m in METRICS))

            if use_wandb:
                wandb.init(
                    project="perturbation-eval",
                    job_type="rhaister_titration_cells",
                    name=f"titration_cells_h{h}_L{L}",
                    config={"holdout": h, "level": L, "level_actual": L_actual, "axis": "cells", "seed": SEED},
                    reinit=True,
                )
                wandb.log(row)
                wandb.finish()

    out = {
        "experiment": "cell_titration",
        "description": (
            "Cell-line titration by in-memory subsetting of non-holdout "
            "training cells (seed 42, nested prefixes). Holdout cells' "
            "training rows are always kept. 'level' is the requested number "
            "of non-holdout cells; 'actual_levels' records what was used "
            "after clamping to the split's available count."
        ),
        "seed": SEED,
        "holdouts": HOLDOUTS,
        "levels": CELL_LEVELS,
        "actual_levels": actual_levels,
        "metrics": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUTPUT}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for m in METRICS:
        print(f"\n{m}")
        header = "  h \\ L  " + "  ".join(f"{L:>8}" for L in CELL_LEVELS)
        print(header)
        for h in HOLDOUTS:
            vals = "  ".join(f"{results[m][str(h)].get(str(L), float('nan')):>8.4f}" for L in CELL_LEVELS)
            print(f"    {h}    {vals}")


if __name__ == "__main__":
    main()
