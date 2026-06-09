"""Label-permutation ablation for the multi-target model.

Runs the same holdout splits as eval_splits.py but with one set of training
labels (treatments or cells) permuted, breaking that label's signal. Compare
the resulting metrics against eval_splits.py to quantify which label the
model relies on more.

Usage:
    uv run python eval_permute.py <run_name> --permute {treatments,cells} [--wandb]
"""
import json
import os
import sys
import numpy as np
from rhaister import train
from rhaister.prepare_combined import prepare_all

OUTPUT_JSON = "label_permutation_ablation.json"

SPLITS = [f"tahoe_{i}_holdout" for i in range(5, 10)]
METRICS = [
    ("pdex_static/pearson_delta_mean", "pdex_static/pearson_delta_mean"),
    ("pdex_static/auprc_p05", "pdex_static/auprc_p05"),
    ("state/pearson_delta_mean", "state/pearson_delta_mean"),
    ("state/de_overlap_mean", "state/de_overlap_mean"),
    ("state/de_spearman_sig", "state/de_spearman_sig"),
    ("state/pr_auc_mean", "state/pr_auc_mean"),
    ("state/spearman_lfc_sig_mean", "state/spearman_lfc_sig_mean"),
    ("state/discrimination_mean", "state/discrimination_mean"),
]
SEED = 42


def _permute_train_labels(data, mode, rng):
    """Within-group label shuffle.

    The regression in train._compute_regression requires each non-holdout cell
    to share its training treatments with the others (it intersects drug sets
    across cells). A full-array shuffle destroys that intersection and crashes.
    Permuting *within* a group keeps each cell's treatment set intact (and
    each treatment's cell set intact) while still decoupling labels from the
    Y_train row they supervise.
    """
    data = dict(data)
    cells = np.asarray(data["train_cells"])
    treatments = np.asarray(data["train_treatments"])

    if mode == "treatments":
        new_treatments = treatments.copy()
        for c in np.unique(cells):
            idx = np.where(cells == c)[0]
            new_treatments[idx] = rng.permutation(treatments[idx])
        data["train_treatments"] = new_treatments
    elif mode == "cells":
        new_cells = cells.copy()
        for t in np.unique(treatments):
            idx = np.where(treatments == t)[0]
            new_cells[idx] = rng.permutation(cells[idx])
        data["train_cells"] = new_cells
    else:
        raise ValueError(f"unknown permute mode: {mode}")
    return data


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else "permute"
    mode = None
    use_wandb = False
    for i, arg in enumerate(sys.argv):
        if arg == "--permute" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == "--wandb":
            use_wandb = True
    if mode not in {"treatments", "cells"}:
        print("Usage: uv run python eval_permute.py <run_name> --permute {treatments,cells} [--wandb]")
        sys.exit(1)
    if use_wandb:
        import wandb

    results = {name: [] for name, _ in METRICS}

    for split in SPLITS:
        print(f"\n{'='*60}\n{split}  (permute={mode})\n{'='*60}")
        data = prepare_all(split)
        rng = np.random.default_rng(SEED)
        data = _permute_train_labels(data, mode, rng)

        metrics = train.train_and_evaluate(
            split_name=split, log=False, data=data, compute_discrimination=True,
        )

        split_metrics = {name: metrics[key] for name, key in METRICS}
        for name, val in split_metrics.items():
            results[name].append(val)

        if use_wandb:
            wandb.init(
                project="perturbation-eval",
                job_type=f"rhaister_permute_{mode}",
                name=f"eval_permute_{mode}_{run_name}",
                config={"split": split, "permute": mode, "seed": SEED},
                reinit=True,
            )
            wandb.log(split_metrics)
            wandb.finish()

    print(f"\n{'='*60}\nSUMMARY  permute={mode}\n{'='*60}")
    print(f"  {'Metric':<42} {'Mean':>8} {'Std':>8}   per-split")
    print(f"  {'-'*80}")
    for name, _ in METRICS:
        vals = results[name]
        per = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {name:<42} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   {per}")

    # Merge this mode's results into the shared aggregate JSON.
    existing = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            existing = json.load(f)
    existing.setdefault("experiment", "label_permutation_ablation")
    existing.setdefault("description",
        "Ablation of the multi-target perturbation model by permuting one "
        "label at a time in the training data. Within-group shuffle: "
        "'treatments' shuffles train_treatments within each train cell; "
        "'cells' shuffles train_cells within each train treatment.")
    existing.setdefault("splits", list(SPLITS))
    existing.setdefault("seed", SEED)
    modes = existing.setdefault("modes", {})
    mode_entry = {
        "description": f"train_{mode} shuffled within each train {'cell' if mode == 'treatments' else 'treatment'}",
        "metrics": {},
    }
    for name, _ in METRICS:
        vals = results[name]
        mode_entry["metrics"][name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "per_split": {split: float(v) for split, v in zip(SPLITS, vals)},
        }
    modes[mode] = mode_entry
    with open(OUTPUT_JSON, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)
    print(f"\nwrote {OUTPUT_JSON} (mode={mode})")


if __name__ == "__main__":
    main()
