"""Evaluate the current model on multiple holdout splits.

Logs each split to WandB project 'perturbation-eval', and also writes a local
JSON aggregate (rhaister_results.json) so figures can consume the results
without a wandb roundtrip.

Usage:
    python eval_splits.py <run_name> [--mode pdex|pcl|static]
"""
import json
import os
import sys
import numpy as np
import wandb
from rhaister import train

OUTPUT_JSON = "rhaister_results.json"

SPLITS = [f"tahoe_{i}_holdout" for i in range(5, 10)]

METRICS_BY_MODE = {
    "pdex": [
        ("pdex_static/pearson_delta_mean", "pdex_static/pearson_delta_mean"),
        ("pdex_static/auprc_p05", "pdex_static/auprc_p05"),
        ("state/pearson_delta_mean", "state/pearson_delta_mean"),
        ("state/de_overlap_mean", "state/de_overlap_mean"),
        ("state/de_spearman_sig", "state/de_spearman_sig"),
        ("state/pr_auc_mean", "state/pr_auc_mean"),
        ("state/spearman_lfc_sig_mean", "state/spearman_lfc_sig_mean"),
        ("state/discrimination_mean", "state/discrimination_mean"),
    ],
}

# Parse arguments
run_name = sys.argv[1] if len(sys.argv) > 1 else "unified"
mode = "pdex"
for i, arg in enumerate(sys.argv):
    if arg == "--mode" and i + 1 < len(sys.argv):
        mode = sys.argv[i + 1]

if mode not in METRICS_BY_MODE:
    print(f"Unknown mode '{mode}'. Available: {list(METRICS_BY_MODE.keys())}")
    sys.exit(1)

METRICS = METRICS_BY_MODE[mode]
results = {name: [] for name, _ in METRICS}
per_split_records = {}

for split in SPLITS:
    print(f"\n{'='*60}")
    print(f"Evaluating on {split}")
    print(f"{'='*60}")

    metrics = train.train_and_evaluate(split_name=split, log=False, compute_discrimination=True)

    # Collect results
    wandb_metrics = {}
    for name, key in METRICS:
        val = metrics[key]
        results[name].append(val)
        wandb_metrics[name] = val
    per_split_records[split] = {name: float(val) for name, val in wandb_metrics.items()}

    # Log to perturbation-eval
    split_config = f"configs/{split}/generalization_converted_cell_lines_3b.toml"
    wandb.init(
        project="perturbation-eval",
        job_type="rhaister",
        name=f"eval_rhaister_{run_name}",
        config={"split_config": split_config, "mode": mode},
        reinit=True,
    )
    wandb.log(wandb_metrics)
    wandb.finish()

# Write local aggregate so figure scripts can read without re-fetching from wandb.
# Schema mirrors fetch_wandb.py output: {run_name: {split: {metric: value}}}.
existing = {}
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON) as f:
        existing = json.load(f)
run_key = f"eval_rhaister_{run_name}"
existing[run_key] = per_split_records
with open(OUTPUT_JSON, "w") as f:
    json.dump(existing, f, indent=2, sort_keys=True)
print(f"\nwrote {OUTPUT_JSON} under key '{run_key}'")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"  {'Metric':<42} {'Mean':>8} {'Std':>8}   per-split")
print(f"  {'-'*80}")
for name, _ in METRICS:
    vals = results[name]
    per = "  ".join(f"{v:.4f}" for v in vals)
    print(f"  {name:<42} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   {per}")
