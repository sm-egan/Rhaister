"""Fetch baseline and rhaister model results from WandB 'perturbation-eval' project.

Saves to baseline_results.json and rhaister_results.json at repo root.

Usage:
    python scripts/fetch_wandb.py
"""
import json
import os
import re
import sys


import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = [f"tahoe_{i}_holdout" for i in range(5, 10)]

METRIC_KEYS = [
    "pdex_static/pearson_delta_mean",
    "pdex_static/auprc_p05",
    "state/pearson_delta_mean",
    "state/spearman_lfc_sig_mean",
    "state/pr_auc_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
    "state/discrimination_mean",  # may be absent
]


def _summary_metrics(run):
    summary = dict(run.summary)
    return {k: summary[k] for k in METRIC_KEYS if k in summary}


def fetch_baselines():
    """Fetch baseline runs. Uses the old 'eval_baseline_{name}_{split}' convention.
    If not available, falls back to time-based inference on recent batches."""
    api = wandb.Api()
    runs = list(api.runs("perturbation-eval", filters={"jobType": "baseline"}, order="-created_at"))

    # Prefer runs with split in name (older convention: eval_baseline_NAME_tahoe_N_holdout)
    named = {}
    for r in runs:
        m = re.match(r"eval_baseline_(\w+?)_(tahoe_\d+_holdout)$", r.name)
        if m:
            baseline_name, split = m.groups()
            # Keep most recent per (split, baseline)
            key = (split, baseline_name)
            if key not in named:
                named[key] = r

    if len(named) >= 20:
        print(f"Using {len(named)} named baseline runs")
        results = {split: {} for split in SPLITS}
        for (split, baseline_name), run in named.items():
            if split in results:
                results[split][baseline_name] = _summary_metrics(run)
        return results

    # Fallback: infer from time batches (4 consecutive runs per split, earliest = tahoe_5_holdout)
    print(f"Only {len(named)} named runs; inferring splits from time batches")
    recent = [r for r in runs if not re.match(r"eval_baseline_\w+_tahoe_\d+_holdout$", r.name)]
    recent = recent[:20]  # most recent 20 = 5 splits × 4 baselines
    recent.reverse()  # oldest first

    results = {split: {} for split in SPLITS}
    for i, run in enumerate(recent):
        split_idx = i // 4
        split = SPLITS[split_idx]
        # run.name is like "baseline_additive"
        baseline_name = run.name.replace("baseline_", "")
        results[split][baseline_name] = _summary_metrics(run)
    return results


def fetch_rhaister(run_name_suffix=None):
    """Fetch rhaister eval_splits runs. Groups by time batch (5 consecutive = one full sweep).
    If run_name_suffix given, only fetches runs matching eval_rhaister_{suffix}."""
    api = wandb.Api()
    filters = {"jobType": "rhaister"}
    if run_name_suffix:
        filters["displayName"] = f"eval_rhaister_{run_name_suffix}"
    runs = list(api.runs("perturbation-eval", filters=filters, order="-created_at"))

    # Group by run name, take 5 most recent of same name (one per split)
    by_name = {}
    for r in runs:
        by_name.setdefault(r.name, []).append(r)

    out = {}
    for name, name_runs in by_name.items():
        if len(name_runs) < 5:
            continue
        # Most recent 5, oldest first → splits 5..9
        sweep = sorted(name_runs[:5], key=lambda r: r.created_at)
        split_results = {}
        for i, run in enumerate(sweep):
            split_results[SPLITS[i]] = _summary_metrics(run)
        out[name] = split_results
    return out


def main():
    print("Fetching baselines...")
    baselines = fetch_baselines()
    baseline_path = os.path.join(REPO_ROOT, "baseline_results.json")
    with open(baseline_path, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
    print(f"  saved to {baseline_path}")
    for split in SPLITS:
        print(f"  {split}: {list(baselines[split].keys())}")

    print("\nFetching rhaister eval_splits results...")
    rhaister = fetch_rhaister()
    rhaister_path = os.path.join(REPO_ROOT, "rhaister_results.json")
    with open(rhaister_path, "w") as f:
        json.dump(rhaister, f, indent=2, sort_keys=True)
    print(f"  saved to {rhaister_path}")
    for name in rhaister:
        print(f"  {name}: {list(rhaister[name].keys())}")


if __name__ == "__main__":
    main()
