"""Fetch STATE eval metrics from WandB and save to state_results.json.

The external evaluator ran the same splits and metrics as we do, under the
model name 'STATE'. We pull the state/* summary metrics from the specific
runs in vevotx/perturbation-eval that cover the tahoe (5..9) and
replogle_nadig fewshot (per cell line) splits.

Output schema mirrors model_results.json:
    {"<split>": {"STATE": {metric: value, ...}}, ...}

Usage:
    uv run python scripts/fetch_state.py
"""
from __future__ import annotations

import json
import os
import sys


import wandb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = "vevotx/perturbation-eval"
MODEL_NAME = "STATE"
OUT_PATH = os.path.join(REPO_ROOT, "state_results.json")

# split key -> ordered list of run names to try. First match (most recent
# matching run) wins. The -pds suffix marks the rerun that also logs
# state/discrimination_mean (PDS); we prefer those when available and fall
# back to the original runs for the two splits that haven't been re-run yet
# (tahoe_5_holdout, replogle_nadig/split_0). replogle_nadig cell-line ->
# split idx comes from splits/replogle_nadig/split_*/split.toml.
SPLIT_TO_RUNS = {
    "tahoe_5_holdout": ["eval_state-st+hvg-pds_tahoe5", "eval_state-st+hvg_tahoe5"],
    "tahoe_6_holdout": ["eval_state-st+hvg-pds_tahoe6", "eval_state-st+hvg_tahoe6"],
    "tahoe_7_holdout": ["eval_state-st+hvg-pds_tahoe7", "eval_state-st+hvg_tahoe7"],
    "tahoe_8_holdout": ["eval_state-st+hvg-pds_tahoe8", "eval_state-st+hvg_tahoe8"],
    "tahoe_9_holdout": ["eval_state-st+hvg-pds_tahoe9", "eval_state-st+hvg_tahoe9"],
    "replogle_nadig/split_0": ["eval_st-hvg-replogle-fewshot-hepg2-pds",  "eval_st-hvg-replogle-fewshot-hepg2"],
    "replogle_nadig/split_1": ["eval_st-hvg-replogle-fewshot-jurkat-pds", "eval_st-hvg-replogle-fewshot-jurkat"],
    "replogle_nadig/split_2": ["eval_st-hvg-replogle-fewshot-k562-pds",   "eval_st-hvg-replogle-fewshot-k562"],
    "replogle_nadig/split_3": ["eval_st-hvg-replogle-fewshot-rpe1-pds",   "eval_st-hvg-replogle-fewshot-rpe1"],
}

# Metrics we surface in figures (six State metrics + pdex auxiliaries when
# present). Missing keys are silently skipped — some runs only log a subset.
METRIC_KEYS = [
    "state/pearson_delta_mean",
    "state/spearman_lfc_sig_mean",
    "state/pr_auc_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
    "state/discrimination_mean",
    "pdex_static/pearson_delta_mean",
    "pdex_static/auprc_p05",
]


def _summary_metrics(run) -> dict:
    summary = dict(run.summary)
    return {k: float(summary[k]) for k in METRIC_KEYS if k in summary}


def fetch() -> dict:
    api = wandb.Api()
    out: dict[str, dict] = {}
    for split, candidate_names in SPLIT_TO_RUNS.items():
        for run_name in candidate_names:
            runs = list(api.runs(PROJECT, filters={"displayName": run_name}, order="-created_at"))
            if runs:
                run = runs[0]
                metrics = _summary_metrics(run)
                if not metrics:
                    print(f"  [warn] {run_name} ({run.id}): no recognised metrics", file=sys.stderr)
                out.setdefault(split, {})[MODEL_NAME] = metrics
                print(f"  {split:32s} <- {run_name}  ({len(metrics)} metrics, id={run.id})")
                break
        else:
            print(f"  [miss] {split}: tried {candidate_names}, no matching runs", file=sys.stderr)
    return out


def main() -> None:
    print(f"Fetching {len(SPLIT_TO_RUNS)} {MODEL_NAME} splits from {PROJECT} ...")
    results = fetch()
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUT_PATH}  ({len(results)} splits)")


if __name__ == "__main__":
    main()
