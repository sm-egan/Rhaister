"""Run baselines + full model (with HP_IMPUTE_MISSING=1) on all 10 parse splits.

Writes per-split results into baseline_results.json and model_results.json
INCREMENTALLY — each split is saved as soon as it's done, so a long-running
run can be killed without losing work.

Progress is printed to stdout with flush=True so `tail -F` on a log file
shows live progress. Run with:

    uv run python scripts/run_parse_sweep.py 2>&1 | tee /tmp/parse_sweep.log
"""
import os, sys, json, time

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HP_IMPUTE_MISSING", "1")

import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import baselines as baselines_mod
from rhaister import train

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_JSON = os.path.join(REPO_ROOT, "baseline_results.json")
MODEL_JSON    = os.path.join(REPO_ROOT, "model_results.json")

SPLITS = (
    [f"parse/split_{i}" for i in range(5)]
    + [f"parse/donor_split_{i}" for i in range(5)]
)

def clean(metrics):
    return {k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float, np.floating))}

def merge_save(path, split, payload):
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing[split] = payload
    with open(path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

print(f"splits to run: {SPLITS}", flush=True)

for split in SPLITS:
    print(f"\n==================== {split} ====================", flush=True)
    t0 = time.time()
    print(f"[{split}] baselines starting ...", flush=True)
    bl = baselines_mod.compute_baselines(split)
    bl_payload = {name: clean(m) for name, m in bl.items()}
    merge_save(BASELINE_JSON, split, bl_payload)
    print(f"[{split}] baselines done in {time.time()-t0:.1f}s, saved to {BASELINE_JSON}", flush=True)

    t0 = time.time()
    print(f"[{split}] model starting ...", flush=True)
    m = train.train_and_evaluate(split_name=split, log=False, compute_discrimination=True)
    model_payload = {"full_impute": clean(m)}
    merge_save(MODEL_JSON, split, model_payload)
    print(f"[{split}] model done in {time.time()-t0:.1f}s, saved to {MODEL_JSON}", flush=True)

print("\nSWEEP COMPLETE", flush=True)
