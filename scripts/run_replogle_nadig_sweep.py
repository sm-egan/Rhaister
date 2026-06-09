"""Run baselines + full model on all 4 replogle_nadig fewshot splits.

Mirrors scripts/run_parse_sweep.py. Writes per-split results into
baseline_results.json and model_results.json INCREMENTALLY — each split is
saved as soon as it's done, so a long-running run can be killed without
losing work. Assumes per-split caches have already been written by
scripts/build_replogle_nadig_caches.py (otherwise prepare_all rebuilds them
on the fly per split).

Usage:
    uv run python scripts/run_replogle_nadig_sweep.py 2>&1 | tee /tmp/replogle_nadig_sweep.log
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

SPLITS = [f"replogle_nadig/split_{i}" for i in range(4)]


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


def main():
    print(f"splits to run: {SPLITS}", flush=True)

    for split in SPLITS:
        print(f"\n==================== {split} ====================", flush=True)

        t0 = time.time()
        print(f"[{split}] baselines starting ...", flush=True)
        bl = baselines_mod.compute_baselines(split)
        bl_payload = {name: clean(m) for name, m in bl.items()}
        merge_save(BASELINE_JSON, split, bl_payload)
        print(f"[{split}] baselines done in {time.time()-t0:.1f}s, saved", flush=True)

        t0 = time.time()
        print(f"[{split}] model starting ...", flush=True)
        m = train.train_and_evaluate(
            split_name=split, log=False, compute_discrimination=True,
        )
        merge_save(MODEL_JSON, split, {"full_impute": clean(m)})
        print(f"[{split}] model done in {time.time()-t0:.1f}s, saved", flush=True)

    print("\nSWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
