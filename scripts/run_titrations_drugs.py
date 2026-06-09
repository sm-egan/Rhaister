"""Drug titration: vary the number of fewshot training drugs on the holdout cell.

Uses predefined split files at splits/tahoe/{h}_holdout/titration/titration_{L}_drugs.toml,
which encode a fewshot/test partition of the holdout cell's treatments for
each titration level L. (The canonical-path symlinks at
splits/tahoe/{h}_holdout_titration_{L}/ are broken on this machine, so we
monkey-patch load_split to resolve the names directly.)

The titration test set shrinks as L grows (drugs move from test → fewshot
training), so per-level metrics over the native test set conflate "more
training data" with "smaller/different evaluation set". To isolate the
training-data effect, we additionally evaluate on a FIXED 780-condition test
set defined by splits/tahoe/{h}_holdout/generalization_converted_cell_lines_3b.toml,
which is contained in every titration test set. Output records both
fixed-set ("fixed/...") and full-test-set ("full/...") metrics per level.

Usage:
    uv run python scripts/run_titrations_drugs.py [--wandb]
    uv run python scripts/run_titrations_drugs.py --levels 60 --append   # incremental
"""
import json
import os
import re
import sys
import tomllib


import numpy as np
from rhaister import prepare_combined
from rhaister import train
from rhaister.prepare_combined import evaluate, parse_split_name, _cache_path

_titration_re = re.compile(r"^tahoe_(\d+)_holdout_titration_(\d+)$")
_original_load_split = prepare_combined.load_split


def _patched_load_split(split_name):
    m = _titration_re.match(split_name)
    if not m:
        return _original_load_split(split_name)
    h, L = m.groups()
    toml_path = os.path.join(
        prepare_combined.SPLITS_DIR,
        "tahoe", f"{h}_holdout", "titration", f"titration_{L}_drugs.toml",
    )
    with open(toml_path, "rb") as f:
        config = tomllib.load(f)
    holdout_cells, test_treatments = [], {}
    for key, v in config["fewshot"].items():
        cell = key.split(".")[-1]
        holdout_cells.append(cell)
        test_treatments[cell] = set(v["test"])
    return {"holdout_cells": holdout_cells, "test_treatments": test_treatments}


prepare_combined.load_split = _patched_load_split


def _load_fixed_test_set(holdout):
    """Return {cell_line: set(treatments)} from the generalization split TOML."""
    path = os.path.join(
        prepare_combined.SPLITS_DIR, "tahoe", f"{holdout}_holdout",
        "generalization_converted_cell_lines_3b.toml",
    )
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    fixed = {}
    for key, v in cfg["fewshot"].items():
        cell = key.split(".")[-1]
        fixed[cell] = set(v["test"])
    return fixed


def _wrap_evaluator(data, split_name, fixed):
    """Replace data["evaluate_test"] with a wrapper that returns both full-test-set
    and fixed-subset metrics in one call. Reads Y_test/P_test/D_test from the
    on-disk cache (they're not exposed in the data dict directly).
    """
    dataset, split = parse_split_name(split_name)
    cp = _cache_path(dataset, split)
    Y_test = np.load(os.path.join(cp, "Y_test.npy"))
    P_test = np.load(os.path.join(cp, "P_test.npy"))
    D_test = np.load(os.path.join(cp, "D_test.npy"))
    F_test = np.asarray(data["F_test"])
    test_cells = np.asarray(data["test_cells"])
    test_treatments = np.asarray(data["test_treatments"])
    gene_cols = data["gene_cols"]

    mask = np.array([
        t in fixed.get(c, set()) for c, t in zip(test_cells, test_treatments)
    ])
    n_full, n_fixed = len(test_cells), int(mask.sum())
    if n_fixed == 0:
        raise ValueError(
            f"No test rows match the fixed generalization set for {split_name}; "
            f"check holdout/cell mapping"
        )

    Y_t, P_t, D_t, F_t = Y_test[mask], P_test[mask], D_test[mask], F_test[mask]
    cells_t, treats_t = test_cells[mask], test_treatments[mask]

    called = [False]

    def evaluate_test(Y_pred_fc, D_pred=None, F_pred=None, P_pred=None,
                       compute_discrimination=False):
        if called[0]:
            raise RuntimeError("evaluate_test() was already called.")
        called[0] = True
        Y_pred = np.asarray(Y_pred_fc)
        D_p = np.asarray(D_pred) if D_pred is not None else None
        F_p = np.asarray(F_pred) if F_pred is not None else None
        P_p = np.asarray(P_pred) if P_pred is not None else None

        full = evaluate(
            Y_test, Y_pred, P_test, P_p, D_test, D_p, F_test, F_p,
            test_cells, test_treatments, gene_cols,
            compute_discrimination=compute_discrimination,
        )
        fixed_m = evaluate(
            Y_t, Y_pred[mask],
            P_t, P_p[mask] if P_p is not None else None,
            D_t, D_p[mask] if D_p is not None else None,
            F_t, F_p[mask] if F_p is not None else None,
            cells_t, treats_t, gene_cols,
            compute_discrimination=compute_discrimination,
        )
        out = {f"full/{k}": v for k, v in full.items()}
        out.update({f"fixed/{k}": v for k, v in fixed_m.items()})
        out["full/n_test"] = n_full
        out["fixed/n_test"] = n_fixed
        return out

    data["evaluate_test"] = evaluate_test


HOLDOUTS = [6, 7, 8, 9]
LEVELS = [1, 3, 6, 10, 20, 30, 60]
BASE_METRICS = [
    "pdex_static/pearson_delta_mean",
    "pdex_static/auprc_p05",
    "state/pearson_delta_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
    "state/pr_auc_mean",
    "state/spearman_lfc_sig_mean",
    "state/discrimination_mean",
]
OUTPUT = "titration_drugs_results.json"


def _parse_levels_arg(argv):
    if "--levels" not in argv:
        return list(LEVELS)
    i = argv.index("--levels")
    vals = []
    for tok in argv[i + 1:]:
        if tok.startswith("--"):
            break
        vals.append(int(tok))
    if not vals:
        raise SystemExit("--levels requires at least one integer")
    return vals


def main():
    use_wandb = "--wandb" in sys.argv
    append = "--append" in sys.argv
    levels = _parse_levels_arg(sys.argv)
    if use_wandb:
        import wandb

    # results[scope][metric][holdout][level] = value
    if append and os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            prev = json.load(f)
        results = prev["metrics"]
        n_test_records = prev["n_test"]
        merged_levels = sorted(set(prev.get("levels", [])) | set(levels))
        # Ensure every holdout key exists in case the previous file was shape-incomplete
        for scope in ("fixed", "full"):
            for m in BASE_METRICS:
                for h in HOLDOUTS:
                    results[scope][m].setdefault(str(h), {})
                n_test_records[scope].setdefault(str(h), {})
        print(f"--append: loaded {OUTPUT}; will merge levels={levels} into existing {prev.get('levels', [])}")
    else:
        results = {
            scope: {m: {str(h): {} for h in HOLDOUTS} for m in BASE_METRICS}
            for scope in ("fixed", "full")
        }
        n_test_records = {scope: {str(h): {} for h in HOLDOUTS} for scope in ("fixed", "full")}
        merged_levels = list(levels)

    for h in HOLDOUTS:
        fixed = _load_fixed_test_set(h)
        for L in levels:
            split_name = f"tahoe_{h}_holdout_titration_{L}"
            print(f"\n{'='*60}\n{split_name}\n{'='*60}")

            data = prepare_combined.prepare_all(split_name)
            _wrap_evaluator(data, split_name, fixed)

            metrics = train.train_and_evaluate(
                split_name=split_name, log=False, data=data,
                compute_discrimination=True,
            )

            row = {}
            for scope in ("fixed", "full"):
                for m in BASE_METRICS:
                    key = f"{scope}/{m}"
                    v = float(metrics[key])
                    results[scope][m][str(h)][str(L)] = v
                    row[key] = v
                n_test_records[scope][str(h)][str(L)] = int(metrics[f"{scope}/n_test"])
            print(
                f"  h={h} L={L:>3}  n_fixed={metrics['fixed/n_test']}  "
                f"n_full={metrics['full/n_test']}"
            )
            for m in BASE_METRICS:
                print(f"    {m:<40s} fixed={row[f'fixed/{m}']:.4f}  full={row[f'full/{m}']:.4f}")

            if use_wandb:
                wandb.init(
                    project="perturbation-eval",
                    job_type="rhaister_titration_drugs",
                    name=f"titration_drugs_h{h}_L{L}",
                    config={"holdout": h, "level": L, "axis": "drugs"},
                    reinit=True,
                )
                wandb.log(row)
                wandb.finish()

    out = {
        "experiment": "drug_titration",
        "description": (
            "Drug titration over predefined splits. Each split moves LEVEL drugs "
            "from the holdout cell's test set into its fewshot training pool. "
            "Metrics are reported in two scopes: 'fixed' restricts evaluation to "
            "the 780-condition test set defined in "
            "splits/tahoe/{h}_holdout/generalization_converted_cell_lines_3b.toml "
            "(contained in every titration test set, so the eval set is constant "
            "across L); 'full' uses the full titration test set (changes with L)."
        ),
        "holdouts": HOLDOUTS,
        "levels": merged_levels,
        "n_test": n_test_records,
        "metrics": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUTPUT}")

    print(f"\n{'='*70}\nSUMMARY (fixed-set metrics)\n{'='*70}")
    for m in BASE_METRICS:
        print(f"\n{m}")
        print("  h \\ L  " + "  ".join(f"{L:>8}" for L in merged_levels))
        for h in HOLDOUTS:
            vals = "  ".join(
                f"{results['fixed'][m][str(h)].get(str(L), float('nan')):>8.4f}"
                for L in merged_levels
            )
            print(f"    {h}    {vals}")


if __name__ == "__main__":
    main()
