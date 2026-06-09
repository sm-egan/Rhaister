"""Multi-split R² figure for a sensitivity dataset (publication style).

Per-split values shown as small dots; mean across the 5 splits as a larger
diamond. SVD-residual is intentionally excluded; the Half-sample reference is
included where the dataset declares variants (RIFIVDU only).

Generates one figure per available subset:
  - full:    all baselines except Primary→Secondary, scored on the full test
             set (~380K rows for PRISM, ~155 rows for RIFIVDU per split).
  - matched: all baselines including Primary→Secondary, scored on the
             dose-matched subset (~44K rows per PRISM split). Logged by
             scripts/eval_dose_matched_subset.py.

Outputs:
    figures/{dataset}_r2_full.{pdf,png}     # main
    figures/{dataset}_r2_matched.{pdf,png}  # supplement (PRISM only)

Usage:
    uv run --with plotnine --with pandas python figures/sensitivity_r2.py --dataset EmeraldBay
    uv run --with plotnine --with pandas python figures/sensitivity_r2.py --dataset prism
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_label,
    geom_point,
    geom_segment,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from style import GROUP_COLOR, figure_theme, save_figure

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results_sensitivity.jsonl"
N_SPLITS = 5

# Display order, bottom -> top on the y-axis. SVD residual intentionally
# omitted. Order is shared across full and matched figures so methods land in
# the same vertical position regardless of which subset is plotted.
METHOD_FROM_JSON = {
    "global_mean":          "Global Mean",
    "cell_mean":            "Context Mean",
    "treatment_mean":       "Perturbation Mean",
    "additive":             "Additive",
    "rhaister_v1":          "Rhaister",
    "rhaister_v1_feat_all": "Rhaister (with features)",
    "primary_to_secondary": "Primary→Secondary",
    "adjacent_dose":        "Adjacent dose",
    "a_vs_b":               "Half-sample reference",
}
METHOD_ORDER = list(METHOD_FROM_JSON.values())

METHOD_GROUP = {
    "Global Mean":       "global mean",
    "Context Mean":      "marginal mean",
    "Perturbation Mean": "marginal mean",
    "Additive":          "additive",
    "Adjacent dose":     "replicate ceiling",
    "Primary→Secondary": "replicate ceiling",
    "Rhaister":           "Rhaister",
    "Rhaister (with features)": "Rhaister",
    "Half-sample reference": "replicate ceiling",
}

# Per-dataset label overrides (keyed by the default display name). RIFIVDU is
# the only dataset with both feature sources, so it spells out which Rhaister
# variant is which; elsewhere "Rhaister" is the single sensitivity model.
DATASET_LABELS = {
    "rifivdu": {
        "Rhaister":                 "Rhaister (sensitivity only)",
        "Rhaister (with features)": "Rhaister (sensitivity + tx)",
    },
}


def _relabel(dataset: str):
    """Return (method_from_json, method_order, method_group) with the dataset's
    label overrides applied. Renames preserve y-axis position and color group."""
    overrides = DATASET_LABELS.get(dataset, {})
    ren = lambda name: overrides.get(name, name)  # noqa: E731
    mfj = {k: ren(v) for k, v in METHOD_FROM_JSON.items()}
    order = [ren(m) for m in METHOD_ORDER]
    group = {ren(k): v for k, v in METHOD_GROUP.items()}
    return mfj, order, group


def load_long(dataset: str) -> pd.DataFrame:
    method_from_json, _, _ = _relabel(dataset)
    splits = {f"{dataset}/split_{i}" for i in range(N_SPLITS)}
    rows = []
    with RESULTS.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("split") not in splits or "r2" not in rec:
                continue
            method = method_from_json.get(rec["baseline"])
            if method is None:
                continue
            subset = rec.get("subset")
            if subset is None:
                # Legacy records without a subset field. Primary→Secondary is
                # always a matched-subset measurement by construction; everything
                # else without a label is the full test set.
                subset = "matched" if rec["baseline"] == "primary_to_secondary" else "full"
            rows.append({
                "method": method,
                "split": rec["split"],
                "subset": subset,
                "r2": float(rec["r2"]),
                "_ts": rec.get("timestamp", ""),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No records for {dataset}/split_{{0..{N_SPLITS-1}}} in {RESULTS}")
    df = df.sort_values("_ts").drop_duplicates(["method", "split", "subset"], keep="last")
    return df.drop(columns=["_ts"]).reset_index(drop=True)


def make_plot(df: pd.DataFrame, method_order=METHOD_ORDER,
              method_group=METHOD_GROUP) -> ggplot:
    _seen = set()
    methods_present = []
    for m in method_order:
        if m in set(df["method"]) and m not in _seen:
            methods_present.append(m)
            _seen.add(m)
    method_idx = {m: i + 1 for i, m in enumerate(methods_present)}
    splits = sorted(df["split"].unique())
    n_h = len(splits)

    span = 0.55
    if n_h > 1:
        offsets = [span * (i - (n_h - 1) / 2) / (n_h - 1) for i in range(n_h)]
    else:
        offsets = [0.0]
    split_idx = {s: i for i, s in enumerate(splits)}

    df = df.copy()
    df["y"] = [method_idx[m] + offsets[split_idx[s]] for m, s in zip(df["method"], df["split"])]
    df["group"] = df["method"].map(method_group)

    means = df.groupby("method", observed=True)["r2"].mean().reset_index()
    means["y"] = means["method"].map(method_idx).astype(float)
    means["group"] = means["method"].map(method_group)

    palette = {g: GROUP_COLOR[g] for g in set(df["group"])}
    tick_colors = [GROUP_COLOR[method_group[m]] for m in methods_present]

    r2_min = float(min(0.0, df["r2"].min())) - 0.05
    r2_max = max(1.0, float(df["r2"].max())) + 0.02
    # 0.5-step breaks keep the x-axis uncluttered at single-column (83mm) width.
    breaks = [b for b in (-0.5, 0.0, 0.5, 1.0) if r2_min <= b <= r2_max]

    # Mean-value text per row, matching main_metrics_zeroshot: label sits just
    # right of the row's rightmost point; rows whose points reach the right edge
    # fall back to a label centered above the diamond.
    txt_size = 6
    pad = 0.02
    lbl_dy = 0.34
    right_fits_cutoff = r2_max - 0.12
    # +0.0 collapses a rounded "-0.00" to "0.00"; lstrip drops the leading zero
    # on positives (".31"), matching main_metrics_zeroshot.
    means["label"] = means["r2"].map(
        lambda v: "" if pd.isna(v) else f"{round(v, 2) + 0.0:.2f}".lstrip("0")
    )
    row_max = df.groupby("method", observed=True)["r2"].max().to_dict()
    means["row_max"] = [max(v, row_max.get(m, v)) for m, v in zip(means["method"], means["r2"])]
    means["lbl_x"] = means["row_max"] + pad
    means["lbl_y"] = means["y"] + lbl_dy
    labelled = means[means["label"] != ""]
    means_right = labelled[labelled["row_max"] <= right_fits_cutoff]
    means_above = labelled[labelled["row_max"] > right_fits_cutoff]

    return (
        ggplot()
        + geom_vline(xintercept=0, color="#bcbcbc", size=0.3)
        + geom_segment(
            means, aes(x=0, xend="r2", y="y", yend="y", color="group"), size=0.5,
        )
        + geom_point(
            df, aes(x="r2", y="y", color="group", fill="group"), size=1.4,
        )
        + geom_point(
            means, aes(x="r2", y="y", fill="group"),
            size=3.4, shape="D", color="white", stroke=0.6,
        )
        + geom_label(
            means_right, aes(x="lbl_x", y="y", label="label", color="group"),
            size=txt_size, va="center", ha="left",
            fill="white", label_size=0, label_padding=0.1,
        )
        + geom_text(
            means_above, aes(x="r2", y="lbl_y", label="label", color="group"),
            size=txt_size, va="bottom", ha="center",
        )
        + scale_x_continuous(limits=(r2_min, r2_max), breaks=breaks)
        + scale_y_continuous(
            breaks=list(method_idx.values()),
            labels=methods_present,
            limits=(0.5, len(methods_present) + 0.5),
        )
        + scale_color_manual(values=palette, guide=None)
        + scale_fill_manual(values=palette, guide=None)
        + labs(x="R²", y="")
        # base_size=10 gives the 10pt axis title; tick labels are pinned to 8pt
        # explicitly (value labels are 6pt via txt_size) to sit with the paper's
        # single-column body text.
        + figure_theme(base_size=10)
        + theme(
            axis_title=element_text(size=10),
            axis_text=element_text(size=8),
            axis_text_y=element_text(size=8, color=tick_colors),
            legend_position="none",
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="EmeraldBay or prism")
    p.add_argument("--subset", choices=["full", "matched"],
                   help="Render only this subset (default: render all available)")
    args = p.parse_args()

    df = load_long(args.dataset)
    _, method_order, method_group = _relabel(args.dataset)
    available = sorted(set(df["subset"]))
    targets = [args.subset] if args.subset else available

    for subset in targets:
        df_sub = df[df["subset"] == subset].copy()
        # Primary→Secondary is only measurable on the dose-matched subset (~12%
        # of rows), but it's a reference ceiling we want shown alongside the
        # other ceilings in the main full figure too — so graft its matched rows
        # in. Its R² is therefore on a different subset than the full-coverage
        # rows; that's inherent to the primary screen's dose coverage.
        if subset == "full":
            primary = df[(df["method"] == "Primary→Secondary") & (df["subset"] == "matched")]
            df_sub = pd.concat([df_sub, primary], ignore_index=True)
        if df_sub.empty:
            print(f"skipping subset={subset}: no records")
            continue
        # Single-column width: 83mm = 3.27in.
        save_figure(make_plot(df_sub, method_order, method_group),
                    f"{args.dataset}_r2_{subset}", size=(3.27, 2.6))


if __name__ == "__main__":
    main()
