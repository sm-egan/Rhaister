"""Sensitivity-modality baseline metrics: one facet per metric (MSE, MAE),
one bar per baseline.

Source: results_sensitivity.jsonl (one record per baseline). Data is
single-split for now, so each bar is a point estimate (no error bars).

Usage:
    uv run --with plotnine --with pandas python figures/sensitivity_baselines.py --dataset EmeraldBay
    uv run --with plotnine --with pandas python figures/sensitivity_baselines.py --dataset prism
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_col,
    ggplot,
    labs,
    scale_fill_manual,
    scale_x_discrete,
    scale_y_continuous,
    theme,
)

from style import BLUE_500, GROUP_COLOR, figure_theme, save_figure

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results_sensitivity.jsonl"

# Display order, bottom -> top on the y-axis (worst expected on top).
# Half-sample reference (replicate noise ceiling) sits at the bottom because
# it is a lower bound on achievable error.
METHOD_FROM_JSON = {
    "a_vs_b":         "Half-sample reference",
    "rhaister_v1":    "Rhaister",
    "svd_residual":   "SVD residual",
    "additive":       "Additive",
    "treatment_mean": "Perturbation Mean",
    "cell_mean":      "Context Mean",
    "global_mean":    "Global Mean",
}
METHOD_ORDER = list(METHOD_FROM_JSON.values())

# SVD residual is a new method group not yet in style.GROUP_COLOR. Promote
# the local palette here; if it appears in a second figure, lift to style.py.
LOCAL_GROUP_COLOR = {
    **GROUP_COLOR,
    "low-rank": BLUE_500,
}
METHOD_GROUP_LOCAL = {
    "Global Mean":       "global mean",
    "Context Mean":      "marginal mean",
    "Perturbation Mean": "marginal mean",
    "Additive":          "additive",
    "SVD residual":      "low-rank",
    "Rhaister":          "Rhaister",
    "Half-sample reference": "replicate ceiling",
}

METRIC_FROM_JSON = {"mse": "MSE", "mae": "MAE", "r2": "R²", "pearson": "Pearson r"}
METRIC_ORDER = list(METRIC_FROM_JSON.values())


def load_long(split: str) -> pd.DataFrame:
    rows = []
    with RESULTS.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] != split:
                continue
            method = METHOD_FROM_JSON.get(rec["baseline"])
            if method is None:
                continue
            for raw, pretty in METRIC_FROM_JSON.items():
                if raw not in rec:
                    continue  # tolerate older records that lacked this metric
                rows.append({"method": method, "metric": pretty, "value": float(rec[raw])})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No records for split={split} in {RESULTS}")
    # Append-only file: keep the most recent record per (method, metric).
    df = df.drop_duplicates(["method", "metric"], keep="last").reset_index(drop=True)
    df["group"] = df["method"].map(METHOD_GROUP_LOCAL)
    # Order is set in make_plot from the methods actually present.
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    return df


def make_plot(df: pd.DataFrame) -> ggplot:
    methods_present = [m for m in METHOD_ORDER if m in set(df["method"])]
    df = df[df["method"].isin(methods_present)].copy()
    df["method"] = pd.Categorical(df["method"], categories=methods_present, ordered=True)
    palette = {g: LOCAL_GROUP_COLOR[g] for g in df["group"].unique()}
    tick_colors = [LOCAL_GROUP_COLOR[METHOD_GROUP_LOCAL[m]] for m in methods_present]
    return (
        ggplot(df, aes(x="method", y="value", fill="group"))
        + geom_col(width=0.7)
        + facet_wrap("~ metric", ncol=2, scales="free", dir="h")
        + coord_flip()
        + scale_x_discrete(limits=methods_present)
        + scale_y_continuous(expand=(0, 0, 0.05, 0))
        + scale_fill_manual(values=palette, guide=None)
        + labs(x="", y="metric value")
        + figure_theme()
        + theme(
            axis_text_y=element_text(color=tick_colors),
            legend_position="none",
        )
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="e.g. EmeraldBay, prism")
    p.add_argument("--split", default="split_0", help="Split directory name")
    args = p.parse_args()
    split = f"{args.dataset}/{args.split}"
    df = load_long(split)
    save_figure(make_plot(df), f"{args.dataset}_baselines", size=(7, 4.5))


if __name__ == "__main__":
    main()
