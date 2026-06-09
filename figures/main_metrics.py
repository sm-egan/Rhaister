"""Main metrics figure: one facet per metric, one row per model/baseline.

Per-holdout values shown as small dots; mean across the 5 tahoe holdouts as a
larger diamond.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from style import (
    GROUP_COLOR,
    GROUP_ORDER,
    METHOD_GROUP,
    METHOD_SHORT,
    METRIC_FROM_JSON,
    METRIC_ORDER,
    METRIC_SHORT,
    METRIC_SHORT_ORDER,
    MIN_LINE_SIZE,
    add_mean_value_labels,
    figure_theme,
    save_figure,
)

REPO = Path(__file__).parent.parent
BASELINES = REPO / "baseline_results.json"
RHAISTER = REPO / "rhaister_results.json"
A_VS_B = REPO / "a_vs_b_results.tahoe_only.json"
PERMUTE = REPO / "label_permutation_ablation.json"
STATE = REPO / "state_results.json"

RHAISTER_RUN = "eval_rhaister_simplified_model"
SPLITS = ["tahoe_5_holdout", "tahoe_6_holdout", "tahoe_7_holdout",
          "tahoe_8_holdout", "tahoe_9_holdout"]

# Display names. Order is bottom -> top on the y-axis. The Permute* rows are
# added only in the "with permute" variant (see main()).
METHOD_ORDER = [
    "Global Mean",
    "Context Mean",
    "Perturbation Mean",
    "Additive",
    "STATE",
    "Rhaister",
    "Half-sample reference",
]
PERMUTE_METHODS = ["Permute Perturbation", "Permute Context"]


def _baseline_rows() -> list[dict]:
    data = json.loads(BASELINES.read_text())
    name_map = {
        "global_mean": "Global Mean",
        "cell_mean": "Context Mean",
        "treatment_mean": "Perturbation Mean",
        "additive": "Additive",
    }
    rows = []
    for split in SPLITS:
        for raw, pretty in name_map.items():
            metrics = data[split][raw]
            for json_key, metric_pretty in METRIC_FROM_JSON.items():
                rows.append({
                    "method": pretty,
                    "holdout": split,
                    "metric": metric_pretty,
                    "value": float(metrics[json_key]),
                })
    return rows


def _rhaister_rows() -> list[dict]:
    data = json.loads(RHAISTER.read_text())[RHAISTER_RUN]
    rows = []
    for split in SPLITS:
        metrics = data[split]
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            rows.append({
                "method": "Rhaister",
                "holdout": split,
                "metric": metric_pretty,
                "value": float(metrics[json_key]),
            })
    return rows


def _state_rows() -> list[dict]:
    # External evaluator logs five of the six State metrics; the missing
    # state/discrimination_mean simply yields no point for that facet.
    data = json.loads(STATE.read_text())
    rows = []
    for split in SPLITS:
        metrics = data[split]["STATE"]
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            if json_key not in metrics:
                continue
            rows.append({
                "method": "STATE",
                "holdout": split,
                "metric": metric_pretty,
                "value": float(metrics[json_key]),
            })
    return rows


def _a_vs_b_rows() -> list[dict]:
    data = json.loads(A_VS_B.read_text())["per_split"]
    rows = []
    for split in SPLITS:
        metrics = data[split]["mean"]
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            rows.append({
                "method": "Half-sample reference",
                "holdout": split,
                "metric": metric_pretty,
                "value": float(metrics[json_key]),
            })
    return rows


def _permute_rows() -> list[dict]:
    data = json.loads(PERMUTE.read_text())["modes"]
    name_map = {"treatments": "Permute Perturbation", "cells": "Permute Context"}
    rows = []
    for raw, pretty in name_map.items():
        metrics = data[raw]["metrics"]
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            per_split = metrics[json_key]["per_split"]
            for split in SPLITS:
                rows.append({
                    "method": pretty,
                    "holdout": split,
                    "metric": metric_pretty,
                    "value": float(per_split[split]),
                })
    return rows


def load_long(include_permute: bool) -> pd.DataFrame:
    rows = (
        _baseline_rows()
        + _rhaister_rows()
        + _state_rows()
        + _a_vs_b_rows()
    )
    if include_permute:
        rows += _permute_rows()
    methods = METHOD_ORDER + (PERMUTE_METHODS if include_permute else [])
    df = pd.DataFrame(rows)
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=methods, ordered=True)
    return df


def make_plot(
    df: pd.DataFrame,
    methods: list[str],
    ncol: int = 3,
    small_fonts: bool = False,
    short_labels: bool = False,
) -> ggplot:
    """Lollipop facets, one per metric.

    ``small_fonts`` selects the compact base-8 font scheme (axis title 8,
    strip 8, axis text 7, value labels 5.6) used by the smaller companion
    panels; ``short_labels`` swaps the y-tick method names and metric strip
    titles for their abbreviated forms. The two are independent so the
    full-width hero can use small fonts while keeping its full-length labels.
    """
    method_idx = {m: i + 1 for i, m in enumerate(methods)}
    holdout_idx = {h: i for i, h in enumerate(SPLITS)}
    n_h = len(SPLITS)
    # Centered offsets in (-0.5, 0.5) range, leaving margin between rows.
    span = 0.55
    offsets = [span * (i - (n_h - 1) / 2) / (n_h - 1) for i in range(n_h)]

    df = df.copy()
    df["y"] = [
        method_idx[m] + offsets[holdout_idx[h]]
        for m, h in zip(df["method"], df["holdout"])
    ]
    df["group"] = df["method"].astype(str).map(METHOD_GROUP)
    df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)

    means = (
        df.groupby(["method", "metric"], observed=True)["value"]
          .mean()
          .reset_index()
    )
    means["y"] = means["method"].astype(str).map(method_idx).astype(float)
    means["group"] = means["method"].astype(str).map(METHOD_GROUP)
    means["group"] = pd.Categorical(means["group"], categories=GROUP_ORDER, ordered=True)

    palette = [GROUP_COLOR[g] for g in GROUP_ORDER]
    labels = [METHOD_SHORT[m] for m in methods] if short_labels else methods
    base = 8 if small_fonts else 10
    pt_dot = 1.2
    pt_mean = 2.8

    # Per-row mean value labels; placement handled by add_mean_value_labels.
    txt_size = base * 0.7
    right_fits_cutoff = 0.90

    if short_labels:
        # Remap metric labels to their shorter forms so the strip titles fit.
        df = df.copy()
        df["metric"] = df["metric"].astype(str).map(METRIC_SHORT)
        df["metric"] = pd.Categorical(df["metric"], categories=METRIC_SHORT_ORDER, ordered=True)
        means = means.copy()
        means["metric"] = means["metric"].astype(str).map(METRIC_SHORT)
        means["metric"] = pd.Categorical(means["metric"], categories=METRIC_SHORT_ORDER, ordered=True)

    plot = (
        ggplot()
        + geom_segment(
            means,
            aes(x=0, xend="value", y="y", yend="y", color="group"),
            size=MIN_LINE_SIZE,
        )
        + geom_point(
            df,
            aes(x="value", y="y", color="group", fill="group"),
            size=pt_dot,
        )
        + geom_point(
            means,
            aes(x="value", y="y", fill="group"),
            size=pt_mean, shape="D", color="white", stroke=MIN_LINE_SIZE,
        )
    )
    # top_above=False: high rows (incl. the top one) label to the left, not
    # above the diamond. That frees us from the extra top headroom an "above"
    # label needs, so the strip title can sit close to its own facet.
    plot = add_mean_value_labels(
        plot, means, df, methods[-1],
        txt_size=txt_size, cutoff=right_fits_cutoff, top_above=False,
    )
    return (
        plot
        + facet_wrap("~ metric", ncol=ncol)
        + scale_x_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + scale_y_continuous(
            breaks=list(method_idx.values()),
            labels=labels,
            limits=(0.5, len(methods) + 0.5),
        )
        + scale_color_manual(values=palette, limits=GROUP_ORDER, guide=None)
        + scale_fill_manual(values=palette, limits=GROUP_ORDER, guide=None)
        + labs(x="metric value", y="")
        + figure_theme(base_size=base)
        + theme(
            panel_spacing_y=0.025,
            axis_text_y=element_text(
                color=[GROUP_COLOR[METHOD_GROUP[m]] for m in methods],
                size=base - 1 if small_fonts else None,
            ),
            axis_text_x=element_text(size=base - 1 if small_fonts else None),
            strip_text=element_text(size=base if small_fonts else None),
            legend_position="none",
            plot_margin_top=0.02 if small_fonts else None,
        )
    )


def main() -> None:
    # Width-by-3-cols (default) and a portrait 3-rows-by-2-cols variant.
    # Height scales with method count so row spacing stays consistent across
    # the with/without-permute renders. The full-width hero uses the compact
    # base-8 font scheme (matching the smaller companion panels it sits beside)
    # but keeps its full-length labels; only the 83 mm _3x2_compact variant
    # also abbreviates the labels.
    # Columns: (ncol, suffix, small_fonts, short_labels, sizer).
    layouts = [
        (3, "",              True,  False, lambda n: (6.535, 0.4 * n)),  # full-width 166 mm, 3 cols × 2 rows
        (2, "_3x2",          False, False, lambda n: (6.0, 0.8 * n)),    # 2 cols × 3 rows
        (2, "_3x2_compact",  True,  True,  lambda n: (3.27, 3.27)),      # 83 mm × 83 mm
    ]
    for ncol, layout_suffix, small_fonts, short_labels, sizer in layouts:
        for include_permute, permute_suffix in [(False, ""), (True, "_with_permute")]:
            if short_labels and include_permute:
                continue  # compact target only ships without permute rows
            methods = METHOD_ORDER + (PERMUTE_METHODS if include_permute else [])
            df = load_long(include_permute=include_permute)
            save_figure(
                make_plot(
                    df, methods, ncol=ncol,
                    small_fonts=small_fonts, short_labels=short_labels,
                ),
                f"main_metrics{permute_suffix}{layout_suffix}",
                size=sizer(len(methods)),
            )


if __name__ == "__main__":
    main()
