"""Main-metrics figure for the zeroshot variant.

Compares the zeroshot drug-specific γ_{t,g} diagonal model ("Rhaister
Zeroshot") against:
- Panel-free statistical baselines (Global Mean, Perturbation Mean) computed
  from zeroshot-eligible train data only (excluding the held-out cell's
  panel rows).
- An intermediate zeroshot model (Shared γ_g across drugs).
- Panel-aware ("Fewshot") models that DO use the held-out cell's training
  drugs (Rhaister Fewshot, STATE Fewshot) — shown for context as a ceiling
  for "what panel access buys you."
- Half-sample reference noise ceiling (independent of training).

Per-holdout values as dots; mean across the 5 tahoe holdouts as a diamond.
Zeroshot diagonal numbers come from titration_cells_zeroshot_results.json
(L=45). Other zeroshot baselines come from zeroshot_baselines_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_label,
    geom_point,
    geom_segment,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from style import (
    GRAY_600,
    GROUP_COLOR,
    GROUP_ORDER,
    METHOD_GROUP,
    METRIC_FROM_JSON,
    METRIC_ORDER,
    METRIC_SHORT,
    METRIC_SHORT_ORDER,
    MIN_LINE_SIZE,
    figure_theme,
    save_figure,
)

REPO = Path(__file__).parent.parent
RHAISTER = REPO / "rhaister_results.json"
A_VS_B = REPO / "a_vs_b_results.tahoe_only.json"
STATE = REPO / "state_results.json"
ZEROSHOT_TITR = REPO / "titration_cells_zeroshot_results.json"
ZEROSHOT_BASE = REPO / "zeroshot_baselines_results.json"
STATE_ZEROSHOT = REPO / "state_zeroshot_results.json"

RHAISTER_RUN = "eval_rhaister_simplified_model"
SPLITS = ["tahoe_5_holdout", "tahoe_6_holdout", "tahoe_7_holdout",
          "tahoe_8_holdout", "tahoe_9_holdout"]
HOLDOUTS = [5, 6, 7, 8, 9]

# Bottom -> top. Zeroshot rows are grouped together; fewshot rows above show
# the ceiling that panel access provides.
METHOD_ORDER = [
    "Global Mean Zeroshot",
    "Perturbation Mean Zeroshot",
    "Rhaister Zeroshot",
    "STATE Zeroshot",
    "Half-sample reference",
]
# Display labels. The figure is all-zeroshot by construction, so "0-shot" is
# dropped as redundant; only the few-shot ceiling rows stay explicitly marked.
METHOD_LABEL = {
    "Global Mean Zeroshot":          "Global Mean",
    "Perturbation Mean Zeroshot":    "Perturbation Mean",
    "Rhaister Zeroshot Shared":      "Rhaister (shared γ)",
    "Rhaister Zeroshot":             "Rhaister-O",
    "STATE Zeroshot":                "STATE",
    "STATE Fewshot":                 "STATE Few-shot",
    "Rhaister Fewshot":              "Rhaister Few-shot",
    "Half-sample reference":         "Half-sample reference",
}
METHOD_SHORT = {
    "Global Mean Zeroshot":          "Global Mean",
    "Perturbation Mean Zeroshot":    "Pert. Mean",
    "Rhaister Zeroshot Shared":      "Rhaister (shared γ)",
    "Rhaister Zeroshot":             "Rhaister-O",
    "STATE Zeroshot":                "STATE",
    "STATE Fewshot":                 "STATE few-shot",
    "Rhaister Fewshot":              "Rhaister few-shot",
    "Half-sample reference":         "Half-sample",
}


def _zeroshot_diag_rows() -> list[dict]:
    """Headline drug-specific γ_{t,g} model: pull L=45 from the titration sweep."""
    data = json.loads(ZEROSHOT_TITR.read_text())["metrics"]
    rows = []
    for json_key, metric_pretty in METRIC_FROM_JSON.items():
        if json_key not in data:
            continue
        for h in HOLDOUTS:
            v = data[json_key].get(str(h), {}).get("45")
            if v is None:
                continue
            rows.append({
                "method": "Rhaister Zeroshot",
                "holdout": f"tahoe_{h}_holdout",
                "metric": metric_pretty,
                "value": float(v),
            })
    return rows


def _zeroshot_baseline_rows() -> list[dict]:
    """Global Mean Zeroshot, Perturbation Mean Zeroshot, and Shared γ from the
    dedicated baselines script."""
    if not ZEROSHOT_BASE.exists():
        return []
    data = json.loads(ZEROSHOT_BASE.read_text())["metrics_per_baseline"]
    rows = []
    for baseline_name, per_h in data.items():
        for h_str, row in per_h.items():
            split = f"tahoe_{h_str}_holdout"
            for json_key, metric_pretty in METRIC_FROM_JSON.items():
                if json_key not in row:
                    continue
                rows.append({
                    "method": baseline_name,
                    "holdout": split,
                    "metric": metric_pretty,
                    "value": float(row[json_key]),
                })
    return rows


def _rhaister_fewshot_rows() -> list[dict]:
    """Rhaister with panel data (i.e. the original few-shot model). Relabeled
    from "Rhaister" -> "Rhaister Fewshot" for the zeroshot figure to make the
    panel-access asymmetry explicit."""
    data = json.loads(RHAISTER.read_text())[RHAISTER_RUN]
    rows = []
    for split in SPLITS:
        metrics = data[split]
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            rows.append({
                "method": "Rhaister Fewshot",
                "holdout": split,
                "metric": metric_pretty,
                "value": float(metrics[json_key]),
            })
    return rows


def _state_fewshot_rows() -> list[dict]:
    data = json.loads(STATE.read_text())
    rows = []
    for split in SPLITS:
        metrics = data[split]["STATE"]
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            if json_key not in metrics:
                continue
            rows.append({
                "method": "STATE Fewshot",
                "holdout": split,
                "metric": metric_pretty,
                "value": float(metrics[json_key]),
            })
    return rows


def _state_zeroshot_rows() -> list[dict]:
    """Partial: read whatever (split, metric) values are in state_zeroshot_results.json.
    Empty when more data isn't available yet."""
    if not STATE_ZEROSHOT.exists():
        return []
    data = json.loads(STATE_ZEROSHOT.read_text())
    rows = []
    for split in SPLITS:
        metrics = data.get(split, {})
        for json_key, metric_pretty in METRIC_FROM_JSON.items():
            v = metrics.get(json_key)
            if v is None:
                continue
            rows.append({
                "method": "STATE Zeroshot",
                "holdout": split,
                "metric": metric_pretty,
                "value": float(v),
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


def load_long() -> pd.DataFrame:
    rows = (
        _zeroshot_diag_rows()
        + _zeroshot_baseline_rows()
        + _rhaister_fewshot_rows()
        + _state_fewshot_rows()
        + _state_zeroshot_rows()
        + _a_vs_b_rows()
    )
    df = pd.DataFrame(rows)
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df


def make_plot(df: pd.DataFrame, ncol: int = 3, compact: bool = False) -> ggplot:
    method_idx = {m: i + 1 for i, m in enumerate(METHOD_ORDER)}
    holdout_idx = {h: i for i, h in enumerate(SPLITS)}
    n_h = len(SPLITS)
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
    labels = [METHOD_SHORT[m] if compact else METHOD_LABEL[m] for m in METHOD_ORDER]
    base = 8 if compact else 10
    pt_dot = 1.2
    pt_mean = 2.8

    # Print each model's mean value on its own row, just to the right of the
    # row's rightmost point, so the value sits with the method it describes.
    # Rows whose points reach too far right (no room for a right label) fall
    # back to a label centered above the diamond.
    txt_size = base * 0.7
    lbl_dy = 0.34
    pad = 0.045
    right_fits_cutoff = 0.90
    means["label"] = means["value"].map(
        lambda v: "" if pd.isna(v) else f"{v:.2f}".lstrip("0")
    )
    row_max = df.groupby(["method", "metric"], observed=True)["value"].max().to_dict()
    means["row_max"] = [
        v if pd.isna(v) else max(v, row_max.get((m, me), v))
        for m, me, v in zip(means["method"], means["metric"], means["value"])
    ]
    means["lbl_x"] = means["row_max"] + pad
    means["lbl_y"] = means["y"] + lbl_dy

    if compact:
        df = df.copy()
        df["metric"] = df["metric"].astype(str).map(METRIC_SHORT)
        df["metric"] = pd.Categorical(df["metric"], categories=METRIC_SHORT_ORDER, ordered=True)
        means = means.copy()
        means["metric"] = means["metric"].astype(str).map(METRIC_SHORT)
        means["metric"] = pd.Categorical(means["metric"], categories=METRIC_SHORT_ORDER, ordered=True)

    # facet_wrap only draws the x-axis (ticks + labels) on the bottom row of
    # panels. Add bare reference tick marks to the upper-row panels so every
    # facet has an x scale to read against, while the numeric labels stay only
    # on the bottom row. Upper-row facets are everything except the last row.
    metric_order = METRIC_SHORT_ORDER if compact else METRIC_ORDER
    xbreaks = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_bot = 0.5
    tick_len = 0.16
    n_bottom = ((len(metric_order) - 1) % ncol) + 1
    upper_metrics = metric_order[: len(metric_order) - n_bottom]
    ticks = pd.DataFrame(
        [{"metric": m, "x": x} for m in upper_metrics for x in xbreaks]
    )
    ticks["metric"] = pd.Categorical(ticks["metric"], categories=metric_order, ordered=True)

    labelled = means[means["label"] != ""]
    means_right = labelled[labelled["row_max"] <= right_fits_cutoff]
    means_above = labelled[labelled["row_max"] > right_fits_cutoff]

    return (
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
        + geom_label(
            means_right,
            aes(x="lbl_x", y="y", label="label", color="group"),
            size=txt_size, va="center", ha="left",
            fill="white", label_size=0, label_padding=0.1,
        )
        + geom_text(
            means_above,
            aes(x="value", y="lbl_y", label="label", color="group"),
            size=txt_size, va="bottom", ha="center",
        )
        + geom_segment(
            ticks,
            aes(x="x", xend="x"),
            y=y_bot, yend=y_bot + tick_len,
            color=GRAY_600, size=MIN_LINE_SIZE,
        )
        + facet_wrap("~ metric", ncol=ncol)
        + scale_x_continuous(
            limits=(0, 1),
            breaks=[0, 0.25, 0.5, 0.75, 1.0],
            labels=[f"{b:.2f}".lstrip("0") for b in [0, 0.25, 0.5, 0.75, 1.0]],
        )
        + scale_y_continuous(
            breaks=list(method_idx.values()),
            labels=labels,
            limits=(0.5, len(METHOD_ORDER) + 0.9),
        )
        + scale_color_manual(values=palette, limits=GROUP_ORDER, guide=None)
        + scale_fill_manual(values=palette, limits=GROUP_ORDER, guide=None)
        + labs(x="metric value", y="")
        + figure_theme(base_size=base)
        + theme(
            axis_text_y=element_text(
                color=[GROUP_COLOR[METHOD_GROUP[m]] for m in METHOD_ORDER],
                size=base - 1 if compact else None,
            ),
            axis_text_x=element_text(size=base - 1 if compact else None),
            strip_text=element_text(size=base if compact else None),
            legend_position="none",
            plot_margin_top=0.02 if compact else None,
        )
    )


def main() -> None:
    df = load_long()
    layouts = [
        (3, "",              False, lambda n: (9.0, 0.5 * n)),
        (2, "_3x2",          False, lambda n: (6.0, 0.8 * n)),
        (2, "_3x2_compact",  True,  lambda n: (3.27, 3.27)),
    ]
    n = len(METHOD_ORDER)
    for ncol, layout_suffix, compact, sizer in layouts:
        save_figure(
            make_plot(df, ncol=ncol, compact=compact),
            f"main_metrics_zeroshot{layout_suffix}",
            size=sizer(n),
        )


if __name__ == "__main__":
    main()
