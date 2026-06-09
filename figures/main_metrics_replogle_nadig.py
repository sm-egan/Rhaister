"""Main metrics figure for replogle_nadig (4 fewshot splits, one per cell line).

Same layout as main_metrics.py: one facet per metric, one row per
model/baseline; per-split values shown as small dots, mean across the four
holdouts as a larger diamond. SVD-residual baseline is intentionally omitted —
report it separately when needed.
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
    col_width_in,
    figure_theme,
    save_figure,
)


REPO = Path(__file__).parent.parent
BASELINES = REPO / "baseline_results.json"
MODEL = REPO / f"model_results.json"
A_VS_B = REPO / "a_vs_b_results.json"
STATE = REPO / "state_results.json"

SPLITS = [f"replogle_nadig/split_{i}" for i in range(4)]
RHAISTER_RUN = "full_impute"

METHOD_ORDER = [
    "Global Mean",
    "Context Mean",
    "Perturbation Mean",
    "Additive",
    "STATE",
    "Rhaister",
    "Half-sample reference",
]


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
    data = json.loads(MODEL.read_text())
    rows = []
    for split in SPLITS:
        metrics = data[split][RHAISTER_RUN]
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
    data = json.loads(A_VS_B.read_text())["per_dataset"]["replogle_nadig"]["per_split"]
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
    rows = _baseline_rows() + _rhaister_rows() + _state_rows() + _a_vs_b_rows()
    df = pd.DataFrame(rows)
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df


def make_plot(
    df: pd.DataFrame,
    *,
    metrics: list[str] = METRIC_ORDER,
    ncol: int = 3,
    short: bool = True,
) -> ggplot:
    """Lollipop facets, one per metric.

    ``metrics`` selects/orders the facets by canonical (full) metric name;
    ``ncol`` sets the facet grid width; ``short`` toggles abbreviated strip
    labels (the dense six-metric supplement layout) vs full names (the
    two-row main-text subset).
    """
    method_idx = {m: i + 1 for i, m in enumerate(METHOD_ORDER)}
    holdout_idx = {h: i for i, h in enumerate(SPLITS)}
    n_h = len(SPLITS)
    span = 0.55
    offsets = [span * (i - (n_h - 1) / 2) / (n_h - 1) for i in range(n_h)]

    df = df[df["metric"].isin(metrics)].copy()
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

    base = 8
    metric_map = METRIC_SHORT if short else {m: m for m in METRIC_ORDER}
    metric_cats = [metric_map[m] for m in metrics]
    for frame in (df, means):
        frame["metric"] = frame["metric"].astype(str).map(metric_map)
        frame["metric"] = pd.Categorical(frame["metric"], categories=metric_cats, ordered=True)

    palette = [GROUP_COLOR[g] for g in GROUP_ORDER]
    labels = [METHOD_SHORT[m] for m in METHOD_ORDER]
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
            size=1.2,
        )
        + geom_point(
            means,
            aes(x="value", y="y", fill="group"),
            size=2.2, shape="D", color="white", stroke=MIN_LINE_SIZE,
        )
    )
    # Short panels have no vertical headroom, so high rows label left, not above.
    plot = add_mean_value_labels(
        plot, means, df, METHOD_ORDER[-1],
        txt_size=base * 0.7, cutoff=0.78, top_above=False,
    )
    return (
        plot
        + facet_wrap("~ metric", ncol=ncol)
        + scale_x_continuous(limits=(0, 1), breaks=[0, 0.5, 1.0])
        + scale_y_continuous(
            breaks=list(method_idx.values()),
            labels=labels,
            limits=(0.5, len(METHOD_ORDER) + 0.5),
        )
        + scale_color_manual(values=palette, limits=GROUP_ORDER, guide=None)
        + scale_fill_manual(values=palette, limits=GROUP_ORDER, guide=None)
        + labs(x="metric value", y="")
        + figure_theme(base_size=base)
        + theme(
            axis_text_y=element_text(
                color=[GROUP_COLOR[METHOD_GROUP[m]] for m in METHOD_ORDER],
                size=base - 1,
            ),
            axis_text_x=element_text(size=base - 1),
            strip_text=element_text(size=base),
            legend_position="none",
            plot_margin_top=0.02,
        )
    )


MAIN_METRICS = ["Pearson correlation Δ", "DE overlap"]


def main() -> None:
    df = load_long()
    # Supplement: all six metrics, dense 3×2 grid, short strip labels.
    save_figure(make_plot(df), "main_metrics_replogle_nadig", size=(3.272, 2.3))
    # Main text: Pearson Δ + DE overlap stacked as two row-facets, full names,
    # sized to 1/3 of the 166 mm two-column block so three panels tile across.
    save_figure(
        make_plot(df, metrics=MAIN_METRICS, ncol=1, short=False),
        "main_metrics_replogle_nadig_2metric",
        size=(col_width_in(1 / 3), 3.0),
    )


if __name__ == "__main__":
    main()
