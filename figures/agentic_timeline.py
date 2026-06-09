"""Agentic experimentation timeline: 5 State metrics across the State-tracked
subset of `train_and_evaluate` runs.

State metrics were added to the eval suite from `pre_prune_baseline` (Apr 20)
onward. Earlier records in results.jsonl logged pdex_static metrics only and
are excluded here. The figure plots each State metric and its running max
across the post-Apr-20 timeline, indexed by position within the subset.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    facet_wrap,
    geom_point,
    geom_segment,
    geom_step,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from style import GROUP_COLOR, METRIC_FROM_JSON, figure_theme, save_figure

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results.jsonl"

# Five State metrics actually present in results.jsonl.
STATE_METRICS = [
    "state/pearson_delta_mean",
    "state/pr_auc_mean",
    "state/spearman_lfc_sig_mean",
    "state/de_overlap_mean",
    "state/de_spearman_sig",
]
RUNTIME_LABEL = "Runtime (s)"
METRIC_ORDER = [METRIC_FROM_JSON[k] for k in STATE_METRICS] + [RUNTIME_LABEL]

# perturbai_s0 evaluates a different split (perturbai, not tahoe) so its
# state-metric values aren't comparable to the rest of the agentic run.
EXCLUDE_EXPERIMENTS = {"rhaister_perturbai_s0"}


def load_long() -> pd.DataFrame:
    """Load only records that have State metrics, indexed within that subset."""
    state_records = []
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if r["experiment"] in EXCLUDE_EXPERIMENTS:
            continue
        if "state/pearson_delta_mean" not in r.get("metrics", {}):
            continue
        state_records.append(r)

    rows = []
    for idx, r in enumerate(state_records, start=1):
        m = r["metrics"]
        decision = r.get("decision") or "none"
        for json_key in STATE_METRICS:
            rows.append({
                "experiment_idx": idx,
                "experiment": r["experiment"],
                "decision": decision,
                "metric": METRIC_FROM_JSON[json_key],
                "value": float(m[json_key]),
            })
        rt = r.get("runtime_seconds")
        if rt is not None:
            rows.append({
                "experiment_idx": idx,
                "experiment": r["experiment"],
                "decision": decision,
                "metric": RUNTIME_LABEL,
                "value": float(rt),
            })

    df = pd.DataFrame(rows)
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_ORDER, ordered=True)
    decisions = ["accepted", "rejected", "none"]
    df["decision"] = pd.Categorical(df["decision"], categories=decisions, ordered=True)
    df = df.sort_values(["metric", "experiment_idx"]).reset_index(drop=True)
    return df


def accepted_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Initial baseline (idx=1) + every accepted record, per metric.

    Stepping through these points traces the actual model state across the
    sweep — i.e. where each metric sat after each accepted change.
    """
    keep_idx = sorted(set(df.loc[df["decision"] == "accepted", "experiment_idx"]) | {1})
    return df[df["experiment_idx"].isin(keep_idx)].copy()


DECISION_COLOR = {
    "accepted": GROUP_COLOR["replicate ceiling"],  # green
    "rejected": "#999999",                          # gray
    "none":     "#cccccc",                          # light gray (re-evals/baseline)
}


def make_plot(df: pd.DataFrame) -> ggplot:
    line_color = GROUP_COLOR["Rhaister"]
    n = int(df["experiment_idx"].max())
    decision_levels = ["accepted", "rejected", "none"]
    palette = [DECISION_COLOR[d] for d in decision_levels]
    accepted = accepted_trajectory(df)

    # Annotate one accepted dot so the decision color is self-explanatory.
    # Use the PR-AUC panel: accepted dots cluster around 0.772 with the upper
    # third of the panel empty (~0.78-0.80), giving room for a label and a
    # short downward connector that doesn't overlap any other points.
    annot_metric = METRIC_FROM_JSON["state/pr_auc_mean"]
    annot_anchor = (
        df[(df["metric"] == annot_metric) & (df["decision"] == "accepted")]
        .iloc[-1]
    )
    metric_cat = pd.Categorical([annot_metric], categories=METRIC_ORDER, ordered=True)
    text_x = annot_anchor["experiment_idx"] - 4
    text_y = annot_anchor["value"] + 0.020
    label_df = pd.DataFrame({
        "experiment_idx": [text_x],
        "value": [text_y],
        "metric": metric_cat,
        "label": ["Accepted"],
    })
    segment_df = pd.DataFrame({
        "x": [text_x + 1.2],
        "xend": [annot_anchor["experiment_idx"] - 0.3],
        "y": [text_y],
        "yend": [annot_anchor["value"] + 0.003],
        "metric": metric_cat,
    })

    return (
        ggplot(df, aes(x="experiment_idx"))
        + geom_step(
            accepted,
            aes(x="experiment_idx", y="value"),
            color=line_color, size=0.5, alpha=0.7, direction="hv",
            inherit_aes=False,
        )
        + geom_point(
            aes(y="value", color="decision", fill="decision"),
            size=1.8, alpha=0.95,
        )
        + geom_segment(
            segment_df,
            aes(x="x", xend="xend", y="y", yend="yend"),
            color=DECISION_COLOR["accepted"],
            size=0.3,
            inherit_aes=False,
        )
        + geom_text(
            label_df,
            aes(x="experiment_idx", y="value", label="label"),
            color=DECISION_COLOR["accepted"],
            size=8,
            ha="left",
            va="bottom",
            inherit_aes=False,
        )
        + facet_wrap("~ metric", ncol=2, scales="free_y", dir="h")
        + scale_x_continuous(breaks=list(range(1, n + 1)), limits=(1, n))
        + scale_y_continuous()
        + scale_color_manual(values=palette, limits=decision_levels, guide=None)
        + scale_fill_manual(values=palette, limits=decision_levels, guide=None)
        + labs(x="Experiment number", y="metric value")
        + figure_theme()
        + theme(
            legend_position="none",
            axis_text_x=element_blank(),
        )
    )


def main() -> None:
    df = load_long()
    save_figure(make_plot(df), "agentic_timeline", size=(6.5, 6.5))


if __name__ == "__main__":
    main()
