"""Cell-line titration: one column of metrics, gain over single-context baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_parser = argparse.ArgumentParser()
_parser.add_argument("--align-right", action="store_true",
    help="Anchor curves to the largest L instead of L=1.")
_args, _ = _parser.parse_known_args()
ALIGN_RIGHT = _args.align_right
ALIGN_SUFFIX = "_align_right" if ALIGN_RIGHT else ""
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_log10,
    theme,
)

from style import (
    GRAY_400,
    INK,
    METRIC_FROM_JSON,
    METRIC_SHORT,
    METRIC_SHORT_ORDER,
    MIN_LINE_SIZE,
    figure_theme,
    save_figure,
)

REPO = Path(__file__).parent.parent
TITRATION = REPO / "titration_cells_results.json"


def load_long() -> pd.DataFrame:
    titration = json.loads(TITRATION.read_text())["metrics"]

    rows = []
    for json_key, pretty in METRIC_FROM_JSON.items():
        for holdout, by_L in titration.get(json_key, {}).items():
            for L, value in by_L.items():
                rows.append({
                    "metric": METRIC_SHORT[pretty],
                    "holdout": f"holdout {holdout}",
                    "L": int(L),
                    "value": float(value),
                })

    df = pd.DataFrame(rows)
    df["metric"] = pd.Categorical(df["metric"], categories=METRIC_SHORT_ORDER, ordered=True)
    df["holdout"] = pd.Categorical(
        df["holdout"],
        categories=sorted(df["holdout"].unique()),
        ordered=True,
    )
    return df


def relativize(df: pd.DataFrame, *, align_right: bool) -> pd.DataFrame:
    """Subtract each (metric, holdout) group's anchor value from every point.

    Anchor is the largest L when ``align_right`` is True, otherwise the smallest.
    """
    anchor_L = df["L"].max() if align_right else df["L"].min()
    baseline = (
        df[df["L"] == anchor_L]
        .set_index(["metric", "holdout"])["value"]
        .rename("baseline")
    )
    out = df.merge(baseline, on=["metric", "holdout"], how="left")
    out["gain"] = out["value"] - out["baseline"]
    # Right-anchored view reads as "% of all-context performance" (anchor = 100%);
    # left-anchored view stays an absolute improvement over the single-context baseline.
    out["pct"] = out["value"] / out["baseline"] * 100
    return out


def make_plot(df: pd.DataFrame) -> ggplot:
    breaks = sorted(df["L"].unique())
    y_col = "pct" if ALIGN_RIGHT else "gain"
    return (
        ggplot(df, aes(x="L", y=y_col, group="holdout"))
        + geom_hline(
            yintercept=100 if ALIGN_RIGHT else 0,
            color=GRAY_400, size=MIN_LINE_SIZE, linetype="dotted",
        )
        + geom_line(size=MIN_LINE_SIZE, color=INK)
        + geom_point(size=1.2, color=INK)
        + facet_wrap("~ metric", ncol=1, scales="free_y")
        + scale_x_log10(breaks=breaks)
        + labs(
            x="Reference\ncontexts",
            y=(
                "% of all-context performance"
                if ALIGN_RIGHT
                else "Improvement over single-context baseline"
            ),
        )
        + figure_theme(base_size=8)
        + theme(
            # Match the zeroshot titration figure: titles 8, ticks 7.
            strip_text=element_text(size=8),
            axis_title=element_text(size=8),
            axis_text_x=element_text(size=7),
            axis_text_y=element_text(size=7),
            plot_margin_top=0.02,
        )
    )


def main() -> None:
    df = load_long()
    if ALIGN_RIGHT:
        # L=1 dwarfs the L>=5 structure under right-anchoring; drop it.
        df = df[df["L"] > 1].copy()
    df = relativize(df, align_right=ALIGN_RIGHT)
    # 40 mm wide (1.575"), leaving headroom at the top of the column for
    # illustrations.
    save_figure(make_plot(df), f"cell_titration{ALIGN_SUFFIX}", size=(40 / 25.4, 7.5))


if __name__ == "__main__":
    main()
