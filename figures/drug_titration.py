"""Drug titration: one column of metrics, gain over single-perturbation baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from matplotlib.transforms import ScaledTranslation

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
    FIGURES_DIR,
    GRAY_400,
    INK,
    METRIC_FROM_JSON,
    METRIC_SHORT,
    METRIC_SHORT_ORDER,
    MIN_LINE_SIZE,
    figure_theme,
)

REPO = Path(__file__).parent.parent
TITRATION = REPO / "titration_drugs_results.json"
GENERALIZATION = REPO / "rhaister_results.json"
GENERALIZATION_RUN = "eval_rhaister_simplified_model"
GENERALIZATION_L = 120  # full generalization split anchor


def load_long(scope: str = "fixed") -> pd.DataFrame:
    titration = json.loads(TITRATION.read_text())["metrics"][scope]
    generalization = json.loads(GENERALIZATION.read_text())[GENERALIZATION_RUN]

    rows = []
    for json_key, pretty in METRIC_FROM_JSON.items():
        short = METRIC_SHORT[pretty]
        for holdout, by_L in titration.get(json_key, {}).items():
            for L, value in by_L.items():
                rows.append({
                    "metric": short,
                    "holdout": f"holdout {holdout}",
                    "L": int(L),
                    "value": float(value),
                })
        # L=120 anchor from the generalization split (same fixed test set, same model)
        for split_name, split_metrics in generalization.items():
            if json_key not in split_metrics:
                continue
            holdout = split_name.removeprefix("tahoe_").removesuffix("_holdout")
            if holdout not in titration.get(json_key, {}):
                continue
            rows.append({
                "metric": short,
                "holdout": f"holdout {holdout}",
                "L": GENERALIZATION_L,
                "value": float(split_metrics[json_key]),
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
    # Right-anchored view reads as "% of full-panel performance" (anchor = 100%);
    # left-anchored view stays an absolute improvement over the single-sample baseline.
    out["pct"] = out["value"] / out["baseline"] * 100
    return out


def make_plot(df: pd.DataFrame) -> ggplot:
    # Drop the L=30 tick/label only (data still plotted); it crowds the axis
    # between the 20 and 60 ticks.
    breaks = [b for b in sorted(df["L"].unique()) if b != 30]
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
        + scale_x_log10(breaks=breaks, labels=[str(b) for b in breaks])
        + labs(
            x="Perturbation\npanel size",
            y=(
                "% of full-panel performance"
                if ALIGN_RIGHT
                else "Improvement over single-sample baseline"
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


# At 40 mm width the log-axis labels 6/10, 10/20 and 60/120 nearly touch.
# Nudge the crowded labels along x (points; -left/+right) so each reads
# unambiguously while its tick mark stays put. The slack lives in the 3-6 and
# 20-60 gaps, so the crowded labels move into it (6 and 60 left, 20 slightly
# right); 10 and 120 stay anchored on their ticks.
X_LABEL_NUDGES_PT = {"6": -2.2, "20": 1.2, "60": -2.6}


def _save_nudged_xlabels(plot, name: str, size: tuple[float, float], dpi: int = 300) -> None:
    """Like style.save_figure, but offset individual x-axis tick labels
    horizontally after layout. plotnine's theme can't move single tick labels,
    so we shift them on the drawn matplotlib figure (tick marks are unaffected)."""
    fig = (plot + theme(figure_size=size)).draw(show=False)
    for ax in fig.axes:
        for lbl in ax.get_xticklabels():
            dx = X_LABEL_NUDGES_PT.get(lbl.get_text().strip())
            if dx and lbl.get_visible():
                lbl.set_transform(
                    lbl.get_transform()
                    + ScaledTranslation(dx / 72.0, 0, fig.dpi_scale_trans)
                )
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(out, dpi=dpi, transparent=True)
        print(f"wrote {out}")


def main() -> None:
    df = load_long(scope="fixed")
    df = relativize(df, align_right=ALIGN_RIGHT)
    # 40 mm wide (1.575"), leaving headroom at the top of the column for
    # illustrations.
    _save_nudged_xlabels(make_plot(df), f"drug_titration{ALIGN_SUFFIX}", size=(40 / 25.4, 7.5))


if __name__ == "__main__":
    main()
