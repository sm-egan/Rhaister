"""Rifivdu drug titration: R² vs L for the two model conditions.

Overlays the no-features Rhaister titration on top of the pdex-features
titration. One mean line per condition + per-split dots so the spread is
visible. Same fixed/r2 (drugs[L_MAX:]) scope as the underlying sweeps.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from plotnine import (
    aes,
    element_text,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_x_log10,
    theme,
)

from style import GRAY_400, GROUP_COLOR, MIN_LINE_SIZE, figure_theme, save_figure

REPO = Path(__file__).parent.parent
JSONS = {
    "Sensitivity only": REPO / f"titration_drugs_sensitivity_results.json",
    "Sensitivity + tx": REPO / f"titration_drugs_sensitivity_results__feat_pdex.json",
}


def load_long() -> pd.DataFrame:
    rows = []
    for cond, path in JSONS.items():
        data = json.loads(path.read_text())
        by_split = data["metrics"]["fixed"]["r2"]
        for split, by_L in by_split.items():
            for L, value in by_L.items():
                rows.append({
                    "condition": cond,
                    "split": f"split {split}",
                    "L": int(L),
                    "value": float(value),
                })
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(
        df["condition"],
        categories=["Sensitivity only", "Sensitivity + tx"],
        ordered=True,
    )
    return df


def mean_lines(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["condition", "L"], observed=True)["value"].mean().reset_index()


def make_plot(df: pd.DataFrame) -> ggplot:
    means = mean_lines(df)
    breaks = sorted(df["L"].unique())
    palette = {
        "Sensitivity only": "#A93428",   # darker red — the bare model
        "Sensitivity + tx": GROUP_COLOR["Rhaister"],  # CORAL_400 — feature variant
    }
    return (
        ggplot()
        + geom_hline(yintercept=0, color=GRAY_400, size=MIN_LINE_SIZE, linetype="dotted")
        + geom_point(df, aes(x="L", y="value", color="condition"), size=1.5, alpha=0.45)
        + geom_line(means, aes(x="L", y="value", color="condition"), size=MIN_LINE_SIZE)
        + geom_point(means, aes(x="L", y="value", color="condition"), size=3, shape="D")
        + scale_x_log10(breaks=breaks)
        + scale_color_manual(values=palette, name="")
        + labs(
            x="Perturbation panel size (L)",
            y="R² (fixed eval set)",
        )
        + figure_theme(base_size=10)
        + theme(
            legend_position="top",
            axis_title=element_text(size=9),
        )
    )


def main() -> None:
    df = load_long()
    save_figure(make_plot(df), f"EmeraldBay_drug_titration_with_features", size=(3.4, 2.4))


if __name__ == "__main__":
    main()
