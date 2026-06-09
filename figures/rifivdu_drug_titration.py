"""Rifivdu drug titration: R² gain over the smallest fewshot pool (L=1).

Equivalent of figures/drug_titration.py for the EmeraldBay scalar-target setup.
Only R² is recorded by run_titrations_drugs_sensitivity.py, so this is a
single panel (no metric facet). One line per EmeraldBay split, gain measured
relative to L=1 within each split. Source JSON has both 'fixed' (comparable
across L) and 'full' (per-L test set) scopes — we use 'fixed' as the
headline, matching drug_titration.py's choice.
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
    scale_x_log10,
    theme,
)

from style import GRAY_400, INK, MIN_LINE_SIZE, figure_theme, save_figure

REPO = Path(__file__).parent.parent
TITRATION = REPO / f"titration_drugs_sensitivity_results.json"


def load_long(scope: str = "fixed") -> pd.DataFrame:
    data = json.loads(TITRATION.read_text())
    by_split = data["metrics"][scope]["r2"]
    rows = []
    for split, by_L in by_split.items():
        for L, value in by_L.items():
            rows.append({
                "split": f"split {split}",
                "L": int(L),
                "value": float(value),
            })
    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(
        df["split"],
        categories=sorted(df["split"].unique()),
        ordered=True,
    )
    return df


def relativize_to_L1(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract each split's L=1 R² from every point in that split."""
    L_min = df["L"].min()
    baseline = (
        df[df["L"] == L_min]
        .set_index("split")["value"]
        .rename("baseline")
    )
    out = df.merge(baseline, on="split", how="left")
    out["gain"] = out["value"] - out["baseline"]
    return out


def make_plot(df: pd.DataFrame) -> ggplot:
    breaks = sorted(df["L"].unique())
    return (
        ggplot(df, aes(x="L", y="gain", group="split"))
        + geom_hline(yintercept=0, color=GRAY_400, size=MIN_LINE_SIZE, linetype="dotted")
        + geom_line(size=MIN_LINE_SIZE, color=INK)
        + geom_point(size=1.2, color=INK)
        + scale_x_log10(breaks=breaks)
        + labs(
            x="Perturbation panel size",
            y="R² gain over L=1",
        )
        + figure_theme(base_size=8)
        + theme(
            axis_title=element_text(size=7),
            axis_text_x=element_text(size=6),
            axis_text_y=element_text(size=6),
            plot_margin_top=0.02,
        )
    )


def main() -> None:
    df = load_long(scope="fixed")
    df = relativize_to_L1(df)
    # One-metric panel, slightly wider than drug_titration.py's column width so
    # the y-axis title fits ("R² gain over single-perturbation baseline").
    save_figure(make_plot(df), f"EmeraldBay_drug_titration", size=(2.4, 1.5))


if __name__ == "__main__":
    main()
