"""Shared style + naming conventions for figures.

Figures should:
    from style import METRIC_FROM_JSON, METRIC_ORDER, figure_theme, save_figure
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_label,
    geom_text,
    theme,
    theme_minimal,
)

FIGURES_DIR = Path(__file__).parent

# --- Paper layout ---------------------------------------------------------

# The paper's two-column text block is 166 mm wide; figure widths are sized
# as fractions of it so panels tile cleanly across the page.
TWO_COL_WIDTH_MM = 166
MM_PER_IN = 25.4


def col_width_in(fraction: float = 1.0) -> float:
    """Figure width (inches) for ``fraction`` of the 166 mm two-column block."""
    return TWO_COL_WIDTH_MM * fraction / MM_PER_IN


# --- Naming conventions ---------------------------------------------------

METRIC_FROM_JSON = {
    "state/discrimination_mean":   "Perturbation discrimination",
    "state/pearson_delta_mean":    "Pearson correlation Δ",
    "state/pr_auc_mean":           "PR-AUC",
    "state/spearman_lfc_sig_mean": "Spearman LFC",
    "state/de_overlap_mean":       "DE overlap",
    "state/de_spearman_sig":       "Spearman effect size",
}
METRIC_ORDER = list(METRIC_FROM_JSON.values())

# Shorter strip labels for narrow / column-fit layouts.
METRIC_SHORT = {
    "Perturbation discrimination": "Discrimination",
    "Pearson correlation Δ":       "Pearson Δ",
    "PR-AUC":                      "PR-AUC",
    "Spearman LFC":                "Spearman LFC",
    "DE overlap":                  "DE overlap",
    "Spearman effect size":        "Spearman ES",
}
METRIC_SHORT_ORDER = [METRIC_SHORT[m] for m in METRIC_ORDER]

METHOD_FROM_JSON = {
    "additive":       "Additive",
    "cell_mean":      "Context Mean",
    "treatment_mean": "Perturbation Mean",
    "global_mean":    "Global Mean",
}

# Shorter y-tick labels for narrow / single-column (compact) layouts.
METHOD_SHORT = {
    "Global Mean":           "Global Mean",
    "Context Mean":          "Ctx. Mean",
    "Perturbation Mean":     "Pert. Mean",
    "Additive":              "Additive",
    "STATE":                 "STATE",
    "Rhaister":              "Rhaister",
    "Half-sample reference": "Half-sample",
    "Permute Perturbation":  "Permute Pert.",
    "Permute Context":       "Permute Ctx.",
}

# Method/baseline grouping for color encoding. Methods within a group share a
# color (e.g. Rhaister and its data-level ablations PermuteP/PermuteC).
METHOD_GROUP = {
    "Global Mean":       "global mean",
    "Global Mean Zeroshot":       "global mean",
    "Context Mean":      "marginal mean",
    "Perturbation Mean": "marginal mean",
    "Perturbation Mean Zeroshot": "marginal mean",
    "Additive":          "additive",
    "Rhaister":                 "Rhaister",
    "Rhaister Fewshot":         "Rhaister",
    "Rhaister Zeroshot":        "Rhaister",
    "Rhaister Zeroshot Shared": "Rhaister",
    "Permute Perturbation":     "Rhaister",
    "Permute Context":          "Rhaister",
    "STATE":             "STATE",
    "STATE Fewshot":     "STATE",
    "STATE Zeroshot":    "STATE",
    "Half-sample reference": "replicate ceiling",
}
GROUP_ORDER = [
    "global mean",
    "marginal mean",
    "additive",
    "STATE",
    "Rhaister",
    "replicate ceiling",
]
# Tahoe brand tokens (source: ../../Tahoe-design-system/python/tahoe_colors.py).
# Hardcoded so figures build without sys.path gymnastics; keep in sync.
NAVY_900  = "#082846"  # anchor / "ground truth"
NAVY_500  = "#216BB4"  # mid navy
BLUE_500  = "#058DC7"  # saturated sky
BLUE_300  = "#83BAF2"  # cool baseline
CORAL_400 = "#FF5F31"  # primary accent — our method
CORAL_300 = "#FF9028"  # amber — comparator
INK       = "#1E1D1B"  # plot ink — single-color lines/points
GRAY_600  = "#747775"  # neutral / axis labels
GRAY_400  = "#B5B5B3"  # de-emphasized reference lines
GRAY_300  = "#D9D9D9"  # borders / gridlines

GROUP_COLOR = {
    "global mean":       GRAY_600,
    "marginal mean":     BLUE_300,
    "additive":          NAVY_500,
    "STATE":             CORAL_300,
    "Rhaister":          CORAL_400,
    "replicate ceiling": NAVY_900,
}

# Design system minimum stroke width.
MIN_LINE_PT = 1.5  # use in theme element_line (passed directly as mpl linewidth)
# Plotnine's geom_*(size=...) aesthetic is multiplied by SIZE_FACTOR = sqrt(pi)
# before being passed to matplotlib's linewidth. So for geom calls divide by
# that factor to land at MIN_LINE_PT in the rendered output.
MIN_LINE_SIZE = MIN_LINE_PT / 1.7724538509055159  # ≈ 0.846, for geom_*(size=...)


# --- Theme ----------------------------------------------------------------

def figure_theme(base_size: int = 10) -> theme:
    """Shared plotnine theme. Compose with `+ figure_theme()`."""
    return theme_minimal(base_size=base_size) + theme(
        panel_spacing=0.04,
        panel_grid_minor=element_blank(),
        panel_grid_major_x=element_blank(),
        panel_grid_major_y=element_line(color=GRAY_300, size=MIN_LINE_PT),
        strip_background=element_blank(),
        strip_text=element_text(size=base_size - 1),
        axis_title=element_text(size=base_size),
        axis_ticks_major=element_line(color=GRAY_600, size=MIN_LINE_PT),
        axis_ticks_minor=element_blank(),
        axis_ticks_length=3,
        legend_title=element_text(size=base_size - 1),
        legend_key=element_blank(),
        plot_background=element_blank(),
        panel_background=element_blank(),
    )


# --- Per-row mean value labels --------------------------------------------

def add_mean_value_labels(
    plot,
    means: pd.DataFrame,
    points: pd.DataFrame,
    top_method: str,
    *,
    txt_size: float,
    cutoff: float = 0.90,
    pad: float = 0.045,
    lbl_dy: float = 0.34,
    top_above: bool = True,
):
    """Print each row's mean value on a metric lollipop plot, placed to avoid
    overlap, and return ``plot`` with the label layers added.

    Each row's value (``means['value']``) is drawn as a small white-boxed
    label just right of the row's rightmost point. Rows whose points reach
    past ``cutoff`` have no room on the right, so they fall back: when
    ``top_above`` the top row (``top_method``) puts its label *above* the
    diamond — it has empty headroom there — while all other rows mirror the
    label to the *left* of their leftmost point. With ``top_above=False``
    (short panels with no vertical headroom) every over-``cutoff`` row,
    including the top one, goes left. Left/right labels are vertically
    centered on their own row, so they never collide with the row above.

    ``means`` needs columns ``y``, ``value``, ``group``, ``metric``;
    ``points`` is the per-observation frame (``method``, ``metric``,
    ``value``) used to find each row's left/right extent (pass ``means``
    itself when there is a single point per row).
    """
    means = means.copy()
    means["_label"] = means["value"].map(
        lambda v: "" if pd.isna(v) else f"{v:.2f}".lstrip("0")
    )
    rmax = points.groupby(["method", "metric"], observed=True)["value"].max().to_dict()
    rmin = points.groupby(["method", "metric"], observed=True)["value"].min().to_dict()
    means["_rmax"] = [
        v if pd.isna(v) else max(v, rmax.get((m, me), v))
        for m, me, v in zip(means["method"], means["metric"], means["value"])
    ]
    means["_rmin"] = [
        v if pd.isna(v) else min(v, rmin.get((m, me), v))
        for m, me, v in zip(means["method"], means["metric"], means["value"])
    ]
    means["_lx"] = means["_rmax"] + pad
    means["_lx_left"] = means["_rmin"] - pad
    means["_ly"] = means["y"] + lbl_dy

    lab = means[means["_label"] != ""]
    fits = lab["_rmax"] <= cutoff
    is_top = lab["method"].astype(str) == top_method
    right = lab[fits]
    above = lab[~fits & is_top] if top_above else lab.iloc[0:0]
    left = lab[~fits & ~is_top] if top_above else lab[~fits]

    return (
        plot
        + geom_label(
            right,
            aes(x="_lx", y="y", label="_label", color="group"),
            size=txt_size, va="center", ha="left",
            fill="white", label_size=0, label_padding=0.1,
        )
        + geom_text(
            above,
            aes(x="value", y="_ly", label="_label", color="group"),
            size=txt_size, va="bottom", ha="center",
        )
        + geom_label(
            left,
            aes(x="_lx_left", y="y", label="_label", color="group"),
            size=txt_size, va="center", ha="right",
            fill="white", label_size=0, label_padding=0.1,
        )
    )


# --- Saving ---------------------------------------------------------------

def save_figure(plot, name: str, size: tuple[float, float], dpi: int = 300) -> None:
    """Save a plotnine figure to both PDF and PNG in figures/, transparent background."""
    width, height = size
    for ext in ("pdf", "png"):
        out = FIGURES_DIR / f"{name}.{ext}"
        plot.save(
            out, width=width, height=height, dpi=dpi, verbose=False,
            transparent=True,
        )
        print(f"wrote {out}")
