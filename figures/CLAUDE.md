# figures/

Publication-quality figures for the Rhaister paper. One figure per script;
outputs both PDF (vector) and PNG (300 dpi).

## Tooling

- **plotnine** for all figures. Don't use seaborn.
- matplotlib is acceptable when plotnine can't express something cleanly
  (custom layouts, network/graph diagrams), but plotnine is the default.
- Run via `uv run --with plotnine --with pandas python <figure>.py`.

## Shared style

`style.py` exports:

- `figure_theme(base_size=10)` — shared plotnine theme. Compose with
  `+ figure_theme()`. Transparent backgrounds, faded horizontal gridlines
  only, no strip backgrounds, no minor ticks.
- `save_figure(plot, name, size=(w, h))` — writes both PDF and PNG with
  transparent background to `figures/`.
- `METRIC_FROM_JSON`, `METRIC_ORDER` — canonical display names for the six
  State metrics. Use these when relabeling JSON keys for figures.
- `METHOD_FROM_JSON` — canonical names for baselines (Additive,
  Context Mean, Perturbation Mean, Global Mean).
- `METHOD_GROUP`, `GROUP_ORDER`, `GROUP_COLOR` — five-group color palette
  for methods/baselines: global mean, marginal mean (Context Mean +
  Perturbation Mean), additive, Rhaister (incl. Permute Perturbation /
  Permute Context ablations), replicate ceiling (Half-sample reference).
  Use these whenever a figure shows multiple methods side by side so
  colors stay consistent across the paper.

Refactor new style elements into `style.py` only after a pattern appears
in ≥2 figures.

## Data sources

Figures consume pre-aggregated result files from the parent directory.
Don't load raw experiment data here.

| File | Use |
|---|---|
| `../titration_drugs_results.json` | Drug titration (L sweep) |
| `../titration_drugs_replicate_eval_results.json` | Replicate-eval titration (model→A, model→B) |
| `../rhaister_results.json` | Full-split eval (L=120 anchor) |
| `../a_vs_b_results.tahoe_only.json` | Half-replicate ceilings (B→A, A→B) |
| `../baseline_results.json` | Simple baselines per holdout |
| `../label_permutation_ablation.json` | Permute Perturbation / Permute Context ablation metrics |

## Conventions

- Build figures incrementally, in response to specific narrative needs.
  Don't pre-scaffold panels that haven't been requested.
- Backgrounds are transparent so figures drop cleanly onto colored panels
  in slide / paper editors.
- For per-holdout metrics with significant level variation, prefer
  relative-to-baseline (gain) curves or single mean curves over
  multi-color overlays — the variation between holdouts is "test
  difficulty," not the message.
- Default axis: `scale_x_log10` for L (drugs / cells / data scaling
  axes); breaks at the actual sample levels.
- Don't use alpha to disambiguate overlapping points — dodge them along
  the categorical axis instead (deterministic, not random jitter).
- When showing multiple methods, prefer coloring the y-axis tick labels
  by group over a separate color legend; suppress legends with
  `guide=None` on the scales and `legend_position="none"` in the theme.
- For figures that compare metrics on a common scale (e.g. all six State
  metrics in one figure), fix the x-axis to `(0, 1)` with shared breaks
  so panels are visually comparable.

## Naming

- Script: `<figure_topic>.py` (e.g. `drug_titration.py`)
- Output: `<figure_topic>.{pdf,png}` in this directory.
- Match the script and output names so the script's purpose is obvious.
