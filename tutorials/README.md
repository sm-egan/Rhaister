# Tutorials

## `rhaister_walkthrough.ipynb`

Raw single-cell counts → differential-expression summaries → a trained Rhaister
model → the State-paper metrics, with every step visible.

It runs on **public data with no credentials**. Nothing here needs cluster
access, a GitHub token, or `HF_TOKEN`.

### This is an adapted version

The walkthrough as originally written could not run outside Tahoe Bio: its setup
cell prompted for a GitHub token and cloned a repo that returns 404, the
`sample_tahoe.h5ad` every later step needed was never published, and step 3
imported a private symbol from `cell-eval`, which is in none of the repo's
dependency sets.

This version fixes all three, and **has not been contributed upstream**. So:

- Re-downloading `tahoebio/Rhaister` gets the *original* notebook, not this one.
- This notebook needs the checkout it came with — specifically
  `rhaister/tutorial_data.py`, which does not exist upstream. The setup cell
  looks for that file and stops with an explanation if it can't find it, rather
  than failing obscurely a few cells later.

### Running it

From this checkout:

```bash
uv pip install -e ".[tutorial]"
jupyter lab tutorials/rhaister_walkthrough.ipynb
```

In Colab, upload this checkout (or otherwise make it reachable) and point
`RHAISTER_REPO` at it before running the setup cell. A bare Colab runtime with
nothing but the notebook will not work until these changes are upstream.

### The two data paths

Set `DATA_MODE` in the setup section.

| `DATA_MODE` | Step 3 (the `AnnData → matrices` bridge) | Steps 4-6 (training) | First-run cost |
|---|---|---|---|
| `"fixture"` *(default)* | `tests/fixtures/plate1_CVCL_0023_100genes.h5ad` — 14,896 cells, 1 cell line, 91 drugs, 100 genes | `data/tahoe_walkthrough_de_subset.parquet`, bundled here | none |
| `"tahoe100m"` | `sample_tahoe.h5ad`, rebuilt from `tahoebio/Tahoe-100M` | the matrices step 3 computed | a few hundred MB, ~10 min |

`"fixture"` is offline and takes a couple of minutes end to end. Its single cell
line cannot train the model — a per-cell ridge needs several to regress against —
so steps 4-6 use a published slice of 8 cell lines × 24 drugs instead. Step 3
earns its keep there by checking its computed deltas against the published
values for that cell line (mean per-gene *r* > 0.999), which is what shows the
notebook is teaching the production pipeline rather than a lookalike.

`"tahoe100m"` is the full story on one dataset: single cells in, metrics out.

### Where the data comes from

Everything is public.

| What | Source | Size |
|---|---|---|
| Bundled DE subset | slice of `tahoebio/tahoe-de-rhaister` | 7 MB, committed here |
| Test fixture | committed in `tests/fixtures/` | 5.5 MB |
| Raw single cells (`"tahoe100m"`) | `tahoebio/Tahoe-100M` | ~3,388 shards; the builder reads a few |

The full `tahoebio/tahoe-de-rhaister` dataset is **41.8 GB**, almost all of it a
single 40.6 GB pdex parquet, so the notebook never calls `snapshot_download` on
it. `rhaister.tutorial_data.fetch_de_subset` instead prunes that parquet's row
groups using its footer statistics — it is clustered by `(plate, cell_line,
target)` — and reads only the few dozen covering the request. If you have the
data locally, set `RHAISTER_DATA_ROOT` and it is used in preference to the Hub,
the same precedence `prepare_combined` uses.

Note that in Tahoe **the plate encodes the dose**: plate 1 is the 0.05 µM arm,
plate 2 the 0.5 µM arm. A plate and a dose argument that disagree will match
nothing.

### Rebuilding the inputs

```bash
# Rebuild sample_tahoe.h5ad from the public Tahoe-100M shards
python scripts/build_example_h5ad.py --out sample_tahoe.h5ad

# Rebuild the bundled DE subset (only needed if the default slice changes)
python -c "
from rhaister.tutorial_data import *
fetch_de_subset(WALKTHROUGH_PLATE, WALKTHROUGH_CELL_LINES, walkthrough_treatments(),
                load_gene_panel('tahoe'), cache_path=BUNDLED_SUBSET, use_bundled=False)
"
```

### Expected numbers

Both paths use the same 192 observations with 8 held out, but they get their
matrices from different places, and it shows:

| Metric | `"fixture"` | `"tahoe100m"` | Paper (5 holdouts, full data) |
|---|---|---|---|
| `state/pearson_delta_mean` | 0.854 | 0.744 | 0.87 |
| `state/spearman_lfc_sig_mean` | 0.811 | 0.667 | 0.81 |
| `state/pr_auc_mean` | 0.671 | 0.318 | 0.73 |
| `state/de_overlap_mean` | 0.613 | 0.137 | 0.59 |

`"fixture"` trains on the **published** DE summaries, computed from every cell on
the plate, so it lands close to the paper. `"tahoe100m"` recomputes them from at
most `--cells-per-group` cells (100 by default, against thousands in the real
pipeline), so its p-values and FDRs are far noisier — which hits the two
significance-dependent metrics, PR-AUC and DE overlap, hardest. Raising
`--cells-per-group` closes most of the gap at the cost of a longer build.

Either way these are a small subsample scored on 8 test observations: a smoke
test of the pipeline, not a reproduction of the paper. For that, see the
reproduction steps in `CLAUDE.md`.

### Testing

`tests/test_tutorial_data.py` covers the loaders offline. The tests that reach
the Hub are skipped unless you opt in:

```bash
python -m pytest tests/test_tutorial_data.py -v
RHAISTER_TEST_NETWORK=1 python -m pytest tests/test_tutorial_data.py -v
```
