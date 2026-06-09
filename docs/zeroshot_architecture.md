# RhaisterO Zero-Shot Model Architecture

## Problem

Predict treatment responses for fully held-out cell lines. Test cells have zero treatment-response observations in training; the only cell-side signal is the DMSO baseline expression centroid.

## Model equation

```
d_hat(c, t) =
    mu_tilde(t)
    + (1 - lambda_p) * [gamma_tilde(t) * z_c]
    + lambda_p * Proj_t[gamma_tilde(t) * z_c]
```

| Term | Meaning |
|---|---|
| `mu_tilde(t)` | SVD-denoised treatment mean response (rank 200) |
| `z_c` | Z-scored DMSO baseline expression centroid for cell line c |
| `gamma_tilde(t)` | Treatment-specific per-gene diagonal cell modulation (EB-shrunk) |
| `Proj_t` | Soft Wiener projection onto treatment t's empirical response subspace |
| `lambda_p` | Projection blend weight (0.3, applied to correction only) |

The diagonal interaction `gamma_t,g * z_c,g` is the right complexity level: it captures most of the zero-shot cell-line signal without overfitting the ~45 training cells per treatment.

## Seven-idea model stack

Built as a controlled ablation ladder. Each step's contribution measured on `tahoe_5`:

| Step | Component | Pearson delta | Delta |
|---|---|---|---|
| 1 | Treatment mean baseline | ~0.483 | — |
| 2 | Shared cell-line centroid effect (pooled beta) | ~0.536 | +0.053 |
| 3 | Treatment-specific diagonal cell effect `gamma_t * z_c` | ~0.654 | +0.118 |
| 4 | SVD-denoised treatment means (rank 200) | ~0.674 | +0.020 |
| 5 | Response-subspace projection (soft Wiener, blend 0.3) | 0.6816 | +0.008 |
| 6 | Empirical-Bayes shrinkage for gamma | 0.6917 | +0.010 |
| 7 | Treatment-structured gamma prior | 0.6954 | +0.004 |

### Idea details

**1. Treatment mean** — Average expression delta across all training cells for each treatment. Strong baseline because most variation is treatment-driven.

**2. Shared per-gene slope** — Pooled OLS: `beta_g = sum(z_c * residual_g) / sum(z_c^2)`. A single slope per gene across all cells and treatments.

**3. Treatment-specific per-gene slope** — Per-treatment OLS: `gamma_{t,g}` captures how each drug's effect on gene g scales with baseline expression. The main interaction term.

**4. SVD denoising** — Rank-200 truncated SVD of the treatment mean matrix. Borrows strength across treatments in response space, reducing noise in rare-treatment means.

**5. Response-subspace projection** — Per-treatment SVD of cell deviations defines a response subspace. Predictions are soft-projected via Wiener weights `sv / (sv + lambda)` where `lambda` = 25th percentile of all singular values. A global 50-PC subspace provides the shrinkage target. Blend: `(1 - 0.3) * correction + 0.3 * projected`.

**6. Empirical-Bayes gene-wise shrinkage** — Instead of fixed shrinkage, estimate per-(treatment, gene) `alpha_{t,g} = signal_var / (signal_var + noise_var)`. High-alpha (well-estimated) gammas stay close to raw OLS; noisy ones shrink toward the prior.

**7. Treatment-structured prior** — The EB shrinkage target is not just the shared beta but a structured prior: `gamma_struct_{t,g} = beta_g + w_g^T s_t`, where `s_t` are the treatment's coordinates in response-PCA space. This pools gamma information through treatment similarity.

## FC and p-value prediction heads

The core model predicts expression deltas (D). Two additional heads convert these to fold changes and p-values for the full State metric suite:

**FC head (D→FC mapping):**
- Per-gene OLS: regress FC deviations on D deviations across training data
- Per-gene R²-weighted shrinkage toward expression-bin group means
- Per-treatment scale (shrunk toward global)
- `FC_pred = fc_treat_mean + 0.3 * D_correction * d2fc_scale`

**P-value head (z-score calibration):**
- Per-treatment FC variance across training cells
- Gene DE frequency prior: `z_boost = exp(0.35 * z_prior)` where z_prior is standardized DE frequency
- Per-treatment calibration constant via binary search to match expected BH rejection count
- Vectorized normal-approximation p-values via `erfc`

## What did not work

Extensive experiments showed that richer cell and drug representations did not improve the model:

**Cell-side failures:**
- Row-level DMSO reference (noisy version of plate-averaged centroid)
- Naive shared latent response basis (destroyed cell identity)
- Cross-gene operators `V_t V_t^T z_c` (too many parameters for ~45 cells)
- ESM-C / GLM-style gene and driver embeddings (no stable signal)
- Foundation model cell embeddings (no residual signal beyond centroid)

**Drug-side failures:**
- Drug metadata, Morgan fingerprints, drug embeddings (information present but redundant with response-PCA geometry)
- Drug-feature priors for treatment mean (chemistry ridge worse than response-SVD)
- Internal treatment pooling / KNN smoothing (already saturated by SVD)
- Metric calibration (did not improve primary Pearson)

**Key insight:** `estimation quality > model capacity > external covariates`. Response-derived structure (SVD, response-PCA) outperformed external features. Diagonal cell modulation outperformed dense cross-gene operators.

## Performance

| Split | Pearson delta (D) | FC Pearson | DE overlap | PR-AUC | Spearman LFC sig |
|---|---|---|---|---|---|
| tahoe_5_holdout | 0.695 | 0.291 | 0.367 | 0.416 | 0.506 |

5-split reference (clean core): 0.6478 +/- 0.0701

## Implementation

Located in `train.py` — activated via `zeroshot=True` parameter or `HP_ZEROSHOT=1` env var:
```bash
HP_ZEROSHOT=1 python train.py <experiment_name>
```

Two regression backends selectable via `HP_ZS_MODEL`:
- `ridge` (default): multi-feature ridge on R/L/C/H feature stack (`_compute_regression_zeroshot`)
- `diagonal`: per-(drug, gene) scalar gamma on per-gene z (`_compute_regression_zeroshot_diagonal`)

The diagonal model corresponds to the core equation described above. The ridge model is a cross-gene generalization that concatenates multiple cell-feature blocks.

Data loaded via `prepare_all(split_name, zeroshot=True)`, which builds control-only reference features (R, L, C, H) for held-out cells from `control_expression_path` and `cell_centroid_*_path` configured in `dataset.toml`.
