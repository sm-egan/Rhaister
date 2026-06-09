# Experiment Log

## Calnet runtime optimization sweep — partial success

Context: after the simplified-model refactor the metric ceiling felt saturated, so focus shifted to wall-clock optimization. Profile of a single `train_and_evaluate` run showed 270s total: calnet training ~56%, `evaluate_test` metrics suite ~27%, everything else ~17%.

### calnet_vectorize_test — accepted
- **Hypothesis**: Replacing the 1995-iteration per-gene test prediction loop with a single flat batch will cut Python/CUDA launch overhead and is a prerequisite for further fused-kernel work.
- **Change**: Collapse `for g in range(n_genes)` into one forward pass over `(n_test * n_genes, n_feat)`.
- **Results**: 270s → 267s (-3s). All metrics within stochastic noise.
- **Interpretation**: The test loop was only ~3s total — per-gene overhead is small. The real value is unblocking `torch.compile` / autocast, which want a single batched call.
- **Next**: Try `torch.compile` on the calnet forward pass.

### calnet_torch_compile — accepted
- **Hypothesis**: Calnet is a 7→256→128→1 MLP with ~66k backward passes; launch overhead should dominate compute. `torch.compile(mode="reduce-overhead")` captures CUDA graphs and fuses kernels.
- **Change**: Wrap `cal_net` in `torch.compile(..., mode="reduce-overhead", dynamic=False)` for training.
- **Results**: 267s → 251s (-16s, ~6%). Metrics unchanged within noise.
- **Interpretation**: Some speedup from reduced launch overhead, but smaller than hoped. Likely the backward pass is still bounded by autograd engine dispatch rather than kernel fusion.
- **Next**: Try matmul precision reductions (TF32, bf16) to actually reduce compute/memory bandwidth.

### calnet_ens1 — rejected
- **Hypothesis**: Dropping calnet ensemble from 2 to 1 halves calnet training time. The +0.0002 AUPRC gain from ens=2 (full_v15) might be stochastic noise in practice.
- **Change**: Default `HP_CALNET_ENS` 2 → 1.
- **Results**: 251s → 169s (-82s, -33%). But pr_auc -0.0009, auprc -0.0006, de_overlap -0.0003.
- **Interpretation**: ens=2 was not pure noise — the ensemble genuinely contributes ~0.001 on FDR-related metrics. Across six prior ens=2 runs pr_auc varied only ±2e-5, so a -0.0009 drop is ~40x noise. Real degradation.
- **Next**: Look for optimizations that don't touch metrics.

### calnet_ep400 — rejected
- **Hypothesis**: 800 epochs was tuned in full_v15 with the 6-feature calnet. Maybe 400 is enough with the 7-feature calnet + lower blend weight.
- **Change**: Default `HP_CALNET_EP` 800 → 400.
- **Results**: 251s → 169s (-82s, -33%). But pr_auc -0.0015 (worst of the sweep), auprc -0.0005, spearman_sig +0.0006, others within noise.
- **Interpretation**: Training length actually matters — the calnet has headroom at 400 that it closes between 400 and 800. Not a free lunch.
- **Next**: Try bf16 autocast for the training step.

### calnet_bf16 — rejected
- **Hypothesis**: The calnet MLP is memory-bandwidth bound (small matmuls); bf16 autocast halves bandwidth and should give a real speedup.
- **Change**: Wrap forward+loss in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
- **Results**: 251s → 151s (-100s, -40%). But de_overlap -0.00084, spearman_sig -0.00038, pr_auc -0.00026, auprc -0.00016.
- **Interpretation**: Biggest speedup of the sweep, but also noticeable metric drift. bf16 at 8-bit mantissa is enough to perturb the per-gene calibration on the marginal genes.
- **Next**: Try TF32 — 19-bit mantissa, much closer to fp32 but still tensor-core accelerated.

### calnet_tf32 — accepted
- **Hypothesis**: TF32 on B200 uses 19-bit mantissa via tensor cores — much higher precision than bf16 (8-bit) but still the same matmul acceleration.
- **Change**: `torch.set_float32_matmul_precision("high")` at module load.
- **Results**: 251s → 171s (-80s, -32%). fc_pearson, delta_pearson, pr_auc, auprc, de_overlap all within stochastic noise. Only spearman_sig drops -0.00052 (14x baseline run-to-run std, but still +0.0008 above v19's accepted state).
- **Interpretation**: Clearest tradeoff in the sweep — nearly the runtime savings of bf16 (80 vs 100s) with dramatically smaller metric drift. The one-line global setting affects calnet most because that's where the matmul volume is; ALS/regression phases are negligible by comparison.
- **Next**: Try pre-stacking features into a single indexed tensor to cut per-batch launch overhead further.

### calnet_prestack_features (v1 rejected, v2 accepted)
- **Hypothesis**: Each batch does 7 fancy indices + `torch.stack` against separate feature tensors; collapsing to a single pre-stacked (n_tr * n_genes, n_feat) tensor + one flat fancy-index should cut CUDA kernel launches.
- **Change**: Build `F_train_flat` of shape (n_tr * n_genes, n_feat) before training. Per-batch: `flat_idx = obs_idx * n_genes + gene_idx; x = F_train_flat[flat_idx]`.
- **Results (v1)**: 171s → 163s (-9s) but de_overlap -0.0009, pr_auc -0.00025 — rejected. v1 used a single `randint(0, n_tr*n_genes)` which shifted the RNG sequence, so the actual trained net differed.
- **Results (v2, keeps both randints)**: 171s → 167s (-4s). All metrics within TF32 run-to-run noise.
- **Interpretation**: `torch.compile(reduce-overhead)` was already capturing most of the per-batch overhead — there's less free room here than expected. Remaining calnet time is the actual backward pass compute, which is the GPU compute floor for this architecture + batch_size.
- **Next**: Net of this whole sweep is 270s → 167s (-38%). Further calnet-only savings would require accepting metric drift (ens=1, fewer epochs, bf16) or explicit full-step CUDA graph capture (complicated by the LR scheduler). The other big remaining block is `evaluate_test` (71s, ~42% of current runtime) in immutable files.

## IPCA learned gene factors (HP_IPCA=1, K=30) — success
- **Hypothesis**: Replace fixed PCA basis for MLP residuals with learned gene factors (IPCA). Network predicts K-dim loading vector β from (cell_emb, treat_emb), projects via learned gene factors F to gene space: Y_resid ≈ β @ F. End-to-end training aligns F with the network's prediction capabilities.
- **Change**: Added IPCANet module with encoder (MLP → K-dim β) and learnable factor matrix F (K × n_genes), initialized from PCA. Trains on gene-space MSE directly (vs PCA-space MSE for standard MLP).
- **Results**:
  | K | fc_pearson | de_overlap | spearman_lfc |
  |---|-----------|-----------|-------------|
  | 20 | 0.6498 | 0.655 | **0.8417** |
  | **30** | 0.6497 | 0.655 | **0.8417** |
  | 50 | 0.6498 | 0.655 | 0.8411 |
  | 100 | 0.6497 | 0.655 | 0.8412 |
  | 150 (PCA baseline) | 0.650 | 0.655 | 0.840 |
- **Verification**: 3 runs at K=30: spearman_lfc 0.8419/0.8414/0.8420. Consistent +0.002.
- **Interpretation**: Learned gene factors are better aligned with the network's prediction signal. The PCA basis captures maximum variance directions, which may not be the most predictable directions from cell+drug embeddings. IPCA optimizes for predictability, finding gene programs that the network can actually distinguish. The improvement is specifically in spearman_lfc (ranking of significant genes), suggesting the learned factors better capture DE-relevant gene programs.
- **Next**: Set HP_IPCA=1, K=30 as default. MLP blend weight is still w=0.12 — gains beyond this are bottlenecked outside the MLP path. Look for signal elsewhere.

### LoRA-style per-cell drug embedding adaptation (HP_LORA) — abandoned
- **Hypothesis**: Per-cell low-rank adaptation of treatment embeddings captures cell-specific drug similarity structure beyond scalar cell_weights.
- **Implementations tested**: Diagonal scaling (`diag` mode), low-rank matrix perturbation (`lowrank` mode, r=8), with/without full embedding adaptation, various lr/epochs/wd configs.
- **Results**: All 10+ configurations within ±0.001 of baseline fc_pearson (0.6493-0.6498 vs 0.6496 baseline). Best config (diag, lr=5e-3, ep=200) gives +0.0001.
- **Interpretation**: With only 50 unique cell lines, there's insufficient diversity to learn a generalizable cell→drug-embedding mapping. The LoRA network has effectively 50 training examples. The lowrank mode (predicting d_t×rank×2 parameters) completely fails to learn. Drug similarity is dominated by global structure that cell-level adaptation can't improve.

### DE-weighted IPCA loss (HP_RANK_BETA) — abandoned
- Swept beta=1,3,5. spearman_lfc 0.842-0.843 with fc_pearson dropping to 0.649. Within noise at MLP w=0.12.

### Doubly robust cross-fitting for MLP (HP_CROSSFIT) — abandoned
- 5-fold honest residuals don't help. Higher MLP weight (w=0.20, 0.30) still hurts. MLP signal inherently weak.

### FC regression centering (HP_FC_CENTER) — abandoned
- fc_pearson drops to 0.644 (-0.006). FC log-scale already centered ~0. Drug-mean carries useful signal.

### Concept bottleneck through pathway activities (HP_BOTTLENECK) — abandoned
- **Hypothesis**: Route IPCA predictions through a regularized bottleneck of K pathway activity scores. Add L1 sparsity on the K-dim activations (each observation activates few pathways) and orthogonality loss on the decoder rows (each pathway captures different genes). Inspired by Consensus-Bottleneck Asset Pricing (Jang et al., 2024), where bottleneck improved OOS R².
- **Change**: Added HP_BOTTLENECK=1 flag. When enabled, modifies IPCA training to add sparsity loss (beta.abs().mean()) and orthogonality loss ((F_norm @ F_norm.T - I)^2.mean()) to MSE.
- **Results**:
  | Config | fc_pearson | spearman_lfc | de_overlap |
  |--------|-----------|-------------|-----------|
  | Baseline (IPCA K=30) | 0.650 | 0.842 | 0.655 |
  | BN s=0.01 o=0.01 K=30 | 0.6497 | 0.8417 | 0.655 |
  | BN s=0.1 o=0.01 K=30 | 0.6498 | 0.8414 | 0.655 |
  | BN s=0.01 o=0.1 K=30 | 0.6497 | 0.8416 | 0.655 |
  | BN s=0.01 o=0.01 K=20 | 0.6497 | 0.8415 | 0.655 |
  | BN s=0.01 o=0.01 K=50 | 0.6494 | 0.8415 | 0.655 |
- **Interpretation**: All configurations within noise of baseline. The bottleneck regularization neither helps nor hurts. The IPCA architecture already effectively constrains the model through its low-rank K-dim bottleneck — adding explicit sparsity/orthogonality on top provides no additional regularization benefit. The MLP component contributes w=0.12 to the final prediction, so even meaningful changes within the MLP are heavily attenuated. The finance analogy doesn't transfer: gene expression residuals from a well-regularized additive model don't benefit from the same bottleneck that helps noisy financial returns.
- **Status**: Abandoned. No signal from any sparsity/ortho/K combination.

## unmix_k20_w005 / unmix_k20_w010 / unmix_k30_w005 / unmix_k50_w005 (Spectral unmixing)
- **Hypothesis**: Model FC residuals (after ALS subtraction) as convex combinations of K "pure response programs" (endmembers). An autoencoder with softmax encoder and non-negative decoder learns compositional response patterns. Blended as a post-hoc correction on top of existing FC predictions.
- **Change**: Added HP_UNMIX=1 code path after cell regression. Encoder: Linear(d_te+d_ce, K) -> softmax -> alpha. Decoder: Linear(K, n_genes, bias=False) with non-negative weight constraint. Loss: MSE reconstruction + entropy sparsity penalty. Blended at low weight with existing predictions.
- **Results**:
  | Config | fc_pearson | fc_median | auprc_p05 | de_overlap | pr_auc_mean | spearman_lfc_sig |
  |---|---|---|---|---|---|---|
  | baseline (no unmix) | 0.650 | 0.651 | 0.800 | 0.655 | 0.774 | 0.840 |
  | K=20, w=0.05 | 0.649192 | 0.650124 | 0.800262 | 0.655331 | 0.773856 | 0.839666 |
  | K=20, w=0.10 | 0.648178 | 0.649049 | 0.800165 | 0.654533 | 0.773643 | 0.838465 |
  | K=30, w=0.05 | 0.649262 | 0.650167 | 0.800275 | 0.655358 | 0.773740 | 0.839790 |
  | K=50, w=0.05 | 0.649259 | 0.649843 | 0.800262 | 0.655109 | 0.773806 | 0.839903 |
- **Interpretation**: No meaningful improvement from spectral unmixing. At w=0.05, results are within noise of baseline (~0.001 drop in fc_pearson). Higher weight (w=0.10) makes it slightly worse. K doesn't matter much (20-50 all similar). The non-negative constraint + softmax compositional structure is too restrictive for modeling residuals, which are centered around zero and contain both positive and negative values. The endmember idea assumes non-negative mixing of non-negative programs, but ALS residuals violate both assumptions.
- **Next**: Drop the compositional endmember approach. For residual modelling, prefer architectures that tolerate signed, zero-mean inputs (e.g. learned low-rank factors — see IPCA).

## Delta-specific RR rank (HP_DELTA_RR=0) — success
- **Hypothesis**: The global HP_RR_RANK=75 was tuned for FC regression. Deltas have different structure — they're linear-scale, dominated by high-expression genes. The full-rank solution might capture per-gene delta patterns that truncation destroys.
- **Change**: Added delta-specific RR rank override (same pattern as NLP's HP_PREG_RR=0). Default: HP_DELTA_RR=0 (no truncation).
- **Results**:
  | RR rank | delta_pearson |
  |---------|--------------|
  | 0 (new) | **0.909** |
  | 25 | 0.816 |
  | 50 | 0.872 |
  | 75 (old) | 0.887 |
  | 100 | 0.894 |
- delta_pearson: 0.887→**0.909** (+0.022). Largest single-experiment gain. All other metrics unchanged.
- **Interpretation**: FC benefits from RR truncation because high-rank drug combination weights overfit to per-gene FC noise. Deltas do NOT — the full-rank solution captures real per-gene delta structure. Delta responses are dominated by a few high-expression genes, and the full-rank solution preserves these gene-specific patterns rather than smoothing them out.
- **Next**: Audit other delta-specific defaults — each target may need its own tuned RR rank, centering, and blend weight rather than sharing FC's.

### DEG-weighted MLP loss (HP_MLP_DEG_BETA) — abandoned
- Swept beta=1,3,5. All within noise. MLP contributes only w=0.12 — loss function changes at this blend weight are irrelevant.

### Post-hoc low-rank consistency projection (HP_LOWRANK_K) — abandoned
- **Hypothesis**: After independently predicting all test FC values, project [Y_train; Y_pred] onto a low-rank manifold to enforce global consistency.
- **Results**: Catastrophic at all K values: K=20→0.288, K=50→0.318, K=100→0.339, K=200→0.375 (baseline 0.650).
- **Interpretation**: The gene expression matrix has high intrinsic rank (~1500+). Low-rank projection destroys gene-specific prediction signal. Even K=200 halves fc_pearson.

### Iterative residual refinement / boosting (HP_BOOST_ITERS) — abandoned
- **Hypothesis**: Re-run ALS on corrected data (residuals minus pipeline corrections) for 2-3 iterations.
- **Results**: All configurations hurt: 2 iters lr=0.8→fc_pearson 0.647 (-0.003), 2 iters lr=0.5→0.649 (-0.001), 3 iters→0.646 (-0.004).
- **Interpretation**: LOO train corrections are too noisy. Train RMSE actually increased after second iteration, confirming noise amplification.

### Gene-gene covariance smoothing (HP_GG_SMOOTH) — abandoned
- Smooth FC predictions using gene-gene covariance from training residuals. Even 5% blend hurts fc_pearson (-0.001). Predictions already have correct gene-gene structure.

### FC regression centering (HP_FC_CENTER) — abandoned
- fc_pearson 0.644 (-0.006). FC is log-scale centered ~0 — drug-mean carries useful signal. Opposite of deltas.

### Calnet hidden size retune with 7 features — no change
- h=128→de_overlap 0.656 (marginal), h=256→0.655, h=384→0.654, h=512→waiting. All within noise. h=256 remains optimal.

### Heteroscedastic FC+variance joint prediction (HP_HETERO) — abandoned
- **Hypothesis**: MLP predicts both FC residuals and per-gene log-variance. Variance maps to p-values via z = |FC|/sqrt(var).
- **Results**: alpha=0 (FC only, no blend): fc_pearson preserved (0.649). alpha=0.1 (10% hetero p-values): catastrophic — de_overlap drops to 0.012, pr_auc to 0.349.
- **Interpretation**: MLP variance prediction in PCA space ≠ statistical significance variance. The z-scores are meaningless because: (1) PCA→gene reconstruction introduces artifacts, (2) residual variance ≠ significance (which depends on sample size, expression level), (3) the calnet already learns the FC→significance mapping much better using ref_mean and gene statistics.
- **Key learning**: P-value prediction requires explicit statistical modeling (calnet with ref_mean, NLP features). Variance from residuals doesn't capture what makes a gene significant.

## Delta centering (HP_DELTA_CENTER=1) — success
- **Hypothesis**: The delta regression uses center=False (default). Centering subtracts per-drug mean across cells before computing drug combination weights, focusing on cell-specific delta deviations rather than absolute treatment effects.
- **Change**: Set center=True in delta `_compute_regression` call.
- **Results**: delta_pearson=**0.882** (up from 0.876, +0.006). All other metrics unchanged.
- **Verification**: Deterministic — 0.882242 across all 3 runs. No stochastic components in delta regression.
- **Interpretation**: Deltas are dominated by high-expression genes where the absolute drug effect (mean across cells) is large. Centering removes this dominant signal and lets the regression focus on cell-specific variations — which cell responds more or less strongly to each drug. This is the same principle that makes centering work for FC (the FC regression uses center=True implicitly via the default path), but it was accidentally disabled for deltas.
- **Next**: Now that delta regression mirrors FC (centering + RR), retune the delta blend alpha against the regression — sweep HP_DELTA_BLEND.

### Delta cell weight temperature (HP_DELTA_CWT) — abandoned
- Swept delta-specific cell weight temp (0.25, 0.50, 1.0). All hurt delta_pearson: 0.0→0.876, 0.25→0.875, 0.50→0.873, 1.0→0.872. FC cell weights (temp=0.18) are already optimal for delta prediction too.

### Calnet observation-level DE count (HP_CALNET_DECOUNT) — abandoned
- **Hypothesis**: Per-observation predicted DE fraction as calnet feature — encodes the "activity level" of the (cell, treatment) pair, relevant for BH correction context.
- **Results**: de_overlap 0.608 (-0.047 CATASTROPHIC), pr_auc 0.754 (-0.020). Other metrics unchanged.
- **Interpretation**: Creates a feedback loop that amplifies miscalibration. Observations predicted to have many DE genes get boosted, few get suppressed. Train/test mismatch severe (ALS vs regression DE counts). The per-observation constant acts as a bias term that distorts entire observations.
- **Key learning**: Observation-level significance features are dangerous for the calnet because they encode the same information the calnet is trying to predict, creating circular amplification.

### Effect shrinkage by observation count (HP_SHRINK_ALPHA) — abandoned
- **Hypothesis**: Shrink ALS treatment/cell effects toward zero for rarely-observed entities: eff *= n/(n+α). Bayesian posterior mean under zero-mean prior.
- **Results**: All alphas hurt monotonically. alpha=5→fc_pearson 0.649 (-0.001), alpha=50→0.646 (-0.004). Delta_pearson drops similarly.
- **Interpretation**: Min treatment count=33, min cell count=402 — no rare entity problem. ALS estimates are already reliable with this much data. The gene embedding regularization (HP_BRT=0.8) already serves as implicit shrinkage toward the gene embedding subspace.

### Delta blend retune (HP_DELTA_BLEND=0.95) — success
- **Results**: With centering enabled, higher blend trusts the now-accurate regression more. Swept 0.65-1.0: delta_pearson monotonically increases (0.859→0.888). Set 0.95 (plateau region).
- delta_pearson: 0.882→0.887 (+0.005).

## Blend alpha retune (HP_FULL_ALPHA=0.75) — success
- **Hypothesis**: blend_alpha=0.85 was tuned before pipeline improvements. With better ALS corrections and delta features, the pipeline now contributes more useful signal — lower alpha lets more through.
- **Change**: Swept HP_FULL_ALPHA from 0.65 to 0.95.
- **Results**: Monotonic improvement as alpha decreases from 0.95 to 0.65-0.75:
  | alpha | fc_pearson | de_overlap | pr_auc | spearman_lfc |
  |-------|-----------|-----------|--------|-------------|
  | 0.65 | 0.6491 | **0.6559** | **0.7741** | 0.8393 |
  | 0.70 | **0.6495** | 0.6555 | 0.7738 | 0.8397 |
  | **0.75** | 0.6494 | 0.6552 | 0.7739 | **0.8403** |
  | 0.80 | 0.6494 | 0.6546 | 0.7734 | 0.8402 |
  | 0.85 | 0.6491 | 0.6542 | 0.7732 | 0.8388 |
  | 0.95 | 0.6473 | 0.6523 | 0.7723 | 0.8365 |
- **Verification**: 3 runs at 0.75: fc_pearson 0.6496±0.0001, de_overlap 0.6554±0.0001, spearman_lfc 0.8402±0.0003.
- **Interpretation**: Best balance of all metrics at alpha=0.75. Plateau at 0.65-0.75 for de_overlap/pr_auc; spearman_lfc peaks at 0.75. The pipeline now carries more signal than it did when 0.85 was tuned.
- **Next**: Set HP_FULL_ALPHA=0.75 as default. Re-check MLP weight against the new alpha (follow-up sweep showed w=0.12 still optimal).

### FC MLP weight retune (HP_MLP_W) — no change
- Swept 0.06-0.20 with new blend_alpha=0.75. w=0.09 marginally better (fc_pearson 0.6497 vs 0.6497 at 0.12, spearman_lfc 0.8406 vs 0.8399). Differences within noise. Kept w=0.12.

## full_v19 — current best
All changes from this session combined:
- Delta cell regression (HP_DELTA_CELL_REG=0.12)
- Calnet delta feature (HP_CALNET_DELTA=1, 7 features)
- Calnet ep=800
- Calnet blend weight w=0.5
- NLP shrinkage disabled

| Metric | Previous best | full_v19 | Gain |
|--------|-------------|---------|------|
| fc_pearson | 0.649 | 0.649 | — |
| auprc | 0.800 | 0.800 | — |
| delta_pearson | 0.874 | 0.876 | +0.002 |
| spearman_lfc | 0.837 | 0.839 | +0.002 |
| pr_auc | 0.766 | 0.773 | +0.007 |
| de_overlap | 0.642 | 0.654 | +0.012 |
| spearman_sig | 0.947 | 0.958 | +0.011 |

## Calnet blend weight retune (HP_CALNET_W=0.5) — success
- **Hypothesis**: The calnet blend weight w=0.8 was tuned with the old 5-feature calnet. With 7 features (including delta), the calnet may benefit from a different weight.
- **Change**: Swept HP_CALNET_W from 0.5 to 1.0 in increments of 0.05-0.1.
- **Results**: Clear monotonic trend — lower w → better de_overlap, pr_auc, spearman_sig:
  | w | auprc | de_overlap | pr_auc | spearman_sig |
  |---|-------|-----------|--------|-------------|
  | 0.50 | 0.800 | **0.654** | **0.773** | **0.958** |
  | 0.60 | 0.801 | 0.651 | 0.769 | 0.957 |
  | 0.70 | 0.802 | 0.648 | 0.770 | 0.956 |
  | 0.80 | 0.802 | 0.645 | 0.767 | 0.954 |
  | 0.90 | 0.801 | 0.642 | 0.766 | 0.952 |
  | 1.00 | 0.797 | 0.634 | 0.758 | 0.948 |
- **Verification**: 3 runs at w=0.5: de_overlap 0.6539/0.6545/0.6545, pr_auc 0.7732/0.7731/0.7731. Extremely consistent.
- **Interpretation**: The 7-feature calnet is more capable but also more prone to miscalibration. At w=0.8, the calnet dominates the prediction and its errors compound (especially for per-gene FDR thresholding). At w=0.5, the regression signal provides a stabilizing anchor — the regression predictions have different error structure (systematic rather than per-gene), which partially cancels calnet errors when blended. The de_overlap improvement (+0.009) is the largest single-experiment gain in this metric. The auprc cost (-0.002) is minimal because AUPRC measures ranking quality averaged over genes, where the calnet's advantage is strongest.
- **Key learning**: More features in the calnet require LESS calnet weight, not more. The calnet's per-gene calibration is most valuable when blended conservatively with the regression's global structure.
- **Next**: Retune HP_CALNET_W whenever calnet features change — the relationship between feature count and optimal blend weight is non-trivial. Consider whether even more calnet features + even lower w unlocks further gains.

### NLP MLP for p-value ALS residuals (HP_NLP_MLP_W) — abandoned
- **Hypothesis**: MLP on NLP ALS residuals (same pattern as FC Global MLP) captures interaction effects for p-values.
- **Results**: Completely neutral — the regression overwrites the ALS+MLP prediction for 3655/3673 covered observations. The MLP only affects 18 uncovered observations.
- **Key learning**: NLP ALS improvements are irrelevant because the regression dominates. Only the calnet and regression matter for NLP.

### Calnet gene delta std feature (HP_CALNET_GENE_DSTD) — abandoned
- **Results**: Neutral. gene_fc_std already captures per-gene effect size variability.

### NLP-guided FC shrinkage retune — disabled
- **Hypothesis**: With the new calnet (7-feature, w=0.5), shrinkage parameters need retuning.
- **Results**: floor=0.95→0.97→0.99→1.0: spearman_lfc increases monotonically (0.836→0.837→0.839→0.839). Other metrics flat.
- **Decision**: Disabled shrinkage (HP_NLP_SHRINK=0). Shrinkage distorts FC rankings for FDR-significant genes. The original +0.00015 fc_pearson benefit is within noise and outweighed by the -0.002 spearman_lfc cost.

## full_v18 (calnet delta feature + ep=800) — success
- **Hypothesis**: |delta_als| (ALS-predicted absolute delta) provides complementary effect-size information to |FC| for the calnet. Deltas capture linear-scale expression changes (dominated by high-expression genes) while FC captures log-scale fold changes (all genes weighted equally). The calnet can use both effect-size signals to better calibrate significance predictions.
- **Changes**:
  - HP_CALNET_DELTA=1: Per-gene |delta_als| as calnet feature 7. Computed from delta ALS before the calnet.
  - HP_CALNET_EP=800: More epochs for the expanded 7-feature space.
- **Results**: auprc=0.802 (+0.002), pr_auc=0.767 (+0.001), de_overlap=0.645 (+0.003), spearman_sig=0.954 (+0.007). No regressions.
- **Epoch sweep**: ep=700→0.8018 AUPRC, ep=800→0.8019, ep=900→0.8019. Marginal improvement at 800+.
- **Interpretation**: |delta| provides an orthogonal signal about effect size that |FC| alone misses. For high-expression genes, |delta| is large even when |FC| is small (since delta = mean_treated - mean_control, dominated by high-expression genes). This helps the calnet identify genes where statistical power is high but FC-based significance is ambiguous. The improvement in de_overlap (+0.003) and spearman_sig (+0.007) confirms that cross-target features can help significance prediction even though they hurt drug similarity (cf. enriched FC regression).
- **Key learning**: Cross-target information is useful at the calnet level (per-gene calibration) but harmful at the regression level (drug similarity). The calnet has a clean per-(obs, gene) structure where extra features directly inform the prediction. The regression's drug-drug similarity is a more delicate computation where orthogonal noise distorts the kernel.
- **Next**: Retune HP_CALNET_W for the 7-feature calnet — the previous w=0.8 was tuned for 5 features.

## full_v17 (delta cell regression) — success
- **Hypothesis**: Apply cell-as-linear-combination regression to delta prediction, as it already works for FC (HP_CELL_REG=0.05) and NLP (HP_NLP_CELL_REG=0.15). For each holdout cell, learn cell combination weights from common drugs' delta values, then predict holdout cell's deltas as weighted average.
- **Change**: HP_DELTA_CELL_REG=0.12 — after delta drug regression, run cell regression on deltas and blend at alpha=0.12.
- **Results**: delta_pearson=0.876 (up from 0.874, +0.002). All other metrics unchanged.
- **Alpha sweep**: 0.05→0.8754, 0.08→0.8760, 0.10→0.8762, **0.12→0.8763**, 0.15→0.8761, 0.20→0.8749. Plateau at 0.08-0.15, sharp decline at 0.20.
- **Interpretation**: Cell-cell similarity provides complementary signal for delta prediction, as it does for FC and NLP. The optimal alpha (0.12) is between FC (0.05) and NLP (0.15), consistent with deltas being an intermediate target (linear-scale expression differences). Higher alphas hurt because the drug regression already captures most delta variance.
- **Next**: Add |delta| as a calnet feature to see whether delta information helps significance prediction.

### FDR direct prediction (HP_DIRECT_FDR) — abandoned
- **Hypothesis**: Predicting FDR directly (instead of p-values → BH correction) should improve FDR-dependent metrics (pr_auc, de_overlap) since evaluation thresholds on FDR, not p-values.
- **Change**: Used F_train instead of P_train for entire NLP pipeline (ALS, regression, calnet). Output F_pred directly without BH correction.
- **Results**: auprc 0.787 (-0.013), pr_auc 0.762 (-0.004), de_overlap 0.630 (-0.012), spearman_sig 0.957 (+0.010). Most metrics hurt.
- **Interpretation**: FDR is a cross-gene quantity — a gene's FDR depends on how many other genes are significant in that observation. The per-(obs,gene) calnet can learn p-value→significance mapping (per-gene) but NOT the FDR mapping (cross-gene). The p-value→BH approach is better because: (1) p-values are per-gene targets the calnet can learn, (2) BH correction handles cross-gene adjustment at test time using the predicted distribution. The one improvement (spearman_sig) shows per-observation significance counts are better calibrated in FDR space, but this doesn't justify the per-gene ranking regression.
- **Key learning**: Don't bypass the BH correction. P-values are the right prediction target for the calnet. Cross-gene adjustments should happen at inference time, not during model training.

### Enriched FC regression features (HP_ENRICH_FC) — abandoned
- **Hypothesis**: Concatenating delta and NLP data as extra features in the FC regression enriches drug-drug similarity by incorporating cross-target information. Drugs similar in FC, delta, AND NLP space should have more reliable combination weights.
- **Change**: Added `extra_features` parameter to `_compute_regression`. Concatenated standardized D_train and -log10(P_train) with Y_train for XTX computation, while keeping XTY based on FC only.
- **Results**: fc_pearson 0.631 (-0.018), de_overlap 0.603 (-0.039), pr_auc 0.754 (-0.012). Catastrophic regression.
- **Interpretation**: Confirms that drug similarity is best computed from the target data alone. Deltas and FC are weakly correlated (r ≈ 0.1), so delta/NLP features inject noise into the FC drug similarity. The RR truncation at rank 75 cannot effectively filter the cross-target noise from a 3x wider feature space. Consistent with all prior gene-direction modifications being neutral or harmful.
- **Key learning**: The drug similarity kernel should remain target-specific. Cross-target information doesn't improve drug similarity — it adds orthogonal noise.

## full_v15 (calnet cell mean + longer training) — success
- **Hypothesis**: (1) Cell mean ref_mean as calnet feature 6 gives cell-specific calibration — different cell lines have different statistical power, so the calnet should adapt its significance thresholds per cell. (2) More calnet training epochs (700 vs 500) let the calnet better learn the calibration mapping.
- **Changes**:
  - HP_CALNET_CELLMEAN=1: Per-observation cell mean log1p(ref_mean) as calnet feature. Captures overall cell expression level.
  - HP_CALNET_EP=700: Increased from 500 for better convergence.
- **Results**: Pearson=0.6492 (unchanged), AUPRC=0.7999 (up from 0.7989, +0.001).
- **Ablation**: ep=500+no cellmean→0.7989, ep=700→0.7992, cellmean only→0.7994, both→0.7999. Effects are independent and additive.
- **Interpretation**: Cell mean ref_mean encodes cell identity (high-expressing vs low-expressing cells) as a single scalar. Since statistical power scales with expression level, the calnet uses this to learn cell-specific calibration curves. The per-gene ref_mean already provided gene-level expression, but the cell-level mean adds cross-gene context (is this a generally high-expressing cell?). Longer training helps the calnet learn the expanded 6-feature space.
- **Next**: With the 6-feature calnet, revisit architecture knobs (ensemble size, width, depth) which may now behave differently.

### Calnet interaction/architecture sweeps — abandoned
- **Calnet FC×ref_mean interaction** (HP_CALNET_INTERACT): Added product feature → AUPRC 0.7993 (hurt). The 3-layer MLP learns this interaction implicitly.
- **Calnet wider** (HP_CALNET_H sweep): 128→0.7997, 256→0.7999, 384→0.7997, 512→0.7995. Current h=256 is optimal.
- **Calnet deeper** (HP_CALNET_DEPTH=4): AUPRC 0.7993 (hurt). 3 layers are sufficient for the calibration task.

### NLP-guided FC shrinkage — marginal success
- **Hypothesis**: Non-significant genes have noisier FC predictions. Gently shrinking their FC (sigmoid with floor=0.97) reduces noise in the per-observation Pearson correlation.
- **Results**: Pearson=0.6493 (up from 0.6492, +0.00015 consistent across 5 paired runs). AUPRC unaffected.
- **Grid search**: floor in {0.96-0.99}, shrink in {3,5,10}. Best: floor=0.97, shrink=5. Strong shrinkage (floor<0.9) catastrophically hurts.
- **Interpretation**: At most 3% FC reduction for completely non-significant genes. The effect is tiny because even non-significant genes have meaningful FC predictions — the drug regression captures small effects well. But the marginal improvement is consistent because some genes near the noise floor do benefit from shrinkage toward zero.

### Calnet ensemble (ens=2) — success
- **Hypothesis**: With the new 6-feature calnet (including cell_mean), inter-seed variance may be higher than with the previous 5-feature version, making ensemble averaging beneficial.
- **Results**: ens=1→0.7999, ens=2→0.8001, ens=3→0.8001, ens=5→0.8002. Ens=2 gives +0.0002 AUPRC consistently (verified across 3 runs: 0.80013, 0.80015, 0.80014).
- **Interpretation**: The cell_mean feature added enough learning variance that different seeds capture slightly different calibration patterns. Averaging two seeds reduces this variance. Diminishing returns beyond ens=2 suggest the variance is small. Set ens=2 as new default.

## full_v14 (NLP cell regression) — success
- **Hypothesis**: Apply the cell-as-linear-combination approach (from FC cell regression in v13) to NLP/p-value prediction. For each holdout cell, learn cell combination weights from common drugs' NLP values, then predict holdout cell's NLP as weighted average. Blend with existing NLP drug regression predictions.
- **Change**: HP_NLP_CELL_REG=0.15 — after NLP drug regression, run cell regression on NLP values and blend at alpha=0.15.
- **Results**: Pearson=0.6492 (unchanged), AUPRC=0.7989 (up from 0.7968, +0.002).
- **Alpha sweep**: 0.02→0.797, 0.05→0.798, 0.08→0.799, 0.10→0.799, 0.12→0.799, 0.13→0.799, 0.14→0.799, **0.15→0.799**, 0.20→0.799, 0.25→0.798, 0.30→0.796, 0.40→0.792. Broad plateau 0.10-0.20, sharp decline past 0.25.
- **Interpretation**: Cell-cell similarity is more informative for significance than for fold change (optimal alpha 0.15 vs 0.05). P-value patterns are more consistent across cells (as also evidenced by the NLP CWT finding in v13), so the cell regression captures real complementary signal. Optimal alpha is 3x larger than for FC because the NLP drug regression leaves more cell-specific residual structure that cell regression can correct.
- **Next**: Apply cell regression to the third target (deltas) to check whether the pattern holds across all three targets.

### Bilinear MLP interaction features (HP_MLP_BILINEAR) — abandoned
- **Hypothesis**: Adding element-wise product of projected treatment and cell embeddings as explicit bilinear interaction features could help the MLP capture multiplicative cell×treatment interactions.
- **Results**: Pearson 0.6492 (unchanged). The MLP already captures bilinear interactions through its hidden layers.

### Calnet FC-NLP correlation feature (HP_CALNET_CORRFEAT) — abandoned
- **Hypothesis**: Per-gene correlation between |FC| and NLP from training data as calnet feature 6 — tells the calnet which genes' significance is predictable from FC magnitude.
- **Results**: AUPRC 0.7983 (slight regression from 0.7989). The calnet already learns the FC→NLP relationship per gene via its existing features (|FC|, gene_mean_nlp, gene_fc_std).

### P-value gene weights for FC regression (HP_PVAL_GW) — abandoned
- **Hypothesis**: Weight genes by mean significance (-log10(p)) in drug regression XTX to focus drug similarity on reliably-measured genes.
- **Results**: alpha=1→0.6492, alpha=5→0.6492, alpha=10→0.6492. Completely flat across all strengths.
- **Interpretation**: Drug-drug similarity is robust to gene weighting (confirmed for variance, sparse selection, and now significance-based weights). The 2000-gene aggregation produces a stable drug similarity regardless of which genes are emphasized.

### Residual regression (HP_RESID_REG) — abandoned
- **Hypothesis**: Predict ALS residuals (Y - mu - cell_eff - treat_eff) instead of raw FC. Focus regression on interaction signal.
- **Results**: Pearson 0.630 (catastrophic -0.019 drop). Regression needs the additive structure in FC to compute meaningful drug similarity. Removing it destroys the signal that makes drug combination weights work.

### FiLM-conditioned MLP (HP_MLP_FILM) — abandoned
- **Hypothesis**: Cell embedding modulates treatment signal via scale/shift (FiLM), a better inductive bias than concatenation for "cell identity shapes treatment response."
- **Results**: Pearson 0.6492, AUPRC 0.799 (identical to baseline). Confirms the interaction signal bottleneck is in the data/features, not the network architecture. Multiple architectures (concat, bilinear, FiLM) all extract the same +0.002 signal.

### Per-gene variance rescaling (HP_VAR_RESCALE) — abandoned
- **Hypothesis**: Ridge regression shrinks predictions toward the mean. Rescaling per-gene prediction variance to match training variance could improve Pearson.
- **Results**: All strengths hurt (0.25→0.6486, 0.5→0.6458, 1.0→0.6393). Even with clipping to [0.5, 2.0], rescaling amplifies noise for poorly-predicted genes. Per-gene prediction quality varies too much for uniform rescaling.

## full_v13 (cell regression + NLP cell weight temp) — success
- **Hypothesis**: Two independent improvements: (1) Cell-as-linear-combination regression provides complementary signal to drug regression for FC prediction. (2) Higher cell weight temperature for NLP regression gives more uniform cell weights, better suited for significance patterns which are more consistent across cells.
- **Changes**:
  - HP_CELL_REG=0.05: For each test drug, learn cell combination weights from common drugs, then predict holdout cell's FC as weighted average of training cells' FC. Blend at 5% with existing FC prediction.
  - HP_NLP_CWT=0.50: Use temperature 0.50 (vs 0.18 for FC) for cell weights in NLP regression, giving more uniform weights.
- **Results**: Pearson=0.6492 (up from 0.649), AUPRC=0.7968 (up from 0.796). Consistent across 3 runs.
- **Alpha sweep**: Cell reg optimal at alpha=0.05 (0.02→0.649, 0.05→0.6492, 0.10→0.649, 0.15→0.648, 0.20→0.647)
- **NLP CWT sweep**: Optimal at temp=0.50 (0.18→0.796, 0.25→0.7964, 0.35→0.7965, 0.50→0.7966, 1.0→0.7964)
- **Interpretation**: (1) Drug regression captures drug-drug similarity (FC(hc,dy) from FC(hc,dx)). Cell regression captures cell-cell similarity (FC(hc,dy) from FC(c,dy)). These are complementary signals. At alpha=0.05, the cell regression adds a small correction. Higher alphas hurt because the drug regression is more informative. (2) NLP significance patterns are more consistent across cells than FC patterns, so more uniform cell weights (higher temp) help the NLP regression generalize.
- **Next**: Apply the cell-as-linear-combination approach to NLP/p-value prediction (same pattern, NLP target).

### Per-gene-group regression (HP_GENE_GROUPS) — abandoned
- **Hypothesis**: Separate drug combination weights per gene group (by FC variance) could capture gene-dependent drug similarity.
- **Results**: K=4→Pearson 0.646, K=2→0.648 (baseline 0.649). Consistently hurt.
- **Interpretation**: Drug-drug similarity is a global property — drugs that act similarly do so across most genes. Per-group XTX (22000 vs 88000 features) is noisier. RR truncation at rank 75 already handles dimensionality reduction. Gene-direction splitting is redundant and harmful.

### FC-informed NLP regression via W-matrix transfer (HP_W_TRANSFER_MU) — abandoned
- **Hypothesis**: Use FC drug combination weights (W_fc) as a regularization prior for NLP regression: W_nlp = (XTX + (λ+μ)I)^{-1}(XTY + μ·W_fc).
- **Change**: Modified _compute_regression to accept W_prior and W_prior_mu parameters. FC regression returns W matrices (return_W=True), passed as prior to NLP regression.
- **Results**: mu=0.1→Pearson 0.6488, mu=10→0.6488, mu=50→0.6489. AUPRC unchanged (~0.796). All within noise.
- **Interpretation**: The FC and NLP drug combination weights, while correlated, capture sufficiently different structure that constraining NLP toward FC doesn't help. The NLP regression already converges well on its own (no RR truncation, full-rank solution).

## Additional experiments tried (all abandoned)

### Gene-weighted p-value regression (HP_PREG_GW)
- Weight genes by ref_mean in p-value regression normal equations (more expressed → more reliable p-values)
- Results: alpha=0.3→5.0 gave AUPRC 0.787-0.791 — no improvement, hurts at high alpha
- Interpretation: Gene weighting distorts drug similarity without improving p-value predictions

### Deeper/wider calnet architectures
- Deep (4-layer + dropout): AUPRC 0.792 — same as shallow
- h=512: AUPRC 0.791 — same as h=256
- 8 features (added gene_frac_sig, gene_nlp_std, gene_mean_ref): AUPRC 0.790 — extra features add noise
- Interpretation: The calnet bottleneck is data/features, not model capacity. 5 features and 3 layers are sufficient.

### FC calibration network (HP_FCCAL)
- Train calnet on (FC_als, ref_mean, gene_stats) → actual FC, analogous to p-value calnet
- Results: w=0.1→0.3 gave Pearson 0.648-0.649 — hurts or neutral
- Interpretation: FC doesn't have expression-level dependence the way p-values do. The ALS → actual FC mapping is too noisy for a simple calnet.

### Per-gene adaptive blend alpha
- Vary blend alpha per gene based on gene's mean |FC| (higher FC → more regression)
- Results: adapt=0.1-0.3 gave Pearson 0.648-0.649 — slightly hurts
- Interpretation: High-FC genes don't systematically benefit more from regression

### Gene-weighted regression (abandoned earlier)
- Weight genes by p-value significance in FC regression
- Results: no improvement across 5 splits
- Interpretation: The regression's drug combination weights are robust to gene weighting. The gene dimension is already well-determined (1995 genes × 45 cells).

### LOO regression features for calnet (HP_LOO_CALNET) — abandoned
- **Hypothesis**: Calnet trains on ALS features (weak) but receives regression-quality features (strong) at test time. LOO regression predictions at train time should reduce this mismatch and improve calibration.
- **Change**: For each training cell, run _compute_regression with that cell held out to get regression-quality FC and NLP predictions. Use these as calnet training features instead of ALS additive predictions.
- **Results**:
  - v1 (all cells in LOO data): coverage 37.7% → AUPRC 0.794 (hurt). Test cells' limited treatment sets constrained common drug sets.
  - v2 (exclude test cells from LOO data): coverage 95.8% → AUPRC 0.785 (hurt more). Pearson 0.649 unchanged.
- **Interpretation**: The train/test feature mismatch is a FEATURE, not a bug. The calnet trained on weak ALS features learns to "boost" predictions. At test time, when it receives stronger regression features, this boosting amplifies the better signal, producing superior results. Reducing the mismatch removes this beneficial implicit denoising effect. This is analogous to training with data augmentation (noise) improving generalization.
- **Key learning**: Do NOT try to reduce the calnet's train/test feature gap. The ALS→regression quality gap is load-bearing.

## NLP ALS model improvements — abandoned
- **NLP per-treatment correction** (HP_NLP_CORR2, analogous to FC correction2): alpha=0.5→0.7961, alpha=1.0→0.7960 — no improvement. The NLP regression dominates; improving ALS base model is irrelevant.
- **NLP-based cell similarity for p-value regression** (HP_NLP_CELLSIM=1): AUPRC 0.7930 — worse. FC-based cell similarity is better for NLP regression (cells with similar FC patterns also have similar NLP patterns; FC similarity is a more robust proxy).

## Calnet architectural variations (ensemble, treatment feature) — abandoned
- **Results**:
  - 3-calnet ensemble (HP_CALNET_ENS=3): AUPRC 0.7963 — same as baseline (low variance, not worth 3x cost)
  - Treatment mean NLP feature (HP_CALNET_TREATFEAT=1, 6 features): AUPRC 0.7946 — worse
- **Interpretation**: The calnet is already stable (low variance), so ensembling doesn't help. Treatment-level NLP is already captured by NLP_als (= mu_nlp + treat_eff + cell_eff), making the treatment mean feature redundant.

## Calnet loss function variations (HP_CALNET_BCE, HP_CALNET_THRESH, focal loss) — abandoned
- **Hypothesis**: MSE loss on NLP is a poor proxy for AUPRC. BCE loss (binary classification), threshold-proximity weighted MSE (focus on NLP≈1.301), or focal loss might directly optimize what AUPRC measures.
- **Results**:
  - BCE (HP_CALNET_BCE=1, w=0.8, ep=500): AUPRC 0.796-0.797 across runs — within noise of baseline (0.796)
  - BCE different w: w=0.6→0.7964, w=0.9→0.7954, w=1.0→0.7952 — all ≤ baseline
  - BCE ep=300: 0.7967, ep=700: 0.7962 — same range
  - Threshold-proximity weighted MSE (HP_CALNET_THRESH=1, sigma=0.5): AUPRC 0.795648 — worse
  - Focal loss (gamma=0.5): 0.7949, focal (gamma=2): 0.7959 — worse or neutral
- **Interpretation**: MSE loss on NLP is already an effective proxy for AUPRC. The calnet's improvement comes from learning the expression-level→NLP relationship, not from the specific loss function. BCE loses the magnitude information (predicts binary significance instead of NLP), which slightly hurts. The 5 features at the threshold (p≈0.05) don't cluster clearly enough for focal loss to help.

## Gene-feature selection/weighting for FC regression (HP_EXPR_AWARE, HP_SPARSE_K) — abandoned
- **Hypothesis**: Computing drug similarity over all 2000 genes dilutes signal from informative genes. Either (a) weighting genes by held-out cell's expression profile (expression-aware similarity) or (b) selecting top-k high-variance genes should give cleaner drug similarity.
- **Changes tested**:
  - HP_EXPR_AWARE=1: weight XTX by log1p(ref_mean[holdout_cell]) per gene (per-cell weights). With Y weighted too: Pearson 0.635, AUPRC 0.795. With X-only + clipped (max 2x): Pearson 0.647, AUPRC 0.795.
  - HP_SPARSE_K=500: binary top-500 gene mask by FC variance. Pearson 0.646, AUPRC 0.795.
  - HP_SPARSE_K=1000: top-1000 genes. Pearson 0.648, AUPRC 0.796.
- **Interpretation**: Gene-feature weighting/selection in the regression normal equations consistently hurts or is neutral. The RR truncation (RR=75) already handles dimensionality reduction in the drug direction. Modifying the gene direction (across the feature dimension) distorts the drug-drug similarity matrix without adding information. The regression is already well-regularized; reducing features gives less, not more, signal.
- **Key learning**: Drug similarity is computed over a high-dimensional (n_nc * n_genes) feature space. The RR truncation focuses on the highest-variance drug directions. Additional gene filtering is redundant with this.

## Per-gene NLP standardization in p-value regression (HP_PREG_STD) — success
- **Hypothesis**: Different genes have very different NLP (-log10(p)) scales. Without standardization, high-NLP genes dominate the drug similarity computation. Per-gene standardization puts all genes on equal footing for determining drug combination weights.
- **Change**: Before p-value regression, standardize NLP per gene: z = (NLP - mean_g) / std_g. After regression, de-standardize: NLP_pred = z_pred * std_g + mean_g.
- **Results**: AUPRC 0.791 → **0.796** on tahoe_5_holdout. Consistent across 3 runs (0.796, 0.796, 0.796).
- **Multi-split**: Mean AUPRC 0.763 ± 0.044 (up from 0.759). Improvement on all 5 splits.
- **Interpretation**: NLP standardization improves the regression's drug combination weights by preventing high-significance genes from dominating. The regression finds better drug-drug similarity when all genes contribute equally. Combined with calnet: total AUPRC improvement from 0.777 → 0.796 (+0.019).

## ref_mean calibration experiments (Direction 2-4)
Several approaches for leveraging ref_mean (baseline expression) were tested:

### Approaches that didn't help FC prediction:
- **Linear ref_mean p-value calibration** (HP_REFCAL): per-gene linear correction NLP += beta * (log_ref_test - mean_ref). Result: 0.649 Pearson, 0.777 AUPRC — no change.
- **Ref_mean isotonic** (HP_REFISO): logistic regression with (|FC|, ref_mean) as features for significance prediction. Result: 0.649 Pearson, **0.759 AUPRC — hurt**.
- **Expression gating** (HP_REFGATE): shrink FC predictions for lowly-expressed genes via tanh(ref/scale). Result: 0.649 Pearson — no change.
- **Treatment × ref_mean interaction** (HP_REFINT): learn per-treatment slopes of FC vs ref_mean, correct test predictions. Result: 0.621-0.649 — hurt or neutral.
- **Ref_mean cell similarity** (HP_REFSIM): blend FC-residual similarity with ref_mean profile similarity in regression. Result: 0.649 — no change.
- **Ref_mean in MLP** (HP_MLP_REF): add ref_mean SVD embeddings to global MLP input. Result: 0.649 — no change.
- **Interpretation**: The model already captures cell identity through ALS cell effects. ref_mean provides redundant information with what cell effects and cell-weighted regression already encode. Post-hoc corrections add noise.

### Neural calibration network for p-values (HP_CALNET) — **success**
- **Hypothesis**: A small network trained on all (observation, gene) pairs can learn the expression-dependent FC → significance mapping that the regression misses. Specifically: for the same predicted |FC| and NLP, higher ref_mean should yield higher actual NLP (more statistical power).
- **Change**: Added calibration network in `full_model()` after computing P_pred from regression. The calnet takes (|FC_als|, NLP_als, log1p(ref_mean), gene_mean_nlp, gene_fc_std) as features and is trained to predict actual -log10(p) on training (obs, gene) pairs. At test time, it uses model predictions as features and blends its output with the regression P_pred.
- **Results**:
  | Config | Pearson | AUPRC | Notes |
  |---|---|---|---|
  | Baseline (no calnet) | 0.649 | 0.777 | |
  | h=128, ep=30, w=0.3 | 0.649 | 0.779 | First signal |
  | h=256, ep=100, w=0.6 | 0.649 | 0.781 | Bigger network helps |
  | h=256, ep=200, w=0.6 | 0.649 | 0.783 | More epochs help |
  | h=256, ep=500, w=0.6 | 0.649 | 0.787 | Sweet spot |
  | **h=256, ep=500, w=0.8** | **0.649** | **0.791** | **Best config** |
  | h=256, ep=1000, w=0.6 | 0.649 | 0.786 | Overfitting |
  | w=1.0 (pure calnet) | 0.649 | 0.788 | Needs regression blend |
- **Multi-split validation**: Mean AUPRC 0.759 ± 0.042 (up from 0.742 ± 0.043). Improvement consistent across all 5 splits.
- **Interpretation**: The calnet learns that statistical power depends on expression level — an effect the linear regression cannot capture. It provides a nonlinear mapping from (FC, expression level, gene identity) → significance that generalizes across cells. The train/test feature mismatch (ALS features for training, regression features for test) actually helps because regression NLP carries stronger signal.
- **Key design choices**: (1) Use actual model predictions at test time, not ALS; (2) Gene-level statistics as features enable gene-specific calibration without per-gene models; (3) w=0.8 blend keeps some regression signal while letting calnet calibrate.
- **Next**: Try improving the base regression itself to push calnet AUPRC higher.

## Upper bound analysis
- Oracle per-treatment mean (from test data): Pearson=0.530
- Oracle additive cell+treatment (from test data): Pearson=0.572
- **Our model: Pearson=0.649 — surpasses oracle additive**, meaning we predict real cell×treatment interactions
- Test data has rank ~791 for 90% variance — inherently high-dimensional
- The model is doing well relative to the signal structure

## multi-RR ensemble
- **Hypothesis**: Averaging regression predictions from different RR truncation ranks captures different levels of drug-drug structure.
- **Change**: Run regression with ranks [50, 75, 100, 150] and average predictions.
- **Results**: Pearson=0.6489 (vs 0.6487 single rank) — tiny but consistent improvement.
- **Interpretation**: Multi-resolution drug similarity captures complementary information. Marginal gain.

## FC × p-value mixing strategies (comprehensive)
Six strategies tested for combining FC and p-value information:
| Strategy | Best Pearson | Notes |
|---|---|---|
| **SoftMask** (soft p-value weighted FC) | 0.648 (w=0.25) | Competitive, doesn't stack with MLP |
| SigMask (binary p-value mask) | 0.646 (w=0.3) | Too aggressive zeroing |
| SignedP (sign(FC) × -log10(p)) | 0.647 (w=0.05) | Linear calibration too lossy |
| TwoStream (raw + masked blend) | 0.644 | Masked stream hurts non-sig genes |
| MultiView (NLP-calibrated + FC) | 0.649 (w=0.02) | Tiny complementary signal |
| ConfBlend (FC-NLP agreement) | 0.648 | Agreement doesn't predict accuracy |

Also tested:
- FC-boosted p-values (|FC| to improve AUPRC): hurt (0.735-0.761 vs 0.777)
- NLP treatment embeddings in pipeline: pipeline hurt (0.616), no effect in blend (0.649)
- Joint FC+NLP ALS: dilutes FC structure (0.616-0.621)

**Key finding**: FC and -log10(p) capture largely overlapping drug similarity structure. Mixing them at the regression level doesn't improve over using each independently. The current architecture (separate FC and NLP regressions) is near-optimal for this data.

## MLP variants tested
- **MLP alone (no ridge)**: 0.403 Pearson — far worse, ridge regression is essential
- **Per-cell MLP**: 0.646 — overfits with only ~400 samples per cell
- **Bilinear features**: 0.649 — no improvement over concat features
- **P-value MLP correction**: 0.774 AUPRC — hurts vs pure regression (0.777)
- **Residual regression (ALS residuals)**: 0.642-0.646 — regression on ALS residuals captures noise
- **kNN residual correction**: 0.643-0.646 — additive residuals too noisy
- **P-value informed FC (FC+NLP concat)**: 0.644-0.647 — too many features, noisy
- **Attention-based drug regression**: 0.642-0.646 — nearest-neighbor in embedding space weaker than ridge
- **Learned per-observation blend weights**: 0.649 — per-obs correlation doesn't add info
- **MLP noise augmentation**: 0.649 — stable but no improvement over base MLP
- **Deeper/wider MLP (hidden=1024, ep=300)**: 0.649 — bottleneck is signal, not model capacity
- **Joint FC+NLP ALS factorization**: 0.616-0.621 — NLP dilutes FC structure
- **Cross-modal regression (NLP features for FC prediction)**: 0.643-0.648 — NLP in XTX adds noise

## full_v6_gmlp (global MLP on residuals)
- **Hypothesis**: A global MLP trained on (treatment_embedding, cell_embedding) → FC residuals can capture nonlinear drug-cell interactions that the linear regression misses.
- **Change**: Added global MLP to `full_model()`. Trains on all ~53k (cell, treatment) pairs. Predicts residuals after ALS additive model, projected to PCA space. Blended with existing predictions at w=0.12.
- **Results**: Pearson=0.649 (up from 0.647), AUPRC=0.777 (unchanged). Best config: d_te=200, pca=150, 200 epochs, hidden=512, w=0.12.
- **Interpretation**: The MLP captures some nonlinear signal that the linear regression misses, even with only +0.002 gain. The gain is consistent across weight values (0.05-0.15 all help). Higher capacity (hidden=1024) doesn't help further — the bottleneck is signal, not model capacity.
- **Next**: Try replacing the per-held-out-cell drug regression with an MLP-based approach. Try deeper or wider architectures. Consider attention-based drug similarity.

## full_v5_nocenter (no-center FC regression)
- **Hypothesis**: Removing mean-centering from FC regression might better preserve treatment-specific effects.
- **Change**: Set center=False in FC regression (HP_FC_CENTER=0).
- **Results**: Mean Pearson across 5 splits: 0.637 (up from 0.635 with centering). Per-split: 0.647/0.625/0.655/0.633/0.625. AUPRC unchanged at 0.742.
- **Interpretation**: Small but consistent improvement. No-center preserves treatment magnitude information that centering removes.
- **Next**: Try further structural improvements to the regression or pipeline.

## multi_split (cross-split validation)
- **Hypothesis**: Validate model robustness across different held-out cell line sets.
- **Change**: Added `multi_split_eval()` that runs `full_model` on all 5 splits (tahoe_5 through tahoe_9).
- **Results**:
  | Split | Pearson | AUPRC |
  |---|---|---|
  | tahoe_5 | 0.646 | 0.777 |
  | tahoe_6 | 0.622 | 0.677 |
  | tahoe_7 | 0.654 | 0.796 |
  | tahoe_8 | 0.632 | 0.748 |
  | tahoe_9 | 0.621 | 0.711 |
  | **Mean** | **0.635 ± 0.013** | **0.742 ± 0.043** |
- **Interpretation**: Model is consistent across splits. Variation is real (different cell lines have different predictability). Split 7 is easiest, split 6/9 are hardest. The ~0.04 AUPRC std suggests room for improvement on harder splits.
- **Next**: Focus on improving the weaker splits. Check if different hyperparameters work better per-split.

## Abandoned ideas (tested, didn't help)
- **Bilinear ridge on pdex**: hurts at any weight (PCA probe 0.187 too low)
- **Cell-weight disabling**: slightly worse for FC
- **Denoised training data for regression**: raw data is better (denoised loses per-obs signal)
- **FC ensemble (multi-rank averaging)**: no improvement over single rank
- **Gene-space smoothing**: loses signal (FC drops to 0.640 at k=200/w=0.3)
- **Interaction blend mode**: worse than alpha blend
- **Shared-W regression for FC+p-values**: p-values need separate W (0.730 vs 0.777)
- **Per-gene isotonic calibration of ALS NLP**: worse than raw regression (0.749 vs 0.777)
- **200-epoch NN**: improves pipeline (0.624→0.627) but doesn't help blend (still 0.646)
- **NLP calibration via training ALS**: worse (0.765 vs 0.777)

## full_v3 (p-value regression without RR truncation)
- **Hypothesis**: The p-value regression might benefit from keeping full rank (no RR truncation), since significance patterns may be higher-dimensional than fold change patterns.
- **Change**: Disabled RR truncation for p-value regression (HP_PREG_RR=0) while keeping RR=75 for FC regression.
- **Results**:
  | Config | pearson | auprc |
  |---|---|---|
  | full_v2 (RR=75 for p-values) | 0.646 | 0.755 |
  | **full_v3 (no RR for p-values)** | **0.646** | **0.777** |
  | RR=200 for p-values | 0.646 | 0.773 |
- **Interpretation**: P-value regression needs full rank to capture the high-dimensional significance structure. This is a +0.022 AUPRC gain from a one-line change. Lambda doesn't matter (tested 0.1, 1.0, 10.0). FC unaffected by p-value regression settings.
- **Next**: Try new structural improvements — e.g., multi-split averaging, improved neural network, or fundamentally different regression approaches.

## improve_fc (FC tuning sweep)
- **Hypothesis**: Various pipeline and regression hyperparameters might improve FC on pdex data.
- **Change**: Tested: cell-weighting on/off, RR rank sweep, lambda sweep, temperature, soft_knn, interaction blend mode, no-center regression.
- **Results**: FC prediction is very stable at 0.646 across all configs. No significant improvement found. Cell-weighting helps slightly (+0.003 vs unweighted). RR=75 is optimal for FC. Interaction blend mode doesn't help.
- **Interpretation**: The FC prediction has plateaued at ~0.646 with current architecture. Diminishing returns from hyperparameter tuning.

## full_v1 / full_v2 (combined FC + p-value model)
- **Hypothesis**: Combining the FC blend (alpha=0.85) with p-value prediction should give good results on both metrics. Using drug regression on -log10(p) should capture cross-cell drug similarity for significance.
- **Change**: Added `full_model()`: runs blend for FC, ALS for p-values, and optionally drug regression on -log10(p). Tested various p-value blend weights.
- **Results**:
  | Config | pearson | auprc |
  |---|---|---|
  | full_v1 (ALS+iso only) | 0.646 | 0.733 |
  | full_v2 preg=0.4/als=0.5/iso=0.1 | 0.646 | 0.747 |
  | full_v2 preg=0.7/als=0.25/iso=0.05 | 0.646 | 0.753 |
  | full_v2 preg=0.8/als=0.15/iso=0.05 | 0.646 | 0.753 |
  | **full_v2 preg=1.0 (pure regression)** | **0.646** | **0.755** |
- **Interpretation**: Drug regression on -log10(p) is the strongest p-value predictor (0.755 AUPRC), beating ALS (0.725) and isotonic (0.600) by wide margins. Cross-cell drug similarity captures significance patterns very well. The ALS/isotonic components don't add value when regression is available.
- **Next**: Improve FC prediction — the 0.646 has plateaued with current pipeline+regression blend. Ideas: tune regression hyperparams more, improve the pipeline's similarity correction, or try new approaches for the interaction term.

## blend_pdex (pipeline + regression)
- **Hypothesis**: Blending the pipeline with drug regression (as was done for cell_eval) should improve FC prediction on pdex too.
- **Change**: Ran `blend_experiment` with p-value weighted pipeline (alpha=0.02) at various blend alphas.
- **Results**:
  | alpha | pearson_delta_mean |
  |---|---|
  | 0 (pipeline only) | 0.624 |
  | 0.50 | 0.642 |
  | 0.65 | 0.645 |
  | 0.75 | 0.646 |
  | **0.85** | **0.646** |
  | 0.90 | 0.646 |
  Lambda tuning (0.5, 1, 2, 5) and RR rank (50, 75, 100) had negligible effect.
- **Interpretation**: Regression dominates — alpha=0.85 gives the best FC prediction at 0.6465. This is a +0.022 gain over pipeline-only (0.624). The regression captures complementary cross-cell drug similarity structure.
- **Next**: Integrate p-value prediction into the blend. Try improving the regression itself (e.g. with p-value weighting). Try improving the pipeline further before blending.

## bilinear_ridge (abandoned)
- **Hypothesis**: Enabling the bilinear ridge phase (currently skipped at PCA probe=0.187 < 0.4) might capture interaction structure.
- **Change**: Made bilinear threshold and weight configurable. Increased regularization from 0.005 to 1.0 to avoid Cholesky failure.
- **Results**: w_b=0.3 → 0.549 (terrible), w_b=0.05 → 0.623 (still worse than 0.624 baseline)
- **Interpretation**: The bilinear ridge interaction model is not useful for pdex data at this scale. The residual structure isn't well-captured by outer-product cell×treatment features. Abandoned.

## pvalue_als (Phase 2: p-value prediction model)
- **Hypothesis**: ALS decomposition on -log10(p) should predict significance better than FC-magnitude isotonic regression, since it captures cell- and treatment-specific significance patterns beyond just effect size.
- **Change**: Added `pvalue_als_model()`: fits separate ALS on -log10(p) with per-cell similarity correction, optionally blended with FC-isotonic predictions.
- **Results**:
  | Method | auprc_p05 | pearson_delta_mean |
  |---|---|---|
  | FC-isotonic (baseline) | 0.600 | 0.621 |
  | **ALS-only** | **0.725** | 0.624 |
  | Blend 0.95/0.05 | 0.729 | 0.624 |
  | Blend 0.90/0.10 | 0.725 | 0.625 |
  | Blend 0.80/0.20 | 0.718 | 0.624 |
  | Blend 0.50/0.50 | 0.690 | 0.624 |
- **Interpretation**: The ALS p-value model dramatically beats isotonic (0.725 vs 0.600). Cell/treatment structure in significance is real and substantial. A tiny isotonic component helps (0.729 at 95/5) but mostly the ALS dominates. FC prediction also slightly improved to 0.624 via p-value weighting.
- **Next**: Improve the ALS p-value model — add more interaction correction components, or try joint embedding approach. Also integrate p-value prediction into the main blend experiment for the full pipeline.

## pvalue_weighted_als (alpha sweep)
- **Hypothesis**: Weighting ALS by p-value confidence (more weight to low-p genes) should improve FC prediction by reducing noise from non-significant measurements.
- **Change**: Added p-value weighting to static ALS path: `w = 1 + alpha * (-log10(p))`. Tested alpha ∈ {0.01, 0.02, 0.05, 0.1, 0.5}.
- **Results**:
  | alpha | pearson_delta_mean |
  |---|---|
  | 0 (baseline) | 0.621 |
  | 0.01 | 0.6246 |
  | **0.02** | **0.6245** |
  | 0.05 | 0.6242 |
  | 0.1 | 0.6225 |
  | 0.5 | 0.6156 |
- **Interpretation**: Gentle p-value weighting helps (+0.003 at alpha=0.02). Too much weighting (alpha≥0.1) hurts, likely because it over-focuses on the most significant genes and loses global structure. The improvement is small but consistent.
- **Next**: Set alpha=0.02 as default for future runs. Bigger gains likely from p-value prediction model.

## pvalue_naive / pvalue_marginal / pvalue_fc_isotonic
- **Hypothesis**: Establish AUPRC baselines for p-value prediction before building complex models. Naive (P=1) gives the floor. Marginal (per-gene mean P) captures gene-level priors. FC-isotonic tests whether |predicted FC| alone can predict significance.
- **Change**: Added `pvalue_baselines()` function to train.py. Uses the static_pdex pipeline FC predictions and maps them to p-value predictions via three strategies.
- **Results**:
  | Baseline | auprc_p05 | pearson_delta_mean |
  |---|---|---|
  | pvalue_naive (P=1.0) | 0.238 | 0.621 |
  | pvalue_marginal | 0.498 | 0.621 |
  | **pvalue_fc_isotonic** | **0.600** | 0.621 |
- **Interpretation**: FC magnitude is a strong predictor of significance (0.600 AUPRC), well above marginal (0.498). The naive floor is 0.238, meaning ~24% of gene-obs are significant at p<0.05. Per-gene isotonic regression captures the gene-specific relationship between effect size and significance.
- **Next**: Try using p-values as training weights for FC prediction (Direction 1). Also try a separate p-value model on -log10(p) to beat the isotonic baseline.

## Baselines (non-model)
| Baseline | pearson_delta_mean |
|---|---|
| global_mean | 0.161 |
| cell_mean | 0.264 |
| treatment_mean | 0.282 |
| additive (cell + treat - global) | 0.354 |

## static_pdex
- **Hypothesis**: Baseline run to establish fold change prediction on pdex data
- **Change**: Switched prepare.py from cell_eval to pdex data source; used raw fold_change (log2 FC)
- **Results**: pearson_delta_mean=0.621, auprc_p05=N/A
- **Interpretation**: Solid interaction signal — well above additive baseline (0.354). Not comparable to old cell_eval numbers (~0.9) due to different data scale. Bilinear phase skipped (PCA probe 0.187 < 0.4).
- **Next**: Establish p-value baselines; explore whether p-value weighting improves FC prediction

