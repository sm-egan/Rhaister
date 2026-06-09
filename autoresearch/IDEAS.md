# Research Ideas

Ideas for future experiments. Add new ideas at the top. Each idea should be self-contained enough to act on without re-reading the full experiment history.

The model now predicts three targets — expression deltas (D), log2 fold changes (Y), and FDR (F) — evaluated on six State metrics. The main research direction is building a **cohesive multi-target model** where all targets share structure, rather than running independent pipelines. See PROGRAM.md for current results and architecture.

## Open

### Low-hanging fruit

#### Cell-adaptive blend weights predicted from ref_mean — DIAGNOSTIC INCONCLUSIVE
- Diagnostic attempted but flawed: held-out cells' data remains in ALS training, so pipeline leaks test info via per-cell similarity correction. All cells show best_alpha=0 (pipeline-only), which is an artifact. Proper diagnostic requires drug-level holdout (a custom split), which is expensive. Idea not ruled out but hard to test cheaply.

#### Composite metric-aware loss (from Virtual Cell Challenge winners)
- **Idea**: Replace standard MSE training loss with a composite loss that directly optimizes the evaluation metrics. VCC 1st place used PDS + DES + small-weight MAE. For your State metrics: combine per-gene MSE with a differentiable approximation of AUPRC and/or spearman correlation on significant genes.
- **Why**: VCC winners found that loss function design mattered as much as architecture choice. Standard MSE optimizes average error across all genes equally, but your metrics weight significant genes heavily (spearman_lfc_sig, pr_auc, de_overlap). A metric-aware loss directly targets what you're evaluated on.
- **Approach**: Start simple: add a differentiable ranking loss on significant genes. For each training observation, use the training FDR to identify DE genes, then add a pairwise ranking loss encouraging correct ordering of FC predictions among DE genes. Weight: `loss = (1-β)*MSE + β*ranking_loss_on_DE_genes`. Tune β.
- **Expected signal**: Small-moderate for spearman_lfc and de_overlap. Risk of hurting fc_pearson (genome-wide metric) if ranking loss dominates.
- **Lit**: Virtual Cell Challenge 2025 winning solutions (Arc Institute wrap-up); "Diversity by Design" (ICML 2025).

### Novel ideas (genuinely untried)

#### Structured prediction errors as a correctable signal
- **Idea**: The pipeline's prediction errors on training data have gene-gene covariance structure — when gene A is over-predicted, genes B and C tend to be under-predicted in a specific pattern (because the model systematically mis-allocates signal between correlated genes). Learn this error covariance and use it to correct test predictions.
- **Why nobody has tried this**: Everyone models the DATA structure (low-rank gene expression). Nobody models the ERROR structure (low-rank prediction errors). But prediction errors are not random — they reflect systematic model biases tied to regulatory relationships the model doesn't capture. The CIPHER paper (2025) showed gene-gene covariances contain 11× more information than gene variances alone. The same principle applies to error covariances.
- **Why it might work**: Your ALS + regression captures the dominant additive + drug-similarity structure. The residuals contain STRUCTURED error: if the model under-estimates a kinase pathway gene, it probably under-estimates other kinase genes too. This structured error can be predicted from the error pattern on observed genes.
- **Approach**: (1) Compute leave-one-out training residuals R_loo (or cross-fitted residuals from the cross-fitting idea). (2) Compute gene-gene error covariance Σ_err = R_loo^T R_loo / n. (3) Low-rank approximate Σ_err via SVD (keep top ~50 error modes). (4) For test predictions, there's no direct error observation. BUT: the drug regression weight vector w for each test observation tells you HOW the prediction was constructed. Observations constructed similarly (similar w vectors) should have similar error patterns. Cluster training observations by their w vectors, compute cluster-specific error biases, and subtract the bias for each test observation based on its w-cluster.
- **Expected signal**: Small. The error structure may be too noisy to exploit. But if there are systematic biases (e.g., the model consistently over-predicts kinase genes for certain drug classes), this catches them.

#### Test-time transductive refinement of cell embeddings
- **Idea**: At test time, you have ~735 training observations for each held-out cell line (responses to training drugs). Use these to REFINE the cell embedding specifically for the test cell, beyond what the global ALS provides. Then re-predict all test observations using the refined embedding.
- **Why this is different from what exists**: The current pipeline uses test cell training data in three ways: (1) ALS includes the test cell's training observations, (2) drug regression uses Y(test_cell, training_drugs) as the basis, (3) per-cell similarity correction uses test cell's residuals. But none of these REFINE the cell embedding. The cell embedding is computed from ALS on the full training matrix. You could fine-tune it.
- **Why nobody does this**: Standard models learn embeddings during training and freeze them at test time. The "transductive" idea of optimizing embeddings at test time is common in few-shot learning (MAML inner loop) but not applied to matrix factorization pipelines.
- **Approach**: After the full pipeline predicts Y_pred for all test observations of cell_c, compute the reconstruction error on cell_c's TRAINING observations: `err_train = Y(cell_c, D_train) - pipeline_pred(cell_c, D_train)`. Adjust the cell embedding by gradient descent on this reconstruction error: `ce_g[c] -= lr * d(err_train)/d(ce_g[c])`. With the refined embedding, re-predict test observations. This is 1-3 gradient steps, not full retraining.
- **Expected signal**: Small-moderate. The cell embedding is already pretty good (it's estimated from ~735 training observations). But the refinement specifically targets the residual error, which the global ALS may not capture perfectly for each individual cell.
- **Risk**: Overfitting to the training observations for one cell line. Regularize by constraining the update to be small (early stopping, small lr).

### Cross-field ideas (high priority)

#### IPCA conditional factor loadings (from quantitative finance)
- **Idea**: Model gene expression change as `Y_g(c,d) = g(z_c, z_d)^T · f_g` where g is a neural network mapping observable (cell, drug) features to K-dimensional factor loadings, and f_g are latent gene-response factors. This is Instrumented PCA (IPCA, Kelly/Pruitt/Su, J. Financial Economics 2019) — the foundational model for conditional asset pricing, applied to gene expression.
- **Why**: Your current architecture learns cell and treatment embeddings separately from the prediction of gene expression. IPCA says the factor loadings (how strongly each cell-drug combination activates each latent gene program) should be a direct function of observable features (ref_mean, drug target profiles). This makes the model fully inductive: for a new cell line with known ref_mean, you immediately get factor loadings without needing training data for that cell. The bilinear structure z'Γf is provably optimal under a latent factor DGP.
- **Approach**: Replace the current pipeline's separate embedding + regression steps with a single IPCA-style model. Network g takes [cell_ref_mean_SVD, drug_target_profile] → K-dimensional loading vector. Gene factors f are estimated jointly. The autoencoder extension (Gu/Kelly/Xiu, J. Econometrics 2021) allows g to be nonlinear while maintaining the factor structure. Open-source implementation: bkelly-lab/ipca on GitHub (255★).
- **Expected signal**: Moderate-high. This reframes the entire prediction problem as conditional factor pricing, which finance has shown consistently outperforms unconditional factor models. The key advantage over current architecture: factor loadings depend on *features*, not on entity identity, enabling true zero-shot prediction.
- **Lit**: IPCA (Kelly, Pruitt, Su, JFE 2019); Autoencoder Asset Pricing (Gu, Kelly, Xiu, J. Econometrics 2021); AI Asset Pricing with Transformers (Kelly et al., NBER 2025).

#### Shared regression weights across all genes (from econometrics)
- **Idea**: In `_compute_regression`, use a SINGLE set of drug-combination weights optimized jointly across all 1995 genes, rather than the current per-gene implicit optimization via the matrix normal equations.
- **Why**: Sun, Ben-Michael, Feller (Review of Economics and Statistics, 2025) prove that when multiple outcomes share a latent factor structure, using shared synthetic control weights across all outcomes strictly reduces bias compared to outcome-specific weights. Your 1995 genes share the same cell and treatment factors — gene-specific weights overfit to per-gene noise.
- **Approach**: Currently `_compute_regression` solves `W = (XTX + λI)^{-1} XTY` where Y is (n_dy × n_genes). The W matrix has separate columns per gene-block. Instead, solve for a scalar weight vector w ∈ R^{n_dx}: minimize `||Y_hc - Σ_i w_i Y_i||^2_F` subject to ridge penalty. This gives one weight per training drug, shared across all genes. The key: `XTX` becomes a scalar Gram matrix (n_dx × n_dx) computed over all genes jointly.
- **Expected signal**: Small but principled. Your current regression already pools across genes via the matrix multiplication, but the centering and reduced-rank truncation operate per-gene. Making the weights explicitly shared and then allowing per-gene correction as a second step is the theoretically optimal two-stage estimator.
- **Lit**: Sun, Ben-Michael, Feller (Rev. Econ. Stat. 2025); Athey et al. (JASA 2021) for the matrix completion unification.

#### TensoRF vector-matrix decomposition (from neural radiance fields)
- **Idea**: Decompose the (cell × treatment × gene) tensor using the VM (vector-matrix) factorization from TensoRF: `Y(c,d,g) = Σ_r [v^cell_r · M^{treat,gene}_r + v^treat_r · M^{cell,gene}_r + v^gene_r · M^{cell,treat}_r]`. Each mode gets a vector, each pair of modes gets a matrix. This is strictly more expressive than additive (mu + cell_eff + treat_eff) while much cheaper than full Tucker.
- **Why**: Current ALS decomposes into rank-1 additive terms (cell effects + treatment effects). VM decomposition adds three types of interactions: (1) per-cell modulation of treatment-gene effects, (2) per-treatment modulation of cell-gene profiles, (3) per-gene modulation of cell-treatment interactions. NeRF research (TensoRF, ECCV 2022) showed this captures 3D structure with dramatically fewer parameters than full 3D grids.
- **Approach**: Replace the ALS step with VM decomposition. Initialize v^cell from current cell_eff SVD, M^{treat,gene} from current treat_eff. Add the cross-mode matrices M^{cell,gene} and M^{cell,treat} as new learnable components. Optimize via alternating updates. The gene dimension is large (1995), so M^{cell,treat} (50 × 1137) is the cheapest new component. GitHub: apchenstu/TensoRF (1238★).
- **Expected signal**: Moderate. The question is whether the cross-mode interactions capture signal beyond what similarity correction and MLP already get.
- **Lit**: TensoRF (Chen et al., ECCV 2022); FastNeRF (Garbin et al., ICCV 2021).

#### Doubly robust cross-fitting for two-stage pipeline (from causal inference)
- **Idea**: Cross-fit the ALS → MLP two-stage pipeline. Split training data into K folds. For each fold, train ALS on the other K-1 folds, compute "honest" residuals on the held-out fold, then train the MLP/regression on these honest residuals.
- **Why**: Semenova et al. (Quantitative Economics, 2023) and Abadie et al. (2024) show that when a second-stage model (your MLP) is trained on residuals from a first-stage model (your ALS), overfitting in the first stage biases the second stage. Cross-fitting produces "honest" residuals that the first stage hasn't seen, preventing this leakage. The orthogonal learner converges faster when the CATE (interaction) function is simpler than the first-stage (additive) function — which is exactly your case.
- **Approach**: K=5 fold cross-fitting. For each fold: (1) train ALS on other 4 folds, (2) predict on held-out fold to get residuals, (3) collect all honest residuals. Train MLP on honest residuals. For test prediction, use the full-data ALS (all 5 folds). The cross-fitting only affects MLP training, not ALS for final prediction.
- **Expected signal**: Small. The current MLP weight is only 0.12, suggesting it captures little signal. But cross-fitting could allow a higher weight by reducing overfitting bias.
- **Risk**: 5× computational cost for ALS training. Could approximate with a single 80/20 split.
- **Lit**: Semenova et al. (QE 2023); Abadie et al. (2024); Chernozhukov et al. "Double/Debiased ML" (Econometrica 2018).

#### Concept bottleneck through pathway activities (from finance)
- **Idea**: Route all predictions through an interpretable bottleneck of K pathway activity scores. A nonlinear first stage maps (cell_features, drug_features) → K pathway activations. A linear second stage maps pathway activations → gene expression changes. Forces predictions to be interpretable as "drug X activates pathway Y in cell Z."
- **Approach**: First stage: MLP maps (ref_mean_SVD, treat_emb) → K-dim pathway activation vector. Second stage: linear W_gene (K × n_genes) maps activations → per-gene predictions. Train end-to-end. The bottleneck K controls expressiveness vs. interpretability (K=20-50 pathway programs).
- **Why**: Consensus-Bottleneck Asset Pricing (Jang et al., 2024) showed this architecture actually *improves* out-of-sample R² (10.5% vs 7.6% for unconstrained DNN) in finance by preventing overfitting to spurious feature combinations. The bottleneck acts as implicit regularization. For your problem, the pathway bottleneck prevents the model from memorizing cell-drug-specific patterns and forces generalization through biologically meaningful intermediates.
- **Expected signal**: Moderate. The bottleneck trades off capacity for generalization. With K=30 programs and 1995 genes, you reduce the problem to predicting 30 numbers instead of 1995.
- **Lit**: CB-APM (Jang et al., 2024); D-SPIN gene program networks (Jiang et al., Nature Methods pipeline). GitHub: yewsiang/ConceptBottleneck (251★).

#### Q-matrix pathway masking for neural interaction (from psychometrics)
- **Idea**: Add a binary Q-matrix mask to the neural interaction module, where Q_{d,k} = 1 if drug d targets pathway k. The interaction prediction becomes `Y = f_DNN(Q_d ⊙ (θ_cell - b_drug) ⊙ a_drug)` where θ is cell pathway sensitivity, b is drug activation threshold, a is drug effect magnitude. Only interactions through biologically relevant pathways contribute.
- **Why**: From Cognitive Diagnostic Models (NeuralCDM, 2019) in psychometrics: when predicting student performance on test items, masking the interaction by a Q-matrix (which skills each item requires) dramatically improves prediction and interpretability. Your current MLP has no such constraint — it can learn spurious interactions between any cell feature and any drug feature. The Q-matrix encodes "this drug cannot affect pathways it doesn't target."
- **Approach**: Build Q from DrugBank/ChEMBL target annotations: for each drug, which pathways (KEGG/Reactome) its targets belong to. Map cell features to pathway-level sensitivity scores θ (via ref_mean → pathway activity). The masked inner product Q_d ⊙ (θ_c - b_d) feeds into a small DNN with non-negative weight constraints (higher sensitivity = stronger response). Falls back to current MLP for drugs without Q annotations.
- **Expected signal**: Small-moderate. Depends on quality of drug-pathway annotations. The parameter reduction could help for drugs with few training observations.
- **Risk**: Incomplete pathway annotations could mask real interactions. Need a "residual" pathway for unannotated effects.
- **Lit**: NeuralCDM (Wang et al., 2019); Deep-IRT (Yeung, 2019); Q-Matrix CDM survey (2024).

#### Spectral unmixing: compositional response programs (from remote sensing)
- **Idea**: Model each (cell, drug) gene expression profile as a convex combination of K "pure response programs" (endmembers): `Y(c,d) = Σ_k α_k(c,d) · E_k` where α_k ≥ 0, Σ_k α_k = 1, and E_k are learned endmember spectra (gene programs). The encoder learns α from (cell, drug) features; the decoder weights ARE the endmember spectra.
- **Why**: Current additive model (cell_eff + treat_eff) assumes effects combine linearly without constraint. Spectral unmixing (used in hyperspectral remote sensing) instead says responses are *compositional* — activating one program suppresses others via the sum-to-one constraint. This matches biology: cells allocate transcriptional resources among competing programs (stress response vs. proliferation vs. differentiation). Physics-constrained autoencoders for Raman spectroscopy (PNAS 2024) show this works better than unconstrained NMF.
- **Approach**: Autoencoder where encoder maps (cell_emb, treat_emb) → softmax(α) of dimension K. Decoder is a linear layer with non-negative weights (the endmember matrix E, shape K × n_genes). Train to reconstruct Y_train. The softmax ensures sum-to-one; ReLU on E ensures non-negativity. For multi-target: three decoder matrices E_Y, E_D, E_NLP sharing the same α.
- **Expected signal**: Moderate. The constraint is strong — it assumes all responses live on a simplex in gene-program space. This may not hold for heterogeneous perturbations. But the shared α across targets is a very elegant multi-target solution.
- **Lit**: Physics-constrained autoencoders for spectral unmixing (Georgiev et al., PNAS 2024); D-SPIN orthogonal NMF (Jiang et al., 2023). GitHub: lloydwindrim/hyperspectral-autoencoders (119★).

### Within-field ideas (lower priority — perturbation prediction field has largely converged)

#### Tri-factorization: shared embeddings with per-target core matrices
- **Idea**: Replace independent ALS on Y, D, NLP with a coupled factorization: `Y_k ≈ G_cell · S_k · G_treat^T` where G_cell and G_treat are shared across all three targets but S_k is a small, target-specific core matrix. This is matrix tri-factorization (Zitnik & Zupan, IEEE TPAMI 2015).
- **Why**: Current ALS runs independently per target, learning separate cell and treatment effects that may miss shared structure. Shared G embeddings force the model to find representations that explain all three targets simultaneously. The per-target S_k (size rank_cell × rank_treat) absorbs scale differences, avoiding the normalization problem flagged in the "Shared ALS with concatenated targets" idea.
- **Approach**: After standard per-target ALS, do a joint re-estimation: SVD the per-target treat_eff/cell_eff matrices, then learn shared cell/treatment factors with target-specific projections. Use the shared factors as improved embeddings for downstream drug regression.
- **Expected signal**: Moderate. Cell effects should be highly shared across targets. Treatment effects less so (r≈0.1 between D and Y). But even partial sharing improves embedding quality.
- **Lit**: "The Structure is the Message" (Tan & Meyer, Cell Systems 2024); DFMF (Zitnik, TPAMI 2015); Deep Collective Matrix Factorization (Mariappan & Rajan, 2019). GitHub: david-cortes/cmfrec (125★), MarieRoald/matcouply (17★).

#### LoRA-style per-cell drug embedding adaptation
- **Idea**: Instead of global treatment embeddings used identically across cells, add a cell-specific low-rank adaptation: `te_cell = te_g + te_g @ (A_cell @ B_cell^T)` where A, B ∈ R^(d_t × r) with r << d_t (e.g., r=8) are predicted from cell embeddings by a small network.
- **Why**: Drug effects are cell-context-dependent (the whole interaction signal). Current cell_weights modulate regression with a scalar per cell-pair. LoRA-style adaptation modulates the drug embedding space itself — each cell gets a slightly different "view" of which drugs are similar. This is a richer form of cell-drug interaction.
- **Approach**: Train a small network `f(ce_g, ref_emb) → (A, B)` that outputs the low-rank perturbation per cell. Apply to treatment embeddings before similarity correction and drug regression. Train end-to-end with the neural interaction network.
- **Expected signal**: Moderate. The key question is whether cell-specific drug similarity structure exists beyond scalar cell_weights. The strong global drug regression performance suggests mostly global structure, but per-cell modulation might capture remaining interaction.
- **Lit**: LoRA (Hu et al., 2021); Magnitude Invariant Parametrizations for hypernetworks (Gonzalez Ortiz et al., 2023).

#### Heteroscedastic FC+variance joint prediction
- **Idea**: Replace the separate calnet with a joint prediction head that outputs both FC mean and per-gene variance from shared embeddings. The predicted variance maps to p-values since significance ≈ effect_size / sqrt(variance).
- **Why**: Currently FC and significance are predicted by independent pipelines with minimal interaction. Faithful heteroscedastic regression (Stirn et al., AISTATS 2023) shows you can jointly predict mean+variance without degrading mean quality using stop-gradient techniques. The predicted variance naturally encodes "how confident are we about this gene's FC?" which is what p-values measure.
- **Approach**: Add a variance head to the global MLP: `net(x) → (fc_residual, log_var)`. Train with heteroscedastic loss: `(1/2σ²)*(y - ŷ)² + (1/2)*log(σ²)`. Use stop-gradient on the variance path to protect FC prediction quality. Convert predicted variance to p-values via: `z = |fc_pred| / sqrt(var_pred)`, then `p = 2*(1 - Φ(z))`.
- **Expected signal**: Moderate. Unifies FC and significance prediction. The variance head gets implicit information about gene reliability from the same embeddings. Risk is that the per-gene variance is driven by sample size/ref_mean (already in calnet), not just by FC prediction uncertainty.
- **Lit**: Faithful Heteroscedastic Regression (Stirn et al., AISTATS 2023); PRESCRIBE (NeurIPS 2025) for uncertainty estimation in perturbation prediction.

#### MMoEEx: mixture-of-experts for heterogeneous multi-target MLP
- **Idea**: Replace the single global MLP with K=4 expert networks, each with the same architecture. Three task-specific gating networks (g_Y, g_D, g_NLP) produce softmax weights over experts. An exclusivity loss forces different experts to specialize on different targets.
- **Why**: Current MLP predicts FC residuals only. A naive multi-head MLP fails when targets are weakly correlated (r≈0.1 between D and Y). MMoEEx (Aoki et al., IEEE/ACM TCBB 2022) uses exclusivity constraints to prevent expert collapse while allowing shared structure where it exists.
- **Approach**: K=4 experts (same 512→256→PCA dims architecture). Three gates: `g_k(x) = softmax(W_k @ x)`. Exclusivity loss: penalize experts activated equally by all gates. Each target's prediction = Σ_k g_target_k * expert_k(emb). Balance with UW-SO: `weight_k = softmax(1/sg[L_k] / T)` for automatic loss weighting across targets.
- **Expected signal**: Moderate. MLP currently shows diminishing returns. But multi-target experts could unlock cross-target signal the single-target MLP misses. The real test is whether the experts learn meaningful specializations.
- **Risk**: More parameters + complexity. Current MLP contributes only w=0.12 of the FC blend — adding multi-target heads may not move the needle.
- **Lit**: MMoEEx (Aoki et al., 2022); UW-SO (Kirchdorfer et al., IJCV 2025); FAMO (NeurIPS 2023). GitHub: median-research-group/LibMTL (2535★), Cranial-XIX/FAMO (122★).

#### Cross-attention cell-drug interaction
- **Idea**: Replace the MLP's concatenated input `cat(te_emb, ce_emb)` with a cross-attention mechanism where the drug branch queries specific aspects of the cell representation and vice versa, before feeding into prediction heads.
- **Why**: XPert (Nature Machine Intelligence, Jan 2026) demonstrates a dual-branch transformer with cross-attention that already jointly predicts delta and post-perturbation expression, achieving 36.7% higher Pearson in cold-cell generalization. Cross-attention is richer than bilinear interaction because it's sparse and input-dependent — different drugs attend to different cell features.
- **Approach**: 2-layer cross-attention block. Drug queries attend to cell's per-gene ref_mean profile (not just SVD embedding — preserves gene-level info). Cell queries attend to drug's ALS effect profile. Attended representations concatenated and fed to per-target heads.
- **Expected signal**: Small-moderate. The MLP already captures interactions via concatenation + nonlinearity. Cross-attention adds gene-level specificity that could help for the subset of genes most affected by each drug.
- **Risk**: Ref_mean is per-cell (not per-observation), so cross-attention may not add much beyond what ref_emb already provides.
- **Lit**: XPert (Nature Machine Intelligence, Jan 2026); PertDiT diffusion transformer (Quantitative Biology 2026).

### Older ideas (pre-literature-search)

#### Shared ALS with concatenated targets
- **Idea**: Run ALS on a gene-axis-concatenated matrix `[Y | D | NLP]` (shape n_obs × 3*n_genes) to learn cell and treatment effects that simultaneously explain all three quantities. The resulting embeddings encode information from all views.
- **Why**: Currently ALS runs independently on Y, D, and NLP. The cell and treatment effects from FC may miss structure visible in deltas (linear-scale, high-expression-gene dominated) and vice versa. Joint decomposition forces shared latent structure.
- **Approach**: Stack the three targets along the gene axis (with per-target standardization to equalize scales). Run standard ALS. Split the resulting treatment/cell effects back into per-target components. Use the full-width SVD embeddings for drug similarity in regression.
- **Expected signal**: Moderate. The targets are weakly correlated (r ≈ 0.1 between D and Y), so shared structure may be limited. But cell effects should be highly shared.
- **Risk**: Scale mismatch between targets. Need careful normalization.

#### Multi-task neural head
- **Idea**: Replace the separate MLP (for FC residuals) and calnet (for NLP calibration) with a single multi-task network that predicts residuals for all three targets simultaneously. Input: shared cell + treatment embeddings. Output: three heads for D, Y, NLP residuals.
- **Why**: The current MLP and calnet are separate networks with different architectures and training loops. A shared backbone could learn cross-target interactions (e.g., a gene that is significant in NLP space should have a non-zero delta and FC).
- **Approach**: Network input: concat(cell_emb, treat_emb, ref_mean_emb). Shared backbone: 2-3 layers. Three output heads: D_residual, Y_residual, NLP_residual. Multi-task loss: weighted sum of per-target MSE. The weights could be tuned or use uncertainty weighting.
- **Expected signal**: Moderate. Individual MLP and calnet already show diminishing returns. But cross-target signal may unlock new gains.
- **Risk**: Multi-task optimization can be unstable. One task may dominate.

#### Per-observation adaptive blending
- **Source**: Fixed blend_alpha=0.75 is suboptimal for heterogeneous test observations. Some test drugs have many similar training drugs (regression is reliable), others are unusual (pipeline may be better).
- **Why**: The regression quality varies by observation but the blend weight is fixed.
- **Approach**: Compute per-observation regression confidence (e.g., norm of weight vector, max weight, reconstruction error on common drugs) and use it to modulate alpha. Confidence → alpha mapping learned on held-out validation.
- **Risk**: Only 5 test cells → limited validation data for learning the mapping.

