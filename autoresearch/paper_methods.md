# Methods

## Setup and notation

Let $c \in \{1, \ldots, C\}$ index cell contexts (cell lines, donors, primary cultures) and $p \in \{1, \ldots, P\}$ index perturbations (small molecules, CRISPR guides, cytokines). For each observed pair $(c, p) \in \mathcal{O} \subset \{1,\ldots,C\} \times \{1,\ldots,P\}$ we measure a response $\mathbf{y}_{c,p} \in \mathbb{R}^G$ — a per-gene differential-expression vector ($G > 1$) or a scalar sensitivity readout ($G = 1$). The task is compositional generalization: predict $\mathbf{y}_{c^\star, p^\star}$ for held-out pairs $(c^\star, p^\star) \notin \mathcal{O}$ where each of $c^\star$ and $p^\star$ appears individually in $\mathcal{O}$ but the combination is unseen.

We let $\mathcal{P}_c = \{p : (c,p) \in \mathcal{O}\}$ and $\mathcal{C}_p = \{c : (c,p) \in \mathcal{O}\}$. Held-out cells are $\mathcal{C}^\star$; non-held-out cells are $\mathcal{C}_{\text{tr}} = \{1,\ldots,C\} \setminus \mathcal{C}^\star$.

## 1. Additive baseline (ALS)

We start with a rank-1 additive decomposition

$$y_{c,p,g} \approx \mu_g + \alpha_{c,g} + \beta_{p,g}, \tag{1}$$

fit by weighted alternating least squares over observed entries. With optional confidence weights $w_{c,p}$ (e.g., $w_{c,p} = 1 + \kappa \cdot (-\log_{10} p^{\text{obs}}_{c,p})$ in the FC view, $\kappa = 0.02$; uniform otherwise), the closed-form updates are

$$\alpha_{c,g} \leftarrow \frac{\sum_{p \in \mathcal{P}_c} w_{c,p}\,(y_{c,p,g} - \mu_g - \beta_{p,g})}{\sum_{p \in \mathcal{P}_c} w_{c,p}}, \qquad \beta_{p,g} \leftarrow \frac{\sum_{c \in \mathcal{C}_p} w_{c,p}\,(y_{c,p,g} - \mu_g - \alpha_{c,g})}{\sum_{c \in \mathcal{C}_p} w_{c,p}}. \tag{2}$$

Iterations alternate (5 sweeps in practice). The additive prediction at a held-out pair is

$$\hat y^{\text{add}}_{c^\star, p^\star, g} = \mu_g + \alpha_{c^\star, g} + \beta_{p^\star, g}. \tag{3}$$

## 2. Per-cell perturbation regression

For each held-out cell $c^\star$ we form a per-cell ridge that predicts the response to a held-out perturbation $p^\star$ as a linear combination of $c^\star$'s observed-perturbation responses. Let $D_x = \mathcal{P}_{c^\star}$ (perturbations observed for the held-out cell) and consider non-held-out cells $\mathcal{C}_{\text{tr}}$ that share $D_x$. Stacking the training-cell observations into $X \in \mathbb{R}^{|\mathcal{C}_{\text{tr}}| \times |D_x| \times G}$ and the targets at the held-out perturbation $p^\star$ into $Y \in \mathbb{R}^{|\mathcal{C}_{\text{tr}}| \times G}$, we solve an unweighted ridge

$$\widehat W = \arg\min_{W \in \mathbb{R}^{|D_x| \times G}} \sum_{c \in \mathcal{C}_{\text{tr}}} \big\| Y_c - W^\top X_c \big\|_2^2 + \lambda \|W\|_F^2 \tag{4}$$

subject to an optional rank constraint $\text{rank}(W) \le r$. The rank constraint is implemented by retaining the top-$r$ components of the unconstrained $\widehat W$; in the FC view we set $r = 75$. The prediction is

$$\hat y^{\text{ridge}}_{c^\star, p^\star, g} = \sum_{q \in D_x} \widehat W_{q, g}\, y_{c^\star, q, g} \quad (\text{or}\ + \text{centered offsets when applicable}). \tag{5}$$

For each held-out cell, $D_y \subseteq \mathcal{P}_{c^\star} \cap \bigcap_{c \in \mathcal{C}_{\text{tr}}} \mathcal{P}_c$ — held-out perturbations that intersect with the common training-perturbation set; for pairs where coverage fails we fall back to the additive baseline (3). For sparse datasets (e.g., PRISM) we densify $X$ by ALS-imputation of missing entries to preserve coverage. Where many training cells share the same $D_x$, we use the Woodbury identity to solve in $|\mathcal{C}_{\text{tr}}|$-by-$|\mathcal{C}_{\text{tr}}|$ rather than $|D_x|$-by-$|D_x|$ space.

The per-view prediction is the ridge output where coverage holds, the additive baseline (3) otherwise:

$$\hat y_{c^\star, p^\star, g} = \begin{cases} \hat y^{\text{ridge}}_{c^\star, p^\star, g} & (c^\star, p^\star) \in \text{coverage} \\ \hat y^{\text{add}}_{c^\star, p^\star, g} & \text{otherwise.} \end{cases} \tag{6}$$

In the Δ view we use a soft fallback, $\hat y^{\Delta} = 0.95\,\hat y^{\text{ridge}} + 0.05\,\hat y^{\text{add}}$ where covered, additive only otherwise.

**Note on residual vs raw-$Y$ regression.** A natural alternative is to regress the *interaction residual* $R = Y - \hat y^{\text{add}}$ instead of raw $Y$, then add the additive baseline back at test time. The motivation is that ridge shrinkage would then pull the prediction toward the additive baseline rather than toward zero. In practice we found this had no advantage and was slightly worse on the datasets we tested (Tahoe, RIFIVDU, PRISM): column-mean centering in (4)–(5) already removes a column-wise estimate of $\beta_q$ from each training column, so the raw-$Y$ regression inherits enough of the additive structure from the training cells' responses without needing explicit residualization.

**Note on cell-similarity weighting.** An earlier version of the model replaced the unweighted sum in (4) with a per-row weight $w_{c^\star, c}$ derived from rank positions of a cosine similarity matrix on per-cell interaction residuals (Borda-style reciprocal-position weights, parameter-free). Ablations across Tahoe, Parse, RIFIVDU, PRISM, and Replogle/Nadig showed the weighting either has no effect (mean |Δ| ≤ 0.01 on every Tahoe metric) or slightly hurts (Replogle/Nadig: mean Δ ≈ +0.05–0.06 on r²-like metrics when weighting is *removed*). The unweighted regression is therefore the default; column-mean centering and ALS imputation carry the cell-context structure on their own.

## 3. Multi-view extension (differential expression)

For the DE setting we predict three coupled quantities — log fold change $\mathbf{y}^{\text{FC}}$, $-\log_{10}$ p-value $\mathbf{y}^{\text{NLP}}$, and expression delta $\mathbf{y}^{\Delta}$ — by running the pipeline (1)–(6) in three view-specific instances. Cross-view information sharing happens through one channel:

**Calibration network for significance.** Per-gene NLP is refined by a small MLP $h_\phi: \mathbb{R}^{d_{\text{cal}}} \to \mathbb{R}$ that ingests per-(cell, perturbation, gene) features:

$$\mathbf{x}^{\text{cal}}_{c,p,g} = \big(\,|\hat y^{\text{FC}}_{c,p,g}|,\ \hat y^{\text{NLP}}_{c,p,g},\ \log(1+R_{c,g}),\ \mu^{\text{NLP}}_g,\ \sigma^{\text{FC}}_g,\ \overline{\log(1+R_c)},\ |\hat y^{\Delta,\text{add}}_{c,p,g}|\,\big), \tag{7}$$

where $R_{c,g}$ is baseline (untreated) expression. The network is a 2-layer GELU MLP (128 → 64 → 1), trained on observed NLP with squared error, 200 epochs of cosine-annealed Adam. The calibrated p-value blends with the regression-based blend in significance space:

$$P^{\text{final}}_{c,p,g} = 1 - \big[(1-w_{\text{cal}})(1 - P^{\text{blend}}_{c,p,g}) + w_{\text{cal}}(1 - 10^{-h_\phi(\mathbf{x}^{\text{cal}}_{c,p,g})})\big], \tag{8}$$

with $w_{\text{cal}} = 0.5$. Final FDR is obtained per observation by Benjamini–Hochberg adjustment of $P^{\text{final}}_{c,p,\cdot}$.

## 4. Implementation

Hyperparameters used in our reported FC + NLP + Δ instantiation:

- ALS sweeps $T_{\text{ALS}} = 5$
- FC p-value weighting $\kappa = 0.02$
- Ridge $\lambda = 1.0$ (unweighted across training cells)
- Rank constraint $r_{\text{FC}} = 75$; no rank truncation for NLP or Δ
- Δ ridge–additive blend $0.95 / 0.05$
- Calibration MLP: hidden 128, 200 epochs, single network (no ensemble); calnet blend $w_{\text{cal}} = 0.5$

Earlier versions of the model also blended in (i) a per-row cell-similarity weighting in the ridge (see the "Note on cell-similarity weighting" in §2), (ii) a residual MLP head with a learnable low-rank gene-factor decoder, and (iii) a complementary cell-as-linear-combination regression that predicts the held-out response as a linear combination of training cells' responses. Ablations showed each contributed only marginal accuracy gains (Δ < 0.01 absolute on the State pr_auc / de_overlap / de_spearman_sig metrics on Tahoe) while their optimal blending weights varied substantially across datasets and did not transfer; in the Replogle/Nadig CRISPR setting cell-similarity weighting actively *hurt*. The default configuration retains only the components that contributed robustly across Tahoe, RIFIVDU, PRISM, Parse, and Replogle/Nadig.
