"""
Self-contained metric calculations from cell-eval, for use with precomputed quantities.

Six metrics from the State paper (Adduri et al. 2025), Figure 2:

    Underlying quantity 1: Expression deltas (pseudobulk mean perturbed - mean control)
        - pearson_delta         (Fig 2E: "Pearson correlation")
        - discrimination_score  (Fig 2D: "Perturbation discrimination")

    Underlying quantity 2: DE log fold changes
        - de_spearman_lfc_sig   (Fig 2H: "Fold Change")

    Underlying quantity 3: DE significance (FDR / p-values)
        - pr_auc                (Fig 2F: "P-value")
        - de_overlap            (Fig 2I: "DE Overlap Accuracy")
        - de_spearman_sig       (Fig 2J: "Effect Size Prediction")

Each function takes only the real and predicted arrays/frames needed.
"""

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, pairwise_distances


def _rowwise_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorized row-wise Pearson correlation between (n, p) arrays.

    Returns (n,) array of correlations. Rows with zero variance get r=0.
    """
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    num = (A_c * B_c).sum(axis=1)
    den = np.sqrt((A_c ** 2).sum(axis=1) * (B_c ** 2).sum(axis=1))
    den = np.where(den == 0, 1.0, den)  # avoid division by zero
    r = num / den
    return np.where(np.isfinite(r), r, 0.0)


# ---------------------------------------------------------------------------
# Expression delta metrics
# ---------------------------------------------------------------------------


def pearson_delta(
    real_deltas: np.ndarray,
    pred_deltas: np.ndarray,
) -> np.ndarray:
    """Per-perturbation Pearson correlation between real and predicted expression deltas.

    Fig 2E: "Pearson correlation"

    Args:
        real_deltas: (n_perts, n_genes) real pseudobulk deltas (perturbed - control).
        pred_deltas: (n_perts, n_genes) predicted pseudobulk deltas.

    Returns:
        Pearson r for each perturbation (across genes).
    """
    return _rowwise_pearson(real_deltas, pred_deltas)


def discrimination_score(
    real_deltas: np.ndarray,
    pred_deltas: np.ndarray,
) -> np.ndarray:
    """Perturbation discrimination score using L1 (Manhattan) distance.

    Fig 2D: "Perturbation discrimination"

    For each perturbation, ranks the predicted delta against all real deltas.
    Returns 1 - rank/N per perturbation (cell-eval convention).

    Args:
        real_deltas: (n_perts, n_genes) real pseudobulk deltas.
        pred_deltas: (n_perts, n_genes) predicted pseudobulk deltas.

    Returns:
        Normalized inverse rank for each perturbation (1.0 = perfect).
    """
    n = real_deltas.shape[0]
    scores = np.empty(n)
    for i in range(n):
        distances = pairwise_distances(real_deltas, pred_deltas[i].reshape(1, -1), metric="l1").flatten()
        sorted_indices = np.argsort(distances)
        rank = np.flatnonzero(sorted_indices == i)[0]
        scores[i] = 1.0 - rank / n
    return scores


# ---------------------------------------------------------------------------
# DE metrics
# ---------------------------------------------------------------------------


def de_spearman_lfc_sig(
    de_real: pl.DataFrame,
    de_pred: pl.DataFrame,
    fdr_threshold: float = 0.05,
    target_col: str = "target",
    feature_col: str = "feature",
    fold_change_col: str = "fold_change",
    fdr_col: str = "fdr",
) -> dict[str, float]:
    """Per-perturbation Spearman correlation of log fold changes (significant genes only).

    Fig 2H: "Fold Change"

    Args:
        de_real: Real DE results with columns [target, feature, fold_change, fdr].
        de_pred: Predicted DE results with same columns.
        fdr_threshold: Significance threshold applied to real DE.

    Returns:
        Spearman rho for each perturbation.
    """
    merged = (
        de_real.filter(pl.col(fdr_col) < fdr_threshold)
        .select([target_col, feature_col, fold_change_col])
        .join(
            de_pred.select([target_col, feature_col, fold_change_col]),
            on=[target_col, feature_col],
            suffix="_pred",
            how="left",
        )
        .with_columns(pl.col(f"{fold_change_col}_pred").fill_null(0.0))
    )

    results = {}
    for row in (
        merged.group_by(target_col)
        .agg(
            pl.corr(
                pl.col(fold_change_col).cast(pl.Float64),
                pl.col(f"{fold_change_col}_pred").cast(pl.Float64),
                method="spearman",
            ).alias("spearman")
        )
        .iter_rows()
    ):
        results[row[0]] = row[1]
    return results


def pr_auc(
    de_real: pl.DataFrame,
    de_pred: pl.DataFrame,
    fdr_threshold: float = 0.05,
    target_col: str = "target",
    feature_col: str = "feature",
    fdr_col: str = "fdr",
) -> dict[str, float]:
    """Per-perturbation area under the precision-recall curve for DE gene recovery.

    Fig 2F: "P-value"

    Binary label: gene is significant in real (FDR < threshold).
    Score: -log10(predicted FDR).

    Args:
        de_real: Real DE results with columns [target, feature, fdr].
        de_pred: Predicted DE results with same columns.

    Returns:
        AUPRC for each perturbation.
    """
    real_fdr_col = fdr_col
    pred_fdr_col = fdr_col

    labeled_real = de_real.with_columns((pl.col(real_fdr_col) < fdr_threshold).cast(pl.Float32).alias("label")).select(
        [target_col, feature_col, "label"]
    )

    pred_q = pl.col(pred_fdr_col).fill_null(1.0).clip(1e-10, 1.0)
    merged = (
        labeled_real.join(
            de_pred.select([target_col, feature_col, pred_fdr_col]),
            on=[target_col, feature_col],
            how="left",
            suffix="_pred",
        )
        .drop_nulls(["label"])
        .with_columns(
            pred_q.alias(pred_fdr_col),
            (-pred_q.log10()).alias("nlp"),
        )
    )

    # Extract arrays grouped by target for batch AUPRC computation.
    sorted_df = merged.sort(target_col)
    all_labels = sorted_df["label"].to_numpy().astype(np.float32)
    all_scores = sorted_df["nlp"].to_numpy().astype(np.float64)
    all_targets = sorted_df[target_col].to_numpy()

    # Find group boundaries
    changes = np.concatenate([[0], np.where(all_targets[1:] != all_targets[:-1])[0] + 1, [len(all_targets)]])

    # Reshape into (n_targets, n_genes) if all groups have the same size
    n_targets = len(changes) - 1
    group_size = changes[1] - changes[0]
    uniform = all(changes[i+1] - changes[i] == group_size for i in range(n_targets))

    if uniform:
        # Fast path: reshape + vectorized argsort
        labels_2d = all_labels.reshape(n_targets, group_size)
        scores_2d = all_scores.reshape(n_targets, group_size)
        target_names = all_targets[changes[:-1]]

        # Vectorized AUPRC: sort each row by descending score
        order = np.argsort(-scores_2d, axis=1)
        sorted_labels = np.take_along_axis(labels_2d, order, axis=1)

        # Cumulative precision-recall
        tp_cum = np.cumsum(sorted_labels, axis=1)
        n_pos = tp_cum[:, -1:]  # total positives per target
        precision = tp_cum / np.arange(1, group_size + 1)
        recall_diff = sorted_labels / np.maximum(n_pos, 1)
        ap = (precision * recall_diff).sum(axis=1)

        # Mark invalid (no positives or all positives)
        valid = (n_pos.ravel() > 0) & (n_pos.ravel() < group_size)
        results = {}
        for i in range(n_targets):
            results[target_names[i]] = float(ap[i]) if valid[i] else float("nan")
    else:
        # Fallback: per-target loop (rare — only when gene counts differ across targets)
        results = {}
        for i in range(n_targets):
            start, end = changes[i], changes[i + 1]
            labels = all_labels[start:end]
            scores = all_scores[start:end]
            pos = labels.sum()
            if not (0 < pos < len(labels)):
                results[all_targets[start]] = float("nan")
                continue
            results[all_targets[start]] = float(average_precision_score(labels, scores))
    return results


def de_overlap(
    de_real: pl.DataFrame,
    de_pred: pl.DataFrame,
    k: int | None = None,
    fdr_threshold: float = 0.05,
    target_col: str = "target",
    feature_col: str = "feature",
    fdr_col: str = "fdr",
    sort_col: str = "abs_log2_fold_change",
) -> dict[str, float]:
    """Per-perturbation overlap in top DE genes.

    Fig 2I: "DE Overlap Accuracy"

    For each perturbation, takes top-k significant genes (sorted by |LFC|) from real
    and predicted, and computes the fraction of real top-k found in predicted top-k.

    Args:
        de_real: Real DE results.
        de_pred: Predicted DE results.
        k: Number of top genes. None = all significant genes.
        fdr_threshold: Significance threshold.
        sort_col: Column to sort genes by (descending).

    Returns:
        Overlap fraction for each perturbation.
    """

    def _top_genes(de: pl.DataFrame) -> dict[str, np.ndarray]:
        sig = de.filter(pl.col(fdr_col) < fdr_threshold)
        result = {}
        for pert in sig[target_col].unique().sort().to_list():
            genes = (
                sig.filter(pl.col(target_col) == pert)
                .sort(sort_col, descending=True)
                .select(feature_col)
                .to_series()
                .to_numpy()
            )
            result[pert] = genes
        return result

    real_top = _top_genes(de_real)
    pred_top = _top_genes(de_pred)

    all_perts = sorted(set(real_top) | set(pred_top))
    overlaps = {}
    for pert in all_perts:
        real_genes = real_top.get(pert, np.array([]))
        pred_genes = pred_top.get(pert, np.array([]))

        k_eff = real_genes.size if k is None else min(k, real_genes.size)
        if k_eff == 0:
            overlaps[pert] = 0.0
            continue

        real_subset = real_genes[:k_eff]
        pred_subset = pred_genes[:k_eff]
        overlaps[pert] = np.intersect1d(real_subset, pred_subset).size / k_eff
    return overlaps


def de_spearman_sig(
    de_real: pl.DataFrame,
    de_pred: pl.DataFrame,
    fdr_threshold: float = 0.05,
    target_col: str = "target",
    fdr_col: str = "fdr",
) -> float:
    """Spearman correlation of perturbation effect sizes (number of significant DE genes).

    Fig 2J: "Effect Size Prediction"

    Counts significant genes per perturbation in real and predicted, then computes
    the Spearman correlation across perturbations.

    Args:
        de_real: Real DE results.
        de_pred: Predicted DE results.

    Returns:
        Single Spearman rho across perturbations.
    """
    counts_real = de_real.filter(pl.col(fdr_col) < fdr_threshold).group_by(target_col).len()
    counts_pred = de_pred.filter(pl.col(fdr_col) < fdr_threshold).group_by(target_col).len()

    merged = counts_real.join(
        counts_pred,
        on=target_col,
        suffix="_pred",
        how="left",
    ).fill_null(0)

    if merged.shape[0] == 0:
        return 1.0  # No significant genes in either — perfect agreement by convention

    return float(
        merged.select(
            pl.corr(
                pl.col("len"),
                pl.col("len_pred"),
                method="spearman",
            )
        )
        .to_numpy()
        .flatten()[0]
    )
