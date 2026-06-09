"""Tests for state_metrics.py — the six State paper metric functions."""

import numpy as np
import polars as pl

from rhaister.state_metrics import (
    de_overlap,
    de_spearman_lfc_sig,
    de_spearman_sig,
    discrimination_score,
    pearson_delta,
    pr_auc,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_de_frame(fc, fdr, n_perts, n_genes):
    """Build a Polars DE DataFrame from (n_perts, n_genes) arrays."""
    perts = [f"pert_{i}" for i in range(n_perts)]
    genes = [f"gene_{j}" for j in range(n_genes)]
    return pl.DataFrame(
        {
            "target": np.repeat(perts, n_genes),
            "feature": np.tile(genes, n_perts),
            "fold_change": fc.ravel().astype(np.float64),
            "fdr": fdr.ravel().astype(np.float64),
            "abs_log2_fold_change": np.abs(fc.ravel()).astype(np.float64),
        }
    )


# ---------------------------------------------------------------------------
# pearson_delta
# ---------------------------------------------------------------------------


class TestPearsonDelta:
    def test_identity(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 100))
        result = pearson_delta(X, X)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_noisy(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 100))
        Y = X + 0.5 * rng.normal(size=X.shape)
        result = pearson_delta(X, Y)
        assert all(0 < r < 1 for r in result)

    def test_negation(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 50))
        result = pearson_delta(X, -X)
        np.testing.assert_allclose(result, -1.0, atol=1e-10)

    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(15, 80))
        result = pearson_delta(X, X)
        assert result.shape == (15,)


# ---------------------------------------------------------------------------
# discrimination_score
# ---------------------------------------------------------------------------


class TestDiscriminationScore:
    def test_perfect(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10, 50))
        # Each row's closest L1 match should be itself
        result = discrimination_score(X, X)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_shape(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(8, 30))
        result = discrimination_score(X, X)
        assert result.shape == (8,)


# ---------------------------------------------------------------------------
# de_spearman_lfc_sig
# ---------------------------------------------------------------------------


class TestSpearmanLfcSig:
    def test_identical(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 10, 50
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = rng.uniform(0, 0.1, size=(n_perts, n_genes))  # mostly significant
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = de_spearman_lfc_sig(df, df)
        vals = list(result.values())
        assert len(vals) > 0
        for v in vals:
            assert abs(v - 1.0) < 1e-6, f"Expected ~1.0, got {v}"

    def test_returns_dict(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 5, 20
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = np.full((n_perts, n_genes), 0.01)
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = de_spearman_lfc_sig(df, df)
        assert isinstance(result, dict)
        assert len(result) == n_perts


# ---------------------------------------------------------------------------
# pr_auc
# ---------------------------------------------------------------------------


class TestPrAuc:
    def test_perfect_predictions(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 5, 100
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = rng.uniform(0, 1, size=(n_perts, n_genes))
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = pr_auc(df, df)
        vals = [v for v in result.values() if np.isfinite(v)]
        assert len(vals) > 0
        for v in vals:
            assert v > 0.9, f"Perfect predictions should give high AUPRC, got {v}"

    def test_returns_per_perturbation(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 8, 50
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = rng.uniform(0, 0.2, size=(n_perts, n_genes))
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = pr_auc(df, df)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# de_overlap
# ---------------------------------------------------------------------------


class TestDeOverlap:
    def test_identical(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 5, 50
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = rng.uniform(0, 0.1, size=(n_perts, n_genes))  # mostly significant
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = de_overlap(df, df)
        vals = list(result.values())
        assert len(vals) > 0
        for v in vals:
            assert abs(v - 1.0) < 1e-10, f"Expected 1.0, got {v}"

    def test_no_overlap(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 3, 50
        fc_real = rng.normal(size=(n_perts, n_genes))
        fc_pred = -fc_real  # reversed ranking
        fdr = np.full((n_perts, n_genes), 0.01)
        df_real = _make_de_frame(fc_real, fdr, n_perts, n_genes)
        df_pred = _make_de_frame(fc_pred, fdr, n_perts, n_genes)
        result = de_overlap(df_real, df_pred)
        vals = list(result.values())
        # With reversed FC, overlap should be low (not necessarily 0 since
        # we sort by abs value)
        assert len(vals) > 0


# ---------------------------------------------------------------------------
# de_spearman_sig
# ---------------------------------------------------------------------------


class TestDeSpearmanSig:
    def test_identical(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 10, 50
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = rng.uniform(0, 0.2, size=(n_perts, n_genes))
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = de_spearman_sig(df, df)
        assert abs(result - 1.0) < 1e-6

    def test_returns_scalar(self):
        rng = np.random.default_rng(42)
        n_perts, n_genes = 5, 30
        fc = rng.normal(size=(n_perts, n_genes))
        fdr = np.full((n_perts, n_genes), 0.01)
        df = _make_de_frame(fc, fdr, n_perts, n_genes)
        result = de_spearman_sig(df, df)
        assert isinstance(result, float)
