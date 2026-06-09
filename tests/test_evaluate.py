"""Tests for evaluation orchestration in prepare_combined.py."""

import numpy as np
import pytest

from rhaister.prepare_combined import evaluate, make_evaluator

# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_all_metric_keys_present(self):
        """evaluate() with all inputs should return all expected metric keys."""
        rng = np.random.default_rng(42)
        n, g = 20, 50
        Y = rng.normal(size=(n, g))
        P = rng.uniform(0, 1, size=(n, g))
        D = rng.normal(size=(n, g))
        F = rng.uniform(0, 1, size=(n, g))
        cells = np.array([f"C{i}" for i in range(n)])
        treats = np.array([f"T{i}" for i in range(n)])
        genes = [f"gene_{j}" for j in range(g)]

        result = evaluate(
            Y_true=Y,
            Y_pred=Y,
            P_true=P,
            P_pred=P,
            D_true=D,
            D_pred=D,
            F_true=F,
            F_pred=F,
            test_cells=cells,
            test_treatments=treats,
            gene_cols=genes,
        )

        assert "pdex_static/pearson_delta_mean" in result
        assert "n_samples" in result
        assert result["n_samples"] == n
        assert "state/pearson_delta_mean" in result
        assert "state/pr_auc_mean" in result
        assert "state/de_overlap_mean" in result
        assert "state/de_spearman_sig" in result
        assert "state/spearman_lfc_sig_mean" in result

    def test_perfect_fc_pearson(self):
        """When Y_pred == Y_true, legacy FC pearson should be ~1.0."""
        rng = np.random.default_rng(42)
        n, g = 15, 40
        Y = rng.normal(size=(n, g))
        result = evaluate(Y_true=Y, Y_pred=Y)
        assert result["pdex_static/pearson_delta_mean"] > 0.999

    def test_fc_only(self):
        """evaluate() with only FC arrays (no p-values, no deltas) should still work."""
        rng = np.random.default_rng(42)
        Y = rng.normal(size=(10, 30))
        result = evaluate(Y_true=Y, Y_pred=Y)
        assert "pdex_static/pearson_delta_mean" in result
        assert "state/pr_auc_mean" not in result  # no p-values provided

    def test_pvalue_auprc(self):
        """When P_pred == P_true, AUPRC should be high."""
        rng = np.random.default_rng(42)
        n, g = 20, 100
        Y = rng.normal(size=(n, g))
        P = rng.uniform(0, 1, size=(n, g))
        result = evaluate(Y_true=Y, Y_pred=Y, P_true=P, P_pred=P)
        assert "pdex_static/auprc_p05" in result
        assert result["pdex_static/auprc_p05"] > 0.9


# ---------------------------------------------------------------------------
# make_evaluator() — one-shot semantics
# ---------------------------------------------------------------------------


class TestMakeEvaluator:
    def test_first_call_succeeds(self):
        rng = np.random.default_rng(42)
        n, g = 10, 20
        Y_test = rng.normal(size=(n, g))
        P_test = rng.uniform(0, 1, size=(n, g))
        D_test = rng.normal(size=(n, g))
        F_test = rng.uniform(0, 1, size=(n, g))
        cells = np.array([f"C{i}" for i in range(n)])
        treats = np.array([f"T{i}" for i in range(n)])
        genes = [f"gene_{j}" for j in range(g)]

        eval_fn = make_evaluator(Y_test, P_test, D_test, F_test, cells, treats, genes)
        Y_pred = rng.normal(size=(n, g))
        result = eval_fn(Y_pred)
        assert isinstance(result, dict)
        assert "pdex_static/pearson_delta_mean" in result

    def test_second_call_raises(self):
        rng = np.random.default_rng(42)
        n, g = 10, 20
        Y_test = rng.normal(size=(n, g))
        P_test = rng.uniform(0, 1, size=(n, g))
        D_test = rng.normal(size=(n, g))
        F_test = rng.uniform(0, 1, size=(n, g))
        cells = np.array([f"C{i}" for i in range(n)])
        treats = np.array([f"T{i}" for i in range(n)])
        genes = [f"gene_{j}" for j in range(g)]

        eval_fn = make_evaluator(Y_test, P_test, D_test, F_test, cells, treats, genes)
        Y_pred = rng.normal(size=(n, g))
        eval_fn(Y_pred)  # first call
        with pytest.raises(RuntimeError, match="already called"):
            eval_fn(Y_pred)  # second call should fail
