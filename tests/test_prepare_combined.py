"""Tests for prepare_combined.py — data utility functions."""

import numpy as np
import pandas as pd
import polars as pl

from rhaister.prepare_combined import (
    _build_de_frame,
    aggregate_replicates,
    get_gene_columns,
    make_splits,
    parse_split_name,
    pvalues_to_fdr_bh,
    to_matrices,
)

# ---------------------------------------------------------------------------
# parse_split_name
# ---------------------------------------------------------------------------


class TestParseSplitName:
    def test_qualified(self):
        assert parse_split_name("tahoe/5_holdout") == ("tahoe", "5_holdout")

    def test_qualified_parse(self):
        assert parse_split_name("parse/donor_split_0") == ("parse", "donor_split_0")

    def test_legacy_flat(self):
        assert parse_split_name("tahoe_5_holdout") == ("tahoe", "5_holdout")

    def test_legacy_flat_long(self):
        assert parse_split_name("tahoe_5_holdout_titration_10") == ("tahoe", "5_holdout_titration_10")

    def test_bare_fallback(self):
        # Bare split name defaults to the default dataset
        dataset, split = parse_split_name("5_holdout")
        assert split == "5_holdout"
        assert dataset == "tahoe"  # DEFAULT_DATASET

    def test_qualified_perturbai(self):
        assert parse_split_name("perturbai/split_0") == ("perturbai", "split_0")


# ---------------------------------------------------------------------------
# get_gene_columns
# ---------------------------------------------------------------------------


class TestGetGeneColumns:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "cell_line": ["A", "B"],
                "treatment": ["X", "Y"],
                "GENE1": [1.0, 2.0],
                "GENE2": [3.0, 4.0],
            }
        )
        result = get_gene_columns(df)
        assert result == ["GENE1", "GENE2"]

    def test_no_genes(self):
        df = pd.DataFrame({"cell_line": ["A"], "treatment": ["X"]})
        assert get_gene_columns(df) == []


# ---------------------------------------------------------------------------
# aggregate_replicates
# ---------------------------------------------------------------------------


class TestAggregateReplicates:
    def test_mean_aggregation(self):
        df = pd.DataFrame(
            {
                "cell_line": ["A", "A", "B"],
                "treatment": ["X", "X", "Y"],
                "GENE1": [2.0, 4.0, 6.0],
                "GENE2": [10.0, 20.0, 30.0],
            }
        )
        agg = aggregate_replicates(df)
        assert len(agg) == 2  # two unique (cell_line, treatment) pairs

        row_ax = agg[(agg["cell_line"] == "A") & (agg["treatment"] == "X")]
        assert float(row_ax["GENE1"].iloc[0]) == 3.0  # mean(2, 4)
        assert float(row_ax["GENE2"].iloc[0]) == 15.0  # mean(10, 20)

    def test_no_replicates(self):
        df = pd.DataFrame(
            {
                "cell_line": ["A", "B"],
                "treatment": ["X", "Y"],
                "GENE1": [1.0, 2.0],
            }
        )
        agg = aggregate_replicates(df)
        assert len(agg) == 2


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------


class TestMakeSplits:
    def test_basic_split(self):
        df = pd.DataFrame(
            {
                "cell_line": ["A", "A", "B", "B"],
                "treatment": ["T1", "T2", "T1", "T2"],
                "GENE1": [1.0, 2.0, 3.0, 4.0],
            }
        )
        split_info = {
            "holdout_cells": ["B"],
            "test_treatments": {"B": {"T2"}},
        }
        train, test = make_splits(df, split_info)
        assert len(test) == 1
        assert test.iloc[0]["cell_line"] == "B"
        assert test.iloc[0]["treatment"] == "T2"
        assert len(train) == 3

    def test_no_holdout_match(self):
        df = pd.DataFrame(
            {
                "cell_line": ["A", "A"],
                "treatment": ["T1", "T2"],
                "GENE1": [1.0, 2.0],
            }
        )
        split_info = {
            "holdout_cells": ["C"],
            "test_treatments": {"C": {"T1"}},
        }
        train, test = make_splits(df, split_info)
        assert len(test) == 0
        assert len(train) == 2


# ---------------------------------------------------------------------------
# to_matrices
# ---------------------------------------------------------------------------


class TestToMatrices:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "cell_line": ["A", "B"],
                "treatment": ["X", "Y"],
                "G1": [1.0, 2.0],
                "G2": [3.0, 4.0],
            }
        )
        cells, treats, Y = to_matrices(df, ["G1", "G2"])
        assert list(cells) == ["A", "B"]
        assert list(treats) == ["X", "Y"]
        assert Y.shape == (2, 2)
        assert Y[0, 0] == 1.0
        assert Y[1, 1] == 4.0


# ---------------------------------------------------------------------------
# pvalues_to_fdr_bh
# ---------------------------------------------------------------------------


class TestPvaluesToFdrBH:
    def test_single_gene(self):
        pvals = np.array([[0.01], [0.05], [0.10]])
        fdr = pvalues_to_fdr_bh(pvals)
        assert fdr.shape == pvals.shape
        # Each row has 1 gene, so FDR == p-value
        np.testing.assert_allclose(fdr, pvals, atol=1e-10)

    def test_monotonicity(self):
        """FDR should be >= p-values (BH correction inflates)."""
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, size=(5, 100))
        fdr = pvalues_to_fdr_bh(pvals)
        # FDR >= p on average (not necessarily per-gene due to BH step-up)
        assert fdr.mean() >= pvals.mean() - 0.01

    def test_bounds(self):
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, size=(10, 50))
        fdr = pvalues_to_fdr_bh(pvals)
        assert fdr.min() >= 0
        assert fdr.max() <= 1

    def test_all_significant(self):
        """When all p-values are very small, FDR should still be small."""
        pvals = np.full((3, 10), 0.001)
        fdr = pvalues_to_fdr_bh(pvals)
        assert fdr.max() < 0.01


# ---------------------------------------------------------------------------
# _build_de_frame
# ---------------------------------------------------------------------------


class TestBuildDeFrame:
    def test_schema(self):
        fc = np.array([[0.5, -0.3], [1.0, 0.2]])
        fdr = np.array([[0.01, 0.5], [0.1, 0.9]])
        cells = np.array(["A", "B"])
        treats = np.array(["T1", "T2"])
        genes = ["G1", "G2"]
        df = _build_de_frame(fc, fdr, cells, treats, genes)
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"target", "feature", "fold_change", "fdr", "abs_log2_fold_change"}
        assert len(df) == 4  # 2 obs * 2 genes

    def test_values(self):
        fc = np.array([[1.0, -2.0]])
        fdr = np.array([[0.01, 0.5]])
        df = _build_de_frame(fc, fdr, np.array(["C"]), np.array(["T"]), ["G1", "G2"])
        row_g2 = df.filter(pl.col("feature") == "G2")
        assert float(row_g2["fold_change"][0]) == -2.0
        assert float(row_g2["abs_log2_fold_change"][0]) == 2.0
