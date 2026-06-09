"""Tests for scripts/data_prep/ — pseudobulk delta and pdex computation.

Unit tests use synthetic AnnData objects. The integration test
(test_celleval_matches_reference) compares against the existing plate1
parquet using a single cell line subset from the real h5ad.
"""

import os
import sys
import tempfile

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "data_prep"))

from compute_celleval_deltas import (
    compute_deltas,
    extract_expression,
    pseudobulk_means,
    sparse_normalize_total_log1p,
)


# ---------------------------------------------------------------------------
# sparse_normalize_total_log1p
# ---------------------------------------------------------------------------


class TestSparseNormalize:
    def test_basic(self):
        """Rows should sum to target_sum before log1p."""
        X = sp.csr_matrix(np.array([[1, 2, 3], [4, 0, 6]], dtype=np.float32))
        result = sparse_normalize_total_log1p(X, target_sum=100)
        # Before log1p, row 0: [1/6*100, 2/6*100, 3/6*100] = [16.67, 33.33, 50]
        # After log1p: [log1p(16.67), log1p(33.33), log1p(50)]
        dense = result.toarray()
        expected_pre_log = np.array([[100 / 6, 200 / 6, 300 / 6], [400 / 10, 0, 600 / 10]])
        np.testing.assert_allclose(dense, np.log1p(expected_pre_log).astype(np.float32), rtol=1e-5)

    def test_zero_row(self):
        """A row with all zeros should stay all zeros."""
        X = sp.csr_matrix(np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32))
        result = sparse_normalize_total_log1p(X, target_sum=100)
        assert result.toarray()[0].sum() == 0

    def test_preserves_sparsity(self):
        X = sp.csr_matrix(np.array([[0, 0, 5], [3, 0, 0]], dtype=np.float32))
        result = sparse_normalize_total_log1p(X, target_sum=100)
        assert sp.issparse(result)
        assert result.nnz == 2


# ---------------------------------------------------------------------------
# pseudobulk_means
# ---------------------------------------------------------------------------


class TestPseudobulkMeans:
    def test_simple(self):
        """Two groups, verify means are correct."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        labels = np.array(["A", "A", "B"])
        means, g2i = pseudobulk_means(X, labels)
        assert len(g2i) == 2
        np.testing.assert_allclose(means[g2i["A"]], [2.0, 3.0], rtol=1e-5)
        np.testing.assert_allclose(means[g2i["B"]], [5.0, 6.0], rtol=1e-5)

    def test_sparse_input(self):
        X = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.float32))
        labels = np.array(["G1", "G1"])
        means, g2i = pseudobulk_means(X, labels)
        np.testing.assert_allclose(means[g2i["G1"]], [0.5, 2.0], rtol=1e-5)

    def test_single_cell_groups(self):
        X = np.eye(3, dtype=np.float32)
        labels = np.array(["A", "B", "C"])
        means, g2i = pseudobulk_means(X, labels)
        assert means.shape == (3, 3)
        np.testing.assert_allclose(means[g2i["A"]], [1, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# compute_deltas
# ---------------------------------------------------------------------------


class TestComputeDeltas:
    def test_basic(self):
        """Control subtraction works correctly."""
        means = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        g2i = {"cellA||DMSO": 0, "cellA||drug1": 1, "cellB||DMSO": 2}
        genes = ["G1", "G2"]
        df = compute_deltas(means, g2i, genes, "DMSO", "cell_line", "treatment")
        assert len(df) == 1  # only cellA||drug1 (cellB has no treatment)
        assert float(df["G1"].iloc[0]) == 2.0  # 3 - 1
        assert float(df["G2"].iloc[0]) == 2.0  # 4 - 2

    def test_no_control(self):
        """Groups without control are skipped."""
        means = np.array([[10.0, 20.0]], dtype=np.float32)
        g2i = {"cellA||drug1": 0}
        df = compute_deltas(means, g2i, ["G1", "G2"], "DMSO", "cell_line", "treatment")
        assert len(df) == 0

    def test_file_id_col(self):
        """file_id_col adds an extra column."""
        means = np.array([[1.0], [3.0]], dtype=np.float32)
        g2i = {"X||ctrl": 0, "X||treat": 1}
        df = compute_deltas(
            means, g2i, ["G1"], "ctrl", "group", "treatment",
            file_id="plate1", file_id_col="plate",
        )
        assert "plate" in df.columns
        assert df["plate"].iloc[0] == "plate1"


# ---------------------------------------------------------------------------
# extract_expression
# ---------------------------------------------------------------------------


class TestExtractExpression:
    def test_all_genes_mode(self):
        """all_genes mode normalizes and returns all genes."""
        n_cells, n_genes = 100, 50
        X = sp.random(n_cells, n_genes, density=0.3, format="csr", dtype=np.float32)
        X.data = np.abs(X.data) * 100  # positive counts
        adata = anndata.AnnData(X=X)
        adata.var.index = [f"gene_{i}" for i in range(n_genes)]
        adata.var = adata.var.reset_index()

        cfg = {"gene_mode": "all_genes", "target_sum": 1000}
        result_X, gene_names = extract_expression(adata, cfg)
        assert result_X.shape == (n_cells, n_genes)
        assert len(gene_names) == n_genes
        # After log1p, all values should be non-negative
        if sp.issparse(result_X):
            assert result_X.data.min() >= 0
        else:
            assert result_X.min() >= 0


# ---------------------------------------------------------------------------
# End-to-end with synthetic data
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_synthetic_full_pipeline(self):
        """Run the full delta pipeline on a synthetic h5ad."""
        n_cells, n_genes = 200, 30
        rng = np.random.default_rng(42)

        # Create synthetic data: 2 cell lines, 3 treatments (including control)
        cell_lines = np.array(["CL1"] * 100 + ["CL2"] * 100)
        treatments = np.array(
            ["DMSO"] * 40 + ["drugA"] * 30 + ["drugB"] * 30
            + ["DMSO"] * 50 + ["drugA"] * 50
        )
        X = sp.random(n_cells, n_genes, density=0.5, format="csr", dtype=np.float32, random_state=42)
        X.data = np.abs(X.data) * 100

        adata = anndata.AnnData(X=X)
        adata.obs["cell_line"] = cell_lines
        adata.obs["treatment"] = treatments
        adata.var.index = pd.Index([f"gene_{i}" for i in range(n_genes)], name="gene_name")

        with tempfile.TemporaryDirectory() as tmp:
            # Save h5ad
            h5ad_path = os.path.join(tmp, "test_plate.h5ad")
            adata.write_h5ad(h5ad_path)

            # Write config
            cfg = {
                "dataset": "test",
                "h5ad_dir": tmp,
                "h5ad_pattern": "*.h5ad",
                "treatment_col": "treatment",
                "control": "DMSO",
                "group_col": "cell_line",
                "celleval": {
                    "output_dir": os.path.join(tmp, "output"),
                    "gene_mode": "all_genes",
                    "gene_name_col": "gene_name",
                    "target_sum": 1000,
                },
            }
            cfg_path = os.path.join(tmp, "test.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f)

            # Run via subprocess
            import subprocess

            result = subprocess.run(
                [sys.executable, "scripts/data_prep/compute_celleval_deltas.py", cfg_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"Script failed:\n{result.stderr}"

            # Check output
            out_files = os.listdir(os.path.join(tmp, "output"))
            assert any("test_plate" in f for f in out_files)

            df = pd.read_parquet(os.path.join(tmp, "output", "test_plate.parquet"))
            assert "cell_line" in df.columns
            assert "treatment" in df.columns
            # CL1 has drugA and drugB, CL2 has drugA -> 3 rows
            assert len(df) == 3
            assert set(df["treatment"]) == {"drugA", "drugB"}
            assert set(df["cell_line"]) == {"CL1", "CL2"}

            # Gene columns should be present
            gene_cols = [c for c in df.columns if c.startswith("gene_")]
            assert len(gene_cols) == n_genes


# ---------------------------------------------------------------------------
# Integration test: compare against existing parquet
# ---------------------------------------------------------------------------


FIXTURE_H5AD = os.path.join(os.path.dirname(__file__), "fixtures", "plate1_CVCL_0023_100genes.h5ad")
FIXTURE_REF = os.path.join(os.path.dirname(__file__), "fixtures", "plate1_CVCL_0023_100genes_ref.parquet")


@pytest.mark.skipif(
    not os.path.exists(FIXTURE_H5AD),
    reason="Test fixture not available",
)
class TestCellevalMatchesReference:
    def test_deltas_match_reference(self):
        """Compute deltas from fixture h5ad and compare to reference parquet.

        The fixture is a subsampled (14K cells, 100 genes) extract from plate 1,
        cell line CVCL_0023. The reference is the corresponding slice from the
        original cell_eval_full parquet. Because the fixture is subsampled,
        the pseudobulk means differ slightly — we check correlation > 0.95
        rather than exact equality.
        """
        adata = anndata.read_h5ad(FIXTURE_H5AD)
        adata.var = adata.var.reset_index()

        cfg = {"gene_mode": "all_genes", "gene_name_col": "gene_name", "target_sum": 1872}
        X, gene_names = extract_expression(adata, cfg)

        group_vals = adata.obs["cell_line"].astype(str).values
        treat_vals = adata.obs["drugname_drugconc"].astype(str).values
        labels = np.array([f"{g}||{t}" for g, t in zip(group_vals, treat_vals)])

        means, g2i = pseudobulk_means(X, labels)
        our_df = compute_deltas(
            means,
            g2i,
            gene_names,
            "[('DMSO_TF', 0.0, 'uM')]",
            "cell_line",
            "treatment",
        )

        # Load reference (computed from the full plate, not subsampled)
        ref_df = pd.read_parquet(FIXTURE_REF)

        # Merge and compare
        our_genes = set(our_df.columns) - {"cell_line", "treatment"}
        ref_genes = set(ref_df.columns) - {"cell_line", "treatment"}
        shared = sorted(our_genes & ref_genes)
        assert len(shared) > 50, f"Expected 50+ shared genes, got {len(shared)}"

        merged = our_df.merge(ref_df, on=["cell_line", "treatment"], suffixes=("_new", "_ref"))
        assert len(merged) > 50, f"Expected 50+ matched treatments, got {len(merged)}"

        correlations = []
        for g in shared:
            new_vals = merged[f"{g}_new"].values.astype(np.float64)
            ref_vals = merged[f"{g}_ref"].values.astype(np.float64)
            valid = np.isfinite(new_vals) & np.isfinite(ref_vals)
            if valid.sum() > 5:
                corr = np.corrcoef(new_vals[valid], ref_vals[valid])[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)

        assert len(correlations) > 30, f"Only {len(correlations)} genes had enough data"
        mean_corr = np.mean(correlations)
        # Reference was generated from the same fixture, so should be exact
        assert mean_corr > 0.999, f"Mean gene correlation {mean_corr:.4f} < 0.999"
