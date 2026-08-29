"""Tests for rhaister/tutorial_data.py — the walkthrough's data path.

Everything here runs offline. The bridge test reuses the same fixture pair as
tests/test_data_prep.py: a subsampled plate-1 h5ad plus the matching slice of
the published cell_eval parquet, so it checks that the tutorial reproduces the
production pipeline rather than merely running without error.

Tests that reach HuggingFace are marked `network` and skipped unless
RHAISTER_TEST_NETWORK=1 is set.
"""

import os

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rhaister.prepare_combined import _TREATMENT_SINGLE_RE
from rhaister.tutorial_data import (
    TAHOE_CONTROL,
    _pack_subset,
    _unpack_subset,
    build_data_dict,
    compositional_split,
    de_from_anndata,
    frames_to_observations,
    load_gene_panel,
    stack_observations,
    treatment_label,
    walkthrough_holdout,
    walkthrough_treatments,
)

FIXTURE_H5AD = os.path.join(os.path.dirname(__file__), "fixtures", "plate1_CVCL_0023_100genes.h5ad")
FIXTURE_REF = os.path.join(os.path.dirname(__file__), "fixtures", "plate1_CVCL_0023_100genes_ref.parquet")

requires_network = pytest.mark.skipif(
    os.environ.get("RHAISTER_TEST_NETWORK") != "1",
    reason="set RHAISTER_TEST_NETWORK=1 to run tests that hit HuggingFace",
)


# ---------------------------------------------------------------------------
# treatment_label
# ---------------------------------------------------------------------------


class TestTreatmentLabel:
    def test_matches_pipeline_encoding(self):
        """Labels must parse with the regex prepare_combined uses on real data."""
        label = treatment_label("Bortezomib", 0.05)
        assert label == "[('Bortezomib', 0.05, 'uM')]"
        m = _TREATMENT_SINGLE_RE.match(label)
        assert m is not None
        assert m.group(1) == "Bortezomib"
        assert float(m.group(2)) == 0.05
        assert m.group(3) == "uM"

    def test_drug_names_with_punctuation(self):
        """Real Tahoe drug names carry spaces, hyphens and parentheses."""
        for drug in ("Belumosudil (mesylate)", "BI-3406", "Elimusertib hydrochloride"):
            m = _TREATMENT_SINGLE_RE.match(treatment_label(drug, 0.05))
            assert m is not None and m.group(1) == drug

    def test_control_label_round_trips(self):
        assert treatment_label("DMSO_TF", 0.0) == TAHOE_CONTROL


# ---------------------------------------------------------------------------
# de_from_anndata — the AnnData -> matrices bridge
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def computed():
    """The fixture h5ad and its DE summaries — computed once, pdex is not cheap."""
    adata = anndata.read_h5ad(FIXTURE_H5AD)
    adata.obs["treatment"] = adata.obs["drugname_drugconc"].astype(str)
    return adata, de_from_anndata(adata, control_label=TAHOE_CONTROL, verbose=False)


@pytest.mark.skipif(not os.path.exists(FIXTURE_H5AD), reason="Test fixture not available")
class TestDeFromAnnData:
    def test_shapes_and_keys(self, computed):
        _, out = computed
        n = len(out["gene_cols"])
        assert n == 100
        assert len(out["Y"]) > 50
        for name in ("Y", "P", "F", "D", "R"):
            assert set(out[name]) == set(out["Y"]), f"{name} keys differ from Y"
            for vec in out[name].values():
                assert vec.shape == (n,)
                assert np.isfinite(vec).all(), f"{name} contains non-finite values"

    def test_control_is_not_an_observation(self, computed):
        """The control is the reference, so it must not appear as a target."""
        _, out = computed
        assert all(t != TAHOE_CONTROL for _, t in out["Y"])

    def test_pvalues_and_fdr_in_unit_interval(self, computed):
        _, out = computed
        for name in ("P", "F"):
            vals = np.concatenate(list(out[name].values()))
            assert vals.min() >= 0.0 and vals.max() <= 1.0

    def test_reference_is_shared_within_a_cell_line(self, computed):
        """R is the control baseline — one vector per cell line, not per target."""
        _, out = computed
        vecs = list(out["R"].values())
        for vec in vecs[1:]:
            np.testing.assert_allclose(vec, vecs[0])

    def test_deltas_match_published_reference(self, computed):
        """D must reproduce the published cell_eval values for this cell line.

        Same check as tests/test_data_prep.py::TestCellevalMatchesReference, but
        through the tutorial's entry point — this is what proves the notebook
        teaches the real pipeline and not a lookalike.
        """
        _, out = computed
        ref = pd.read_parquet(FIXTURE_REF)
        genes = [g for g in out["gene_cols"] if g in ref.columns]
        assert len(genes) > 50

        pairs = [
            (str(c), str(t))
            for c, t in zip(ref["cell_line"], ref["treatment"])
            if (str(c), str(t)) in out["D"]
        ]
        assert len(pairs) > 50, f"only {len(pairs)} treatments matched the reference"

        gidx = {g: i for i, g in enumerate(out["gene_cols"])}
        ours = np.array([out["D"][p] for p in pairs])[:, [gidx[g] for g in genes]]
        theirs = ref.set_index(["cell_line", "treatment"]).loc[pairs, genes].to_numpy()

        cors = []
        for j in range(len(genes)):
            valid = np.isfinite(ours[:, j]) & np.isfinite(theirs[:, j])
            if valid.sum() > 5:
                c = np.corrcoef(ours[valid, j], theirs[valid, j])[0, 1]
                if np.isfinite(c):
                    cors.append(c)
        assert len(cors) > 30
        assert np.mean(cors) > 0.999, f"mean gene correlation {np.mean(cors):.4f}"

    def test_gene_panel_restricts_and_orders(self, computed):
        """Panel order wins, and genes absent from the data are dropped."""
        adata, _ = computed
        panel = ["MISSING_GENE_1", *adata.var_names[5:0:-1]]
        out = de_from_anndata(adata, gene_panel=panel, control_label=TAHOE_CONTROL, verbose=False)
        assert out["gene_cols"] == list(adata.var_names[5:0:-1])

    def test_rejects_disjoint_gene_panel(self, computed):
        adata, _ = computed
        with pytest.raises(ValueError, match="No genes in common"):
            de_from_anndata(adata, gene_panel=["NOT_A_GENE"], control_label=TAHOE_CONTROL, verbose=False)


# ---------------------------------------------------------------------------
# build_data_dict
# ---------------------------------------------------------------------------


def _synthetic_observations(n_cells=4, n_treats=5, n_genes=7, seed=0):
    rng = np.random.default_rng(seed)
    keys = [(f"CVCL_{c}", treatment_label(f"Drug{t}", 0.05)) for c in range(n_cells) for t in range(n_treats)]
    mats = {}
    for name, gen in (
        ("Y", lambda: rng.normal(size=n_genes)),
        ("P", lambda: rng.uniform(size=n_genes)),
        ("F", lambda: rng.uniform(size=n_genes)),
        ("D", lambda: rng.normal(size=n_genes)),
        ("R", lambda: rng.uniform(0, 10, size=n_genes)),
    ):
        mats[name] = {k: gen() for k in keys}
    return keys, mats, [f"GENE{i}" for i in range(n_genes)]


class TestBuildDataDict:
    def test_has_every_key_the_model_reads(self):
        keys, m, genes = _synthetic_observations()
        train, test = compositional_split(keys, {"CVCL_0": [treatment_label("Drug0", 0.05)]})
        data = build_data_dict(m["Y"], m["P"], m["F"], m["D"], m["R"], train, test, genes)

        required = {
            "Y_train", "P_train", "D_train", "F_train", "R_train",
            "Y_test", "P_test", "D_test", "F_test", "R_test",
            "train_cells", "train_treatments", "test_cells", "test_treatments",
            "gene_cols", "cell_to_idx", "treat_to_idx", "n_cells", "n_treatments",
            "n_test", "evaluate_test",
        }
        assert required <= set(data)

    def test_shapes_are_consistent(self):
        keys, m, genes = _synthetic_observations()
        train, test = compositional_split(keys, {"CVCL_1": [treatment_label("Drug2", 0.05)]})
        data = build_data_dict(m["Y"], m["P"], m["F"], m["D"], m["R"], train, test, genes)

        assert data["n_test"] == len(test)
        assert data["Y_train"].shape == (len(train), len(genes))
        assert data["Y_test"].shape == (len(test), len(genes))
        for name in ("P", "D", "F", "R"):
            assert data[f"{name}_train"].shape == data["Y_train"].shape
            assert data[f"{name}_test"].shape == data["Y_test"].shape
        assert len(data["train_cells"]) == len(train)
        assert len(data["test_treatments"]) == len(test)

    def test_row_order_follows_keys(self):
        keys, m, genes = _synthetic_observations()
        train, test = compositional_split(keys, {"CVCL_2": [treatment_label("Drug3", 0.05)]})
        data = build_data_dict(m["Y"], m["P"], m["F"], m["D"], m["R"], train, test, genes)
        for i, (cell, treat) in enumerate(test):
            assert data["test_cells"][i] == cell
            assert data["test_treatments"][i] == treat
            np.testing.assert_allclose(data["Y_test"][i], m["Y"][(cell, treat)])

    def test_evaluator_is_one_shot(self):
        keys, m, genes = _synthetic_observations()
        train, test = compositional_split(keys, {"CVCL_0": [treatment_label("Drug1", 0.05)]})
        data = build_data_dict(m["Y"], m["P"], m["F"], m["D"], m["R"], train, test, genes)

        pred = np.asarray(data["Y_test"])
        data["evaluate_test"](pred, D_pred=np.asarray(data["D_test"]))
        with pytest.raises(RuntimeError, match="already called"):
            data["evaluate_test"](pred)

    def test_rejects_overlapping_splits(self):
        keys, m, genes = _synthetic_observations()
        with pytest.raises(ValueError, match="overlap"):
            build_data_dict(m["Y"], m["P"], m["F"], m["D"], m["R"], keys, keys[:2], genes)

    def test_rejects_empty_test_set(self):
        keys, m, genes = _synthetic_observations()
        with pytest.raises(ValueError, match="test_keys is empty"):
            build_data_dict(m["Y"], m["P"], m["F"], m["D"], m["R"], keys, [], genes)


class TestStackObservations:
    def test_empty_needs_gene_count(self):
        assert stack_observations([], {}, n_genes=3).shape == (0, 3)
        with pytest.raises(ValueError, match="n_genes is required"):
            stack_observations([], {})


# ---------------------------------------------------------------------------
# compositional_split
# ---------------------------------------------------------------------------


class TestCompositionalSplit:
    def test_partitions_all_keys(self):
        keys, _, _ = _synthetic_observations()
        holdout = {"CVCL_0": [treatment_label("Drug0", 0.05), treatment_label("Drug1", 0.05)]}
        train, test = compositional_split(keys, holdout)
        assert len(train) + len(test) == len(keys)
        assert set(train).isdisjoint(test)
        assert len(test) == 2

    def test_rejects_entirely_held_out_cell_line(self):
        """A cell line with no training rows leaves the ridge nothing to fit."""
        keys, _, _ = _synthetic_observations(n_cells=2, n_treats=2)
        holdout = {"CVCL_0": [treatment_label(f"Drug{t}", 0.05) for t in range(2)]}
        with pytest.raises(ValueError, match="held out entirely"):
            compositional_split(keys, holdout)

    def test_rejects_entirely_held_out_treatment(self):
        keys, _, _ = _synthetic_observations(n_cells=2, n_treats=3)
        holdout = {f"CVCL_{c}": [treatment_label("Drug0", 0.05)] for c in range(2)}
        with pytest.raises(ValueError, match="never appears in train"):
            compositional_split(keys, holdout)

    def test_rejects_unknown_pairs(self):
        keys, _, _ = _synthetic_observations()
        with pytest.raises(ValueError, match="not present in the data"):
            compositional_split(keys, {"CVCL_9": [treatment_label("Drug0", 0.05)]})

    def test_walkthrough_defaults_are_a_valid_split(self):
        """The notebook's default holdout must survive its own validation."""
        cells = [f"CVCL_{i}" for i in range(8)]
        treats = walkthrough_treatments()
        keys = [(c, t) for c in cells for t in treats]
        holdout = dict(walkthrough_holdout())
        # Remap the real cell ids onto the synthetic ones for a pure logic check.
        holdout = {cells[i]: ts for i, ts in enumerate(holdout.values())}
        train, test = compositional_split(keys, holdout)
        assert len(test) == sum(len(v) for v in holdout.values())
        assert len(train) == len(keys) - len(test)


# ---------------------------------------------------------------------------
# Subset cache round trip
# ---------------------------------------------------------------------------


def _synthetic_frames(cells=("CVCL_0", "CVCL_1"), drugs=("A", "B", "C"), genes=("G1", "G2")):
    rows = [{"cell_line": c, "treatment": treatment_label(d, 0.05)} for c in cells for d in drugs]
    frames = []
    for k in range(5):
        df = pd.DataFrame(rows)
        for j, g in enumerate(genes):
            df[g] = np.arange(len(df), dtype=float) + k * 100 + j
        frames.append(df)
    return tuple(frames), list(genes)


class TestSubsetCache:
    def test_round_trip_preserves_values(self):
        frames, genes = _synthetic_frames()
        packed = _pack_subset(*frames)
        cells = sorted(set(frames[0]["cell_line"]))
        treats = sorted(set(frames[0]["treatment"]))
        out = _unpack_subset(packed, cells, treats)
        assert out is not None
        for original, restored in zip(frames, out):
            expected = original.sort_values(["cell_line", "treatment"]).reset_index(drop=True)
            np.testing.assert_allclose(
                restored[genes].to_numpy(dtype=float), expected[genes].to_numpy(dtype=float), rtol=1e-6
            )

    def test_returns_none_when_cache_misses_request(self):
        frames, _ = _synthetic_frames()
        packed = _pack_subset(*frames)
        assert _unpack_subset(packed, ["CVCL_0", "CVCL_MISSING"], [treatment_label("A", 0.05)]) is None
        assert _unpack_subset(packed, ["CVCL_0"], [treatment_label("ZZZ", 0.05)]) is None

    def test_frames_to_observations_matches_frames(self):
        frames, genes = _synthetic_frames()
        obs = frames_to_observations(*frames, genes)
        assert set(obs) == {"Y", "P", "F", "R", "D"}
        first = frames[0]
        key = (first["cell_line"].iloc[0], first["treatment"].iloc[0])
        np.testing.assert_allclose(obs["Y"][key], first[genes].to_numpy(dtype=float)[0])


# ---------------------------------------------------------------------------
# End to end on synthetic single cells
# ---------------------------------------------------------------------------


class TestBridgeToModelInputs:
    def test_de_from_anndata_feeds_build_data_dict(self):
        """The two halves of the notebook must actually compose."""
        rng = np.random.default_rng(0)
        cells, drugs, n_genes = [f"CVCL_{i}" for i in range(3)], ["A", "B", "C"], 6
        records, blocks = [], []
        for c in cells:
            for d in [*drugs, "DMSO_TF"]:
                label = TAHOE_CONTROL if d == "DMSO_TF" else treatment_label(d, 0.05)
                for _ in range(20):
                    records.append({"cell_line": c, "treatment": label})
                blocks.append(rng.poisson(5, size=(20, n_genes)))

        adata = anndata.AnnData(
            X=sp.csr_matrix(np.vstack(blocks).astype(np.float32)),
            obs=pd.DataFrame(records, index=[f"c{i}" for i in range(len(records))]),
        )
        adata.var_names = [f"GENE{i}" for i in range(n_genes)]

        out = de_from_anndata(adata, control_label=TAHOE_CONTROL, verbose=False)
        keys = sorted(out["Y"])
        assert len(keys) == len(cells) * len(drugs)

        train, test = compositional_split(keys, {cells[0]: [treatment_label("A", 0.05)]})
        data = build_data_dict(
            out["Y"], out["P"], out["F"], out["D"], out["R"], train, test, out["gene_cols"]
        )
        assert data["Y_train"].shape == (len(train), n_genes)
        assert np.isfinite(data["Y_train"]).all()


# ---------------------------------------------------------------------------
# Network-dependent
# ---------------------------------------------------------------------------


class TestGenePanel:
    def test_loads_tahoe_panel(self):
        panel = load_gene_panel("tahoe")
        assert len(panel) == 2000
        assert all(isinstance(g, str) for g in panel)


@requires_network
class TestFetchDeSubset:
    def test_fetches_requested_slice(self, tmp_path):
        from rhaister.tutorial_data import WALKTHROUGH_CELL_LINES, WALKTHROUGH_PLATE, fetch_de_subset

        cells = WALKTHROUGH_CELL_LINES[:2]
        treats = walkthrough_treatments()[:2]
        frames = fetch_de_subset(
            WALKTHROUGH_PLATE, cells, treats, load_gene_panel("tahoe"),
            cache_path=str(tmp_path / "subset.parquet"), use_bundled=False, verbose=False,
        )
        fc = frames[0]
        assert set(fc["cell_line"]) <= set(cells)
        assert set(fc["treatment"]) <= set(treats)
        assert len(fc) == len(cells) * len(treats)
