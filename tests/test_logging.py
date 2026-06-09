"""Tests for logging functions in prepare_combined.py."""

import json
import os
import tempfile

import pytest

from rhaister import prepare_combined

# ---------------------------------------------------------------------------
# Helpers — redirect RESULTS_JSONL to a temp file
# ---------------------------------------------------------------------------


class _TmpResultsFile:
    """Context manager that temporarily redirects RESULTS_JSONL to a temp file."""

    def __init__(self):
        self.tmpfile = None
        self.original = None

    def __enter__(self):
        self.tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        self.tmpfile.close()
        self.original = prepare_combined.RESULTS_JSONL
        prepare_combined.RESULTS_JSONL = self.tmpfile.name
        return self.tmpfile.name

    def __exit__(self, *args):
        prepare_combined.RESULTS_JSONL = self.original
        try:
            os.unlink(self.tmpfile.name)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# log_result
# ---------------------------------------------------------------------------


class TestLogResult:
    def test_appends_valid_jsonl(self):
        with _TmpResultsFile() as path:
            metrics = {
                "pdex_static/pearson_delta_mean": 0.65,
                "state/pr_auc_mean": 0.77,
            }
            # Disable wandb for test
            orig_wandb = os.environ.get("WANDB_MODE")
            os.environ["WANDB_MODE"] = "disabled"
            try:
                prepare_combined.log_result("test_exp", metrics, notes="unit test")
            finally:
                if orig_wandb is None:
                    os.environ.pop("WANDB_MODE", None)
                else:
                    os.environ["WANDB_MODE"] = orig_wandb

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["experiment"] == "test_exp"
            assert record["metrics"]["pdex_static/pearson_delta_mean"] == 0.65
            assert record["notes"] == "unit test"
            assert "timestamp" in record

    def test_appends_multiple(self):
        with _TmpResultsFile() as path:
            orig_wandb = os.environ.get("WANDB_MODE")
            os.environ["WANDB_MODE"] = "disabled"
            try:
                prepare_combined.log_result("exp1", {"m": 1.0})
                prepare_combined.log_result("exp2", {"m": 2.0})
            finally:
                if orig_wandb is None:
                    os.environ.pop("WANDB_MODE", None)
                else:
                    os.environ["WANDB_MODE"] = orig_wandb

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["experiment"] == "exp1"
            assert json.loads(lines[1])["experiment"] == "exp2"


# ---------------------------------------------------------------------------
# update_decision
# ---------------------------------------------------------------------------


class TestUpdateDecision:
    def test_updates_correct_record(self):
        with _TmpResultsFile() as path:
            # Seed with two records
            records = [
                {"timestamp": "T1", "experiment": "exp1", "metrics": {}, "notes": "", "decision": None},
                {"timestamp": "T2", "experiment": "exp2", "metrics": {}, "notes": "", "decision": None},
            ]
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            prepare_combined.update_decision("exp1", "accepted")

            with open(path) as f:
                updated = [json.loads(line) for line in f]
            assert updated[0]["decision"] == "accepted"
            assert updated[1]["decision"] is None

    def test_updates_most_recent(self):
        with _TmpResultsFile() as path:
            records = [
                {"timestamp": "T1", "experiment": "dup", "metrics": {}, "notes": "", "decision": None},
                {"timestamp": "T2", "experiment": "dup", "metrics": {}, "notes": "", "decision": None},
            ]
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            prepare_combined.update_decision("dup", "rejected")

            with open(path) as f:
                updated = [json.loads(line) for line in f]
            assert updated[0]["decision"] is None  # first unchanged
            assert updated[1]["decision"] == "rejected"  # most recent updated

    def test_bad_experiment_raises(self):
        with _TmpResultsFile() as path:
            records = [
                {"timestamp": "T1", "experiment": "exp1", "metrics": {}, "notes": "", "decision": None},
            ]
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            with pytest.raises(ValueError, match="No JSONL entry found"):
                prepare_combined.update_decision("nonexistent", "accepted")

    def test_bad_decision_raises(self):
        with _TmpResultsFile():
            with pytest.raises(ValueError, match="must be"):
                prepare_combined.update_decision("exp1", "maybe")
