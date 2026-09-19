import json
from pathlib import Path

import pytest

from wildfire.experiment_registry import (
    ExperimentRecord,
    validate_experiment_record,
    write_experiment_record,
)


def _record(**overrides) -> ExperimentRecord:
    values = {
        "experiment_id": "af-hard-neg-folds-v1",
        "timestamp": "2026-09-18T20:00:00+00:00",
        "git_commit_sha": "a" * 40,
        "dataset_fingerprint": "b" * 64,
        "task": "AF",
        "candidate_name": "HARD_NEG_CONTEXT",
        "candidate_version": "1",
        "configuration": {"alpha_steps": 20},
        "seed": 42,
        "fold_manifest_sha256": "c" * 64,
        "train_events": ["event-1", "event-2"],
        "holdout_events": ["event-3"],
        "per_fold_metrics": [{"fold": 0, "f1_af": 0.8}],
        "aggregate_metrics": {"f1_af": 0.8, "score": 0.7},
        "official_score": 0.7,
        "reference_score": 0.69,
        "delta_score": 0.01,
        "runtime_seconds": 12.5,
        "peak_ram_mb": 512.0,
        "gpu_info": {"status": "NOT_PROBED"},
        "bootstrap_result": {"probability_delta_positive": 0.97},
        "state": "PROMOTED",
        "promoted": True,
        "promotion_reason": "cross-fit gain exceeded epsilon",
        "rejected_reason": None,
    }
    values.update(overrides)
    return ExperimentRecord(**values)


def test_experiment_record_writes_machine_readable_json(tmp_path: Path):
    record = _record()

    path = write_experiment_record(tmp_path, record)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "af-hard-neg-folds-v1.json"
    assert payload["git_commit_sha"] == "a" * 40
    assert payload["delta_score"] == pytest.approx(0.01)
    assert payload["promoted"] is True


def test_experiment_record_rejects_inconsistent_delta():
    record = _record(delta_score=0.5)

    with pytest.raises(ValueError, match="delta_score"):
        validate_experiment_record(record)


def test_rejected_experiment_requires_reason():
    record = _record(
        state="REJECTED",
        promoted=False,
        promotion_reason=None,
        rejected_reason=None,
    )

    with pytest.raises(ValueError, match="rejected_reason"):
        validate_experiment_record(record)


def test_promoted_experiment_requires_reason():
    record = _record(promotion_reason=None)

    with pytest.raises(ValueError, match="promotion_reason"):
        validate_experiment_record(record)
