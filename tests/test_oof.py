from pathlib import Path

import numpy as np
import pytest

from wildfire.model_config import ModelConfig
from wildfire.oof import OOFPool, OOFRecord, load_oof_directory, save_oof_record


def _record(chip_id: str, task: str, score, target) -> OOFRecord:
    score_array = np.asarray(score, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.uint8)
    return OOFRecord(
        chip_id=chip_id,
        task=task,
        score=score_array,
        target=target_array,
        valid=np.ones_like(target_array, dtype=bool),
    )


def test_pooled_oof_calibration_can_reach_perfect_score():
    pool = OOFPool()
    pool.add(_record("af_1", "AF", [[0.1, 0.9]], [[0, 1]]))
    pool.add(_record("af_2", "AF", [[0.2, 0.8]], [[0, 1]]))
    pool.add(_record("bs_1", "BS", [[0.0, 0.2, 0.5, 0.8]], [[0, 1, 2, 3]]))
    pool.add(_record("bs_2", "BS", [[0.05, 0.25, 0.55, 0.85]], [[0, 1, 2, 3]]))

    calibrated, report = pool.calibrate_and_evaluate(
        ModelConfig(),
        bs_max_candidates=32,
        bs_passes=4,
    )

    assert report["score"] == 1.0
    assert report["f1_af"] == 1.0
    assert report["iou_burn"] == 1.0
    assert report["miou_severity"] == 1.0
    assert calibrated.af.threshold >= 0.8


def test_oof_rejects_duplicate_chip():
    pool = OOFPool()
    record = _record("af_1", "AF", [[0.1]], [[0]])
    pool.add(record)
    with pytest.raises(ValueError, match="duplicate"):
        pool.add(record)


def test_oof_record_roundtrip(tmp_path: Path):
    record = _record("bs_7", "BS", [[0.1, 0.8]], [[0, 3]])
    save_oof_record(record, tmp_path / "bs_7.npz")
    pool = load_oof_directory(tmp_path)
    loaded = pool.records[0]
    assert loaded.chip_id == "bs_7"
    assert loaded.task == "BS"
    assert np.array_equal(loaded.score, record.score)
    assert np.array_equal(loaded.target, record.target)


def test_oof_expected_coverage_is_strict():
    pool = OOFPool()
    pool.add(_record("af_1", "AF", [[0.1]], [[0]]))
    with pytest.raises(ValueError, match="coverage mismatch"):
        pool.validate_expected({"af_1", "af_2"})
