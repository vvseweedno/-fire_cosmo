import numpy as np

from wildfire.model_config import ModelConfig
from wildfire.training import calibrate_af_threshold, threshold_grid


def test_threshold_grid_is_deterministic_and_inclusive():
    assert threshold_grid(2.0, 3.0, 0.5) == [2.0, 2.5, 3.0]


def test_af_threshold_calibration_maximises_micro_f1():
    score = np.array([[0.0, 2.2, 4.2, 7.0]], dtype=np.float32)
    target = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)

    calibrated, trace = calibrate_af_threshold(
        [(score, target, valid)],
        [2.0, 4.0, 6.0],
        ModelConfig(),
    )
    assert calibrated.af.threshold == 4.0
    assert calibrated.training["af_threshold_calibration"]["best_f1_train"] == 1.0
    assert len(trace) == 3


def test_af_calibration_counts_gt_outside_model_valid_as_fn():
    score = np.array([[10.0, 10.0]], dtype=np.float32)
    target = np.array([[1, 1]], dtype=np.uint8)
    model_valid = np.array([[1, 0]], dtype=bool)

    calibrated, trace = calibrate_af_threshold(
        [(score, target, model_valid)],
        [4.0],
        ModelConfig(),
    )
    assert trace[0]["tp"] == 1
    assert trace[0]["fn"] == 1
    assert np.isclose(trace[0]["f1"], 2.0 / 3.0)
    assert np.isclose(
        calibrated.training["af_threshold_calibration"]["best_f1_train"],
        2.0 / 3.0,
    )
