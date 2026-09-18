import numpy as np

from wildfire.constants import LC_CROP, LC_TREE
from wildfire.fusion import BurnFusionComponents
from wildfire.metrics import binary_iou, severity_miou
from wildfire.model_config import ModelConfig
from wildfire.training import (
    apply_landcover_thresholds,
    calibrate_af_threshold,
    calibrate_af_threshold_exact,
    calibrate_bs_cloud_sar_fallback,
    calibrate_bs_landcover_thresholds,
    calibrate_bs_thresholds,
    threshold_grid,
)


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


def test_exact_af_calibration_counts_gt_outside_model_valid_as_fn():
    score = np.array([[10.0, 10.0]], dtype=np.float32)
    target = np.array([[1, 1]], dtype=np.uint8)
    model_valid = np.array([[1, 0]], dtype=bool)

    calibrated, result = calibrate_af_threshold_exact(
        [(score, target, model_valid)],
        ModelConfig(),
    )
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert np.isclose(result["f1"], 2.0 / 3.0)
    assert np.isclose(
        calibrated.training["af_threshold_calibration"]["f1"],
        2.0 / 3.0,
    )


def test_bs_calibration_learns_ordered_thresholds():
    score = np.array([[0.0, 0.05, 0.2, 0.25, 0.5, 0.55, 0.8, 0.85]], dtype=np.float32)
    target = np.array([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)

    calibrated, result = calibrate_bs_thresholds(
        [(score, target, valid)],
        ModelConfig(),
        max_candidates=32,
        passes=4,
    )
    assert np.isclose(result["bs_subscore"], 0.65)
    assert calibrated.bs.default_thresholds == calibrated.bs.forest_thresholds
    assert calibrated.bs.default_thresholds[0] < calibrated.bs.default_thresholds[1]
    assert calibrated.bs.default_thresholds[1] < calibrated.bs.default_thresholds[2]


def test_cloud_sar_optimizer_can_recover_cloudy_severity_pixels():
    optical = np.array([[0.0, 0.2, 0.5, 0.8, 0.0, 0.0, 0.0]], dtype=np.float32)
    sar = np.array([[0.0, 0.0, 0.0, 0.0, 0.2, 0.35, 0.6]], dtype=np.float32)
    clear = np.array([[1, 1, 1, 1, 0, 0, 0]], dtype=bool)
    base_valid = np.ones_like(clear)
    sar_available = np.ones_like(clear)
    target = np.array([[0, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)

    components = BurnFusionComponents(
        optical_score=optical,
        sar_score=sar,
        optical_valid=clear,
        base_valid=base_valid,
        sar_available=sar_available,
    )
    calibrated, result = calibrate_bs_cloud_sar_fallback(
        [(components, target)],
        ModelConfig(),
        cloud_weight_candidates=(0.0, 1.0),
        max_candidates=32,
        passes=4,
    )

    assert calibrated.bs.cloud_sar_weight == 1.0
    assert np.isclose(result["bs_subscore"], 0.65)


def test_landcover_threshold_refinement_can_recover_crop_severity():
    score = np.array(
        [[0.0, 0.15, 0.35, 0.60, 0.0, 0.05, 0.10, 0.15]],
        dtype=np.float32,
    )
    target = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)
    landcover = np.array(
        [[LC_TREE, LC_TREE, LC_TREE, LC_TREE, LC_CROP, LC_CROP, LC_CROP, LC_CROP]],
        dtype=np.int16,
    )

    base = ModelConfig()
    baseline_prediction = apply_landcover_thresholds(
        score,
        valid,
        landcover,
        base,
    )
    baseline_subscore = (
        0.35 * binary_iou(baseline_prediction > 0, target > 0)
        + 0.30 * severity_miou(baseline_prediction, target)
    )

    calibrated, result = calibrate_bs_landcover_thresholds(
        [(score, target, valid, landcover)],
        base,
        max_candidates=32,
        passes=3,
    )

    assert calibrated.bs.crop_thresholds != base.bs.crop_thresholds
    assert result["bs_subscore"] >= baseline_subscore
    assert result["miou_severity"] > severity_miou(baseline_prediction, target)
