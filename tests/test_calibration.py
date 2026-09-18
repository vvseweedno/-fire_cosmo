import numpy as np

from wildfire.calibration import (
    apply_ordered_thresholds,
    exact_f1_threshold,
    optimize_ordered_thresholds,
)


def test_exact_f1_threshold_finds_true_boundary():
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    target = np.array([0, 0, 1, 1], dtype=np.uint8)
    result = exact_f1_threshold(scores, target)
    assert np.isclose(result["threshold"], 0.8)
    assert result["f1"] == 1.0
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_exact_threshold_counts_positive_outside_valid_as_fn():
    scores = np.array([0.9, 0.8], dtype=np.float32)
    target = np.array([1, 1], dtype=np.uint8)
    valid = np.array([1, 0], dtype=bool)
    result = exact_f1_threshold(scores, target, valid)
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert np.isclose(result["f1"], 2.0 / 3.0)


def test_ordered_threshold_optimizer_improves_bs_subscore():
    scores = np.array([0.0, 0.05, 0.2, 0.25, 0.5, 0.55, 0.8, 0.85])
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)

    initial = apply_ordered_thresholds(scores, (0.10, 0.27, 0.44), valid)
    assert not np.array_equal(initial, target)

    result = optimize_ordered_thresholds(
        scores,
        target,
        valid,
        initial=(0.10, 0.27, 0.44),
        max_candidates=32,
        passes=4,
    )
    final = apply_ordered_thresholds(scores, result["thresholds"], valid)
    assert np.array_equal(final, target)
    assert np.isclose(result["bs_subscore"], 0.65)
