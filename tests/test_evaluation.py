import numpy as np

from wildfire.evaluation import (
    BinaryAccumulator,
    CompetitionEvaluator,
    SeverityAccumulator,
    evaluation_mask,
)


def test_binary_accumulator_aggregates_globally():
    metric = BinaryAccumulator()
    metric.update(np.array([[1, 0]]), np.array([[1, 1]]))
    metric.update(np.array([[0, 1]]), np.array([[0, 1]]))
    assert metric.tp == 2
    assert metric.fp == 0
    assert metric.fn == 1
    assert np.isclose(metric.f1, 0.8)


def test_severity_accumulator_uses_one_for_absent_class():
    metric = SeverityAccumulator()
    metric.update(
        np.array([[0, 0]], dtype=np.uint8),
        np.array([[0, 1]], dtype=np.uint8),
    )
    assert metric.iou_by_class() == {1: 0.0, 2: 1.0, 3: 1.0}
    assert np.isclose(metric.miou, 2.0 / 3.0)


def test_competition_summary_requires_both_tasks_for_composite_score():
    evaluator = CompetitionEvaluator()
    evaluator.update_af(np.array([[1]]), np.array([[1]]))
    assert evaluator.summary()["score"] is None

    evaluator.update_bs(
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
    )
    assert evaluator.summary()["score"] == 1.0


def test_valid_mask_is_opt_in_for_official_evaluation():
    target = np.array([[1, 0]], dtype=np.uint8)
    channels = {"VALID_MASK": np.array([[1, 0]], dtype=np.uint8)}
    assert np.array_equal(
        evaluation_mask(channels, target),
        np.array([[True, True]]),
    )
    assert np.array_equal(
        evaluation_mask(channels, target, use_valid_mask=True),
        np.array([[True, False]]),
    )
