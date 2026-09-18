import numpy as np

from wildfire.evaluation import BinaryAccumulator, CompetitionEvaluator, SeverityAccumulator


def test_binary_accumulator_aggregates_globally():
    metric = BinaryAccumulator()
    metric.update(np.array([[1, 0]]), np.array([[1, 1]]))
    metric.update(np.array([[0, 1]]), np.array([[0, 1]]))
    assert metric.tp == 2
    assert metric.fp == 0
    assert metric.fn == 1
    assert np.isclose(metric.f1, 0.8)


def test_severity_accumulator_ignores_absent_class_in_miou():
    metric = SeverityAccumulator()
    metric.update(
        np.array([[0, 1], [2, 2]], dtype=np.uint8),
        np.array([[0, 1], [2, 3]], dtype=np.uint8),
    )
    assert np.isclose(metric.miou, (1.0 + 0.5 + 0.0) / 3.0)


def test_competition_summary_requires_both_tasks_for_composite_score():
    evaluator = CompetitionEvaluator()
    evaluator.update_af(np.array([[1]]), np.array([[1]]))
    assert evaluator.summary()["score"] is None

    evaluator.update_bs(
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
    )
    assert evaluator.summary()["score"] == 1.0


def test_valid_mask_excludes_pixels():
    evaluator = CompetitionEvaluator()
    evaluator.update_af(
        np.array([[1, 1]], dtype=np.uint8),
        np.array([[1, 0]], dtype=np.uint8),
        np.array([[1, 0]], dtype=np.uint8),
    )
    assert evaluator.summary()["f1_af"] == 1.0
