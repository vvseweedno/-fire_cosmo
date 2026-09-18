import numpy as np

from wildfire.metrics import binary_f1, binary_iou, competition_score, severity_miou


def test_binary_metrics_perfect():
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert binary_f1(target, target) == 1.0
    assert binary_iou(target, target) == 1.0


def test_severity_miou():
    target = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    pred = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    assert np.isclose(severity_miou(pred, target), (1.0 + 0.5 + 0.0) / 3)


def test_competition_score_weights():
    af = np.array([[0, 1]], dtype=np.uint8)
    bs = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    result = competition_score(af, af, bs, bs)
    assert result["score"] == 1.0
