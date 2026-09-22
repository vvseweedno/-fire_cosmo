import numpy as np

from wildfire.diagnostics import (
    af_error_summary,
    bs_error_summary,
    candidate_diversity,
)


def test_af_error_summary_matches_hand_counts():
    target = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    pred = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    report = af_error_summary(pred, target)

    assert report["tp"] == 2
    assert report["fp"] == 1
    assert report["fn"] == 1
    assert report["tn"] == 2
    assert report["precision"] == 2 / 3
    assert report["recall"] == 2 / 3
    assert report["f1"] == 2 / 3


def test_af_error_summary_breaks_errors_down_by_landcover():
    target = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    pred = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    landcover = np.array([[50, 80], [10, 10]], dtype=np.int16)

    report = af_error_summary(pred, target, landcover=landcover)

    assert report["false_positive_landcover"] == {"10": 0, "50": 1, "80": 1}
    assert report["false_negative_landcover"] == {"10": 1, "50": 0, "80": 0}


def test_bs_error_summary_reports_confusion_and_iou():
    target = np.array([[0, 1, 2], [3, 2, 0]], dtype=np.uint8)
    pred = np.array([[0, 1, 1], [3, 0, 2]], dtype=np.uint8)

    report = bs_error_summary(pred, target)

    confusion = report["severity_confusion"]
    assert confusion["2"]["1"] == 1
    assert confusion["2"]["0"] == 1
    assert confusion["0"]["2"] == 1
    assert report["burn"]["tp"] == 3
    assert report["burn"]["fp"] == 1
    assert report["burn"]["fn"] == 1


def test_candidate_diversity_measures_complementary_errors():
    target = np.array([0, 1, 1, 0, 1, 0], dtype=np.uint8)
    predictions = {
        "A": np.array([0, 1, 0, 0, 1, 1], dtype=np.uint8),
        "B": np.array([0, 0, 1, 0, 1, 1], dtype=np.uint8),
    }

    report = candidate_diversity(predictions, target)
    pair = report["pairs"][0]

    assert pair["left"] == "A"
    assert pair["right"] == "B"
    assert pair["error_overlap_pixels"] == 1
    assert pair["error_symmetric_difference_pixels"] == 2
    assert pair["left_only_error_pixels"] == 1
    assert pair["right_only_error_pixels"] == 1
    assert pair["disagreement_rate"] == 2 / 6


def test_candidate_diversity_rejects_shape_mismatch():
    try:
        candidate_diversity(
            {"A": np.zeros(3), "B": np.zeros(4)},
            np.zeros(3),
        )
    except ValueError as exc:
        assert "prediction shape differs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("shape mismatch must fail")
