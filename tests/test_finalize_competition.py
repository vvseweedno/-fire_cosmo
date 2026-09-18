from scripts.finalize_competition import _assert_reproducible, _selected_metrics


def test_selected_metrics_uses_promoted_crossfit_bs_metrics():
    baseline = {
        "crossfit": {
            "f1_af": 0.9,
            "iou_burn": 0.5,
            "miou_severity": 0.4,
            "score": 0.61,
        }
    }
    candidates = {
        "promotion_allowed": True,
        "ensemble": {
            "iou_burn": 0.6,
            "miou_severity": 0.5,
            "bs_subscore": 0.36,
        },
    }
    promoted, metrics = _selected_metrics(baseline, candidates)
    assert promoted is True
    assert metrics["f1_af"] == 0.9
    assert metrics["iou_burn"] == 0.6
    assert metrics["miou_severity"] == 0.5
    assert abs(metrics["score"] - 0.675) < 1e-12


def test_selected_metrics_keeps_baseline_when_promotion_is_denied():
    baseline = {
        "crossfit": {
            "f1_af": 0.9,
            "iou_burn": 0.5,
            "miou_severity": 0.4,
            "score": 0.61,
        }
    }
    candidates = {
        "promotion_allowed": False,
        "ensemble": {
            "iou_burn": 0.8,
            "miou_severity": 0.8,
            "bs_subscore": 0.52,
        },
    }
    promoted, metrics = _selected_metrics(baseline, candidates)
    assert promoted is False
    assert metrics == baseline["crossfit"]


def test_reproducibility_requires_same_deployable_parameters():
    first = {
        "promoted": True,
        "deployment_signature": {"bs": {"score_weights": {"BASE": 0.5, "DNDMI_Z": 0.5}}},
        "metrics": {
            "f1_af": 0.9,
            "iou_burn": 0.6,
            "miou_severity": 0.5,
            "score": 0.675,
        },
    }
    second = {
        "promoted": True,
        "deployment_signature": first["deployment_signature"],
        "metrics": {
            "f1_af": 0.9,
            "iou_burn": 0.6,
            "miou_severity": 0.5,
            "score": 0.675,
        },
    }
    deltas = _assert_reproducible(first, second, tolerance=0.005)
    assert all(value == 0.0 for value in deltas.values())
