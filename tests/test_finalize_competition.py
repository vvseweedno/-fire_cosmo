from scripts.finalize_competition import _assert_reproducible, _selected_metrics


def _baseline():
    return {
        "crossfit": {
            "f1_af": 0.9,
            "iou_burn": 0.5,
            "miou_severity": 0.4,
            "score": 0.61,
        }
    }


def test_selected_metrics_combines_independently_promoted_af_and_bs():
    af = {
        "promotion_allowed": True,
        "ensemble_f1": 0.92,
    }
    bs = {
        "promotion_allowed": True,
        "ensemble": {
            "iou_burn": 0.6,
            "miou_severity": 0.5,
            "bs_subscore": 0.36,
        },
    }
    promotions, metrics = _selected_metrics(_baseline(), af, bs)
    assert promotions == {"af": True, "bs": True}
    assert metrics["f1_af"] == 0.92
    assert metrics["iou_burn"] == 0.6
    assert metrics["miou_severity"] == 0.5
    assert abs(metrics["score"] - (0.35 * 0.92 + 0.35 * 0.6 + 0.30 * 0.5)) < 1e-12


def test_selected_metrics_keeps_baseline_components_when_promotions_are_denied():
    af = {
        "promotion_allowed": False,
        "ensemble_f1": 0.99,
    }
    bs = {
        "promotion_allowed": False,
        "ensemble": {
            "iou_burn": 0.8,
            "miou_severity": 0.8,
            "bs_subscore": 0.52,
        },
    }
    promotions, metrics = _selected_metrics(_baseline(), af, bs)
    assert promotions == {"af": False, "bs": False}
    assert metrics == _baseline()["crossfit"]


def test_reproducibility_requires_same_deployable_parameters():
    first = {
        "promotions": {"af": True, "bs": True},
        "deployment_signature": {
            "af": {"score_weights": {"BASE": 0.5, "I45_Z": 0.5}},
            "bs": {"score_weights": {"BASE": 0.5, "DNDMI_Z": 0.5}},
        },
        "metrics": {
            "f1_af": 0.9,
            "iou_burn": 0.6,
            "miou_severity": 0.5,
            "score": 0.675,
        },
    }
    second = {
        "promotions": dict(first["promotions"]),
        "deployment_signature": first["deployment_signature"],
        "metrics": dict(first["metrics"]),
    }
    deltas = _assert_reproducible(first, second, tolerance=1e-8)
    assert all(value == 0.0 for value in deltas.values())
