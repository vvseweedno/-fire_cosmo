from wildfire.release_gate import evaluate_proven_release, official_score


def _metrics(f1: float, iou: float, miou: float) -> dict[str, float]:
    payload = {
        "f1_af": f1,
        "iou_burn": iou,
        "miou_severity": miou,
    }
    payload["score"] = official_score(payload)
    return payload


def _evidence() -> dict[str, object]:
    baseline = _metrics(0.70, 0.60, 0.50)
    final = _metrics(0.72, 0.62, 0.52)
    return {
        "ci": {"green": True},
        "preflight": {"train": {"ok": True}, "test": {"ok": True}},
        "leakage_audit": {
            "status": "PASS",
            "pass": True,
            "strict_event_grouping": True,
            "chip_fallbacks": 0,
            "event_group_coverage": 1.0,
        },
        "crossfit": {"baseline": baseline, "final": final},
        "bootstrap": {
            "n_boot_used": 2000,
            "bootstrap_95_ci": [0.002, 0.020],
            "probability_delta_positive": 0.98,
            "score_a": final["score"],
            "score_b": baseline["score"],
            "delta_score": final["score"] - baseline["score"],
        },
        "reproducibility": {
            "run_1": final,
            "run_2": dict(final),
            "deployment_signature_equal": True,
        },
        "submission": {
            "run_1": {"valid": True, "sha256": "abc", "errors": []},
            "run_2": {"valid": True, "sha256": "abc", "errors": []},
        },
    }


def test_proven_release_requires_every_gate():
    report = evaluate_proven_release(_evidence())
    assert report["proven"] is True
    assert report["status"] == "PASS"
    assert all(item["pass"] for item in report["checks"].values())


def test_proven_release_rejects_non_improving_final_score():
    evidence = _evidence()
    evidence["crossfit"]["final"] = dict(evidence["crossfit"]["baseline"])
    report = evaluate_proven_release(evidence)
    assert report["proven"] is False
    assert report["checks"]["final_score_above_baseline"]["pass"] is False


def test_proven_release_rejects_inconclusive_bootstrap():
    evidence = _evidence()
    evidence["bootstrap"]["bootstrap_95_ci"] = [-0.001, 0.020]
    report = evaluate_proven_release(evidence)
    assert report["proven"] is False
    assert report["checks"]["bootstrap_stability"]["pass"] is False


def test_proven_release_rejects_different_submission_bytes():
    evidence = _evidence()
    evidence["submission"]["run_2"]["sha256"] = "different"
    report = evaluate_proven_release(evidence)
    assert report["proven"] is False
    assert report["checks"]["submission_validator_and_sha256"]["pass"] is False


def test_proven_release_rejects_chip_level_group_fallback():
    evidence = _evidence()
    evidence["leakage_audit"] = {
        "strict_event_grouping": False,
        "chip_fallbacks": 3,
        "event_group_coverage": 0.8,
    }
    report = evaluate_proven_release(evidence)
    assert report["proven"] is False
    assert report["checks"]["strict_leakage_grouping"]["pass"] is False


def test_proven_release_rejects_failed_explicit_leakage_audit():
    evidence = _evidence()
    evidence["leakage_audit"]["status"] = "FAIL"
    evidence["leakage_audit"]["pass"] = False
    report = evaluate_proven_release(evidence)
    assert report["proven"] is False
    assert report["checks"]["leakage_audit_pass"]["pass"] is False


def test_proven_release_rejects_bootstrap_crossfit_score_mismatch():
    evidence = _evidence()
    evidence["bootstrap"]["score_a"] += 0.01
    report = evaluate_proven_release(evidence)
    assert report["proven"] is False
    assert report["checks"]["bootstrap_stability"]["pass"] is False
