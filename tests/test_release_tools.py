from pathlib import Path

from scripts.benchmark_inference import runtime_tier
from scripts.readiness_scorecard import build_scorecard
from scripts.reproduce_final import build_manifest


def test_runtime_tier_matches_official_boundaries():
    assert runtime_tier(29.999) == "8/8"
    assert runtime_tier(30.0) == "6/8"
    assert runtime_tier(300.0) == "6/8"
    assert runtime_tier(720.0) == "4/8"
    assert runtime_tier(1800.0) == "2/8"
    assert runtime_tier(None) == "PENDING"


def test_scorecard_never_claims_ready_without_evidence(tmp_path: Path):
    scorecard = build_scorecard(tmp_path)

    assert scorecard["release_status"] == "NOT READY"
    assert all(value is False for value in scorecard["hard_gates"].values())


def test_manifest_is_pending_without_measured_final_run(tmp_path: Path):
    manifest = build_manifest(tmp_path)

    assert manifest["status"] == "PENDING"
    assert manifest["release_ready"] is False
