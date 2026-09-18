from pathlib import Path

import numpy as np

from scripts.optimize_af_candidate_ensemble import run as optimize_run
from wildfire.af_candidates import AF_CANDIDATE_NAMES
from wildfire.af_ensemble_validation import crossfit_af_candidate_ensemble
from wildfire.model_config import load_model_config
from wildfire.oof import OOFRecord, save_oof_record


def _candidate_fixture(root: Path) -> dict[str, object]:
    target = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)
    baseline = np.full(target.shape, 0.5, dtype=np.float32)
    strong = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0], dtype=np.float32)

    chips = ["af_0", "af_1", "af_2", "af_3"]
    for name in AF_CANDIDATE_NAMES:
        score = strong if name == "I45_Z" else baseline
        for chip_id in chips:
            save_oof_record(
                OOFRecord(
                    chip_id=chip_id,
                    task="AF",
                    score=score,
                    target=target,
                    valid=valid,
                ),
                root / name / f"{chip_id}.npz",
            )

    return {
        "folds": [
            {"fold": 0, "validation": ["af_0", "af_1"]},
            {"fold": 1, "validation": ["af_2", "af_3"]},
        ]
    }


def test_crossfit_af_candidate_ensemble_promotes_only_measured_gain(tmp_path: Path):
    root = tmp_path / "candidates"
    manifest = _candidate_fixture(root)

    report = crossfit_af_candidate_ensemble(
        root,
        manifest,
        alpha_steps=4,
    )

    assert report["promotion_allowed"] is True
    assert report["delta_f1"] > 0
    assert report["delta_total_score"] > 0
    assert report["ensemble_f1"] > report["baseline_f1"]


def test_pooled_af_optimizer_freezes_weights_and_threshold(tmp_path: Path):
    root = tmp_path / "candidates"
    _candidate_fixture(root)
    output_config = tmp_path / "ensemble_config.json"
    output_report = tmp_path / "ensemble_report.json"

    report = optimize_run(
        root,
        output_config=output_config,
        output_report=output_report,
        alpha_steps=4,
    )

    config = load_model_config(output_config)
    assert output_report.exists()
    assert report["weights"].get("I45_Z", 0.0) > 0
    assert config.af.score_weights.get("I45_Z", 0.0) > 0
    assert np.isclose(sum(config.af.score_weights.values()), 1.0)
    assert np.isfinite(config.af.threshold)
