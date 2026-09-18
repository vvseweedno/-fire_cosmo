from pathlib import Path

import numpy as np

from scripts.optimize_bs_candidate_ensemble import run as optimize_run
from wildfire.bs_candidates import BS_CANDIDATE_NAMES
from wildfire.bs_ensemble_validation import crossfit_bs_candidate_ensemble
from wildfire.model_config import ModelConfig, load_model_config
from wildfire.oof import OOFRecord, save_oof_record


def _candidate_fixture(root: Path) -> dict[str, object]:
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)
    baseline = np.full(target.shape, 0.5, dtype=np.float32)
    strong = np.array(
        [0.0, 0.05, 0.20, 0.25, 0.50, 0.55, 0.80, 0.85],
        dtype=np.float32,
    )
    landcover = np.full(target.shape, -1, dtype=np.int16)

    chips = ["bs_0", "bs_1", "bs_2", "bs_3"]
    for name in BS_CANDIDATE_NAMES:
        score = strong if name == "RBR_Z" else baseline
        for chip_id in chips:
            save_oof_record(
                OOFRecord(
                    chip_id=chip_id,
                    task="BS",
                    score=score,
                    target=target,
                    valid=valid,
                    landcover=landcover,
                ),
                root / name / f"{chip_id}.npz",
            )

    return {
        "folds": [
            {"fold": 0, "validation": ["bs_0", "bs_1"]},
            {"fold": 1, "validation": ["bs_2", "bs_3"]},
        ]
    }


def test_crossfit_candidate_ensemble_promotes_only_measured_gain(tmp_path: Path):
    root = tmp_path / "candidates"
    manifest = _candidate_fixture(root)

    report = crossfit_bs_candidate_ensemble(
        root,
        manifest,
        ModelConfig(),
        alpha_steps=4,
        threshold_candidates=16,
        threshold_passes=2,
        landcover_passes=1,
    )

    assert report["promotion_allowed"] is True
    assert report["delta_total_score"] > 0
    assert report["ensemble"]["bs_subscore"] > report["baseline"]["bs_subscore"]


def test_pooled_optimizer_freezes_candidate_weights_into_inference_config(tmp_path: Path):
    root = tmp_path / "candidates"
    _candidate_fixture(root)
    output_config = tmp_path / "ensemble_config.json"
    output_report = tmp_path / "ensemble_report.json"

    report = optimize_run(
        root,
        output_config=output_config,
        output_report=output_report,
        alpha_steps=4,
        threshold_candidates=16,
        threshold_passes=2,
        landcover_passes=1,
    )

    config = load_model_config(output_config)
    assert output_report.exists()
    assert report["weights"].get("RBR_Z", 0.0) > 0
    assert config.bs.score_weights.get("RBR_Z", 0.0) > 0
    assert np.isclose(sum(config.bs.score_weights.values()), 1.0)
