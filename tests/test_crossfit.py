import numpy as np
import pytest

from wildfire.crossfit import crossfit_calibrate_and_evaluate
from wildfire.model_config import ModelConfig
from wildfire.oof import OOFPool, OOFRecord


def _record(chip_id: str, task: str, score, target) -> OOFRecord:
    score_array = np.asarray(score, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.uint8)
    return OOFRecord(
        chip_id=chip_id,
        task=task,
        score=score_array,
        target=target_array,
        valid=np.ones_like(target_array, dtype=bool),
    )


def test_crossfit_uses_all_holdout_chips_and_can_score_perfectly():
    pool = OOFPool()
    folds = []
    for fold in range(3):
        af_id = f"af_{fold}"
        bs_id = f"bs_{fold}"
        pool.add(_record(af_id, "AF", [[0.1, 0.9]], [[0, 1]]))
        pool.add(
            _record(
                bs_id,
                "BS",
                [[0.0, 0.2, 0.5, 0.8]],
                [[0, 1, 2, 3]],
            )
        )
        folds.append(
            {
                "fold": fold,
                "validation": [af_id, bs_id],
                "train": [],
            }
        )

    deployment, report = crossfit_calibrate_and_evaluate(
        pool,
        {"folds": folds},
        ModelConfig(),
        bs_max_candidates=24,
        bs_passes=3,
    )

    assert report["primary_validation"] == "cross_fitted_oof"
    assert report["crossfit"]["score"] == 1.0
    assert len(report["folds"]) == 3
    assert "crossfit_validation" in deployment.training


def test_crossfit_rejects_duplicate_fold_assignment():
    pool = OOFPool()
    pool.add(_record("af_1", "AF", [[0.1, 0.9]], [[0, 1]]))
    pool.add(_record("bs_1", "BS", [[0.0, 0.8]], [[0, 3]]))
    manifest = {
        "folds": [
            {"fold": 0, "validation": ["af_1", "bs_1"]},
            {"fold": 1, "validation": ["af_1"]},
        ]
    }
    with pytest.raises(ValueError, match="multiple folds"):
        crossfit_calibrate_and_evaluate(pool, manifest, ModelConfig())
