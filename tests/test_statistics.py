import numpy as np
import pytest

from wildfire.statistics import GroupedPrediction, paired_group_bootstrap


def _record(chip_id: str, group_id: str, task: str, pred, target):
    return GroupedPrediction(
        chip_id=chip_id,
        group_id=group_id,
        task=task,
        prediction=np.asarray(pred, dtype=np.uint8),
        target=np.asarray(target, dtype=np.uint8),
    )


def test_paired_group_bootstrap_prefers_better_experiment():
    experiment_a = [
        _record("af_1", "event:1", "AF", [[0, 1]], [[0, 1]]),
        _record("bs_1", "event:1", "BS", [[0, 1, 2, 3]], [[0, 1, 2, 3]]),
        _record("af_2", "event:2", "AF", [[0, 1]], [[0, 1]]),
        _record("bs_2", "event:2", "BS", [[0, 1, 2, 3]], [[0, 1, 2, 3]]),
        _record("af_3", "event:3", "AF", [[0, 1]], [[0, 1]]),
        _record("bs_3", "event:3", "BS", [[0, 1, 2, 3]], [[0, 1, 2, 3]]),
    ]
    experiment_b = [
        _record("af_1", "event:1", "AF", [[0, 0]], [[0, 1]]),
        _record("bs_1", "event:1", "BS", [[0, 0, 0, 0]], [[0, 1, 2, 3]]),
        _record("af_2", "event:2", "AF", [[0, 0]], [[0, 1]]),
        _record("bs_2", "event:2", "BS", [[0, 0, 0, 0]], [[0, 1, 2, 3]]),
        _record("af_3", "event:3", "AF", [[0, 0]], [[0, 1]]),
        _record("bs_3", "event:3", "BS", [[0, 0, 0, 0]], [[0, 1, 2, 3]]),
    ]

    report = paired_group_bootstrap(
        experiment_a,
        experiment_b,
        n_boot=300,
        seed=7,
    )
    assert report["delta_score"] > 0
    assert report["bootstrap_95_ci"][0] > 0
    assert report["interpretation"] == "supports_a"


def test_paired_group_bootstrap_rejects_target_mismatch():
    experiment_a = [
        _record("af_1", "event:1", "AF", [[1]], [[1]]),
        _record("bs_1", "event:2", "BS", [[1]], [[1]]),
    ]
    experiment_b = [
        _record("af_1", "event:1", "AF", [[1]], [[0]]),
        _record("bs_1", "event:2", "BS", [[1]], [[1]]),
    ]

    with pytest.raises(ValueError, match="target mismatch"):
        paired_group_bootstrap(experiment_a, experiment_b, n_boot=100)
