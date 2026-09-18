import numpy as np

from wildfire.af_candidates import (
    AF_CANDIDATE_NAMES,
    active_fire_score_candidates,
    fuse_af_candidate_scores,
)
from wildfire.baselines import active_fire_score
from wildfire.features import local_mean_3x3, robust_z
from wildfire.model_config import AFConfig, ModelConfig


def _channels() -> dict[str, np.ndarray]:
    i4 = np.array(
        [
            [300, 301, 302, 303, 304],
            [300, 301, 340, 303, 304],
            [300, 301, 380, 303, 304],
            [300, 301, 350, 303, 304],
            [300, 301, 302, 303, 304],
        ],
        dtype=np.float32,
    )
    i5 = np.full(i4.shape, 295.0, dtype=np.float32)
    i3 = np.full(i4.shape, 20.0, dtype=np.float32)
    landcover = np.full(i4.shape, 10, dtype=np.int16)
    landcover[0, 0] = 80
    return {
        "I3": i3,
        "I4": i4,
        "I5": i5,
        "LANDCOVER": landcover,
        "VALID_MASK": np.ones(i4.shape, dtype=np.uint8),
    }


def test_default_af_candidate_is_exact_previous_formula():
    channels = _channels()
    config = ModelConfig()
    valid = np.asarray(channels["VALID_MASK"]) > 0
    z4 = robust_z(channels["I4"], valid)
    z5 = robust_z(channels["I5"], valid)
    expected = (
        config.af.z4_weight * z4
        + config.af.z5_weight * z5
        + config.af.local_anomaly_weight * (z4 - local_mean_3x3(z4))
    )
    expected -= config.af.i3_sunglint_penalty * np.maximum(
        robust_z(channels["I3"], valid),
        0.0,
    )
    expected = expected.copy()
    expected[channels["LANDCOVER"] == 80] -= config.af.water_snow_penalty

    actual, actual_valid = active_fire_score(channels, config)
    assert np.array_equal(actual_valid, valid)
    assert np.allclose(actual[valid], expected[valid])


def test_af_candidates_are_aligned_and_finite_inside_valid():
    candidates, valid = active_fire_score_candidates(_channels(), ModelConfig())

    assert tuple(candidates) == AF_CANDIDATE_NAMES
    assert all(score.shape == valid.shape for score in candidates.values())
    for score in candidates.values():
        assert np.all(np.isfinite(score[valid]))


def test_frozen_af_weights_are_applied_as_convex_blend():
    candidates, valid = active_fire_score_candidates(_channels(), ModelConfig())
    weights = {"BASE": 0.25, "I45_Z": 0.75}

    expected = 0.25 * candidates["BASE"] + 0.75 * candidates["I45_Z"]
    actual = fuse_af_candidate_scores(candidates, weights, valid)

    assert np.allclose(actual[valid], expected[valid])


def test_model_config_can_deploy_non_baseline_af_weights():
    config = ModelConfig(
        af=AFConfig(score_weights={"BASE": 0.4, "I45_Z": 0.6})
    )
    score, valid = active_fire_score(_channels(), config)
    baseline, _ = active_fire_score(_channels(), ModelConfig())

    assert np.all(np.isfinite(score[valid]))
    assert not np.allclose(score[valid], baseline[valid])
