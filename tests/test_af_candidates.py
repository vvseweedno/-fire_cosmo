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


def test_hard_negative_context_rewards_supported_thermal_structure():
    size = 13
    base = (
        300.0
        + np.arange(size, dtype=np.float32)[:, None] * 0.08
        + np.arange(size, dtype=np.float32)[None, :] * 0.03
    )
    i4 = base.copy()

    # Same peak temperature in two contexts: one isolated pixel and one compact
    # 3x3 thermal structure. The contextual candidate should prefer the latter.
    i4[2, 2] = 350.0
    i4[8:11, 8:11] = 350.0

    channels = {
        "I3": np.full((size, size), 20.0, dtype=np.float32),
        "I4": i4,
        "I5": np.full((size, size), 295.0, dtype=np.float32),
        "LANDCOVER": np.full((size, size), 10, dtype=np.int16),
        "VALID_MASK": np.ones((size, size), dtype=np.uint8),
    }

    candidates, valid = active_fire_score_candidates(channels, ModelConfig())
    contextual = candidates["HARD_NEG_CONTEXT"]

    assert valid[2, 2]
    assert valid[9, 9]
    assert contextual[9, 9] > contextual[2, 2]


def test_hard_negative_context_penalizes_known_landcover_false_positive_contexts():
    channels = _channels()
    candidates, valid = active_fire_score_candidates(channels, ModelConfig())
    contextual = candidates["HARD_NEG_CONTEXT"]

    # Water/snow context is explicitly penalized; the candidate remains finite
    # and legal for cross-fitted selection rather than becoming a hard rule.
    assert np.isfinite(contextual[valid]).all()
    assert contextual[0, 0] < candidates["I45_Z"][0, 0]


def test_persistent_context_candidate_is_neutral_without_explicit_prior():
    candidates, valid = active_fire_score_candidates(_channels(), ModelConfig())

    assert np.allclose(
        candidates["HARD_NEG_CONTEXT_PERSISTENT"][valid],
        candidates["HARD_NEG_CONTEXT"][valid],
    )


def test_persistent_context_candidate_soft_penalizes_recurrent_heat():
    channels = _channels()
    prior = np.zeros_like(channels["I4"], dtype=np.float32)
    prior[2, 2] = 1.0
    channels["PERSISTENT_HEAT_PRIOR"] = prior

    candidates, valid = active_fire_score_candidates(channels, ModelConfig())

    assert valid[2, 2]
    assert (
        candidates["HARD_NEG_CONTEXT_PERSISTENT"][2, 2]
        < candidates["HARD_NEG_CONTEXT"][2, 2]
    )
    assert np.isclose(
        candidates["HARD_NEG_CONTEXT"][2, 2]
        - candidates["HARD_NEG_CONTEXT_PERSISTENT"][2, 2],
        1.5,
    )
