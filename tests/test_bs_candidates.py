import numpy as np

from wildfire.baselines import burn_severity_score
from wildfire.bs_candidates import (
    BS_CANDIDATE_NAMES,
    burn_score_candidates,
    fuse_candidate_scores,
)
from wildfire.fusion import burn_fusion_components, fuse_burn_score
from wildfire.model_config import BSConfig, ModelConfig


def _channels() -> dict[str, np.ndarray]:
    shape = (3, 4)
    b8a_pre = np.array(
        [
            [0.80, 0.75, 0.70, 0.65],
            [0.78, 0.72, 0.66, 0.60],
            [0.74, 0.68, 0.62, 0.56],
        ],
        dtype=np.float32,
    )
    b12_pre = np.full(shape, 0.20, dtype=np.float32)
    b8a_post = np.array(
        [
            [0.75, 0.60, 0.45, 0.30],
            [0.70, 0.55, 0.40, 0.25],
            [0.65, 0.50, 0.35, 0.20],
        ],
        dtype=np.float32,
    )
    b12_post = np.array(
        [
            [0.22, 0.28, 0.35, 0.45],
            [0.24, 0.30, 0.38, 0.48],
            [0.26, 0.32, 0.40, 0.52],
        ],
        dtype=np.float32,
    )
    b4_pre = np.full(shape, 0.15, dtype=np.float32)
    b4_post = np.linspace(0.16, 0.30, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    b11_pre = np.full(shape, 0.25, dtype=np.float32)
    b11_post = np.linspace(0.27, 0.45, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    return {
        "B8A_PRE": b8a_pre,
        "B12_PRE": b12_pre,
        "B8A_POST": b8a_post,
        "B12_POST": b12_post,
        "B4_PRE": b4_pre,
        "B4_POST": b4_post,
        "B11_PRE": b11_pre,
        "B11_POST": b11_post,
        "VALID_MASK": np.ones(shape, dtype=np.uint8),
    }


def test_default_candidate_ensemble_is_exact_previous_base_score():
    channels = _channels()
    config = ModelConfig()

    components = burn_fusion_components(channels)
    expected, expected_valid = fuse_burn_score(
        components,
        clear_sar_weight=config.bs.sar_weight,
        cloud_sar_weight=config.bs.cloud_sar_weight,
        sar_clip=config.bs.sar_clip,
    )
    actual, actual_valid = burn_severity_score(channels, config)

    assert np.array_equal(actual_valid, expected_valid)
    assert np.array_equal(np.isneginf(actual), np.isneginf(expected))
    assert np.allclose(actual[actual_valid], expected[expected_valid])


def test_all_candidate_names_are_aligned_even_when_optional_bands_are_missing():
    channels = {
        key: value
        for key, value in _channels().items()
        if key not in {"B4_PRE", "B4_POST", "B11_PRE", "B11_POST"}
    }
    candidates, valid = burn_score_candidates(channels, ModelConfig())

    assert tuple(candidates) == BS_CANDIDATE_NAMES
    assert all(score.shape == valid.shape for score in candidates.values())
    assert np.array_equal(candidates["DNDVI_Z"], candidates["BASE"])
    assert np.array_equal(candidates["DNDMI_Z"], candidates["BASE"])


def test_frozen_score_weights_are_applied_as_convex_blend():
    candidates, valid = burn_score_candidates(_channels(), ModelConfig())
    weights = {"BASE": 0.25, "RBR_Z": 0.75}

    expected = 0.25 * candidates["BASE"] + 0.75 * candidates["RBR_Z"]
    actual = fuse_candidate_scores(candidates, weights, valid)

    assert np.allclose(actual[valid], expected[valid])


def test_model_config_can_deploy_non_baseline_candidate_weights():
    config = ModelConfig(
        bs=BSConfig(score_weights={"BASE": 0.4, "RBR_Z": 0.6})
    )
    score, valid = burn_severity_score(_channels(), config)
    base, _ = burn_severity_score(_channels(), ModelConfig())

    assert np.all(np.isfinite(score[valid]))
    assert not np.allclose(score[valid], base[valid])
