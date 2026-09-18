from dataclasses import replace

import numpy as np

from wildfire.baselines import burn_severity_score, predict_active_fire, predict_burn_severity
from wildfire.model_config import ModelConfig


def test_active_fire_finds_strong_local_i4_anomaly():
    i4 = np.full((7, 7), 300.0, dtype=np.float32)
    i5 = np.full((7, 7), 295.0, dtype=np.float32)
    i4[3, 3] = 380.0
    channels = {"I4": i4, "I5": i5, "VALID_MASK": np.ones((7, 7), dtype=np.uint8)}
    pred = predict_active_fire(channels, threshold=3.0)
    assert pred[3, 3] == 1


def test_burn_severity_masks_cloud_and_water():
    shape = (2, 3)
    pre_nir = np.full(shape, 0.7, dtype=np.float32)
    pre_swir = np.full(shape, 0.2, dtype=np.float32)
    post_nir = np.full(shape, 0.3, dtype=np.float32)
    post_swir = np.full(shape, 0.5, dtype=np.float32)
    scl = np.array([[4, 9, 4], [4, 4, 4]], dtype=np.uint8)
    lc = np.array([[10, 10, 80], [30, 40, 50]], dtype=np.uint8)

    pred = predict_burn_severity(
        {
            "B8A_PRE": pre_nir,
            "B12_PRE": pre_swir,
            "B8A_POST": post_nir,
            "B12_POST": post_swir,
            "SCL_PRE": scl,
            "SCL_POST": scl,
            "LANDCOVER": lc,
        }
    )
    assert pred[0, 0] > 0
    assert pred[0, 1] == 0
    assert pred[0, 2] == 0
    assert pred[1, 2] == 0


def test_spectral_consensus_falls_back_when_optional_bands_are_missing():
    shape = (3, 3)
    channels = {
        "B8A_PRE": np.full(shape, 0.7, dtype=np.float32),
        "B12_PRE": np.full(shape, 0.2, dtype=np.float32),
        "B8A_POST": np.full(shape, 0.3, dtype=np.float32),
        "B12_POST": np.full(shape, 0.5, dtype=np.float32),
    }
    base = ModelConfig()
    spectral = replace(
        base,
        bs=replace(
            base.bs,
            score_recipe="spectral_consensus",
            index_consensus_weight=0.1,
        ),
    )

    base_score, base_valid = burn_severity_score(channels, base)
    spectral_score, spectral_valid = burn_severity_score(channels, spectral)

    assert np.array_equal(base_valid, spectral_valid)
    assert np.allclose(base_score, spectral_score)
