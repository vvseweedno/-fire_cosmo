import numpy as np

from wildfire.features import (
    active_fire_physics_features,
    bais2,
    burn_physics_features,
    nbr,
    rbr,
)


def test_rbr_matches_definition():
    pre = np.array([[0.6]], dtype=np.float32)
    post = np.array([[0.2]], dtype=np.float32)
    expected = 0.4 / 1.601
    assert np.allclose(rbr(pre, post), expected)


def test_bais2_is_finite_for_reflectance_inputs():
    shape = (2, 2)
    result = bais2(
        np.full(shape, 0.12, dtype=np.float32),
        np.full(shape, 0.20, dtype=np.float32),
        np.full(shape, 0.25, dtype=np.float32),
        np.full(shape, 0.35, dtype=np.float32),
        np.full(shape, 0.30, dtype=np.float32),
    )
    assert result.shape == shape
    assert np.isfinite(result).all()


def test_burn_physics_features_include_multitemporal_deltas():
    shape = (2, 2)
    channels = {
        "B4_PRE": np.full(shape, 0.20, dtype=np.float32),
        "B6_PRE": np.full(shape, 0.25, dtype=np.float32),
        "B7_PRE": np.full(shape, 0.30, dtype=np.float32),
        "B8A_PRE": np.full(shape, 0.70, dtype=np.float32),
        "B11_PRE": np.full(shape, 0.25, dtype=np.float32),
        "B12_PRE": np.full(shape, 0.20, dtype=np.float32),
        "B4_POST": np.full(shape, 0.15, dtype=np.float32),
        "B6_POST": np.full(shape, 0.15, dtype=np.float32),
        "B7_POST": np.full(shape, 0.16, dtype=np.float32),
        "B8A_POST": np.full(shape, 0.30, dtype=np.float32),
        "B11_POST": np.full(shape, 0.40, dtype=np.float32),
        "B12_POST": np.full(shape, 0.50, dtype=np.float32),
    }
    features = burn_physics_features(channels)
    assert {"DNBR", "RBR", "RDNBR", "DNDVI", "DNDMI", "DBAIS2"} <= set(features)
    assert np.all(features["DNBR"] > 0)


def test_active_fire_features_capture_local_hotspot():
    i4 = np.full((9, 9), 300.0, dtype=np.float32)
    i5 = np.full((9, 9), 295.0, dtype=np.float32)
    i4[4, 4] = 360.0
    features = active_fire_physics_features({"I4": i4, "I5": i5})
    assert features["I4_ANOMALY_5"][4, 4] > 0
    assert features["I4_MINUS_I5"][4, 4] > features["I4_MINUS_I5"][0, 0]
    assert np.allclose(nbr(np.ones((1, 1)), np.ones((1, 1))), 0.0)
