import numpy as np

from wildfire.fusion import (
    BurnFusionComponents,
    burn_fusion_components,
    fuse_burn_score,
)


def test_zero_cloud_weight_reproduces_masked_cloud_behavior():
    components = BurnFusionComponents(
        optical_score=np.array([[0.5, 0.9]], dtype=np.float32),
        sar_score=np.array([[0.0, 2.0]], dtype=np.float32),
        optical_valid=np.array([[1, 0]], dtype=bool),
        base_valid=np.array([[1, 1]], dtype=bool),
        sar_available=np.array([[1, 1]], dtype=bool),
    )
    score, valid = fuse_burn_score(
        components,
        clear_sar_weight=0.0,
        cloud_sar_weight=0.0,
        sar_clip=3.0,
    )
    assert valid.tolist() == [[True, False]]
    assert np.isclose(score[0, 0], 0.5)
    assert np.isneginf(score[0, 1])


def test_nonzero_cloud_weight_enables_only_sar_available_pixels():
    components = BurnFusionComponents(
        optical_score=np.zeros((1, 2), dtype=np.float32),
        sar_score=np.array([[2.0, 2.0]], dtype=np.float32),
        optical_valid=np.zeros((1, 2), dtype=bool),
        base_valid=np.ones((1, 2), dtype=bool),
        sar_available=np.array([[1, 0]], dtype=bool),
    )
    score, valid = fuse_burn_score(
        components,
        clear_sar_weight=0.0,
        cloud_sar_weight=0.25,
        sar_clip=3.0,
    )
    assert valid.tolist() == [[True, False]]
    assert np.isclose(score[0, 0], 0.5)
    assert np.isneginf(score[0, 1])


def test_scl_cloud_can_fall_back_to_sar():
    shape = (3, 3)
    channels = {
        "B8A_PRE": np.full(shape, 0.7, dtype=np.float32),
        "B12_PRE": np.full(shape, 0.2, dtype=np.float32),
        "B8A_POST": np.full(shape, 0.3, dtype=np.float32),
        "B12_POST": np.full(shape, 0.5, dtype=np.float32),
        "SCL_PRE": np.full(shape, 4, dtype=np.uint8),
        "SCL_POST": np.full(shape, 4, dtype=np.uint8),
        "VH_PRE": np.full(shape, -12.0, dtype=np.float32),
        "VH_POST": np.full(shape, -12.0, dtype=np.float32),
    }
    channels["SCL_POST"][1, 1] = 9
    channels["VH_PRE"][1, 1] = -6.0

    components = burn_fusion_components(channels)
    assert not components.optical_valid[1, 1]
    assert components.sar_available[1, 1]
    assert components.sar_score[1, 1] > 0
