import numpy as np
import pytest
from affine import Affine

from wildfire.area import burned_area_hectares


_TRANSFORM = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_burned_area_rejects_non_finite_burn_mask_values(bad_value):
    burned = np.array([[1.0, bad_value], [0.0, 1.0]])
    with pytest.raises(ValueError, match="NaN or infinite"):
        burned_area_hectares(burned, transform=_TRANSFORM, crs="EPSG:32637")


def test_burned_area_rejects_non_binary_burn_mask_values():
    burned = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="only binary 0/1"):
        burned_area_hectares(burned, transform=_TRANSFORM, crs="EPSG:32637")


def test_burned_area_rejects_non_finite_valid_mask_values():
    burned = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    valid = np.array([[1.0, np.nan], [1.0, 1.0]])
    with pytest.raises(ValueError, match="valid_mask must not contain NaN or infinite"):
        burned_area_hectares(burned, transform=_TRANSFORM, crs="EPSG:32637", valid_mask=valid)


def test_binary_uint8_masks_remain_supported():
    burned = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    valid = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    area = burned_area_hectares(burned, transform=_TRANSFORM, crs="EPSG:32637", valid_mask=valid)
    assert area == pytest.approx(0.08)
