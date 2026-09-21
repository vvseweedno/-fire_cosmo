import numpy as np
import pytest
from affine import Affine

from wildfire.area import burned_area_hectares, projected_pixel_area_m2


def test_projected_pixel_area_uses_affine_determinant():
    transform = Affine(20.0, 0.0, 100.0, 0.0, -20.0, 200.0)

    assert projected_pixel_area_m2(transform, "EPSG:32637") == 400.0


def test_burned_area_hectares_counts_only_burned_valid_pixels():
    burned = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    valid = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)

    area = burned_area_hectares(
        burned,
        transform=transform,
        crs="EPSG:32637",
        valid_mask=valid,
    )

    assert area == pytest.approx(0.08)


def test_burned_area_hectares_rejects_geographic_degree_grid():
    burned = np.ones((2, 2), dtype=np.uint8)
    transform = Affine(0.0001, 0.0, 38.0, 0.0, -0.0001, 47.0)

    with pytest.raises(ValueError, match="geodesic area"):
        burned_area_hectares(
            burned,
            transform=transform,
            crs="EPSG:4326",
        )


def test_burned_area_hectares_rejects_web_mercator_map_area():
    burned = np.ones((2, 2), dtype=np.uint8)
    transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0)

    with pytest.raises(ValueError, match="Web Mercator"):
        burned_area_hectares(
            burned,
            transform=transform,
            crs="EPSG:3857",
        )


def test_burned_area_hectares_rejects_shape_mismatch():
    burned = np.ones((2, 2), dtype=np.uint8)
    valid = np.ones((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape differs"):
        burned_area_hectares(
            burned,
            transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            crs="EPSG:32637",
            valid_mask=valid,
        )


def test_burned_area_hectares_rejects_empty_mask():
    burned = np.empty((0, 0), dtype=np.uint8)

    with pytest.raises(ValueError, match="burned_mask must not be empty"):
        burned_area_hectares(
            burned,
            transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            crs="EPSG:32637",
        )


def test_burned_area_hectares_rejects_valid_mask_without_observations():
    burned = np.zeros((2, 2), dtype=np.uint8)
    valid = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="no valid observation pixels"):
        burned_area_hectares(
            burned,
            transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            crs="EPSG:32637",
            valid_mask=valid,
        )


@pytest.mark.parametrize(
    "bad_value, error",
    [
        (np.nan, "non-finite"),
        (np.inf, "non-finite"),
        (-1.0, "only 0/1"),
        (2.0, "only 0/1"),
    ],
)
def test_burned_area_hectares_rejects_invalid_burned_values(bad_value, error):
    burned = np.array([[1.0, bad_value], [0.0, 1.0]])

    with pytest.raises(ValueError, match=error):
        burned_area_hectares(
            burned,
            transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            crs="EPSG:32637",
        )


def test_burned_area_hectares_rejects_invalid_valid_mask_values():
    burned = np.ones((2, 2), dtype=np.uint8)
    valid = np.array([[1.0, np.nan], [1.0, 1.0]])

    with pytest.raises(ValueError, match="valid_mask contains non-finite"):
        burned_area_hectares(
            burned,
            transform=Affine(20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
            crs="EPSG:32637",
            valid_mask=valid,
        )
