"""Geospatial area calculations with explicit CRS safety."""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.crs import CRS


def projected_pixel_area_m2(
    transform: Affine,
    crs: CRS | str,
) -> float:
    """Return pixel area in square metres for a metric projected CRS."""
    resolved = CRS.from_user_input(crs)
    if resolved.is_geographic:
        raise ValueError(
            "geographic CRS requires geodesic area calculation; "
            "degree-sized pixels cannot be treated as metres"
        )
    units = (resolved.linear_units or "").lower()
    if units not in {"metre", "meter", "metres", "meters", "m"}:
        raise ValueError(f"unsupported projected CRS linear units: {units!r}")
    area = abs(transform.a * transform.e - transform.b * transform.d)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("pixel transform has non-positive/invalid area")
    return float(area)


def _binary_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    """Return boolean data only from explicit finite binary raster values."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D raster")
    if not (
        np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must contain boolean or numeric binary values")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must not contain NaN or infinite values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary 0/1 values")
    return array.astype(bool, copy=False)


def burned_area_hectares(
    burned_mask: np.ndarray,
    *,
    transform: Affine,
    crs: CRS | str,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Calculate burned area only when pixel area and masks are grounded."""
    burned = _binary_mask(burned_mask, name="burned_mask")
    if valid_mask is not None:
        valid = _binary_mask(valid_mask, name="valid_mask")
        if valid.shape != burned.shape:
            raise ValueError("valid_mask shape differs from burned_mask")
        burned = burned & valid
    pixel_area = projected_pixel_area_m2(transform, crs)
    burned_pixels = int(np.count_nonzero(burned))
    return burned_pixels * pixel_area / 10_000.0
