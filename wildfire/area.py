"""Geospatial area calculations with explicit CRS safety."""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.crs import CRS


def projected_pixel_area_m2(
    transform: Affine,
    crs: CRS | str,
) -> float:
    """Return pixel area in square metres for a metric projected CRS.

    Geographic degree grids are deliberately rejected. They require geodesic
    area calculation and must never be converted to hectares by treating
    degrees as metres. Other non-projected CRSs (for example geocentric XYZ)
    are also rejected because their metre-valued coordinates do not describe
    a planar ground surface. Web Mercator is rejected because its area
    distortion varies strongly with latitude.
    """

    resolved = CRS.from_user_input(crs)
    if resolved.is_geographic:
        raise ValueError(
            "geographic CRS requires geodesic area calculation; "
            "degree-sized pixels cannot be treated as metres"
        )
    if not resolved.is_projected:
        raise ValueError(
            "burned-area hectares require a projected ground-surface CRS"
        )
    if resolved.to_epsg() == 3857:
        raise ValueError(
            "EPSG:3857 Web Mercator is not valid for burned-area hectares; "
            "reproject to an appropriate metric analysis CRS"
        )
    units = (resolved.linear_units or "").lower()
    if units not in {"metre", "meter", "metres", "meters", "m"}:
        raise ValueError(f"unsupported projected CRS linear units: {units!r}")

    # Determinant handles north-up and rotated/sheared affine transforms.
    area = abs(transform.a * transform.e - transform.b * transform.d)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("pixel transform has non-positive/invalid area")
    return float(area)


def _binary_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    """Return a boolean raster only for explicit finite binary mask values."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D raster")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain boolean or numeric 0/1 values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    return array.astype(bool, copy=False)


def burned_area_hectares(
    burned_mask: np.ndarray,
    *,
    transform: Affine,
    crs: CRS | str,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Calculate burned area only when mask and pixel area are grounded."""

    burned = _binary_mask(burned_mask, name="burned_mask")

    if valid_mask is not None:
        valid = _binary_mask(valid_mask, name="valid_mask")
        if valid.shape != burned.shape:
            raise ValueError("valid_mask shape differs from burned_mask")
        if not np.any(valid):
            raise ValueError("valid_mask contains no valid observation pixels")
        # A positive burn classification outside the observation-valid support
        # is contradictory evidence. Silently clipping it would hide an
        # upstream masking/alignment bug and could under-report hectares.
        if np.any(burned & ~valid):
            raise ValueError("burned_mask contains burn pixels outside valid_mask")

    pixel_area = projected_pixel_area_m2(transform, crs)
    burned_pixels = int(np.count_nonzero(burned))
    return burned_pixels * pixel_area / 10_000.0
