"""Numerically safe domain features reused by baselines and future ML pipelines."""

from __future__ import annotations

import numpy as np


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-6)
    return out


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    nir_f = np.asarray(nir, dtype=np.float32)
    swir_f = np.asarray(swir2, dtype=np.float32)
    return safe_ratio(nir_f - swir_f, nir_f + swir_f)


def dnbr(
    nir_pre: np.ndarray,
    swir_pre: np.ndarray,
    nir_post: np.ndarray,
    swir_post: np.ndarray,
) -> np.ndarray:
    return nbr(nir_pre, swir_pre) - nbr(nir_post, swir_post)


def robust_z(array: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(x) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(x))
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)
    median = float(np.median(x[mask]))
    mad = float(np.median(np.abs(x[mask] - median)))
    scale = max(1.4826 * mad, 1e-6)
    return (x - median) / scale


def local_mean_3x3(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    padded = np.pad(x, 1, mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return out / 9.0
