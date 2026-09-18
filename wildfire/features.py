"""Numerically safe physics features for AF and burn-severity modelling."""

from __future__ import annotations

import numpy as np


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float32)
    den = np.asarray(denominator, dtype=np.float32)
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-6)
    return out


def nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio."""
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


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir_f = np.asarray(nir, dtype=np.float32)
    red_f = np.asarray(red, dtype=np.float32)
    return safe_ratio(nir_f - red_f, nir_f + red_f)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    nir_f = np.asarray(nir, dtype=np.float32)
    swir_f = np.asarray(swir1, dtype=np.float32)
    return safe_ratio(nir_f - swir_f, nir_f + swir_f)


def nbr2(swir2: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    swir2_f = np.asarray(swir2, dtype=np.float32)
    swir1_f = np.asarray(swir1, dtype=np.float32)
    return safe_ratio(swir2_f - swir1_f, swir2_f + swir1_f)


def mirbi(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    swir1_f = np.asarray(swir1, dtype=np.float32)
    swir2_f = np.asarray(swir2, dtype=np.float32)
    return (10.0 * swir2_f - 9.8 * swir1_f + 2.0).astype(np.float32)


def bais2(
    red: np.ndarray,
    red_edge2: np.ndarray,
    red_edge3: np.ndarray,
    nir_narrow: np.ndarray,
    swir2: np.ndarray,
) -> np.ndarray:
    """Burned Area Index for Sentinel-2 on BOA reflectance inputs."""
    r = np.asarray(red, dtype=np.float32)
    re2 = np.asarray(red_edge2, dtype=np.float32)
    re3 = np.asarray(red_edge3, dtype=np.float32)
    n2 = np.asarray(nir_narrow, dtype=np.float32)
    s2 = np.asarray(swir2, dtype=np.float32)

    product_ratio = safe_ratio(re2 * re3 * n2, r)
    first = 1.0 - np.sqrt(np.clip(product_ratio, 0.0, None))
    denominator = np.sqrt(np.clip(s2 + n2, 1e-6, None))
    second = safe_ratio(s2 - n2, denominator) + 1.0
    result = first * second
    return np.where(np.isfinite(result), result, 0.0).astype(np.float32)


def rbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
    """Relativized Burn Ratio."""
    pre = np.asarray(nbr_pre, dtype=np.float32)
    post = np.asarray(nbr_post, dtype=np.float32)
    return safe_ratio(pre - post, pre + 1.001)


def rdnbr(
    nbr_pre: np.ndarray,
    nbr_post: np.ndarray,
    *,
    nbr_scale: float = 1000.0,
) -> np.ndarray:
    """RdNBR represented on the conventional NBR*1000 scale."""
    pre = np.asarray(nbr_pre, dtype=np.float32)
    post = np.asarray(nbr_post, dtype=np.float32)
    delta = pre - post
    denominator = np.sqrt(np.clip(np.abs(pre), 1e-6, None))
    return (float(nbr_scale) * safe_ratio(delta, denominator)).astype(np.float32)


def robust_z(array: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(x) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(x))
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)
    median = float(np.median(x[mask]))
    mad = float(np.median(np.abs(x[mask] - median)))
    scale = max(1.4826 * mad, 1e-6)
    return ((x - median) / scale).astype(np.float32)


def local_mean(array: np.ndarray, size: int) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("local window size must be a positive odd integer")
    x = np.asarray(array, dtype=np.float32)
    radius = size // 2
    padded = np.pad(x, radius, mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return out / float(size * size)


def local_mean_3x3(array: np.ndarray) -> np.ndarray:
    return local_mean(array, 3)


def active_fire_physics_features(
    channels: dict[str, np.ndarray],
    valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """NASA-inspired contextual AF features without copying product labels."""
    i4 = np.asarray(channels["I4"], dtype=np.float32)
    i5 = np.asarray(channels["I5"], dtype=np.float32)
    if i4.shape != i5.shape:
        raise ValueError("I4 and I5 shapes differ")

    mask = np.isfinite(i4) & np.isfinite(i5)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)

    delta45 = i4 - i5
    z4 = robust_z(i4, mask)
    z5 = robust_z(i5, mask)
    zd45 = robust_z(delta45, mask)

    features = {
        "I4_Z": z4,
        "I5_Z": z5,
        "I4_MINUS_I5": delta45.astype(np.float32),
        "I45_Z": zd45,
        "I45_NORMALIZED": safe_ratio(delta45, np.abs(i4) + np.abs(i5)),
        "I4_ANOMALY_3": (z4 - local_mean(z4, 3)).astype(np.float32),
        "I4_ANOMALY_5": (z4 - local_mean(z4, 5)).astype(np.float32),
        "I4_ANOMALY_9": (z4 - local_mean(z4, 9)).astype(np.float32),
        "I45_ANOMALY_5": (zd45 - local_mean(zd45, 5)).astype(np.float32),
    }
    if "I3" in channels:
        features["I3_Z"] = robust_z(channels["I3"], mask)
    return features


def burn_physics_features(channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build multitemporal Sentinel/SAR features when source bands exist."""
    required = {"B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"}
    if not required <= set(channels):
        raise ValueError(f"missing required burn bands: {sorted(required - set(channels))}")

    pre_nbr = nbr(channels["B8A_PRE"], channels["B12_PRE"])
    post_nbr = nbr(channels["B8A_POST"], channels["B12_POST"])
    features: dict[str, np.ndarray] = {
        "NBR_PRE": pre_nbr,
        "NBR_POST": post_nbr,
        "DNBR": (pre_nbr - post_nbr).astype(np.float32),
        "RBR": rbr(pre_nbr, post_nbr),
        "RDNBR": rdnbr(pre_nbr, post_nbr),
    }

    if {"B4_PRE", "B4_POST"} <= set(channels):
        pre_ndvi = ndvi(channels["B8A_PRE"], channels["B4_PRE"])
        post_ndvi = ndvi(channels["B8A_POST"], channels["B4_POST"])
        features.update(
            {
                "NDVI_PRE": pre_ndvi,
                "NDVI_POST": post_ndvi,
                "DNDVI": (pre_ndvi - post_ndvi).astype(np.float32),
            }
        )

    if {"B11_PRE", "B11_POST"} <= set(channels):
        pre_ndmi = ndmi(channels["B8A_PRE"], channels["B11_PRE"])
        post_ndmi = ndmi(channels["B8A_POST"], channels["B11_POST"])
        pre_nbr2 = nbr2(channels["B12_PRE"], channels["B11_PRE"])
        post_nbr2 = nbr2(channels["B12_POST"], channels["B11_POST"])
        pre_mirbi = mirbi(channels["B11_PRE"], channels["B12_PRE"])
        post_mirbi = mirbi(channels["B11_POST"], channels["B12_POST"])
        features.update(
            {
                "NDMI_PRE": pre_ndmi,
                "NDMI_POST": post_ndmi,
                "DNDMI": (pre_ndmi - post_ndmi).astype(np.float32),
                "NBR2_PRE": pre_nbr2,
                "NBR2_POST": post_nbr2,
                "DNBR2": (pre_nbr2 - post_nbr2).astype(np.float32),
                "MIRBI_PRE": pre_mirbi,
                "MIRBI_POST": post_mirbi,
                "DMIRBI": (post_mirbi - pre_mirbi).astype(np.float32),
            }
        )

    bais_required = {
        "B4_PRE",
        "B6_PRE",
        "B7_PRE",
        "B4_POST",
        "B6_POST",
        "B7_POST",
    }
    if bais_required <= set(channels):
        pre_bais2 = bais2(
            channels["B4_PRE"],
            channels["B6_PRE"],
            channels["B7_PRE"],
            channels["B8A_PRE"],
            channels["B12_PRE"],
        )
        post_bais2 = bais2(
            channels["B4_POST"],
            channels["B6_POST"],
            channels["B7_POST"],
            channels["B8A_POST"],
            channels["B12_POST"],
        )
        features.update(
            {
                "BAIS2_PRE": pre_bais2,
                "BAIS2_POST": post_bais2,
                "DBAIS2": (post_bais2 - pre_bais2).astype(np.float32),
            }
        )

    for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"):
        pre_key = f"{band}_PRE"
        post_key = f"{band}_POST"
        if pre_key in channels and post_key in channels:
            features[f"D{band}"] = (
                np.asarray(channels[post_key], dtype=np.float32)
                - np.asarray(channels[pre_key], dtype=np.float32)
            )

    if {"VV_PRE", "VV_POST"} <= set(channels):
        features["DVV"] = (
            np.asarray(channels["VV_POST"], dtype=np.float32)
            - np.asarray(channels["VV_PRE"], dtype=np.float32)
        )
    if {"VH_PRE", "VH_POST"} <= set(channels):
        features["DVH"] = (
            np.asarray(channels["VH_POST"], dtype=np.float32)
            - np.asarray(channels["VH_PRE"], dtype=np.float32)
        )

    return features
