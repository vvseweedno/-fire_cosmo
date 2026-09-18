"""Fast deterministic baselines derived from the case physics.

They are intentionally simple and auditable. Their purpose is to produce a valid first
submission and a measurable baseline immediately after the official dataset arrives.
"""

from __future__ import annotations

import numpy as np

from wildfire.constants import (
    INVALID_SCL,
    LC_BARE,
    LC_BUILT,
    LC_CROP,
    LC_GRASS,
    LC_MANGROVE,
    LC_MOSS,
    LC_SHRUB,
    LC_SNOW,
    LC_TREE,
    LC_WATER,
    LC_WETLAND,
)
from wildfire.features import dnbr, local_mean_3x3, robust_z


def _valid_mask(channels: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if "VALID_MASK" in channels:
        valid &= np.asarray(channels["VALID_MASK"]) > 0
    return valid


def predict_active_fire(
    channels: dict[str, np.ndarray],
    threshold: float = 4.0,
) -> np.ndarray:
    """VIIRS active-fire baseline using MIR/TIR contrast and spatial anomaly."""
    i4 = np.asarray(channels["I4"], dtype=np.float32)
    i5 = np.asarray(channels["I5"], dtype=np.float32)
    if i4.shape != i5.shape:
        raise ValueError("I4 and I5 shapes differ")

    valid = _valid_mask(channels, i4.shape)
    z4 = robust_z(i4, valid)
    z5 = robust_z(i5, valid)
    local_anomaly = z4 - local_mean_3x3(z4)

    score = 1.10 * z4 - 0.30 * z5 + 0.85 * local_anomaly
    if "I3" in channels:
        # I3 is useful as a conservative sunglint/context penalty.
        score -= 0.10 * np.maximum(robust_z(channels["I3"], valid), 0.0)

    if "LANDCOVER" in channels:
        lc = np.asarray(channels["LANDCOVER"])
        score = score.copy()
        score[np.isin(lc, [LC_WATER, LC_SNOW])] -= 4.0
        score[lc == LC_BUILT] -= 2.0
        score[lc == LC_BARE] -= 0.5

    mask = (score >= threshold) & valid & np.isfinite(score)
    return mask.astype(np.uint8)


def _threshold_arrays(
    landcover: np.ndarray | None,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.full(shape, 0.10, dtype=np.float32)
    moderate = np.full(shape, 0.27, dtype=np.float32)
    high = np.full(shape, 0.44, dtype=np.float32)
    if landcover is None:
        return low, moderate, high

    lc = np.asarray(landcover)
    natural_open = np.isin(lc, [LC_SHRUB, LC_GRASS, LC_WETLAND, LC_MANGROVE, LC_MOSS])
    low[natural_open], moderate[natural_open], high[natural_open] = 0.08, 0.20, 0.35

    crop = lc == LC_CROP
    low[crop], moderate[crop], high[crop] = 0.18, 0.32, 0.48

    forest = lc == LC_TREE
    low[forest], moderate[forest], high[forest] = 0.10, 0.27, 0.44
    return low, moderate, high


def predict_burn_severity(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Sentinel-2 dNBR baseline with SCL and land-cover-aware severity thresholds."""
    d = dnbr(
        channels["B8A_PRE"],
        channels["B12_PRE"],
        channels["B8A_POST"],
        channels["B12_POST"],
    )
    valid = _valid_mask(channels, d.shape)
    for key in ("SCL_PRE", "SCL_POST"):
        if key in channels:
            valid &= ~np.isin(np.asarray(channels[key]), list(INVALID_SCL))

    landcover = channels.get("LANDCOVER")
    low, moderate, high = _threshold_arrays(landcover, d.shape)

    # Optional SAR support: strong post-fire VH decrease can slightly lower the
    # burn threshold, but never creates a burn without optical evidence.
    adjusted = d.copy()
    if "VH_PRE" in channels and "VH_POST" in channels:
        sar_delta = robust_z(
            np.asarray(channels["VH_PRE"], dtype=np.float32)
            - np.asarray(channels["VH_POST"], dtype=np.float32),
            valid,
        )
        adjusted += 0.025 * np.clip(sar_delta, 0.0, 3.0)

    result = np.zeros(d.shape, dtype=np.uint8)
    result[(adjusted >= low) & valid] = 1
    result[(adjusted >= moderate) & valid] = 2
    result[(adjusted >= high) & valid] = 3

    if landcover is not None:
        lc = np.asarray(landcover)
        result[np.isin(lc, [LC_WATER, LC_SNOW, LC_BUILT])] = 0
    return result


def predict(channels: dict[str, np.ndarray], task: str) -> np.ndarray:
    task_upper = task.upper()
    if task_upper == "AF":
        return predict_active_fire(channels)
    if task_upper == "BS":
        return predict_burn_severity(channels)
    raise ValueError(f"Unknown task: {task}")
