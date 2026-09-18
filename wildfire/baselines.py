"""Fast, auditable baselines derived from the case physics."""

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
from wildfire.model_config import ModelConfig


def _valid_mask(channels: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if "VALID_MASK" in channels:
        valid &= np.asarray(channels["VALID_MASK"]) > 0
    return valid


def active_fire_score(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = (config or ModelConfig()).af
    i4 = np.asarray(channels["I4"], dtype=np.float32)
    i5 = np.asarray(channels["I5"], dtype=np.float32)
    if i4.shape != i5.shape:
        raise ValueError("I4 and I5 shapes differ")

    valid = _valid_mask(channels, i4.shape)
    z4 = robust_z(i4, valid)
    z5 = robust_z(i5, valid)
    local_anomaly = z4 - local_mean_3x3(z4)

    score = (
        cfg.z4_weight * z4
        + cfg.z5_weight * z5
        + cfg.local_anomaly_weight * local_anomaly
    )
    if "I3" in channels:
        score -= cfg.i3_sunglint_penalty * np.maximum(
            robust_z(channels["I3"], valid),
            0.0,
        )

    if "LANDCOVER" in channels:
        lc = np.asarray(channels["LANDCOVER"])
        score = score.copy()
        score[np.isin(lc, [LC_WATER, LC_SNOW])] -= cfg.water_snow_penalty
        score[lc == LC_BUILT] -= cfg.built_penalty
        score[lc == LC_BARE] -= cfg.bare_penalty

    score = np.where(np.isfinite(score), score, -np.inf).astype(np.float32, copy=False)
    return score, valid


def predict_active_fire(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    resolved = config or ModelConfig()
    score, valid = active_fire_score(channels, resolved)
    decision_threshold = resolved.af.threshold if threshold is None else float(threshold)
    return ((score >= decision_threshold) & valid).astype(np.uint8)


def burn_severity_score(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous BS score and the pixels where this baseline can predict."""
    resolved = config or ModelConfig()
    score = dnbr(
        channels["B8A_PRE"],
        channels["B12_PRE"],
        channels["B8A_POST"],
        channels["B12_POST"],
    )
    valid = _valid_mask(channels, score.shape)
    for key in ("SCL_PRE", "SCL_POST"):
        if key in channels:
            valid &= ~np.isin(np.asarray(channels[key]), list(INVALID_SCL))

    if "LANDCOVER" in channels:
        landcover = np.asarray(channels["LANDCOVER"])
        valid &= ~np.isin(landcover, [LC_WATER, LC_SNOW, LC_BUILT])

    adjusted = score.copy()
    if "VH_PRE" in channels and "VH_POST" in channels:
        sar_delta = robust_z(
            np.asarray(channels["VH_PRE"], dtype=np.float32)
            - np.asarray(channels["VH_POST"], dtype=np.float32),
            valid,
        )
        adjusted += resolved.bs.sar_weight * np.clip(
            sar_delta,
            0.0,
            resolved.bs.sar_clip,
        )

    adjusted = np.where(np.isfinite(adjusted), adjusted, -np.inf)
    return adjusted.astype(np.float32, copy=False), valid


def _threshold_arrays(
    landcover: np.ndarray | None,
    shape: tuple[int, int],
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = np.full(shape, config.bs.default_thresholds[0], dtype=np.float32)
    moderate = np.full(shape, config.bs.default_thresholds[1], dtype=np.float32)
    high = np.full(shape, config.bs.default_thresholds[2], dtype=np.float32)
    if landcover is None:
        return low, moderate, high

    lc = np.asarray(landcover)
    natural_open = np.isin(lc, [LC_SHRUB, LC_GRASS, LC_WETLAND, LC_MANGROVE, LC_MOSS])
    low[natural_open] = config.bs.natural_open_thresholds[0]
    moderate[natural_open] = config.bs.natural_open_thresholds[1]
    high[natural_open] = config.bs.natural_open_thresholds[2]

    crop = lc == LC_CROP
    low[crop] = config.bs.crop_thresholds[0]
    moderate[crop] = config.bs.crop_thresholds[1]
    high[crop] = config.bs.crop_thresholds[2]

    forest = lc == LC_TREE
    low[forest] = config.bs.forest_thresholds[0]
    moderate[forest] = config.bs.forest_thresholds[1]
    high[forest] = config.bs.forest_thresholds[2]
    return low, moderate, high


def predict_burn_severity(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> np.ndarray:
    resolved = config or ModelConfig()
    score, valid = burn_severity_score(channels, resolved)
    landcover = channels.get("LANDCOVER")
    low, moderate, high = _threshold_arrays(landcover, score.shape, resolved)

    result = np.zeros(score.shape, dtype=np.uint8)
    result[(score >= low) & valid] = 1
    result[(score >= moderate) & valid] = 2
    result[(score >= high) & valid] = 3
    return result


def predict(
    channels: dict[str, np.ndarray],
    task: str,
    config: ModelConfig | None = None,
) -> np.ndarray:
    task_upper = task.upper()
    if task_upper == "AF":
        return predict_active_fire(channels, config)
    if task_upper == "BS":
        return predict_burn_severity(channels, config)
    raise ValueError(f"Unknown task: {task}")
