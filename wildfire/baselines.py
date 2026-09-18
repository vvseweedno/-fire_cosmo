"""Fast, auditable baselines derived from the case physics."""

from __future__ import annotations

import numpy as np

from wildfire.af_candidates import active_fire_score_candidates, fuse_af_candidate_scores
from wildfire.bs_candidates import burn_score_candidates, fuse_candidate_scores
from wildfire.constants import (
    LC_CROP,
    LC_GRASS,
    LC_MANGROVE,
    LC_MOSS,
    LC_SHRUB,
    LC_TREE,
    LC_WETLAND,
)
from wildfire.model_config import ModelConfig


def active_fire_score(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous AF score from the frozen metric-gated candidate ensemble."""
    resolved = config or ModelConfig()
    candidates, valid = active_fire_score_candidates(channels, resolved)
    score = fuse_af_candidate_scores(candidates, resolved.af.score_weights, valid)
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
    """Continuous BS score from the frozen metric-gated candidate ensemble."""
    resolved = config or ModelConfig()
    candidates, valid = burn_score_candidates(channels, resolved)
    score = fuse_candidate_scores(candidates, resolved.bs.score_weights, valid)
    return score, valid


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
