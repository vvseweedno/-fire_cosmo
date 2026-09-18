"""Fast, auditable baselines derived from the case physics."""

from __future__ import annotations

import numpy as np

from wildfire.constants import (
    LC_CROP,
    LC_GRASS,
    LC_MANGROVE,
    LC_MOSS,
    LC_SHRUB,
    LC_TREE,
    LC_WETLAND,
)
from wildfire.features import burn_physics_features, local_mean_3x3, robust_z
from wildfire.fusion import burn_fusion_components, fuse_burn_score
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
        score[np.isin(lc, [80, 70])] -= cfg.water_snow_penalty
        score[lc == 50] -= cfg.built_penalty
        score[lc == 60] -= cfg.bare_penalty

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


def _spectral_consensus(
    channels: dict[str, np.ndarray],
    optical_valid: np.ndarray,
) -> np.ndarray | None:
    """Return robust consensus evidence from independent burn-sensitive indices.

    The baseline dNBR itself is intentionally excluded so this term only changes
    pixel ranking when independent spectral evidence agrees. Missing optional
    bands simply reduce the number of voters.
    """
    features = burn_physics_features(channels)
    evidence_names = ("DNDVI", "DNDMI", "DMIRBI", "DBAIS2")
    evidence: list[np.ndarray] = []
    for name in evidence_names:
        value = features.get(name)
        if value is None:
            continue
        evidence.append(robust_z(value, optical_valid))
    if not evidence:
        return None

    stack = np.stack(evidence, axis=0)
    consensus = np.median(stack, axis=0)
    return np.where(np.isfinite(consensus), consensus, 0.0).astype(np.float32)


def burn_severity_score(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous BS score with optional OOF-promoted spectral consensus."""
    resolved = config or ModelConfig()
    components = burn_fusion_components(channels)
    score, valid = fuse_burn_score(
        components,
        clear_sar_weight=resolved.bs.sar_weight,
        cloud_sar_weight=resolved.bs.cloud_sar_weight,
        sar_clip=resolved.bs.sar_clip,
    )

    if resolved.bs.score_recipe == "dnbr_sar":
        return score, valid
    if resolved.bs.score_recipe != "spectral_consensus":
        raise ValueError(f"Unsupported BS score recipe: {resolved.bs.score_recipe}")

    weight = float(resolved.bs.index_consensus_weight)
    if weight <= 0:
        return score, valid

    consensus = _spectral_consensus(channels, components.optical_valid)
    if consensus is None:
        return score, valid

    refined = np.asarray(score, dtype=np.float32).copy()
    clear = components.optical_valid & np.isfinite(refined)
    refined[clear] += weight * consensus[clear]
    refined = np.where(valid & np.isfinite(refined), refined, -np.inf)
    return refined.astype(np.float32, copy=False), valid


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
