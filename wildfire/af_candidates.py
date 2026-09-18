"""Deterministic active-fire score candidates for metric-gated ensembling.

BASE reproduces the current production score exactly.  Additional candidates
are label-free VIIRS physics/context features.  The OOF optimizer may select
them only when they improve cross-fitted AF F1; BASE always remains available.
"""

from __future__ import annotations

import numpy as np

from wildfire.features import (\n    active_fire_physics_features,\n    local_mean_3x3,\n    robust_z,\n)
from wildfire.model_config import ModelConfig


AF_CANDIDATE_NAMES: tuple[str, ...] = (
    "BASE",
    "I4_Z",
    "I45_Z",
    "I45_NORMALIZED",
    "I4_ANOMALY_3",
    "I4_ANOMALY_5",
    "I4_ANOMALY_9",
    "I45_ANOMALY_5",
)


def _valid_mask(
    channels: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if "VALID_MASK" in channels:
        valid &= np.asarray(channels["VALID_MASK"]) > 0
    return valid


def _base_score(
    channels: dict[str, np.ndarray],
    config: ModelConfig,
    valid: np.ndarray,
) -> np.ndarray:
    cfg = config.af
    i4 = np.asarray(channels["I4"], dtype=np.float32)
    i5 = np.asarray(channels["I5"], dtype=np.float32)
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

    return np.where(valid & np.isfinite(score), score, 0.0).astype(
        np.float32,
        copy=False,
    )


def active_fire_score_candidates(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    resolved = config or ModelConfig()
    i4 = np.asarray(channels["I4"], dtype=np.float32)
    i5 = np.asarray(channels["I5"], dtype=np.float32)
    if i4.shape != i5.shape:
        raise ValueError("I4 and I5 shapes differ")

    valid = _valid_mask(channels, i4.shape)
    valid &= np.isfinite(i4) & np.isfinite(i5)

    features = active_fire_physics_features(channels, valid)
    candidates: dict[str, np.ndarray] = {
        "BASE": _base_score(channels, resolved, valid),
    }

    feature_map = {
        "I4_Z": "I4_Z",
        "I45_Z": "I45_Z",
        "I45_NORMALIZED": "I45_NORMALIZED",
        "I4_ANOMALY_3": "I4_ANOMALY_3",
        "I4_ANOMALY_5": "I4_ANOMALY_5",
        "I4_ANOMALY_9": "I4_ANOMALY_9",
        "I45_ANOMALY_5": "I45_ANOMALY_5",
    }
    for candidate_name, feature_name in feature_map.items():
        score = np.asarray(features[feature_name], dtype=np.float32)
        candidates[candidate_name] = np.where(
            valid & np.isfinite(score),
            score,
            0.0,
        ).astype(np.float32, copy=False)

    return candidates, valid


def fuse_af_candidate_scores(
    candidates: dict[str, np.ndarray],
    weights: dict[str, float],
    valid: np.ndarray,
) -> np.ndarray:
    if not weights:
        raise ValueError("AF score_weights must not be empty")

    normalized: dict[str, float] = {}
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip().upper()
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("AF score weights must be non-negative")
        if weight > 0:
            normalized[name] = normalized.get(name, 0.0) + weight

    total = float(sum(normalized.values()))
    if total <= 0:
        raise ValueError("AF score weights must contain a positive weight")

    missing = sorted(set(normalized) - set(candidates))
    if missing:
        raise ValueError(f"unknown AF score candidates in config: {missing}")

    shape = next(iter(candidates.values())).shape
    score = np.zeros(shape, dtype=np.float32)
    for name, weight in normalized.items():
        candidate = np.asarray(candidates[name], dtype=np.float32)
        if candidate.shape != shape:
            raise ValueError("AF score candidate shapes differ")
        score += float(weight / total) * candidate

    mask = np.asarray(valid, dtype=bool)
    if mask.shape != shape:
        raise ValueError("AF candidate valid mask shape differs")
    return np.where(mask & np.isfinite(score), score, -np.inf).astype(
        np.float32,
        copy=False,
    )
