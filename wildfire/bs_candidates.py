"""Deterministic burn-severity score candidates for metric-gated ensembling.

Every candidate is derived only from organiser-provided input channels.  The
BASE candidate is the existing cloud/SAR-aware dNBR score exactly, so enabling
candidate search never removes the previous solution from the search space.

Optional optical candidates fall back to BASE when their required bands are
missing or optical pixels are invalid.  This gives every candidate the same
valid-mask contract and makes OOF alignment safe.
"""

from __future__ import annotations  # noqa: I001

import numpy as np

from wildfire.features import burn_physics_features, robust_z
from wildfire.fusion import burn_fusion_components, fuse_burn_score
from wildfire.model_config import ModelConfig


BS_CANDIDATE_NAMES: tuple[str, ...] = (
    "BASE",
    "RBR_Z",
    "RDNBR_Z",
    "DNDVI_Z",
    "DNDMI_Z",
    "DNBR2_Z",
    "DMIRBI_Z",
    "DBAIS2_Z",
)

_FEATURE_BY_CANDIDATE = {
    "RBR_Z": "RBR",
    "RDNBR_Z": "RDNBR",
    "DNDVI_Z": "DNDVI",
    "DNDMI_Z": "DNDMI",
    "DNBR2_Z": "DNBR2",
    "DMIRBI_Z": "DMIRBI",
    "DBAIS2_Z": "DBAIS2",
}


def burn_score_candidates(
    channels: dict[str, np.ndarray],
    config: ModelConfig | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return aligned BS score candidates and one shared valid mask."""

    resolved = config or ModelConfig()
    components = burn_fusion_components(channels)
    base, valid = fuse_burn_score(
        components,
        clear_sar_weight=resolved.bs.sar_weight,
        cloud_sar_weight=resolved.bs.cloud_sar_weight,
        sar_clip=resolved.bs.sar_clip,
    )

    # Ensemble arithmetic should never see inf outside the shared valid mask.
    # The validity mask, not sentinel score values, controls evaluation.
    base_safe = np.where(valid & np.isfinite(base), base, 0.0).astype(
        np.float32,
        copy=False,
    )
    candidates: dict[str, np.ndarray] = {"BASE": base_safe}

    features = burn_physics_features(channels)
    optical_valid = components.optical_valid & valid

    for candidate_name, feature_name in _FEATURE_BY_CANDIDATE.items():
        feature = features.get(feature_name)
        if feature is None or not np.any(optical_valid):
            candidates[candidate_name] = base_safe.copy()
            continue

        feature_score = robust_z(feature, optical_valid)
        candidate = base_safe.copy()
        candidate[optical_valid] = feature_score[optical_valid]
        candidate = np.where(
            valid & np.isfinite(candidate),
            candidate,
            0.0,
        ).astype(np.float32, copy=False)
        candidates[candidate_name] = candidate

    return candidates, valid


def fuse_candidate_scores(
    candidates: dict[str, np.ndarray],
    weights: dict[str, float],
    valid: np.ndarray,
) -> np.ndarray:
    """Apply a convex score ensemble from a frozen model configuration."""

    if not weights:
        raise ValueError("BS score_weights must not be empty")

    normalized: dict[str, float] = {}
    for raw_name, raw_weight in weights.items():
        name = str(raw_name).strip().upper()
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("BS score weights must be non-negative")
        if weight > 0:
            normalized[name] = normalized.get(name, 0.0) + weight

    total = float(sum(normalized.values()))
    if total <= 0:
        raise ValueError("BS score weights must contain a positive weight")

    missing = sorted(set(normalized) - set(candidates))
    if missing:
        raise ValueError(f"unknown BS score candidates in config: {missing}")

    shape = next(iter(candidates.values())).shape
    score = np.zeros(shape, dtype=np.float32)
    for name, weight in normalized.items():
        candidate = np.asarray(candidates[name], dtype=np.float32)
        if candidate.shape != shape:
            raise ValueError("BS score candidate shapes differ")
        score += float(weight / total) * candidate

    mask = np.asarray(valid, dtype=bool)
    if mask.shape != shape:
        raise ValueError("BS candidate valid mask shape differs")
    return np.where(mask & np.isfinite(score), score, -np.inf).astype(
        np.float32,
        copy=False,
    )
