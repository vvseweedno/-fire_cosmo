"""Cloud-aware optical/SAR fusion primitives for burn severity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wildfire.constants import (
    LC_BUILT,
    LC_SNOW,
    LC_WATER,
    OPTICAL_BASELINE_SCL,
)
from wildfire.features import dnbr, robust_z


@dataclass(frozen=True)
class BurnFusionComponents:
    optical_score: np.ndarray
    sar_score: np.ndarray
    optical_valid: np.ndarray
    base_valid: np.ndarray
    sar_available: np.ndarray

    def __post_init__(self) -> None:
        shape = self.optical_score.shape
        for value in (
            self.sar_score,
            self.optical_valid,
            self.base_valid,
            self.sar_available,
        ):
            if value.shape != shape:
                raise ValueError("burn fusion component shapes differ")


def _base_valid_mask(
    channels: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if "VALID_MASK" in channels:
        valid &= np.asarray(channels["VALID_MASK"]) > 0
    if "LANDCOVER" in channels:
        landcover = np.asarray(channels["LANDCOVER"])
        valid &= ~np.isin(landcover, [LC_WATER, LC_SNOW, LC_BUILT])
    return valid


def _optical_valid_mask(
    channels: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    for key in ("SCL_PRE", "SCL_POST"):
        if key in channels:
            valid &= np.isin(np.asarray(channels[key]), list(OPTICAL_BASELINE_SCL))
    return valid


def _sar_evidence(
    channels: dict[str, np.ndarray],
    reference_valid: np.ndarray,
    base_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a signed SAR change score and per-pixel availability.

    VH is preferred because it reproduces the previous deterministic baseline.
    VV is used only when VH is absent. The sign is deliberately not assumed to
    be universally correct: OOF calibration can choose positive, negative or
    zero fallback weight.
    """
    pair: tuple[str, str] | None = None
    if {"VH_PRE", "VH_POST"} <= set(channels):
        pair = ("VH_PRE", "VH_POST")
    elif {"VV_PRE", "VV_POST"} <= set(channels):
        pair = ("VV_PRE", "VV_POST")

    if pair is None:
        return (
            np.zeros(reference_valid.shape, dtype=np.float32),
            np.zeros(reference_valid.shape, dtype=bool),
        )

    pre = np.asarray(channels[pair[0]], dtype=np.float32)
    post = np.asarray(channels[pair[1]], dtype=np.float32)
    if pre.shape != reference_valid.shape or post.shape != reference_valid.shape:
        raise ValueError("SAR and optical shapes differ")

    available = base_valid & np.isfinite(pre) & np.isfinite(post)
    robust_reference = reference_valid & available
    if not np.any(robust_reference):
        robust_reference = available

    score = robust_z(pre - post, robust_reference)
    score = np.where(available & np.isfinite(score), score, 0.0).astype(
        np.float32,
        copy=False,
    )
    return score, available


def burn_fusion_components(
    channels: dict[str, np.ndarray],
) -> BurnFusionComponents:
    optical = dnbr(
        channels["B8A_PRE"],
        channels["B12_PRE"],
        channels["B8A_POST"],
        channels["B12_POST"],
    ).astype(np.float32, copy=False)

    base_valid = _base_valid_mask(channels, optical.shape)
    optical_valid = base_valid & _optical_valid_mask(channels, optical.shape)
    sar_score, sar_available = _sar_evidence(
        channels,
        optical_valid,
        base_valid,
    )
    return BurnFusionComponents(
        optical_score=optical,
        sar_score=sar_score,
        optical_valid=optical_valid,
        base_valid=base_valid,
        sar_available=sar_available,
    )


def fuse_burn_score(
    components: BurnFusionComponents,
    *,
    clear_sar_weight: float,
    cloud_sar_weight: float,
    sar_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse dNBR with SAR while preserving the old baseline as a special case.

    cloud_sar_weight == 0 means cloudy/shadowed optical pixels remain masked,
    exactly matching the previous behavior. Non-zero values enable a pure-SAR
    fallback only where SAR is independently available.
    """
    if sar_clip <= 0:
        raise ValueError("sar_clip must be positive")

    clipped_sar = np.clip(
        components.sar_score,
        -float(sar_clip),
        float(sar_clip),
    )
    score = np.zeros_like(components.optical_score, dtype=np.float32)

    clear = components.optical_valid
    score[clear] = (
        components.optical_score[clear]
        + float(clear_sar_weight) * clipped_sar[clear]
    )

    cloud_fallback = (
        components.base_valid
        & ~components.optical_valid
        & components.sar_available
        & (abs(float(cloud_sar_weight)) > 1e-12)
    )
    score[cloud_fallback] = float(cloud_sar_weight) * clipped_sar[cloud_fallback]

    valid = clear | cloud_fallback
    score = np.where(valid & np.isfinite(score), score, -np.inf)
    return score.astype(np.float32, copy=False), valid
