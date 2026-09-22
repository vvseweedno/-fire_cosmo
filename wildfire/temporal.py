"""Temporal priors for optional active-fire false-positive suppression."""

from __future__ import annotations

import numpy as np


def persistent_heat_prior(
    detections: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    min_valid_observations: int = 3,
) -> np.ndarray:
    """Estimate per-pixel thermal-detection recurrence on an aligned grid.

    Parameters
    ----------
    detections:
        Boolean-like array with shape (time, height, width). These are historical
        thermal anomaly detections, not labels.
    valid:
        Optional boolean array with the same shape. Invalid/cloud/missing
        observations do not enter the recurrence denominator.
    min_valid_observations:
        Pixels with fewer valid historical observations receive prior=0 so a
        short history cannot become a strong static-source penalty.

    The caller is responsible for supplying observations that are genuinely
    co-registered to the same grid. No temporal/spatial relation is guessed.
    """

    history = np.asarray(detections)
    if history.ndim != 3:
        raise ValueError("detections must have shape (time, height, width)")
    if history.shape[0] < 1:
        raise ValueError("detections must contain at least one observation")
    if min_valid_observations < 1:
        raise ValueError("min_valid_observations must be positive")

    observed = history.astype(bool)
    if valid is None:
        valid_mask = np.ones(history.shape, dtype=bool)
    else:
        valid_mask = np.asarray(valid, dtype=bool)
        if valid_mask.shape != history.shape:
            raise ValueError("valid history shape differs from detections")

    valid_count = np.sum(valid_mask, axis=0, dtype=np.int32)
    hot_count = np.sum(observed & valid_mask, axis=0, dtype=np.int32)

    recurrence = np.zeros(history.shape[1:], dtype=np.float32)
    eligible = valid_count >= int(min_valid_observations)
    np.divide(
        hot_count,
        valid_count,
        out=recurrence,
        where=eligible,
        casting="unsafe",
    )
    return np.clip(recurrence, 0.0, 1.0).astype(np.float32, copy=False)
