"""Deterministic metric-aware calibration before heavier ML models are trained."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from wildfire.calibration import exact_f1_threshold, optimize_ordered_thresholds
from wildfire.model_config import ModelConfig


AFTrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray]
BSTrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray]


def threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if maximum < minimum:
        raise ValueError("threshold maximum must be >= minimum")
    count = int(np.floor((maximum - minimum) / step + 1e-9)) + 1
    values = [minimum + index * step for index in range(count)]
    if values[-1] < maximum - 1e-9:
        values.append(maximum)
    return [round(float(value), 10) for value in values]


def _materialize_samples(
    samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    count = 0
    for score, target, valid in samples:
        s = np.asarray(score, dtype=np.float32)
        t = np.asarray(target)
        v = np.asarray(valid, dtype=bool)
        if s.shape != t.shape or s.shape != v.shape:
            raise ValueError("score, target and valid shapes must match")
        scores.append(s.ravel())
        targets.append(t.ravel())
        valid_masks.append(v.ravel())
        count += 1
    if not scores:
        raise ValueError("no training samples were supplied")
    return (
        np.concatenate(scores),
        np.concatenate(targets),
        np.concatenate(valid_masks),
        count,
    )


def calibrate_af_threshold_exact(
    samples: Iterable[AFTrainingSample],
    base_config: ModelConfig,
) -> tuple[ModelConfig, dict[str, float | int]]:
    scores, targets, valid, sample_count = _materialize_samples(samples)
    best = exact_f1_threshold(scores, targets, valid)

    metadata = dict(base_config.training)
    metadata["af_threshold_calibration"] = {
        "method": "exact_micro_f1_sweep_on_train_partition",
        "samples": sample_count,
        **best,
    }
    calibrated = replace(
        base_config,
        af=replace(base_config.af, threshold=float(best["threshold"])),
        training=metadata,
    )
    return calibrated, best


def calibrate_af_threshold(
    samples: Iterable[AFTrainingSample],
    thresholds: Iterable[float],
    base_config: ModelConfig,
) -> tuple[ModelConfig, list[dict[str, float | int]]]:
    candidates = [float(value) for value in thresholds]
    if not candidates:
        raise ValueError("threshold grid is empty")

    scores, targets, valid, sample_count = _materialize_samples(samples)
    trace: list[dict[str, float | int]] = []
    for threshold in candidates:
        prediction = (scores >= threshold) & valid
        truth = targets > 0
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        denominator = 2 * tp + fp + fn
        f1 = 1.0 if denominator == 0 else (2.0 * tp) / denominator
        trace.append(
            {
                "threshold": threshold,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    best = max(
        trace,
        key=lambda row: (
            float(row["f1"]),
            -int(row["fp"]),
            float(row["threshold"]),
        ),
    )
    metadata = dict(base_config.training)
    metadata["af_threshold_calibration"] = {
        "method": "grid_micro_f1_on_train_partition",
        "samples": sample_count,
        "best_threshold": float(best["threshold"]),
        "best_f1_train": float(best["f1"]),
        "trace": trace,
    }
    calibrated = replace(
        base_config,
        af=replace(base_config.af, threshold=float(best["threshold"])),
        training=metadata,
    )
    return calibrated, trace


def calibrate_bs_thresholds(
    samples: Iterable[BSTrainingSample],
    base_config: ModelConfig,
    *,
    max_candidates: int = 64,
    passes: int = 3,
) -> tuple[ModelConfig, dict[str, object]]:
    scores, targets, valid, sample_count = _materialize_samples(samples)
    result = optimize_ordered_thresholds(
        scores,
        targets,
        valid,
        initial=base_config.bs.default_thresholds,
        max_candidates=max_candidates,
        passes=passes,
    )
    thresholds = tuple(float(value) for value in result["thresholds"])

    bs = replace(
        base_config.bs,
        default_thresholds=thresholds,
        natural_open_thresholds=thresholds,
        crop_thresholds=thresholds,
        forest_thresholds=thresholds,
    )
    metadata = dict(base_config.training)
    metadata["bs_threshold_calibration"] = {
        "method": "ordered_coordinate_search_on_competition_bs_subscore",
        "samples": sample_count,
        **result,
    }
    calibrated = replace(base_config, bs=bs, training=metadata)
    return calibrated, result
