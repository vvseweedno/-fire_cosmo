"""Deterministic metric-aware calibration before heavier ML models are trained."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from wildfire.calibration import exact_f1_threshold, optimize_ordered_thresholds
from wildfire.constants import (
    LC_CROP,
    LC_GRASS,
    LC_MANGROVE,
    LC_MOSS,
    LC_SHRUB,
    LC_TREE,
    LC_WETLAND,
)
from wildfire.evaluation import BinaryAccumulator, SeverityAccumulator
from wildfire.fusion import BurnFusionComponents, fuse_burn_score
from wildfire.model_config import ModelConfig


AFTrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray]
BSTrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray]
BSFusionTrainingSample = tuple[BurnFusionComponents, np.ndarray]
BSLandcoverTrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


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


def calibrate_bs_cloud_sar_fallback(
    samples: Iterable[BSFusionTrainingSample],
    base_config: ModelConfig,
    *,
    cloud_weight_candidates: Iterable[float] = (
        -0.50,
        -0.25,
        -0.10,
        -0.05,
        0.0,
        0.05,
        0.10,
        0.25,
        0.50,
    ),
    max_candidates: int = 64,
    passes: int = 3,
) -> tuple[ModelConfig, dict[str, object]]:
    """Select cloud SAR fallback only when it improves the BS competition objective.

    Candidate zero reproduces the previous masking behavior, so the optimizer
    always has a route back to the baseline on the same calibration pixels.
    """
    materialized = list(samples)
    if not materialized:
        raise ValueError("no BS fusion training samples were supplied")

    candidates = sorted(
        {
            float(value)
            for value in (*cloud_weight_candidates, base_config.bs.cloud_sar_weight, 0.0)
        }
    )

    rows: list[dict[str, object]] = []
    best_config = base_config
    best_result: dict[str, object] | None = None
    best_key: tuple[float, float] | None = None

    for cloud_weight in candidates:
        score_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []

        for components, target in materialized:
            score, valid = fuse_burn_score(
                components,
                clear_sar_weight=base_config.bs.sar_weight,
                cloud_sar_weight=cloud_weight,
                sar_clip=base_config.bs.sar_clip,
            )
            target_array = np.asarray(target)
            if target_array.shape != score.shape:
                raise ValueError("BS fusion target and score shapes differ")
            score_parts.append(score.ravel())
            target_parts.append(target_array.ravel())
            valid_parts.append(valid.ravel())

        scores = np.concatenate(score_parts)
        targets = np.concatenate(target_parts)
        valid = np.concatenate(valid_parts)

        result = optimize_ordered_thresholds(
            scores,
            targets,
            valid,
            initial=base_config.bs.default_thresholds,
            max_candidates=max_candidates,
            passes=passes,
        )
        row = {
            "cloud_sar_weight": cloud_weight,
            **result,
        }
        rows.append(row)

        key = (
            float(result["bs_subscore"]),
            -abs(cloud_weight - base_config.bs.cloud_sar_weight),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_result = row
            thresholds = tuple(float(value) for value in result["thresholds"])
            best_config = replace(
                base_config,
                bs=replace(
                    base_config.bs,
                    cloud_sar_weight=cloud_weight,
                    default_thresholds=thresholds,
                    natural_open_thresholds=thresholds,
                    crop_thresholds=thresholds,
                    forest_thresholds=thresholds,
                ),
            )

    assert best_result is not None
    metadata = dict(best_config.training)
    metadata["bs_cloud_sar_calibration"] = {
        "method": "metric_aware_grid_with_zero_fallback_anchor",
        "samples": len(materialized),
        "baseline_anchor_cloud_sar_weight": 0.0,
        "chosen_cloud_sar_weight": best_config.bs.cloud_sar_weight,
        "candidates": rows,
    }
    return replace(best_config, training=metadata), best_result


def _bs_score(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float]:
    burn = BinaryAccumulator()
    severity = SeverityAccumulator()
    burn.update(prediction > 0, target > 0, valid)
    severity.update(prediction, target, valid)
    return burn.iou, severity.miou, 0.35 * burn.iou + 0.30 * severity.miou


def _landcover_masks(landcover: np.ndarray) -> dict[str, np.ndarray]:
    lc = np.asarray(landcover)
    natural = np.isin(
        lc,
        [LC_SHRUB, LC_GRASS, LC_WETLAND, LC_MANGROVE, LC_MOSS],
    )
    crop = lc == LC_CROP
    forest = lc == LC_TREE
    return {
        "default": ~(natural | crop | forest),
        "natural_open": natural,
        "crop": crop,
        "forest": forest,
    }


def _predict_landcover_thresholds(
    scores: np.ndarray,
    valid: np.ndarray,
    landcover: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    groups = _landcover_masks(landcover)
    thresholds = {
        "default": config.bs.default_thresholds,
        "natural_open": config.bs.natural_open_thresholds,
        "crop": config.bs.crop_thresholds,
        "forest": config.bs.forest_thresholds,
    }
    prediction = np.zeros(scores.shape, dtype=np.uint8)
    for group_name, group_mask in groups.items():
        low, moderate, high = thresholds[group_name]
        active = valid & group_mask
        prediction[(scores >= low) & active] = 1
        prediction[(scores >= moderate) & active] = 2
        prediction[(scores >= high) & active] = 3
    return prediction


def calibrate_bs_landcover_thresholds(
    samples: Iterable[BSLandcoverTrainingSample],
    base_config: ModelConfig,
    *,
    max_candidates: int = 64,
    passes: int = 2,
) -> tuple[ModelConfig, dict[str, object]]:
    """Monotonic land-cover-specific threshold refinement.

    Each proposed group-specific threshold set is first fitted only on that
    group's pixels, then accepted only if the *global* weighted BS competition
    subscore does not decrease.  The current configuration is therefore always
    a legal fallback on the calibration pool.
    """

    score_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    landcover_parts: list[np.ndarray] = []
    sample_count = 0

    for score, target, valid, landcover in samples:
        s = np.asarray(score, dtype=np.float32)
        t = np.asarray(target)
        v = np.asarray(valid, dtype=bool)
        lc = np.asarray(landcover)
        if not (s.shape == t.shape == v.shape == lc.shape):
            raise ValueError("BS landcover calibration arrays must have matching shapes")
        score_parts.append(s.ravel())
        target_parts.append(t.ravel())
        valid_parts.append(v.ravel())
        landcover_parts.append(lc.ravel())
        sample_count += 1

    if not score_parts:
        raise ValueError("no BS landcover calibration samples were supplied")

    scores = np.concatenate(score_parts)
    targets = np.concatenate(target_parts)
    valid = np.concatenate(valid_parts)
    landcover = np.concatenate(landcover_parts)

    current = base_config
    current_prediction = _predict_landcover_thresholds(
        scores,
        valid,
        landcover,
        current,
    )
    current_iou, current_miou, current_score = _bs_score(
        current_prediction,
        targets,
        valid,
    )
    trace: list[dict[str, object]] = [
        {
            "step": "baseline",
            "iou_burn": current_iou,
            "miou_severity": current_miou,
            "bs_subscore": current_score,
        }
    ]

    for pass_index in range(max(1, passes)):
        changed = False
        group_masks = _landcover_masks(landcover)
        for group_name in ("default", "natural_open", "crop", "forest"):
            group_valid = valid & group_masks[group_name]
            if not np.any(group_valid):
                continue

            attr = {
                "default": "default_thresholds",
                "natural_open": "natural_open_thresholds",
                "crop": "crop_thresholds",
                "forest": "forest_thresholds",
            }[group_name]
            initial = getattr(current.bs, attr)
            local = optimize_ordered_thresholds(
                scores,
                targets,
                group_valid,
                initial=initial,
                max_candidates=max_candidates,
                passes=2,
            )
            candidate_thresholds = tuple(float(value) for value in local["thresholds"])
            candidate_bs = replace(current.bs, **{attr: candidate_thresholds})
            candidate = replace(current, bs=candidate_bs)
            prediction = _predict_landcover_thresholds(
                scores,
                valid,
                landcover,
                candidate,
            )
            iou_burn, miou_severity, candidate_score = _bs_score(
                prediction,
                targets,
                valid,
            )
            accepted = candidate_score >= current_score - 1e-12
            trace.append(
                {
                    "pass": pass_index,
                    "group": group_name,
                    "candidate_thresholds": candidate_thresholds,
                    "iou_burn": iou_burn,
                    "miou_severity": miou_severity,
                    "bs_subscore": candidate_score,
                    "accepted": accepted,
                }
            )
            if accepted and candidate_thresholds != initial:
                current = candidate
                current_score = candidate_score
                changed = True

        if not changed:
            break

    final_prediction = _predict_landcover_thresholds(
        scores,
        valid,
        landcover,
        current,
    )
    final_iou, final_miou, final_score = _bs_score(
        final_prediction,
        targets,
        valid,
    )

    metadata = dict(current.training)
    metadata["bs_landcover_threshold_calibration"] = {
        "method": "group_local_proposal_global_metric_acceptance",
        "samples": sample_count,
        "baseline_preserved": True,
        "iou_burn": final_iou,
        "miou_severity": final_miou,
        "bs_subscore": final_score,
        "trace": trace,
    }
    return replace(current, training=metadata), {
        "iou_burn": final_iou,
        "miou_severity": final_miou,
        "bs_subscore": final_score,
        "thresholds": {
            "default": current.bs.default_thresholds,
            "natural_open": current.bs.natural_open_thresholds,
            "crop": current.bs.crop_thresholds,
            "forest": current.bs.forest_thresholds,
        },
        "trace": trace,
    }
