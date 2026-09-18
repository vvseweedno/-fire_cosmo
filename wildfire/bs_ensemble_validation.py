"""Fold-wise validation for the deployable BS candidate ensemble.

Weights, global thresholds and land-cover thresholds for fold k are calibrated
only on the other folds.  Because AF is unchanged, the delta in the weighted BS
subscore is also the delta in the full competition Score.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from wildfire.bs_candidates import BS_CANDIDATE_NAMES
from wildfire.ensemble import optimize_bs_ensemble
from wildfire.evaluation import BinaryAccumulator, SeverityAccumulator
from wildfire.model_config import ModelConfig
from wildfire.oof import OOFRecord, load_oof_directory
from wildfire.training import (
    apply_landcover_thresholds,
    calibrate_bs_landcover_thresholds,
)


def _assignment(manifest: dict[str, object]) -> dict[str, int]:
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("fold manifest must contain at least two folds")

    result: dict[str, int] = {}
    for fallback_index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError("fold entries must be objects")
        fold_id = int(fold.get("fold", fallback_index))
        validation = fold.get("validation")
        if not isinstance(validation, list) or not all(
            isinstance(item, str) for item in validation
        ):
            raise ValueError("each fold must contain validation chip ids")
        for chip_id in validation:
            if chip_id in result:
                raise ValueError(f"chip appears in multiple folds: {chip_id}")
            result[chip_id] = fold_id
    return result


def _load_candidates(
    root: str | Path,
) -> dict[str, dict[str, OOFRecord]]:
    base = Path(root)
    result: dict[str, dict[str, OOFRecord]] = {}
    for name in BS_CANDIDATE_NAMES:
        pool = load_oof_directory(base / name)
        records = {
            record.chip_id: record
            for record in pool.records
            if record.task.upper() == "BS"
        }
        if not records:
            raise RuntimeError(f"{name}: no BS records")
        result[name] = records

    reference = set(result["BASE"])
    for name, records in result.items():
        if set(records) != reference:
            raise RuntimeError(f"{name}: chip set differs from BASE")
        for chip_id in reference:
            base_record = result["BASE"][chip_id]
            record = records[chip_id]
            if not np.array_equal(record.target, base_record.target):
                raise RuntimeError(f"{name}/{chip_id}: target mismatch")
            if not np.array_equal(record.valid, base_record.valid):
                raise RuntimeError(f"{name}/{chip_id}: valid mismatch")
    return result


def _landcover(record: OOFRecord) -> np.ndarray:
    if record.landcover is not None:
        return np.asarray(record.landcover)
    return np.full(record.score.shape, -1, dtype=np.int16)


def _fuse_chip(
    chip_id: str,
    candidates: dict[str, dict[str, OOFRecord]],
    weights: dict[str, float],
) -> np.ndarray:
    base = candidates["BASE"][chip_id]
    valid = np.asarray(base.valid, dtype=bool)
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("ensemble weight sum must be positive")

    fused = np.zeros(base.score.shape, dtype=np.float32)
    for name, weight in weights.items():
        score = np.asarray(candidates[name][chip_id].score, dtype=np.float32)
        safe = np.where(valid & np.isfinite(score), score, 0.0)
        fused += float(weight / total) * safe
    return fused


def _concat(
    ids: list[str],
    candidates: dict[str, dict[str, OOFRecord]],
    names: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    base = candidates["BASE"]
    target = np.concatenate([base[chip_id].target.ravel() for chip_id in ids])
    valid = np.concatenate(
        [base[chip_id].valid.astype(bool).ravel() for chip_id in ids]
    )
    scores = {
        name: np.concatenate(
            [candidates[name][chip_id].score.ravel() for chip_id in ids]
        )
        for name in names
    }
    return scores, target, valid


def _fit_fold_config(
    calibration_ids: list[str],
    candidates: dict[str, dict[str, OOFRecord]],
    base_config: ModelConfig,
    names: tuple[str, ...],
    *,
    alpha_steps: int,
    threshold_candidates: int,
    threshold_passes: int,
    landcover_passes: int,
) -> tuple[ModelConfig, dict[str, object]]:
    scores, target, valid = _concat(calibration_ids, candidates, names)
    ensemble = optimize_bs_ensemble(
        scores,
        target,
        valid,
        initial_thresholds=base_config.bs.default_thresholds,
        alpha_steps=alpha_steps,
        threshold_candidates=threshold_candidates,
        threshold_passes=threshold_passes,
    )
    weights = {
        str(name): float(value)
        for name, value in dict(ensemble["weights"]).items()
    }
    thresholds = tuple(float(value) for value in ensemble["thresholds"])
    config = replace(
        base_config,
        bs=replace(
            base_config.bs,
            score_weights=weights,
            default_thresholds=thresholds,
            natural_open_thresholds=thresholds,
            crop_thresholds=thresholds,
            forest_thresholds=thresholds,
        ),
    )

    samples = []
    for chip_id in calibration_ids:
        record = candidates["BASE"][chip_id]
        samples.append(
            (
                _fuse_chip(chip_id, candidates, weights),
                np.asarray(record.target),
                np.asarray(record.valid, dtype=bool),
                _landcover(record),
            )
        )
    config, landcover_result = calibrate_bs_landcover_thresholds(
        samples,
        config,
        max_candidates=threshold_candidates,
        passes=landcover_passes,
    )
    return config, {
        "ensemble": ensemble,
        "landcover": landcover_result,
    }


def _evaluate(
    ids: list[str],
    candidates: dict[str, dict[str, OOFRecord]],
    config: ModelConfig,
    burn: BinaryAccumulator,
    severity: SeverityAccumulator,
) -> None:
    for chip_id in ids:
        record = candidates["BASE"][chip_id]
        score = _fuse_chip(chip_id, candidates, config.bs.score_weights)
        prediction = apply_landcover_thresholds(
            score,
            np.asarray(record.valid, dtype=bool),
            _landcover(record),
            config,
        )
        # Official scoring includes all target pixels; model-invalid pixels are
        # simply predicted background and remain false negatives when positive.
        burn.update(prediction > 0, np.asarray(record.target) > 0)
        severity.update(prediction, np.asarray(record.target))


def _summary(burn: BinaryAccumulator, severity: SeverityAccumulator) -> dict[str, object]:
    subscore = 0.35 * burn.iou + 0.30 * severity.miou
    return {
        "iou_burn": burn.iou,
        "miou_severity": severity.miou,
        "bs_subscore": subscore,
        "burn_counts": burn.as_dict(),
        "severity": severity.as_dict(),
    }


def crossfit_bs_candidate_ensemble(
    candidate_root: str | Path,
    fold_manifest: dict[str, object],
    base_config: ModelConfig,
    *,
    alpha_steps: int = 12,
    threshold_candidates: int = 64,
    threshold_passes: int = 2,
    landcover_passes: int = 2,
) -> dict[str, object]:
    candidates = _load_candidates(candidate_root)
    assignment = _assignment(fold_manifest)

    bs_ids = sorted(candidates["BASE"])
    missing_assignment = sorted(set(bs_ids) - set(assignment))
    if missing_assignment:
        raise ValueError(
            "BS OOF records are missing fold assignments: "
            + ", ".join(missing_assignment[:10])
        )

    fold_ids = sorted({assignment[chip_id] for chip_id in bs_ids})
    ensemble_burn = BinaryAccumulator()
    ensemble_severity = SeverityAccumulator()
    baseline_burn = BinaryAccumulator()
    baseline_severity = SeverityAccumulator()
    fold_reports: list[dict[str, object]] = []

    for fold_id in fold_ids:
        holdout = [chip_id for chip_id in bs_ids if assignment[chip_id] == fold_id]
        calibration = [chip_id for chip_id in bs_ids if assignment[chip_id] != fold_id]
        if not holdout or not calibration:
            raise ValueError(f"fold {fold_id} has empty calibration or holdout BS set")

        ensemble_config, ensemble_fit = _fit_fold_config(
            calibration,
            candidates,
            base_config,
            BS_CANDIDATE_NAMES,
            alpha_steps=alpha_steps,
            threshold_candidates=threshold_candidates,
            threshold_passes=threshold_passes,
            landcover_passes=landcover_passes,
        )
        baseline_config, baseline_fit = _fit_fold_config(
            calibration,
            candidates,
            base_config,
            ("BASE",),
            alpha_steps=alpha_steps,
            threshold_candidates=threshold_candidates,
            threshold_passes=threshold_passes,
            landcover_passes=landcover_passes,
        )

        fold_ensemble_burn = BinaryAccumulator()
        fold_ensemble_severity = SeverityAccumulator()
        fold_baseline_burn = BinaryAccumulator()
        fold_baseline_severity = SeverityAccumulator()

        _evaluate(
            holdout,
            candidates,
            ensemble_config,
            ensemble_burn,
            ensemble_severity,
        )
        _evaluate(
            holdout,
            candidates,
            ensemble_config,
            fold_ensemble_burn,
            fold_ensemble_severity,
        )
        _evaluate(
            holdout,
            candidates,
            baseline_config,
            baseline_burn,
            baseline_severity,
        )
        _evaluate(
            holdout,
            candidates,
            baseline_config,
            fold_baseline_burn,
            fold_baseline_severity,
        )

        ensemble_summary = _summary(fold_ensemble_burn, fold_ensemble_severity)
        baseline_summary = _summary(fold_baseline_burn, fold_baseline_severity)
        fold_reports.append(
            {
                "fold": fold_id,
                "calibration_chips": len(calibration),
                "holdout_chips": len(holdout),
                "ensemble": ensemble_summary,
                "baseline": baseline_summary,
                "delta_bs_subscore": (
                    float(ensemble_summary["bs_subscore"])
                    - float(baseline_summary["bs_subscore"])
                ),
                "ensemble_fit": ensemble_fit,
                "baseline_fit": baseline_fit,
            }
        )

    ensemble_summary = _summary(ensemble_burn, ensemble_severity)
    baseline_summary = _summary(baseline_burn, baseline_severity)
    delta = float(ensemble_summary["bs_subscore"]) - float(
        baseline_summary["bs_subscore"]
    )

    return {
        "primary_validation": "cross_fitted_candidate_ensemble",
        "folds": fold_reports,
        "ensemble": ensemble_summary,
        "baseline": baseline_summary,
        "delta_bs_subscore": delta,
        "delta_total_score": delta,
        "promotion_allowed": delta > 0.0,
        "note": (
            "AF is unchanged, so delta_total_score equals the BS subscore delta. "
            "Every fold tunes weights and thresholds only on the other folds."
        ),
    }
