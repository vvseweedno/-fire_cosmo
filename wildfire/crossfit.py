"""Cross-fitted OOF calibration for less optimistic model comparison."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from wildfire.calibration import (
    apply_ordered_thresholds,
    exact_f1_threshold,
    optimize_ordered_thresholds,
)
from wildfire.evaluation import CompetitionEvaluator
from wildfire.model_config import ModelConfig
from wildfire.oof import OOFPool, OOFRecord


def _assignment_from_manifest(manifest: dict[str, object]) -> dict[str, int]:
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("fold manifest must contain at least two folds")

    assignment: dict[str, int] = {}
    for fallback_index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError("fold entries must be objects")
        fold_index = int(fold.get("fold", fallback_index))
        validation = fold.get("validation")
        if not isinstance(validation, list) or not all(
            isinstance(item, str) for item in validation
        ):
            raise ValueError("each fold must contain validation chip ids")
        for chip_id in validation:
            if chip_id in assignment:
                raise ValueError(f"chip appears in multiple folds: {chip_id}")
            assignment[chip_id] = fold_index

    if not assignment:
        raise ValueError("fold manifest has no validation chips")
    return assignment


def _concat_task(
    records: tuple[OOFRecord, ...],
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = tuple(record for record in records if record.task.upper() == task)
    if not selected:
        raise ValueError(f"calibration records contain no {task} samples")
    return (
        np.concatenate([record.score.ravel() for record in selected]),
        np.concatenate([record.target.ravel() for record in selected]),
        np.concatenate([record.valid.astype(bool).ravel() for record in selected]),
    )


def crossfit_calibrate_and_evaluate(
    pool: OOFPool,
    fold_manifest: dict[str, object],
    base_config: ModelConfig,
    *,
    bs_max_candidates: int = 64,
    bs_passes: int = 3,
) -> tuple[ModelConfig, dict[str, object]]:
    """Evaluate each fold using thresholds calibrated only on the other folds.

    Returns:
    - a final deployment config calibrated on all OOF predictions;
    - a report whose primary score is the cross-fitted estimate.
    """
    assignment = _assignment_from_manifest(fold_manifest)
    pool.validate_expected(set(assignment))

    records = pool.records
    fold_ids = sorted(set(assignment.values()))
    evaluator = CompetitionEvaluator()
    fold_reports: list[dict[str, object]] = []

    for fold_id in fold_ids:
        calibration_records = tuple(
            record for record in records if assignment[record.chip_id] != fold_id
        )
        holdout_records = tuple(
            record for record in records if assignment[record.chip_id] == fold_id
        )
        if not holdout_records:
            raise ValueError(f"fold {fold_id} has no OOF records")

        af_scores, af_target, af_valid = _concat_task(calibration_records, "AF")
        af_result = exact_f1_threshold(af_scores, af_target, af_valid)

        bs_scores, bs_target, bs_valid = _concat_task(calibration_records, "BS")
        bs_result = optimize_ordered_thresholds(
            bs_scores,
            bs_target,
            bs_valid,
            initial=base_config.bs.default_thresholds,
            max_candidates=bs_max_candidates,
            passes=bs_passes,
        )
        thresholds = tuple(float(value) for value in bs_result["thresholds"])
        af_threshold = float(af_result["threshold"])

        fold_evaluator = CompetitionEvaluator()
        af_holdout = 0
        bs_holdout = 0
        for record in holdout_records:
            if record.task.upper() == "AF":
                prediction = (
                    (np.asarray(record.score) >= af_threshold)
                    & np.asarray(record.valid, dtype=bool)
                ).astype(np.uint8)
                evaluator.update_af(prediction, record.target)
                fold_evaluator.update_af(prediction, record.target)
                af_holdout += 1
            else:
                prediction = apply_ordered_thresholds(
                    record.score,
                    thresholds,
                    record.valid,
                )
                evaluator.update_bs(prediction, record.target)
                fold_evaluator.update_bs(prediction, record.target)
                bs_holdout += 1

        fold_summary = fold_evaluator.summary()
        fold_reports.append(
            {
                "fold": fold_id,
                "calibration_chips": len(calibration_records),
                "holdout_chips": len(holdout_records),
                "holdout_AF": af_holdout,
                "holdout_BS": bs_holdout,
                "af_threshold": af_threshold,
                "bs_thresholds": thresholds,
                "f1_af": fold_summary["f1_af"],
                "iou_burn": fold_summary["iou_burn"],
                "miou_severity": fold_summary["miou_severity"],
                "score": fold_summary["score"],
            }
        )

    crossfit_summary = evaluator.summary()

    deployment_config, pooled_report = pool.calibrate_and_evaluate(
        base_config,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
    )
    metadata = dict(deployment_config.training)
    metadata["crossfit_validation"] = {
        "folds": len(fold_ids),
        "score": crossfit_summary["score"],
        "f1_af": crossfit_summary["f1_af"],
        "iou_burn": crossfit_summary["iou_burn"],
        "miou_severity": crossfit_summary["miou_severity"],
    }
    deployment_config = replace(deployment_config, training=metadata)

    return deployment_config, {
        "primary_validation": "cross_fitted_oof",
        "crossfit": crossfit_summary,
        "folds": fold_reports,
        "pooled_calibration_for_deployment": {
            "score_on_same_oof_after_global_calibration": pooled_report["score"],
            "warning": (
                "Use this pooled value for threshold/ensemble fitting, not as the "
                "primary unbiased model-comparison estimate."
            ),
            "calibration": pooled_report["calibration"],
        },
    }
