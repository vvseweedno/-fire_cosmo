"""Cross-fitted validation for the deployable active-fire candidate ensemble."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wildfire.af_candidates import AF_CANDIDATE_NAMES
from wildfire.ensemble import optimize_af_ensemble
from wildfire.evaluation import BinaryAccumulator
from wildfire.oof import OOFRecord, load_oof_directory


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
    for name in AF_CANDIDATE_NAMES:
        pool = load_oof_directory(base / name)
        records = {
            record.chip_id: record
            for record in pool.records
            if record.task.upper() == "AF"
        }
        if not records:
            raise RuntimeError(f"{name}: no AF records")
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


def _fit(
    calibration_ids: list[str],
    candidates: dict[str, dict[str, OOFRecord]],
    names: tuple[str, ...],
    *,
    alpha_steps: int,
) -> dict[str, object]:
    scores, target, valid = _concat(calibration_ids, candidates, names)
    return optimize_af_ensemble(
        scores,
        target,
        valid,
        alpha_steps=alpha_steps,
    )


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


def _evaluate(
    ids: list[str],
    candidates: dict[str, dict[str, OOFRecord]],
    fit: dict[str, object],
    accumulator: BinaryAccumulator,
) -> None:
    raw_weights = fit["weights"]
    if not isinstance(raw_weights, dict):
        raise ValueError("ensemble fit is missing weights")
    weights = {str(name): float(value) for name, value in raw_weights.items()}
    threshold = float(fit["threshold"])

    for chip_id in ids:
        record = candidates["BASE"][chip_id]
        valid = np.asarray(record.valid, dtype=bool)
        score = _fuse_chip(chip_id, candidates, weights)
        prediction = (score >= threshold) & valid
        # Official AF F1 is all-pixel. Invalid-model pixels remain background and
        # positive targets there are therefore false negatives.
        accumulator.update(prediction, np.asarray(record.target) > 0)


def crossfit_af_candidate_ensemble(
    candidate_root: str | Path,
    fold_manifest: dict[str, object],
    *,
    alpha_steps: int = 20,
) -> dict[str, object]:
    candidates = _load_candidates(candidate_root)
    assignment = _assignment(fold_manifest)
    af_ids = sorted(candidates["BASE"])

    missing_assignment = sorted(set(af_ids) - set(assignment))
    if missing_assignment:
        raise ValueError(
            "AF OOF records are missing fold assignments: "
            + ", ".join(missing_assignment[:10])
        )

    fold_ids = sorted({assignment[chip_id] for chip_id in af_ids})
    ensemble_acc = BinaryAccumulator()
    baseline_acc = BinaryAccumulator()
    fold_reports: list[dict[str, object]] = []

    for fold_id in fold_ids:
        holdout = [chip_id for chip_id in af_ids if assignment[chip_id] == fold_id]
        calibration = [chip_id for chip_id in af_ids if assignment[chip_id] != fold_id]
        if not holdout or not calibration:
            raise ValueError(f"fold {fold_id} has empty calibration or holdout AF set")

        ensemble_fit = _fit(
            calibration,
            candidates,
            AF_CANDIDATE_NAMES,
            alpha_steps=alpha_steps,
        )
        baseline_fit = _fit(
            calibration,
            candidates,
            ("BASE",),
            alpha_steps=alpha_steps,
        )

        fold_ensemble = BinaryAccumulator()
        fold_baseline = BinaryAccumulator()
        _evaluate(holdout, candidates, ensemble_fit, ensemble_acc)
        _evaluate(holdout, candidates, ensemble_fit, fold_ensemble)
        _evaluate(holdout, candidates, baseline_fit, baseline_acc)
        _evaluate(holdout, candidates, baseline_fit, fold_baseline)

        fold_reports.append(
            {
                "fold": fold_id,
                "calibration_chips": len(calibration),
                "holdout_chips": len(holdout),
                "ensemble_f1": fold_ensemble.f1,
                "baseline_f1": fold_baseline.f1,
                "delta_f1": fold_ensemble.f1 - fold_baseline.f1,
                "ensemble_fit": ensemble_fit,
                "baseline_fit": baseline_fit,
            }
        )

    delta_f1 = ensemble_acc.f1 - baseline_acc.f1
    return {
        "primary_validation": "cross_fitted_af_candidate_ensemble",
        "folds": fold_reports,
        "ensemble_f1": ensemble_acc.f1,
        "baseline_f1": baseline_acc.f1,
        "delta_f1": delta_f1,
        "delta_total_score": 0.35 * delta_f1,
        "promotion_epsilon": epsilon,\n        "promotion_allowed": delta_f1 > epsilon,
        "ensemble_counts": ensemble_acc.as_dict(),
        "baseline_counts": baseline_acc.as_dict(),
        "note": (
            "BS is unchanged, so delta_total_score is exactly 0.35 * delta_f1. "
            "Each fold tunes weights and threshold only on the other folds. " \\n            f"Promotion additionally requires delta_f1 > {epsilon:g}."
        ),
    }
