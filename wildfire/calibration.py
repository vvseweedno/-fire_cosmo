"""Metric-aware calibration utilities for AF and BS.

These routines optimize the competition metrics only on the supplied
training/OOF partition. They never inspect test labels.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from wildfire.metrics import binary_iou, severity_miou


def exact_f1_threshold(
    scores: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Find the exact >= threshold that maximizes micro-F1.

    Predictions can be constrained by valid, but target pixels outside that
    mask remain part of the official score and therefore remain false
    negatives when positive.
    """
    score = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(target).ravel() > 0
    if score.shape != truth.shape:
        raise ValueError("scores and target shapes differ")

    allowed = np.ones_like(truth, dtype=bool)
    if valid is not None:
        allowed = np.asarray(valid, dtype=bool).ravel()
        if allowed.shape != truth.shape:
            raise ValueError("valid and target shapes differ")

    usable = allowed & np.isfinite(score)
    if not np.any(usable):
        raise ValueError("no finite model scores are available for threshold calibration")

    s = score[usable]
    y = truth[usable]
    order = np.argsort(s, kind="mergesort")[::-1]
    s = s[order]
    y = y[order]

    total_positive = int(np.count_nonzero(truth))
    max_score = float(s[0])
    no_prediction_threshold = float(np.nextafter(max_score, np.inf))

    def _f1(tp: int, fp: int, fn: int) -> float:
        denominator = 2 * tp + fp + fn
        return 1.0 if denominator == 0 else (2.0 * tp) / denominator

    best: dict[str, float | int] = {
        "threshold": no_prediction_threshold,
        "f1": _f1(0, 0, total_positive),
        "tp": 0,
        "fp": 0,
        "fn": total_positive,
        "predicted_positive": 0,
    }

    tp = 0
    fp = 0
    fn = total_positive
    index = 0
    while index < s.size:
        value = s[index]
        end = index + 1
        while end < s.size and s[end] == value:
            end += 1

        positives = int(np.count_nonzero(y[index:end]))
        negatives = (end - index) - positives
        tp += positives
        fp += negatives
        fn -= positives
        f1 = _f1(tp, fp, fn)

        candidate = {
            "threshold": float(value),
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_positive": tp + fp,
        }
        candidate_key = (
            float(candidate["f1"]),
            -int(candidate["fp"]),
            float(candidate["threshold"]),
        )
        best_key = (
            float(best["f1"]),
            -int(best["fp"]),
            float(best["threshold"]),
        )
        if candidate_key > best_key:
            best = candidate
        index = end

    return best


def apply_ordered_thresholds(
    scores: np.ndarray,
    thresholds: tuple[float, float, float],
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Map a continuous severity score to ordered classes 0/1/2/3."""
    low, moderate, high = (float(value) for value in thresholds)
    if not low < moderate < high:
        raise ValueError("severity thresholds must be strictly increasing")

    score = np.asarray(scores, dtype=np.float32)
    usable = np.isfinite(score)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != score.shape:
            raise ValueError("valid and scores shapes differ")
        usable &= mask

    result = np.zeros(score.shape, dtype=np.uint8)
    result[(score >= low) & usable] = 1
    result[(score >= moderate) & usable] = 2
    result[(score >= high) & usable] = 3
    return result


def bs_competition_subscore(prediction: np.ndarray, target: np.ndarray) -> float:
    """Contribution of the BS task to total competition score, max 0.65."""
    pred = np.asarray(prediction)
    truth = np.asarray(target)
    return 0.35 * binary_iou(pred > 0, truth > 0) + 0.30 * severity_miou(pred, truth)


def _candidate_thresholds(
    scores: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    max_candidates: int,
    anchors: Iterable[float] = (),
) -> np.ndarray:
    if max_candidates < 8:
        raise ValueError("max_candidates must be at least 8")

    score = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(target).ravel()
    usable = np.asarray(valid, dtype=bool).ravel() & np.isfinite(score)
    values = score[usable]
    if values.size == 0:
        raise ValueError("no finite valid BS scores are available")

    unique = np.unique(values)
    candidates: list[float] = []
    if unique.size <= max_candidates:
        candidates.extend(float(value) for value in unique)
    else:
        candidates.extend(
            float(value)
            for value in np.quantile(values, np.linspace(0.0, 1.0, max_candidates))
        )
        per_class = max(4, max_candidates // 8)
        for class_id in (0, 1, 2, 3):
            class_values = score[usable & (truth == class_id)]
            if class_values.size:
                candidates.extend(
                    float(value)
                    for value in np.quantile(
                        class_values,
                        np.linspace(0.0, 1.0, per_class),
                    )
                )

    candidates.extend(float(value) for value in anchors)
    clean = sorted({round(value, 10) for value in candidates if np.isfinite(value)})
    return np.asarray(clean, dtype=np.float64)


def optimize_ordered_thresholds(
    scores: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    initial: tuple[float, float, float],
    max_candidates: int = 64,
    passes: int = 3,
) -> dict[str, object]:
    """Coordinate-search ordered BS thresholds against the actual score formula."""
    if passes < 1:
        raise ValueError("passes must be positive")

    score = np.asarray(scores, dtype=np.float32)
    truth = np.asarray(target)
    mask = np.asarray(valid, dtype=bool)
    if score.shape != truth.shape or score.shape != mask.shape:
        raise ValueError("scores, target and valid shapes must match")
    if not initial[0] < initial[1] < initial[2]:
        raise ValueError("initial thresholds must be strictly increasing")

    candidates = _candidate_thresholds(
        score,
        truth,
        mask,
        max_candidates=max_candidates,
        anchors=initial,
    )
    current = tuple(float(value) for value in initial)
    current_prediction = apply_ordered_thresholds(score, current, mask)
    current_score = bs_competition_subscore(current_prediction, truth)
    trace: list[dict[str, object]] = []

    for pass_index in range(passes):
        pass_improved = False
        for threshold_index in range(3):
            best_thresholds = current
            best_score = current_score
            best_distance = sum(
                abs(current[index] - initial[index]) for index in range(3)
            )

            for candidate in candidates:
                proposed = list(current)
                proposed[threshold_index] = float(candidate)
                if not proposed[0] < proposed[1] < proposed[2]:
                    continue

                proposed_tuple = tuple(proposed)
                prediction = apply_ordered_thresholds(score, proposed_tuple, mask)
                objective = bs_competition_subscore(prediction, truth)
                distance = sum(
                    abs(proposed_tuple[index] - initial[index]) for index in range(3)
                )
                if (objective, -distance) > (best_score, -best_distance):
                    best_thresholds = proposed_tuple
                    best_score = objective
                    best_distance = distance

            if best_thresholds != current:
                pass_improved = True
                current = best_thresholds
                current_score = best_score

            trace.append(
                {
                    "pass": pass_index,
                    "coordinate": threshold_index,
                    "thresholds": list(current),
                    "bs_subscore": float(current_score),
                }
            )

        if not pass_improved:
            break

    final_prediction = apply_ordered_thresholds(score, current, mask)
    return {
        "thresholds": current,
        "bs_subscore": float(current_score),
        "iou_burn": float(binary_iou(final_prediction > 0, truth > 0)),
        "miou_severity": float(severity_miou(final_prediction, truth)),
        "candidate_count": int(candidates.size),
        "trace": trace,
    }
