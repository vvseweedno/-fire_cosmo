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


def _sample_for_quantiles(values: np.ndarray, limit: int = 2_000_000) -> np.ndarray:
    """Deterministically cap quantile work without random sampling."""
    if values.size <= limit:
        return values
    step = int(np.ceil(values.size / limit))
    return values[::step][:limit]


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

    candidates: list[float] = []
    if values.size <= max_candidates:
        candidates.extend(float(value) for value in np.unique(values))
    else:
        sampled = _sample_for_quantiles(values)
        candidates.extend(
            float(value)
            for value in np.quantile(
                sampled,
                np.linspace(0.0, 1.0, max_candidates),
            )
        )
        per_class = max(4, max_candidates // 8)
        for class_id in (0, 1, 2, 3):
            class_values = score[usable & (truth == class_id)]
            if class_values.size:
                class_sample = _sample_for_quantiles(class_values)
                candidates.extend(
                    float(value)
                    for value in np.quantile(
                        class_sample,
                        np.linspace(0.0, 1.0, per_class),
                    )
                )

    candidates.extend(float(value) for value in anchors)
    clean = sorted({round(value, 10) for value in candidates if np.isfinite(value)})
    return np.asarray(clean, dtype=np.float64)


def _binned_truth_counts(
    score: np.ndarray,
    truth: np.ndarray,
    usable: np.ndarray,
    candidates: np.ndarray,
    *,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """Aggregate truth classes into candidate-defined score bins.

    Columns are: non-burn, severity 1, severity 2, severity 3, other burn.
    The output is tiny: (candidate_count + 1) x 5.
    """
    bins = np.zeros((candidates.size + 1, 5), dtype=np.int64)
    flat_score = np.asarray(score, dtype=np.float64).ravel()
    flat_truth = np.asarray(truth).ravel()
    flat_usable = np.asarray(usable, dtype=bool).ravel()
    usable_indices = np.flatnonzero(flat_usable)

    for start in range(0, usable_indices.size, chunk_size):
        index = usable_indices[start : start + chunk_size]
        values = flat_score[index]
        labels = flat_truth[index]

        score_bin = np.searchsorted(candidates, values, side="right")
        category = np.zeros(labels.shape, dtype=np.int8)
        category[labels == 1] = 1
        category[labels == 2] = 2
        category[labels == 3] = 3
        category[(labels > 0) & ~np.isin(labels, (1, 2, 3))] = 4

        packed = score_bin * 5 + category
        counts = np.bincount(
            packed,
            minlength=(candidates.size + 1) * 5,
        ).reshape(candidates.size + 1, 5)
        bins += counts.astype(np.int64, copy=False)

    return bins


def _threshold_objective_from_bins(
    prefix: np.ndarray,
    candidates: np.ndarray,
    thresholds: tuple[float, float, float],
    total_truth_burn: int,
    total_truth_classes: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Return burn IoU, severity mIoU, weighted BS subscore."""

    def boundary(threshold: float) -> int:
        index = int(np.searchsorted(candidates, threshold, side="left"))
        return min(max(index + 1, 0), prefix.shape[0] - 1)

    b1, b2, b3 = (boundary(value) for value in thresholds)
    if not b1 < b2 < b3:
        raise ValueError("threshold boundaries must be strictly increasing")

    end = prefix.shape[0] - 1
    class1 = prefix[b2] - prefix[b1]
    class2 = prefix[b3] - prefix[b2]
    class3 = prefix[end] - prefix[b3]
    burn_pred = prefix[end] - prefix[b1]

    burn_tp = int(np.sum(burn_pred[1:]))
    burn_fp = int(burn_pred[0])
    burn_denom = total_truth_burn + burn_fp
    burn_iou = 1.0 if burn_denom == 0 else burn_tp / burn_denom

    intervals = (class1, class2, class3)
    ious: list[float] = []
    for class_index, counts in enumerate(intervals, start=1):
        predicted = int(np.sum(counts))
        intersection = int(counts[class_index])
        truth_count = total_truth_classes[class_index - 1]
        union = truth_count + predicted - intersection
        ious.append(1.0 if union == 0 else intersection / union)

    severity_miou_value = float(np.mean(ious))
    objective = 0.35 * burn_iou + 0.30 * severity_miou_value
    return float(burn_iou), severity_miou_value, float(objective)


def optimize_ordered_thresholds(
    scores: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    initial: tuple[float, float, float],
    max_candidates: int = 64,
    passes: int = 3,
) -> dict[str, object]:
    """Coordinate-search ordered BS thresholds against the actual score formula.

    Candidate evaluation uses a compact histogram of score bins rather than
    materialising a full prediction mask for every trial. This preserves the
    competition objective while making wider threshold searches practical.
    """
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
    usable = mask & np.isfinite(score)
    bins = _binned_truth_counts(score, truth, usable, candidates)
    prefix = np.zeros((bins.shape[0] + 1, bins.shape[1]), dtype=np.int64)
    prefix[1:] = np.cumsum(bins, axis=0)

    flat_truth = truth.ravel()
    total_truth_burn = int(np.count_nonzero(flat_truth > 0))
    total_truth_classes = tuple(
        int(np.count_nonzero(flat_truth == class_id))
        for class_id in (1, 2, 3)
    )

    current = tuple(float(value) for value in initial)
    _, _, current_score = _threshold_objective_from_bins(
        prefix,
        candidates,
        current,
        total_truth_burn,
        total_truth_classes,
    )
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
                _, _, objective = _threshold_objective_from_bins(
                    prefix,
                    candidates,
                    proposed_tuple,
                    total_truth_burn,
                    total_truth_classes,
                )
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

    iou_burn, miou_severity, final_score = _threshold_objective_from_bins(
        prefix,
        candidates,
        current,
        total_truth_burn,
        total_truth_classes,
    )
    return {
        "thresholds": current,
        "bs_subscore": float(final_score),
        "iou_burn": float(iou_burn),
        "miou_severity": float(miou_severity),
        "candidate_count": int(candidates.size),
        "trace": trace,
    }
