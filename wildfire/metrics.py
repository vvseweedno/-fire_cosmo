"""Competition-oriented pixel metrics."""

from __future__ import annotations

import numpy as np


def _binary_counts(pred: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    p = np.asarray(pred).astype(bool)
    t = np.asarray(target).astype(bool)
    if p.shape != t.shape:
        raise ValueError("Prediction and target shapes differ")
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    return tp, fp, fn


def binary_f1(pred: np.ndarray, target: np.ndarray) -> float:
    tp, fp, fn = _binary_counts(pred, target)
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else (2.0 * tp) / denom


def binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    tp, fp, fn = _binary_counts(pred, target)
    denom = tp + fp + fn
    return 1.0 if denom == 0 else tp / denom


def severity_miou(
    pred: np.ndarray,
    target: np.ndarray,
    classes: tuple[int, ...] = (1, 2, 3),
) -> float:
    """Mean IoU over severity classes; absent-in-both classes do not inflate the score."""
    p = np.asarray(pred)
    t = np.asarray(target)
    if p.shape != t.shape:
        raise ValueError("Prediction and target shapes differ")

    values: list[float] = []
    for class_id in classes:
        p_c = p == class_id
        t_c = t == class_id
        union = int(np.count_nonzero(p_c | t_c))
        if union == 0:
            continue
        intersection = int(np.count_nonzero(p_c & t_c))
        values.append(intersection / union)
    return float(np.mean(values)) if values else 1.0


def competition_score(
    af_pred: np.ndarray,
    af_target: np.ndarray,
    bs_pred: np.ndarray,
    bs_target: np.ndarray,
) -> dict[str, float]:
    af_f1 = binary_f1(af_pred, af_target)
    burn_iou = binary_iou(np.asarray(bs_pred) > 0, np.asarray(bs_target) > 0)
    sev_miou = severity_miou(bs_pred, bs_target)
    score = 0.35 * af_f1 + 0.35 * burn_iou + 0.30 * sev_miou
    return {
        "f1_af": af_f1,
        "iou_burn": burn_iou,
        "miou_severity": sev_miou,
        "score": score,
    }
