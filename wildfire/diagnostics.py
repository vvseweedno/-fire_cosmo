"""Error and diversity diagnostics for leakage-safe OOF predictions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _mask_for(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(pred)
    t = np.asarray(target)
    if p.shape != t.shape:
        raise ValueError("prediction and target shapes differ")
    if valid is None:
        mask = np.ones(t.shape, dtype=bool)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != t.shape:
            raise ValueError("valid mask shape differs from target")
    return p, t, mask


def af_error_summary(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    landcover: np.ndarray | None = None,
) -> dict[str, object]:
    """Return deterministic AF confusion/error diagnostics."""

    p, t, mask = _mask_for(pred, target, valid)
    p = p.astype(bool)
    t = t.astype(bool)

    tp_mask = p & t & mask
    fp_mask = p & ~t & mask
    fn_mask = ~p & t & mask
    tn_mask = ~p & ~t & mask

    tp = int(np.count_nonzero(tp_mask))
    fp = int(np.count_nonzero(fp_mask))
    fn = int(np.count_nonzero(fn_mask))
    tn = int(np.count_nonzero(tn_mask))
    precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1_denom = 2 * tp + fp + fn
    f1 = 1.0 if f1_denom == 0 else 2.0 * tp / f1_denom

    report: dict[str, object] = {
        "pixels": int(np.count_nonzero(mask)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_target_pixels": int(np.count_nonzero(t & mask)),
        "positive_prediction_pixels": int(np.count_nonzero(p & mask)),
    }

    if landcover is not None:
        lc = np.asarray(landcover)
        if lc.shape != t.shape:
            raise ValueError("landcover shape differs from target")
        classes = np.unique(lc[mask])
        report["false_positive_landcover"] = {
            str(int(class_id)): int(np.count_nonzero(fp_mask & (lc == class_id)))
            for class_id in classes
            if np.issubdtype(type(class_id), np.integer)
            or float(class_id).is_integer()
        }
        report["false_negative_landcover"] = {
            str(int(class_id)): int(np.count_nonzero(fn_mask & (lc == class_id)))
            for class_id in classes
            if np.issubdtype(type(class_id), np.integer)
            or float(class_id).is_integer()
        }

    return report


def bs_error_summary(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    classes: tuple[int, ...] = (0, 1, 2, 3),
) -> dict[str, object]:
    """Return burned/unburned and severity-class diagnostics."""

    p, t, mask = _mask_for(pred, target, valid)
    burn = af_error_summary(p > 0, t > 0, mask)

    confusion: dict[str, dict[str, int]] = {}
    iou: dict[str, float] = {}
    for true_class in classes:
        row: dict[str, int] = {}
        for pred_class in classes:
            row[str(pred_class)] = int(
                np.count_nonzero((t == true_class) & (p == pred_class) & mask)
            )
        confusion[str(true_class)] = row

    for class_id in classes[1:]:
        pred_c = (p == class_id) & mask
        target_c = (t == class_id) & mask
        union = int(np.count_nonzero(pred_c | target_c))
        intersection = int(np.count_nonzero(pred_c & target_c))
        iou[str(class_id)] = 1.0 if union == 0 else intersection / union

    return {
        "pixels": int(np.count_nonzero(mask)),
        "burn": burn,
        "severity_confusion": confusion,
        "severity_iou": iou,
        "miou_severity": float(np.mean(list(iou.values()))) if iou else None,
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 1e-12 or b_std <= 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def candidate_diversity(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure complementary prediction errors for candidate selection."""

    if len(predictions) < 2:
        raise ValueError("at least two candidates are required")

    truth = np.asarray(target)
    mask = np.ones(truth.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != truth.shape:
        raise ValueError("valid mask shape differs from target")

    resolved: dict[str, np.ndarray] = {}
    for name, pred in predictions.items():
        arr = np.asarray(pred)
        if arr.shape != truth.shape:
            raise ValueError(f"{name}: prediction shape differs from target")
        resolved[str(name)] = arr

    names = sorted(resolved)
    pairs: list[dict[str, object]] = []
    for index, left_name in enumerate(names):
        left = resolved[left_name]
        left_error = (left != truth) & mask
        for right_name in names[index + 1 :]:
            right = resolved[right_name]
            right_error = (right != truth) & mask
            disagreement = (left != right) & mask
            left_only_error = left_error & ~right_error
            right_only_error = right_error & ~left_error

            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "prediction_correlation": _correlation(left[mask], right[mask]),
                    "disagreement_rate": float(
                        np.count_nonzero(disagreement) / max(np.count_nonzero(mask), 1)
                    ),
                    "error_overlap_pixels": int(np.count_nonzero(left_error & right_error)),
                    "error_symmetric_difference_pixels": int(
                        np.count_nonzero(left_error ^ right_error)
                    ),
                    "left_only_error_pixels": int(np.count_nonzero(left_only_error)),
                    "right_only_error_pixels": int(np.count_nonzero(right_only_error)),
                }
            )

    return {
        "pixels": int(np.count_nonzero(mask)),
        "candidates": names,
        "pairs": pairs,
    }
