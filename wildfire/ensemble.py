"""Greedy convex ensemble search on pooled OOF scores.

The optimizer is deliberately conservative: it starts from the best single
model and every blending step includes alpha=1.0, which preserves the current
ensemble exactly. Therefore the selected OOF objective is monotonic
non-decreasing by construction.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wildfire.calibration import exact_f1_threshold, optimize_ordered_thresholds


def _validate_score_maps(
    score_maps: Mapping[str, np.ndarray],
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(score_maps) < 1:
        raise ValueError("at least one model score map is required")

    truth = np.asarray(target)
    mask = np.asarray(valid, dtype=bool)
    if truth.shape != mask.shape:
        raise ValueError("target and valid shapes differ")

    result: dict[str, np.ndarray] = {}
    for name, score in score_maps.items():
        if not name:
            raise ValueError("model name must not be empty")
        array = np.asarray(score, dtype=np.float32)
        if array.shape != truth.shape:
            raise ValueError(f"{name}: score shape differs from target")
        if np.any(mask & ~np.isfinite(array)):
            raise ValueError(f"{name}: non-finite score inside valid pixels")
        # Convex blending must not evaluate 0 * +/-inf outside the metric mask.
        result[name] = np.where(mask, array, 0.0).astype(np.float32, copy=False)
    return result


def _alpha_grid(steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    return np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("ensemble weight sum must be positive")
    cleaned = {
        name: float(value / total)
        for name, value in weights.items()
        if value > 1e-12
    }
    total2 = float(sum(cleaned.values()))
    return {name: value / total2 for name, value in cleaned.items()}


def optimize_af_ensemble(
    score_maps: Mapping[str, np.ndarray],
    target: np.ndarray,
    valid: np.ndarray,
    *,
    alpha_steps: int = 20,
    min_improvement: float = 0.0,
) -> dict[str, object]:
    """Greedy convex blend optimized for exact pooled micro-F1."""
    scores = _validate_score_maps(score_maps, target, valid)
    truth = np.asarray(target)
    mask = np.asarray(valid, dtype=bool)

    singles: dict[str, dict[str, float | int]] = {
        name: exact_f1_threshold(score, truth, mask)
        for name, score in scores.items()
    }
    best_name = max(
        singles,
        key=lambda name: (
            float(singles[name]["f1"]),
            -int(singles[name]["fp"]),
            name,
        ),
    )

    current_score = scores[best_name].copy()
    current_result = singles[best_name]
    weights = {best_name: 1.0}
    remaining = set(scores) - {best_name}
    trace: list[dict[str, object]] = [
        {
            "step": 0,
            "added": best_name,
            "weights": dict(weights),
            "f1": float(current_result["f1"]),
            "threshold": float(current_result["threshold"]),
        }
    ]

    alphas = _alpha_grid(alpha_steps)
    while remaining:
        step_best: tuple[str, float, np.ndarray, dict[str, float | int]] | None = None
        step_key: tuple[float, int, float, str] | None = None

        for name in sorted(remaining):
            candidate = scores[name]
            for alpha in alphas:
                fused = (
                    float(alpha) * current_score
                    + (1.0 - float(alpha)) * candidate
                )
                result = exact_f1_threshold(fused, truth, mask)
                key = (
                    float(result["f1"]),
                    -int(result["fp"]),
                    float(alpha),
                    name,
                )
                if step_key is None or key > step_key:
                    step_key = key
                    step_best = (name, float(alpha), fused, result)

        assert step_best is not None
        name, alpha, fused, result = step_best
        improvement = float(result["f1"]) - float(current_result["f1"])
        if improvement <= float(min_improvement):
            break

        weights = {
            model_name: value * alpha
            for model_name, value in weights.items()
        }
        weights[name] = weights.get(name, 0.0) + (1.0 - alpha)
        weights = _renormalize_weights(weights)
        current_score = fused
        current_result = result
        remaining.remove(name)
        trace.append(
            {
                "step": len(trace),
                "added": name,
                "alpha_keep_current": alpha,
                "weights": dict(weights),
                "f1": float(result["f1"]),
                "threshold": float(result["threshold"]),
                "delta_f1": improvement,
            }
        )

    return {
        "task": "AF",
        "weights": weights,
        "f1": float(current_result["f1"]),
        "threshold": float(current_result["threshold"]),
        "tp": int(current_result["tp"]),
        "fp": int(current_result["fp"]),
        "fn": int(current_result["fn"]),
        "best_single_model": best_name,
        "best_single_f1": float(singles[best_name]["f1"]),
        "trace": trace,
    }


def optimize_bs_ensemble(
    score_maps: Mapping[str, np.ndarray],
    target: np.ndarray,
    valid: np.ndarray,
    *,
    initial_thresholds: tuple[float, float, float],
    alpha_steps: int = 12,
    threshold_candidates: int = 48,
    threshold_passes: int = 2,
    min_improvement: float = 0.0,
) -> dict[str, object]:
    """Greedy convex blend optimized for the weighted BS competition subscore."""
    scores = _validate_score_maps(score_maps, target, valid)
    truth = np.asarray(target)
    mask = np.asarray(valid, dtype=bool)

    singles: dict[str, dict[str, object]] = {
        name: optimize_ordered_thresholds(
            score,
            truth,
            mask,
            initial=initial_thresholds,
            max_candidates=threshold_candidates,
            passes=threshold_passes,
        )
        for name, score in scores.items()
    }
    best_name = max(
        singles,
        key=lambda name: (
            float(singles[name]["bs_subscore"]),
            float(singles[name]["iou_burn"]),
            float(singles[name]["miou_severity"]),
            name,
        ),
    )

    current_score = scores[best_name].copy()
    current_result = singles[best_name]
    weights = {best_name: 1.0}
    remaining = set(scores) - {best_name}
    trace: list[dict[str, object]] = [
        {
            "step": 0,
            "added": best_name,
            "weights": dict(weights),
            "bs_subscore": float(current_result["bs_subscore"]),
            "thresholds": list(current_result["thresholds"]),
        }
    ]

    alphas = _alpha_grid(alpha_steps)
    while remaining:
        step_best: tuple[str, float, np.ndarray, dict[str, object]] | None = None
        step_key: tuple[float, float, float, float, str] | None = None

        for name in sorted(remaining):
            candidate = scores[name]
            for alpha in alphas:
                fused = (
                    float(alpha) * current_score
                    + (1.0 - float(alpha)) * candidate
                )
                result = optimize_ordered_thresholds(
                    fused,
                    truth,
                    mask,
                    initial=tuple(float(x) for x in current_result["thresholds"]),
                    max_candidates=threshold_candidates,
                    passes=threshold_passes,
                )
                key = (
                    float(result["bs_subscore"]),
                    float(result["iou_burn"]),
                    float(result["miou_severity"]),
                    float(alpha),
                    name,
                )
                if step_key is None or key > step_key:
                    step_key = key
                    step_best = (name, float(alpha), fused, result)

        assert step_best is not None
        name, alpha, fused, result = step_best
        improvement = float(result["bs_subscore"]) - float(current_result["bs_subscore"])
        if improvement <= float(min_improvement):
            break

        weights = {
            model_name: value * alpha
            for model_name, value in weights.items()
        }
        weights[name] = weights.get(name, 0.0) + (1.0 - alpha)
        weights = _renormalize_weights(weights)
        current_score = fused
        current_result = result
        remaining.remove(name)
        trace.append(
            {
                "step": len(trace),
                "added": name,
                "alpha_keep_current": alpha,
                "weights": dict(weights),
                "bs_subscore": float(result["bs_subscore"]),
                "iou_burn": float(result["iou_burn"]),
                "miou_severity": float(result["miou_severity"]),
                "thresholds": list(result["thresholds"]),
                "delta_bs_subscore": improvement,
            }
        )

    return {
        "task": "BS",
        "weights": weights,
        "bs_subscore": float(current_result["bs_subscore"]),
        "iou_burn": float(current_result["iou_burn"]),
        "miou_severity": float(current_result["miou_severity"]),
        "thresholds": tuple(float(x) for x in current_result["thresholds"]),
        "best_single_model": best_name,
        "best_single_bs_subscore": float(singles[best_name]["bs_subscore"]),
        "trace": trace,
    }
