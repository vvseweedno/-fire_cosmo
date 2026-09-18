"""Streaming dataset-level evaluation for the official competition metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_bool_mask(valid: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if valid is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"Valid mask shape {mask.shape} differs from target shape {shape}")
    return mask


@dataclass
class BinaryAccumulator:
    """Accumulate pixel-level binary confusion counts across chips."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        p = np.asarray(pred).astype(bool)
        t = np.asarray(target).astype(bool)
        if p.shape != t.shape:
            raise ValueError(f"Prediction shape {p.shape} differs from target shape {t.shape}")
        mask = _as_bool_mask(valid, t.shape)
        self.tp += int(np.count_nonzero(p & t & mask))
        self.fp += int(np.count_nonzero(p & ~t & mask))
        self.fn += int(np.count_nonzero(~p & t & mask))

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 1.0 if denom == 0 else (2.0 * self.tp) / denom

    @property
    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return 1.0 if denom == 0 else self.tp / denom

    def as_dict(self) -> dict[str, int | float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "f1": self.f1,
            "iou": self.iou,
        }


@dataclass
class SeverityAccumulator:
    """Accumulate class-wise intersections/unions for severity classes 1/2/3."""

    classes: tuple[int, ...] = (1, 2, 3)
    intersections: dict[int, int] = field(default_factory=dict)
    unions: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for class_id in self.classes:
            self.intersections.setdefault(class_id, 0)
            self.unions.setdefault(class_id, 0)

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        p = np.asarray(pred)
        t = np.asarray(target)
        if p.shape != t.shape:
            raise ValueError(f"Prediction shape {p.shape} differs from target shape {t.shape}")
        mask = _as_bool_mask(valid, t.shape)

        for class_id in self.classes:
            p_c = (p == class_id) & mask
            t_c = (t == class_id) & mask
            self.intersections[class_id] += int(np.count_nonzero(p_c & t_c))
            self.unions[class_id] += int(np.count_nonzero(p_c | t_c))

    def iou_by_class(self) -> dict[int, float | None]:
        result: dict[int, float | None] = {}
        for class_id in self.classes:
            union = self.unions[class_id]
            result[class_id] = None if union == 0 else self.intersections[class_id] / union
        return result

    @property
    def miou(self) -> float:
        values = [value for value in self.iou_by_class().values() if value is not None]
        return float(np.mean(values)) if values else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "miou": self.miou,
            "iou_by_class": {
                str(class_id): value for class_id, value in self.iou_by_class().items()
            },
            "intersection_by_class": {
                str(class_id): self.intersections[class_id] for class_id in self.classes
            },
            "union_by_class": {
                str(class_id): self.unions[class_id] for class_id in self.classes
            },
        }


@dataclass
class CompetitionEvaluator:
    """Dataset-level evaluator matching the AF/BS score composition."""

    af: BinaryAccumulator = field(default_factory=BinaryAccumulator)
    burn: BinaryAccumulator = field(default_factory=BinaryAccumulator)
    severity: SeverityAccumulator = field(default_factory=SeverityAccumulator)
    af_chips: int = 0
    bs_chips: int = 0

    def update_af(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        self.af.update(pred, target, valid)
        self.af_chips += 1

    def update_bs(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        self.burn.update(np.asarray(pred) > 0, np.asarray(target) > 0, valid)
        self.severity.update(pred, target, valid)
        self.bs_chips += 1

    def summary(self) -> dict[str, object]:
        f1_af = self.af.f1 if self.af_chips else None
        iou_burn = self.burn.iou if self.bs_chips else None
        miou_severity = self.severity.miou if self.bs_chips else None

        score = None
        if f1_af is not None and iou_burn is not None and miou_severity is not None:
            score = 0.35 * f1_af + 0.35 * iou_burn + 0.30 * miou_severity

        return {
            "chips": {"AF": self.af_chips, "BS": self.bs_chips},
            "f1_af": f1_af,
            "iou_burn": iou_burn,
            "miou_severity": miou_severity,
            "score": score,
            "af_counts": self.af.as_dict(),
            "burn_counts": self.burn.as_dict(),
            "severity": self.severity.as_dict(),
        }


def evaluation_mask(
    channels: dict[str, np.ndarray],
    target: np.ndarray,
    ignore_value: float | None = None,
) -> np.ndarray:
    """Build an explicit evaluation mask without inventing undocumented nodata rules."""
    mask = np.isfinite(np.asarray(target))
    if "VALID_MASK" in channels:
        valid = np.asarray(channels["VALID_MASK"]) > 0
        if valid.shape != mask.shape:
            raise ValueError("VALID_MASK shape differs from TARGET")
        mask &= valid
    if ignore_value is not None:
        mask &= np.asarray(target) != ignore_value
    return mask
