"""Pooled out-of-fold calibration and evaluation.

The official metrics are micro-style pixel metrics, so model selection must use
all OOF pixels pooled together rather than averaging per-fold F1/IoU values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from wildfire.calibration import (
    apply_ordered_thresholds,
    exact_f1_threshold,
    optimize_ordered_thresholds,
)
from wildfire.evaluation import CompetitionEvaluator
from wildfire.model_config import ModelConfig


@dataclass(frozen=True)
class OOFRecord:
    chip_id: str
    task: str
    score: np.ndarray
    target: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        task = self.task.upper()
        if task not in {"AF", "BS"}:
            raise ValueError(f"unsupported OOF task: {self.task}")
        if not self.chip_id:
            raise ValueError("chip_id must not be empty")
        if self.score.shape != self.target.shape or self.score.shape != self.valid.shape:
            raise ValueError(f"{self.chip_id}: score/target/valid shapes differ")


class OOFPool:
    """Collect each validation chip exactly once and evaluate it globally."""

    def __init__(self) -> None:
        self._records: dict[str, OOFRecord] = {}

    def add(self, record: OOFRecord) -> None:
        if record.chip_id in self._records:
            raise ValueError(f"duplicate OOF chip: {record.chip_id}")
        self._records[record.chip_id] = record

    @property
    def records(self) -> tuple[OOFRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def validate_expected(self, expected_chip_ids: set[str]) -> None:
        actual = set(self._records)
        missing = sorted(expected_chip_ids - actual)
        unexpected = sorted(actual - expected_chip_ids)
        if missing or unexpected:
            raise ValueError(
                f"OOF chip coverage mismatch; missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    @staticmethod
    def _concat(
        records: tuple[OOFRecord, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not records:
            raise ValueError("OOF record set is empty")
        return (
            np.concatenate([record.score.ravel() for record in records]),
            np.concatenate([record.target.ravel() for record in records]),
            np.concatenate([record.valid.astype(bool).ravel() for record in records]),
        )

    def calibrate_and_evaluate(
        self,
        base_config: ModelConfig,
        *,
        bs_max_candidates: int = 64,
        bs_passes: int = 3,
    ) -> tuple[ModelConfig, dict[str, object]]:
        af_records = tuple(record for record in self.records if record.task.upper() == "AF")
        bs_records = tuple(record for record in self.records if record.task.upper() == "BS")
        if not af_records or not bs_records:
            raise ValueError("pooled OOF evaluation requires both AF and BS records")

        af_scores, af_target, af_valid = self._concat(af_records)
        af_result = exact_f1_threshold(af_scores, af_target, af_valid)

        bs_scores, bs_target, bs_valid = self._concat(bs_records)
        bs_result = optimize_ordered_thresholds(
            bs_scores,
            bs_target,
            bs_valid,
            initial=base_config.bs.default_thresholds,
            max_candidates=bs_max_candidates,
            passes=bs_passes,
        )
        bs_thresholds = tuple(float(value) for value in bs_result["thresholds"])

        evaluator = CompetitionEvaluator()
        af_threshold = float(af_result["threshold"])
        for record in af_records:
            prediction = (
                (np.asarray(record.score) >= af_threshold)
                & np.asarray(record.valid, dtype=bool)
            ).astype(np.uint8)
            evaluator.update_af(prediction, record.target)

        for record in bs_records:
            prediction = apply_ordered_thresholds(
                record.score,
                bs_thresholds,
                record.valid,
            )
            evaluator.update_bs(prediction, record.target)

        metadata = dict(base_config.training)
        metadata["oof_calibration"] = {
            "chips": len(self._records),
            "af_chips": len(af_records),
            "bs_chips": len(bs_records),
            "af": af_result,
            "bs": bs_result,
        }
        calibrated = replace(
            base_config,
            af=replace(base_config.af, threshold=af_threshold),
            bs=replace(
                base_config.bs,
                default_thresholds=bs_thresholds,
                natural_open_thresholds=bs_thresholds,
                crop_thresholds=bs_thresholds,
                forest_thresholds=bs_thresholds,
            ),
            training=metadata,
        )

        report = evaluator.summary()
        report.update(
            {
                "calibration": {
                    "af": af_result,
                    "bs": bs_result,
                },
                "oof_chips": {
                    "total": len(self._records),
                    "AF": len(af_records),
                    "BS": len(bs_records),
                },
            }
        )
        return calibrated, report


def save_oof_record(record: OOFRecord, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        chip_id=np.asarray(record.chip_id),
        task=np.asarray(record.task.upper()),
        score=np.asarray(record.score, dtype=np.float32),
        target=np.asarray(record.target),
        valid=np.asarray(record.valid, dtype=np.uint8),
    )


def load_oof_record(path: str | Path) -> OOFRecord:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {"chip_id", "task", "score", "target", "valid"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{source}: missing OOF arrays {sorted(missing)}")
        chip_id = str(payload["chip_id"].item())
        task = str(payload["task"].item())
        return OOFRecord(
            chip_id=chip_id,
            task=task,
            score=np.asarray(payload["score"], dtype=np.float32),
            target=np.asarray(payload["target"]),
            valid=np.asarray(payload["valid"], dtype=bool),
        )


def load_oof_directory(path: str | Path) -> OOFPool:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise ValueError(f"no OOF .npz records found in {root}")

    pool = OOFPool()
    for file_path in files:
        pool.add(load_oof_record(file_path))
    return pool
