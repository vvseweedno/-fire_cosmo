"""Submission creation and structural validation."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wildfire.constants import AF_CLASS_IDS, BS_CLASS_IDS
from wildfire.rle import decode_binary_mask, encode_class


@dataclass(frozen=True)
class Prediction:
    chip_id: str
    task: str
    mask: np.ndarray


def rows_for_prediction(prediction: Prediction) -> list[dict[str, str | int]]:
    task = prediction.task.upper()
    class_ids = AF_CLASS_IDS if task == "AF" else BS_CLASS_IDS if task == "BS" else ()
    if not class_ids:
        raise ValueError(f"Unknown task: {prediction.task}")
    return [
        {
            "chip_id": prediction.chip_id,
            "class_id": class_id,
            "rle": encode_class(prediction.mask, class_id),
        }
        for class_id in class_ids
    ]


def write_submission(predictions: Iterable[Prediction], output: str | Path) -> int:
    rows: list[dict[str, str | int]] = []
    for prediction in predictions:
        rows.extend(rows_for_prediction(prediction))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chip_id", "class_id", "rle"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def validate_submission(
    submission: str | Path,
    expected: dict[str, tuple[str, tuple[int, int]]],
) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, int]] = set()

    with Path(submission).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["chip_id", "class_id", "rle"]:
            errors.append("Header must be exactly: chip_id,class_id,rle")
            return errors

        for line_no, row in enumerate(reader, start=2):
            chip_id = row.get("chip_id", "")
            if chip_id not in expected:
                errors.append(f"line {line_no}: unknown chip_id {chip_id!r}")
                continue
            task, shape = expected[chip_id]
            try:
                class_id = int(row.get("class_id", ""))
            except ValueError:
                errors.append(f"line {line_no}: invalid class_id")
                continue

            allowed = AF_CLASS_IDS if task == "AF" else BS_CLASS_IDS
            if class_id not in allowed:
                errors.append(f"line {line_no}: class_id {class_id} invalid for {task}")
                continue
            key = (chip_id, class_id)
            if key in seen:
                errors.append(f"line {line_no}: duplicate row {key}")
                continue
            seen.add(key)
            try:
                decode_binary_mask(row.get("rle", ""), shape)
            except (TypeError, ValueError) as exc:
                errors.append(f"line {line_no}: bad RLE: {exc}")

    for chip_id, (task, _) in expected.items():
        class_ids = AF_CLASS_IDS if task == "AF" else BS_CLASS_IDS
        for class_id in class_ids:
            if (chip_id, class_id) not in seen:
                errors.append(f"missing row: {(chip_id, class_id)}")
    return errors
