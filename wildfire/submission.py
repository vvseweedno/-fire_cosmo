"""Submission creation and strict validation against sample_submission.csv."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
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


@dataclass(frozen=True)
class TemplateRow:
    chip_id: str
    class_id: int


def read_submission_template(path: str | Path) -> list[TemplateRow]:
    """Read the organiser-provided sample_submission.csv preserving row order."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    rows: list[TemplateRow] = []
    seen: set[tuple[str, int]] = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if fields != ["chip_id", "class_id", "rle"]:
            raise ValueError(
                "sample_submission.csv header must be exactly chip_id,class_id,rle"
            )
        for line_no, row in enumerate(reader, start=2):
            chip_id = (row.get("chip_id") or "").strip()
            if not chip_id:
                raise ValueError(f"sample_submission.csv line {line_no}: empty chip_id")
            try:
                class_id = int(row.get("class_id") or "")
            except ValueError as exc:
                raise ValueError(
                    f"sample_submission.csv line {line_no}: invalid class_id"
                ) from exc
            key = (chip_id, class_id)
            if key in seen:
                raise ValueError(f"sample_submission.csv line {line_no}: duplicate {key}")
            seen.add(key)
            rows.append(TemplateRow(chip_id=chip_id, class_id=class_id))
    return rows


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


def _validate_csv_atom(value: str, field: str) -> None:
    if any(token in value for token in (",", "\n", "\r", '"')):
        raise ValueError(f"{field} contains unsupported CSV characters: {value!r}")


def _write_rows(rows: Iterable[dict[str, str | int]], output: str | Path) -> int:
    """Write exact organiser-style CSV with the RLE field always double-quoted."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("chip_id,class_id,rle\n")
        for row in rows:
            chip_id = str(row["chip_id"])
            class_id = int(row["class_id"])
            rle = str(row["rle"])
            _validate_csv_atom(chip_id, "chip_id")
            _validate_csv_atom(rle, "rle")
            handle.write(f'{chip_id},{class_id},"{rle}"\n')
            count += 1
    return count


def write_submission(predictions: Iterable[Prediction], output: str | Path) -> int:
    rows: list[dict[str, str | int]] = []
    for prediction in predictions:
        rows.extend(rows_for_prediction(prediction))
    return _write_rows(rows, output)


def write_submission_from_template(
    predictions: Mapping[str, Prediction],
    template: Iterable[TemplateRow],
    output: str | Path,
) -> int:
    """Write exactly the organiser-provided (chip_id, class_id) rows and order."""
    rows: list[dict[str, str | int]] = []
    for template_row in template:
        prediction = predictions.get(template_row.chip_id)
        if prediction is None:
            raise ValueError(f"Missing prediction for chip {template_row.chip_id}")

        task = prediction.task.upper()
        allowed = AF_CLASS_IDS if task == "AF" else BS_CLASS_IDS if task == "BS" else ()
        if template_row.class_id not in allowed:
            raise ValueError(
                f"Template class {template_row.class_id} is invalid for "
                f"{template_row.chip_id} task {task}"
            )
        rows.append(
            {
                "chip_id": template_row.chip_id,
                "class_id": template_row.class_id,
                "rle": encode_class(prediction.mask, template_row.class_id),
            }
        )
    return _write_rows(rows, output)


def validate_submission(
    submission: str | Path,
    expected: dict[str, tuple[str, tuple[int, int]]],
) -> list[str]:
    """Backward-compatible structural validator when no sample template is available."""
    template: list[TemplateRow] = []
    shapes: dict[str, tuple[int, int]] = {}
    for chip_id, (task, shape) in expected.items():
        class_ids = AF_CLASS_IDS if task == "AF" else BS_CLASS_IDS
        template.extend(TemplateRow(chip_id, class_id) for class_id in class_ids)
        shapes[chip_id] = shape
    return validate_submission_against_template(submission, template, shapes)


def validate_submission_against_template(
    submission: str | Path,
    template: Iterable[TemplateRow],
    shapes: Mapping[str, tuple[int, int]],
) -> list[str]:
    """Validate exact pair set, strict RLE syntax, bounds, and class non-overlap."""
    source = Path(submission)
    errors: list[str] = []
    expected_rows = list(template)
    expected_pairs = {(row.chip_id, row.class_id) for row in expected_rows}

    raw_lines = source.read_text(encoding="utf-8-sig").splitlines()
    if not raw_lines or raw_lines[0] != "chip_id,class_id,rle":
        return ["Header must be exactly: chip_id,class_id,rle"]

    exact_row_pattern = re.compile(r'^[^,\r\n"]+,\d+,"[^"]*"$')
    for line_no, line in enumerate(raw_lines[1:], start=2):
        if not exact_row_pattern.fullmatch(line):
            errors.append(
                f"line {line_no}: row must be chip_id,class_id,\"rle\" with quoted RLE"
            )

    seen: set[tuple[str, int]] = set()
    decoded_by_chip: dict[str, list[np.ndarray]] = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=2):
            chip_id = row.get("chip_id", "")
            try:
                class_id = int(row.get("class_id", ""))
            except ValueError:
                errors.append(f"line {line_no}: invalid class_id")
                continue

            key = (chip_id, class_id)
            if key not in expected_pairs:
                errors.append(f"line {line_no}: unexpected row {key}")
                continue
            if key in seen:
                errors.append(f"line {line_no}: duplicate row {key}")
                continue
            seen.add(key)

            shape = shapes.get(chip_id)
            if shape is None:
                errors.append(f"line {line_no}: missing shape metadata for {chip_id}")
                continue
            try:
                decoded = decode_binary_mask(row.get("rle", ""), shape)
            except (TypeError, ValueError) as exc:
                errors.append(f"line {line_no}: bad RLE: {exc}")
                continue
            decoded_by_chip.setdefault(chip_id, []).append(decoded)

    missing = expected_pairs - seen
    for key in sorted(missing):
        errors.append(f"missing row: {key}")

    if len(seen) != len(expected_rows):
        errors.append(
            f"row count mismatch: expected {len(expected_rows)}, got {len(seen)} unique rows"
        )

    for chip_id, masks in decoded_by_chip.items():
        if len(masks) < 2:
            continue
        occupied = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            if np.any((occupied > 0) & (mask > 0)):
                errors.append(f"{chip_id}: class masks overlap")
                break
            occupied += mask.astype(np.uint8)

    return errors
