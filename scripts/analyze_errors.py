"""Analyze saved leakage-safe cross-fit prediction errors."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wildfire.diagnostics import af_error_summary, bs_error_summary
from wildfire.metadata import read_meta_csv


def _load_record(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"prediction", "target"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        prediction = np.asarray(payload["prediction"])
        target = np.asarray(payload["target"])
        valid = (
            np.asarray(payload["valid"], dtype=bool)
            if "valid" in payload.files
            else None
        )
    if prediction.shape != target.shape:
        raise ValueError(f"{path}: prediction/target shapes differ")
    if valid is not None and valid.shape != target.shape:
        raise ValueError(f"{path}: valid/target shapes differ")
    return prediction, target, valid


def _summary(
    task: str,
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    valids: list[np.ndarray],
) -> dict[str, object]:
    pred = np.concatenate([array.ravel() for array in predictions])
    target = np.concatenate([array.ravel() for array in targets])
    valid = np.concatenate([array.ravel() for array in valids])
    if task == "AF":
        return af_error_summary(pred, target, valid)
    return bs_error_summary(pred, target, valid)


def run(
    prediction_dir: str | Path,
    *,
    task: str,
    output: str | Path | None = None,
    meta_csv: str | Path | None = None,
) -> dict[str, object]:
    resolved_task = task.upper()
    if resolved_task not in {"AF", "BS"}:
        raise ValueError("task must be AF or BS")

    root = Path(prediction_dir)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise ValueError(f"no prediction .npz files found in {root}")

    meta = read_meta_csv(meta_csv) if meta_csv is not None else None
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    per_chip: dict[str, object] = {}
    event_arrays: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"prediction": [], "target": [], "valid": []}
    )

    for path in files:
        chip_id = path.stem
        prediction, target, valid = _load_record(path)
        resolved_valid = (
            np.ones(target.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        predictions.append(prediction)
        targets.append(target)
        valids.append(resolved_valid)

        if resolved_task == "AF":
            per_chip[chip_id] = af_error_summary(prediction, target, resolved_valid)
        else:
            per_chip[chip_id] = bs_error_summary(prediction, target, resolved_valid)

        if meta is not None:
            item = meta.get(chip_id)
            if item is None:
                raise ValueError(f"{chip_id}: missing from metadata")
            event_id = item.split_group
            event_arrays[event_id]["prediction"].append(prediction)
            event_arrays[event_id]["target"].append(target)
            event_arrays[event_id]["valid"].append(resolved_valid)

    per_event: dict[str, object] = {}
    for event_id, arrays in sorted(event_arrays.items()):
        per_event[event_id] = _summary(
            resolved_task,
            arrays["prediction"],
            arrays["target"],
            arrays["valid"],
        )

    report = {
        "task": resolved_task,
        "prediction_dir": str(root),
        "chips": len(files),
        "aggregate": _summary(resolved_task, predictions, targets, valids),
        "per_chip": per_chip,
        "per_event": per_event,
        "grouping": "organiser event/group id" if meta is not None else None,
    }

    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--task", choices=("AF", "BS"), required=True)
    parser.add_argument("--meta-csv")
    parser.add_argument(
        "--output",
        default="artifacts/error_analysis/error_analysis.json",
    )
    args = parser.parse_args()

    report = run(
        args.prediction_dir,
        task=args.task,
        output=args.output,
        meta_csv=args.meta_csv,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
