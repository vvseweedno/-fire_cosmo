"""Measure prediction/error diversity across saved candidate predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wildfire.diagnostics import candidate_diversity
from wildfire.metadata import read_meta_csv


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate must use NAME=DIR")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("candidate name must not be empty")
    return name, Path(raw_path)


def _load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"prediction", "target"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        prediction = np.asarray(payload["prediction"])
        target = np.asarray(payload["target"])
    if prediction.shape != target.shape:
        raise ValueError(f"{path}: prediction/target shapes differ")
    return prediction, target


def run(
    candidates: dict[str, Path],
    *,
    output: str | Path | None = None,
    meta_csv: str | Path | None = None,
) -> dict[str, object]:
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")

    file_sets = {
        name: {path.name: path for path in sorted(root.glob("*.npz"))}
        for name, root in candidates.items()
    }
    if any(not files for files in file_sets.values()):
        empty = sorted(name for name, files in file_sets.items() if not files)
        raise ValueError(f"candidate directories contain no .npz files: {empty}")

    reference_name = sorted(file_sets)[0]
    reference_files = set(file_sets[reference_name])
    for name, files in file_sets.items():
        if set(files) != reference_files:
            raise ValueError(f"{name}: prediction chip set differs from {reference_name}")

    meta = read_meta_csv(meta_csv) if meta_csv is not None else None
    concatenated: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    truth_parts: list[np.ndarray] = []
    per_event_errors: dict[str, dict[str, int]] = defaultdict(dict)

    for filename in sorted(reference_files):
        reference_target: np.ndarray | None = None
        chip_predictions: dict[str, np.ndarray] = {}

        for name in sorted(candidates):
            prediction, target = _load_prediction(file_sets[name][filename])
            if reference_target is None:
                reference_target = target
            elif not np.array_equal(target, reference_target):
                raise ValueError(f"{filename}: target mismatch between candidates")
            chip_predictions[name] = prediction
            concatenated[name].append(prediction)

        assert reference_target is not None
        truth_parts.append(reference_target)

        if meta is not None:
            chip_id = Path(filename).stem
            item = meta.get(chip_id)
            if item is None:
                raise ValueError(f"{chip_id}: missing from metadata")
            group_id = item.split_group
            for name, prediction in chip_predictions.items():
                errors = int(np.count_nonzero(prediction != reference_target))
                per_event_errors[group_id][name] = (
                    per_event_errors[group_id].get(name, 0) + errors
                )

    target = np.concatenate([array.ravel() for array in truth_parts])
    prediction_arrays = {
        name: np.concatenate([array.ravel() for array in parts])
        for name, parts in concatenated.items()
    }
    report = candidate_diversity(prediction_arrays, target)

    if meta is not None:
        event_wins: list[dict[str, object]] = []
        names = sorted(candidates)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                left_wins = 0
                right_wins = 0
                ties = 0
                for errors in per_event_errors.values():
                    left_error = errors[left]
                    right_error = errors[right]
                    if left_error < right_error:
                        left_wins += 1
                    elif right_error < left_error:
                        right_wins += 1
                    else:
                        ties += 1
                event_wins.append(
                    {
                        "left": left,
                        "right": right,
                        "left_event_wins": left_wins,
                        "right_event_wins": right_wins,
                        "ties": ties,
                    }
                )
        report["event_level_complementary_wins"] = event_wins
        report["events"] = len(per_event_errors)

    report["candidate_dirs"] = {
        name: str(path) for name, path in sorted(candidates.items())
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
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate prediction directory as NAME=DIR; repeat at least twice.",
    )
    parser.add_argument("--meta-csv")
    parser.add_argument(
        "--output",
        default="artifacts/candidate_diversity.json",
    )
    args = parser.parse_args()

    parsed = dict(_parse_candidate(value) for value in args.candidate)
    if len(parsed) != len(args.candidate):
        raise ValueError("candidate names must be unique")
    report = run(parsed, output=args.output, meta_csv=args.meta_csv)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
