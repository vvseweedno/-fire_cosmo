"""Validate organiser-provided dataset structure before train/inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from wildfire.constants import AF_CLASS_IDS, BS_CLASS_IDS
from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.metadata import read_meta_csv
from wildfire.submission import read_submission_template, validate_template_task_contract

REQUIRED_CHANNELS = {
    "AF": {"I1", "I2", "I3", "I4", "I5"},
    "BS": {"B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"},
}


def run(
    data_dir: str | Path,
    *,
    mode: str = "test",
    deep: bool = False,
) -> dict[str, object]:
    root = Path(data_dir)
    resolved_mode = mode.lower()
    if resolved_mode not in {"train", "test"}:
        raise ValueError("mode must be 'train' or 'test'")

    meta = read_meta_csv(root / "meta.csv")
    chips = {chip.chip_id: chip for chip in discover_chips(root)}

    template_rows = 0
    if resolved_mode == "test":
        template = read_submission_template(root / "sample_submission.csv")
        required_ids = list(dict.fromkeys(row.chip_id for row in template))
        template_rows = len(template)
    else:
        required_ids = sorted(meta)

    errors: list[str] = []
    warnings: list[str] = []
    task_counts: Counter[str] = Counter()
    optional_presence: Counter[str] = Counter()

    if resolved_mode == "test":
        errors.extend(
            validate_template_task_contract(
                template,
                {chip_id: item.kind for chip_id, item in meta.items()},
            )
        )
        untemplated = sorted(set(meta) - set(required_ids))
        if untemplated:
            errors.append(
                "meta.csv contains chips absent from sample_submission.csv: "
                + ", ".join(untemplated[:10])
            )

    extra = sorted(set(chips) - set(required_ids))
    if extra:
        warnings.append(f"{len(extra)} discovered chips are not required by {resolved_mode} metadata")

    for chip_id in required_ids:
        item = meta.get(chip_id)
        chip = chips.get(chip_id)
        if item is None:
            errors.append(f"{chip_id}: missing from meta.csv")
            continue
        if chip is None:
            errors.append(f"{chip_id}: raster channels were not discovered")
            continue

        available = set(chip.channels)
        try:
            inferred = infer_task(chip.channels)
        except ValueError as exc:
            errors.append(f"{chip_id}: {exc}")
            continue

        expected = item.kind.upper()
        if inferred != expected:
            errors.append(f"{chip_id}: meta kind={expected} but channels infer task={inferred}")
            continue

        missing = sorted(REQUIRED_CHANNELS[expected] - available)
        if missing:
            errors.append(f"{chip_id}: missing required channels {missing}")
            continue
        if resolved_mode == "train" and "TARGET" not in available:
            errors.append(f"{chip_id}: training chip is missing TARGET")
            continue

        task_counts[expected] += 1
        for channel in available:
            if channel not in REQUIRED_CHANNELS[expected]:
                optional_presence[channel] += 1

        if deep:
            try:
                channels = load_channels(chip)
            except Exception as exc:
                errors.append(f"{chip_id}: channel load failed: {type(exc).__name__}: {exc}")
                continue

            shapes = {name: tuple(array.shape) for name, array in channels.items()}
            if not shapes:
                errors.append(f"{chip_id}: no channels loaded")
                continue
            unique_shapes = set(shapes.values())
            if len(unique_shapes) != 1:
                errors.append(f"{chip_id}: channel shapes differ: {shapes}")
                continue
            shape = next(iter(unique_shapes))
            if shape != item.shape:
                errors.append(f"{chip_id}: raster shape={shape} but meta.csv shape={item.shape}")

            for name, array in channels.items():
                values = np.asarray(array)
                if values.ndim != 2:
                    errors.append(
                        f"{chip_id}: channel {name} must be a 2-D raster, got {values.shape}"
                    )
                    continue
                if not np.issubdtype(values.dtype, np.number):
                    errors.append(
                        f"{chip_id}: channel {name} has non-numeric dtype {values.dtype}"
                    )
                    continue
                finite = np.isfinite(values)
                if not np.any(finite):
                    errors.append(f"{chip_id}: channel {name} contains no finite pixels")
                    continue
                missing = int(values.size - np.count_nonzero(finite))
                if missing:
                    warnings.append(
                        f"{chip_id}: channel {name} has {missing}/{values.size} non-finite pixels"
                    )

                if name == "TARGET" and resolved_mode == "train":
                    valid = values[finite]
                    allowed = (0, *AF_CLASS_IDS) if expected == "AF" else (0, *BS_CLASS_IDS)
                    integral = np.equal(valid, np.rint(valid))
                    observed = set(np.unique(valid[integral].astype(np.int64)).tolist())
                    if not np.all(integral) or not observed <= set(allowed):
                        errors.append(
                            f"{chip_id}: TARGET labels must be integer classes {list(allowed)}, "
                            f"observed {sorted(observed)}"
                        )

    return {
        "ok": not errors,
        "mode": resolved_mode,
        "data_dir": str(root),
        "deep": deep,
        "template_rows": template_rows,
        "required_chips": len(required_ids),
        "discovered_chips": len(chips),
        "task_counts": dict(task_counts),
        "optional_channel_presence": dict(sorted(optional_presence.items())),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = run(args.data_dir, mode=args.mode, deep=args.deep)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
