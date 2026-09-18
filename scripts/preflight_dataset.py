"""Validate organiser-provided dataset structure before inference."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.metadata import read_meta_csv
from wildfire.submission import read_submission_template


REQUIRED_CHANNELS = {
    "AF": {"I4", "I5"},
    "BS": {"B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"},
}


def run(data_dir: str | Path, *, deep: bool = False) -> dict[str, object]:
    root = Path(data_dir)
    meta = read_meta_csv(root / "meta.csv")
    template = read_submission_template(root / "sample_submission.csv")
    chips = {chip.chip_id: chip for chip in discover_chips(root)}
    template_ids = list(dict.fromkeys(row.chip_id for row in template))

    errors: list[str] = []
    warnings: list[str] = []
    task_counts: Counter[str] = Counter()
    optional_presence: Counter[str] = Counter()

    extra = sorted(set(chips) - set(template_ids))
    if extra:
        warnings.append(
            f"{len(extra)} discovered chips are not referenced by sample_submission.csv"
        )

    for chip_id in template_ids:
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
            errors.append(
                f"{chip_id}: meta kind={expected} but channels infer task={inferred}"
            )
            continue

        missing = sorted(REQUIRED_CHANNELS[expected] - available)
        if missing:
            errors.append(f"{chip_id}: missing required channels {missing}")
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
                errors.append(
                    f"{chip_id}: raster shape={shape} but meta.csv shape={item.shape}"
                )

    return {
        "ok": not errors,
        "data_dir": str(root),
        "deep": deep,
        "template_rows": len(template),
        "template_chips": len(template_ids),
        "discovered_chips": len(chips),
        "task_counts": dict(task_counts),
        "optional_channel_presence": dict(sorted(optional_presence.items())),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = run(args.data_dir, deep=args.deep)
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
