"""Validate RLE structure and required rows against discovered test chips."""

from __future__ import annotations

import argparse

from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.submission import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    expected: dict[str, tuple[str, tuple[int, int]]] = {}
    for chip in discover_chips(args.data_dir):
        channels = load_channels(chip)
        task = infer_task(channels)
        first = next(iter(channels.values()))
        expected[chip.chip_id] = (task, tuple(first.shape))

    errors = validate_submission(args.submission, expected)
    if errors:
        print("INVALID SUBMISSION")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print(f"VALID SUBMISSION: {len(expected)} chips")


if __name__ == "__main__":
    main()
