"""Inspect the official dataset before hard-coding its layout."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from wildfire.io import discover_chips, infer_task, load_channels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    chips = discover_chips(Path(args.data_dir))
    print(f"recognised chips: {len(chips)}")
    tasks: Counter[str] = Counter()

    for chip in chips:
        channels = load_channels(chip)
        try:
            task = infer_task(channels)
        except ValueError:
            task = "UNKNOWN"
        tasks[task] += 1
        shapes = {name: tuple(array.shape) for name, array in channels.items()}
        print(f"{chip.chip_id}: task={task} channels={sorted(channels)} shapes={shapes}")

    print("task counts:", dict(tasks))


if __name__ == "__main__":
    main()
