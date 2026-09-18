"""Competition entry point.

Required usage:
    python inference.py --data-dir /path/to/test --output submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wildfire.baselines import predict
from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.submission import Prediction, write_submission


def run(data_dir: str | Path, output: str | Path) -> int:
    chips = discover_chips(data_dir)
    if not chips:
        raise RuntimeError(
            "No recognised chips found. Run scripts/inspect_dataset.py and update aliases if needed."
        )

    predictions: list[Prediction] = []
    for chip in chips:
        channels = load_channels(chip)
        task = infer_task(channels)
        mask = predict(channels, task)
        predictions.append(Prediction(chip_id=chip.chip_id, task=task, mask=mask))

    return write_submission(predictions, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = run(args.data_dir, args.output)
    print(f"Wrote {rows} submission rows to {args.output}")


if __name__ == "__main__":
    main()
