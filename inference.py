"""Competition entry point required by the case specification.

Usage:
    python inference.py --data-dir /path/to/test --output /path/to/submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wildfire.baselines import predict
from wildfire.io import discover_chips, load_channels
from wildfire.metadata import read_meta_csv
from wildfire.model_config import load_model_config
from wildfire.submission import (
    Prediction,
    read_submission_template,
    write_submission_from_template,
)


def run(
    data_dir: str | Path,
    output: str | Path,
    model_config: str | Path = "configs/baseline.json",
) -> int:
    root = Path(data_dir)
    template_path = root / "sample_submission.csv"
    meta_path = root / "meta.csv"

    template = read_submission_template(template_path)
    meta = read_meta_csv(meta_path)
    config = load_model_config(model_config)

    discovered = {chip.chip_id: chip for chip in discover_chips(root)}
    required_chip_ids = list(dict.fromkeys(row.chip_id for row in template))

    predictions: dict[str, Prediction] = {}
    for chip_id in required_chip_ids:
        chip_meta = meta.get(chip_id)
        if chip_meta is None:
            raise RuntimeError(f"{chip_id}: absent from meta.csv")
        chip = discovered.get(chip_id)
        if chip is None:
            raise RuntimeError(
                f"{chip_id}: data files were not recognised. "
                "Run scripts/inspect_dataset.py and adapt wildfire/io.py to the official layout."
            )

        channels = load_channels(chip)
        task = chip_meta.kind.upper()
        mask = predict(channels, task, config)
        if tuple(mask.shape) != chip_meta.shape:
            raise RuntimeError(
                f"{chip_id}: prediction shape {mask.shape} != meta.csv shape {chip_meta.shape}"
            )
        predictions[chip_id] = Prediction(chip_id=chip_id, task=task, mask=mask)

    rows = write_submission_from_template(predictions, template, output)
    if rows != len(template):
        raise RuntimeError(f"Expected {len(template)} submission rows, wrote {rows}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/baseline.json")
    args = parser.parse_args()
    rows = run(args.data_dir, args.output, args.model_config)
    print(f"Wrote {rows} template-aligned submission rows to {args.output}")


if __name__ == "__main__":
    main()
