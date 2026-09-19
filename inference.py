"""Competition entry point required by the case specification.

Usage:
    python inference.py --data-dir /path/to/test --output /path/to/submission.csv
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from wildfire.baselines import predict
from wildfire.io import discover_chips, load_channels
from wildfire.metadata import read_meta_csv
from wildfire.model_config import load_model_config
from wildfire.submission import (
    Prediction,
    read_submission_template,
    validate_submission_against_template,
    validate_template_task_contract,
    write_submission_from_template,
)


def run(
    data_dir: str | Path,
    output: str | Path,
    model_config: str | Path = "configs/baseline.json",
) -> int:
    root = Path(data_dir)
    output_path = Path(output)
    template_path = root / "sample_submission.csv"
    meta_path = root / "meta.csv"

    template = read_submission_template(template_path)
    meta = read_meta_csv(meta_path)
    task_contract_errors = validate_template_task_contract(
        template,
        {chip_id: item.kind for chip_id, item in meta.items()},
    )
    if task_contract_errors:
        raise RuntimeError(
            "sample_submission.csv and meta.csv task contract failed: "
            + "; ".join(task_contract_errors)
        )
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

        task = chip_meta.kind.upper()
        try:
            channels = load_channels(chip)
            mask = predict(channels, task, config)
        except Exception as exc:
            files = sorted({str(source.path) for source in chip.channels.values()})
            raise RuntimeError(
                f"{chip_id}: inference failed for task={task}; "
                f"files={files}; {type(exc).__name__}: {exc}"
            ) from exc
        if tuple(mask.shape) != chip_meta.shape:
            raise RuntimeError(
                f"{chip_id}: prediction shape {mask.shape} != meta.csv shape {chip_meta.shape}"
            )
        predictions[chip_id] = Prediction(chip_id=chip_id, task=task, mask=mask)

    # Stage the candidate beside the requested destination.  Validation happens
    # before os.replace(), so a failed run can never clobber a previously valid
    # submission and successful publication is atomic on the destination FS.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    staged_path = Path(staged_name)
    try:
        rows = write_submission_from_template(predictions, template, staged_path)
        if rows != len(template):
            raise RuntimeError(f"Expected {len(template)} submission rows, wrote {rows}")

        submission_errors = validate_submission_against_template(
            staged_path,
            template,
            {chip_id: meta[chip_id].shape for chip_id in required_chip_ids},
            tasks={chip_id: meta[chip_id].kind for chip_id in required_chip_ids},
            expected_row_count=len(template),
        )
        if submission_errors:
            raise RuntimeError(
                "generated submission failed strict validation: "
                + "; ".join(submission_errors)
            )
        os.replace(staged_path, output_path)
    finally:
        staged_path.unlink(missing_ok=True)
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
