"""Reproducible training entry point.

The first-stage trainer deliberately calibrates only the AF decision threshold on
the TRAIN partition. Validation stays untouched for model selection/reporting.

Example:
    python train.py \
      --data-dir /path/to/train \
      --split-manifest splits/seed42.json \
      --output artifacts/baseline_config.json
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from wildfire.baselines import active_fire_score
from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.model_config import load_model_config, save_model_config
from wildfire.split import read_split_manifest
from wildfire.training import AFTrainingSample, calibrate_af_threshold, threshold_grid


def _af_samples(
    data_dir: str | Path,
    train_chip_ids: set[str],
    config_path: str | Path,
) -> Iterator[AFTrainingSample]:
    config = load_model_config(config_path)
    discovered = {chip.chip_id: chip for chip in discover_chips(data_dir)}
    missing = sorted(train_chip_ids - set(discovered))
    if missing:
        raise RuntimeError(
            "Split manifest references chips not discovered in data: "
            + ", ".join(missing[:10])
        )

    for chip_id in sorted(train_chip_ids):
        chip = discovered[chip_id]
        channels = load_channels(chip)
        if "TARGET" not in channels:
            raise RuntimeError(f"{chip_id}: TARGET is missing")
        if infer_task(channels) != "AF":
            continue
        score, model_valid = active_fire_score(channels, config)
        target = np.asarray(channels["TARGET"])
        yield score, target, model_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--output", default="artifacts/baseline_config.json")
    parser.add_argument("--af-threshold-min", type=float, default=2.0)
    parser.add_argument("--af-threshold-max", type=float, default=8.0)
    parser.add_argument("--af-threshold-step", type=float, default=0.25)
    args = parser.parse_args()

    manifest = read_split_manifest(args.split_manifest)
    train_chip_ids = set(manifest["train"])
    base_config = load_model_config(args.base_config)
    thresholds = threshold_grid(
        args.af_threshold_min,
        args.af_threshold_max,
        args.af_threshold_step,
    )
    calibrated, trace = calibrate_af_threshold(
        _af_samples(args.data_dir, train_chip_ids, args.base_config),
        thresholds,
        base_config,
    )
    save_model_config(calibrated, args.output)

    best = calibrated.training["af_threshold_calibration"]
    print(
        "AF threshold calibrated on train partition: "
        f"threshold={best['best_threshold']}, micro-F1={best['best_f1_train']:.6f}, "
        f"candidates={len(trace)}"
    )
    print(f"Saved model config to {args.output}")


if __name__ == "__main__":
    main()
