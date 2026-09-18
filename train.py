"""Metric-aware deterministic training entry point."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from wildfire.baselines import active_fire_score
from wildfire.fusion import burn_fusion_components
from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.model_config import ModelConfig, load_model_config, save_model_config
from wildfire.split import read_split_manifest
from wildfire.training import (
    AFTrainingSample,
    BSFusionTrainingSample,
    calibrate_af_threshold_exact,
    calibrate_bs_cloud_sar_fallback,
)


def _selected_chips(data_dir: str | Path, train_chip_ids: set[str]):
    discovered = {chip.chip_id: chip for chip in discover_chips(data_dir)}
    missing = sorted(train_chip_ids - set(discovered))
    if missing:
        raise RuntimeError(
            "Split manifest references chips not discovered in data: "
            + ", ".join(missing[:10])
        )
    return discovered


def _af_samples(
    discovered,
    train_chip_ids: set[str],
    config: ModelConfig,
) -> Iterator[AFTrainingSample]:
    for chip_id in sorted(train_chip_ids):
        chip = discovered[chip_id]
        channels = load_channels(chip)
        if "TARGET" not in channels:
            raise RuntimeError(f"{chip_id}: TARGET is missing")
        if infer_task(channels) != "AF":
            continue
        score, model_valid = active_fire_score(channels, config)
        yield score, np.asarray(channels["TARGET"]), model_valid


def _bs_fusion_samples(
    discovered,
    train_chip_ids: set[str],
) -> Iterator[BSFusionTrainingSample]:
    for chip_id in sorted(train_chip_ids):
        chip = discovered[chip_id]
        channels = load_channels(chip)
        if "TARGET" not in channels:
            raise RuntimeError(f"{chip_id}: TARGET is missing")
        if infer_task(channels) != "BS":
            continue
        yield burn_fusion_components(channels), np.asarray(channels["TARGET"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--output", default="artifacts/baseline_config.json")
    parser.add_argument("--bs-max-candidates", type=int, default=64)
    parser.add_argument("--bs-passes", type=int, default=3)
    args = parser.parse_args()

    manifest = read_split_manifest(args.split_manifest)
    train_chip_ids = set(manifest["train"])
    base_config = load_model_config(args.base_config)
    discovered = _selected_chips(args.data_dir, train_chip_ids)

    af_config, af_result = calibrate_af_threshold_exact(
        _af_samples(discovered, train_chip_ids, base_config),
        base_config,
    )
    final_config, bs_result = calibrate_bs_cloud_sar_fallback(
        _bs_fusion_samples(discovered, train_chip_ids),
        af_config,
        max_candidates=args.bs_max_candidates,
        passes=args.bs_passes,
    )
    save_model_config(final_config, args.output)

    print(
        "AF exact threshold: "
        f"{af_result['threshold']:.6f}; train micro-F1={af_result['f1']:.6f}"
    )
    print(
        "BS cloud-aware calibration: "
        f"cloud_sar_weight={final_config.bs.cloud_sar_weight:.6f}; "
        f"thresholds={tuple(round(float(x), 6) for x in bs_result['thresholds'])}; "
        f"burn-IoU={bs_result['iou_burn']:.6f}; "
        f"severity-mIoU={bs_result['miou_severity']:.6f}; "
        f"weighted BS subscore={bs_result['bs_subscore']:.6f}"
    )
    print(f"Saved calibrated model config to {args.output}")


if __name__ == "__main__":
    main()
