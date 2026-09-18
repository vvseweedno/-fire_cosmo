"""Compare two calibrated OOF experiments with event-level bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.calibration import apply_ordered_thresholds
from wildfire.metadata import read_meta_csv
from wildfire.model_config import load_model_config
from wildfire.oof import load_oof_directory
from wildfire.statistics import GroupedPrediction, paired_group_bootstrap


def _to_grouped_predictions(oof_dir: str, meta_path: str, config_path: str):
    pool = load_oof_directory(oof_dir)
    meta = read_meta_csv(meta_path)
    config = load_model_config(config_path)

    records: list[GroupedPrediction] = []
    for record in pool.records:
        chip_meta = meta.get(record.chip_id)
        if chip_meta is None:
            raise RuntimeError(f"{record.chip_id}: missing from meta.csv")

        if record.task.upper() == "AF":
            prediction = (
                (np.asarray(record.score) >= config.af.threshold)
                & np.asarray(record.valid, dtype=bool)
            ).astype(np.uint8)
        else:
            prediction = apply_ordered_thresholds(
                record.score,
                config.bs.default_thresholds,
                record.valid,
            )

        records.append(
            GroupedPrediction(
                chip_id=record.chip_id,
                group_id=chip_meta.split_group,
                task=record.task,
                prediction=prediction,
                target=np.asarray(record.target),
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-a", required=True)
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--oof-b", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output", default="outputs/oof_ablation_comparison.json")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=99173)
    args = parser.parse_args()

    experiment_a = _to_grouped_predictions(args.oof_a, args.meta, args.config_a)
    experiment_b = _to_grouped_predictions(args.oof_b, args.meta, args.config_b)
    report = paired_group_bootstrap(
        experiment_a,
        experiment_b,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved paired OOF comparison to {output}")


if __name__ == "__main__":
    main()
