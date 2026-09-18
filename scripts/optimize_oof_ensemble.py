"""Optimize convex AF/BS ensemble weights from aligned OOF score maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.ensemble import optimize_af_ensemble, optimize_bs_ensemble
from wildfire.model_config import load_model_config
from wildfire.oof import load_oof_directory


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must use NAME=/path/to/oof")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--model must use NAME=/path/to/oof")
    return name.strip(), Path(raw_path)


def _aligned_task_arrays(models: list[tuple[str, Path]], task: str):
    loaded = [(name, load_oof_directory(path)) for name, path in models]
    base_name, base_pool = loaded[0]
    base_records = {
        record.chip_id: record
        for record in base_pool.records
        if record.task.upper() == task
    }
    if not base_records:
        raise RuntimeError(f"{base_name}: no {task} OOF records")

    chip_ids = sorted(base_records)
    target_parts = []
    valid_parts = []
    model_parts: dict[str, list[np.ndarray]] = {name: [] for name, _ in loaded}

    for chip_id in chip_ids:
        base = base_records[chip_id]
        target_parts.append(base.target.ravel())
        valid_parts.append(base.valid.ravel())

    for name, pool in loaded:
        records = {
            record.chip_id: record
            for record in pool.records
            if record.task.upper() == task
        }
        if set(records) != set(chip_ids):
            raise RuntimeError(f"{name}: {task} OOF chip set differs from {base_name}")
        for chip_id in chip_ids:
            record = records[chip_id]
            base = base_records[chip_id]
            if not np.array_equal(record.target, base.target):
                raise RuntimeError(f"{name}/{chip_id}: target mismatch")
            if not np.array_equal(record.valid, base.valid):
                raise RuntimeError(f"{name}/{chip_id}: valid-mask mismatch")
            model_parts[name].append(record.score.ravel())

    return (
        {name: np.concatenate(parts) for name, parts in model_parts.items()},
        np.concatenate(target_parts),
        np.concatenate(valid_parts).astype(bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=_parse_model,
        help="Repeat NAME=/path/to/oof for each candidate model.",
    )
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--output", default="outputs/oof_ensemble.json")
    parser.add_argument("--af-alpha-steps", type=int, default=20)
    parser.add_argument("--bs-alpha-steps", type=int, default=12)
    args = parser.parse_args()

    if len(args.model) < 2:
        raise RuntimeError("at least two --model entries are required")

    config = load_model_config(args.base_config)
    af_scores, af_target, af_valid = _aligned_task_arrays(args.model, "AF")
    bs_scores, bs_target, bs_valid = _aligned_task_arrays(args.model, "BS")

    af = optimize_af_ensemble(
        af_scores,
        af_target,
        af_valid,
        alpha_steps=args.af_alpha_steps,
    )
    bs = optimize_bs_ensemble(
        bs_scores,
        bs_target,
        bs_valid,
        initial_thresholds=config.bs.default_thresholds,
        alpha_steps=args.bs_alpha_steps,
    )

    report = {
        "models": [name for name, _ in args.model],
        "AF": af,
        "BS": bs,
        "note": (
            "Weights were optimized only on aligned pooled OOF scores. "
            "Freeze them before hidden-test inference."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
