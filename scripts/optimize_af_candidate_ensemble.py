"""Fit deployable AF candidate weights on pooled OOF after cross-fit promotion."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wildfire.af_candidates import AF_CANDIDATE_NAMES
from wildfire.ensemble import optimize_af_ensemble
from wildfire.model_config import load_model_config, save_model_config
from wildfire.oof import OOFRecord, load_oof_directory


def _load_aligned(
    candidate_root: str | Path,
) -> tuple[
    list[str],
    dict[str, dict[str, OOFRecord]],
    dict[str, OOFRecord],
]:
    root = Path(candidate_root)
    pools: dict[str, dict[str, OOFRecord]] = {}

    for name in AF_CANDIDATE_NAMES:
        directory = root / name
        if not directory.exists():
            raise FileNotFoundError(directory)
        pool = load_oof_directory(directory)
        records = {
            record.chip_id: record
            for record in pool.records
            if record.task.upper() == "AF"
        }
        if not records:
            raise RuntimeError(f"{name}: no AF OOF records")
        pools[name] = records

    base_records = pools["BASE"]
    chip_ids = sorted(base_records)
    base_set = set(chip_ids)

    for name, records in pools.items():
        if set(records) != base_set:
            raise RuntimeError(f"{name}: AF OOF chip set differs from BASE")
        for chip_id in chip_ids:
            record = records[chip_id]
            base = base_records[chip_id]
            if not np.array_equal(record.target, base.target):
                raise RuntimeError(f"{name}/{chip_id}: target mismatch")
            if not np.array_equal(record.valid, base.valid):
                raise RuntimeError(f"{name}/{chip_id}: valid-mask mismatch")

    return chip_ids, pools, base_records


def run(
    candidate_root: str | Path,
    *,
    base_config_path: str | Path = "configs/baseline.json",
    output_config: str | Path = "artifacts/af_ensemble_config.json",
    output_report: str | Path = "outputs/af_ensemble_report.json",
    alpha_steps: int = 24,
) -> dict[str, object]:
    chip_ids, pools, base_records = _load_aligned(candidate_root)

    target = np.concatenate(
        [base_records[chip_id].target.ravel() for chip_id in chip_ids]
    )
    valid = np.concatenate(
        [base_records[chip_id].valid.astype(bool).ravel() for chip_id in chip_ids]
    )
    scores = {
        name: np.concatenate(
            [records[chip_id].score.ravel() for chip_id in chip_ids]
        )
        for name, records in pools.items()
    }

    result = optimize_af_ensemble(
        scores,
        target,
        valid,
        alpha_steps=alpha_steps,
    )
    weights = {
        str(name): float(value)
        for name, value in dict(result["weights"]).items()
    }
    threshold = float(result["threshold"])

    base_config = load_model_config(base_config_path)
    config = replace(
        base_config,
        af=replace(
            base_config.af,
            score_weights=weights,
            threshold=threshold,
        ),
    )
    metadata = dict(config.training)
    metadata["af_candidate_ensemble"] = {
        "candidate_root": str(candidate_root),
        "weights": weights,
        "pooled_search": result,
        "warning": (
            "Pooled OOF fitting selects deployment parameters. Promote this "
            "configuration only after cross-fitted validation is positive."
        ),
    }
    config = replace(config, training=metadata)
    save_model_config(config, output_config)

    report = {
        "candidates": list(AF_CANDIDATE_NAMES),
        "chips": len(chip_ids),
        "weights": weights,
        "pooled_f1": result["f1"],
        "best_single_model": result["best_single_model"],
        "best_single_f1": result["best_single_f1"],
        "threshold": threshold,
        "output_config": str(output_config),
        "warning": (
            "Pooled OOF fitting is for deployment parameter selection, not an "
            "unbiased hidden-test estimate."
        ),
    }
    output = Path(output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--output-config", default="artifacts/af_ensemble_config.json")
    parser.add_argument("--output-report", default="outputs/af_ensemble_report.json")
    parser.add_argument("--alpha-steps", type=int, default=24)
    args = parser.parse_args()

    report = run(
        args.candidate_root,
        base_config_path=args.base_config,
        output_config=args.output_config,
        output_report=args.output_report,
        alpha_steps=args.alpha_steps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
