"""Optimize deployable BS candidate weights from aligned pooled OOF records.

The BASE candidate is always included, so the search space contains the current
production score.  The final config stores convex candidate weights and then
recalibrates land-cover-specific severity thresholds on the fused OOF score.

Use cross-fitted evaluation before promoting the result as a measured gain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wildfire.bs_candidates import BS_CANDIDATE_NAMES
from wildfire.ensemble import optimize_bs_ensemble
from wildfire.model_config import load_model_config, save_model_config
from wildfire.oof import OOFRecord, load_oof_directory
from wildfire.training import calibrate_bs_landcover_thresholds


def _load_aligned(
    candidate_root: str | Path,
) -> tuple[
    list[str],
    dict[str, dict[str, OOFRecord]],
    dict[str, OOFRecord],
]:
    root = Path(candidate_root)
    pools: dict[str, dict[str, OOFRecord]] = {}

    for name in BS_CANDIDATE_NAMES:
        directory = root / name
        if not directory.exists():
            raise FileNotFoundError(directory)
        pool = load_oof_directory(directory)
        records = {
            record.chip_id: record
            for record in pool.records
            if record.task.upper() == "BS"
        }
        if not records:
            raise RuntimeError(f"{name}: no BS OOF records")
        pools[name] = records

    base_records = pools["BASE"]
    chip_ids = sorted(base_records)
    base_set = set(chip_ids)

    for name, records in pools.items():
        if set(records) != base_set:
            raise RuntimeError(f"{name}: BS OOF chip set differs from BASE")
        for chip_id in chip_ids:
            record = records[chip_id]
            base = base_records[chip_id]
            if not np.array_equal(record.target, base.target):
                raise RuntimeError(f"{name}/{chip_id}: target mismatch")
            if not np.array_equal(record.valid, base.valid):
                raise RuntimeError(f"{name}/{chip_id}: valid-mask mismatch")

    return chip_ids, pools, base_records


def _pooled_arrays(
    chip_ids: list[str],
    pools: dict[str, dict[str, OOFRecord]],
    base_records: dict[str, OOFRecord],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    target = np.concatenate([base_records[chip_id].target.ravel() for chip_id in chip_ids])
    valid = np.concatenate(
        [base_records[chip_id].valid.astype(bool).ravel() for chip_id in chip_ids]
    )
    scores = {
        name: np.concatenate(
            [records[chip_id].score.ravel() for chip_id in chip_ids]
        )
        for name, records in pools.items()
    }
    return scores, target, valid


def _fused_score(
    chip_id: str,
    pools: dict[str, dict[str, OOFRecord]],
    weights: dict[str, float],
) -> np.ndarray:
    base = pools["BASE"][chip_id]
    valid = np.asarray(base.valid, dtype=bool)
    fused = np.zeros(base.score.shape, dtype=np.float32)
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("ensemble weight sum must be positive")

    for name, weight in weights.items():
        if name not in pools:
            raise ValueError(f"ensemble references unknown candidate {name}")
        score = np.asarray(pools[name][chip_id].score, dtype=np.float32)
        safe = np.where(valid & np.isfinite(score), score, 0.0)
        fused += float(weight / total) * safe
    return fused


def run(
    candidate_root: str | Path,
    *,
    base_config_path: str | Path = "configs/baseline.json",
    output_config: str | Path = "artifacts/bs_ensemble_config.json",
    output_report: str | Path = "outputs/bs_ensemble_report.json",
    alpha_steps: int = 16,
    threshold_candidates: int = 96,
    threshold_passes: int = 3,
    landcover_passes: int = 2,
) -> dict[str, object]:
    chip_ids, pools, base_records = _load_aligned(candidate_root)
    scores, target, valid = _pooled_arrays(chip_ids, pools, base_records)
    base_config = load_model_config(base_config_path)

    ensemble = optimize_bs_ensemble(
        scores,
        target,
        valid,
        initial_thresholds=base_config.bs.default_thresholds,
        alpha_steps=alpha_steps,
        threshold_candidates=threshold_candidates,
        threshold_passes=threshold_passes,
    )
    weights = {
        str(name): float(value)
        for name, value in dict(ensemble["weights"]).items()
    }
    global_thresholds = tuple(float(value) for value in ensemble["thresholds"])

    config = replace(
        base_config,
        bs=replace(
            base_config.bs,
            score_weights=weights,
            default_thresholds=global_thresholds,
            natural_open_thresholds=global_thresholds,
            crop_thresholds=global_thresholds,
            forest_thresholds=global_thresholds,
        ),
    )

    landcover_samples = []
    for chip_id in chip_ids:
        record = base_records[chip_id]
        landcover = (
            np.asarray(record.landcover)
            if record.landcover is not None
            else np.full(record.score.shape, -1, dtype=np.int16)
        )
        landcover_samples.append(
            (
                _fused_score(chip_id, pools, weights),
                np.asarray(record.target),
                np.asarray(record.valid, dtype=bool),
                landcover,
            )
        )

    config, landcover = calibrate_bs_landcover_thresholds(
        landcover_samples,
        config,
        max_candidates=threshold_candidates,
        passes=landcover_passes,
    )
    metadata = dict(config.training)
    metadata["bs_candidate_ensemble"] = {
        "candidate_root": str(candidate_root),
        "weights": weights,
        "pooled_search": ensemble,
        "landcover_refinement": landcover,
        "warning": (
            "This is pooled OOF deployment fitting. Use a fold-wise cross-fitted "
            "comparison before calling the improvement unbiased."
        ),
    }
    config = replace(config, training=metadata)
    save_model_config(config, output_config)

    report = {
        "candidates": list(BS_CANDIDATE_NAMES),
        "chips": len(chip_ids),
        "weights": weights,
        "pooled_bs_subscore_before_landcover": ensemble["bs_subscore"],
        "pooled_iou_burn_before_landcover": ensemble["iou_burn"],
        "pooled_miou_severity_before_landcover": ensemble["miou_severity"],
        "landcover_refinement": landcover,
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
    parser.add_argument("--output-config", default="artifacts/bs_ensemble_config.json")
    parser.add_argument("--output-report", default="outputs/bs_ensemble_report.json")
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument("--threshold-candidates", type=int, default=96)
    parser.add_argument("--threshold-passes", type=int, default=3)
    parser.add_argument("--landcover-passes", type=int, default=2)
    args = parser.parse_args()

    report = run(
        args.candidate_root,
        base_config_path=args.base_config,
        output_config=args.output_config,
        output_report=args.output_report,
        alpha_steps=args.alpha_steps,
        threshold_candidates=args.threshold_candidates,
        threshold_passes=args.threshold_passes,
        landcover_passes=args.landcover_passes,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
