"""Run deterministic baselines on labelled chips and save real measured metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from wildfire.baselines import predict
from wildfire.evaluation import CompetitionEvaluator, evaluation_mask
from wildfire.io import discover_chips, infer_task, load_channels


def _chip_metrics(
    task: str,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    evaluator = CompetitionEvaluator()
    if task == "AF":
        evaluator.update_af(pred, target, valid)
        summary = evaluator.summary()
        return {
            "f1": float(summary["f1_af"]),
            "pixels_valid": int(np.count_nonzero(valid)),
        }

    evaluator.update_bs(pred, target, valid)
    summary = evaluator.summary()
    return {
        "iou_burn": float(summary["iou_burn"]),
        "miou_severity": float(summary["miou_severity"]),
        "pixels_valid": int(np.count_nonzero(valid)),
    }


def run(
    data_dir: str | Path,
    ignore_value: float | None = None,
) -> dict[str, object]:
    evaluator = CompetitionEvaluator()
    per_chip: list[dict[str, object]] = []
    missing_target: list[str] = []
    started = perf_counter()

    chips = discover_chips(data_dir)
    for chip in chips:
        channels = load_channels(chip)
        if "TARGET" not in channels:
            missing_target.append(chip.chip_id)
            continue

        task = infer_task(channels)
        target = np.asarray(channels["TARGET"])
        pred_started = perf_counter()
        pred = predict(channels, task)
        latency_ms = (perf_counter() - pred_started) * 1000.0

        if pred.shape != target.shape:
            raise ValueError(
                f"{chip.chip_id}: prediction shape {pred.shape} != target shape {target.shape}"
            )

        valid = evaluation_mask(channels, target, ignore_value)
        if task == "AF":
            evaluator.update_af(pred, target, valid)
        else:
            evaluator.update_bs(pred, target, valid)

        chip_result: dict[str, object] = {
            "chip_id": chip.chip_id,
            "task": task,
            "latency_ms": latency_ms,
        }
        chip_result.update(_chip_metrics(task, pred, target, valid))
        per_chip.append(chip_result)

    elapsed = perf_counter() - started
    summary = evaluator.summary()
    summary.update(
        {
            "discovered_chips": len(chips),
            "evaluated_chips": len(per_chip),
            "missing_target_chips": missing_target,
            "elapsed_seconds": elapsed,
            "mean_latency_ms": (
                float(np.mean([float(row["latency_ms"]) for row in per_chip]))
                if per_chip
                else None
            ),
            "per_chip": per_chip,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/baseline_metrics.json")
    parser.add_argument("--ignore-value", type=float, default=None)
    args = parser.parse_args()

    report = run(args.data_dir, args.ignore_value)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({key: value for key, value in report.items() if key != "per_chip"}, indent=2))
    print(f"Saved full report to {output}")


if __name__ == "__main__":
    main()
