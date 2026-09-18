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
from wildfire.split import read_split_manifest


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
            "pixels_scored": int(np.count_nonzero(valid)),
        }

    evaluator.update_bs(pred, target, valid)
    summary = evaluator.summary()
    return {
        "iou_burn": float(summary["iou_burn"]),
        "miou_severity": float(summary["miou_severity"]),
        "pixels_scored": int(np.count_nonzero(valid)),
    }


def run(
    data_dir: str | Path,
    ignore_value: float | None = None,
    *,
    use_valid_mask: bool = False,
    selected_chip_ids: set[str] | None = None,
) -> dict[str, object]:
    evaluator = CompetitionEvaluator()
    per_chip: list[dict[str, object]] = []
    missing_target: list[str] = []
    started = perf_counter()

    chips = discover_chips(data_dir)
    if selected_chip_ids is not None:
        chips = [chip for chip in chips if chip.chip_id in selected_chip_ids]
        discovered_ids = {chip.chip_id for chip in chips}
        missing_from_data = sorted(selected_chip_ids - discovered_ids)
        if missing_from_data:
            raise RuntimeError(
                "Split manifest references chips not discovered in data: "
                + ", ".join(missing_from_data[:10])
            )

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

        valid = evaluation_mask(
            channels,
            target,
            ignore_value,
            use_valid_mask=use_valid_mask,
        )
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
            "selected_chips": len(chips),
            "evaluated_chips": len(per_chip),
            "missing_target_chips": missing_target,
            "elapsed_seconds": elapsed,
            "mean_latency_ms": (
                float(np.mean([float(row["latency_ms"]) for row in per_chip]))
                if per_chip
                else None
            ),
            "evaluation_policy": {
                "micro_averaging": True,
                "use_valid_mask": use_valid_mask,
                "ignore_value": ignore_value,
            },
            "per_chip": per_chip,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/baseline_metrics.json")
    parser.add_argument("--ignore-value", type=float, default=None)
    parser.add_argument("--split-manifest")
    parser.add_argument(
        "--partition",
        choices=("train", "validation"),
        default="validation",
        help="Partition to evaluate when --split-manifest is provided.",
    )
    parser.add_argument(
        "--use-valid-mask",
        action="store_true",
        help="Research ablation only; official score pools all target pixels.",
    )
    args = parser.parse_args()

    selected: set[str] | None = None
    if args.split_manifest:
        manifest = read_split_manifest(args.split_manifest)
        selected = set(manifest[args.partition])

    report = run(
        args.data_dir,
        args.ignore_value,
        use_valid_mask=args.use_valid_mask,
        selected_chip_ids=selected,
    )
    report["partition"] = args.partition if args.split_manifest else "all"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({key: value for key, value in report.items() if key != "per_chip"}, indent=2))
    print(f"Saved full report to {output}")


if __name__ == "__main__":
    main()
