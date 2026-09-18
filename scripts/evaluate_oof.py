"""Calibrate thresholds and compute the official score on pooled OOF pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.model_config import load_model_config, save_model_config
from wildfire.oof import load_oof_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", required=True)
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--report", default="outputs/oof_report.json")
    parser.add_argument("--calibrated-config", default="artifacts/oof_config.json")
    parser.add_argument("--bs-max-candidates", type=int, default=64)
    parser.add_argument("--bs-passes", type=int, default=3)
    args = parser.parse_args()

    pool = load_oof_directory(args.oof_dir)
    base_config = load_model_config(args.base_config)
    calibrated, report = pool.calibrate_and_evaluate(
        base_config,
        bs_max_candidates=args.bs_max_candidates,
        bs_passes=args.bs_passes,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_model_config(calibrated, args.calibrated_config)

    print(
        json.dumps(
            {
                "f1_af": report["f1_af"],
                "iou_burn": report["iou_burn"],
                "miou_severity": report["miou_severity"],
                "score": report["score"],
                "oof_chips": report["oof_chips"],
            },
            indent=2,
        )
    )
    print(f"Saved OOF report to {args.report}")
    print(f"Saved OOF-calibrated config to {args.calibrated_config}")


if __name__ == "__main__":
    main()
