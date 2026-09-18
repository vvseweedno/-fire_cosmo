"""Cross-fitted OOF evaluation plus final pooled deployment calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.crossfit import crossfit_calibrate_and_evaluate
from wildfire.model_config import load_model_config, save_model_config
from wildfire.oof import load_oof_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--report", default="outputs/crossfit_oof_report.json")
    parser.add_argument(
        "--deployment-config",
        default="artifacts/crossfit_deployment_config.json",
    )
    parser.add_argument("--bs-max-candidates", type=int, default=64)
    parser.add_argument("--bs-passes", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("fold manifest root must be an object")

    pool = load_oof_directory(args.oof_dir)
    base_config = load_model_config(args.base_config)
    deployment_config, report = crossfit_calibrate_and_evaluate(
        pool,
        manifest,
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
    save_model_config(deployment_config, args.deployment_config)

    summary = report["crossfit"]
    print(
        json.dumps(
            {
                "validation": "cross_fitted_oof",
                "f1_af": summary["f1_af"],
                "iou_burn": summary["iou_burn"],
                "miou_severity": summary["miou_severity"],
                "score": summary["score"],
            },
            indent=2,
        )
    )
    print(f"Saved cross-fitted report to {args.report}")
    print(f"Saved pooled deployment calibration to {args.deployment_config}")


if __name__ == "__main__":
    main()
