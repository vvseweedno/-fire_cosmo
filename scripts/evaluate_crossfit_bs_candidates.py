"""Cross-fit the BS candidate ensemble before promoting pooled deployment weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.bs_ensemble_validation import crossfit_bs_candidate_ensemble
from wildfire.model_config import load_model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument(
        "--output",
        default="outputs/crossfit_bs_candidate_ensemble.json",
    )
    parser.add_argument("--alpha-steps", type=int, default=12)
    parser.add_argument("--threshold-candidates", type=int, default=64)
    parser.add_argument("--threshold-passes", type=int, default=2)
    parser.add_argument("--landcover-passes", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fold manifest root must be an object")

    report = crossfit_bs_candidate_ensemble(
        args.candidate_root,
        manifest,
        load_model_config(args.base_config),
        alpha_steps=args.alpha_steps,
        threshold_candidates=args.threshold_candidates,
        threshold_passes=args.threshold_passes,
        landcover_passes=args.landcover_passes,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
