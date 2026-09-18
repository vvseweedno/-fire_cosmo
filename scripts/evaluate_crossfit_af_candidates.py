"""Cross-fit AF candidate weights before promoting pooled deployment parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.af_ensemble_validation import crossfit_af_candidate_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument(
        "--output",
        default="outputs/crossfit_af_candidate_ensemble.json",
    )
    parser.add_argument("--alpha-steps", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    args = parser.parse_args()

    manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fold manifest root must be an object")

    report = crossfit_af_candidate_ensemble(
        args.candidate_root,
        manifest,
        alpha_steps=args.alpha_steps,
        epsilon=args.epsilon,
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
