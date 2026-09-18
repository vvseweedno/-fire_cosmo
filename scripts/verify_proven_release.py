"""Verify that a release satisfies the strict engineering PROVEN contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.release_gate import evaluate_proven_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", default="artifacts/final_validation.json")
    parser.add_argument("--score-epsilon", type=float, default=1e-4)
    parser.add_argument("--repro-tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence root must be an object")

    report = evaluate_proven_release(
        payload,
        score_epsilon=args.score_epsilon,
        repro_tolerance=args.repro_tolerance,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["proven"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
