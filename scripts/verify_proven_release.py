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
    parser.add_argument("--workflow-run-id", type=int)
    parser.add_argument("--workflow-sha")
    parser.add_argument("--workflow-conclusion")
    args = parser.parse_args()

    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence root must be an object")

    supplied_ci = (
        args.workflow_run_id is not None
        or args.workflow_sha is not None
        or args.workflow_conclusion is not None
    )
    if supplied_ci:
        if (
            args.workflow_run_id is None
            or args.workflow_sha is None
            or args.workflow_conclusion is None
        ):
            raise ValueError(
                "workflow CI attachment requires --workflow-run-id, "
                "--workflow-sha and --workflow-conclusion together"
            )
        existing = payload.get("ci")
        source_sha = (
            str(existing.get("source_commit_sha") or "")
            if isinstance(existing, dict)
            else ""
        )
        if not source_sha:
            raise ValueError("release evidence has no source_commit_sha")
        payload["ci"] = {
            "source_commit_sha": source_sha,
            "workflow_run_id": args.workflow_run_id,
            "workflow_sha": args.workflow_sha,
            "workflow_conclusion": args.workflow_conclusion,
        }

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
