"""Validate submission.csv against official sample_submission.csv and meta.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from wildfire.metadata import read_meta_csv
from wildfire.submission import (
    read_submission_template,
    validate_submission_against_template,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=447,
        help="Current official template row count (default: 447).",
    )
    args = parser.parse_args()

    root = Path(args.data_dir)
    template = read_submission_template(root / "sample_submission.csv")
    meta = read_meta_csv(root / "meta.csv")
    shapes = {chip_id: item.shape for chip_id, item in meta.items()}
    tasks = {chip_id: item.kind for chip_id, item in meta.items()}

    errors = validate_submission_against_template(
        args.submission,
        template,
        shapes,
        tasks=tasks,
        expected_row_count=args.expected_rows,
    )
    if errors:
        print("INVALID SUBMISSION")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print(f"VALID SUBMISSION: {len(template)} template rows")


if __name__ == "__main__":
    main()
