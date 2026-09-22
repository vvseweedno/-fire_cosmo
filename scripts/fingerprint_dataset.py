"""Create stable dataset evidence before official experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.dataset_evidence import dataset_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/data_audit.json",
    )
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Hash all files byte-for-byte instead of only metadata/sidecars.",
    )
    args = parser.parse_args()

    report = dataset_fingerprint(args.data_dir, full_hash=args.full_hash)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
