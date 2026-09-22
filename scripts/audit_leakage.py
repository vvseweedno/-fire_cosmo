"""Run explicit leakage checks and emit machine-readable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.leakage import audit_leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-manifest")
    parser.add_argument(
        "--output",
        default="artifacts/leakage_audit.json",
    )
    args = parser.parse_args()

    manifest = None
    if args.fold_manifest:
        manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("fold manifest root must be an object")

    report = audit_leakage(args.data_dir, manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
