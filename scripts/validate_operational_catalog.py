#!/usr/bin/env python3
"""Validate a judge-facing result catalog before serving or demoing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wildfire.service import analytical_summary, load_results


def validate_catalog(path: str | Path) -> dict[str, Any]:
    """Parse a catalog and force all summary/evidence gates to execute."""
    features = load_results(path)
    summary = analytical_summary(features)
    return {
        "status": "VALID",
        "catalog": str(path),
        "feature_count": len(features),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed if an operational GeoJSON catalog violates service evidence gates."
    )
    parser.add_argument("catalog", type=Path, help="GeoJSON FeatureCollection to validate")
    args = parser.parse_args()

    try:
        report = validate_catalog(args.catalog)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
