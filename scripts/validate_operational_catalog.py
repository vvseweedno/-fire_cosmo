#!/usr/bin/env python3
"""Validate a judge-facing result catalog before serving or demoing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from wildfire.service import analytical_summary, load_results


def _validate_unique_feature_ids(features: list[dict[str, Any]]) -> None:
    """Reject ambiguous catalog identities before judge-facing publication."""
    seen: set[str] = set()
    for feature in features:
        feature_id = str(feature["id"])
        if feature_id in seen:
            raise ValueError(f"duplicate feature id: {feature_id}")
        seen.add(feature_id)


def validate_catalog(path: str | Path) -> dict[str, Any]:
    """Parse a catalog and force all runtime, identity, summary, and evidence gates to execute."""
    catalog_path = Path(path)
    features = load_results(catalog_path)
    _validate_unique_feature_ids(features)
    summary = analytical_summary(features)
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    return {
        "status": "VALID",
        "catalog": str(path),
        "catalog_sha256": digest,
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
