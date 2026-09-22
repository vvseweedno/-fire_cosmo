#!/usr/bin/env python3
"""Validate a judge-facing result catalog before serving or demoing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from wildfire.service import analytical_summary, load_results


_CALENDAR_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_unique_feature_ids(features: list[dict[str, Any]]) -> None:
    """Reject ambiguous catalog identities before judge-facing publication."""
    seen: set[str] = set()
    for feature in features:
        feature_id = str(feature["id"])
        if feature_id in seen:
            raise ValueError(f"duplicate feature id: {feature_id}")
        seen.add(feature_id)


def _validate_wgs84_bounds(features: list[dict[str, Any]]) -> None:
    """Reject impossible lon/lat values in GeoJSON operational output."""
    for feature in features:
        geometry = feature["geometry"]
        positions = ([geometry["coordinates"]] if geometry["type"] == "Point" else geometry["coordinates"][0])
        for position in positions:
            lon, lat = float(position[0]), float(position[1])
            if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
                raise ValueError(f"feature {feature['id']} has coordinates outside GeoJSON WGS84 bounds")


def _validate_temporal_metadata(features: list[dict[str, Any]]) -> None:
    """Reject malformed, ambiguous, or internally conflicting acquisition dates."""
    for feature in features:
        properties = feature["properties"]
        parsed_values: dict[str, datetime] = {}
        for field in ("acquired_at", "date"):
            raw = properties.get(field)
            if raw is None:
                continue
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"feature {feature['id']} has invalid {field}")
            if field == "date" and not _CALENDAR_DATE_RE.fullmatch(raw):
                raise ValueError(f"feature {feature['id']} date must be an ISO calendar date (YYYY-MM-DD)")
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"feature {feature['id']} has invalid {field}") from exc
            if field == "acquired_at" and parsed.tzinfo is None:
                raise ValueError(f"feature {feature['id']} has ambiguous acquired_at without timezone")
            parsed_values[field] = parsed
        if "acquired_at" in parsed_values and "date" in parsed_values:
            if parsed_values["acquired_at"].date() != parsed_values["date"].date():
                raise ValueError(f"feature {feature['id']} date disagrees with acquired_at calendar date")


def validate_catalog(path: str | Path) -> dict[str, Any]:
    """Parse a catalog and force all runtime, identity, summary, and evidence gates to execute."""
    catalog_path = Path(path)
    features = load_results(catalog_path)
    _validate_unique_feature_ids(features)
    _validate_wgs84_bounds(features)
    _validate_temporal_metadata(features)
    summary = analytical_summary(features)
    catalog_bytes = catalog_path.read_bytes()
    digest = hashlib.sha256(catalog_bytes).hexdigest()
    return {
        "status": "VALID",
        "catalog": str(path),
        "catalog_sha256": digest,
        "catalog_size_bytes": len(catalog_bytes),
        "feature_count": len(features),
        "feature_ids": [str(feature["id"]) for feature in features],
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail closed if an operational GeoJSON catalog violates service evidence gates.")
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
