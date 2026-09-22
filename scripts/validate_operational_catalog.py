#!/usr/bin/env python3
"""Validate a judge-facing result catalog before serving or demoing it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from wildfire.service import analytical_summary, load_results


def _coordinate_pair(value: Any, *, feature_id: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"feature {feature_id} has malformed coordinates")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature {feature_id} has non-numeric coordinates") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"feature {feature_id} has non-finite coordinates")
    return x, y


def _validate_geometry(feature: dict[str, Any]) -> None:
    """Fail closed on malformed geometries before they reach the judge-facing API."""
    feature_id = str(feature.get("id", "unknown"))
    geometry = feature["geometry"]
    coordinates = geometry.get("coordinates")
    if geometry["type"] == "Point":
        _coordinate_pair(coordinates, feature_id=feature_id)
        return
    if not isinstance(coordinates, list) or not coordinates or not isinstance(coordinates[0], list):
        raise ValueError(f"feature {feature_id} has malformed Polygon coordinates")
    ring = coordinates[0]
    if len(ring) < 4:
        raise ValueError(f"feature {feature_id} Polygon ring must contain at least four positions")
    points = [_coordinate_pair(point, feature_id=feature_id) for point in ring]
    if points[0] != points[-1]:
        raise ValueError(f"feature {feature_id} Polygon ring is not closed")
    if len(set(points[:-1])) < 3:
        raise ValueError(f"feature {feature_id} Polygon ring has fewer than three distinct vertices")


def validate_catalog(path: str | Path) -> dict[str, Any]:
    """Parse a catalog and force all geometry, summary, and evidence gates to execute."""
    features = load_results(path)
    for feature in features:
        _validate_geometry(feature)
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
