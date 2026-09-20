"""Strict, dependency-light monitoring AOI parsing.

The organiser AOI is a GeoJSON FeatureCollection. This module intentionally
validates only explicit geometry/properties; it never reconstructs private-test
boundaries or derives labels from geography.
"""

from __future__ import annotations  # noqa: I001

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MonitoringAOI:
    feature_id: str
    name: str
    role: str
    subjects: str
    seasons: str
    months: str
    utm_zones: str
    area_km2: float | None
    crs: str
    ring: tuple[tuple[float, float], ...]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        longitudes = [point[0] for point in self.ring]
        latitudes = [point[1] for point in self.ring]
        return (
            min(longitudes),
            min(latitudes),
            max(longitudes),
            max(latitudes),
        )


def _as_text(properties: dict[str, Any], key: str) -> str:
    value = properties.get(key, "")
    return "" if value is None else str(value)


def _validate_ring(raw_ring: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw_ring, list) or len(raw_ring) < 4:
        raise ValueError("AOI polygon exterior ring must contain at least 4 positions")

    ring: list[tuple[float, float]] = []
    for index, position in enumerate(raw_ring):
        if not isinstance(position, list) or len(position) < 2:
            raise ValueError(f"AOI position {index} must contain longitude and latitude")
        try:
            lon = float(position[0])
            lat = float(position[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"AOI position {index} is not numeric") from exc
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise ValueError(f"AOI position {index} must contain finite coordinates")
        if not (-180.0 <= lon <= 180.0):
            raise ValueError(f"AOI longitude out of range at position {index}: {lon}")
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"AOI latitude out of range at position {index}: {lat}")
        ring.append((lon, lat))

    if ring[0] != ring[-1]:
        raise ValueError("AOI polygon exterior ring must be explicitly closed")
    return tuple(ring)


def load_monitoring_aoi(
    path: str | Path,
    *,
    feature_id: str = "aoi",
) -> MonitoringAOI:
    """Load one explicitly named organiser AOI feature from GeoJSON."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("AOI GeoJSON root must be a FeatureCollection")

    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("AOI GeoJSON features must be a list")

    matches = [
        feature
        for feature in features
        if isinstance(feature, dict) and str(feature.get("id", "")) == feature_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"AOI GeoJSON must contain exactly one feature with id={feature_id!r}"
        )

    feature = matches[0]
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        raise ValueError("monitoring AOI geometry must be a Polygon")

    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) != 1:
        raise ValueError("monitoring AOI must contain exactly one exterior polygon ring")
    ring = _validate_ring(coordinates[0])

    properties = feature.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("monitoring AOI properties must be an object")

    crs = _as_text(properties, "crs")
    if crs and crs.upper() != "EPSG:4326":
        raise ValueError(f"unsupported monitoring AOI CRS: {crs}")

    raw_area = properties.get("area_km2")
    area_km2: float | None
    if raw_area in (None, ""):
        area_km2 = None
    else:
        try:
            area_km2 = float(raw_area)
        except (TypeError, ValueError) as exc:
            raise ValueError("AOI area_km2 must be numeric when provided") from exc
        if not math.isfinite(area_km2) or area_km2 <= 0:
            raise ValueError("AOI area_km2 must be finite and positive")

    return MonitoringAOI(
        feature_id=feature_id,
        name=_as_text(properties, "name"),
        role=_as_text(properties, "role"),
        subjects=_as_text(properties, "subjects"),
        seasons=_as_text(properties, "seasons"),
        months=_as_text(properties, "months"),
        utm_zones=_as_text(properties, "utm_zones"),
        area_km2=area_km2,
        crs=crs or "EPSG:4326",
        ring=ring,
    )


def monitoring_aoi_summary(aoi: MonitoringAOI) -> dict[str, object]:
    min_lon, min_lat, max_lon, max_lat = aoi.bbox
    return {
        "feature_id": aoi.feature_id,
        "name": aoi.name,
        "role": aoi.role,
        "subjects": aoi.subjects,
        "seasons": aoi.seasons,
        "months": aoi.months,
        "utm_zones": aoi.utm_zones,
        "area_km2": aoi.area_km2,
        "crs": aoi.crs,
        "vertices_including_closure": len(aoi.ring),
        "bbox": {
            "min_lon": min_lon,
            "min_lat": min_lat,
            "max_lon": max_lon,
            "max_lat": max_lat,
        },
    }
