"""Offline result catalog and spatial-temporal service helpers."""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[1] / "service" / "demo_results.geojson"


def results_path() -> Path:
    return Path(os.getenv("WILDFIRE_RESULTS_GEOJSON", str(DEFAULT_RESULTS_PATH)))


def load_results(path: str | Path | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path is not None else results_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError("results catalog must be a GeoJSON FeatureCollection")
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("results catalog features must be a list")
    features: list[dict[str, Any]] = []
    for index, feature in enumerate(raw_features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"results feature {index} must be a GeoJSON Feature")
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        if not isinstance(geometry, dict) or geometry.get("type") not in {"Point", "Polygon"}:
            raise ValueError(f"results feature {index} must be a Point or Polygon")
        if not isinstance(properties, dict):
            raise ValueError(f"results feature {index} properties must be an object")
        feature_id = str(feature.get("id") or f"feature-{index}")
        features.append({"type": "Feature", "id": feature_id, "geometry": geometry, "properties": properties})
    return features


def parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None or not value.strip():
        return None
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("bbox must be min_x,min_y,max_x,max_y") from exc
    if len(parts) != 4:
        raise ValueError("bbox must contain four coordinates")
    min_x, min_y, max_x, max_y = parts
    if not all(math.isfinite(item) for item in parts) or not (min_x < max_x and min_y < max_y):
        raise ValueError("bbox must satisfy finite min_x < max_x and min_y < max_y")
    return parts


def normalize_polygon(polygon: list[list[float]] | None) -> tuple[tuple[float, float], ...] | None:
    if polygon is None:
        return None
    if len(polygon) < 4:
        raise ValueError("polygon must contain at least four coordinate pairs including closure")
    ring: list[tuple[float, float]] = []
    for point in polygon:
        if len(point) != 2:
            raise ValueError("polygon coordinates must be [x, y] pairs")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon coordinates must be finite")
        ring.append((x, y))
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(set(ring[:-1])) < 3:
        raise ValueError("polygon must contain at least three distinct vertices")
    twice_area = sum(left[0] * right[1] - right[0] * left[1] for left, right in zip(ring, ring[1:]))
    if math.isclose(twice_area, 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("polygon must have non-zero area")
    return tuple(ring)


def _feature_date(feature: dict[str, Any]) -> date | None:
    raw = feature["properties"].get("acquired_at") or feature["properties"].get("date")
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()


def _point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _geometry_bbox(feature: dict[str, Any]) -> tuple[float, float, float, float]:
    geometry = feature["geometry"]
    if geometry["type"] == "Point":
        x, y = geometry["coordinates"][:2]
        return float(x), float(y), float(x), float(y)
    ring = geometry["coordinates"][0]
    xs = [float(point[0]) for point in ring]
    ys = [float(point[1]) for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return not (left[2] < right[0] or left[0] > right[2] or left[3] < right[1] or left[1] > right[3])


def _orientation(left: tuple[float, float], middle: tuple[float, float], right: tuple[float, float]) -> float:
    return (middle[0] - left[0]) * (right[1] - left[1]) - (middle[1] - left[1]) * (right[0] - left[0])


def _on_segment(left: tuple[float, float], point: tuple[float, float], right: tuple[float, float]) -> bool:
    return min(left[0], right[0]) - 1e-12 <= point[0] <= max(left[0], right[0]) + 1e-12 and min(left[1], right[1]) - 1e-12 <= point[1] <= max(left[1], right[1]) + 1e-12


def _segments_intersect(first_left: tuple[float, float], first_right: tuple[float, float], second_left: tuple[float, float], second_right: tuple[float, float]) -> bool:
    epsilon = 1e-12
    orientations = (_orientation(first_left, first_right, second_left), _orientation(first_left, first_right, second_right), _orientation(second_left, second_right, first_left), _orientation(second_left, second_right, first_right))
    if (((orientations[0] > epsilon and orientations[1] < -epsilon) or (orientations[0] < -epsilon and orientations[1] > epsilon)) and ((orientations[2] > epsilon and orientations[3] < -epsilon) or (orientations[2] < -epsilon and orientations[3] > epsilon))):
        return True
    return any(abs(orientation) <= epsilon and _on_segment(left, point, right) for orientation, left, point, right in ((orientations[0], first_left, second_left, first_right), (orientations[1], first_left, second_right, first_right), (orientations[2], second_left, first_left, second_right), (orientations[3], second_left, first_right, second_right)))


def _point_in_ring(point: tuple[float, float], ring: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    for left, right in zip(ring, ring[1:]):
        if abs(_orientation(left, right, point)) <= 1e-12 and _on_segment(left, point, right):
            return True
        if (left[1] > point[1]) != (right[1] > point[1]):
            intersection_x = (right[0] - left[0]) * (point[1] - left[1]) / (right[1] - left[1]) + left[0]
            if point[0] < intersection_x:
                inside = not inside
    return inside


def _geometry_intersects_ring(feature: dict[str, Any], ring: tuple[tuple[float, float], ...]) -> bool:
    geometry = feature["geometry"]
    if geometry["type"] == "Point":
        coordinates = geometry["coordinates"]
        return _point_in_ring((float(coordinates[0]), float(coordinates[1])), ring)
    feature_ring = tuple((float(point[0]), float(point[1])) for point in geometry["coordinates"][0])
    if any(_point_in_ring(point, ring) for point in feature_ring[:-1]):
        return True
    if any(_point_in_ring(point, feature_ring) for point in ring[:-1]):
        return True
    return any(_segments_intersect(left, right, other_left, other_right) for left, right in zip(feature_ring, feature_ring[1:]) for other_left, other_right in zip(ring, ring[1:]))


def filter_results(features: list[dict[str, Any]], *, bbox: tuple[float, float, float, float] | None = None, polygon: tuple[tuple[float, float], ...] | None = None, start_date: date | None = None, end_date: date | None = None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    polygon_bbox = None
    if polygon is not None:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        polygon_bbox = (min(xs), min(ys), max(xs), max(ys))
    for feature in features:
        feature_date = _feature_date(feature)
        if start_date is not None and (feature_date is None or feature_date < start_date):
            continue
        if end_date is not None and (feature_date is None or feature_date > end_date):
            continue
        feature_bbox = _geometry_bbox(feature)
        if bbox is not None and not _bbox_intersects(feature_bbox, bbox):
            continue
        if polygon is not None and (polygon_bbox is None or not _bbox_intersects(feature_bbox, polygon_bbox) or not _geometry_intersects_ring(feature, polygon)):
            continue
        selected.append(feature)
    return selected


def _area_ha(feature: dict[str, Any]) -> float:
    properties = feature["properties"]
    raw_area = properties.get("area_ha")
    if raw_area is None:
        raise ValueError(f"feature {feature.get('id')} has no area_ha; supply area from a projected raster")
    area = float(raw_area)
    if not math.isfinite(area) or area < 0:
        raise ValueError(f"feature {feature.get('id')} has invalid area_ha")
    pixel_count = properties.get("pixel_count")
    pixel_area_m2 = properties.get("pixel_area_m2")
    if (pixel_count is None) != (pixel_area_m2 is None):
        raise ValueError(f"feature {feature.get('id')} has incomplete pixel geometry metadata")
    if pixel_count is None:
        raise ValueError(f"feature {feature.get('id')} has no pixel geometry metadata; area requires projected-raster provenance")
    try:
        count = float(pixel_count)
        pixel_area = float(pixel_area_m2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature {feature.get('id')} has invalid pixel geometry metadata") from exc
    if not math.isfinite(count) or count < 0 or not count.is_integer():
        raise ValueError(f"feature {feature.get('id')} has invalid pixel_count")
    if not math.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError(f"feature {feature.get('id')} has invalid pixel_area_m2")
    expected = count * pixel_area / 10_000.0
    if not math.isfinite(expected) or not math.isclose(area, expected, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"feature {feature.get('id')} area_ha disagrees with pixel geometry metadata")
    return area


def analytical_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    contours = [feature for feature in features if feature["properties"].get("kind") in {"burned_area", "burn", "severity"}]
    area_by_severity = {str(class_id): 0.0 for class_id in (1, 2, 3)}
    for feature in contours:
        severity = int(feature["properties"].get("severity_class", 0))
        if severity not in (1, 2, 3):
            raise ValueError(f"feature {feature.get('id')} has invalid severity_class")
        area_by_severity[str(severity)] += _area_ha(feature)
    total = sum(area_by_severity.values())
    return {"feature_count": len(features), "active_fire_count": sum(feature["properties"].get("kind") in {"active_fire", "af"} for feature in features), "burned_area_contour_count": len(contours), "total_burn_area_ha": total, "area_by_severity_ha": area_by_severity, "area_source": "projected-raster pixel metadata carried by each contour"}


def feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def query_results(features: list[dict[str, Any]], *, bbox: tuple[float, float, float, float] | None = None, polygon: tuple[tuple[float, float], ...] | None = None, start_date: date | None = None, end_date: date | None = None) -> dict[str, Any]:
    selected = filter_results(features, bbox=bbox, polygon=polygon, start_date=start_date, end_date=end_date)
    active = [feature for feature in selected if feature["properties"].get("kind") in {"active_fire", "af"}]
    burned = [feature for feature in selected if feature["properties"].get("kind") in {"burned_area", "burn", "severity"}]
    return {"query": {"bbox": bbox, "polygon": polygon, "start_date": start_date.isoformat() if start_date else None, "end_date": end_date.isoformat() if end_date else None}, "active_fire_points": feature_collection(active), "burned_area_contours": feature_collection(burned), "summary": analytical_summary(selected)}
