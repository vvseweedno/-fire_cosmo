import json
from pathlib import Path

import pytest

from wildfire.aoi import load_monitoring_aoi, monitoring_aoi_summary


def _write_aoi(
    path: Path,
    *,
    close_ring: bool = True,
    crs: str = "EPSG:4326",
    area_km2: object = 435273,
) -> None:
    ring = [
        [38.3, 47.1],
        [42.0, 45.0],
        [48.0, 46.8],
        [48.0, 52.3],
        [38.3, 47.1] if close_ring else [38.4, 47.2],
    ]
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "aoi",
                "properties": {
                    "name": "Нижнее Поволжье и Подонье",
                    "role": "Граница территории мониторинга природных пожаров",
                    "subjects": "test subjects",
                    "seasons": "2019–2025",
                    "months": "04–10",
                    "utm_zones": "EPSG:32637, EPSG:32638",
                    "area_km2": area_km2,
                    "crs": crs,
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_monitoring_aoi_preserves_explicit_organizer_properties(tmp_path: Path):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path)

    aoi = load_monitoring_aoi(path)
    summary = monitoring_aoi_summary(aoi)

    assert aoi.feature_id == "aoi"
    assert aoi.name == "Нижнее Поволжье и Подонье"
    assert aoi.seasons == "2019–2025"
    assert aoi.months == "04–10"
    assert aoi.crs == "EPSG:4326"
    assert aoi.area_km2 == 435273.0
    assert summary["bbox"] == {
        "min_lon": 38.3,
        "min_lat": 45.0,
        "max_lon": 48.0,
        "max_lat": 52.3,
    }


def test_load_monitoring_aoi_requires_closed_ring(tmp_path: Path):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path, close_ring=False)

    with pytest.raises(ValueError, match="explicitly closed"):
        load_monitoring_aoi(path)


def test_load_monitoring_aoi_rejects_degenerate_ring(tmp_path: Path):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["features"][0]["geometry"]["coordinates"][0] = [
        [40.0, 50.0],
        [40.0, 50.0],
        [41.0, 51.0],
        [40.0, 50.0],
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 3 distinct vertices"):
        load_monitoring_aoi(path)


def test_load_monitoring_aoi_rejects_unexpected_crs(tmp_path: Path):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path, crs="EPSG:3857")

    with pytest.raises(ValueError, match="unsupported monitoring AOI CRS"):
        load_monitoring_aoi(path)


@pytest.mark.parametrize("area_km2", ["nan", "inf", "-inf", 0, -1])
def test_load_monitoring_aoi_rejects_invalid_area_metadata(
    tmp_path: Path, area_km2: object
):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path, area_km2=area_km2)

    with pytest.raises(ValueError, match="finite and positive"):
        load_monitoring_aoi(path)


@pytest.mark.parametrize("bad_coordinate", ["nan", "inf", "-inf"])
def test_load_monitoring_aoi_rejects_non_finite_coordinates(
    tmp_path: Path, bad_coordinate: str
):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ring = payload["features"][0]["geometry"]["coordinates"][0]
    ring[0][0] = bad_coordinate
    ring[-1][0] = bad_coordinate
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="finite coordinates"):
        load_monitoring_aoi(path)


def test_load_monitoring_aoi_does_not_fall_back_to_other_features(tmp_path: Path):
    path = tmp_path / "aoi.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "utm_32637",
                "properties": {"crs": "EPSG:4326"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[38, 45], [42, 45], [42, 52], [38, 45]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one feature"):
        load_monitoring_aoi(path)
