# ruff: noqa: I001
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _write_aoi(path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "aoi",
                "properties": {
                    "name": "Нижнее Поволжье и Подонье",
                    "role": "Граница территории мониторинга природных пожаров",
                    "subjects": "test",
                    "seasons": "2019–2025",
                    "months": "04–10",
                    "utm_zones": "EPSG:32637, EPSG:32638",
                    "area_km2": 435273,
                    "crs": "EPSG:4326",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [38.3, 47.1],
                            [48.0, 46.8],
                            [48.0, 52.3],
                            [38.3, 47.1],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_api_spec_exposes_public_two_stage_contract():
    response = client.get("/api/spec")

    assert response.status_code == 200
    payload = response.json()
    assert "active burning" in payload["public_task"]["stage_1"]
    assert "Sentinel-2" in payload["public_task"]["stage_2"]
    assert payload["aoi_endpoint"] == "/api/aoi"
    assert payload["readiness_endpoint"] == "/api/readiness"
    assert "working_score_formula" in payload


def test_api_aoi_is_explicitly_unavailable_without_configuration(monkeypatch):
    monkeypatch.delenv("WILDFIRE_AOI_GEOJSON", raising=False)

    response = client.get("/api/aoi")

    assert response.status_code == 503
    assert "AOI is not configured" in response.json()["detail"]


def test_api_aoi_returns_only_configured_geojson_metadata(tmp_path: Path, monkeypatch):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path)
    monkeypatch.setenv("WILDFIRE_AOI_GEOJSON", str(path))

    response = client.get("/api/aoi")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Нижнее Поволжье и Подонье"
    assert payload["crs"] == "EPSG:4326"
    assert payload["area_km2"] == 435273.0
    assert payload["bbox"] == {
        "min_lon": 38.3,
        "min_lat": 46.8,
        "max_lon": 48.0,
        "max_lat": 52.3,
    }


def test_api_models_never_claims_unverified_accuracy():
    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["accuracy"] == {
        "status": "UNVERIFIED_ON_ORGANIZER_LABELS",
        "metrics": None,
    }
    assert payload["capabilities"]["active_fire"][
        "implemented_inference_adapters"
    ] == ["VIIRS"]
    assert "MODIS" in payload["capabilities"]["active_fire"]["public_sensor_families"]


def test_api_readiness_reports_real_blockers_without_aoi(monkeypatch):
    monkeypatch.delenv("WILDFIRE_AOI_GEOJSON", raising=False)

    response = client.get("/api/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "PARTIAL"
    assert payload["checks"]["service"]["ready"] is True
    assert payload["checks"]["aoi"]["ready"] is False
    assert payload["checks"]["organizer_labelled_metrics"]["ready"] is False
    assert set(payload["blockers"]) == {"aoi", "organizer_labelled_metrics"}


def test_api_readiness_accepts_valid_explicit_aoi(tmp_path: Path, monkeypatch):
    path = tmp_path / "aoi.geojson"
    _write_aoi(path)
    monkeypatch.setenv("WILDFIRE_AOI_GEOJSON", str(path))

    response = client.get("/api/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["checks"]["aoi"]["ready"] is True
    assert payload["blockers"] == ["organizer_labelled_metrics"]
