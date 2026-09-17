from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.api.routes import fires
from app.core.config import settings
from app.main import app
from app.services.burned_area_mapper import BurnedAreaMapper


@pytest.mark.asyncio
async def test_nbr_uses_float_math_for_uint16_bands():
    mapper = BurnedAreaMapper()

    nbr = await mapper.calculate_nbr(
        np.array([[1000]], dtype=np.uint16),
        np.array([[3000]], dtype=np.uint16),
    )

    assert nbr[0, 0] == pytest.approx(-0.5)


@pytest.mark.asyncio
async def test_burned_area_fixture_counts_only_affected_pixels():
    mapper = BurnedAreaMapper()

    geojson, result = await mapper.process_sentinel2_pair(
        pre_scene_path=None,
        post_scene_path="data/fixtures/sentinel2",
        event_id="fire_demo",
        center_lon=92.0,
        center_lat=56.0,
    )

    assert result.area_ha > 0
    assert result.method == "dNBR_pixel_count"
    assert result.severity_summary is not None
    assert result.area_ha == result.severity_summary.total_affected_ha
    assert all(feature["properties"]["severity"] != "unburned" for feature in geojson["features"])


def test_api_analyze_persists_events_and_burned_area(monkeypatch):
    monkeypatch.setattr(settings, "offline_mode", True)
    fires._detection_service = None
    fires._filter_service = None
    fires._clustering_service = None
    fires._burned_area_mapper = None
    fires._burned_area_store.clear()
    fires._report_store.clear()

    client = TestClient(app)
    response = client.post(
        "/api/v1/fires/analyze",
        json={
            "bbox": [91.8, 55.85, 92.2, 56.15],
            "start_date": "2026-07-15",
            "end_date": "2026-07-16",
            "sensors": ["MODIS", "VIIRS"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["events_found"] > 0
    event_id = payload["events"][0]["event_id"]
    assert payload["events"][0]["area_ha"] > 0

    events_response = client.get("/api/v1/events")
    assert events_response.status_code == 200
    assert events_response.json()["total"] == payload["events_found"]

    burned_response = client.get(f"/api/v1/events/{event_id}/burned.geojson")
    assert burned_response.status_code == 200
    assert burned_response.json()["features"]

    missing_response = client.get("/api/v1/events/not-real/burned.geojson")
    assert missing_response.status_code == 404
