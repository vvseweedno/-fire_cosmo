import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_offline_summary_returns_area_by_severity():
    response = client.get("/api/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_burn_area_ha"] == pytest.approx(1.4)
    assert payload["area_by_severity_ha"] == {"1": 0.0, "2": 1.0, "3": 0.4}
    assert payload["active_fire_count"] == 2


def test_spatial_temporal_query_filters_demo_features():
    response = client.get(
        "/api/query",
        params={
            "bbox": "37.5,55.7,37.65,55.78",
            "start_date": "2026-07-01",
            "end_date": "2026-07-02",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["active_fire_points"]["features"]] == ["demo-af-001"]
    assert [item["id"] for item in payload["burned_area_contours"]["features"]] == ["demo-burn-001"]
    assert payload["summary"]["total_burn_area_ha"] == pytest.approx(1.0)


def test_query_post_accepts_polygon_and_geojson_export_is_machine_readable():
    response = client.post(
        "/api/query",
        json={
            "polygon": [[37.5, 55.7], [37.65, 55.7], [37.65, 55.78], [37.5, 55.7]],
            "start_date": "2026-07-01",
            "end_date": "2026-07-02",
        },
    )
    assert response.status_code == 200

    exported = client.get("/api/export/geojson")
    assert exported.status_code == 200
    assert exported.headers["content-type"].startswith("application/geo+json")
    assert exported.json()["type"] == "FeatureCollection"


@pytest.mark.parametrize(
    "polygon",
    [
        [[37.5, 55.7], [37.5, 55.7], [37.6, 55.8], [37.5, 55.7]],
        [[37.5, 55.7], [37.6, 55.8], [37.7, 55.9], [37.5, 55.7]],
    ],
)
def test_query_post_rejects_degenerate_polygons(polygon):
    response = client.post("/api/query", json={"polygon": polygon})

    assert response.status_code == 400


def test_polygon_filter_does_not_use_bbox_overlap_as_a_match():
    response = client.post(
        "/api/query",
        json={
            "polygon": [[37.60, 55.85], [37.70, 55.85], [37.70, 55.73], [37.60, 55.85]],
            "start_date": "2026-07-01",
            "end_date": "2026-07-02",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_fire_points"]["features"] == []
    assert payload["burned_area_contours"]["features"] == []
    assert payload["summary"]["total_burn_area_ha"] == 0.0


def test_offline_map_is_available_without_external_tiles():
    response = client.get("/map")

    assert response.status_code == 200
    assert "Offline wildfire map" in response.text
    assert "<svg" in response.text
