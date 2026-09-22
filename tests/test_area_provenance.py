import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_summary_rejects_burn_area_without_complete_raster_provenance(
    tmp_path,
    monkeypatch,
):
    cases = [
        {"kind": "burned_area", "severity_class": 2, "area_ha": 1.0},
        {
            "kind": "burned_area",
            "severity_class": 2,
            "area_ha": 1.0,
            "pixel_count": 25,
        },
        {
            "kind": "burned_area",
            "severity_class": 2,
            "area_ha": 1.0,
            "pixel_area_m2": 400.0,
        },
    ]

    for index, properties in enumerate(cases):
        catalog = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": f"unproven-area-{index}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": properties,
                }
            ],
        }
        path = tmp_path / f"results-{index}.geojson"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        monkeypatch.setenv("WILDFIRE_RESULTS_GEOJSON", str(path))

        response = client.get("/api/summary")

        assert response.status_code == 400
        assert "pixel geometry metadata" in response.json()["detail"]
