import json

import pytest

from scripts.validate_operational_catalog import validate_catalog


def test_validator_accepts_demo_catalog():
    report = validate_catalog("service/demo_results.geojson")

    assert report["status"] == "VALID"
    assert report["feature_count"] == 4
    assert report["summary"]["total_burn_area_ha"] == pytest.approx(1.4)


def test_validator_rejects_unproven_hectares(tmp_path):
    catalog = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "unproven-area",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "properties": {
                    "kind": "burned_area",
                    "severity_class": 2,
                    "area_ha": 1.0,
                },
            }
        ],
    }
    path = tmp_path / "results.geojson"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="projected-raster provenance"):
        validate_catalog(path)


def test_validator_rejects_invalid_severity_even_with_area_provenance(tmp_path):
    catalog = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "invalid-severity",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
                "properties": {
                    "kind": "burned_area",
                    "severity_class": 4,
                    "area_ha": 1.0,
                    "pixel_count": 100,
                    "pixel_area_m2": 100.0,
                },
            }
        ],
    }
    path = tmp_path / "invalid-severity.geojson"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid severity_class"):
        validate_catalog(path)
