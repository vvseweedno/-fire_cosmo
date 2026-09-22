import json

import pytest

from scripts.validate_operational_catalog import validate_catalog


def _write_catalog(tmp_path, geometry):
    catalog = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "geometry-check",
                "geometry": geometry,
                "properties": {"kind": "active_fire"},
            }
        ],
    }
    path = tmp_path / "catalog.geojson"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Point", "coordinates": [float("nan"), 55.0]},
        {"type": "Point", "coordinates": [37.0]},
        {
            "type": "Polygon",
            "coordinates": [[[37.0, 55.0], [38.0, 55.0], [38.0, 56.0], [37.0, 56.0]]],
        },
        {
            "type": "Polygon",
            "coordinates": [[[37.0, 55.0], [37.0, 55.0], [37.0, 55.0], [37.0, 55.0]]],
        },
    ],
)
def test_catalog_validator_rejects_malformed_geometry(tmp_path, geometry):
    path = _write_catalog(tmp_path, geometry)

    with pytest.raises(ValueError):
        validate_catalog(path)


def test_catalog_validator_accepts_finite_point_geometry(tmp_path):
    path = _write_catalog(tmp_path, {"type": "Point", "coordinates": [37.0, 55.0]})

    report = validate_catalog(path)

    assert report["status"] == "VALID"
    assert report["feature_count"] == 1
