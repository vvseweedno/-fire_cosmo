import json

import pytest

from scripts.validate_operational_catalog import validate_catalog


def _write_catalog(tmp_path, coordinates):
    path = tmp_path / "catalog.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "af-1",
                "geometry": {"type": "Point", "coordinates": coordinates},
                "properties": {"kind": "active_fire"},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_operational_catalog_accepts_wgs84_boundary_coordinates(tmp_path):
    report = validate_catalog(_write_catalog(tmp_path, [180.0, 90.0]))
    assert report["status"] == "VALID"


@pytest.mark.parametrize("coordinates", ([180.0001, 0.0], [-180.0001, 0.0], [0.0, 90.0001], [0.0, -90.0001]))
def test_operational_catalog_rejects_coordinates_outside_wgs84(tmp_path, coordinates):
    with pytest.raises(ValueError, match="outside GeoJSON WGS84 bounds"):
        validate_catalog(_write_catalog(tmp_path, coordinates))
