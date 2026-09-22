import json

import pytest

from scripts.validate_operational_catalog import validate_catalog


def _write_catalog(tmp_path, properties):
    path = tmp_path / "catalog.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "af-1",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {"kind": "active_fire", **properties},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_operational_catalog_accepts_iso_temporal_metadata(tmp_path):
    report = validate_catalog(
        _write_catalog(tmp_path, {"acquired_at": "2026-01-02T03:04:05Z", "date": "2026-01-02"})
    )
    assert report["status"] == "VALID"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("acquired_at", "not-a-date"),
        ("date", "2026-99-99"),
        ("acquired_at", ""),
        ("date", 20260922),
    ),
)
def test_operational_catalog_rejects_malformed_temporal_metadata(tmp_path, field, value):
    with pytest.raises(ValueError, match=rf"invalid {field}"):
        validate_catalog(_write_catalog(tmp_path, {field: value}))
