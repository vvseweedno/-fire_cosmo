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


def test_operational_catalog_accepts_explicit_acquisition_offset(tmp_path):
    report = validate_catalog(
        _write_catalog(tmp_path, {"acquired_at": "2026-01-02T06:04:05+03:00"})
    )
    assert report["status"] == "VALID"


def test_operational_catalog_rejects_conflicting_acquisition_calendar_date(tmp_path):
    with pytest.raises(ValueError, match="date disagrees with acquired_at calendar date"):
        validate_catalog(
            _write_catalog(
                tmp_path,
                {"acquired_at": "2026-01-02T23:59:59Z", "date": "2026-01-03"},
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("acquired_at", "not-a-date", "invalid acquired_at"),
        ("date", "2026-99-99", "invalid date"),
        ("acquired_at", "", "invalid acquired_at"),
        ("date", 20260922, "invalid date"),
        ("acquired_at", "2026-01-02T03:04:05", "ambiguous acquired_at without timezone"),
        ("date", "2026-01-02T03:04:05Z", "date must be an ISO calendar date"),
    ),
)
def test_operational_catalog_rejects_malformed_or_ambiguous_temporal_metadata(
    tmp_path, field, value, message
):
    with pytest.raises(ValueError, match=message):
        validate_catalog(_write_catalog(tmp_path, {field: value}))
