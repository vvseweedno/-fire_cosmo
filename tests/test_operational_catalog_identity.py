import json
from pathlib import Path

import pytest

from scripts.validate_operational_catalog import validate_catalog


def _write_catalog(path: Path, ids: list[str]) -> Path:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": feature_id,
                "geometry": {"type": "Point", "coordinates": [92.0 + index, 56.0]},
                "properties": {"kind": "active_fire"},
            }
            for index, feature_id in enumerate(ids)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_operational_catalog_rejects_duplicate_feature_ids(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "duplicate.geojson", ["event-1", "event-1"])

    with pytest.raises(ValueError, match="duplicate feature id: event-1"):
        validate_catalog(catalog)


def test_operational_catalog_accepts_unique_feature_ids(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "unique.geojson", ["event-1", "event-2"])

    report = validate_catalog(catalog)

    assert report["status"] == "VALID"
    assert report["feature_count"] == 2
