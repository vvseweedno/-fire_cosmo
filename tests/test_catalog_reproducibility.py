import hashlib
import json

from scripts.validate_operational_catalog import validate_catalog


def test_operational_catalog_report_binds_evidence_to_exact_bytes(tmp_path):
    catalog = tmp_path / "catalog.geojson"
    payload = b'{"type":"FeatureCollection","features":[]}'
    catalog.write_bytes(payload)

    report = validate_catalog(catalog)

    assert report["status"] == "VALID"
    assert report["feature_count"] == 0
    assert report["feature_ids"] == []
    assert report["catalog_size_bytes"] == len(payload)
    assert report["catalog_sha256"] == hashlib.sha256(payload).hexdigest()


def test_operational_catalog_digest_changes_when_catalog_bytes_change(tmp_path):
    first = tmp_path / "first.geojson"
    second = tmp_path / "second.geojson"
    first.write_bytes(b'{"type":"FeatureCollection","features":[]}')
    second.write_bytes(b'{"type": "FeatureCollection", "features": []}')

    assert validate_catalog(first)["catalog_sha256"] != validate_catalog(second)["catalog_sha256"]


def test_operational_catalog_report_lists_validated_feature_ids_in_catalog_order(tmp_path):
    catalog = tmp_path / "catalog.geojson"
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "af-observation-002",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {"kind": "active_fire"},
            },
            {
                "type": "Feature",
                "id": "af-observation-001",
                "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                "properties": {"kind": "active_fire"},
            },
        ],
    }
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_catalog(catalog)

    assert report["feature_count"] == 2
    assert report["feature_ids"] == ["af-observation-002", "af-observation-001"]
