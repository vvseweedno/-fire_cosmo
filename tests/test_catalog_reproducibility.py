import hashlib

from scripts.validate_operational_catalog import validate_catalog


def test_operational_catalog_report_binds_evidence_to_exact_bytes(tmp_path):
    catalog = tmp_path / "catalog.geojson"
    payload = b'{"type":"FeatureCollection","features":[]}'
    catalog.write_bytes(payload)

    report = validate_catalog(catalog)

    assert report["status"] == "VALID"
    assert report["feature_count"] == 0
    assert report["catalog_sha256"] == hashlib.sha256(payload).hexdigest()


def test_operational_catalog_digest_changes_when_catalog_bytes_change(tmp_path):
    first = tmp_path / "first.geojson"
    second = tmp_path / "second.geojson"
    first.write_bytes(b'{"type":"FeatureCollection","features":[]}')
    second.write_bytes(b'{"type": "FeatureCollection", "features": []}')

    assert validate_catalog(first)["catalog_sha256"] != validate_catalog(second)["catalog_sha256"]
