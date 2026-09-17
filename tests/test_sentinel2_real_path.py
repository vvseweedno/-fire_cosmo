from datetime import datetime, timezone

import numpy as np

from app.adapters.sentinel2_adapter import Sentinel2Adapter
from app.services.sentinel2_raster import Sentinel2RasterProcessor


def test_stac_parser_keeps_required_assets_and_provenance():
    adapter = Sentinel2Adapter("https://earth-search.aws.element84.com/v1")
    payload = {
        "features": [
            {
                "id": "S2_TEST",
                "collection": "sentinel-2-l2a",
                "bbox": [91.0, 55.0, 92.0, 56.0],
                "properties": {
                    "datetime": "2026-07-15T03:00:00Z",
                    "eo:cloud_cover": 7.5,
                    "grid:code": "MGRS-46V",
                },
                "links": [{"rel": "self", "href": "https://example.test/item.json"}],
                "assets": {
                    "nir": {"href": "https://example.test/B08.tif"},
                    "swir22": {"href": "https://example.test/B12.tif"},
                    "scl": {"href": "https://example.test/SCL.tif"},
                },
            }
        ]
    }

    scenes = adapter._parse_stac_response(payload)
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene.scene_id == "S2_TEST"
    assert scene.collection == "sentinel-2-l2a"
    assert scene.assets["B08"].endswith("B08.tif")
    assert scene.assets["B12"].endswith("B12.tif")
    assert scene.assets["SCL"].endswith("SCL.tif")
    assert scene.cloud_cover == 7.5


def test_nbr_uses_float_math_without_uint16_underflow():
    processor = Sentinel2RasterProcessor()
    nir = np.array([[1000]], dtype=np.uint16)
    swir = np.array([[3000]], dtype=np.uint16)
    nbr = processor._nbr(nir, swir)
    assert np.isclose(nbr[0, 0], -0.5)


def test_scl_invalid_classes_are_excluded():
    processor = Sentinel2RasterProcessor()
    nir = np.ones((1, 4), dtype=np.float32) * 3000
    swir = np.ones((1, 4), dtype=np.float32) * 1000
    scl = np.array([[4, 3, 9, 11]], dtype=np.uint8)
    valid = processor._valid_mask(nir, swir, scl)
    assert valid.tolist() == [[True, False, False, False]]


def test_severity_area_thresholds_are_monotonic():
    processor = Sentinel2RasterProcessor(low=0.10, moderate=0.27, high=0.44)
    dnbr = np.array([[0.0, 0.15, 0.30, 0.60]], dtype=np.float32)
    severity = processor._classify(dnbr, np.ones_like(dnbr, dtype=bool))
    assert severity.tolist() == [[0, 1, 2, 3]]


def test_pixel_area_uses_affine_determinant():
    from affine import Affine

    transform = Affine(20, 0, 0, 0, -20, 0)
    assert Sentinel2RasterProcessor._pixel_area_m2(transform) == 400
