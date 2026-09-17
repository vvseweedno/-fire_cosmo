from datetime import datetime, timezone

import numpy as np
import pytest

from app.adapters.sentinel2_adapter import Sentinel2Adapter
from app.core.schemas import Sentinel2Scene
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
                    "nir08": {
                        "href": "https://example.test/B8A.tif",
                        "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
                    },
                    "swir22": {
                        "href": "https://example.test/B12.tif",
                        "raster:bands": [{"scale": 0.0001, "offset": -0.1, "nodata": 0}],
                    },
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
    assert scene.assets["B8A"].endswith("B8A.tif")
    assert scene.assets["B12"].endswith("B12.tif")
    assert scene.assets["SCL"].endswith("SCL.tif")
    assert scene.asset_metadata["B8A"]["scale"] == 0.0001
    assert scene.asset_metadata["B8A"]["offset"] == -0.1
    assert scene.cloud_cover == 7.5


def test_processor_prefers_b8a_over_b08():
    scene = Sentinel2Scene(
        scene_id="scene",
        datetime=datetime.now(timezone.utc),
        cloud_cover=1,
        assets={"B8A": "b8a.tif", "B08": "b08.tif", "B12": "b12.tif"},
    )
    assert Sentinel2RasterProcessor._nir_key(scene) == "B8A"


def test_radiometric_scale_and_offset_are_applied_before_nbr():
    raw = np.array([[0, 2000, 5000]], dtype=np.float32)
    calibrated = Sentinel2RasterProcessor._apply_calibration(
        raw, scale=0.0001, offset=-0.1, nodata=0
    )
    assert np.isnan(calibrated[0, 0])
    assert calibrated[0, 1] == pytest.approx(0.1)
    assert calibrated[0, 2] == pytest.approx(0.4)


def test_nbr_uses_float_math_without_uint16_underflow():
    processor = Sentinel2RasterProcessor()
    nir = np.array([[1000]], dtype=np.uint16)
    swir = np.array([[3000]], dtype=np.uint16)
    nbr = processor._nbr(nir, swir)
    assert np.isclose(nbr[0, 0], -0.5)


def test_scl_invalid_classes_include_shadows_water_cloud_and_snow():
    processor = Sentinel2RasterProcessor()
    nir = np.ones((1, 7), dtype=np.float32) * 0.4
    swir = np.ones((1, 7), dtype=np.float32) * 0.2
    scl = np.array([[4, 2, 3, 6, 9, 10, 11]], dtype=np.uint8)
    valid = processor._valid_mask(nir, swir, scl)
    assert valid.tolist() == [[True, False, False, False, False, False, False]]


def test_severity_area_thresholds_are_monotonic():
    processor = Sentinel2RasterProcessor(low=0.10, moderate=0.27, high=0.44)
    dnbr = np.array([[0.0, 0.15, 0.30, 0.60]], dtype=np.float32)
    severity = processor._classify(dnbr, np.ones_like(dnbr, dtype=bool))
    assert severity.tolist() == [[0, 1, 2, 3]]


def test_invalid_severity_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        Sentinel2RasterProcessor(low=0.3, moderate=0.2, high=0.4)


def test_pixel_area_uses_affine_determinant():
    from affine import Affine

    transform = Affine(20, 0, 0, 0, -20, 0)
    assert Sentinel2RasterProcessor._pixel_area_m2(transform) == 400
