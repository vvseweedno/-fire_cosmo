from app.adapters.modis_adapter import ModisAdapter
from app.adapters.viirs_adapter import ViirsAdapter
from app.core.schemas import ConfidenceLevel
from app.services.fire_detection import FireDetectionService


def test_firms_date_range_is_chunked_to_api_maximum():
    chunks = list(ModisAdapter._iter_date_chunks("2026-07-01", "2026-07-12"))
    assert [days for _, days, _ in chunks] == [5, 5, 2]
    assert all(1 <= days <= 5 for _, days, _ in chunks)


def test_modis_numeric_confidence_uses_0_to_100_scale_correctly():
    adapter = ModisAdapter(offline_mode=True)
    assert adapter._parse_confidence("10") == ConfidenceLevel.LOW
    assert adapter._parse_confidence("30") == ConfidenceLevel.NOMINAL
    assert adapter._parse_confidence("80") == ConfidenceLevel.HIGH
    assert adapter._parse_confidence("0.8") == ConfidenceLevel.HIGH


def test_viirs_categorical_confidence_is_preserved():
    adapter = ViirsAdapter(offline_mode=True)
    assert adapter._parse_confidence("l") == ConfidenceLevel.LOW
    assert adapter._parse_confidence("n") == ConfidenceLevel.NOMINAL
    assert adapter._parse_confidence("h") == ConfidenceLevel.HIGH


def test_candidate_ids_are_deterministic_for_same_source_record():
    csv_text = """latitude,longitude,brightness,confidence,acq_date,acq_time,satellite,frp,daynight
56.0001,92.0001,330,85,2026-07-15,0315,Terra,12.5,D
"""
    adapter = ModisAdapter(offline_mode=True)
    first = adapter._parse_csv_response(csv_text, [91, 55, 93, 57], "2026-07-15", "2026-07-15")
    second = adapter._parse_csv_response(csv_text, [91, 55, 93, 57], "2026-07-15", "2026-07-15")
    assert first[0].id == second[0].id
    assert first[0].source == "MODIS_NRT"


def test_default_detection_does_not_claim_landsat_thermal_pipeline():
    service = FireDetectionService.create_default(offline_mode=True)
    assert set(service.adapters) == {"MODIS", "VIIRS"}
