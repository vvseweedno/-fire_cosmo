from datetime import UTC, datetime

import pytest

from wildfire.operational.ingest import ObservationDescriptor, observation_readiness


def _viirs_observation() -> ObservationDescriptor:
    return ObservationDescriptor(
        observation_id="stage-contract",
        sensor_family="VIIRS",
        acquired_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        crs="EPSG:4326",
        bbox=(38.0, 45.0, 39.0, 46.0),
        channels=("I4", "I5"),
        source="test",
    )


@pytest.mark.parametrize("stage", [None, 1, True, (), []])
def test_readiness_rejects_non_string_stage_metadata(stage):
    with pytest.raises(ValueError, match="stage must be AF or BS"):
        observation_readiness(_viirs_observation(), stage)  # type: ignore[arg-type]


def test_readiness_canonicalizes_string_stage_metadata():
    readiness = observation_readiness(_viirs_observation(), "  af  ")  # type: ignore[arg-type]
    assert readiness["stage"] == "AF"
    assert readiness["ready"] is True
