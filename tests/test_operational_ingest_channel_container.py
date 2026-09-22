from datetime import UTC, datetime

import pytest

from wildfire.operational.ingest import ObservationDescriptor, validate_observation


def test_observation_rejects_mutable_channel_container():
    observation = ObservationDescriptor(
        observation_id="obs-mutable-channels",
        sensor_family="VIIRS",
        acquired_at=datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        crs="EPSG:4326",
        bbox=(38.0, 45.0, 39.0, 46.0),
        channels=("I4", "I5"),
        source="test",
    )
    object.__setattr__(observation, "channels", ["I4", "I5"])

    with pytest.raises(ValueError, match="channels must be a tuple"):
        validate_observation(observation)
