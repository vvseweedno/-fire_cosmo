import pytest

from wildfire.operational.ingest import ObservationDescriptor, validate_observation


def test_crs84_observation_rejects_out_of_range_longitude():
    observation = ObservationDescriptor(
        observation_id="crs84-out-of-range",
        sensor_family="VIIRS",
        acquired_at=__import__("datetime").datetime.datetime(
            2026,
            7,
            1,
            tzinfo=__import__("datetime").datetime.UTC,
        ),
        crs="OGC:CRS84",
        bbox=(181.0, 45.0, 182.0, 46.0),
        channels=("I4", "I5"),
        source="regression-test",
    )

    with pytest.raises(ValueError, match="longitude is outside"):
        validate_observation(observation)
