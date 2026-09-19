from datetime import UTC, datetime, tzinfo

import pytest

from wildfire.operational.ingest import (
    ObservationDescriptor,
    build_burn_pair,
    observation_readiness,
    operational_capabilities,
    validate_observation,
)


def _obs(
    *,
    observation_id: str = "obs-1",
    sensor: str = "VIIRS",
    channels: tuple[str, ...] = ("I4", "I5"),
    acquired_at: datetime | None = None,
    crs: str = "EPSG:4326",
    bbox: tuple[float, float, float, float] = (38.0, 45.0, 39.0, 46.0),
    source: str = "test",
) -> ObservationDescriptor:
    return ObservationDescriptor(
        observation_id=observation_id,
        sensor_family=sensor,
        acquired_at=acquired_at or datetime(2026, 7, 1, 10, 0, tzinfo=UTC),
        crs=crs,
        bbox=bbox,
        channels=channels,
        source=source,
    )


def test_viirs_i4_i5_observation_is_currently_af_ready():
    observation = _obs()

    validate_observation(observation)
    readiness = observation_readiness(observation, "AF")

    assert readiness["public_contract_sensor"] is True
    assert readiness["inference_adapter_implemented"] is True
    assert readiness["missing_required_channels"] == []
    assert readiness["ready"] is True


def test_public_modis_sensor_is_not_falsely_marked_implemented():
    observation = _obs(sensor="MODIS", channels=("B21", "B31"))

    readiness = observation_readiness(observation, "AF")

    assert readiness["public_contract_sensor"] is True
    assert readiness["inference_adapter_implemented"] is False
    assert readiness["ready"] is False


def test_viirs_missing_i5_is_not_inference_ready():
    observation = _obs(channels=("I4",))

    readiness = observation_readiness(observation, "AF")

    assert readiness["missing_required_channels"] == ["I5"]
    assert readiness["ready"] is False


def test_observation_requires_explicit_source_provenance():
    observation = _obs(source="   ")

    with pytest.raises(ValueError, match="provenance is required"):
        validate_observation(observation)


def test_observation_requires_timezone_aware_timestamp():
    naive = _obs(acquired_at=datetime(2026, 7, 1, 10, 0))

    with pytest.raises(ValueError, match="timezone-aware"):
        validate_observation(naive)


def test_observation_rejects_timezone_without_utc_offset():
    class MissingOffsetTimezone(tzinfo):
        def utcoffset(self, dt):
            return None

        def dst(self, dt):
            return None

        def tzname(self, dt):
            return "missing-offset"

    invalid = _obs(
        acquired_at=datetime(2026, 7, 1, 10, 0, tzinfo=MissingOffsetTimezone())
    )

    with pytest.raises(ValueError, match="valid UTC offset"):
        validate_observation(invalid)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_observation_rejects_non_finite_bbox_coordinates(non_finite):
    observation = _obs(
        crs="EPSG:3857",
        bbox=(0.0, 0.0, non_finite, 1.0),
    )

    with pytest.raises(ValueError, match="bbox coordinates must be finite"):
        validate_observation(observation)


def test_capabilities_separate_public_and_implemented_sensor_families():
    capabilities = operational_capabilities()

    assert capabilities["active_fire"]["public_sensor_families"] == [
        "MODIS",
        "VIIRS",
        "LANDSAT",
    ]
    assert capabilities["active_fire"]["implemented_inference_adapters"] == [
        "VIIRS"
    ]
    assert capabilities["burn_assessment"]["implemented_inference_adapters"] == [
        "SENTINEL-2"
    ]


def test_capability_channel_requirements_match_readiness_contract():
    capabilities = operational_capabilities()

    assert capabilities["active_fire"]["implemented_viirs_minimum_channels"] == [
        "I4",
        "I5",
    ]
    assert capabilities["burn_assessment"][
        "implemented_sentinel2_minimum_channels"
    ] == ["B12", "B8A"]
    assert observation_readiness(_obs(), "AF")["required_channels"] == ["I4", "I5"]


def test_build_burn_pair_requires_ordered_coregistered_sentinel2():
    pre = _obs(
        observation_id="pre",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
    )
    post = _obs(
        observation_id="post",
        sensor="S2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 7, 1, 9, 0, tzinfo=UTC),
    )

    pair = build_burn_pair(pre, post)

    assert pair.pre.observation_id == "pre"
    assert pair.post.observation_id == "post"


def test_build_burn_pair_rejects_same_observation_id():
    pre = _obs(
        observation_id="same-scene",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    post = _obs(
        observation_id="same-scene",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="distinct observation_id"):
        build_burn_pair(pre, post)


def test_build_burn_pair_rejects_reversed_time():
    pre = _obs(
        observation_id="pre",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    post = _obs(
        observation_id="post",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 7, 1, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="earlier"):
        build_burn_pair(pre, post)


def test_build_burn_pair_rejects_non_coregistered_bbox():
    pre = _obs(
        observation_id="pre",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    post = _obs(
        observation_id="post",
        sensor="Sentinel-2",
        channels=("B8A", "B12"),
        acquired_at=datetime(2026, 7, 1, tzinfo=UTC),
        bbox=(38.1, 45.0, 39.1, 46.0),
    )

    with pytest.raises(ValueError, match="bounding boxes differ"):
        build_burn_pair(pre, post)
