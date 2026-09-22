from datetime import UTC, datetime

import pytest

from wildfire.operational.ingest import ObservationDescriptor, build_burn_pair, validate_observation


def _sentinel2(*, observation_id: str, acquired_at: datetime, crs: str) -> ObservationDescriptor:
    return ObservationDescriptor(
        observation_id=observation_id,
        sensor_family="Sentinel-2",
        acquired_at=acquired_at,
        crs=crs,
        bbox=(38.0, 45.0, 39.0, 46.0),
        channels=("B8A", "B12"),
        source="test",
    )


def test_validate_observation_rejects_unresolvable_crs():
    observation = _sentinel2(
        observation_id="scene",
        acquired_at=datetime(2026, 7, 1, tzinfo=UTC),
        crs="not-a-real-crs",
    )

    with pytest.raises(ValueError, match="valid coordinate reference system"):
        validate_observation(observation)


def test_burn_pair_accepts_semantically_equivalent_crs_encodings():
    pre = _sentinel2(
        observation_id="pre",
        acquired_at=datetime(2026, 6, 1, tzinfo=UTC),
        crs="EPSG:4326",
    )
    post = _sentinel2(
        observation_id="post",
        acquired_at=datetime(2026, 7, 1, tzinfo=UTC),
        crs="urn:ogc:def:crs:EPSG::4326",
    )

    pair = build_burn_pair(pre, post)

    assert pair.pre is pre
    assert pair.post is post
