from datetime import UTC, datetime

import pytest

from wildfire.operational.ingest import ObservationDescriptor, build_burn_pair, validate_observation


def _sentinel2(*, observation_id: str, crs: str, month: int) -> ObservationDescriptor:
    return ObservationDescriptor(
        observation_id=observation_id,
        sensor_family="Sentinel-2",
        acquired_at=datetime(2026, month, 1, tzinfo=UTC),
        crs=crs,
        bbox=(38.0, 45.0, 39.0, 46.0),
        channels=("B8A", "B12"),
        source="test",
    )


def test_observation_rejects_nonempty_but_invalid_crs():
    observation = _sentinel2(observation_id="invalid-crs", crs="not-a-real-crs", month=6)
    with pytest.raises(ValueError, match="valid coordinate reference system"):
        validate_observation(observation)


def test_burn_pair_accepts_equivalent_crs_representations():
    pre = _sentinel2(observation_id="pre", crs="EPSG:4326", month=6)
    post = _sentinel2(observation_id="post", crs="OGC:CRS84", month=7)

    pair = build_burn_pair(pre, post)

    assert pair.pre.observation_id == "pre"
    assert pair.post.observation_id == "post"
