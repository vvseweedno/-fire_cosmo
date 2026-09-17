from datetime import datetime, timedelta

import pytest

from app.core.schemas import ConfidenceLevel, FireCandidate, FireStatus
from app.services.fire_clustering import FireClusteringService


def _point(point_id: str, minutes: int = 0, lat: float = 56.0, lon: float = 92.0):
    return FireCandidate(
        id=point_id,
        sensor="VIIRS",
        source="VIIRS_SNPP_NRT",
        datetime=datetime(2026, 7, 15, 3, 0) + timedelta(minutes=minutes),
        latitude=lat,
        longitude=lon,
        brightness_temp_k=340.0,
        frp_mw=10.0,
        confidence=ConfidenceLevel.HIGH,
        daynight="D",
    )


@pytest.mark.asyncio
async def test_repeated_point_does_not_duplicate_event_or_point():
    service = FireClusteringService(min_points_per_event=1)
    point = _point("viirs_stable_1")

    first = await service.cluster_points([point])
    second = await service.cluster_points([point])

    assert first[0].id == second[0].id
    assert second[0].point_count == 1
    assert len(service.get_all_events()) == 1


@pytest.mark.asyncio
async def test_mapped_event_is_reactivated_by_new_nearby_observation():
    service = FireClusteringService(min_points_per_event=1)
    first_event = (await service.cluster_points([_point("p1")]))[0]
    first_event.status = FireStatus.MAPPED

    touched = await service.cluster_points([_point("p2", minutes=20, lat=56.001, lon=92.001)])

    assert len(touched) == 1
    assert touched[0].id == first_event.id
    assert touched[0].point_count == 2
    assert touched[0].status == FireStatus.ACTIVE


@pytest.mark.asyncio
async def test_cluster_result_contains_only_events_touched_by_current_batch():
    service = FireClusteringService(min_points_per_event=1)
    first = await service.cluster_points([_point("p1")])
    second = await service.cluster_points([_point("p2", lat=57.0, lon=94.0)])

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].id != second[0].id
    assert len(service.get_all_events()) == 2


@pytest.mark.asyncio
async def test_event_id_is_deterministic_across_clean_services():
    point = _point("stable-source-id")
    service_a = FireClusteringService(min_points_per_event=1)
    service_b = FireClusteringService(min_points_per_event=1)

    event_a = (await service_a.cluster_points([point.model_copy(deep=True)]))[0]
    event_b = (await service_b.cluster_points([point.model_copy(deep=True)]))[0]

    assert event_a.id == event_b.id
