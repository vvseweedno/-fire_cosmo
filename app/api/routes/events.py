# Events API routes
from fastapi import APIRouter, HTTPException

from app.core.schemas import EventReport


router = APIRouter(prefix="/events", tags=["Events"])


@router.get("")
async def list_events():
    """Получить список всех пожарных событий."""
    from app.api.routes.fires import get_services

    _, _, clustering, _ = get_services()
    events = clustering.get_all_events()
    return {
        "events": [
            {
                "event_id": e.id,
                "first_seen": e.first_seen.isoformat(),
                "last_seen": e.last_seen.isoformat(),
                "centroid": [e.centroid_lon, e.centroid_lat],
                "sensors": e.sensors,
                "status": e.status.value,
                "point_count": e.point_count,
                "area_ha": e.area_ha,
                "severity": e.severity.value if e.severity else None,
            }
            for e in events
        ],
        "total": len(events),
    }


@router.get("/{event_id}")
async def get_event(event_id: str):
    """Получить детали события по ID."""
    from app.api.routes.fires import get_services, _provenance_store

    _, _, clustering, _ = get_services()
    event = clustering.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return {
        "event_id": event.id,
        "first_seen": event.first_seen.isoformat(),
        "last_seen": event.last_seen.isoformat(),
        "centroid": [event.centroid_lon, event.centroid_lat],
        "fire_points": [
            {
                "id": p.id,
                "sensor": p.sensor,
                "datetime": p.datetime.isoformat(),
                "latitude": p.latitude,
                "longitude": p.longitude,
                "brightness_temp_k": p.brightness_temp_k,
                "confidence": p.confidence.value,
                "filters_passed": p.filters_passed,
                "filters_failed": p.filters_failed,
                "reason": p.reason,
            }
            for p in event.fire_points
        ],
        "sensors": event.sensors,
        "status": event.status.value,
        "point_count": event.point_count,
        "max_frp": event.max_frp,
        "area_ha": event.area_ha,
        "severity": event.severity.value if event.severity else None,
        "provenance": _provenance_store.get(event_id),
    }


@router.get("/{event_id}/burned.geojson")
async def get_burned_area(event_id: str):
    """Получить GeoJSON полигонов гари для события."""
    from app.api.routes.fires import _burned_area_store, get_services

    _, _, clustering, _ = get_services()
    if not clustering.get_event_by_id(event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    if event_id not in _burned_area_store:
        raise HTTPException(status_code=404, detail="Burned area has not been mapped yet")
    return _burned_area_store[event_id]


@router.get("/{event_id}/report")
async def get_event_report(event_id: str):
    """Получить воспроизводимый отчет: никаких вымышленных scene IDs."""
    from app.core.schemas import Sentinel2Scene, SeveritySummary
    from app.api.routes.fires import _report_store, _provenance_store, get_services

    _, _, clustering, _ = get_services()
    event = clustering.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    burned_area = _report_store.get(event_id)
    if burned_area is None:
        raise HTTPException(status_code=404, detail="Burned area has not been mapped yet")

    provenance = _provenance_store.get(event_id, {})
    pre_payload = provenance.get("pre_scene")
    post_payload = provenance.get("post_scene")
    pre_scene = Sentinel2Scene(**pre_payload) if pre_payload else None
    post_scene = Sentinel2Scene(**post_payload) if post_payload else None
    is_fixture = provenance.get("mode") == "offline_fixture"

    limitations = []
    warnings = []
    if is_fixture:
        limitations.append("Offline demo uses a local dNBR fixture; scene provenance is intentionally absent.")
        warnings.append("Fixture output must not be presented as a real Sentinel-2 observation.")
    else:
        if not provenance.get("cloud_mask"):
            limitations.append("Cloud-mask provenance is unavailable.")
        if pre_scene is None or post_scene is None:
            warnings.append("Sentinel-2 scene provenance is incomplete.")

    report = EventReport(
        event_id=event_id,
        region_bbox=provenance.get("analysis_bounds_wgs84"),
        first_seen=event.first_seen,
        last_seen=event.last_seen,
        sources=event.sensors,
        sentinel2_pre=pre_scene,
        sentinel2_post=post_scene,
        burned_area=burned_area,
        severity_summary=burned_area.severity_summary or SeveritySummary(),
        confidence="medium" if is_fixture else "high",
        warnings=warnings,
        limitations=limitations,
    )
    payload = report.model_dump(mode="json")
    payload["processing_provenance"] = provenance
    return payload
