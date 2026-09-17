# Events API routes
from fastapi import APIRouter, HTTPException

from app.core.schemas import EventReport


router = APIRouter(prefix="/events", tags=["Events"])


@router.get("")
async def list_events():
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
    from app.api.routes.fires import _provenance_store, get_services

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
                "source": p.source,
                "sensor": p.sensor,
                "datetime": p.datetime.isoformat(),
                "latitude": p.latitude,
                "longitude": p.longitude,
                "brightness_temp_k": p.brightness_temp_k,
                "frp_mw": p.frp_mw,
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
    from app.api.routes.fires import _burned_area_store, get_services

    _, _, clustering, _ = get_services()
    if not clustering.get_event_by_id(event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    if event_id not in _burned_area_store:
        raise HTTPException(status_code=404, detail="Burned area has not been mapped yet")
    return _burned_area_store[event_id]


@router.get("/{event_id}/report")
async def get_event_report(event_id: str):
    """Return a reproducible report without fabricated scene metadata."""
    from app.api.routes.fires import _provenance_store, _report_store, get_services
    from app.core.schemas import Sentinel2Scene, SeveritySummary

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

    limitations = [
        "Burn severity thresholds are configurable heuristics and require validation for the target ecosystem.",
        "Forest-only affected area is not claimed until a validated forest/land-cover mask is connected.",
    ]
    warnings = []
    confidence = "medium"

    if is_fixture:
        confidence = "demo_fixture"
        limitations.append("Offline demo uses a local dNBR fixture; real scene provenance is intentionally absent.")
        warnings.append("Fixture output must not be presented as a real Sentinel-2 observation.")
    else:
        if not provenance.get("cloud_surface_mask"):
            limitations.append("SCL/cloud/surface-mask provenance is unavailable.")
        if pre_scene is None or post_scene is None:
            warnings.append("Sentinel-2 scene provenance is incomplete.")
        if pre_scene and post_scene and ("SCL" not in pre_scene.assets or "SCL" not in post_scene.assets):
            warnings.append("One or both scenes lack SCL; cloud/surface masking is incomplete.")
        if not (pre_scene and post_scene):
            confidence = "low"

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
        confidence=confidence,
        warnings=warnings,
        limitations=limitations,
    )
    payload = report.model_dump(mode="json")
    payload["processing_provenance"] = provenance
    return payload
