# Events API routes
from fastapi import APIRouter, HTTPException
from typing import List, Optional

from app.core.schemas import FireEvent, BurnedAreaResult, EventReport


router = APIRouter(prefix="/events", tags=["Events"])


# Глобальное хранилище событий (в реальности - БД)
_events_store: dict = {}


@router.get("")
async def list_events():
    """Получить список всех пожарных событий"""
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
                "severity": e.severity.value if e.severity else None
            }
            for e in events
        ],
        "total": len(events)
    }


@router.get("/{event_id}")
async def get_event(event_id: str):
    """Получить детали события по ID"""
    from app.api.routes.fires import get_services

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
                "confidence": p.confidence.value
            }
            for p in event.fire_points
        ],
        "sensors": event.sensors,
        "status": event.status.value,
        "point_count": event.point_count,
        "max_frp": event.max_frp,
        "area_ha": event.area_ha,
        "severity": event.severity.value if event.severity else None
    }


@router.get("/{event_id}/burned.geojson")
async def get_burned_area(event_id: str):
    """Получить GeoJSON полигонов гари для события"""
    from app.api.routes.fires import _burned_area_store, get_services

    _, _, clustering, _ = get_services()
    if not clustering.get_event_by_id(event_id):
        raise HTTPException(status_code=404, detail="Event not found")
    if event_id not in _burned_area_store:
        raise HTTPException(status_code=404, detail="Burned area has not been mapped yet")
    return _burned_area_store[event_id]


@router.get("/{event_id}/report")
async def get_event_report(event_id: str):
    """Получить полный отчет о событии"""
    from app.core.schemas import Sentinel2Scene, SeveritySummary
    from app.api.routes.fires import _report_store, get_services

    _, _, clustering, _ = get_services()
    event = clustering.get_event_by_id(event_id)
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    burned_area = _report_store.get(event_id)
    if burned_area is None:
        raise HTTPException(status_code=404, detail="Burned area has not been mapped yet")

    severity_summary = burned_area.severity_summary or SeveritySummary()

    report = EventReport(
        event_id=event_id,
        region_bbox=None,
        first_seen=event.first_seen,
        last_seen=event.last_seen,
        sources=event.sensors,
        sentinel2_pre=Sentinel2Scene(
            scene_id="S2A_MSIL2A_20240101T000000_N0400_T10SEG_20240101T000000",
            datetime=event.first_seen,
            cloud_cover=5.2
        ) if event.fire_points else None,
        sentinel2_post=Sentinel2Scene(
            scene_id="S2B_MSIL2A_20240115T000000_N0400_T10SEG_20240115T000000",
            datetime=event.last_seen,
            cloud_cover=8.7
        ) if event.fire_points else None,
        burned_area=burned_area,
        severity_summary=severity_summary,
        confidence="medium" if burned_area.uncertainty == "demo_fixture" else "high",
        warnings=[],
        limitations=[
            "Демо-режим использует локальную dNBR-фикстуру вместо скачанного Sentinel-2 raster pair",
            "Для финальной защиты подключите реальные Sentinel-2 L2A сцены и сохраните их идентификаторы"
        ]
    )
    
    return report.model_dump(mode='json')
