# Fires API routes
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional

from app.core.schemas import FireCandidate, AnalyzeRequest
from app.services.fire_detection import FireDetectionService
from app.services.false_positive_filter import FalsePositiveFilter
from app.services.fire_clustering import FireClusteringService
from app.services.burned_area_mapper import BurnedAreaMapper


router = APIRouter(prefix="/fires", tags=["Fires"])


# Глобальный сервис (в реальности лучше через dependency injection)
_detection_service: Optional[FireDetectionService] = None
_filter_service: Optional[FalsePositiveFilter] = None
_clustering_service: Optional[FireClusteringService] = None
_burned_area_mapper: Optional[BurnedAreaMapper] = None
_burned_area_store: dict = {}
_report_store: dict = {}


def get_services():
    """Получить или создать сервисы"""
    global _detection_service, _filter_service, _clustering_service, _burned_area_mapper
    
    if _detection_service is None:
        from app.core.config import settings
        
        _detection_service = FireDetectionService.create_default(
            firms_api_key=settings.firms_map_key,
            offline_mode=settings.offline_mode,
            cache_dir=settings.cache_dir
        )
    
    if _filter_service is None:
        _filter_service = FalsePositiveFilter()
    
    if _clustering_service is None:
        _clustering_service = FireClusteringService()

    if _burned_area_mapper is None:
        from app.core.config import settings
        _burned_area_mapper = BurnedAreaMapper(output_dir=settings.output_dir)
    
    return _detection_service, _filter_service, _clustering_service, _burned_area_mapper


@router.get("")
async def list_fires(
    bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    start_date: str = Query(..., description="Start date ISO 8601"),
    end_date: str = Query(..., description="End date ISO 8601"),
    sensor: Optional[str] = Query(None, description="Sensor type: MODIS, VIIRS"),
    min_confidence: str = Query("nominal", description="Minimum confidence level")
):
    """
    Получить список очагов пожаров
    
    Возвращает GeoJSON FeatureCollection с точками термических аномалий
    """
    try:
        bbox_coords = [float(x) for x in bbox.split(",")]
        if len(bbox_coords) != 4:
            raise HTTPException(status_code=400, detail="Invalid bbox format")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid bbox coordinates")
    
    detection_service, filter_service, clustering_service, _ = get_services()
    
    # Детекция
    sensors = [sensor] if sensor else None
    points = await detection_service.detect_fires(bbox_coords, start_date, end_date, sensors)
    
    # Фильтрация ложных
    filtered_points = await filter_service.filter_points(points)
    
    # Конвертация в GeoJSON
    features = []
    for point in filtered_points:
        if point.is_valid:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point.longitude, point.latitude]
                },
                "properties": {
                    "id": point.id,
                    "sensor": point.sensor,
                    "datetime": point.datetime.isoformat(),
                    "brightness_temp_k": point.brightness_temp_k,
                    "frp_mw": point.frp_mw,
                    "confidence": point.confidence.value,
                    "daynight": point.daynight
                }
            })
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "total": len(features)
    }


@router.post("/analyze")
async def analyze_region(request: AnalyzeRequest):
    """
    Запустить полный анализ региона
    
    1. Детекция очагов
    2. Кластеризация в события
    3. Подбор Sentinel-2
    4. Расчет площади гари
    """
    from uuid import uuid4
    
    detection_service, filter_service, clustering_service, burned_area_mapper = get_services()
    
    # Детекция
    points = await detection_service.detect_fires(
        request.bbox,
        request.start_date,
        request.end_date,
        request.sensors
    )
    
    # Фильтрация
    filtered_points = await filter_service.filter_points(points)
    
    # Кластеризация
    events = await clustering_service.cluster_points(filtered_points)

    # Картирование гарей по Sentinel-2/dNBR fixture или будущему raster path
    event_summaries = []
    for event in events:
        burned_geojson, burned_result = await burned_area_mapper.process_sentinel2_pair(
            pre_scene_path=None,
            post_scene_path="data/fixtures/sentinel2",
            event_id=event.id,
            center_lon=event.centroid_lon,
            center_lat=event.centroid_lat
        )
        event.area_ha = burned_result.area_ha
        if burned_result.severity_summary:
            summary = burned_result.severity_summary
            dominant = max(
                [
                    ("low_severity", summary.low_severity_ha),
                    ("moderate_severity", summary.moderate_severity_ha),
                    ("high_severity", summary.high_severity_ha),
                ],
                key=lambda item: item[1]
            )[0]
            from app.core.schemas import SeverityLevel, FireStatus
            event.severity = SeverityLevel(dominant)
            event.status = FireStatus.MAPPED
        event.burned_area_geojson = burned_geojson
        _burned_area_store[event.id] = burned_geojson
        _report_store[event.id] = burned_result
        event_summaries.append({
            "event_id": event.id,
            "centroid": [event.centroid_lon, event.centroid_lat],
            "point_count": event.point_count,
            "sensors": event.sensors,
            "area_ha": burned_result.area_ha,
            "severity_summary": burned_result.severity_summary.model_dump() if burned_result.severity_summary else None,
            "burned_geojson_url": f"/api/v1/events/{event.id}/burned.geojson",
            "report_url": f"/api/v1/events/{event.id}/report"
        })
    
    job_id = f"job_{uuid4().hex[:8]}"
    
    return {
        "job_id": job_id,
        "status": "completed",  # Для демо синхронно
        "events_found": len(events),
        "fire_points_found": len(points),
        "fire_points_after_filter": len(filtered_points),
        "events": event_summaries,
        "message": f"Found {len(filtered_points)} fire points, clustered into {len(events)} mapped events"
    }
