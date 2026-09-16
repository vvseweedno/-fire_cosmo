# Fires API routes
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional

from app.core.schemas import FireCandidate, AnalyzeRequest
from app.services.fire_detection import FireDetectionService
from app.services.false_positive_filter import FalsePositiveFilter
from app.services.fire_clustering import FireClusteringService


router = APIRouter(prefix="/fires", tags=["Fires"])


# Глобальный сервис (в реальности лучше через dependency injection)
_detection_service: Optional[FireDetectionService] = None
_filter_service: Optional[FalsePositiveFilter] = None
_clustering_service: Optional[FireClusteringService] = None


def get_services():
    """Получить или создать сервисы"""
    global _detection_service, _filter_service, _clustering_service
    
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
    
    return _detection_service, _filter_service, _clustering_service


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
    
    detection_service, filter_service, clustering_service = get_services()
    
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
    
    detection_service, filter_service, clustering_service = get_services()
    
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
    
    job_id = f"job_{uuid4().hex[:8]}"
    
    return {
        "job_id": job_id,
        "status": "completed",  # Для демо синхронно
        "events_found": len(events),
        "message": f"Found {len(filtered_points)} fire points, clustered into {len(events)} events"
    }
