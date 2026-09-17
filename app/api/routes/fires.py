# Fires API routes
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.schemas import AnalyzeRequest, ConfidenceLevel
from app.services.burned_area_mapper import BurnedAreaMapper
from app.services.false_positive_filter import FalsePositiveFilter
from app.services.fire_clustering import FireClusteringService
from app.services.fire_detection import FireDetectionService


router = APIRouter(prefix="/fires", tags=["Fires"])


_detection_service: Optional[FireDetectionService] = None
_filter_service: Optional[FalsePositiveFilter] = None
_clustering_service: Optional[FireClusteringService] = None
_burned_area_mapper: Optional[BurnedAreaMapper] = None
_burned_area_store: dict = {}
_report_store: dict = {}
_provenance_store: dict = {}


def get_services():
    """Получить или создать сервисы."""
    global _detection_service, _filter_service, _clustering_service, _burned_area_mapper

    from app.core.config import settings

    if _detection_service is None:
        _detection_service = FireDetectionService.create_default(
            firms_api_key=settings.firms_map_key,
            offline_mode=settings.offline_mode,
            cache_dir=settings.cache_dir,
        )
    if _filter_service is None:
        _filter_service = FalsePositiveFilter(
            whitelist_path=settings.industrial_whitelist_path,
            min_confidence=settings.default_min_confidence,
        )
    if _clustering_service is None:
        _clustering_service = FireClusteringService(
            min_points_per_event=max(1, settings.min_fire_point_cluster_size)
        )
    if _burned_area_mapper is None:
        _burned_area_mapper = BurnedAreaMapper(output_dir=settings.output_dir)
    return _detection_service, _filter_service, _clustering_service, _burned_area_mapper


def _candidate_payload(point) -> dict:
    return {
        "id": point.id,
        "source": point.source,
        "sensor": point.sensor,
        "datetime": point.datetime.isoformat(),
        "latitude": point.latitude,
        "longitude": point.longitude,
        "brightness_temp_k": point.brightness_temp_k,
        "frp_mw": point.frp_mw,
        "confidence": point.confidence.value,
        "daynight": point.daynight,
        "satellite": point.satellite,
        "is_valid": point.is_valid,
        "filters_passed": point.filters_passed,
        "filters_failed": point.filters_failed,
        "reason": point.reason,
    }


@router.get("")
async def list_fires(
    bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    start_date: str = Query(..., description="Start date ISO 8601"),
    end_date: str = Query(..., description="End date ISO 8601"),
    sensor: Optional[str] = Query(None, description="Sensor type: MODIS, VIIRS"),
    min_confidence: str = Query("nominal", description="Minimum confidence level"),
):
    """Получить GeoJSON принятых термических аномалий и полный аудит фильтрации."""
    try:
        bbox_coords = [float(x) for x in bbox.split(",")]
        if len(bbox_coords) != 4:
            raise HTTPException(status_code=400, detail="Invalid bbox format")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid bbox coordinates") from exc

    try:
        confidence_threshold = ConfidenceLevel(min_confidence.lower())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="min_confidence must be one of: low, nominal, high",
        ) from exc

    detection_service, filter_service, _, _ = get_services()
    filter_service.min_confidence = confidence_threshold
    sensors = [sensor.upper()] if sensor else None
    points = await detection_service.detect_fires(bbox_coords, start_date, end_date, sensors)
    filtered_points = await filter_service.filter_points(points)

    features = []
    for point in filtered_points:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [point.longitude, point.latitude],
                },
                "properties": _candidate_payload(point),
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "total": len(features),
        "filter_audit": filter_service.audit_summary(),
        "rejected_candidates": [_candidate_payload(p) for p in filter_service.last_rejected],
    }


async def _map_event(event, burned_area_mapper, request_bbox):
    """Run fixture mapping offline or real STAC/COG mapping online."""
    from app.core.config import settings

    if settings.offline_mode:
        geojson, result = await burned_area_mapper.process_sentinel2_pair(
            pre_scene_path=None,
            post_scene_path="data/fixtures/sentinel2",
            event_id=event.id,
            center_lon=event.centroid_lon,
            center_lat=event.centroid_lat,
        )
        provenance = {
            "mode": "offline_fixture",
            "fixture": "data/fixtures/sentinel2/dNBR.json",
            "warning": "Demo fixture; not a claim about a real satellite scene.",
        }
        return geojson, result, provenance

    from app.adapters.sentinel2_adapter import Sentinel2Adapter
    from app.services.sentinel2_raster import Sentinel2RasterProcessor

    stac = Sentinel2Adapter(settings.sentinel2_stac_api)
    pre_scene, post_scene = await stac.find_best_pair(
        bbox=request_bbox,
        reference_date=event.first_seen,
        days_before=settings.pre_image_days_before,
        days_after=settings.post_image_days_after,
        max_cloud_cover=settings.max_cloud_cover,
    )
    if pre_scene is None or post_scene is None:
        raise RuntimeError("No reproducible pre/post Sentinel-2 L2A pair found")

    processor = Sentinel2RasterProcessor(
        low=settings.severity_unburned_max,
        moderate=settings.severity_low_max,
        high=settings.severity_moderate_max,
    )
    analysis = await asyncio.to_thread(
        processor.process_pair,
        pre_scene,
        post_scene,
        event.id,
        event.centroid_lon,
        event.centroid_lat,
    )
    return analysis.geojson, analysis.result, analysis.provenance


@router.post("/analyze")
async def analyze_region(request: AnalyzeRequest):
    """Run detection -> filtering -> clustering -> Sentinel-2 burn mapping."""
    from uuid import uuid4

    from app.core.schemas import FireStatus, SeverityLevel

    detection_service, filter_service, clustering_service, burned_area_mapper = get_services()
    filter_service.min_confidence = request.min_confidence

    points = await detection_service.detect_fires(
        request.bbox,
        request.start_date,
        request.end_date,
        [sensor.upper() for sensor in request.sensors],
    )
    filtered_points = await filter_service.filter_points(points)
    filter_audit = filter_service.audit_summary()
    events = await clustering_service.cluster_points(filtered_points)

    event_summaries = []
    mapped_count = 0
    for event in events:
        try:
            burned_geojson, burned_result, provenance = await _map_event(
                event, burned_area_mapper, request.bbox
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
                    key=lambda item: item[1],
                )[0]
                event.severity = SeverityLevel(dominant)
            event.status = FireStatus.MAPPED
            event.burned_area_geojson = burned_geojson
            _burned_area_store[event.id] = burned_geojson
            _report_store[event.id] = burned_result
            _provenance_store[event.id] = provenance
            mapped_count += 1
            event_summaries.append(
                {
                    "event_id": event.id,
                    "status": "burn_mapping_completed",
                    "centroid": [event.centroid_lon, event.centroid_lat],
                    "point_count": event.point_count,
                    "sensors": event.sensors,
                    "area_ha": burned_result.area_ha,
                    "severity_summary": (
                        burned_result.severity_summary.model_dump()
                        if burned_result.severity_summary
                        else None
                    ),
                    "burned_geojson_url": f"/api/v1/events/{event.id}/burned.geojson",
                    "report_url": f"/api/v1/events/{event.id}/report",
                    "provenance": provenance,
                }
            )
        except Exception as exc:
            # Detection/event is still valid; only stage 2 is unavailable.
            event.status = FireStatus.ACTIVE
            _burned_area_store.pop(event.id, None)
            _report_store.pop(event.id, None)
            _provenance_store.pop(event.id, None)
            event_summaries.append(
                {
                    "event_id": event.id,
                    "status": "burn_mapping_unavailable",
                    "centroid": [event.centroid_lon, event.centroid_lat],
                    "point_count": event.point_count,
                    "sensors": event.sensors,
                    "area_ha": None,
                    "reason": str(exc),
                }
            )

    if not events:
        overall_status = "completed"
    elif mapped_count == len(events):
        overall_status = "completed"
    else:
        overall_status = "partial"

    job_id = f"job_{uuid4().hex[:8]}"
    return {
        "job_id": job_id,
        "status": overall_status,
        "stages": {
            "detection": "completed",
            "filtering": "completed",
            "clustering": "completed",
            "burn_mapping": (
                "completed"
                if not events or mapped_count == len(events)
                else "partial"
            ),
        },
        "events_found": len(events),
        "events_mapped": mapped_count,
        "fire_points_found": len(points),
        "fire_points_after_filter": len(filtered_points),
        "filter_audit": filter_audit,
        "rejected_candidates": [_candidate_payload(p) for p in filter_service.last_rejected],
        "events": event_summaries,
        "message": (
            f"Found {len(filtered_points)} accepted fire candidates; "
            f"mapped {mapped_count}/{len(events)} events"
        ),
    }
