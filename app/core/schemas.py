# Pydantic schemas for API request/response
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceLevel(str, Enum):
    LOW = "low"
    NOMINAL = "nominal"
    HIGH = "high"


class FireStatus(str, Enum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    MAPPED = "mapped"
    FAILED = "failed"


class SeverityLevel(str, Enum):
    UNBURNED = "unburned"
    LOW = "low_severity"
    MODERATE = "moderate_severity"
    HIGH = "high_severity"
    NODATA = "nodata"


class FireCandidate(BaseModel):
    """Нормализованный кандидат термической аномалии."""

    id: str
    sensor: str
    source: Optional[str] = None
    datetime: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    brightness_temp_k: float
    frp_mw: Optional[float] = None
    confidence: ConfidenceLevel
    daynight: str
    satellite: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)
    is_valid: bool = True
    false_positive_score: float = 0.0
    filters_passed: List[str] = Field(default_factory=list)
    filters_failed: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class FireEventCreate(BaseModel):
    event_id: Optional[str] = None
    first_seen: datetime
    last_seen: datetime
    centroid_lat: float
    centroid_lon: float
    sensors: List[str] = Field(default_factory=list)
    status: FireStatus = FireStatus.ACTIVE
    point_count: int = 1
    max_frp: Optional[float] = None


class FireEvent(FireEventCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    fire_points: List[FireCandidate] = Field(default_factory=list)
    area_ha: Optional[float] = None
    severity: Optional[SeverityLevel] = None
    burned_area_geojson: Optional[Dict[str, Any]] = None


class Sentinel2Scene(BaseModel):
    """Sentinel-2 L2A scene with reproducible STAC/raster provenance."""

    scene_id: str
    datetime: datetime
    cloud_cover: float
    tile_id: Optional[str] = None
    collection: Optional[str] = None
    stac_item_url: Optional[str] = None
    bbox: Optional[List[float]] = None
    assets: Dict[str, str] = Field(default_factory=dict)
    # Per canonical asset: STAC raster:bands scale/offset/nodata where supplied.
    asset_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class BurnedAreaSelection(BaseModel):
    event_id: str
    pre_scene_id: Optional[str] = None
    post_scene_id: Optional[str] = None
    pre_datetime: Optional[datetime] = None
    post_datetime: Optional[datetime] = None
    cloud_cover_pre: Optional[float] = None
    cloud_cover_post: Optional[float] = None
    confidence: str = "high"


class SeveritySummary(BaseModel):
    unburned_ha: float = 0.0
    low_severity_ha: float = 0.0
    moderate_severity_ha: float = 0.0
    high_severity_ha: float = 0.0
    total_affected_ha: float = 0.0


class BurnedAreaResult(BaseModel):
    event_id: str
    area_ha: float
    area_m2: float
    projection_used: str
    method: str
    uncertainty: str = "medium"
    severity_summary: Optional[SeveritySummary] = None


class EventReport(BaseModel):
    event_id: str
    region_bbox: Optional[List[float]] = None
    first_seen: datetime
    last_seen: datetime
    sources: List[str]
    sentinel2_pre: Optional[Sentinel2Scene] = None
    sentinel2_post: Optional[Sentinel2Scene] = None
    burned_area: BurnedAreaResult
    severity_summary: SeveritySummary
    confidence: str
    warnings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    bbox: List[float] = Field(min_length=4, max_length=4)
    start_date: str
    end_date: str
    sensors: List[str] = Field(default_factory=lambda: ["MODIS", "VIIRS"])
    min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = "queued"
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "fire-burned-area-service"
    version: str = "1.0.0"
    offline_mode: bool = False
