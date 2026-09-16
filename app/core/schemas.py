# Pydantic schemas for API request/response
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


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


# === Fire Detection Schemas ===

class FireCandidate(BaseModel):
    """Кандидат в очаги пожара (нормализованная точка)"""
    id: str
    sensor: str  # MODIS | VIIRS | LANDSAT
    datetime: datetime
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    brightness_temp_k: float
    frp_mw: Optional[float] = None
    confidence: ConfidenceLevel
    daynight: str  # D | N
    satellite: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation flags
    is_valid: bool = True
    false_positive_score: float = 0.0
    filters_passed: List[str] = Field(default_factory=list)
    filters_failed: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class FireEventCreate(BaseModel):
    """Создание пожарного события"""
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
    """Пожарное событие с полной информацией"""
    id: str
    fire_points: List[FireCandidate] = Field(default_factory=list)
    area_ha: Optional[float] = None
    severity: Optional[SeverityLevel] = None
    burned_area_geojson: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


# === Burned Area Schemas ===

class Sentinel2Scene(BaseModel):
    """Снимок Sentinel-2"""
    scene_id: str
    datetime: datetime
    cloud_cover: float
    tile_id: Optional[str] = None


class BurnedAreaSelection(BaseModel):
    """Результат подбора снимков для анализа гари"""
    event_id: str
    pre_scene_id: Optional[str] = None
    post_scene_id: Optional[str] = None
    pre_datetime: Optional[datetime] = None
    post_datetime: Optional[datetime] = None
    cloud_cover_pre: Optional[float] = None
    cloud_cover_post: Optional[float] = None
    confidence: str = "high"  # high | medium | low


class SeveritySummary(BaseModel):
    """Статистика по степеням поражения"""
    unburned_ha: float = 0.0
    low_severity_ha: float = 0.0
    moderate_severity_ha: float = 0.0
    high_severity_ha: float = 0.0
    total_affected_ha: float = 0.0


class BurnedAreaResult(BaseModel):
    """Результат расчета площади гари"""
    event_id: str
    area_ha: float
    area_m2: float
    projection_used: str
    method: str  # polygon_area | raster_pixel_count
    uncertainty: str = "medium"  # low | medium | high
    severity_summary: Optional[SeveritySummary] = None


# === Report Schemas ===

class EventReport(BaseModel):
    """Полный отчет о событии"""
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


# === API Request/Response ===

class AnalyzeRequest(BaseModel):
    """Запрос на анализ региона"""
    bbox: List[float] = Field(min_length=4, max_length=4)  # [min_lon, min_lat, max_lon, max_lat]
    start_date: str
    end_date: str
    sensors: List[str] = ["MODIS", "VIIRS", "LANDSAT"]
    min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL


class AnalyzeResponse(BaseModel):
    """Ответ на запрос анализа"""
    job_id: str
    status: str = "queued"
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Статус сервиса"""
    status: str = "ok"
    service: str = "fire-burned-area-service"
    version: str = "1.0.0"
    offline_mode: bool = False
