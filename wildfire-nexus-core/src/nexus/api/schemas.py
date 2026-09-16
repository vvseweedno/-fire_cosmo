"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="System status")
    events_active: int = Field(..., description="Number of active fire events")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ThermalPointSchema(BaseModel):
    """Thermal point data."""

    id: str
    detected_at: datetime
    latitude: float
    longitude: float
    brightness: float
    confidence: float
    frp: float | None
    source: str


class FireEventSchema(BaseModel):
    """Fire event data."""

    id: str
    status: str
    first_seen: datetime
    last_seen: datetime
    centroid_lat: float
    centroid_lon: float
    point_count: int
    max_frp: float
    area_estimate_ha: float
    risk_level: str
    nearest_settlement: str | None

    model_config = {"from_attributes": True}


class FireEventDetailSchema(FireEventSchema):
    """Fire event with additional details."""

    last_alert_at: datetime | None
    last_alert_level: str | None


class StatsSchema(BaseModel):
    """Statistics response."""

    total_events: int
    active_events: int
    critical_events: int
    warning_events: int
    info_events: int
    total_points_24h: int
    period_start: datetime
    period_end: datetime
