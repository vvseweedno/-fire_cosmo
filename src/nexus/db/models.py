"""SQLAlchemy database models."""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class ThermalPoint(Base):
    """Thermal point detected by satellite (FIRMS data)."""

    __tablename__ = "thermal_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # 'VIIRS' | 'MODIS'
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    brightness: Mapped[float] = mapped_column(Float, nullable=False)  # Kelvin
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1
    frp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # MW (Fire Radiative Power)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)  # Original FIRMS response
    event_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)  # FK to FireEvent

    __table_args__ = (Index("idx_thermal_detected", "detected_at"), Index("idx_thermal_event", "event_id"))

    def __repr__(self) -> str:
        return f"<ThermalPoint(id={self.id}, lat={self.latitude}, lon={self.longitude}, source={self.source})>"


class FireEvent(Base):
    """Fire event - tracked wildfire process (multiple thermal points over time)."""

    __tablename__ = "fire_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")  # active | contained | extinguished
    first_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    centroid_lat: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_lon: Mapped[float] = mapped_column(Float, nullable=False)
    point_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_frp: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    area_estimate_ha: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # Hectares
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False, default="info")  # critical | warning | info
    nearest_settlement: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_alert_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_alert_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    __table_args__ = (Index("idx_event_status", "status", "last_seen"),)

    def __repr__(self) -> str:
        return f"<FireEvent(id={self.id}, status={self.status}, risk={self.risk_level})>"


class Settlement(Base):
    """Settlement (village, town, city) for proximity analysis."""

    __tablename__ = "settlements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    population: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    region: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    __table_args__ = (Index("idx_settlement_location", "latitude", "longitude"),)

    def __repr__(self) -> str:
        return f"<Settlement(id={self.id}, name={self.name}, pop={self.population})>"
