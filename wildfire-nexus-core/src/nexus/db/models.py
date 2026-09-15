"""
SQLAlchemy database models for Wildfire Nexus Core.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class ThermalPoint(Base):
    """
    Thermal point detected by satellite sensors (MODIS/VIIRS).
    
    Represents a single hot spot detection from NASA FIRMS data.
    """

    __tablename__ = "thermal_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid4().hex)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # 'VIIRS' | 'MODIS'
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    brightness: Mapped[float] = mapped_column(Float, nullable=False)  # Kelvin
    confidence: Mapped[float] = mapped_column(Float, nullable=False)  # 0-1
    frp: Mapped[float | None] = mapped_column(Float, nullable=True)  # MW (Fire Radiative Power)
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=True)  # Original API response
    
    # Foreign key to event
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    __table_args__ = (
        Index("ix_thermal_points_detected_at", "detected_at"),
        Index("ix_thermal_points_event_id", "event_id"),
    )

    def __repr__(self) -> str:
        return f"<ThermalPoint(id={self.id}, lat={self.latitude}, lon={self.longitude})>"


class FireEvent(Base):
    """
    Fire event tracked over time.
    
    Represents a wildfire as a process with lifecycle, aggregating multiple thermal points.
    """

    __tablename__ = "fire_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: uuid4().hex)
    status: Mapped[str] = mapped_column(String(20), default="active")  # 'active' | 'contained' | 'extinguished'
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    centroid_lat: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_lon: Mapped[float] = mapped_column(Float, nullable=False)
    point_count: Mapped[int] = mapped_column(Integer, default=1)
    max_frp: Mapped[float] = mapped_column(Float, default=0.0)
    area_estimate_ha: Mapped[float] = mapped_column(Float, default=0.0)  # Hectares
    risk_level: Mapped[str] = mapped_column(String(20), default="info")  # 'critical' | 'warning' | 'info'
    nearest_settlement: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_alert_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_alert_level: Mapped[str | None] = mapped_column(String(20), nullable=True)

    __table_args__ = (
        Index("ix_fire_events_status", "status"),
        Index("ix_fire_events_last_seen", "last_seen"),
        Index("ix_fire_events_risk_level", "risk_level"),
    )

    def __repr__(self) -> str:
        return f"<FireEvent(id={self.id}, status={self.status}, risk={self.risk_level})>"


class Settlement(Base):
    """
    Settlement (village, town, city) for proximity analysis.
    """

    __tablename__ = "settlements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    population: Mapped[int] = mapped_column(Integer, default=0)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("ix_settlements_name", "name"),
        Index("ix_settlements_population", "population"),
    )

    def __repr__(self) -> str:
        return f"<Settlement(id={self.id}, name={self.name}, pop={self.population})>"
