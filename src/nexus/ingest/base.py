"""Base classes and types for ingest module."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class BoundingBox:
    """Geographic bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @classmethod
    def from_list(cls, bbox: list[float]) -> "BoundingBox":
        """Create from [min_lon, min_lat, max_lon, max_lat] list."""
        if len(bbox) != 4:
            raise ValueError(f"Bounding box must have 4 values, got {len(bbox)}")
        return cls(
            min_lon=bbox[0],
            min_lat=bbox[1],
            max_lon=bbox[2],
            max_lat=bbox[3],
        )


@dataclass
class ThermalPoint:
    """Thermal point from FIRMS API."""

    latitude: float
    longitude: float
    brightness: float  # Kelvin
    confidence: float  # 0-1
    source: str  # 'VIIRS' | 'MODIS'
    detected_at: datetime
    frp: Optional[float] = None  # MW (Fire Radiative Power)
    scan: Optional[float] = None  # Scan error
    track: Optional[float] = None  # Track error
    acq_date: Optional[str] = None  # Acquisition date
    acq_time: Optional[str] = None  # Acquisition time
    satellite: Optional[str] = None  # Satellite name
    instrument: Optional[str] = None  # Instrument name
    version: Optional[str] = None  # Data version
    bright_t31: Optional[float] = None  # Brightness temperature T31
    frp_confidence: Optional[str] = None  # FRP confidence level
    type: Optional[str] = None  # Fire type (0-presumed vegetation, 1-active oil/gas)


@dataclass
class WeatherSnapshot:
    """Weather data snapshot from Open-Meteo."""

    latitude: float
    longitude: float
    timestamp: datetime
    temperature_2m: float  # °C
    relative_humidity_2m: float  # %
    wind_speed_10m: float  # m/s
    wind_direction_10m: float  # degrees
    precipitation: float = 0.0  # mm
    cloud_cover: float = 0.0  # %
