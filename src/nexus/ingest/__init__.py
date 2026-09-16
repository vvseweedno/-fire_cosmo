"""Ingest module for loading external data."""

from .base import BoundingBox
from .firms import FirmsIngester, ThermalPoint
from .weather import WeatherIngester, WeatherSnapshot

__all__ = [
    "BoundingBox",
    "ThermalPoint",
    "WeatherSnapshot",
    "FirmsIngester",
    "WeatherIngester",
]
