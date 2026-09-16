"""
Ingest module for loading external data sources.
"""

from .base import BoundingBox
from .firms import fetch_firms_data, process_firms_points
from .weather import fetch_weather_data

__all__ = [
    "BoundingBox",
    "fetch_firms_data",
    "process_firms_points",
    "fetch_weather_data",
]
