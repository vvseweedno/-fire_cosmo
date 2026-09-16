"""Thermal point filtering module."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..ingest.base import ThermalPoint
from ..settings import get_settings

logger = logging.getLogger(__name__)


class ThermalFilter:
    """Filter for thermal anomaly points to remove false positives."""

    def __init__(self, whitelist_path: Optional[str] = None) -> None:
        """Initialize thermal filter.
        
        Args:
            whitelist_path: Path to industrial whitelist JSON file
        """
        self.settings = get_settings()
        self.whitelist_path = whitelist_path or str(
            Path(__file__).parent.parent.parent / "data" / "whitelist_industrial.json"
        )
        self._whitelist: list[dict] = []
        self._load_whitelist()

    def _load_whitelist(self) -> None:
        """Load industrial whitelist from file."""
        try:
            path = Path(self.whitelist_path)
            if path.exists():
                with open(path, "r") as f:
                    self._whitelist = json.load(f)
                logger.info(f"Loaded {len(self._whitelist)} industrial zones from whitelist")
            else:
                logger.warning(f"Whitelist file not found: {self.whitelist_path}")
                self._whitelist = []
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")
            self._whitelist = []

    def filter_point(self, point: ThermalPoint) -> tuple[bool, Optional[str]]:
        """Check if a thermal point passes the filter.
        
        Args:
            point: ThermalPoint to check
            
        Returns:
            Tuple of (passes_filter, rejection_reason)
        """
        # Check brightness threshold based on source
        min_brightness = (
            self.settings.min_brightness_viirs
            if point.source == "VIIRS"
            else self.settings.min_brightness_modis
        )
        
        if point.brightness < min_brightness:
            return False, f"Brightness {point.brightness}K below threshold {min_brightness}K"
        
        # Check confidence threshold
        if point.confidence < self.settings.min_confidence:
            return False, f"Confidence {point.confidence} below threshold {self.settings.min_confidence}"
        
        # Check FRP if available
        if point.frp is not None and point.frp < self.settings.min_frp:
            return False, f"FRP {point.frp}MW below threshold {self.settings.min_frp}MW"
        
        # Check industrial whitelist
        if self._is_in_whitelist(point):
            return False, "Point in industrial whitelist zone"
        
        return True, None

    def _is_in_whitelist(self, point: ThermalPoint) -> bool:
        """Check if point is within an industrial zone from whitelist."""
        if not self._whitelist:
            return False
        
        for zone in self._whitelist:
            zone_lat = zone.get("latitude", 0)
            zone_lon = zone.get("longitude", 0)
            radius_km = zone.get("radius_km", 2.0)
            
            distance = self._haversine_distance(
                point.latitude, point.longitude,
                zone_lat, zone_lon
            )
            
            if distance <= radius_km:
                logger.debug(f"Point matches industrial zone: {zone.get('name', 'unknown')}")
                return True
        
        return False

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers using Haversine formula."""
        from math import asin, cos, radians, sin, sqrt
        
        R = 6371.0  # Earth's radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * asin(sqrt(a))
        
        return R * c
