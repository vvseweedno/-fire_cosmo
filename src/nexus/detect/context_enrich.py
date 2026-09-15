"""Context enrichment module for thermal points."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from shapely.geometry import Point

from ..db.models import Settlement
from ..ingest.base import ThermalPoint, WeatherSnapshot
from ..settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class EnrichedPoint:
    """Thermal point enriched with context data."""

    # Original point data
    latitude: float
    longitude: float
    brightness: float
    confidence: float
    source: str
    detected_at: str
    frp: Optional[float] = None
    
    # Context data
    nearest_settlement: Optional[str] = None
    distance_to_settlement_km: Optional[float] = None
    settlement_population: Optional[int] = None
    wind_towards_settlement: bool = False
    fire_weather_index: int = 0  # 0-5 scale
    
    # Weather data
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    
    # Additional context
    raw_point: Optional[ThermalPoint] = field(default=None, repr=False)
    weather: Optional[WeatherSnapshot] = field(default=None, repr=False)


class ContextEnricher:
    """Enriches thermal points with contextual information."""

    def __init__(self, settlements: Optional[list[Settlement]] = None) -> None:
        """Initialize context enricher.
        
        Args:
            settlements: List of settlements for proximity analysis
        """
        self.settings = get_settings()
        self.settlements = settlements or []

    async def enrich(self, point: ThermalPoint, weather: Optional[WeatherSnapshot] = None) -> EnrichedPoint:
        """Enrich a thermal point with context.
        
        Args:
            point: ThermalPoint to enrich
            weather: Optional weather snapshot
            
        Returns:
            EnrichedPoint with all context data
        """
        enriched = EnrichedPoint(
            latitude=point.latitude,
            longitude=point.longitude,
            brightness=point.brightness,
            confidence=point.confidence,
            source=point.source,
            detected_at=point.detected_at.isoformat(),
            frp=point.frp,
            raw_point=point,
            weather=weather,
        )
        
        # Find nearest settlement
        if self.settlements:
            nearest = self._find_nearest_settlement(point.latitude, point.longitude)
            if nearest:
                enriched.nearest_settlement = nearest.name
                enriched.distance_to_settlement_km = nearest.distance_km
                enriched.settlement_population = nearest.population
                
                # Check wind direction towards settlement
                if weather and nearest.distance_km <= self.settings.warning_distance_km:
                    enriched.wind_towards_settlement = self._is_wind_towards_settlement(
                        weather, point.latitude, point.longitude,
                        nearest.latitude, nearest.longitude
                    )
        
        # Calculate fire weather index
        if weather:
            enriched.temperature = weather.temperature_2m
            enriched.humidity = weather.relative_humidity_2m
            enriched.wind_speed = weather.wind_speed_10m
            enriched.wind_direction = weather.wind_direction_10m
            enriched.fire_weather_index = self._calculate_fire_weather_index(weather)
        
        return enriched

    def _find_nearest_settlement(self, lat: float, lon: float) -> Optional["_SettlementDistance"]:
        """Find the nearest settlement to a point."""
        if not self.settlements:
            return None
        
        point = Point(lon, lat)
        min_distance = float("inf")
        nearest = None
        
        for settlement in self.settlements:
            settle_point = Point(settlement.longitude, settlement.latitude)
            distance_km = self._haversine_distance(lat, lon, settlement.latitude, settlement.longitude)
            
            if distance_km < min_distance:
                min_distance = distance_km
                nearest = _SettlementDistance(
                    id=settlement.id,
                    name=settlement.name,
                    population=settlement.population,
                    latitude=settlement.latitude,
                    longitude=settlement.longitude,
                    distance_km=distance_km,
                )
        
        return nearest

    def _is_wind_towards_settlement(
        self,
        weather: WeatherSnapshot,
        fire_lat: float,
        fire_lon: float,
        settle_lat: float,
        settle_lon: float,
    ) -> bool:
        """Check if wind is blowing from fire towards settlement."""
        if weather.wind_speed_10m < 1.0:  # Wind too weak
            return False
        
        # Calculate bearing from fire to settlement
        from math import atan2, cos, radians, sin
        
        fire_rad = (radians(fire_lat), radians(fire_lon))
        settle_rad = (radians(settle_lat), radians(settle_lon))
        
        delta_lon = settle_rad[1] - fire_rad[1]
        
        y = sin(delta_lon) * cos(settle_rad[0])
        x = cos(fire_rad[0]) * sin(settle_rad[0]) - sin(fire_rad[0]) * cos(settle_rad[0]) * cos(delta_lon)
        
        bearing_to_settlement = (atan2(y, x) * 180 / 3.14159 + 360) % 360
        
        # Wind direction is where wind comes FROM, so we need opposite direction
        wind_blowing_to = (weather.wind_direction_10m + 180) % 360
        
        # Check if wind is blowing towards settlement (within ±30°)
        angle_diff = abs(wind_blowing_to - bearing_to_settlement)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        return angle_diff <= 30

    def _calculate_fire_weather_index(self, weather: WeatherSnapshot) -> int:
        """Calculate simplified fire weather index (0-5)."""
        index = 0
        
        # Temperature contribution
        if weather.temperature_2m > 30:
            index += 2
        elif weather.temperature_2m > 25:
            index += 1
        
        # Humidity contribution (lower = more dangerous)
        if weather.relative_humidity_2m < 20:
            index += 2
        elif weather.relative_humidity_2m < 40:
            index += 1
        
        # Wind contribution
        if weather.wind_speed_10m > 10:
            index += 2
        elif weather.wind_speed_10m > 5:
            index += 1
        
        # Precipitation (recent rain reduces risk)
        if weather.precipitation > 5:
            index = max(0, index - 1)
        
        return min(5, max(0, index))

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        from math import asin, cos, radians, sin, sqrt
        
        R = 6371.0  # Earth's radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * asin(sqrt(a))
        
        return R * c


@dataclass
class _SettlementDistance:
    """Internal data class for settlement with distance."""
    
    id: int
    name: str
    population: int
    latitude: float
    longitude: float
    distance_km: float
