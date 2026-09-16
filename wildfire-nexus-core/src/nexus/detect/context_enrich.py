"""
Context enrichment for thermal points.

Adds settlement proximity, weather data, and fire weather index.
"""

import logging
from math import atan2, cos, radians, sin, sqrt
from typing import Any

from ..db.connection import get_session_sync
from ..db.models import Settlement
from ..settings import get_settings
from .thermal_filter import _distance_km

logger = logging.getLogger(__name__)


def _get_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate bearing from point 1 to point 2 in degrees (0-360).
    
    Returns:
        Bearing in degrees (0 = North, 90 = East, etc.)
    """
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lon = radians(lon2 - lon1)
    
    x = sin(delta_lon) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(delta_lon)
    
    bearing = atan2(x, y)
    bearing_degrees = (bearing * 180 / 3.14159 + 360) % 360
    
    return bearing_degrees


def find_nearest_settlement(lat: float, lon: float) -> dict[str, Any] | None:
    """
    Find nearest settlement to a point.
    
    Args:
        lat: Latitude of point
        lon: Longitude of point
        
    Returns:
        Settlement dict with name, population, distance or None
    """
    session = get_session_sync()
    try:
        # Get all settlements (for 2000 settlements, linear scan is <5ms)
        settlements = session.query(Settlement).all()
        
        nearest = None
        min_distance = float('inf')
        
        for settlement in settlements:
            distance = _distance_km(lat, lon, settlement.latitude, settlement.longitude)
            
            if distance < min_distance:
                min_distance = distance
                nearest = {
                    "name": settlement.name,
                    "population": settlement.population,
                    "latitude": settlement.latitude,
                    "longitude": settlement.longitude,
                    "distance_km": round(distance, 2),
                }
        
        return nearest
    finally:
        session.close()


def check_wind_towards_settlement(
    settlement_lat: float,
    settlement_lon: float,
    fire_lat: float,
    fire_lon: float,
    wind_direction: float | None,
    wind_tolerance_degrees: int = 30,
) -> bool:
    """
    Check if wind is blowing from fire towards settlement.
    
    Args:
        settlement_lat: Settlement latitude
        settlement_lon: Settlement longitude
        fire_lat: Fire latitude
        fire_lon: Fire longitude
        wind_direction: Wind direction in degrees (0-360, direction wind is coming FROM)
        wind_tolerance_degrees: Tolerance angle in degrees
        
    Returns:
        True if wind is blowing towards settlement within tolerance
    """
    if wind_direction is None:
        return False
    
    # Calculate bearing from fire to settlement
    bearing_to_settlement = _get_bearing(fire_lat, fire_lon, settlement_lat, settlement_lon)
    
    # Wind direction is where wind comes FROM, so wind blows TO the opposite direction
    wind_blows_towards = (wind_direction + 180) % 360
    
    # Check if wind direction is within tolerance of bearing to settlement
    angle_diff = abs(wind_blows_towards - bearing_to_settlement)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    return angle_diff <= wind_tolerance_degrees


def calculate_fire_weather_index(
    temperature: float | None,
    humidity: float | None,
    wind_speed: float | None,
) -> int:
    """
    Calculate simplified fire weather index (0-5).
    
    Based on temperature, humidity, and wind speed.
    
    Args:
        temperature: Temperature in Celsius
        humidity: Relative humidity in percent
        wind_speed: Wind speed in m/s
        
    Returns:
        Index from 0 (low risk) to 5 (extreme risk)
    """
    score = 0
    
    # Temperature contribution (0-2 points)
    if temperature is not None:
        if temperature > 35:
            score += 2
        elif temperature > 25:
            score += 1
    
    # Humidity contribution (0-2 points)
    if humidity is not None:
        if humidity < 20:
            score += 2
        elif humidity < 40:
            score += 1
    
    # Wind contribution (0-1 points)
    if wind_speed is not None:
        if wind_speed > 10:
            score += 1
    
    return min(score, 5)


def enrich_point_with_context(point: dict[str, Any], weather: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Enrich thermal point with context data.
    
    Args:
        point: Thermal point dictionary
        weather: Optional weather data dictionary
        
    Returns:
        Enriched point dictionary with additional context fields
    """
    lat = point.get("latitude", 0)
    lon = point.get("longitude", 0)
    
    # Find nearest settlement
    nearest_settlement = find_nearest_settlement(lat, lon)
    
    # Extract weather data
    wind_speed = None
    wind_direction = None
    temperature = None
    humidity = None
    
    if weather:
        wind_speed = weather.get("wind_speed")
        wind_direction = weather.get("wind_direction")
        temperature = weather.get("temperature")
        humidity = weather.get("humidity")
    
    # Check if wind blows towards settlement
    wind_towards_settlement = False
    if nearest_settlement and wind_direction is not None:
        wind_towards_settlement = check_wind_towards_settlement(
            nearest_settlement["latitude"],
            nearest_settlement["longitude"],
            lat,
            lon,
            wind_direction,
        )
    
    # Calculate fire weather index
    fwi = calculate_fire_weather_index(temperature, humidity, wind_speed)
    
    # Build enriched point
    enriched = {
        **point,
        "nearest_settlement": nearest_settlement["name"] if nearest_settlement else None,
        "settlement_population": nearest_settlement["population"] if nearest_settlement else None,
        "distance_to_settlement_km": nearest_settlement["distance_km"] if nearest_settlement else None,
        "wind_towards_settlement": wind_towards_settlement,
        "fire_weather_index": fwi,
        "weather": weather,
    }
    
    logger.debug(
        f"Enriched point at ({lat}, {lon}): "
        f"settlement={enriched['nearest_settlement']}, "
        f"distance={enriched['distance_to_settlement_km']}km, "
        f"fwi={fwi}"
    )
    
    return enriched
