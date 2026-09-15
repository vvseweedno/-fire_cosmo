"""
Weather data loader from Open-Meteo API.

Fetches weather conditions for context enrichment (wind, temperature, humidity).
API: https://open-meteo.com/
"""

import logging
from datetime import datetime
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo API endpoint
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# Simple in-memory cache: {(lat, lon, hour): (data, timestamp)}
_weather_cache: dict[tuple, tuple[dict, datetime]] = {}


async def fetch_weather_data(lat: float, lon: float) -> dict[str, Any] | None:
    """
    Fetch weather data for a specific location.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with weather parameters or None if unavailable
        
    Note:
        This is an optional module - system degrades gracefully if unavailable
    """
    try:
        # Check cache first (cache by hour)
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        cache_key = (round(lat, 2), round(lon, 2), current_hour)
        
        if cache_key in _weather_cache:
            cached_data, cached_at = _weather_cache[cache_key]
            if datetime.utcnow() - cached_at < timedelta(hours=1):
                logger.debug(f"Weather cache hit for ({lat}, {lon})")
                return cached_data
        
        # Fetch from API
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "timezone": "auto",
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Parse response
        current = data.get("current", {})
        weather = {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "fetched_at": datetime.utcnow().isoformat(),
        }
        
        # Cache result
        _weather_cache[cache_key] = (weather, datetime.utcnow())
        
        logger.debug(f"Fetched weather for ({lat}, {lon}): {weather}")
        return weather
        
    except Exception as e:
        logger.warning(f"Weather fetch failed for ({lat}, {lon}): {e}")
        # Graceful degradation - return None, system continues without weather
        return None


def get_cached_weather(lat: float, lon: float) -> dict[str, Any] | None:
    """
    Get cached weather data without fetching.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Cached weather data or None
    """
    current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    cache_key = (round(lat, 2), round(lon, 2), current_hour)
    
    if cache_key in _weather_cache:
        return _weather_cache[cache_key][0]
    
    return None


# Import timedelta
from datetime import timedelta
