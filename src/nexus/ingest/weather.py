"""Weather data ingester from Open-Meteo API."""

import asyncio
import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import httpx

from .base import WeatherSnapshot

logger = logging.getLogger(__name__)


class WeatherIngester:
    """Ingester for weather data from Open-Meteo API."""

    WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self) -> None:
        """Initialize weather ingester."""
        self.cache: dict[tuple[float, float, int], WeatherSnapshot] = {}

    async def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[WeatherSnapshot]:
        """Fetch weather data for a specific location.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            timestamp: Timestamp (defaults to now). Weather is cached by hour.
            
        Returns:
            WeatherSnapshot or None if failed
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Round to hour for caching
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
        cache_key = (round(latitude, 2), round(longitude, 2), int(hour_key.timestamp()))
        
        # Check cache
        if cache_key in self.cache:
            logger.debug(f"Weather cache hit for ({latitude}, {longitude})")
            return self.cache[cache_key]
        
        try:
            weather = await self._fetch_from_api(latitude, longitude)
            if weather:
                self.cache[cache_key] = weather
            return weather
        except Exception as e:
            logger.warning(f"Failed to fetch weather for ({latitude}, {longitude}): {e}")
            # Degrade gracefully - return None, system continues without weather
            return None

    async def _fetch_from_api(self, latitude: float, longitude: float) -> Optional[WeatherSnapshot]:
        """Fetch weather from Open-Meteo API."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "cloud_cover",
            ],
            "timezone": "UTC",
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Retry logic
            for attempt in range(2):
                try:
                    response = await client.get(self.WEATHER_API_URL, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    return self._parse_response(data, latitude, longitude)
                except httpx.TimeoutException:
                    if attempt == 1:
                        raise
                    wait_time = 2 ** attempt
                    logger.warning(f"Weather API timeout, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Weather API error: {e}")
                    return None
        
        return None

    def _parse_response(self, data: dict, latitude: float, longitude: float) -> Optional[WeatherSnapshot]:
        """Parse Open-Meteo API response."""
        try:
            current = data.get("current", {})
            if not current:
                return None
            
            return WeatherSnapshot(
                latitude=latitude,
                longitude=longitude,
                timestamp=datetime.now(timezone.utc),
                temperature_2m=current.get("temperature_2m", 0.0),
                relative_humidity_2m=current.get("relative_humidity_2m", 0.0),
                wind_speed_10m=current.get("wind_speed_10m", 0.0),
                wind_direction_10m=current.get("wind_direction_10m", 0.0),
                precipitation=current.get("precipitation", 0.0),
                cloud_cover=current.get("cloud_cover", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to parse weather response: {e}")
            return None
