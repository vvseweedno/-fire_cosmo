"""
NASA FIRMS (Fire Information for Resource Management System) data loader.

Fetches thermal anomaly data from NASA FIRMS API.
API: https://firms.modaps.eosdis.nasa.gov/api/
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from ..settings import get_settings
from .base import BoundingBox

logger = logging.getLogger(__name__)

# FIRMS API endpoints
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area"

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_key(region: BoundingBox, date_range: tuple[str, str], source: str) -> str:
    """Generate cache key from parameters."""
    key_str = f"{region.to_firms_format()}_{date_range}_{source}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_from_cache(cache_key: str, ttl_hours: int) -> list[dict] | None:
    """Load data from cache if valid."""
    cache_file = CACHE_DIR / f"firms_{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file) as f:
            data = json.load(f)
        
        # Check cache age
        cached_at = datetime.fromisoformat(data["_cached_at"])
        if datetime.utcnow() - cached_at > timedelta(hours=ttl_hours):
            logger.debug(f"Cache expired for key {cache_key}")
            return None
        
        logger.debug(f"Loaded {len(data.get('points', []))} points from cache")
        return data.get("points")
    except Exception as e:
        logger.warning(f"Cache load error: {e}")
        return None


def _save_to_cache(cache_key: str, points: list[dict]) -> None:
    """Save data to cache."""
    cache_file = CACHE_DIR / f"firms_{cache_key}.json"
    
    data = {
        "_cached_at": datetime.utcnow().isoformat(),
        "points": points,
    }
    
    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
        logger.debug(f"Saved {len(points)} points to cache")
    except Exception as e:
        logger.warning(f"Cache save error: {e}")


async def fetch_firms_data(
    region: BoundingBox,
    date_range: tuple[datetime, datetime],
    source: str = "VIIRS",
) -> list[dict[str, Any]]:
    """
    Fetch thermal points from NASA FIRMS API.
    
    Args:
        region: Geographic bounding box
        date_range: (start_date, end_date) tuple
        source: Satellite source ('VIIRS' or 'MODIS')
    
    Returns:
        List of thermal point dictionaries
        
    Raises:
        httpx.HTTPError: If API request fails after retries
    """
    settings = get_settings()
    
    if not settings.firms_map_key:
        logger.warning("FIRMS_MAP_KEY not set, returning empty result")
        return []
    
    # Format dates
    start_str = date_range[0].strftime("%Y-%m-%d")
    end_str = date_range[1].strftime("%Y-%m-%d")
    date_tuple = (start_str, end_str)
    
    # Check cache
    cache_ttl = settings.ingest.get("cache_ttl_hours", 6)
    cache_key = _get_cache_key(region, date_tuple, source)
    cached_data = _load_from_cache(cache_key, cache_ttl)
    if cached_data is not None:
        return cached_data
    
    # Build API URL
    # FIRMS API format: /api/area/{source}/{lat},{lon},{lat},{lon}/{dates}/{MAP_KEY}
    url = (
        f"{FIRMS_BASE_URL}/{source}/"
        f"{region.to_firms_format()}/"
        f"{start_str}_{end_str}/"
        f"{settings.firms_map_key}"
    )
    
    # Request with retries
    max_retries = 3
    retry_delays = [1, 2, 4]  # Exponential backoff
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                
                if response.status_code == 429:
                    logger.warning("FIRMS API rate limited, waiting 60s")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(60)
                        continue
                
                response.raise_for_status()
                data = response.json()
                
                # Parse CSV response (FIRMS returns CSV)
                points = _parse_firms_csv(data, source)
                
                # Cache result
                _save_to_cache(cache_key, points)
                
                logger.info(f"Fetched {len(points)} thermal points from FIRMS")
                return points
                
        except httpx.HTTPError as e:
            logger.warning(f"FIRMS API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delays[attempt])
            else:
                logger.error(f"FIRMS API failed after {max_retries} attempts")
                raise
    
    return []


def _parse_firms_csv(csv_data: str, source: str) -> list[dict[str, Any]]:
    """
    Parse FIRMS CSV response into list of dictionaries.
    
    Args:
        csv_data: Raw CSV string from FIRMS API
        source: Satellite source name
        
    Returns:
        List of thermal point dictionaries
    """
    if not csv_data or isinstance(csv_data, dict):
        # Handle case where API returns error message
        logger.warning(f"FIRMS returned non-CSV data: {csv_data}")
        return []
    
    lines = csv_data.strip().split("\n")
    if len(lines) < 2:
        return []
    
    # FIRMS VIIRS columns:
    # latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, confidence, version, bright_t31, frp, daynight
    headers = lines[0].split(",")
    
    points = []
    for line in lines[1:]:
        try:
            values = line.split(",")
            if len(values) < len(headers):
                continue
            
            point = dict(zip(headers, values))
            
            # Convert types
            point["latitude"] = float(point.get("latitude", 0))
            point["longitude"] = float(point.get("longitude", 0))
            point["brightness"] = float(point.get("brightness", 0))
            point["confidence"] = float(point.get("confidence", 0)) / 100.0  # Convert to 0-1
            point["frp"] = float(point.get("frp", 0)) if point.get("frp") else None
            point["source"] = source
            point["detected_at"] = f"{point.get('acq_date', '')} {point.get('acq_time', '0')}:00"
            
            points.append(point)
        except (ValueError, KeyError) as e:
            logger.debug(f"Skipping malformed row: {e}")
            continue
    
    return points


def process_firms_points(raw_points: list[dict]) -> list[dict]:
    """
    Process raw FIRMS points into standardized format.
    
    Args:
        raw_points: Raw points from FIRMS API
        
    Returns:
        List of processed thermal point dictionaries
    """
    processed = []
    
    for point in raw_points:
        try:
            processed_point = {
                "latitude": float(point.get("latitude", 0)),
                "longitude": float(point.get("longitude", 0)),
                "brightness": float(point.get("brightness", 0)),
                "confidence": float(point.get("confidence", 0)),
                "frp": float(point["frp"]) if point.get("frp") else None,
                "source": point.get("source", "VIIRS"),
                "detected_at": point.get("detected_at", datetime.utcnow().isoformat()),
                "raw_payload": point,
            }
            processed.append(processed_point)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to process point: {e}")
            continue
    
    logger.info(f"Processed {len(processed)}/{len(raw_points)} FIRMS points")
    return processed


# Import asyncio for sleep
import asyncio
