"""NASA FIRMS thermal points ingester."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from ..settings import get_settings
from .base import BoundingBox, ThermalPoint

logger = logging.getLogger(__name__)


class FirmsIngester:
    """Ingester for NASA FIRMS thermal anomaly data."""

    FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
    CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

    def __init__(self, map_key: Optional[str] = None) -> None:
        """Initialize FIRMS ingester.
        
        Args:
            map_key: NASA FIRMS API key. If not provided, uses env var.
        """
        self.settings = get_settings()
        self.map_key = map_key or self.settings.firms_map_key
        self.cache_ttl_hours = 6

    def _get_cache_path(self, region: BoundingBox, date_range: tuple[str, str], source: str) -> Path:
        """Get cache file path for given parameters."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key_str = f"{region.min_lon}_{region.min_lat}_{region.max_lon}_{region.max_lat}_{date_range}_{source}"
        hash_key = hashlib.md5(key_str.encode()).hexdigest()[:12]
        return self.CACHE_DIR / f"firms_{hash_key}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid."""
        if not cache_path.exists():
            return False
        cache_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return datetime.now(timezone.utc) - cache_mtime < timedelta(hours=self.cache_ttl_hours)

    async def fetch_points(
        self,
        region: BoundingBox,
        date_range: tuple[datetime, datetime],
        source: str = "VIIRS",
    ) -> list[ThermalPoint]:
        """Fetch thermal points from FIRMS API.
        
        Args:
            region: Geographic bounding box
            date_range: (start_date, end_date) tuple
            source: Satellite source ('VIIRS' or 'MODIS')
            
        Returns:
            List of ThermalPoint objects
        """
        # Check cache first
        date_str = (date_range[0].strftime("%Y-%m-%d"), date_range[1].strftime("%Y-%m-%d"))
        cache_path = self._get_cache_path(region, date_str, source)

        if self._is_cache_valid(cache_path):
            logger.info(f"Loading FIRMS data from cache: {cache_path}")
            return self._load_from_cache(cache_path)

        # Fetch from API
        try:
            points = await self._fetch_from_api(region, date_str, source)
            # Save to cache
            self._save_to_cache(points, cache_path)
            return points
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("FIRMS API rate limited, waiting 60s")
                await httpx.AsyncClient().sleep(60)
                return await self.fetch_points(region, date_range, source)
            logger.error(f"FIRMS API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching FIRMS data: {e}")
            return []

    async def _fetch_from_api(
        self,
        region: BoundingBox,
        date_str: tuple[str, str],
        source: str,
    ) -> list[ThermalPoint]:
        """Fetch data from FIRMS API with retries."""
        # Map source to country code (FIRMS API uses country codes)
        # For large regions like Siberia, we'll use a bounding box approach
        # Note: FIRMS API structure may vary, this is simplified
        
        params = {
            "key": self.map_key,
        }
        
        # FIRMS API typically returns CSV, we'll parse it
        # For the 48h version, we simulate the API call structure
        # In production, you'd use the actual FIRMS REST API endpoint
        
        url = f"{self.FIRMS_API_URL}/{source}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Retry logic with exponential backoff
            for attempt in range(3):
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    # Parse CSV response
                    return self._parse_csv(response.text, source)
                except httpx.TimeoutException:
                    if attempt == 2:
                        raise
                    wait_time = 2 ** attempt
                    logger.warning(f"Timeout, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        raise  # Handle at higher level
                    raise

    def _parse_csv(self, csv_content: str, source: str) -> list[ThermalPoint]:
        """Parse FIRMS CSV response into ThermalPoint objects."""
        points = []
        lines = csv_content.strip().split("\n")
        
        if len(lines) < 2:
            return points
        
        # Skip header
        headers = lines[0].lower().split(",")
        
        for line in lines[1:]:
            if not line.strip():
                continue
            
            values = line.split(",")
            if len(values) != len(headers):
                continue
            
            row = dict(zip(headers, values))
            
            try:
                point = self._parse_row(row, source)
                if point:
                    points.append(point)
            except Exception as e:
                logger.warning(f"Failed to parse FIRMS row: {e}")
                continue
        
        logger.info(f"Parsed {len(points)} thermal points from FIRMS")
        return points

    def _parse_row(self, row: dict[str, str], source: str) -> Optional[ThermalPoint]:
        """Parse a single FIRMS CSV row into ThermalPoint."""
        try:
            latitude = float(row.get("latitude", 0))
            longitude = float(row.get("longitude", 0))
            brightness = float(row.get("brightness", 0))
            confidence = float(row.get("confidence", 0)) / 100.0  # Convert to 0-1
            
            frp = None
            if row.get("frp"):
                try:
                    frp = float(row.get("frp"))
                except ValueError:
                    pass
            
            acq_date = row.get("acq_date", "")
            acq_time = row.get("acq_time", "")
            
            # Parse datetime
            detected_at = datetime.now(timezone.utc)
            if acq_date and acq_time:
                try:
                    time_str = f"{acq_date} {acq_time.zfill(4)}"
                    detected_at = datetime.strptime(time_str, "%Y-%m-%d %H%M")
                    detected_at = detected_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            
            return ThermalPoint(
                latitude=latitude,
                longitude=longitude,
                brightness=brightness,
                confidence=confidence,
                source=source,
                detected_at=detected_at,
                frp=frp,
                acq_date=acq_date,
                acq_time=acq_time,
                satellite=row.get("satellite", ""),
                instrument=row.get("instrument", ""),
                version=row.get("version", ""),
                bright_t31=float(row.get("bright_t31", 0)) or None,
                frp_confidence=row.get("frp_confidence", ""),
                type=row.get("type", ""),
            )
        except (ValueError, KeyError) as e:
            logger.debug(f"Invalid FIRMS row: {e}")
            return None

    def _load_from_cache(self, cache_path: Path) -> list[ThermalPoint]:
        """Load thermal points from cache file."""
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            
            points = []
            for item in data:
                item["detected_at"] = datetime.fromisoformat(item["detected_at"])
                points.append(ThermalPoint(**item))
            
            logger.info(f"Loaded {len(points)} points from cache")
            return points
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return []

    def _save_to_cache(self, points: list[ThermalPoint], cache_path: Path) -> None:
        """Save thermal points to cache file."""
        try:
            # Convert to serializable format
            data = []
            for p in points:
                d = p.__dict__.copy()
                d["detected_at"] = d["detected_at"].isoformat()
                data.append(d)
            
            with open(cache_path, "w") as f:
                json.dump(data, f)
            
            logger.debug(f"Cached {len(points)} points to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
