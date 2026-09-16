# MODIS Adapter for NASA FIRMS data
import logging
import csv
from io import StringIO
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

import httpx

from app.adapters.base import BaseFireAdapter
from app.core.schemas import FireCandidate, ConfidenceLevel


logger = logging.getLogger(__name__)


class ModisAdapter(BaseFireAdapter):
    """Адаптер для данных MODIS (NASA FIRMS)"""
    
    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/country"
    
    def get_sensor_type(self) -> str:
        return "MODIS"
    
    async def fetch_fire_points(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        country: str = "USA"  # По умолчанию USA, можно менять
    ) -> List[FireCandidate]:
        """
        Загрузить данные MODIS из NASA FIRMS
        
        Примечание: FIRMS API использует CSV формат и требует MAP_KEY
        """
        # Офлайн режим - загрузка из фикстур
        if self.offline_mode:
            return await self._load_from_fixture(bbox, start_date, end_date)
        
        if not self.api_key:
            logger.warning("MODIS API key not set, returning empty list")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # FIRMS API endpoint для загрузки за период
                url = f"{self.base_url}/{country}/modis/{start_date}/{end_date}"
                params = {"key": self.api_key}
                
                response = await client.get(url, params=params)
                
                if response.status_code == 429:
                    logger.warning("FIRMS API rate limit, waiting...")
                    await asyncio.sleep(60)
                    return await self.fetch_fire_points(bbox, start_date, end_date, country)
                
                response.raise_for_status()
                
                # Парсинг CSV ответа
                return self._parse_csv_response(response.text, bbox)
                
        except httpx.HTTPError as e:
            self._log_error(f"HTTP error fetching MODIS data", e)
            return []
        except Exception as e:
            self._log_error(f"Unexpected error fetching MODIS data", e)
            return []
    
    async def _load_from_fixture(self, bbox: List[float], start_date: str, end_date: str) -> List[FireCandidate]:
        """Загрузка данных из локальной фикстуры"""
        fixture_file = "modis_demo.csv"
        fixture_path = self._get_fixture_path(fixture_file)
        
        if not fixture_path.exists():
            logger.error(
                f"MODIS fixture not found at {fixture_path}. "
                f"Run: python scripts/download_demo_data.py --offline"
            )
            raise FileNotFoundError(
                f"MODIS fixture '{fixture_file}' not found. "
                f"Please run: python scripts/download_demo_data.py --offline"
            )
        
        logger.info(f"Loading MODIS data from fixture: {fixture_path}")
        
        points = []
        with open(fixture_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                try:
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                    
                    # Фильтрация по bbox
                    if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                        continue
                    
                    brightness = float(row.get('bright', 0))
                    confidence_val = row.get('confidence', 'nominal').lower()
                    
                    # Маппинг confidence
                    confidence_map = {
                        'low': ConfidenceLevel.LOW,
                        'nominal': ConfidenceLevel.NOMINAL,
                        'high': ConfidenceLevel.HIGH
                    }
                    confidence = confidence_map.get(confidence_val, ConfidenceLevel.NOMINAL)
                    
                    # Парсинг даты
                    acq_date = row.get('acq_date', '2024-01-01')
                    acq_time = row.get('acq_time', '1200')
                    satellite = row.get('satellite', 'Terra')
                    daynight = row.get('daynight', 'D')
                    
                    dt = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M")
                    
                    point = FireCandidate(
                        id=f"modis_{row.get('id', i)}",
                        sensor="MODIS",
                        datetime=dt,
                        latitude=lat,
                        longitude=lon,
                        brightness_temp_k=brightness,
                        frp_mw=float(row['frp']) if row.get('frp') else None,
                        confidence=confidence,
                        daynight=daynight,
                        satellite=satellite,
                        source="demo_fixture",
                        raw=dict(row)
                    )
                    points.append(point)
                    
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid MODIS fixture row: {e}")
                    continue
        
        logger.info(f"Loaded {len(points)} MODIS points from fixture")
        return points
    
    def _parse_csv_response(self, csv_data: str, bbox: List[float]) -> List[FireCandidate]:
        """Распарсить CSV ответ от FIRMS"""
        points = []
        reader = csv.DictReader(StringIO(csv_data))
        
        for row in reader:
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                
                # Фильтрация по bbox
                if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                    continue
                
                brightness = float(row.get('brightness', 0))
                confidence_val = row.get('confidence', 'nominal').lower()
                
                # Маппинг confidence
                confidence_map = {
                    'low': ConfidenceLevel.LOW,
                    'nominal': ConfidenceLevel.NOMINAL,
                    'high': ConfidenceLevel.HIGH
                }
                confidence = confidence_map.get(confidence_val, ConfidenceLevel.NOMINAL)
                
                # Парсинг даты
                acq_date = row.get('acq_date', '2024-01-01')
                acq_time = row.get('acq_time', '1200')
                satellite = row.get('satellite', 'Terra')
                daynight = row.get('daynight', 'D')
                
                dt = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M")
                
                point = FireCandidate(
                    id=f"modis_{row.get('id', len(points))}",
                    sensor="MODIS",
                    datetime=dt,
                    latitude=lat,
                    longitude=lon,
                    brightness_temp_k=brightness,
                    frp_mw=float(row['frp']) if row.get('frp') else None,
                    confidence=confidence,
                    daynight=daynight,
                    satellite=satellite,
                    raw=dict(row)
                )
                points.append(point)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid MODIS row: {e}")
                continue
        
        return points
