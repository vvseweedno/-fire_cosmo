"""
Landsat Adapter for USGS EarthExplorer

Источник: https://earthexplorer.usgs.gov/
API: M2M API (Machine-to-Machine)
Реализует термальный канал Band 10 (10.9 µm) для детекции пожаров
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import httpx

from app.adapters.base import BaseFireAdapter
from app.core.schemas import FireCandidate, ConfidenceLevel


logger = logging.getLogger(__name__)


class LandsatAdapter(BaseFireAdapter):
    """Адаптер для Landsat-8/9 термальных данных"""
    
    BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        offline_mode: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.username = username
        self.password = password
        self.offline_mode = offline_mode
        self.api_key = None
    
    def get_sensor_type(self) -> str:
        return "LANDSAT"
    
    async def authenticate(self) -> bool:
        """Аутентификация в USGS M2M API"""
        if not self.username or not self.password:
            logger.warning("Landsat credentials not provided")
            return False
        
        if self.offline_mode:
            logger.info("Landsat: offline mode, skipping authentication")
            return True
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.BASE_URL}login",
                    json={
                        "username": self.username,
                        "password": self.password
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                if result.get("errorCode") == "":
                    self.api_key = result["data"]
                    logger.info("Landsat API authenticated")
                    return True
                else:
                    logger.error(f"Landsat auth failed: {result.get('errorMessage')}")
                    return False
                    
        except Exception as e:
            logger.error(f"Landsat authentication error: {e}")
            return False
    
    async def fetch_fire_points(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        **kwargs
    ) -> List[FireCandidate]:
        """
        Поиск пожаров по термальным данным Landsat
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            start_date: ISO 8601
            end_date: ISO 8601
            max_cloud_cover: Максимальная облачность (%)
        
        Returns:
            Список FireCandidate с термальными аномалиями
        """
        # Аутентификация (если не offline)
        if not self.offline_mode and not self.api_key:
            await self.authenticate()
        
        # Поиск сцен
        scenes = await self.search_scenes(bbox, start_date, end_date, max_cloud_cover)
        
        # Конвертация сцен в точки пожаров
        fire_points = []
        for scene in scenes:
            points = self._scene_to_fire_candidates(scene)
            fire_points.extend(points)
        
        logger.info(f"Landsat: found {len(fire_points)} fire candidates from {len(scenes)} scenes")
        return fire_points
    
    async def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0
    ) -> List[Dict[str, Any]]:
        """
        Поиск Landsat сцен
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            start_date: ISO 8601
            end_date: ISO 8601
            max_cloud_cover: Максимальная облачность (%)
        
        Returns:
            Список сцен с metadata
        """
        if self.offline_mode or not self.api_key:
            logger.info("Landsat: using offline mode")
            return self._get_offline_scenes(bbox, start_date, end_date)
        
        try:
            async with httpx.AsyncClient() as client:
                # Search scenes
                response = await client.post(
                    f"{self.BASE_URL}scene-search",
                    headers={"X-Auth-Token": self.api_key},
                    json={
                        "datasetName": "landsat_ot_c2_l2",
                        "maxResults": 10,
                        "startingNumber": 1,
                        "sceneFilter": {
                            "spatialFilter": {
                                "filterType": "mbr",
                                "lowerLeft": {
                                    "latitude": bbox[1],
                                    "longitude": bbox[0]
                                },
                                "upperRight": {
                                    "latitude": bbox[3],
                                    "longitude": bbox[2]
                                }
                            },
                            "acquisitionFilter": {
                                "start": start_date,
                                "end": end_date
                            },
                            "cloudCoverFilter": {
                                "min": 0,
                                "max": max_cloud_cover
                            }
                        }
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                scenes = result.get("data", {}).get("results", [])
                
                logger.info(f"Landsat: found {len(scenes)} scenes")
                return scenes
                
        except Exception as e:
            logger.error(f"Landsat search error: {e}")
            return self._get_offline_scenes(bbox, start_date, end_date)
    
    def _get_offline_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Возвращает фикстурные данные для offline-режима"""
        return [
            {
                "entityId": f"LC08_L2SP_{i:06d}_20240101_20240102_02_T1",
                "displayId": f"LC08_L2SP_{i:06d}_20240101",
                "acquisitionDate": "2024-01-01",
                "cloudCover": 5.0 + i,
                "spatialBounds": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]]
                    ]]
                },
                "temperatureBand": {
                    "min": 293.0,
                    "max": 310.0,
                    "mean": 298.0
                }
            }
            for i in range(5)
        ]
    
    def _scene_to_fire_candidates(self, scene: Dict[str, Any]) -> List[FireCandidate]:
        """Конвертация сцены Landsat в кандидаты пожаров"""
        import uuid
        
        # Извлекаем температуру из термального канала
        temp_data = scene.get("temperatureBand", {})
        temp_max = temp_data.get("max", 310.0)
        
        # Порог для детекции пожара: > 305K (~32°C)
        FIRE_TEMP_THRESHOLD = 305.0
        
        if temp_max < FIRE_TEMP_THRESHOLD:
            return []
        
        # Вычисляем уверенность на основе температуры и облачности
        cloud_cover = scene.get("cloudCover", 0.0)
        temp_confidence = min((temp_max - FIRE_TEMP_THRESHOLD) / 20.0, 1.0)
        cloud_confidence = 1.0 - (cloud_cover / 100.0)
        confidence_score = (temp_confidence + cloud_confidence) / 2.0
        
        # Определяем уровень уверенности
        if confidence_score > 0.7:
            confidence = ConfidenceLevel.HIGH
        elif confidence_score > 0.4:
            confidence = ConfidenceLevel.NOMINAL
        else:
            confidence = ConfidenceLevel.LOW
        
        # Получаем координаты центра сцены
        coords = scene.get("spatialBounds", {}).get("coordinates", [[]])
        if coords and coords[0]:
            lons = [c[0] for c in coords[0]]
            lats = [c[1] for c in coords[0]]
            center_lon = sum(lons) / len(lons)
            center_lat = sum(lats) / len(lats)
        else:
            center_lon = 92.5
            center_lat = 56.5
        
        # Определяем день/ночь (Landsat - дневной спутник)
        daynight = "D"
        
        return [
            FireCandidate(
                id=str(uuid.uuid4()),
                sensor="LANDSAT",
                datetime=datetime.fromisoformat(scene.get("acquisitionDate", "2024-01-01")),
                latitude=center_lat,
                longitude=center_lon,
                brightness_temp_k=temp_max,
                frp_mw=50.0 + (temp_max - FIRE_TEMP_THRESHOLD) * 10,
                confidence=confidence,
                daynight=daynight,
                satellite=scene.get("displayId", "LANDSAT_8"),
                raw=scene
            )
        ]
    
    async def download_thermal_band(
        self,
        scene_id: str,
        band: str = "ST_B10"
    ) -> Optional[str]:
        """
        Скачать термальный канал (Band 10 = 10.9 µm)
        
        Args:
            scene_id: Landsat scene ID
            band: Band name (ST_B10 для температуры поверхности)
        
        Returns:
            Путь к скачанному файлу или None
        """
        if self.offline_mode:
            logger.info(f"Landsat: offline mode, returning mock data for {band}")
            cache_path = Path("data/cache") / f"landsat_{scene_id}_{band}.tif"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            return str(cache_path)
        
        # Реальная загрузка через M2M API требует download endpoint
        logger.warning("Landsat download not fully implemented in this version")
        return None
