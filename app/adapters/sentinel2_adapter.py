# Sentinel-2 Adapter for STAC API
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import httpx

from app.core.schemas import Sentinel2Scene


logger = logging.getLogger(__name__)


class Sentinel2Adapter:
    """Адаптер для поиска снимков Sentinel-2 через STAC API"""
    
    def __init__(self, stac_api_url: str, token: str | None = None):
        self.stac_api_url = stac_api_url
        self.token = token
        self.search_endpoint = f"{stac_api_url}/search"
    
    async def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        limit: int = 10
    ) -> List[Sentinel2Scene]:
        """
        Поиск снимков Sentinel-2 через STAC API
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            start_date: ISO 8601 дата
            end_date: ISO 8601 дата
            max_cloud_cover: Максимальная облачность в %
            limit: Лимит результатов
            
        Returns:
            Список Sentinel2Scene
        """
        try:
            # STAC Query
            query = {
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{start_date}/{end_date}",
                "limit": limit,
                "filter": {
                    "op": "<",
                    "args": [
                        {"property": "eo:cloud_cover"},
                        max_cloud_cover
                    ]
                }
            }
            
            headers = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.search_endpoint,
                    json=query,
                    headers=headers
                )
                response.raise_for_status()
                
                data = response.json()
                return self._parse_stac_response(data)
                
        except httpx.HTTPError as e:
            logger.error(f"STAC API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching Sentinel-2: {e}")
            return []
    
    def _parse_stac_response(self, data: Dict[str, Any]) -> List[Sentinel2Scene]:
        """Распарсить ответ STAC API"""
        scenes = []
        
        features = data.get('features', [])
        for feature in features:
            try:
                props = feature.get('properties', {})
                scene_id = feature.get('id', '')
                
                # Парсинг даты
                dt_str = props.get('datetime', '')
                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                
                # Облачность
                cloud_cover = props.get('eo:cloud_cover', 0.0)
                
                # Tile ID
                tile_id = props.get('grid:code', None)
                
                scene = Sentinel2Scene(
                    scene_id=scene_id,
                    datetime=dt,
                    cloud_cover=cloud_cover,
                    tile_id=tile_id
                )
                scenes.append(scene)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid STAC feature: {e}")
                continue
        
        return scenes
    
    async def find_best_pair(
        self,
        bbox: List[float],
        reference_date: datetime,
        days_before: int = 90,
        days_after: int = 30,
        max_cloud_cover: float = 20.0
    ) -> tuple[Optional[Sentinel2Scene], Optional[Sentinel2Scene]]:
        """
        Найти лучшую пару снимков до/после пожара
        
        Returns:
            (pre_scene, post_scene) или (None, post_scene) если до-снимка нет
        """
        pre_start = reference_date - timedelta(days=days_before)
        pre_end = reference_date
        
        post_start = reference_date
        post_end = reference_date + timedelta(days=days_after)
        
        # Поиск снимков до и после
        pre_scenes = await self.search_scenes(
            bbox, 
            pre_start.isoformat(), 
            pre_end.isoformat(),
            max_cloud_cover
        )
        
        post_scenes = await self.search_scenes(
            bbox,
            post_start.isoformat(),
            post_end.isoformat(),
            max_cloud_cover
        )
        
        # Выбор лучших (наименьшая облачность)
        pre_scene = min(pre_scenes, key=lambda s: s.cloud_cover) if pre_scenes else None
        post_scene = min(post_scenes, key=lambda s: s.cloud_cover) if post_scenes else None
        
        return pre_scene, post_scene
    
    async def get_from_fixture(self, fixture_path: str) -> tuple[Sentinel2Scene, Sentinel2Scene]:
        """Загрузка фикстурных данных для демо"""
        import json
        from pathlib import Path
        
        path = Path(fixture_path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                pre = Sentinel2Scene(**data.get('pre', {}))
                post = Sentinel2Scene(**data.get('post', {}))
                return pre, post
        
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
