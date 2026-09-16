# Fire Detection Service
import logging
from typing import List, Dict, Type
from datetime import datetime

from app.core.schemas import FireCandidate, ConfidenceLevel, FireEvent as FireEventSchema
from app.adapters.base import BaseFireAdapter
from app.adapters.modis_adapter import ModisAdapter
from app.adapters.viirs_adapter import ViirsAdapter


logger = logging.getLogger(__name__)


class FireDetectionService:
    """Сервис детекции очагов пожаров по тепловым данным"""
    
    def __init__(
        self,
        adapters: Dict[str, BaseFireAdapter],
        min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL
    ):
        self.adapters = adapters
        self.min_confidence = min_confidence
    
    @classmethod
    def create_default(
        cls,
        firms_api_key: str | None = None,
        offline_mode: bool = False,
        cache_dir: str = "./data/cache"
    ) -> 'FireDetectionService':
        """Создать сервис с адаптерами по умолчанию"""
        
        adapters: Dict[str, BaseFireAdapter] = {}
        
        if firms_api_key:
            adapters['MODIS'] = ModisAdapter(api_key=firms_api_key, offline_mode=offline_mode, cache_dir=cache_dir)
            adapters['VIIRS'] = ViirsAdapter(api_key=firms_api_key, offline_mode=offline_mode, cache_dir=cache_dir)
        else:
            logger.warning("FIRMS API key not set, fire detection will use fixtures only")
        
        return cls(adapters=adapters, min_confidence=ConfidenceLevel.NOMINAL)
    
    async def detect_fires(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        sensors: List[str] | None = None
    ) -> List[FireCandidate]:
        """
        Детектировать очаги пожаров
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            start_date: ISO 8601 дата
            end_date: ISO 8601 дата
            sensors: Список сенсоров ['MODIS', 'VIIRS'] или None для всех
            
        Returns:
            Список FireCandidate
        """
        if sensors is None:
            sensors = list(self.adapters.keys())
        
        all_points: List[FireCandidate] = []
        
        for sensor_name in sensors:
            adapter = self.adapters.get(sensor_name)
            if not adapter:
                logger.warning(f"Adapter for {sensor_name} not found, skipping")
                continue
            
            try:
                points = await adapter.fetch_fire_points(bbox, start_date, end_date)
                
                # Если точек нет и работаем в офлайн-режиме, пробуем фикстуры
                if not points and hasattr(adapter, 'fetch_from_cache_or_fixture'):
                    points = await adapter.fetch_from_cache_or_fixture(bbox, start_date, end_date)
                
                all_points.extend(points)
                logger.info(f"Loaded {len(points)} points from {sensor_name}")
                
            except Exception as e:
                logger.error(f"Error detecting fires with {sensor_name}: {e}")
                continue
        
        # Фильтрация по confidence
        filtered_points = self._filter_by_confidence(all_points)
        logger.info(f"Total fire candidates after filtering: {len(filtered_points)}")
        
        return filtered_points
    
    def _filter_by_confidence(self, points: List[FireCandidate]) -> List[FireCandidate]:
        """Отфильтровать точки по уровню уверенности"""
        confidence_order = [ConfidenceLevel.LOW, ConfidenceLevel.NOMINAL, ConfidenceLevel.HIGH]
        min_idx = confidence_order.index(self.min_confidence)
        
        return [
            p for p in points 
            if confidence_order.index(p.confidence) >= min_idx
        ]
    
    async def get_all_available_sensors(self) -> List[str]:
        """Вернуть список доступных сенсоров"""
        return list(self.adapters.keys())
