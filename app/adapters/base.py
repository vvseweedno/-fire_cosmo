# Base adapter for fire data sources
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from app.core.schemas import FireCandidate


class BaseFireAdapter(ABC):
    """Базовый класс адаптера для источников данных о пожарах"""

    def __init__(self, cache_dir: str = "./data/cache", offline_mode: bool = False, fixtures_dir: str = "./data/fixtures", api_key: Optional[str] = None):
        self.cache_dir = cache_dir
        self.offline_mode = offline_mode
        self.fixtures_dir = Path(fixtures_dir)
        self.api_key = api_key

    @abstractmethod
    async def fetch_fire_points(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> List[FireCandidate]:
        """
        Получить точки термических аномалий

        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            start_date: ISO 8601 дата
            end_date: ISO 8601 дата

        Returns:
            Список нормализованных FireCandidate
        """
        pass

    @abstractmethod
    def get_sensor_type(self) -> str:
        """Вернуть тип сенсора (MODIS, VIIRS, LANDSAT)"""
        pass

    def _cache_key(self, bbox: List[float], start_date: str, end_date: str) -> str:
        """Сгенерировать ключ кэша"""
        import hashlib
        key_str = f"{bbox}_{start_date}_{end_date}_{self.get_sensor_type()}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _log_error(self, message: str, error: Exception):
        """Логирование ошибки с маскировкой чувствительных данных"""
        import logging
        logger = logging.getLogger(__name__)
        # Маскировать чувствительные данные
        error_str = str(error)
        if self.api_key and self.api_key in error_str:
            error_str = error_str.replace(self.api_key, "***MASKED***")
        logger.error(f"{self.get_sensor_type()} adapter error: {message} - {error_str}")

    def _get_fixture_path(self, filename: str) -> Path:
        """Получить путь к фикстуре"""
        return self.fixtures_dir / "fires" / filename

    def _check_fixture_exists(self, filename: str) -> bool:
        """Проверить существование фикстуры"""
        return self._get_fixture_path(filename).exists()
