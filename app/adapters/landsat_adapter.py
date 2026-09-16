# Landsat Adapter (stub with fixtures)
from typing import List
from pathlib import Path
import json

from app.adapters.base import BaseFireAdapter
from app.core.schemas import FireCandidate


class LandsatAdapter(BaseFireAdapter):
    """Адаптер для Landsat (заглушка с фикстурами)"""

    def get_sensor_type(self) -> str:
        return "LANDSAT"

    async def fetch_fire_points(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> List[FireCandidate]:
        """Загрузка из фикстур"""
        return await self.fetch_from_cache_or_fixture(bbox, start_date, end_date)

    async def fetch_from_cache_or_fixture(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str
    ) -> List[FireCandidate]:
        """Загрузка из фикстур"""
        fixture_path = Path(self.fixtures_dir) / "landsat_fixture.json"
        if fixture_path.exists():
            with open(fixture_path) as f:
                data = json.load(f)
                return [FireCandidate(**p) for p in data.get('points', [])]
        return []
