# Sentinel-2 Adapter for STAC API
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import httpx

from app.core.schemas import Sentinel2Scene


logger = logging.getLogger(__name__)


class Sentinel2Adapter:
    """Адаптер поиска Sentinel-2 L2A через STAC API с raster provenance."""

    ASSET_ALIASES = {
        "B08": ("B08", "nir", "nir08"),
        "B12": ("B12", "swir22", "swir2"),
        "SCL": ("SCL", "scl"),
    }

    def __init__(self, stac_api_url: str, token: str | None = None):
        self.stac_api_url = stac_api_url.rstrip("/")
        self.token = token
        self.search_endpoint = f"{self.stac_api_url}/search"

    async def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        limit: int = 10,
    ) -> List[Sentinel2Scene]:
        """Поиск Sentinel-2 L2A; сохраняет ссылки B08/B12/SCL из STAC item."""
        query = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "limit": limit,
            # STAC Query extension is supported by Earth Search and many compatible APIs.
            "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        }

        headers = {"Accept": "application/geo+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.post(self.search_endpoint, json=query, headers=headers)
                response.raise_for_status()
                return self._parse_stac_response(response.json())
        except httpx.HTTPError as exc:
            logger.error("STAC API error: %s", exc)
            return []
        except Exception as exc:
            logger.error("Unexpected error searching Sentinel-2: %s", exc)
            return []

    @classmethod
    def _select_assets(cls, raw_assets: Dict[str, Any]) -> Dict[str, str]:
        selected: Dict[str, str] = {}
        for canonical, aliases in cls.ASSET_ALIASES.items():
            for alias in aliases:
                asset = raw_assets.get(alias)
                if isinstance(asset, dict) and asset.get("href"):
                    selected[canonical] = asset["href"]
                    break
        return selected

    def _parse_stac_response(self, data: Dict[str, Any]) -> List[Sentinel2Scene]:
        scenes: List[Sentinel2Scene] = []
        for feature in data.get("features", []):
            try:
                props = feature.get("properties", {})
                scene_id = feature.get("id", "")
                dt_str = props.get("datetime") or props.get("start_datetime")
                if not dt_str:
                    raise ValueError("missing datetime")
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                assets = self._select_assets(feature.get("assets", {}))
                if "B08" not in assets or "B12" not in assets:
                    logger.warning("Skipping %s: no B08/B12-compatible STAC assets", scene_id)
                    continue

                scenes.append(
                    Sentinel2Scene(
                        scene_id=scene_id,
                        datetime=dt,
                        cloud_cover=float(props.get("eo:cloud_cover", 100.0)),
                        tile_id=props.get("grid:code") or props.get("s2:mgrs_tile"),
                        collection=feature.get("collection"),
                        stac_item_url=(feature.get("links") or [{}])[0].get("href"),
                        bbox=feature.get("bbox"),
                        assets=assets,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping invalid STAC feature: %s", exc)
        return scenes

    async def find_best_pair(
        self,
        bbox: List[float],
        reference_date: datetime,
        days_before: int = 90,
        days_after: int = 30,
        max_cloud_cover: float = 20.0,
    ) -> tuple[Optional[Sentinel2Scene], Optional[Sentinel2Scene]]:
        """Найти пару до/после; предпочесть минимум облачности, затем близость к пожару."""
        pre_start = reference_date - timedelta(days=days_before)
        post_end = reference_date + timedelta(days=days_after)

        pre_scenes = await self.search_scenes(
            bbox, pre_start.isoformat(), reference_date.isoformat(), max_cloud_cover
        )
        post_scenes = await self.search_scenes(
            bbox, reference_date.isoformat(), post_end.isoformat(), max_cloud_cover
        )

        pre_scene = min(
            pre_scenes,
            key=lambda s: (s.cloud_cover, abs((reference_date - s.datetime).total_seconds())),
        ) if pre_scenes else None
        post_scene = min(
            post_scenes,
            key=lambda s: (s.cloud_cover, abs((s.datetime - reference_date).total_seconds())),
        ) if post_scenes else None
        return pre_scene, post_scene

    async def get_from_fixture(self, fixture_path: str) -> tuple[Sentinel2Scene, Sentinel2Scene]:
        import json
        from pathlib import Path

        path = Path(fixture_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return Sentinel2Scene(**data.get("pre", {})), Sentinel2Scene(**data.get("post", {}))
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
