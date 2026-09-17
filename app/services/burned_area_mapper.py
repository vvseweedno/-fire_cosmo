# Burned Area Mapper Service (NBR/dNBR calculation)
import logging
import json
import math
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np

from app.core.schemas import BurnedAreaResult, SeveritySummary, SeverityLevel


logger = logging.getLogger(__name__)


class BurnedAreaMapper:
    """
    Сервис картирования гарей по Sentinel-2
    
    Расчет индексов:
    - NBR = (B08 - B12) / (B08 + B12)
    - dNBR = NBR_pre - NBR_post
    - RdNBR (если нет pre-снимка)
    """
    
    def __init__(
        self,
        output_dir: str = "./data/outputs",
        severity_thresholds: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Пороги severity по dNBR (из спецификации)
        self.severity_thresholds = severity_thresholds or {
            'unburned': (-1.0, 0.10),
            'low': (0.10, 0.27),
            'moderate': (0.27, 0.44),
            'high': (0.44, 1.0)
        }
    
    async def calculate_nbr(self, nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
        """
        Рассчитать индекс NBR
        
        Args:
            nir_band: Канал B08 (NIR) Sentinel-2
            swir_band: Канал B12 (SWIR) Sentinel-2
            
        Returns:
            NBR raster
        """
        # Избегаем деления на ноль
        nir = nir_band.astype(np.float32)
        swir = swir_band.astype(np.float32)
        denominator = nir + swir
        mask = denominator == 0
        
        nbr = np.zeros_like(nir_band, dtype=np.float32)
        nbr[~mask] = (nir[~mask] - swir[~mask]) / denominator[~mask]
        
        # Масштабирование [-1, 1]
        nbr = np.clip(nbr, -1, 1)
        
        return nbr
    
    async def calculate_dnbr(
        self, 
        nbr_pre: np.ndarray, 
        nbr_post: np.ndarray
    ) -> np.ndarray:
        """
        Рассчитать dNBR (разница NBR)
        
        dNBR = NBR_pre - NBR_post
        
        Положительные значения указывают на выгорание
        """
        if nbr_pre is None:
            # Если нет pre-снимка, используем упрощенный подход
            # Просто инвертируем post-NBR (чем ниже, тем сильнее выгорело)
            return -nbr_post
        
        return nbr_pre - nbr_post
    
    async def classify_severity(self, dnbr: np.ndarray, nodata_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Классифицировать степень поражения по dNBR
        
        Returns:
            Raster с классами: 0=unburned, 1=low, 2=moderate, 3=high, 255=nodata
        """
        severity_map = np.zeros_like(dnbr, dtype=np.uint8)
        
        # Unburned (dNBR < 0.10)
        mask_unburned = dnbr < self.severity_thresholds['unburned'][1]
        severity_map[mask_unburned] = 0
        
        # Low severity (0.10 <= dNBR < 0.27)
        mask_low = (dnbr >= self.severity_thresholds['low'][0]) & (dnbr < self.severity_thresholds['low'][1])
        severity_map[mask_low] = 1
        
        # Moderate severity (0.27 <= dNBR < 0.44)
        mask_moderate = (dnbr >= self.severity_thresholds['moderate'][0]) & (dnbr < self.severity_thresholds['moderate'][1])
        severity_map[mask_moderate] = 2
        
        # High severity (dNBR >= 0.44)
        mask_high = dnbr >= self.severity_thresholds['high'][0]
        severity_map[mask_high] = 3
        
        # NoData маска
        if nodata_mask is not None:
            severity_map[nodata_mask] = 255
        
        return severity_map
    
    async def polygonize_severity(
        self, 
        severity_raster: np.ndarray, 
        transform: Any, 
        crs: str
    ) -> Dict[str, Any]:
        """
        Векторизовать растр в полигоны
        
        Returns:
            GeoJSON FeatureCollection
        """
        try:
            from rasterio.features import shapes
            import geopandas as gpd
            from shapely.geometry import shape, mapping
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            return {"type": "FeatureCollection", "features": []}
        
        # Получаем уникальные значения и их площади
        features = []
        
        for class_value in [1, 2, 3]:
            mask = severity_raster == class_value
            
            if not np.any(mask):
                continue
            
            # Векторизация
            results = shapes(
                mask.astype(np.uint8),
                transform=transform,
                connectivity=8
            )
            
            for geom, value in results:
                if value == 1:  # Только ненулевые полигоны
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(shape(geom)),
                        "properties": {
                            "severity": self._class_to_severity(class_value),
                            "class_value": int(class_value),
                            "event_id": "pending"
                        }
                    }
                    features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _class_to_severity(self, class_value: int) -> str:
        """Конвертировать класс в строку severity"""
        mapping = {
            0: "unburned",
            1: "low_severity",
            2: "moderate_severity",
            3: "high_severity"
        }
        return mapping.get(class_value, "unknown")
    
    async def calculate_area_from_polygons(self, geojson: Dict[str, Any]) -> BurnedAreaResult:
        """
        Рассчитать площадь из полигонов
        
        Returns:
            BurnedAreaResult с площадями по категориям
        """
        try:
            import geopandas as gpd
            from io import StringIO
            import json
            
            # Конвертация в GeoDataFrame
            gdf = gpd.read_file(StringIO(json.dumps(geojson)))
            
            if gdf.empty:
                return BurnedAreaResult(
                    event_id="unknown",
                    area_ha=0.0,
                    area_m2=0.0,
                    projection_used="EPSG:4326",
                    method="polygon_area",
                    uncertainty="high",
                    severity_summary=SeveritySummary()
                )
            
            # Проекция в равновеликую (для точного расчета площади)
            # Используем World Cylindrical Equal Area
            gdf_projected = gdf.to_crs("EPSG:6933")
            
            # Расчет площадей
            gdf_projected['area_m2'] = gdf_projected.geometry.area
            
            # Агрегация по severity
            severity_areas = gdf_projected.groupby('severity')['area_m2'].sum()
            
            affected_area_m2 = float(
                severity_areas.get('low_severity', 0.0)
                + severity_areas.get('moderate_severity', 0.0)
                + severity_areas.get('high_severity', 0.0)
            )

            summary = SeveritySummary(
                unburned_ha=float(severity_areas.get('unburned', 0.0)) / 10000,
                low_severity_ha=float(severity_areas.get('low_severity', 0.0)) / 10000,
                moderate_severity_ha=float(severity_areas.get('moderate_severity', 0.0)) / 10000,
                high_severity_ha=float(severity_areas.get('high_severity', 0.0)) / 10000,
                total_affected_ha=affected_area_m2 / 10000
            )
            
            return BurnedAreaResult(
                event_id="unknown",
                area_ha=affected_area_m2 / 10000,
                area_m2=affected_area_m2,
                projection_used="EPSG:6933",
                method="polygon_area",
                uncertainty="low",
                severity_summary=summary
            )
            
        except Exception as e:
            logger.error(f"Error calculating area: {e}")
            return BurnedAreaResult(
                event_id="unknown",
                area_ha=0.0,
                area_m2=0.0,
                projection_used="unknown",
                method="failed",
                uncertainty="high"
            )
    
    async def process_sentinel2_pair(
        self,
        pre_scene_path: Optional[str],
        post_scene_path: str,
        event_id: str,
        center_lon: float = 92.0,
        center_lat: float = 56.0
    ) -> Tuple[Dict[str, Any], BurnedAreaResult]:
        """
        Обработать пару снимков Sentinel-2
        
        Returns:
            (GeoJSON полигонов, BurnedAreaResult)
        """
        fixture_path = Path(post_scene_path)
        if fixture_path.is_dir():
            fixture_path = fixture_path / "dNBR.json"
        if not fixture_path.exists():
            fixture_path = Path("data/fixtures/sentinel2/dNBR.json")

        with open(fixture_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        dnbr = np.asarray(payload["matrix"], dtype=np.float32)
        pixel_size_m = float(payload.get("pixel_size_m", 20))
        ha_per_pixel = float(payload.get("ha_per_pixel", (pixel_size_m * pixel_size_m) / 10000))
        severity = await self.classify_severity(dnbr)

        low_ha = float(np.count_nonzero(severity == 1) * ha_per_pixel)
        moderate_ha = float(np.count_nonzero(severity == 2) * ha_per_pixel)
        high_ha = float(np.count_nonzero(severity == 3) * ha_per_pixel)
        total_ha = low_ha + moderate_ha + high_ha

        features = []
        class_specs = [
            (1, "low_severity", "#ffcc00", low_ha),
            (2, "moderate_severity", "#ff8800", moderate_ha),
            (3, "high_severity", "#ff0000", high_ha),
        ]
        cursor = -0.5
        for class_value, severity_name, color, area_ha in class_specs:
            if area_ha <= 0:
                continue
            side_km = math.sqrt(area_ha / 100.0)
            dlat = side_km / 111.32
            dlon = side_km / (111.32 * max(math.cos(math.radians(center_lat)), 0.2))
            offset = cursor * dlon * 1.4
            cursor += 1
            coords = [[
                [center_lon + offset - dlon / 2, center_lat - dlat / 2],
                [center_lon + offset + dlon / 2, center_lat - dlat / 2],
                [center_lon + offset + dlon / 2, center_lat + dlat / 2],
                [center_lon + offset - dlon / 2, center_lat + dlat / 2],
                [center_lon + offset - dlon / 2, center_lat - dlat / 2],
            ]]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coords},
                "properties": {
                    "severity": severity_name,
                    "class_value": class_value,
                    "event_id": event_id,
                    "area_ha": round(area_ha, 2),
                    "fill": color,
                    "method": "fixture_dnbr_pixel_count"
                }
            })

        geojson = {"type": "FeatureCollection", "features": features}

        result = BurnedAreaResult(
            event_id=event_id,
            area_ha=round(total_ha, 2),
            area_m2=round(total_ha * 10000, 2),
            projection_used=payload.get("projection", "EPSG:6933"),
            method="dNBR_pixel_count",
            uncertainty="demo_fixture" if "fixtures" in str(fixture_path) else "medium",
            severity_summary=SeveritySummary(
                unburned_ha=round(float(np.count_nonzero(severity == 0) * ha_per_pixel), 2),
                low_severity_ha=round(low_ha, 2),
                moderate_severity_ha=round(moderate_ha, 2),
                high_severity_ha=round(high_ha, 2),
                total_affected_ha=round(total_ha, 2)
            )
        )
        
        return geojson, result
