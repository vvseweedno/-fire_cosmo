# Burned Area Mapper Service (NBR/dNBR calculation)
import logging
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
        denominator = nir_band + swir_band
        mask = denominator == 0
        
        nbr = np.zeros_like(nir_band, dtype=np.float32)
        nbr[~mask] = (nir_band[~mask] - swir_band[~mask]) / denominator[~mask]
        
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
        
        for i, class_value in enumerate([0, 1, 2, 3]):
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
            
            summary = SeveritySummary(
                unburned_ha=float(severity_areas.get('unburned', 0.0)) / 10000,
                low_severity_ha=float(severity_areas.get('low_severity', 0.0)) / 10000,
                moderate_severity_ha=float(severity_areas.get('moderate_severity', 0.0)) / 10000,
                high_severity_ha=float(severity_areas.get('high_severity', 0.0)) / 10000,
                total_affected_ha=float(gdf_projected['area_m2'].sum()) / 10000
            )
            
            total_area_m2 = gdf_projected['area_m2'].sum()
            
            return BurnedAreaResult(
                event_id="unknown",
                area_ha=float(total_area_m2) / 10000,
                area_m2=float(total_area_m2),
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
        event_id: str
    ) -> Tuple[Dict[str, Any], BurnedAreaResult]:
        """
        Обработать пару снимков Sentinel-2
        
        Returns:
            (GeoJSON полигонов, BurnedAreaResult)
        """
        # Заглушка для реальной реализации
        # В реальности здесь будет загрузка растров через rasterio
        
        logger.warning("Using fixture/mock data for burned area calculation")
        
        # Создаем фиктивный результат
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.5, 37.8],
                        [-122.4, 37.8],
                        [-122.4, 37.9],
                        [-122.5, 37.9],
                        [-122.5, 37.8]
                    ]]
                },
                "properties": {
                    "severity": "moderate_severity",
                    "event_id": event_id
                }
            }]
        }
        
        mock_result = BurnedAreaResult(
            event_id=event_id,
            area_ha=125.5,
            area_m2=1255000.0,
            projection_used="EPSG:6933",
            method="polygon_area",
            uncertainty="medium",
            severity_summary=SeveritySummary(
                unburned_ha=0.0,
                low_severity_ha=25.0,
                moderate_severity_ha=75.5,
                high_severity_ha=25.0,
                total_affected_ha=125.5
            )
        )
        
        return mock_geojson, mock_result
