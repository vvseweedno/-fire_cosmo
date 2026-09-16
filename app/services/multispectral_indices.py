"""
Мультиспектральные индексы для анализа пожаров

Индексы:
- NBR = (B08 - B12) / (B08 + B12)
- SAVI = ((B08 - B04) / (B08 + B04 + L)) * (1 + L)
- NDWI = (B03 - B08) / (B03 + B08)
- dNBR = NBR_pre - NBR_post
- RdNBR = dNBR / sqrt(|NBR_pre| + 1e-8)
- Fire Risk Index = vegetation_dryness*30 + soil_dryness*25 + fuel_factor*25 + history*20
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)


class MultispectralIndices:
    """
    Калькулятор мультиспектральных индексов для ДЗЗ
    
    Поддерживаемые индексы:
    - NBR (Normalized Burn Ratio)
    - SAVI (Soil Adjusted Vegetation Index)
    - NDWI (Normalized Difference Water Index)
    - dNBR (Difference NBR)
    - RdNBR (Relative dNBR)
    - Fire Risk Index
    """
    
    @staticmethod
    def calculate_nbr(b08: np.ndarray, b12: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Рассчитать индекс NBR (Normalized Burn Ratio)
        
        NBR = (B08 - B12) / (B08 + B12)
        
        где:
        - B08: NIR канал (Near Infrared)
        - B12: SWIR канал (Short Wave Infrared)
        
        Args:
            b08: Канал NIR
            b12: Канал SWIR
            eps: epsilon для избежания деления на ноль
            
        Returns:
            NBR raster в диапазоне [-1, 1]
        """
        numerator = b08.astype(np.float32) - b12.astype(np.float32)
        denominator = b08.astype(np.float32) + b12.astype(np.float32) + eps
        
        nbr = numerator / denominator
        
        # Clip to [-1, 1]
        nbr = np.clip(nbr, -1.0, 1.0)
        
        logger.debug(f"NBR calculated: min={nbr.min():.3f}, max={nbr.max():.3f}")
        
        return nbr
    
    @staticmethod
    def calculate_savi(
        b04: np.ndarray, 
        b08: np.ndarray, 
        L: float = 0.5,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Рассчитать индекс SAVI (Soil Adjusted Vegetation Index)
        
        SAVI = ((B08 - B04) / (B08 + B04 + L)) * (1 + L)
        
        где:
        - B04: Red канал
        - B08: NIR канал
        - L: коэффициент коррекции почвы (обычно 0.5)
        
        Args:
            b04: Канал Red
            b08: Канал NIR
            L: Soil brightness correction factor
            eps: epsilon для избежания деления на ноль
            
        Returns:
            SAVI raster
        """
        numerator = b08.astype(np.float32) - b04.astype(np.float32)
        denominator = b08.astype(np.float32) + b04.astype(np.float32) + L + eps
        
        savi = (numerator / denominator) * (1.0 + L)
        
        logger.debug(f"SAVI calculated: min={savi.min():.3f}, max={savi.max():.3f}")
        
        return savi
    
    @staticmethod
    def calculate_ndwi(b03: np.ndarray, b08: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Рассчитать индекс NDWI (Normalized Difference Water Index)
        
        NDWI = (B03 - B08) / (B03 + B08)
        
        где:
        - B03: Green канал
        - B08: NIR канал
        
        Args:
            b03: Канал Green
            b08: Канал NIR
            eps: epsilon для избежания деления на ноль
            
        Returns:
            NDWI raster в диапазоне [-1, 1]
        """
        numerator = b03.astype(np.float32) - b08.astype(np.float32)
        denominator = b03.astype(np.float32) + b08.astype(np.float32) + eps
        
        ndwi = numerator / denominator
        
        # Clip to [-1, 1]
        ndwi = np.clip(ndwi, -1.0, 1.0)
        
        logger.debug(f"NDWI calculated: min={ndwi.min():.3f}, max={ndwi.max():.3f}")
        
        return ndwi
    
    @staticmethod
    def calculate_dnbr(nbr_pre: np.ndarray, nbr_post: np.ndarray) -> np.ndarray:
        """
        Рассчитать dNBR (Difference NBR)
        
        dNBR = NBR_pre - NBR_post
        
        Положительные значения указывают на выгорание
        
        Args:
            nbr_pre: NBR до пожара
            nbr_post: NBR после пожара
            
        Returns:
            dNBR raster
        """
        if nbr_pre is None:
            # Если нет pre-снимка, используем упрощенный подход
            logger.warning("No pre-fire NBR available, using inverted post-NBR")
            return -nbr_post
        
        dnbr = nbr_pre.astype(np.float32) - nbr_post.astype(np.float32)
        
        logger.debug(f"dNBR calculated: min={dnbr.min():.3f}, max={dnbr.max():.3f}")
        
        return dnbr
    
    @staticmethod
    def calculate_rdnbr(dnbr: np.ndarray, nbr_pre: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Рассчитать RdNBR (Relative dNBR)
        
        RdNBR = dNBR / sqrt(|NBR_pre| + eps)
        
        Нормализует dNBR относительно предпожарного состояния растительности
        
        Args:
            dnbr: dNBR raster
            nbr_pre: NBR до пожара
            eps: epsilon для избежания деления на ноль
            
        Returns:
            RdNBR raster
        """
        if nbr_pre is None:
            logger.warning("No pre-fire NBR available, returning dNBR unchanged")
            return dnbr
        
        denominator = np.sqrt(np.abs(nbr_pre.astype(np.float32)) + eps)
        
        rdnbr = dnbr.astype(np.float32) / denominator
        
        logger.debug(f"RdNBR calculated: min={rdnbr.min():.3f}, max={rdnbr.max():.3f}")
        
        return rdnbr
    
    @staticmethod
    def calculate_fire_risk_index(
        ndvi: np.ndarray,
        swir: np.ndarray,
        land_cover: np.ndarray,
        burned_history: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Рассчитать индекс риска пожара (Fire Risk Index)
        
        risk = vegetation_dryness*30 + soil_dryness*25 + fuel_factor*25 + history*20
        
        Args:
            ndvi: NDVI raster (vegetation health)
            swir: SWIR канал (soil moisture proxy)
            land_cover: Карта типов земной поверхности
            burned_history: История пожаров (опционально)
            weights: Веса компонентов {vegetation, soil, fuel, history}
            
        Returns:
            Fire Risk Index raster [0, 100]
        """
        if weights is None:
            weights = {
                'vegetation': 30.0,
                'soil': 25.0,
                'fuel': 25.0,
                'history': 20.0,
            }
        
        # Vegetation dryness: низкий NDVI = сухая растительность
        # Нормализуем NDVI от [-1, 1] к [0, 1], затем инвертируем
        ndvi_normalized = (ndvi.astype(np.float32) + 1.0) / 2.0  # [0, 1]
        vegetation_dryness = 1.0 - np.clip(ndvi_normalized, 0.0, 1.0)
        
        # Soil dryness: высокий SWIR = сухая почва
        # Нормализуем SWIR (предполагаем диапазон [0, 10000] для Sentinel-2)
        soil_dryness = np.clip(swir.astype(np.float32) / 10000.0, 0.0, 1.0)
        
        # Fuel factor: зависит от land_cover
        # Типы: 1=forest, 2=grassland, 3=agriculture, 4=urban, 5=water
        fuel_factors = {
            1: 1.0,   # forest - высокий риск
            2: 0.9,   # grassland
            3: 0.7,   # agriculture
            4: 0.3,   # urban - низкий риск
            5: 0.0,   # water - нет риска
        }
        
        fuel_factor = np.zeros_like(land_cover, dtype=np.float32)
        for code, factor in fuel_factors.items():
            fuel_factor[land_cover == code] = factor
        
        # History factor: предыдущие пожары увеличивают риск
        if burned_history is not None:
            history_factor = np.clip(burned_history.astype(np.float32), 0.0, 1.0)
        else:
            history_factor = np.zeros_like(land_cover, dtype=np.float32)
        
        # Взвешенная сумма
        total_weight = sum(weights.values())
        
        risk_index = (
            weights['vegetation'] * vegetation_dryness +
            weights['soil'] * soil_dryness +
            weights['fuel'] * fuel_factor +
            weights['history'] * history_factor
        ) / total_weight * 100.0
        
        # Clip to [0, 100]
        risk_index = np.clip(risk_index, 0.0, 100.0)
        
        logger.info(
            f"Fire Risk Index calculated: min={risk_index.min():.2f}, "
            f"max={risk_index.max():.2f}, mean={risk_index.mean():.2f}"
        )
        
        return risk_index
    
    @staticmethod
    def calculate_ndvi(b04: np.ndarray, b08: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Рассчитать индекс NDVI (Normalized Difference Vegetation Index)
        
        NDVI = (B08 - B04) / (B08 + B04)
        
        Args:
            b04: Канал Red
            b08: Канал NIR
            eps: epsilon для избежания деления на ноль
            
        Returns:
            NDVI raster в диапазоне [-1, 1]
        """
        numerator = b08.astype(np.float32) - b04.astype(np.float32)
        denominator = b08.astype(np.float32) + b04.astype(np.float32) + eps
        
        ndvi = numerator / denominator
        
        # Clip to [-1, 1]
        ndvi = np.clip(ndvi, -1.0, 1.0)
        
        logger.debug(f"NDVI calculated: min={ndvi.min():.3f}, max={ndvi.max():.3f}")
        
        return ndvi
    
    @staticmethod
    def calculate_baismm(b04: np.ndarray, b08: np.ndarray, b11: np.ndarray, b12: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Рассчитать индекс BAISMM (Burned Area Index for Sentinel-2 Modified)
        
        BAISMM = (1 - ((NIR × RE2 × RE3) / SWIR1 × SWIR2)) × (1 - NDVI)
        
        Специализированный индекс для детекции гарей по Sentinel-2
        
        Args:
            b04: Red канал
            b08: NIR канал
            b11: SWIR1 канал
            b12: SWIR2 канал
            eps: epsilon
            
        Returns:
            BAISMM raster
        """
        # Упрощенная версия без RE2/RE3 каналов
        ndvi = MultispectralIndices.calculate_ndvi(b04, b08, eps)
        
        nir_swir_ratio = (b08.astype(np.float32) * eps) / (b11.astype(np.float32) * b12.astype(np.float32) + eps)
        
        baismm = (1.0 - nir_swir_ratio) * (1.0 - ndvi)
        
        logger.debug(f"BAISMM calculated: min={baismm.min():.3f}, max={baismm.max():.3f}")
        
        return baismm
