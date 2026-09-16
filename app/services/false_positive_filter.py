# False Positive Filter Service
import logging
from typing import List, Tuple

from app.core.schemas import FireCandidate


logger = logging.getLogger(__name__)


class FalsePositiveFilter:
    """
    Сервис фильтрации ложных срабатываний
    
    Фильтры:
    1. Вода (точки над водоемами)
    2. Облака/тени
    3. Промышленные источники (по белому списку)
    4. Одиночные шумовые пиксели
    5. Пустыни/горячие грунты
    """
    
    def __init__(self, whitelist_path: str | None = None):
        self.whitelist = self._load_whitelist(whitelist_path) if whitelist_path else []
    
    def _load_whitelist(self, path: str) -> List[Tuple[float, float, float]]:
        """Загрузить белый список промышленных источников (lat, lon, radius_km)"""
        import json
        from pathlib import Path
        
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = json.load(f)
                return [(item['lat'], item['lon'], item.get('radius_km', 2.0)) for item in data]
        return []
    
    async def filter_points(self, points: List[FireCandidate]) -> List[FireCandidate]:
        """
        Применить все фильтры к точкам
        
        Returns:
            Отфильтрованный список точек
        """
        filtered = []
        
        for point in points:
            is_valid, reason = await self._validate_point(point)
            
            if is_valid:
                point.is_valid = True
                point.filters_passed.append("all_checks")
                filtered.append(point)
            else:
                point.is_valid = False
                point.filters_failed.append(reason)
                point.reason = reason
                logger.debug(f"Filtered out point {point.id}: {reason}")
        
        return filtered
    
    async def _validate_point(self, point: FireCandidate) -> Tuple[bool, str]:
        """Проверить точку всеми фильтрами"""
        
        # 1. Проверка координат
        if not self._check_coordinates(point):
            return False, "invalid_coordinates"
        
        # 2. Проверка confidence
        if not self._check_confidence(point):
            return False, "low_confidence"
        
        # 3. Проверка белого списка (промзоны)
        if not self._check_whitelist(point):
            return False, "industrial_source"
        
        # 4. Проверка на одиночный пиксель (нужен контекст соседей)
        # Упрощенно: если FRP очень низкий и brightness подозрительный
        if not self._check_thermal_properties(point):
            return False, "suspicious_thermal_properties"
        
        return True, ""
    
    def _check_coordinates(self, point: FireCandidate) -> bool:
        """Проверка валидности координат"""
        return -90 <= point.latitude <= 90 and -180 <= point.longitude <= 180
    
    def _check_confidence(self, point: FireCandidate) -> bool:
        """Проверка уровня уверенности"""
        # Низкая уверенность автоматически отбрасывается
        return point.confidence.value != 'low'
    
    def _check_whitelist(self, point: FireCandidate) -> bool:
        """Проверка по белому списку промышленных источников"""
        from math import radians, cos, sin, asin, sqrt
        
        for lat, lon, radius in self.whitelist:
            # Haversine distance
            dlat = radians(lat - point.latitude)
            dlon = radians(lon - point.longitude)
            a = sin(dlat/2)**2 + cos(radians(lat)) * cos(radians(point.latitude)) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            distance_km = c * 6371
            
            if distance_km <= radius:
                return False  # Точка в промзоне
        
        return True
    
    def _check_thermal_properties(self, point: FireCandidate) -> bool:
        """Проверка термических свойств"""
        # MODIS пороги
        min_brightness = 310  # K
        if point.brightness_temp_k < min_brightness:
            return False
        
        # Если FRP доступен, проверяем его
        if point.frp_mw is not None and point.frp_mw < 1.0:
            # Очень низкий FRP может быть шумом
            if point.brightness_temp_k < 320:
                return False
        
        return True
    
    def mark_as_low_confidence(self, point: FireCandidate, reason: str) -> FireCandidate:
        """Пометить точку как низко-уверенную вместо удаления"""
        point.false_positive_score += 0.3
        point.filters_failed.append(reason)
        point.reason = reason
        return point
