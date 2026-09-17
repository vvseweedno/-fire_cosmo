# Fire Clustering Service (Event Tracker)
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from uuid import uuid4

from shapely.geometry import Point
from shapely.strtree import STRtree as STRTree
import numpy as np

from app.core.schemas import FireCandidate, FireEvent, FireStatus


logger = logging.getLogger(__name__)


class FireClusteringService:
    """
    Сервис кластеризации точек пожаров в события
    
    Объединяет близкие точки во времени и пространстве
    """
    
    def __init__(
        self,
        spatial_radius_km: float = 3.0,
        temporal_window_hours: float = 24.0,
        min_points_per_event: int = 1
    ):
        self.spatial_radius_km = spatial_radius_km
        self.temporal_window_hours = temporal_window_hours
        self.min_points_per_event = min_points_per_event
        
        # Хранилище активных событий
        self.events: Dict[str, FireEvent] = {}
        self._event_tree: Optional[STRTree] = None
        self._event_index: Dict[int, str] = {}  # index -> event_id
    
    async def cluster_points(self, points: List[FireCandidate]) -> List[FireEvent]:
        """
        Скластеризовать точки в пожарные события
        
        Args:
            points: Список FireCandidate
            
        Returns:
            Список FireEvent
        """
        if not points:
            return []
        
        # Сортировка по времени
        sorted_points = sorted(points, key=lambda p: p.datetime)
        
        for point in sorted_points:
            await self._process_point(point)
        
        # Фильтрация событий с малым количеством точек
        valid_events = [
            e for e in self.events.values()
            if len(e.fire_points) >= self.min_points_per_event
        ]
        
        logger.info(f"Clustered {len(points)} points into {len(valid_events)} events")
        return valid_events
    
    async def _process_point(self, point: FireCandidate):
        """Обработать одну точку"""
        nearby_events = await self._find_nearby_events(point.latitude, point.longitude, point.datetime)
        
        if nearby_events:
            # Привязать к ближайшему событию
            target_event = nearby_events[0]
            await self._update_event_with_point(target_event, point)
        else:
            # Создать новое событие
            await self._create_new_event(point)
    
    async def _find_nearby_events(self, lat: float, lon: float, observed_at: datetime) -> List[FireEvent]:
        """Найти события в радиусе с использованием STRTree"""
        if not self.events:
            return []
        
        # Построить spatial index (один раз)
        if self._event_tree is None:
            self._rebuild_event_tree()
        
        # Найти кандидатов в bounding box
        from math import cos, radians

        point = Point(lon, lat)
        lat_radius = self.spatial_radius_km / 111.32
        lon_radius = self.spatial_radius_km / (111.32 * max(cos(radians(lat)), 0.2))
        bbox = point.buffer(max(lat_radius, lon_radius))
        candidates = self._event_tree.query(bbox)
        
        # Проверить точное расстояние
        nearby = []
        for idx in candidates:
            event_id = self._event_index[idx]
            event = self.events[event_id]
            if event.status != FireStatus.ACTIVE:
                continue
            if abs((observed_at - event.last_seen).total_seconds()) > self.temporal_window_hours * 3600:
                continue
            
            distance = self._haversine_distance(
                lat, lon,
                event.centroid_lat, event.centroid_lon
            )
            if distance <= self.spatial_radius_km:
                nearby.append(event)
        
        return sorted(
            nearby,
            key=lambda e: self._haversine_distance(lat, lon, e.centroid_lat, e.centroid_lon)
        )
    
    def _rebuild_event_tree(self):
        """Перестроить пространственный индекс"""
        if not self.events:
            return
        
        geometries = [
            Point(e.centroid_lon, e.centroid_lat)
            for e in self.events.values()
        ]
        self._event_tree = STRTree(geometries)
        self._event_index = {i: eid for i, eid in enumerate(self.events.keys())}
    
    async def _update_event_with_point(self, event: FireEvent, point: FireCandidate):
        """Обновить событие новой точкой"""
        event.fire_points.append(point)
        event.last_seen = point.datetime
        event.point_count = len(event.fire_points)
        
        # Обновление максимума FRP
        if point.frp_mw and (event.max_frp is None or point.frp_mw > event.max_frp):
            event.max_frp = point.frp_mw
        
        # Пересчет центроида
        await self._recalculate_centroid(event)
        
        # Добавление сенсора
        if point.sensor not in event.sensors:
            event.sensors.append(point.sensor)
        
        # Инвалидировать spatial index
        self._event_tree = None
        
        logger.debug(f"Updated event {event.id} with point {point.id}")
    
    async def _create_new_event(self, point: FireCandidate):
        """Создать новое пожарное событие"""
        event_id = f"fire_{uuid4().hex[:8]}"
        
        event = FireEvent(
            id=event_id,
            event_id=event_id,
            first_seen=point.datetime,
            last_seen=point.datetime,
            centroid_lat=point.latitude,
            centroid_lon=point.longitude,
            sensors=[point.sensor],
            status=FireStatus.ACTIVE,
            point_count=1,
            max_frp=point.frp_mw,
            fire_points=[point]
        )
        
        self.events[event_id] = event
        self._event_tree = None
        logger.info(f"Created new event {event_id} at ({point.latitude}, {point.longitude})")
    
    async def _recalculate_centroid(self, event: FireEvent):
        """Пересчитать центроид события"""
        if not event.fire_points:
            return
        
        lats = [p.latitude for p in event.fire_points]
        lons = [p.longitude for p in event.fire_points]
        
        event.centroid_lat = sum(lats) / len(lats)
        event.centroid_lon = sum(lons) / len(lons)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Расстояние между точками в км"""
        from math import radians, cos, sin, asin, sqrt
        
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    async def mark_decayed_events(self, hours_threshold: float = 12.0):
        """Пометить затухшие события"""
        now = datetime.utcnow()
        threshold = now - timedelta(hours=hours_threshold)
        
        decayed_count = 0
        for event in self.events.values():
            if event.status == FireStatus.ACTIVE and event.last_seen < threshold:
                event.status = FireStatus.HISTORICAL
                decayed_count += 1
        
        if decayed_count > 0:
            logger.info(f"Marked {decayed_count} events as historical")
        
        return decayed_count
    
    def get_event_by_id(self, event_id: str) -> Optional[FireEvent]:
        """Получить событие по ID"""
        return self.events.get(event_id)
    
    def get_all_events(self) -> List[FireEvent]:
        """Получить все события"""
        return list(self.events.values())
    
    async def clear_events(self):
        """Очистить все события"""
        self.events.clear()
        self._event_tree = None
        self._event_index.clear()
        logger.info("Cleared all events")
