# Fire Clustering Service (Event Tracker)
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from shapely.geometry import Point
from shapely.strtree import STRtree as STRTree

from app.core.schemas import FireCandidate, FireEvent, FireStatus


logger = logging.getLogger(__name__)


class FireClusteringService:
    """Group spatially/temporally close thermal detections into persistent events."""

    def __init__(
        self,
        spatial_radius_km: float = 3.0,
        temporal_window_hours: float = 24.0,
        min_points_per_event: int = 1,
    ):
        self.spatial_radius_km = spatial_radius_km
        self.temporal_window_hours = temporal_window_hours
        self.min_points_per_event = min_points_per_event
        self.events: Dict[str, FireEvent] = {}
        self._point_to_event: Dict[str, str] = {}
        self._event_tree: Optional[STRTree] = None
        self._event_index: Dict[int, str] = {}

    async def cluster_points(self, points: List[FireCandidate]) -> List[FireEvent]:
        """Cluster one analysis batch and return only events touched by that batch."""
        if not points:
            return []

        touched: set[str] = set()
        for point in sorted(points, key=lambda p: (p.datetime, p.id)):
            event = await self._process_point(point)
            touched.add(event.id)

        valid_events = [
            self.events[event_id]
            for event_id in sorted(touched)
            if len(self.events[event_id].fire_points) >= self.min_points_per_event
        ]
        logger.info("Clustered %d points into %d touched events", len(points), len(valid_events))
        return valid_events

    async def _process_point(self, point: FireCandidate) -> FireEvent:
        existing_event_id = self._point_to_event.get(point.id)
        if existing_event_id and existing_event_id in self.events:
            return self.events[existing_event_id]

        nearby_events = await self._find_nearby_events(
            point.latitude, point.longitude, point.datetime
        )
        if nearby_events:
            target_event = nearby_events[0]
            await self._update_event_with_point(target_event, point)
            return target_event
        return await self._create_new_event(point)

    async def _find_nearby_events(
        self, lat: float, lon: float, observed_at: datetime
    ) -> List[FireEvent]:
        if not self.events:
            return []
        if self._event_tree is None:
            self._rebuild_event_tree()

        from math import cos, radians

        point = Point(lon, lat)
        lat_radius = self.spatial_radius_km / 111.32
        lon_radius = self.spatial_radius_km / (111.32 * max(cos(radians(lat)), 0.2))
        bbox = point.buffer(max(lat_radius, lon_radius))
        candidates = self._event_tree.query(bbox)

        nearby = []
        for raw_idx in candidates:
            idx = int(raw_idx)
            event_id = self._event_index[idx]
            event = self.events[event_id]
            if event.status == FireStatus.FAILED:
                continue
            if abs((observed_at - event.last_seen).total_seconds()) > self.temporal_window_hours * 3600:
                continue
            distance = self._haversine_distance(
                lat, lon, event.centroid_lat, event.centroid_lon
            )
            if distance <= self.spatial_radius_km:
                nearby.append(event)

        return sorted(
            nearby,
            key=lambda e: self._haversine_distance(
                lat, lon, e.centroid_lat, e.centroid_lon
            ),
        )

    def _rebuild_event_tree(self):
        if not self.events:
            self._event_tree = None
            self._event_index.clear()
            return
        event_ids = list(self.events.keys())
        geometries = [
            Point(self.events[eid].centroid_lon, self.events[eid].centroid_lat)
            for eid in event_ids
        ]
        self._event_tree = STRTree(geometries)
        self._event_index = {i: eid for i, eid in enumerate(event_ids)}

    async def _update_event_with_point(self, event: FireEvent, point: FireCandidate):
        if point.id in self._point_to_event:
            return
        event.fire_points.append(point)
        event.first_seen = min(event.first_seen, point.datetime)
        event.last_seen = max(event.last_seen, point.datetime)
        event.point_count = len(event.fire_points)
        if point.frp_mw is not None and (
            event.max_frp is None or point.frp_mw > event.max_frp
        ):
            event.max_frp = point.frp_mw
        await self._recalculate_centroid(event)
        if point.sensor not in event.sensors:
            event.sensors.append(point.sensor)
        # A newly observed point makes the event active again; burn mapping can
        # then be recomputed for the updated event.
        event.status = FireStatus.ACTIVE
        self._point_to_event[point.id] = event.id
        self._event_tree = None
        logger.debug("Updated event %s with point %s", event.id, point.id)

    @staticmethod
    def _event_id_from_first_point(point: FireCandidate) -> str:
        digest = hashlib.sha1(point.id.encode("utf-8")).hexdigest()[:12]
        return f"fire_{digest}"

    async def _create_new_event(self, point: FireCandidate) -> FireEvent:
        event_id = self._event_id_from_first_point(point)
        # Extremely unlikely SHA-prefix collision: extend deterministically.
        if event_id in self.events and point.id not in self._point_to_event:
            digest = hashlib.sha1((point.id + "|event").encode("utf-8")).hexdigest()[:16]
            event_id = f"fire_{digest}"

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
            fire_points=[point],
        )
        self.events[event_id] = event
        self._point_to_event[point.id] = event_id
        self._event_tree = None
        logger.info("Created event %s at (%s, %s)", event_id, point.latitude, point.longitude)
        return event

    async def _recalculate_centroid(self, event: FireEvent):
        if not event.fire_points:
            return
        event.centroid_lat = sum(p.latitude for p in event.fire_points) / len(event.fire_points)
        event.centroid_lon = sum(p.longitude for p in event.fire_points) / len(event.fire_points)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import asin, cos, radians, sin, sqrt

        radius_km = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return radius_km * (2 * asin(sqrt(a)))

    async def mark_decayed_events(self, hours_threshold: float = 12.0):
        now = datetime.utcnow()
        threshold = now - timedelta(hours=hours_threshold)
        decayed_count = 0
        for event in self.events.values():
            if event.status == FireStatus.ACTIVE and event.last_seen < threshold:
                event.status = FireStatus.HISTORICAL
                decayed_count += 1
        if decayed_count:
            logger.info("Marked %d events as historical", decayed_count)
        return decayed_count

    def get_event_by_id(self, event_id: str) -> Optional[FireEvent]:
        return self.events.get(event_id)

    def get_all_events(self) -> List[FireEvent]:
        return list(self.events.values())

    async def clear_events(self):
        self.events.clear()
        self._point_to_event.clear()
        self._event_tree = None
        self._event_index.clear()
        logger.info("Cleared all events")
