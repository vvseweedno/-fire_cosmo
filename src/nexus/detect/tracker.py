"""Fire event tracker - converts points to tracked events."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from shapely.geometry import Point, MultiPoint

from ..db.models import FireEvent, ThermalPoint
from ..settings import get_settings

logger = logging.getLogger(__name__)


class FireTracker:
    """Tracks wildfires as events over time from thermal points."""

    def __init__(self) -> None:
        """Initialize fire tracker."""
        self.settings = get_settings()
        self._active_events: dict[str, FireEvent] = {}

    async def process_point(
        self,
        point: ThermalPoint,
        session,
    ) -> FireEvent:
        """Process a thermal point and associate with an event.
        
        Args:
            point: ThermalPoint to process
            session: SQLAlchemy async session
            
        Returns:
            Associated or created FireEvent
        """
        # Find existing active events within tracking radius
        nearby_event = await self._find_nearby_event(point, session)
        
        if nearby_event:
            # Update existing event
            event = await self._update_event(nearby_event, point, session)
        else:
            # Create new event
            event = await self._create_event(point, session)
        
        return event

    async def _find_nearby_event(
        self,
        point: ThermalPoint,
        session,
    ) -> Optional[FireEvent]:
        """Find active fire event near the point."""
        from sqlalchemy import select
        
        radius_km = self.settings.tracking_radius_km
        
        # Query active events
        stmt = select(FireEvent).where(FireEvent.status == "active")
        result = await session.execute(stmt)
        events = result.scalars().all()
        
        for event in events:
            distance = self._haversine_distance(
                point.latitude, point.longitude,
                event.centroid_lat, event.centroid_lon
            )
            
            if distance <= radius_km:
                logger.debug(f"Point matches event {event.id} at {distance:.2f}km")
                return event
        
        return None

    async def _update_event(
        self,
        event: FireEvent,
        point: ThermalPoint,
        session,
    ) -> FireEvent:
        """Update existing event with new point."""
        event.last_seen = point.detected_at
        event.point_count += 1
        
        if point.frp and point.frp > event.max_frp:
            event.max_frp = point.frp
        
        # Recalculate centroid (simplified - average of all points)
        # In production, you'd store all point locations and recalculate properly
        weight = 1.0 / event.point_count
        event.centroid_lat = event.centroid_lat * (1 - weight) + point.latitude * weight
        event.centroid_lon = event.centroid_lon * (1 - weight) + point.longitude * weight
        
        # Estimate area (very simplified - based on point count and max FRP)
        event.area_estimate_ha = self._estimate_area(event)
        
        logger.info(f"Updated event {event.id}: points={event.point_count}, area={event.area_estimate_ha:.1f}ha")
        
        return event

    async def _create_event(self, point: ThermalPoint, session) -> FireEvent:
        """Create new fire event from point."""
        from sqlalchemy import select
        
        event = FireEvent(
            status="active",
            first_seen=point.detected_at,
            last_seen=point.detected_at,
            centroid_lat=point.latitude,
            centroid_lon=point.longitude,
            point_count=1,
            max_frp=point.frp or 0.0,
            area_estimate_ha=10.0,  # Initial estimate
            risk_level="info",
        )
        
        session.add(event)
        await session.flush()  # Get ID
        
        logger.info(f"Created new event {event.id} at ({point.latitude}, {point.longitude})")
        
        return event

    def _estimate_area(self, event: FireEvent) -> float:
        """Estimate fire area in hectares based on event data."""
        # Simplified estimation: base area + FRP contribution
        base_area = 10.0  # Minimum 10 hectares
        
        # Add area based on point count (more detections = larger fire)
        point_contribution = event.point_count * 5.0
        
        # Add area based on FRP (higher energy = larger fire)
        frp_contribution = event.max_frp * 0.5 if event.max_frp else 0.0
        
        # Apply correction factor (convex hull approximation)
        correction_factor = 1.3
        
        estimated = (base_area + point_contribution + frp_contribution) * correction_factor
        
        return min(estimated, 10000.0)  # Cap at 10,000 ha

    async def check_decay(self, session) -> int:
        """Check for events that have decayed (no new points).
        
        Args:
            session: SQLAlchemy async session
            
        Returns:
            Number of events marked as extinguished
        """
        from sqlalchemy import select
        
        threshold = datetime.now(timezone.utc) - timedelta(hours=self.settings.event_decay_hours)
        
        stmt = select(FireEvent).where(
            FireEvent.status == "active",
            FireEvent.last_seen < threshold
        )
        
        result = await session.execute(stmt)
        expired_events = result.scalars().all()
        
        for event in expired_events:
            event.status = "extinguished"
            logger.info(f"Event {event.id} marked as extinguished (no points for {self.settings.event_decay_hours}h)")
        
        return len(expired_events)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers."""
        from math import asin, cos, radians, sin, sqrt
        
        R = 6371.0  # Earth's radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * asin(sqrt(a))
        
        return R * c
