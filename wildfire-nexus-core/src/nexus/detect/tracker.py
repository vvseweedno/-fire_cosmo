"""
Fire event tracker - converts thermal points into wildfire events.

Tracks fires as processes with lifecycle, not just point detections.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from shapely.geometry import MultiPoint

from ..db.connection import get_session_sync
from ..db.models import FireEvent, ThermalPoint
from ..settings import get_settings

logger = logging.getLogger(__name__)


class FireTracker:
    """
    Tracks fire events by aggregating thermal points over time.
    
    Key features:
    - Groups nearby points into single events
    - Maintains event lifecycle (active → contained/extinguished)
    - Estimates fire area from point cluster
    """
    
    def __init__(self):
        settings = get_settings()
        self.tracking_radius_km = settings.detect.get("tracking_radius_km", 3.0)
        self.event_decay_hours = settings.detect.get("event_decay_hours", 12)
    
    def _find_matching_event(
        self,
        lat: float,
        lon: float,
        session,
    ) -> FireEvent | None:
        """
        Find active event within tracking radius.
        
        Args:
            lat: Point latitude
            lon: Point longitude
            session: DB session
            
        Returns:
            Matching FireEvent or None
        """
        # Get all active events
        active_events = session.query(FireEvent).filter(
            FireEvent.status == "active"
        ).all()
        
        # Find nearest active event within radius
        for event in active_events:
            distance = self._calculate_distance(
                lat, lon, event.centroid_lat, event.centroid_lon
            )
            
            if distance <= self.tracking_radius_km:
                return event
        
        return None
    
    def _calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate distance between two points in km."""
        from math import asin, cos, radians, sin, sqrt
        
        R = 6371  # Earth radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * asin(sqrt(a))
        
        return R * c
    
    def _estimate_area(self, points: list[ThermalPoint]) -> float:
        """
        Estimate fire area in hectares from point cluster.
        
        Uses convex hull with 1.3 correction factor.
        
        Args:
            points: List of thermal points belonging to event
            
        Returns:
            Estimated area in hectares
        """
        if len(points) < 3:
            # Not enough points for polygon, estimate from point count
            return len(points) * 0.5  # Rough estimate: 0.5 ha per point
        
        coords = [(p.longitude, p.latitude) for p in points]
        
        try:
            multi_point = MultiPoint(coords)
            hull = multi_point.convex_hull
            
            # Area in square degrees, convert to hectares
            # Approximate: 1 sq degree ≈ 111km × 111km × cos(lat)
            area_sq_degrees = hull.area
            
            # Get centroid for latitude correction
            centroid = hull.centroid
            lat_rad = radians(centroid.y)
            
            # Convert to square kilometers
            area_sq_km = area_sq_degrees * (111 ** 2) * cos(lat_rad)
            
            # Convert to hectares (1 sq km = 100 ha)
            area_ha = area_sq_km * 100
            
            # Apply correction factor (convex hull underestimates)
            return area_ha * 1.3
            
        except Exception as e:
            logger.warning(f"Area estimation failed: {e}")
            return len(points) * 0.5
    
    def process_point(self, point_data: dict[str, Any]) -> FireEvent:
        """
        Process a thermal point and update/create event.
        
        Args:
            point_data: Enriched thermal point dictionary
            
        Returns:
            Associated FireEvent
        """
        session = get_session_sync()
        try:
            lat = point_data["latitude"]
            lon = point_data["longitude"]
            
            # Try to find matching active event
            event = self._find_matching_event(lat, lon, session)
            
            if event:
                # Update existing event
                event.last_seen = datetime.utcnow()
                event.point_count += 1
                
                # Update max FRP
                frp = point_data.get("frp")
                if frp and frp > event.max_frp:
                    event.max_frp = frp
                
                # Recalculate centroid (weighted average)
                total_points = event.point_count
                weight = 1.0 / total_points
                event.centroid_lat = event.centroid_lat * (1 - weight) + lat * weight
                event.centroid_lon = event.centroid_lon * (1 - weight) + lon * weight
                
                # Update area estimate
                event_points = session.query(ThermalPoint).filter(
                    ThermalPoint.event_id == event.id
                ).all()
                event.area_estimate_ha = self._estimate_area(event_points)
                
                logger.debug(f"Updated event {event.id} with new point")
            else:
                # Create new event
                event = FireEvent(
                    id=uuid4().hex,
                    status="active",
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    centroid_lat=lat,
                    centroid_lon=lon,
                    point_count=1,
                    max_frp=point_data.get("frp", 0) or 0,
                    area_estimate_ha=0.5,  # Initial estimate
                    risk_level=point_data.get("risk_level", "info"),
                    nearest_settlement=point_data.get("nearest_settlement"),
                )
                session.add(event)
                session.flush()  # Get generated ID
                
                logger.info(f"Created new fire event {event.id} at ({lat}, {lon})")
            
            # Create thermal point record
            thermal_point = ThermalPoint(
                id=uuid4().hex,
                detected_at=datetime.fromisoformat(point_data["detected_at"]) if isinstance(point_data.get("detected_at"), str) else point_data.get("detected_at", datetime.utcnow()),
                source=point_data.get("source", "VIIRS"),
                latitude=lat,
                longitude=lon,
                brightness=point_data.get("brightness", 0),
                confidence=point_data.get("confidence", 0),
                frp=point_data.get("frp"),
                raw_payload=point_data.get("raw_payload"),
                event_id=event.id,
            )
            session.add(thermal_point)
            
            session.commit()
            
            return event
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to process point: {e}")
            raise
        finally:
            session.close()
    
    def mark_extinguished_events(self) -> int:
        """
        Mark events as extinguished if no new points received.
        
        Returns:
            Number of events marked as extinguished
        """
        session = get_session_sync()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.event_decay_hours)
            
            # Find active events with no recent points
            old_events = session.query(FireEvent).filter(
                FireEvent.status == "active",
                FireEvent.last_seen < cutoff_time,
            ).all()
            
            count = 0
            for event in old_events:
                event.status = "extinguished"
                count += 1
                logger.info(f"Marked event {event.id} as extinguished")
            
            session.commit()
            return count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to mark extinguished events: {e}")
            return 0
        finally:
            session.close()
