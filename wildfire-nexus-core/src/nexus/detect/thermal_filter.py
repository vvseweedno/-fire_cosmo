"""
Thermal point filtering based on thresholds and whitelist.

Filters out false positives (industrial sources, low-confidence detections).
"""

import json
import logging
from pathlib import Path
from typing import Any

from ..settings import get_settings

logger = logging.getLogger(__name__)

# Whitelist path
WHITELIST_PATH = Path(__file__).parent.parent.parent / "data" / "whitelist_industrial.json"


def _load_whitelist() -> list[dict[str, Any]]:
    """Load industrial zones whitelist."""
    if not WHITELIST_PATH.exists():
        return []
    
    try:
        with open(WHITELIST_PATH) as f:
            data = json.load(f)
        return data.get("industrial_zones", [])
    except Exception as e:
        logger.warning(f"Failed to load whitelist: {e}")
        return []


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate approximate distance between two points in km.
    
    Uses Haversine formula for reasonable accuracy.
    """
    from math import asin, cos, radians, sin, sqrt
    
    R = 6371  # Earth's radius in km
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * asin(sqrt(a))
    
    return R * c


def filter_thermal_point(point: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Filter a thermal point based on thresholds and whitelist.
    
    Args:
        point: Thermal point dictionary with brightness, confidence, frp, etc.
        
    Returns:
        Tuple of (passed_filter, rejection_reason)
        - (True, None) if point passes all filters
        - (False, reason) if point should be rejected
    """
    settings = get_settings()
    detect_config = settings.detect
    
    # Extract point data
    source = point.get("source", "VIIRS")
    brightness = point.get("brightness", 0)
    confidence = point.get("confidence", 0)
    frp = point.get("frp")
    lat = point.get("latitude", 0)
    lon = point.get("longitude", 0)
    
    # Threshold checks
    min_brightness = (
        detect_config.get("min_brightness_viirs", 320)
        if source == "VIIRS"
        else detect_config.get("min_brightness_modis", 310)
    )
    min_confidence = detect_config.get("min_confidence", 0.4)
    min_frp = detect_config.get("min_frp", 5.0)
    
    # Check brightness
    if brightness < min_brightness:
        return False, f"brightness_too_low ({brightness}K < {min_brightness}K)"
    
    # Check confidence
    if confidence < min_confidence:
        return False, f"confidence_too_low ({confidence} < {min_confidence})"
    
    # Check FRP if available
    if frp is not None and frp < min_frp:
        return False, f"frp_too_low ({frp}MW < {min_frp}MW)"
    
    # Check whitelist (industrial zones)
    whitelist = _load_whitelist()
    whitelist_radius_km = 2.0  # Fixed 2km radius for industrial zones
    
    for zone in whitelist:
        zone_lat = zone.get("latitude")
        zone_lon = zone.get("longitude")
        
        if zone_lat is None or zone_lon is None:
            continue
        
        distance = _distance_km(lat, lon, zone_lat, zone_lon)
        
        if distance <= whitelist_radius_km:
            # Check if this is a stable industrial source
            # For now, we reject all points near known industrial zones
            # In production, you'd track stability over time
            return False, f"near_industrial_zone ({zone.get('name', 'unknown')}, {distance:.1f}km)"
    
    # All filters passed
    return True, None


def batch_filter_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter multiple thermal points.
    
    Args:
        points: List of thermal point dictionaries
        
    Returns:
        List of points that passed the filter
    """
    filtered = []
    rejection_reasons: dict[str, int] = {}
    
    for point in points:
        passed, reason = filter_thermal_point(point)
        
        if passed:
            filtered.append(point)
        else:
            # Track rejection reasons for logging
            if reason:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    
    logger.info(
        f"Filtered {len(filtered)}/{len(points)} points, "
        f"rejected {len(points) - len(filtered)}"
    )
    
    if rejection_reasons:
        for reason, count in rejection_reasons.items():
            logger.debug(f"Rejection reason '{reason}': {count} points")
    
    return filtered
