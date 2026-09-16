"""
Risk prioritization for fire events.

Classifies fires as critical/warning/info based on threat to settlements.
"""

import logging
from typing import Any

from ..settings import get_settings

logger = logging.getLogger(__name__)


def calculate_risk_level(enriched_point: dict[str, Any]) -> str:
    """
    Calculate risk level for a fire event.
    
    Rules (in priority order):
    
    CRITICAL:
      - Distance to settlement with population >1000 < 10 km
      - AND wind blowing towards settlement (±30°)
      - AND wind speed > 5 m/s
    
    WARNING:
      - Distance to any settlement < 15 km
      - OR max_frp > 100 MW
    
    INFO:
      - Everything else
    
    Args:
        enriched_point: Point with context data
        
    Returns:
        Risk level string: 'critical', 'warning', or 'info'
    """
    settings = get_settings()
    alert_config = settings.alert
    
    # Extract relevant fields
    distance_km = enriched_point.get("distance_to_settlement_km")
    population = enriched_point.get("settlement_population", 0) or 0
    wind_towards = enriched_point.get("wind_towards_settlement", False)
    weather = enriched_point.get("weather", {})
    wind_speed = weather.get("wind_speed") if weather else None
    frp = enriched_point.get("frp")
    
    # Get thresholds from config
    critical_distance = alert_config.get("critical_distance_km", 10)
    warning_distance = alert_config.get("warning_distance_km", 15)
    min_pop_critical = alert_config.get("min_population_critical", 1000)
    
    # Check CRITICAL conditions
    if (
        distance_km is not None
        and distance_km < critical_distance
        and population > min_pop_critical
        and wind_towards
        and wind_speed is not None
        and wind_speed > 5
    ):
        logger.info(
            f"CRITICAL risk: {distance_km}km from settlement ({population} pop), "
            f"wind towards={wind_towards}, speed={wind_speed}m/s"
        )
        return "critical"
    
    # Check WARNING conditions
    if distance_km is not None and distance_km < warning_distance:
        logger.debug(f"WARNING risk: {distance_km}km from settlement")
        return "warning"
    
    if frp is not None and frp > 100:
        logger.debug(f"WARNING risk: high FRP ({frp}MW)")
        return "warning"
    
    # Default to INFO
    logger.debug("INFO risk: no elevated risk factors")
    return "info"


def batch_calculate_risk(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Calculate risk levels for multiple points.
    
    Args:
        points: List of enriched point dictionaries
        
    Returns:
        Points with added risk_level field
    """
    for point in points:
        point["risk_level"] = calculate_risk_level(point)
    
    return points
