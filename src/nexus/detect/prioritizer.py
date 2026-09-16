"""Risk prioritization module for fire events."""

import logging
from enum import Enum
from typing import Optional

from ..db.models import FireEvent
from ..settings import get_settings

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class Prioritizer:
    """Assigns risk levels to fire events based on context."""

    def __init__(self) -> None:
        """Initialize prioritizer."""
        self.settings = get_settings()

    def calculate_risk(
        self,
        event: FireEvent,
        distance_to_settlement_km: Optional[float] = None,
        settlement_population: Optional[int] = None,
        wind_towards_settlement: bool = False,
        wind_speed_ms: Optional[float] = None,
        fire_weather_index: int = 0,
    ) -> RiskLevel:
        """Calculate risk level for a fire event.
        
        Args:
            event: FireEvent to assess
            distance_to_settlement_km: Distance to nearest settlement
            settlement_population: Population of nearest settlement
            wind_towards_settlement: Whether wind blows towards settlement
            wind_speed_ms: Wind speed in m/s
            fire_weather_index: Fire weather index (0-5)
            
        Returns:
            RiskLevel (critical/warning/info)
        """
        # CRITICAL criteria (all must be met)
        if self._is_critical(
            distance_to_settlement_km,
            settlement_population,
            wind_towards_settlement,
            wind_speed_ms,
        ):
            logger.info(f"Event {event.id}: CRITICAL - threat to populated area")
            return RiskLevel.CRITICAL
        
        # WARNING criteria (any)
        if self._is_warning(
            distance_to_settlement_km,
            event.max_frp,
        ):
            logger.info(f"Event {event.id}: WARNING - potential threat")
            return RiskLevel.WARNING
        
        # Default to INFO
        return RiskLevel.INFO

    def _is_critical(
        self,
        distance_km: Optional[float],
        population: Optional[int],
        wind_towards: bool,
        wind_speed: Optional[float],
    ) -> bool:
        """Check if event meets CRITICAL criteria."""
        # All conditions must be met:
        # - Distance to settlement with pop > 1000 < 10 km
        # - Wind towards settlement
        # - Wind speed > 5 m/s
        
        if distance_km is None or population is None:
            return False
        
        if distance_km >= self.settings.critical_distance_km:
            return False
        
        if population < self.settings.min_population_critical:
            return False
        
        if not wind_towards:
            return False
        
        if wind_speed is None or wind_speed < 5.0:
            return False
        
        return True

    def _is_warning(
        self,
        distance_km: Optional[float],
        max_frp: float,
    ) -> bool:
        """Check if event meets WARNING criteria."""
        # Any condition:
        # - Distance to any settlement < 15 km
        # - Max FRP > 100 MW
        
        if distance_km is not None and distance_km < self.settings.warning_distance_km:
            return True
        
        if max_frp > 100.0:
            return True
        
        return False

    def apply_to_event(
        self,
        event: FireEvent,
        distance_to_settlement_km: Optional[float] = None,
        settlement_population: Optional[int] = None,
        wind_towards_settlement: bool = False,
        wind_speed_ms: Optional[float] = None,
        fire_weather_index: int = 0,
    ) -> RiskLevel:
        """Calculate and apply risk level to event.
        
        Args:
            event: FireEvent to update
            distance_to_settlement_km: Distance to nearest settlement
            settlement_population: Population of nearest settlement
            wind_towards_settlement: Whether wind blows towards settlement
            wind_speed_ms: Wind speed in m/s
            fire_weather_index: Fire weather index (0-5)
            
        Returns:
            Calculated RiskLevel
        """
        risk_level = self.calculate_risk(
            event,
            distance_to_settlement_km,
            settlement_population,
            wind_towards_settlement,
            wind_speed_ms,
            fire_weather_index,
        )
        
        event.risk_level = risk_level.value
        return risk_level
