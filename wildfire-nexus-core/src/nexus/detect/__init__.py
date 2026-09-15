"""
Detection module for wildfire analysis.
"""

from .context_enrich import enrich_point_with_context
from .deduplicator import should_send_alert
from .prioritizer import calculate_risk_level
from .thermal_filter import filter_thermal_point
from .tracker import FireTracker

__all__ = [
    "filter_thermal_point",
    "enrich_point_with_context",
    "FireTracker",
    "should_send_alert",
    "calculate_risk_level",
]
