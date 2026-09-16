"""Deduplication module for fire event alerts."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..db.models import FireEvent
from ..settings import get_settings

logger = logging.getLogger(__name__)


class Deduplicator:
    """Prevents duplicate alerts for the same fire event."""

    def __init__(self) -> None:
        """Initialize deduplicator."""
        self.settings = get_settings()

    def should_send_alert(self, event: FireEvent, new_risk_level: str) -> tuple[bool, str]:
        """Check if an alert should be sent for this event.
        
        Args:
            event: FireEvent to check
            new_risk_level: Current risk level (critical/warning/info)
            
        Returns:
            Tuple of (should_send, reason)
        """
        # Never send alerts for info level
        if new_risk_level == "info":
            return False, "Info level - no alert needed"
        
        # Check if alert was sent recently
        if event.last_alert_at:
            time_since_alert = datetime.now(timezone.utc) - event.last_alert_at
            
            if time_since_alert < timedelta(hours=self.settings.dedup_interval_hours):
                # Check if risk level increased
                if self._risk_increased(event.last_alert_level, new_risk_level):
                    return True, f"Risk level increased from {event.last_alert_level} to {new_risk_level}"
                else:
                    return False, f"Alert sent {time_since_alert.total_seconds()/3600:.1f}h ago, no escalation"
        
        return True, "New alert or dedup interval expired"

    def _risk_increased(self, old_level: Optional[str], new_level: str) -> bool:
        """Check if risk level has increased."""
        risk_order = {"info": 0, "warning": 1, "critical": 2}
        
        if old_level is None:
            return True
        
        old_rank = risk_order.get(old_level, 0)
        new_rank = risk_order.get(new_level, 0)
        
        return new_rank > old_rank

    def record_alert_sent(self, event: FireEvent, risk_level: str) -> None:
        """Record that an alert was sent for this event.
        
        Args:
            event: FireEvent
            risk_level: Risk level of the sent alert
        """
        event.last_alert_at = datetime.now(timezone.utc)
        event.last_alert_level = risk_level
        logger.debug(f"Recorded alert for event {event.id} at level {risk_level}")
