"""
Alert deduplication - prevents duplicate notifications for same event.
"""

import logging
from datetime import datetime, timedelta

from ..db.models import FireEvent
from ..settings import get_settings

logger = logging.getLogger(__name__)


def should_send_alert(event: FireEvent, current_risk_level: str) -> tuple[bool, str]:
    """
    Determine if an alert should be sent for this event.
    
    Implements deduplication logic:
    - Don't send alerts more frequently than dedup_interval_hours
    - Exception: send immediately if risk level increases
    
    Args:
        event: FireEvent instance
        current_risk_level: Current calculated risk level
        
    Returns:
        Tuple of (should_send, reason)
    """
    settings = get_settings()
    dedup_interval = settings.detect.get("dedup_interval_hours", 6)
    
    # Risk level priority order
    risk_priority = {"info": 0, "warning": 1, "critical": 2}
    
    # Check if alert was ever sent
    if event.last_alert_at is None:
        return True, "first_alert"
    
    # Check if risk level increased
    current_priority = risk_priority.get(current_risk_level, 0)
    last_priority = risk_priority.get(event.last_alert_level or "info", 0)
    
    if current_priority > last_priority:
        return True, f"risk_increased ({event.last_alert_level} → {current_risk_level})"
    
    # Check dedup interval
    time_since_alert = datetime.utcnow() - event.last_alert_at
    
    if time_since_alert < timedelta(hours=dedup_interval):
        hours_remaining = dedup_interval - time_since_alert.total_seconds() / 3600
        return False, f"dedup_active ({hours_remaining:.1f}h until next alert)"
    
    # Same risk level, but dedup interval passed - allow UPDATE alert
    return True, "periodic_update"


def update_event_alert_status(
    event: FireEvent,
    risk_level: str,
    session,
) -> None:
    """
    Update event's alert tracking fields.
    
    Args:
        event: FireEvent instance
        risk_level: Risk level that was alerted
        session: DB session
    """
    event.last_alert_at = datetime.utcnow()
    event.last_alert_level = risk_level
    session.add(event)
    logger.debug(f"Updated alert status for event {event.id}: {risk_level}")
