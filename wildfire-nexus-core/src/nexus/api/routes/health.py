"""
Health check and system status API routes.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ...db.connection import get_session
from ...db.models import FireEvent, ThermalPoint
from ..schemas import HealthResponse, StatsSchema

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check(session: Session = Depends(get_session)) -> dict:
    """
    System health check endpoint.
    
    Returns:
        System status with active event count
    """
    active_count = session.query(FireEvent).filter(
        FireEvent.status == "active"
    ).count()
    
    return {
        "status": "ok",
        "events_active": active_count,
        "timestamp": datetime.utcnow(),
    }


@router.get("/stats", response_model=StatsSchema)
def get_stats(session: Session = Depends(get_session)) -> dict:
    """
    Get system statistics.
    
    Returns:
        Aggregated statistics for the last 24 hours
    """
    now = datetime.utcnow()
    period_start = now - timedelta(hours=24)
    
    # Count events by status
    total_events = session.query(FireEvent).count()
    active_events = session.query(FireEvent).filter(FireEvent.status == "active").count()
    critical_events = session.query(FireEvent).filter(
        FireEvent.risk_level == "critical"
    ).count()
    warning_events = session.query(FireEvent).filter(
        FireEvent.risk_level == "warning"
    ).count()
    info_events = session.query(FireEvent).filter(
        FireEvent.risk_level == "info"
    ).count()
    
    # Count thermal points in last 24h
    total_points_24h = session.query(ThermalPoint).filter(
        ThermalPoint.detected_at >= period_start
    ).count()
    
    return {
        "total_events": total_events,
        "active_events": active_events,
        "critical_events": critical_events,
        "warning_events": warning_events,
        "info_events": info_events,
        "total_points_24h": total_points_24h,
        "period_start": period_start,
        "period_end": now,
    }
