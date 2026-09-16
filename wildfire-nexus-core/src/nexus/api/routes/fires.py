"""
Fire events API routes.
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...db.connection import get_session
from ...db.models import FireEvent, ThermalPoint
from ..schemas import FireEventDetailSchema, FireEventSchema, StatsSchema

router = APIRouter(prefix="/api/v1", tags=["fires"])


@router.get("/events", response_model=list[FireEventSchema])
def list_events(
    status: str | None = "active",
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session),
) -> list[FireEvent]:
    """
    List fire events with optional filtering.
    
    Args:
        status: Filter by status (active/contained/extinguished)
        limit: Maximum number of results
        offset: Pagination offset
        
    Returns:
        List of fire events
    """
    query = session.query(FireEvent)
    
    if status:
        query = query.filter(FireEvent.status == status)
    
    query = query.order_by(FireEvent.last_seen.desc())
    
    return query.offset(offset).limit(limit).all()


@router.get("/events/{event_id}", response_model=FireEventDetailSchema)
def get_event(event_id: str, session: Session = Depends(get_session)) -> FireEvent:
    """
    Get details for a specific fire event.
    
    Args:
        event_id: Event UUID
        
    Returns:
        Fire event details
        
    Raises:
        HTTPException: If event not found
    """
    event = session.query(FireEvent).filter(FireEvent.id == event_id).first()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return event


@router.get("/events/{event_id}/points")
def get_event_points(event_id: str, session: Session = Depends(get_session)) -> list[dict]:
    """
    Get thermal points for a specific event.
    
    Args:
        event_id: Event UUID
        
    Returns:
        List of thermal points
        
    Raises:
        HTTPException: If event not found
    """
    event = session.query(FireEvent).filter(FireEvent.id == event_id).first()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    points = session.query(ThermalPoint).filter(
        ThermalPoint.event_id == event_id
    ).order_by(ThermalPoint.detected_at.desc()).all()
    
    return [
        {
            "id": p.id,
            "detected_at": p.detected_at.isoformat(),
            "latitude": p.latitude,
            "longitude": p.longitude,
            "brightness": p.brightness,
            "confidence": p.confidence,
            "frp": p.frp,
            "source": p.source,
        }
        for p in points
    ]
