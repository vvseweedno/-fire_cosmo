"""
Main entry point for Wildfire Nexus Core.

Runs the full system: ingest, detect, alert, and API server.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .db.connection import init_db, get_session_sync
from .db.models import FireEvent
from .detect import (
    FireTracker,
    enrich_point_with_context,
    filter_thermal_point,
    calculate_risk_level,
)
from .detect.deduplicator import should_send_alert, update_event_alert_status
from .ingest import BoundingBox, fetch_firms_data, process_firms_points
from .ingest.weather import fetch_weather_data
from .alert.telegram_bot import send_telegram_alert_sync
from .settings import get_settings
from .api.app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def ingest_and_process():
    """
    Main ingestion and processing loop.
    
    1. Fetch thermal points from FIRMS
    2. Filter noise
    3. Enrich with context
    4. Track events
    5. Send alerts for critical events
    """
    settings = get_settings()
    
    if not settings.regions:
        logger.warning("No regions configured, skipping ingest")
        return
    
    tracker = FireTracker()
    
    for region_config in settings.regions:
        region_name = region_config.get("name", "unknown")
        bbox_coords = region_config.get("bbox", [80.0, 50.0, 105.0, 62.0])
        
        region = BoundingBox(
            min_lon=bbox_coords[0],
            min_lat=bbox_coords[1],
            max_lon=bbox_coords[2],
            max_lat=bbox_coords[3],
        )
        
        logger.info(f"Processing region: {region_name}")
        
        # Fetch data for last 24 hours
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=24)
        
        try:
            # Fetch from FIRMS
            raw_points = await fetch_firms_data(
                region=region,
                date_range=(start_date, end_date),
                source=settings.ingest.get("firms_source", "VIIRS"),
            )
            
            if not raw_points:
                logger.info(f"No points fetched for {region_name}")
                continue
            
            logger.info(f"Fetched {len(raw_points)} raw points for {region_name}")
            
            # Process points
            processed = process_firms_points(raw_points)
            
            # Filter
            filtered = []
            for point in processed:
                passed, reason = filter_thermal_point(point)
                if passed:
                    filtered.append(point)
                else:
                    logger.debug(f"Filtered out point: {reason}")
            
            logger.info(f"Filtered to {len(filtered)} points for {region_name}")
            
            # Enrich and track
            for point in filtered:
                # Fetch weather (optional, degrades gracefully)
                weather = await fetch_weather_data(
                    point["latitude"],
                    point["longitude"],
                )
                
                # Enrich with context
                enriched = enrich_point_with_context(point, weather)
                
                # Calculate risk
                enriched["risk_level"] = calculate_risk_level(enriched)
                
                # Track event
                event = tracker.process_point(enriched)
                
                # Check if alert should be sent
                session = get_session_sync()
                try:
                    should_alert, reason = should_send_alert(event, event.risk_level)
                    
                    if should_alert and event.risk_level in ["critical", "warning"]:
                        alert_type = "UPDATE" if "update" in reason else "NEW"
                        
                        # Send Telegram alert (non-blocking)
                        send_telegram_alert_sync(event, alert_type)
                        
                        # Update alert status
                        update_event_alert_status(session, event, event.risk_level)
                        session.commit()
                finally:
                    session.close()
            
            # Mark old events as extinguished
            extinguished_count = tracker.mark_extinguished_events()
            if extinguished_count > 0:
                logger.info(f"Marked {extinguished_count} events as extinguished")
                
        except Exception as e:
            logger.error(f"Error processing region {region_name}: {e}", exc_info=True)


def run_scheduler():
    """Run the main scheduler."""
    settings = get_settings()
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Create scheduler
    scheduler = AsyncIOScheduler()
    
    # Add job for regular ingestion
    ingest_interval = settings.ingest.get("firms_interval_minutes", 30)
    
    scheduler.add_job(
        ingest_and_process,
        trigger="interval",
        minutes=ingest_interval,
        id="ingest_firms",
        name="Fetch and process FIRMS data",
        next_run_time=datetime.utcnow(),  # Run immediately on start
    )
    
    logger.info(f"Scheduled ingest job every {ingest_interval} minutes")
    
    # Start scheduler
    scheduler.start()
    
    return scheduler


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    # Run scheduler in background
    scheduler = run_scheduler()
    
    # Start API server
    api_host = settings.api.get("host", "0.0.0.0")
    api_port = settings.api.get("port", 8000)
    
    logger.info(f"Starting API server on {api_host}:{api_port}")
    
    try:
        uvicorn.run(app, host=api_host, port=api_port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        scheduler.shutdown()
