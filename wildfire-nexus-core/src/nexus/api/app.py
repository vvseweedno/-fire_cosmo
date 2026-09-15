"""
FastAPI application configuration.
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..db.connection import init_db
from .routes import fires, health

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Wildfire Nexus Core API",
    description="API for wildfire detection and tracking system",
    version="1.0.0",
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and log startup."""
    init_db()
    logger.info("Database initialized")
    logger.info("Wildfire Nexus Core API started")


# Include routers
app.include_router(health.router)
app.include_router(fires.router)

# Serve static files (web map)
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
    logger.info(f"Static files served from {static_path}")


@app.get("/")
async def root():
    """Root endpoint - redirects to web map."""
    return {"message": "Wildfire Nexus Core API", "docs": "/docs"}
