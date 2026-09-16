# Main FastAPI Application
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.api.routes import fires, events, report, health


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    logger.info("Starting Wildfire Nexus Core service...")
    logger.info(f"Offline mode: {settings.offline_mode}")
    logger.info(f"Data directory: {settings.data_dir}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Wildfire Nexus Core service...")


app = FastAPI(
    title="Wildfire Nexus Core",
    description="Хакатонный сервис ДЗЗ: поиск очагов горения и картирование гарей",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - ограниченные origins для безопасности
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Static files (карта)
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except RuntimeError:
    logger.warning("Static directory not found, skipping static files mount")

# API Routes
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(fires.router, prefix="/api/v1", tags=["Fires"])
app.include_router(events.router, prefix="/api/v1", tags=["Events"])
app.include_router(report.router, prefix="/api/v1", tags=["Reports"])


@app.get("/")
async def root():
    """Главная страница - карта"""
    return FileResponse("app/static/index.html")


@app.get("/health")
async def health_check():
    """Legacy health endpoint"""
    return {"status": "ok", "service": "wildfire-nexus-core"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.app_env == "development")
    )
