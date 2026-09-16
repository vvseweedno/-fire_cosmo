# Health check endpoint
from fastapi import APIRouter

from app.core.schemas import HealthResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    from app.core.config import settings
    
    return HealthResponse(
        status="ok",
        service="fire-burned-area-service",
        version="1.0.0",
        offline_mode=settings.offline_mode
    )
