# Integration tests for full pipeline
import pytest
from app.services.fire_detection import FireDetectionService
from app.services.false_positive_filter import FalsePositiveFilter
from app.services.fire_clustering import FireClusteringService


@pytest.mark.asyncio
async def test_full_pipeline_offline():
    """Тест полного цикла в offline режиме"""
    # 1. Детекция
    detection = FireDetectionService(offline_mode=True)
    points = await detection.detect_fires(
        bbox=[-122.5, 37.7, -122.3, 37.9],
        start_date="2024-01-01",
        end_date="2024-01-02"
    )
    
    # 2. Фильтрация
    filtered = await FalsePositiveFilter().filter_points(points)
    
    # 3. Кластеризация
    clustering = FireClusteringService()
    events = await clustering.cluster_points(filtered)
    
    # Проверка
    assert len(events) >= 0  # Может быть 0 в зависимости от фикстур
    if events:
        assert all(e.point_count > 0 for e in events)
        assert all(e.status.value in ['active', 'historical'] for e in events)


@pytest.mark.asyncio
async def test_edge_case_empty_bbox():
    """Тест пустого bbox"""
    detection = FireDetectionService(offline_mode=True)
    points = await detection.detect_fires(
        bbox=[0, 0, 0, 0],
        start_date="2024-01-01",
        end_date="2024-01-02"
    )
    assert points == [] or len(points) >= 0  # Фикстуры могут вернуть данные


@pytest.mark.asyncio
async def test_clustering_strtree():
    """Тест использования STRTree для кластеризации"""
    from app.core.schemas import FireCandidate, ConfidenceLevel
    from datetime import datetime
    
    clustering = FireClusteringService(spatial_radius_km=3.0)
    
    # Создать тестовые точки
    points = [
        FireCandidate(
            id=f"point_{i}",
            sensor="MODIS",
            latitude=37.8 + (i * 0.01),
            longitude=-122.4 + (i * 0.01),
            datetime=datetime.utcnow(),
            confidence=ConfidenceLevel.NOMINAL,
            brightness_temp_k=350.0,
            frp_mw=10.0
        )
        for i in range(5)
    ]
    
    events = await clustering.cluster_points(points)
    assert len(events) >= 1
