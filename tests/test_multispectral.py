"""
Tests for Multispectral Indices (NBR, SAVI, NDWI, dNBR)
"""

import pytest
import asyncio
import numpy as np
from app.services.multispectral_indices import MultispectralIndices
from app.services.burned_area_mapper import BurnedAreaMapper


class TestNBR:
    """Тесты индекса NBR"""
    
    def test_nbr_range(self):
        """NBR должен быть в диапазоне [-1, 1]"""
        b08 = np.random.uniform(0.3, 0.6, (100, 100))
        b12 = np.random.uniform(0.1, 0.3, (100, 100))
        nbr = MultispectralIndices.calculate_nbr(b08, b12)
        
        assert nbr.min() >= -1.0
        assert nbr.max() <= 1.0
    
    def test_nbr_formula(self):
        """Проверка формулы NBR = (B08 - B12) / (B08 + B12)"""
        b08 = np.array([[0.5, 0.6], [0.4, 0.5]])
        b12 = np.array([[0.2, 0.3], [0.1, 0.2]])
        
        nbr = MultispectralIndices.calculate_nbr(b08, b12)
        expected = (b08 - b12) / (b08 + b12)
        
        assert np.allclose(nbr, expected, atol=1e-6)
    
    def test_nbr_healthy_vegetation(self):
        """Здоровая растительность: высокий NIR, низкий SWIR → NBR > 0"""
        b08 = np.full((50, 50), 0.6)  # Высокий NIR
        b12 = np.full((50, 50), 0.2)  # Низкий SWIR
        
        nbr = MultispectralIndices.calculate_nbr(b08, b12)
        assert nbr.mean() > 0.5
    
    def test_nbr_burned_area(self):
        """Выгоревшая территория: низкий NIR, высокий SWIR → NBR < 0"""
        b08 = np.full((50, 50), 0.2)  # Низкий NIR после пожара
        b12 = np.full((50, 50), 0.4)  # Высокий SWIR после пожара
        
        nbr = MultispectralIndices.calculate_nbr(b08, b12)
        assert nbr.mean() < 0.0
    
    def test_nbr_zero_denominator(self):
        """Обработка случая деления на ноль"""
        b08 = np.zeros((10, 10))
        b12 = np.zeros((10, 10))
        
        nbr = MultispectralIndices.calculate_nbr(b08, b12)
        
        # Должен вернуть нули без ошибок
        assert np.all(nbr == 0.0)


class TestSAVI:
    """Тесты индекса SAVI"""
    
    def test_savi_range(self):
        """SAVI должен быть в разумном диапазоне"""
        b04 = np.random.uniform(0.1, 0.4, (50, 50))
        b08 = np.random.uniform(0.3, 0.7, (50, 50))
        savi = MultispectralIndices.calculate_savi(b04, b08)
        
        assert savi.min() >= -1.0
        assert savi.max() <= 1.0
    
    def test_savi_formula(self):
        """Проверка формулы SAVI = ((B08 - B04) / (B08 + B04 + L)) * (1 + L)"""
        b04 = np.array([[0.2, 0.3], [0.1, 0.2]])
        b08 = np.array([[0.5, 0.6], [0.4, 0.5]])
        L = 0.5
        
        savi = MultispectralIndices.calculate_savi(b04, b08, L)
        expected = ((b08 - b04) / (b08 + b04 + L)) * (1.0 + L)
        
        assert np.allclose(savi, expected, atol=1e-6)
    
    def test_savi_high_vegetation(self):
        """Высокая растительность: SAVI > 0.5"""
        b04 = np.full((50, 50), 0.1)  # Низкий Red
        b08 = np.full((50, 50), 0.7)  # Высокий NIR
        
        savi = MultispectralIndices.calculate_savi(b04, b08)
        assert savi.mean() > 0.5


class TestNDWI:
    """Тесты индекса NDWI"""
    
    def test_ndwi_range(self):
        """NDWI должен быть в диапазоне [-1, 1]"""
        b03 = np.random.uniform(0.1, 0.5, (50, 50))
        b08 = np.random.uniform(0.2, 0.6, (50, 50))
        ndwi = MultispectralIndices.calculate_ndwi(b03, b08)
        
        assert ndwi.min() >= -1.0
        assert ndwi.max() <= 1.0
    
    def test_ndwi_water_body(self):
        """Водный объект: высокий NDWI > 0.3"""
        b03 = np.full((50, 50), 0.4)  # Высокий Green (вода)
        b08 = np.full((50, 50), 0.1)  # Низкий NIR (вода поглощает)
        
        ndwi = MultispectralIndices.calculate_ndwi(b03, b08)
        assert ndwi.mean() > 0.3
    
    def test_ndwi_dry_vegetation(self):
        """Сухая растительность: низкий NDWI < 0"""
        b03 = np.full((50, 50), 0.1)  # Низкий Green
        b08 = np.full((50, 50), 0.5)  # Высокий NIR
        
        ndwi = MultispectralIndices.calculate_ndwi(b03, b08)
        assert ndwi.mean() < 0.0


class TestDNBR:
    """Тесты индекса dNBR"""
    
    def test_dnbr_positive_for_burned(self):
        """После пожара NBR падает → dNBR > 0"""
        nbr_pre = np.full((50, 50), 0.5)   # До пожара
        nbr_post = np.full((50, 50), 0.2)  # После пожара
        
        dnbr = MultispectralIndices.calculate_dnbr(nbr_pre, nbr_post)
        assert dnbr.mean() > 0.2
    
    def test_dnbr_formula(self):
        """Проверка формулы dNBR = NBR_pre - NBR_post"""
        nbr_pre = np.random.uniform(0.3, 0.7, (50, 50))
        nbr_post = np.random.uniform(0.1, 0.4, (50, 50))
        
        dnbr = MultispectralIndices.calculate_dnbr(nbr_pre, nbr_post)
        expected = nbr_pre - nbr_post
        
        assert np.allclose(dnbr, expected, atol=1e-6)
    
    def test_dnbr_no_pre_image(self):
        """Если нет pre-снимка, возвращается -NBR_post"""
        nbr_post = np.random.uniform(0.2, 0.6, (50, 50))
        
        dnbr = MultispectralIndices.calculate_dnbr(None, nbr_post)
        expected = -nbr_post
        
        assert np.allclose(dnbr, expected, atol=1e-6)
    
    def test_dnbr_unburned(self):
        """Невыгоревшая территория: dNBR ≈ 0"""
        nbr_pre = np.full((50, 50), 0.4)
        nbr_post = np.full((50, 50), 0.38)  # Почти без изменений
        
        dnbr = MultispectralIndices.calculate_dnbr(nbr_pre, nbr_post)
        assert abs(dnbr.mean()) < 0.05


class TestRdNBR:
    """Тесты индекса RdNBR"""
    
    def test_rdnbr_formula(self):
        """Проверка формулы RdNBR = dNBR / sqrt(|NBR_pre| + eps)"""
        nbr_pre = np.full((50, 50), 0.5)
        nbr_post = np.full((50, 50), 0.2)
        dnbr = nbr_pre - nbr_post
        
        rdnbr = MultispectralIndices.calculate_rdnbr(dnbr, nbr_pre)
        expected = dnbr / np.sqrt(np.abs(nbr_pre) + 1e-8)
        
        assert np.allclose(rdnbr, expected, atol=1e-6)
    
    def test_rdnbr_no_pre_image(self):
        """Если нет pre-снимка, возвращается dNBR без изменений"""
        dnbr = np.random.uniform(0.1, 0.5, (50, 50))
        
        rdnbr = MultispectralIndices.calculate_rdnbr(dnbr, None)
        
        assert np.allclose(rdnbr, dnbr, atol=1e-6)


class TestSeverityClassification:
    """Тесты классификации severity"""
    
    @pytest.mark.asyncio
    async def test_severity_thresholds(self):
        """Проверка порогов: unburned < 0.10, low 0.10-0.27, moderate 0.27-0.44, high > 0.44"""
        mapper = BurnedAreaMapper()
        
        # Unburned (dNBR < 0.10)
        dnbr_low = np.full((10, 10), 0.05)
        severity = await mapper.classify_severity(dnbr_low)
        # Все пиксели должны быть unburned (0) или low (1) в зависимости от реализации
        assert np.all(severity <= 1)
        
        # Low severity (0.10 <= dNBR < 0.27)
        dnbr_low_sev = np.full((10, 10), 0.20)
        severity = await mapper.classify_severity(dnbr_low_sev)
        assert np.all(severity == 1)  # low
        
        # Moderate severity (0.27 <= dNBR < 0.44)
        dnbr_mod = np.full((10, 10), 0.35)
        severity = await mapper.classify_severity(dnbr_mod)
        assert np.all(severity == 2)  # moderate
        
        # High severity (dNBR >= 0.44)
        dnbr_high = np.full((10, 10), 0.50)
        severity = await mapper.classify_severity(dnbr_high)
        assert np.all(severity == 3)  # high
    
    @pytest.mark.asyncio
    async def test_severity_mixed_values(self):
        """Смешанные значения severity"""
        mapper = BurnedAreaMapper()
        
        dnbr = np.array([
            [0.05, 0.15, 0.35, 0.55],
            [0.08, 0.20, 0.40, 0.60]
        ])
        
        severity = await mapper.classify_severity(dnbr)
        
        # Проверка каждого пикселя
        assert severity[0, 0] == 0  # unburned
        assert severity[0, 1] == 1  # low
        assert severity[0, 2] == 2  # moderate
        assert severity[0, 3] == 3  # high
        assert severity[1, 0] == 0  # unburned
        assert severity[1, 1] == 1  # low
        assert severity[1, 2] == 2  # moderate
        assert severity[1, 3] == 3  # high
    
    @pytest.mark.asyncio
    async def test_severity_with_nodata(self):
        """Классификация с маской nodata"""
        mapper = BurnedAreaMapper()
        
        dnbr = np.full((10, 10), 0.3)
        nodata_mask = np.zeros((10, 10), dtype=bool)
        nodata_mask[5:7, 5:7] = True
        
        severity = await mapper.classify_severity(dnbr, nodata_mask)
        
        # Основной фон - moderate
        assert severity[0, 0] == 2
        # NoData области = 255
        assert severity[5, 5] == 255
        assert severity[6, 6] == 255


class TestFireRiskIndex:
    """Тесты Fire Risk Index"""
    
    def test_fire_risk_index_range(self):
        """Fire Risk Index должен быть в диапазоне [0, 100]"""
        ndvi = np.random.uniform(-1, 1, (50, 50))
        swir = np.random.uniform(0, 10000, (50, 50))
        land_cover = np.random.randint(1, 6, (50, 50))
        
        risk = MultispectralIndices.calculate_fire_risk_index(
            ndvi=ndvi, swir=swir, land_cover=land_cover
        )
        
        assert risk.min() >= 0.0
        assert risk.max() <= 100.0
    
    def test_fire_risk_index_high_risk(self):
        """Высокий риск: сухая растительность, сухая почва, лес"""
        ndvi = np.full((50, 50), -0.5)  # Сухая растительность
        swir = np.full((50, 50), 8000)  # Сухая почва
        land_cover = np.full((50, 50), 1)  # Лес
        
        risk = MultispectralIndices.calculate_fire_risk_index(
            ndvi=ndvi, swir=swir, land_cover=land_cover
        )
        
        assert risk.mean() > 65  # Высокий риск (ожидаем ~67.5)
    
    def test_fire_risk_index_low_risk(self):
        """Низкий риск: здоровая растительность, влажная почва, вода"""
        ndvi = np.full((50, 50), 0.8)  # Здоровая растительность
        swir = np.full((50, 50), 1000)  # Влажная почва
        land_cover = np.full((50, 50), 5)  # Вода
        
        risk = MultispectralIndices.calculate_fire_risk_index(
            ndvi=ndvi, swir=swir, land_cover=land_cover
        )
        
        assert risk.mean() < 20  # Низкий риск
    
    def test_fire_risk_index_custom_weights(self):
        """Fire Risk Index с пользовательскими весами"""
        ndvi = np.full((50, 50), 0.0)
        swir = np.full((50, 50), 5000)
        land_cover = np.full((50, 50), 2)
        
        weights = {
            'vegetation': 50.0,
            'soil': 50.0,
            'fuel': 0.0,
            'history': 0.0,
        }
        
        risk = MultispectralIndices.calculate_fire_risk_index(
            ndvi=ndvi, swir=swir, land_cover=land_cover, weights=weights
        )
        
        assert risk.min() >= 0.0
        assert risk.max() <= 100.0


class TestBAISMM:
    """Тесты индекса BAISMM"""
    
    def test_baismm_exists(self):
        """BAISMM должен рассчитываться без ошибок"""
        b04 = np.random.uniform(0.1, 0.4, (50, 50))
        b08 = np.random.uniform(0.3, 0.7, (50, 50))
        b11 = np.random.uniform(0.2, 0.5, (50, 50))
        b12 = np.random.uniform(0.1, 0.4, (50, 50))
        
        baismm = MultispectralIndices.calculate_baismm(b04, b08, b11, b12)
        
        assert baismm.shape == (50, 50)
        assert not np.any(np.isnan(baismm))
    
    def test_baismm_zero_channels(self):
        """BAISMM при нулевых каналах"""
        b04 = np.zeros((10, 10))
        b08 = np.zeros((10, 10))
        b11 = np.zeros((10, 10))
        b12 = np.zeros((10, 10))
        
        baismm = MultispectralIndices.calculate_baismm(b04, b08, b11, b12)
        
        # Не должно быть NaN или inf
        assert not np.any(np.isnan(baismm))
        assert not np.any(np.isinf(baismm))


class TestNDVI:
    """Тесты индекса NDVI"""
    
    def test_ndvi_range(self):
        """NDVI должен быть в диапазоне [-1, 1]"""
        b04 = np.random.uniform(0.1, 0.4, (50, 50))
        b08 = np.random.uniform(0.3, 0.7, (50, 50))
        ndvi = MultispectralIndices.calculate_ndvi(b04, b08)
        
        assert ndvi.min() >= -1.0
        assert ndvi.max() <= 1.0
    
    def test_ndvi_dense_vegetation(self):
        """Густая растительность: NDVI > 0.6"""
        b04 = np.full((50, 50), 0.05)  # Низкий Red
        b08 = np.full((50, 50), 0.7)   # Высокий NIR
        
        ndvi = MultispectralIndices.calculate_ndvi(b04, b08)
        assert ndvi.mean() > 0.6
    
    def test_ndvi_bare_soil(self):
        """Голая почва: NDVI около 0.1-0.2"""
        b04 = np.full((50, 50), 0.25)
        b08 = np.full((50, 50), 0.30)
        
        ndvi = MultispectralIndices.calculate_ndvi(b04, b08)
        assert 0.0 < ndvi.mean() < 0.3
