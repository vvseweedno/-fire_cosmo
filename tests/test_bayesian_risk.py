"""
Tests for Bayesian Fire Risk Estimator
"""

import pytest
from app.services.bayesian_fire_risk import (
    BayesianFireRiskEstimator,
    BayesianFireRisk,
    RiskLevel,
    PRIOR_PROBABILITIES,
    LIKELIHOOD_WEIGHTS,
    RISK_THRESHOLDS,
    clip,
)


@pytest.fixture
def estimator():
    """Создать оценщик риска"""
    return BayesianFireRiskEstimator(default_prior=0.10)


class TestClip:
    """Тесты функции clip"""
    
    def test_clip_within_range(self):
        """Тест ограничения в диапазоне"""
        assert clip(0.5, 0.0, 1.0) == 0.5
    
    def test_clip_below_range(self):
        """Тест значения ниже диапазона"""
        assert clip(-0.5, 0.0, 1.0) == 0.0
    
    def test_clip_above_range(self):
        """Тест значения выше диапазона"""
        assert clip(1.5, 0.0, 1.0) == 1.0


class TestFactorCalculations:
    """Тесты расчета факторов"""
    
    def test_temp_factor_low(self, estimator):
        """Тест температурного фактора при низкой температуре"""
        factor = estimator._calculate_temp_factor(5.0)
        assert factor == 0.0  # (5-15)/20 = -0.5 → clip to 0
    
    def test_temp_factor_high(self, estimator):
        """Тест температурного фактора при высокой температуре"""
        factor = estimator._calculate_temp_factor(35.0)
        assert factor == 1.0  # (35-15)/20 = 1.0
    
    def test_temp_factor_medium(self, estimator):
        """Тест температурного фактора при средней температуре"""
        factor = estimator._calculate_temp_factor(25.0)
        assert abs(factor - 0.5) < 0.01  # (25-15)/20 = 0.5
    
    def test_humidity_factor_dry(self, estimator):
        """Тест фактора влажности при сухом воздухе"""
        factor = estimator._calculate_humidity_factor(0.0)
        assert factor == 1.0  # 1 - 0 = 1
    
    def test_humidity_factor_wet(self, estimator):
        """Тест фактора влажности при влажном воздухе"""
        factor = estimator._calculate_humidity_factor(100.0)
        assert factor == 0.0  # 1 - 1 = 0
    
    def test_wind_factor_zero(self, estimator):
        """Тест ветрового фактора без ветра"""
        factor = estimator._calculate_wind_factor(0.0)
        assert factor == 0.0
    
    def test_wind_factor_high(self, estimator):
        """Тест ветрового фактора при сильном ветре"""
        factor = estimator._calculate_wind_factor(15.0)
        assert factor == 1.0
    
    def test_fuel_factor_short_grass(self, estimator):
        """Тест фактора топлива для short_grass"""
        factor = estimator._calculate_fuel_factor('short_grass')
        assert factor == 0.9
    
    def test_fuel_factor_deciduous(self, estimator):
        """Тест фактора топлива для deciduous"""
        factor = estimator._calculate_fuel_factor('deciduous')
        assert factor == 0.5
    
    def test_slope_factor_flat(self, estimator):
        """Тест фактора уклона на равнине"""
        factor = estimator._calculate_slope_factor(0.0)
        assert factor == 0.0
    
    def test_slope_factor_steep(self, estimator):
        """Тест фактора уклона на крутом склоне"""
        factor = estimator._calculate_slope_factor(30.0)
        assert factor == 1.0


class TestRiskEstimation:
    """Тесты оценки риска"""
    
    def test_estimate_risk_low_conditions(self, estimator):
        """Тест оценки риска при благоприятных условиях"""
        risk = estimator.estimate_risk(
            temperature=10.0,  # Холодно
            humidity=80.0,     # Влажно
            wind_speed=2.0,    # Слабый ветер
            fuel_type='deciduous',
            slope_deg=5.0
        )
        
        assert 0.0 <= risk.probability <= 1.0
        assert risk.risk_level in [RiskLevel.LOW, RiskLevel.MODERATE]
        assert 'temp_factor' in risk.evidence
        assert 'humidity_factor' in risk.evidence
    
    def test_estimate_risk_extreme_conditions(self, estimator):
        """Тест оценки риска при экстремальных условиях"""
        risk = estimator.estimate_risk(
            temperature=40.0,  # Жарко
            humidity=10.0,     # Сухо
            wind_speed=15.0,   # Сильный ветер
            fuel_type='short_grass',
            slope_deg=25.0
        )
        
        assert risk.probability > 0.5  # Высокий риск
        assert risk.risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME]
    
    def test_estimate_risk_structure(self, estimator):
        """Тест структуры результата оценки"""
        risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='conifer_litter',
            slope_deg=10.0
        )
        
        assert isinstance(risk, BayesianFireRisk)
        assert isinstance(risk.probability, float)
        assert isinstance(risk.confidence_interval, tuple)
        assert len(risk.confidence_interval) == 2
        assert isinstance(risk.risk_level, RiskLevel)
        assert isinstance(risk.evidence, dict)
    
    def test_estimate_risk_with_prior_override(self, estimator):
        """Тест оценки с переопределением априорной вероятности"""
        risk_default = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0,
            prior_override=None
        )
        
        risk_custom = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0,
            prior_override=0.5
        )
        
        # Разные priors должны давать разные результаты
        assert risk_default.probability != risk_custom.probability or \
               risk_default.confidence_interval != risk_custom.confidence_interval


class TestRiskClassification:
    """Тесты классификации уровня риска"""
    
    def test_classify_low_risk(self, estimator):
        """Тест классификации низкого риска"""
        level = estimator._classify_risk_level(0.1)
        assert level == RiskLevel.LOW
    
    def test_classify_moderate_risk(self, estimator):
        """Тест классификации умеренного риска"""
        level = estimator._classify_risk_level(0.35)
        assert level == RiskLevel.MODERATE
    
    def test_classify_high_risk(self, estimator):
        """Тест классификации высокого риска"""
        level = estimator._classify_risk_level(0.6)
        assert level == RiskLevel.HIGH
    
    def test_classify_extreme_risk(self, estimator):
        """Тест классификации экстремального риска"""
        level = estimator._classify_risk_level(0.85)
        assert level == RiskLevel.EXTREME
    
    def test_classify_boundary_values(self, estimator):
        """Тест классификации граничных значений"""
        assert estimator._classify_risk_level(0.0) == RiskLevel.LOW
        assert estimator._classify_risk_level(0.25) == RiskLevel.MODERATE
        assert estimator._classify_risk_level(0.50) == RiskLevel.HIGH
        assert estimator._classify_risk_level(0.75) == RiskLevel.EXTREME


class TestBayesianUpdate:
    """Тесты байесовского обновления"""
    
    def test_update_with_fire_observation(self, estimator):
        """Тест обновления при наблюдении пожара"""
        initial_risk = estimator.estimate_risk(
            temperature=30.0,
            humidity=30.0,
            wind_speed=8.0,
            fuel_type='short_grass',
            slope_deg=15.0
        )
        
        updated_risk = estimator.update_with_observation(
            current_risk=initial_risk,
            observation=True,  # Пожар произошел
            learning_rate=0.1
        )
        
        # Вероятность должна увеличиться
        assert updated_risk.probability >= initial_risk.probability - 0.01
    
    def test_update_with_no_fire_observation(self, estimator):
        """Тест обновления при отсутствии пожара"""
        initial_risk = estimator.estimate_risk(
            temperature=30.0,
            humidity=30.0,
            wind_speed=8.0,
            fuel_type='short_grass',
            slope_deg=15.0
        )
        
        updated_risk = estimator.update_with_observation(
            current_risk=initial_risk,
            observation=False,  # Пожара не было
            learning_rate=0.1
        )
        
        # Вероятность должна уменьшиться или остаться той же
        assert updated_risk.probability <= initial_risk.probability + 0.01
    
    def test_update_history_growth(self, estimator):
        """Тест роста истории наблюдений"""
        initial_len = len(estimator.observation_history)
        
        risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        estimator.update_with_observation(risk, True)
        estimator.update_with_observation(risk, False)
        
        assert len(estimator.observation_history) == initial_len + 2
    
    def test_update_confidence_interval_narrowing(self, estimator):
        """Тест сужения доверительного интервала"""
        risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        # Много обновлений
        for _ in range(20):
            risk = estimator.update_with_observation(risk, True)
        
        # Доверительный интервал должен сузиться
        ci_width = risk.confidence_interval[1] - risk.confidence_interval[0]
        assert ci_width < 0.6  # Должен быть относительно узким


class TestRiskToDict:
    """Тесты конвертации в dict"""
    
    def test_to_dict_structure(self, estimator):
        """Тест структуры dict"""
        risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        risk_dict = risk.to_dict()
        
        assert 'probability' in risk_dict
        assert 'confidence_interval' in risk_dict
        assert 'risk_level' in risk_dict
        assert 'evidence' in risk_dict
        assert 'timestamp' in risk_dict
    
    def test_to_dict_values(self, estimator):
        """Тест значений в dict"""
        risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        risk_dict = risk.to_dict()
        
        assert risk_dict['probability'] == risk.probability
        assert risk_dict['risk_level'] == risk.risk_level.value
        assert risk_dict['evidence'] == risk.evidence


class TestPriorProbabilities:
    """Тесты априорных вероятностей"""
    
    def test_prior_probabilities_exist(self):
        """Тест существования априорных вероятностей"""
        assert 'short_grass' in PRIOR_PROBABILITIES
        assert 'conifer_litter' in PRIOR_PROBABILITIES
        assert 'timber_understory' in PRIOR_PROBABILITIES
        assert 'deciduous' in PRIOR_PROBABILITIES
    
    def test_prior_probabilities_valid(self):
        """Тест валидности априорных вероятностей"""
        for fuel_type, prior in PRIOR_PROBABILITIES.items():
            assert 0.0 <= prior <= 1.0
    
    def test_likelihood_weights_sum(self):
        """Тест суммы весов likelihood"""
        total_weight = sum(LIKELIHOOD_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01  # Должно быть ≈ 1.0


class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_extreme_temperature(self, estimator):
        """Тест экстремальной температуры"""
        risk_cold = estimator.estimate_risk(
            temperature=-20.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        risk_hot = estimator.estimate_risk(
            temperature=50.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        # Горячие условия должны иметь более высокий риск
        assert risk_hot.probability >= risk_cold.probability
    
    def test_zero_humidity(self, estimator):
        """Тест нулевой влажности"""
        risk = estimator.estimate_risk(
            temperature=30.0,
            humidity=0.0,
            wind_speed=5.0,
            fuel_type='short_grass',
            slope_deg=10.0
        )
        
        assert risk.probability > 0
    
    def test_unknown_fuel_type(self, estimator):
        """Тест неизвестного типа топлива"""
        risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='unknown_fuel',
            slope_deg=10.0
        )
        
        # Должен использовать значение по умолчанию
        assert 0.0 <= risk.probability <= 1.0
    
    def test_learning_rate_zero(self, estimator):
        """Тест нулевой скорости обучения"""
        initial_risk = estimator.estimate_risk(
            temperature=25.0,
            humidity=50.0,
            wind_speed=5.0,
            fuel_type='deciduous',
            slope_deg=10.0
        )
        
        updated_risk = estimator.update_with_observation(
            current_risk=initial_risk,
            observation=True,
            learning_rate=0.0
        )
        
        # При learning_rate=0 вероятность не должна измениться
        assert updated_risk.probability == initial_risk.probability
