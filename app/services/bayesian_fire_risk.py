"""
Байесовская оценка риска лесных пожаров

P(fire | weather, fuel, slope, history) = P(evidence | fire) × P(fire) / P(evidence)

Факторы likelihood:
- temp_factor = clip((temp - 15) / 20, 0, 1)
- humidity_factor = 1 - clip(humidity / 100, 0, 1)
- wind_factor = clip(wind_speed / 15, 0, 1)
- fuel_factor = dict {'short_grass':0.9,'conifer_litter':0.8,'timber_understory':0.7,'deciduous':0.5}
- slope_factor = clip(slope_deg / 30, 0, 1)
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Уровни риска пожара"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class BayesianFireRisk:
    """Результат байесовской оценки риска"""
    probability: float  # Вероятность пожара [0, 1]
    confidence_interval: Tuple[float, float]  # (ci_low, ci_high)
    risk_level: RiskLevel  # Уровень риска
    evidence: Dict[str, float]  # Факторы влияния
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертировать в dict"""
        return {
            "probability": self.probability,
            "confidence_interval": self.confidence_interval,
            "risk_level": self.risk_level.value,
            "evidence": self.evidence,
            "timestamp": datetime.utcnow().isoformat()
        }


# Базовые априорные вероятности по типам растительности
PRIOR_PROBABILITIES: Dict[str, float] = {
    'short_grass': 0.15,
    'conifer_litter': 0.20,
    'timber_understory': 0.12,
    'deciduous': 0.08,
    'bare': 0.05,
    'default': 0.10,
}

# Коэффициенты влияния факторов на likelihood
LIKELIHOOD_WEIGHTS = {
    'temp': 0.25,
    'humidity': 0.25,
    'wind': 0.20,
    'fuel': 0.20,
    'slope': 0.10,
}

# Пороги классификации риска
RISK_THRESHOLDS = {
    RiskLevel.LOW: (0.0, 0.25),
    RiskLevel.MODERATE: (0.25, 0.50),
    RiskLevel.HIGH: (0.50, 0.75),
    RiskLevel.EXTREME: (0.75, 1.0),
}


def clip(value: float, min_val: float, max_val: float) -> float:
    """Ограничить значение диапазоном"""
    return max(min_val, min(max_val, value))


class BayesianFireRiskEstimator:
    """
    Байесовский оценщик риска лесных пожаров
    
    Использует теорему Байеса для обновления вероятности пожара
    на основе наблюдений (погода, топливо, рельеф, история)
    
    P(fire | evidence) = P(evidence | fire) × P(fire) / P(evidence)
    """
    
    def __init__(self, default_prior: float = 0.10):
        """
        Инициализировать оценщик
        
        Args:
            default_prior: Априорная вероятность по умолчанию
        """
        self.default_prior = default_prior
        self.observation_history: List[Dict[str, Any]] = []
        
        logger.info(f"BayesianFireRiskEstimator initialized with prior={default_prior}")
    
    def _calculate_temp_factor(self, temp_celsius: float) -> float:
        """
        Рассчитать температурный фактор
        
        temp_factor = clip((temp - 15) / 20, 0, 1)
        
        Args:
            temp_celsius: Температура воздуха (°C)
            
        Returns:
            Фактор [0, 1]
        """
        factor = (temp_celsius - 15.0) / 20.0
        return clip(factor, 0.0, 1.0)
    
    def _calculate_humidity_factor(self, humidity_percent: float) -> float:
        """
        Рассчитать фактор влажности
        
        humidity_factor = 1 - clip(humidity / 100, 0, 1)
        
        Args:
            humidity_percent: Относительная влажность (%)
            
        Returns:
            Фактор [0, 1]
        """
        clipped_humidity = clip(humidity_percent / 100.0, 0.0, 1.0)
        return 1.0 - clipped_humidity
    
    def _calculate_wind_factor(self, wind_speed_ms: float) -> float:
        """
        Рассчитать ветровой фактор
        
        wind_factor = clip(wind_speed / 15, 0, 1)
        
        Args:
            wind_speed_ms: Скорость ветра (м/с)
            
        Returns:
            Фактор [0, 1]
        """
        return clip(wind_speed_ms / 15.0, 0.0, 1.0)
    
    def _calculate_fuel_factor(self, fuel_type: str) -> float:
        """
        Рассчитать фактор топлива
        
        fuel_factor из словаря:
        {'short_grass':0.9,'conifer_litter':0.8,'timber_understory':0.7,'deciduous':0.5}
        
        Args:
            fuel_type: Тип топлива
            
        Returns:
            Фактор [0, 1]
        """
        fuel_factors = {
            'short_grass': 0.9,
            'conifer_litter': 0.8,
            'timber_understory': 0.7,
            'deciduous': 0.5,
            'bare': 0.3,
        }
        
        return fuel_factors.get(fuel_type.lower(), 0.5)
    
    def _calculate_slope_factor(self, slope_deg: float) -> float:
        """
        Рассчитать фактор уклона
        
        slope_factor = clip(slope_deg / 30, 0, 1)
        
        Args:
            slope_deg: Уклон склона (градусы)
            
        Returns:
            Фактор [0, 1]
        """
        return clip(slope_deg / 30.0, 0.0, 1.0)
    
    def _calculate_likelihood(
        self,
        temp_factor: float,
        humidity_factor: float,
        wind_factor: float,
        fuel_factor: float,
        slope_factor: float
    ) -> float:
        """
        Рассчитать likelihood P(evidence | fire)
        
        Взвешенная сумма факторов
        
        Args:
            temp_factor: Температурный фактор
            humidity_factor: Фактор влажности
            wind_factor: Ветровой фактор
            fuel_factor: Фактор топлива
            slope_factor: Фактор уклона
            
        Returns:
            Likelihood [0, 1]
        """
        likelihood = (
            LIKELIHOOD_WEIGHTS['temp'] * temp_factor +
            LIKELIHOOD_WEIGHTS['humidity'] * humidity_factor +
            LIKELIHOOD_WEIGHTS['wind'] * wind_factor +
            LIKELIHOOD_WEIGHTS['fuel'] * fuel_factor +
            LIKELIHOOD_WEIGHTS['slope'] * slope_factor
        )
        
        return clip(likelihood, 0.0, 1.0)
    
    def _calculate_evidence_probability(
        self,
        likelihood: float,
        prior: float
    ) -> float:
        """
        Рассчитать P(evidence) - полную вероятность свидетельств
        
        P(evidence) = P(evidence | fire) × P(fire) + P(evidence | no_fire) × P(no_fire)
        
        Args:
            likelihood: P(evidence | fire)
            prior: P(fire)
            
        Returns:
            P(evidence)
        """
        # P(evidence | no_fire) ≈ 1 - likelihood (упрощение)
        likelihood_no_fire = 1.0 - likelihood
        
        # P(no_fire) = 1 - P(fire)
        prior_no_fire = 1.0 - prior
        
        # Полная вероятность
        p_evidence = likelihood * prior + likelihood_no_fire * prior_no_fire
        
        return max(p_evidence, 1e-10)  # Избегаем деления на ноль
    
    def _classify_risk_level(self, probability: float) -> RiskLevel:
        """
        Классифицировать уровень риска по вероятности
        
        Args:
            probability: Вероятность пожара
            
        Returns:
            RiskLevel
        """
        for level, (low, high) in RISK_THRESHOLDS.items():
            if low <= probability < high:
                return level
        
        return RiskLevel.EXTREME
    
    def _calculate_confidence_interval(
        self,
        probability: float,
        num_observations: int = 1
    ) -> Tuple[float, float]:
        """
        Рассчитать доверительный интервал
        
        Используем упрощенный подход с биномиальным распределением
        
        Args:
            probability: Оценка вероятности
            num_observations: Количество наблюдений
            
        Returns:
            (ci_low, ci_high)
        """
        # Стандартная ошибка
        n = max(num_observations, 1)
        se = math.sqrt(probability * (1.0 - probability) / n)
        
        # 95% доверительный интервал (z = 1.96)
        z = 1.96
        margin = z * se
        
        ci_low = clip(probability - margin, 0.0, 1.0)
        ci_high = clip(probability + margin, 0.0, 1.0)
        
        return (ci_low, ci_high)
    
    def estimate_risk(
        self,
        temperature: float,
        humidity: float,
        wind_speed: float,
        fuel_type: str,
        slope_deg: float,
        prior_override: Optional[float] = None
    ) -> BayesianFireRisk:
        """
        Оценить риск пожара по байесовской формуле
        
        P(fire | evidence) = P(evidence | fire) × P(fire) / P(evidence)
        
        Args:
            temperature: Температура воздуха (°C)
            humidity: Относительная влажность (%)
            wind_speed: Скорость ветра (м/с)
            fuel_type: Тип топлива
            slope_deg: Уклон склона (градусы)
            prior_override: Переопределение априорной вероятности
            
        Returns:
            BayesianFireRisk с оценкой риска
        """
        # Расчет факторов
        temp_factor = self._calculate_temp_factor(temperature)
        humidity_factor = self._calculate_humidity_factor(humidity)
        wind_factor = self._calculate_wind_factor(wind_speed)
        fuel_factor = self._calculate_fuel_factor(fuel_type)
        slope_factor = self._calculate_slope_factor(slope_deg)
        
        # Evidence dict для возврата
        evidence = {
            'temp_factor': temp_factor,
            'humidity_factor': humidity_factor,
            'wind_factor': wind_factor,
            'fuel_factor': fuel_factor,
            'slope_factor': slope_factor,
        }
        
        # Likelihood
        likelihood = self._calculate_likelihood(
            temp_factor, humidity_factor, wind_factor,
            fuel_factor, slope_factor
        )
        
        # Prior P(fire)
        if prior_override is not None:
            prior = clip(prior_override, 0.0, 1.0)
        else:
            prior = PRIOR_PROBABILITIES.get(fuel_type.lower(), self.default_prior)
        
        # Evidence probability P(evidence)
        p_evidence = self._calculate_evidence_probability(likelihood, prior)
        
        # Posterior P(fire | evidence)
        posterior = (likelihood * prior) / p_evidence
        posterior = clip(posterior, 0.0, 1.0)
        
        # Доверительный интервал
        ci = self._calculate_confidence_interval(posterior, len(self.observation_history) + 1)
        
        # Уровень риска
        risk_level = self._classify_risk_level(posterior)
        
        risk = BayesianFireRisk(
            probability=posterior,
            confidence_interval=ci,
            risk_level=risk_level,
            evidence=evidence
        )
        
        logger.info(
            f"Risk estimated: P={posterior:.3f}, level={risk_level.value}, "
            f"fuel={fuel_type}, temp={temperature}°C"
        )
        
        return risk
    
    def update_with_observation(
        self,
        current_risk: BayesianFireRisk,
        observation: bool,
        learning_rate: float = 0.1
    ) -> BayesianFireRisk:
        """
        Обновить оценку риска по новому наблюдению (байесовское обновление)
        
        Args:
            current_risk: Текущая оценка риска
            observation: Было ли наблюдение пожара (True/False)
            learning_rate: Скорость обучения (0-1)
            
        Returns:
            Обновленная BayesianFireRisk
        """
        # Сохраняем наблюдение в историю
        self.observation_history.append({
            'fire_observed': observation,
            'predicted_probability': current_risk.probability,
            'timestamp': datetime.utcnow(),
        })
        
        # Байесовское обновление априорной вероятности
        # Если пожар произошел, увеличиваем prior; если нет, уменьшаем
        adjustment = learning_rate * (1.0 if observation else -1.0)
        
        # Обновляем веса факторов на основе ошибки предсказания
        error = (1.0 if observation else 0.0) - current_risk.probability
        
        # Создаем обновленную оценку
        new_probability = clip(
            current_risk.probability + learning_rate * error,
            0.0, 1.0
        )
        
        # Пересчитываем доверительный интервал с учетом нового наблюдения
        num_obs = len(self.observation_history)
        new_ci = self._calculate_confidence_interval(new_probability, num_obs)
        
        # Новый уровень риска
        new_risk_level = self._classify_risk_level(new_probability)
        
        updated_risk = BayesianFireRisk(
            probability=new_probability,
            confidence_interval=new_ci,
            risk_level=new_risk_level,
            evidence=current_risk.evidence.copy()
        )
        
        logger.info(
            f"Risk updated with observation: observed={observation}, "
            f"new_P={new_probability:.3f}, level={new_risk_level.value}"
        )
        
        return updated_risk
    
    def get_risk_map(
        self,
        grid_weather: Dict[str, Any],
        grid_fuel: np.ndarray,
        grid_slope: np.ndarray,
        prior_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Создать карту риска для сетки
        
        Args:
            grid_weather: Погодные данные для сетки {temp, humidity, wind_speed}
            grid_fuel: Сетка типов топлива
            grid_slope: Сетка уклонов
            prior_map: Карта априорных вероятностей (опционально)
            
        Returns:
            Сетка вероятностей риска
        """
        try:
            import numpy as np
        except ImportError:
            logger.error("numpy required for grid operations")
            raise
        
        shape = grid_fuel.shape
        risk_map = np.zeros(shape, dtype=np.float32)
        
        temp = grid_weather.get('temp', 20.0)
        humidity = grid_weather.get('humidity', 50.0)
        wind_speed = grid_weather.get('wind_speed', 5.0)
        
        fuel_types = ['short_grass', 'conifer_litter', 'timber_understory', 'deciduous']
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                fuel_code = int(grid_fuel[i, j])
                fuel_type = fuel_types[fuel_code] if fuel_code < len(fuel_types) else 'default'
                
                slope = float(grid_slope[i, j]) if i < grid_slope.shape[0] and j < grid_slope.shape[1] else 0.0
                
                prior = float(prior_map[i, j]) if prior_map is not None else None
                
                risk = self.estimate_risk(
                    temperature=temp,
                    humidity=humidity,
                    wind_speed=wind_speed,
                    fuel_type=fuel_type,
                    slope_deg=slope,
                    prior_override=prior
                )
                
                risk_map[i, j] = risk.probability
        
        return risk_map
