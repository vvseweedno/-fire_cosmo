"""
Интеграционные тесты для математических моделей
"""

import pytest
import numpy as np
from app.services.fire_physics import FirePhysicsService, FuelType
from app.services.fire_cellular_automata import FireCellularAutomata
from app.services.bayesian_fire_risk import BayesianFireRiskEstimator
from app.services.multispectral_indices import MultispectralIndices


class TestPhysicsIntegration:
    """Интеграционные тесты модели Ротермеля"""
    
    def test_full_scenario_wind_impact(self):
        """Полный сценарий: влияние ветра на распространение"""
        physics = FirePhysicsService()
        
        # Без ветра
        result_no_wind = physics.simulate_fire_scenario(
            center_lat=55.0, center_lon=82.0, hours=24,
            weather={"wind_speed": 0.0, "wind_direction": 0.0, "temp": 25.0, "humidity": 40.0},
            terrain={"slope": 10.0, "aspect": "S"}
        )
        
        # С сильным ветром
        result_strong_wind = physics.simulate_fire_scenario(
            center_lat=55.0, center_lon=82.0, hours=24,
            weather={"wind_speed": 15.0, "wind_direction": 90.0, "temp": 25.0, "humidity": 40.0},
            terrain={"slope": 10.0, "aspect": "S"}
        )
        
        # Сильный ветер должен давать большее распространение
        assert result_strong_wind["spread_distance_m"] > result_no_wind["spread_distance_m"]
    
    def test_fuel_type_comparison(self):
        """Сравнение разных типов топлива"""
        physics = FirePhysicsService()
        
        distances = {}
        for fuel_type in [FuelType.SHORT_GRASS, FuelType.TIMBER_UNDERSTORY, FuelType.CONIFER_LITTER]:
            result = physics.predict_spread_polygon(
                center_lat=55.0, center_lon=82.0, hours=12,
                wind_speed_ms=10.0, wind_direction_deg=90.0,
                fuel_type=fuel_type
            )
            distances[fuel_type.value] = result.spread_distance_m
        
        # Все расстояния должны быть положительными
        assert all(d > 0 for d in distances.values())
        
        # Разные типы топлива дают разные скорости
        assert len(set(distances.values())) > 1


class TestCellularAutomataIntegration:
    """Интеграционные тесты клеточных автоматов"""
    
    def test_wind_direction_impact(self):
        """Влияние направления ветра на форму пожара"""
        # Проверяем что ветер влияет на направление распространения
        # Ветер на восток (90°)
        ca_east = FireCellularAutomata(grid_shape=(100, 100))
        ca_east.set_ignition_point(50, 50)
        ca_east.set_wind(15.0, 90.0)
        grid_east = ca_east.simulate(hours=12)
        
        # Без ветра (контрольный тест)
        ca_calm = FireCellularAutomata(grid_shape=(100, 100))
        ca_calm.set_ignition_point(50, 50)
        ca_calm.set_wind(0.0, 0.0)
        grid_calm = ca_calm.simulate(hours=12)
        
        # Считаем количество сгоревших клеток
        burned_east = int(np.sum(grid_east == 2))
        burned_calm = int(np.sum(grid_calm == 2))
        
        # С ветром пожар должен распространяться больше
        assert burned_east > burned_calm, f"С ветром должно сгореть больше клеток: {burned_east} vs {burned_calm}"
    
    def test_obstacle_blocking(self):
        """Препятствия блокируют распространение"""
        ca = FireCellularAutomata(grid_shape=(30, 30))
        ca.set_ignition_point(15, 15)
        
        # Добавляем водную преграду
        ca.grid[15:20, 10:20] = 3  # CellState.WATER
        
        grid = ca.simulate(hours=6)
        
        # Вода не должна гореть
        water_cells = np.sum(grid[15:20, 10:20] == 2)
        assert water_cells == 0


class TestBayesianIntegration:
    """Интеграционные тесты байесовской оценки"""
    
    def test_learning_from_observations(self):
        """Обучение на последовательных наблюдениях"""
        estimator = BayesianFireRiskEstimator()
        
        # Начальная оценка
        risk = estimator.estimate_risk(
            temperature=30.0, humidity=30.0, wind_speed=10.0,
            fuel_type="short_grass", slope_deg=15.0
        )
        initial_prob = risk.probability
        
        # Наблюдаем пожар 5 раз
        for _ in range(5):
            risk = estimator.update_with_observation(
                current_risk=risk,
                observation=True,
                learning_rate=0.2
            )
        
        # Вероятность должна вырасти
        assert risk.probability > initial_prob
    
    def test_risk_map_generation(self):
        """Генерация карты риска для сетки"""
        estimator = BayesianFireRiskEstimator()
        
        grid_fuel = np.random.randint(0, 4, size=(10, 10))
        grid_slope = np.random.uniform(0, 30, size=(10, 10))
        
        risk_map = estimator.get_risk_map(
            grid_weather={"temp": 25.0, "humidity": 40.0, "wind_speed": 8.0},
            grid_fuel=grid_fuel,
            grid_slope=grid_slope
        )
        
        assert risk_map.shape == (10, 10)
        assert np.all((risk_map >= 0) & (risk_map <= 1))


class TestMultispectralIntegration:
    """Интеграционные тесты мультиспектральных индексов"""
    
    def test_nbr_and_dnbr_pipeline(self):
        """Полный пайплайн: NBR → dNBR → classification"""
        # Синтетические данные
        b08_pre = np.random.uniform(0.3, 0.6, size=(100, 100))
        b12_pre = np.random.uniform(0.1, 0.3, size=(100, 100))
        
        b08_post = b08_pre * 0.7  # После пожара NIR падает
        b12_post = b12_pre * 1.3  # SWIR растёт
        
        # Расчёт NBR
        nbr_pre = MultispectralIndices.calculate_nbr(b08_pre, b12_pre)
        nbr_post = MultispectralIndices.calculate_nbr(b08_post, b12_post)
        
        # Расчёт dNBR
        dnbr = MultispectralIndices.calculate_dnbr(nbr_pre, nbr_post)
        
        # dNBR должен быть положительным (выгорание)
        assert np.mean(dnbr) > 0
        
        # RdNBR
        rdnbr = MultispectralIndices.calculate_rdnbr(dnbr, nbr_pre)
        assert rdnbr.shape == dnbr.shape
    
    def test_fire_risk_index_calculation(self):
        """Расчёт комплексного индекса риска"""
        ndvi = np.random.uniform(0.2, 0.8, size=(50, 50))
        swir = np.random.uniform(2000, 8000, size=(50, 50))
        land_cover = np.random.randint(1, 6, size=(50, 50))
        burned_history = np.random.uniform(0, 1, size=(50, 50))
        
        risk_index = MultispectralIndices.calculate_fire_risk_index(
            ndvi=ndvi,
            swir=swir,
            land_cover=land_cover,
            burned_history=burned_history
        )
        
        assert risk_index.shape == (50, 50)
        assert np.all((risk_index >= 0) & (risk_index <= 100))
