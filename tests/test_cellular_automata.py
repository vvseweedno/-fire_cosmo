"""
Tests for Fire Cellular Automata model
"""

import pytest
import numpy as np
from app.services.fire_cellular_automata import (
    FireCellularAutomata,
    FireCellularAutomataConfig,
    CellState,
)


@pytest.fixture
def ca_small():
    """Создать модель клеточных автоматов малого размера для тестов"""
    return FireCellularAutomata(grid_shape=(20, 20))


@pytest.fixture
def ca_config():
    """Создать кастомную конфигурацию"""
    return FireCellularAutomataConfig(
        cell_size_m=100.0,
        dt_minutes=15.0,
        p_base=0.3,
        p_wind_max=0.4,
        p_fuel_max=0.3,
    )


class TestInitialization:
    """Тесты инициализации модели"""
    
    def test_grid_initial_state(self, ca_small):
        """Тест начального состояния сетки"""
        assert ca_small.grid.shape == (20, 20)
        assert np.all(ca_small.grid == CellState.NOT_BURNING.value)
    
    def test_fuel_grid_default(self, ca_small):
        """Тест сетки топлива по умолчанию"""
        assert ca_small.fuel_grid.shape == (20, 20)
        assert np.all(ca_small.fuel_grid == 0)  # short_grass по умолчанию
    
    def test_custom_config(self, ca_config):
        """Тест кастомной конфигурации"""
        ca = FireCellularAutomata(grid_shape=(10, 10), config=ca_config)
        assert ca.config.p_base == 0.3
        assert ca.config.cell_size_m == 100.0


class TestIgnition:
    """Тесты поджигания клеток"""
    
    def test_ignite_single_cell(self, ca_small):
        """Тест поджигания одной клетки"""
        ca_small.ignite_cell(10, 10)
        assert ca_small.grid[10, 10] == CellState.BURNING.value
    
    def test_ignite_cell_out_of_bounds(self, ca_small):
        """Тест поджигания вне границ"""
        ca_small.ignite_cell(-1, 0)
        ca_small.ignite_cell(100, 0)
        assert np.all(ca_small.grid == CellState.NOT_BURNING.value)
    
    def test_ignite_region(self, ca_small):
        """Тест поджигания области"""
        ca_small.ignite_region(5, 5, 10, 10)
        
        region = ca_small.grid[5:10, 5:10]
        assert np.all(region == CellState.BURNING.value)
    
    def test_ignite_on_obstacle(self, ca_small):
        """Тест что препятствия не загораются"""
        ca_small.set_obstacle(5, 5, 10, 10, CellState.WATER)
        ca_small.ignite_region(5, 5, 10, 10)
        
        region = ca_small.grid[5:10, 5:10]
        assert np.all(region == CellState.WATER.value)


class TestObstacles:
    """Тесты препятствий"""
    
    def test_set_water_obstacle(self, ca_small):
        """Тест установки водного препятствия"""
        ca_small.set_obstacle(0, 0, 5, 5, CellState.WATER)
        assert np.all(ca_small.grid[0:5, 0:5] == CellState.WATER.value)
    
    def test_set_road_obstacle(self, ca_small):
        """Тест установки дорожного препятствия"""
        ca_small.set_obstacle(10, 10, 15, 15, CellState.ROAD)
        assert np.all(ca_small.grid[10:15, 10:15] == CellState.ROAD.value)
    
    def test_obstacle_bounds_clipping(self, ca_small):
        """Тест обрезки границ препятствия"""
        ca_small.set_obstacle(-5, -5, 100, 100, CellState.WATER)
        assert np.all(ca_small.grid == CellState.WATER.value)


class TestWind:
    """Тесты влияния ветра"""
    
    def test_set_wind(self, ca_small):
        """Тест установки параметров ветра"""
        ca_small.set_wind(10.0, 90.0)
        assert ca_small.wind_speed == 10.0
    
    def test_wind_direction_uniform(self, ca_small):
        """Тест равномерности направления ветра"""
        ca_small.set_wind(5.0, 45.0)
        assert np.all(ca_small.wind_direction == ca_small.wind_direction[0, 0])


class TestFuelTypes:
    """Тесты типов топлива"""
    
    def test_set_fuel_type(self, ca_small):
        """Тест установки типа топлива"""
        ca_small.set_fuel_type(0, 0, 10, 10, 2)
        assert np.all(ca_small.fuel_grid[0:10, 0:10] == 2)
    
    def test_fuel_factor_lookup(self, ca_small):
        """Тест поиска коэффициента топлива"""
        factors = {
            0: 1.0,   # short_grass
            1: 0.9,   # conifer_litter
            2: 0.8,   # timber_understory
            3: 0.5,   # deciduous
            4: 0.2,   # bare
            5: 0.0,   # water
        }
        
        for code, expected in factors.items():
            actual = ca_small._get_fuel_factor(code)
            assert abs(actual - expected) < 0.01


class TestSimulation:
    """Тесты симуляции распространения"""
    
    def test_simulation_basic(self, ca_small):
        """Базовый тест симуляции"""
        ca_small.ignite_cell(10, 10)
        ca_small.set_wind(5.0, 90.0)
        
        final_grid = ca_small.simulate(hours=1.0, dt_minutes=15.0)
        
        # Должны быть сгоревшие клетки
        burned = np.isin(final_grid, [CellState.BURNING.value, CellState.BURNED.value])
        assert np.sum(burned) > 1  # Больше чем исходная клетка
    
    def test_simulation_extinguish(self, ca_small):
        """Тест затухания пожара без топлива"""
        # Устанавливаем низкую вероятность возгорания
        config = FireCellularAutomataConfig(p_base=0.01, p_wind_max=0.01, p_fuel_max=0.01)
        ca = FireCellularAutomata(grid_shape=(10, 10), config=config)
        ca.ignite_cell(5, 5)
        
        np.random.seed(42)  # Для воспроизводимости
        final_grid = ca.simulate(hours=0.5, dt_minutes=15.0)
        
        # Пожар должен потухнуть из-за низкой вероятности
        burning = final_grid == CellState.BURNING.value
        assert np.sum(burning) == 0
    
    def test_simulation_time_scaling(self, ca_small):
        """Тест масштабирования времени симуляции"""
        ca_small.ignite_cell(10, 10)
        ca_small.set_wind(5.0, 90.0)
        
        np.random.seed(42)
        grid_1h = ca_small.get_grid_copy()
        ca_small.simulate(hours=1.0, dt_minutes=15.0)
        area_1h = ca_small.get_burned_area_ha()
        
        ca_small.reset()
        ca_small.ignite_cell(10, 10)
        ca_small.set_wind(5.0, 90.0)
        np.random.seed(42)
        ca_small.simulate(hours=2.0, dt_minutes=15.0)
        area_2h = ca_small.get_burned_area_ha()
        
        # Площадь должна расти со временем
        assert area_2h >= area_1h
    
    def test_simulation_no_initial_fire(self, ca_small):
        """Тест симуляции без начального огня"""
        final_grid = ca_small.simulate(hours=1.0)
        assert np.all(final_grid == CellState.NOT_BURNING.value)


class TestBurnedArea:
    """Тесты расчета площади"""
    
    def test_burned_area_calculation(self, ca_small):
        """Тест расчета площади пожара"""
        ca_small.ignite_region(0, 0, 10, 10)
        
        # 10x10 клеток по 100м = 1000м x 1000м = 1 км² = 100 га
        area = ca_small.get_burned_area_ha()
        expected_area = 100.0  # га
        
        assert abs(area - expected_area) < 0.01
    
    def test_burned_area_empty(self, ca_small):
        """Тест площади при отсутствии пожара"""
        area = ca_small.get_burned_area_ha()
        assert area == 0.0
    
    def test_burned_area_partial(self, ca_small):
        """Тест частичной площади"""
        ca_small.ignite_cell(5, 5)
        ca_small.ignite_cell(6, 6)
        
        # 2 клетки по 100м = 2 × 10000 м² = 20000 м² = 2 га
        area = ca_small.get_burned_area_ha()
        expected_area = 2.0  # га
        
        assert abs(area - expected_area) < 0.01


class TestStatistics:
    """Тесты статистики пожара"""
    
    def test_fire_statistics_structure(self, ca_small):
        """Тест структуры статистики"""
        stats = ca_small.get_fire_statistics()
        
        assert 'total_cells' in stats
        assert 'not_burning' in stats
        assert 'burning' in stats
        assert 'burned' in stats
        assert 'burned_area_ha' in stats
        
        assert stats['total_cells'] == 20 * 20
    
    def test_fire_statistics_after_ignition(self, ca_small):
        """Тест статистики после поджигания"""
        ca_small.ignite_region(0, 0, 5, 5)
        
        stats = ca_small.get_fire_statistics()
        assert stats['burning'] == 25
        assert stats['not_burning'] == 20*20 - 25


class TestReset:
    """Тесты сброса состояния"""
    
    def test_reset_clears_fire(self, ca_small):
        """Тест что сброс гасит огонь"""
        ca_small.ignite_region(0, 0, 10, 10)
        ca_small.reset()
        
        assert np.all(ca_small.grid == CellState.NOT_BURNING.value)


class TestGeoJSON:
    """Тесты конвертации в GeoJSON"""
    
    def test_geojson_structure(self, ca_small):
        """Тест структуры GeoJSON"""
        ca_small.ignite_cell(10, 10)
        
        geojson = ca_small.to_geojson(origin_lat=55.0, origin_lon=82.0)
        
        assert geojson['type'] == 'FeatureCollection'
        assert 'features' in geojson
        assert len(geojson['features']) > 0
    
    def test_geojson_properties(self, ca_small):
        """Тест свойств GeoJSON"""
        ca_small.ignite_cell(5, 5)
        
        geojson = ca_small.to_geojson()
        
        if len(geojson['features']) > 0:
            feature = geojson['features'][0]
            assert 'geometry' in feature
            assert 'properties' in feature
            assert feature['geometry']['type'] == 'Polygon'
            assert 'state' in feature['properties']


class TestEdgeCases:
    """Тесты граничных случаев"""
    
    def test_small_grid(self):
        """Тест очень маленькой сетки"""
        ca = FireCellularAutomata(grid_shape=(2, 2))
        ca.ignite_cell(0, 0)
        ca.simulate(hours=0.25)
        
        assert ca.grid.shape == (2, 2)
    
    def test_zero_wind_speed(self, ca_small):
        """Тест нулевой скорости ветра"""
        ca_small.set_wind(0.0, 0.0)
        ca_small.ignite_cell(10, 10)
        
        # Симуляция должна работать без ошибок
        ca_small.simulate(hours=0.5)
        assert True
    
    def test_large_simulation(self):
        """Тест большой симуляции"""
        ca = FireCellularAutomata(grid_shape=(50, 50))
        ca.ignite_cell(25, 25)
        ca.set_wind(10.0, 45.0)
        
        final_grid = ca.simulate(hours=4.0, dt_minutes=15.0)
        
        assert final_grid.shape == (50, 50)
        burned = np.isin(final_grid, [CellState.BURNING.value, CellState.BURNED.value])
        assert np.sum(burned) > 0
