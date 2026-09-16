"""
Tests for Fire Physics Service (Rothermel model)
"""

import pytest
import math
from app.services.fire_physics import (
    FirePhysicsService,
    FuelType,
    FUEL_MODELS,
    ASPECT_MULTIPLIERS,
    EPSILON,
    QIG,
    BETA,
)


@pytest.fixture
def physics_service():
    """Создать сервис физики пожара с топливом по умолчанию"""
    return FirePhysicsService(default_fuel_type=FuelType.SHORT_GRASS)


class TestBaseRate:
    """Тесты расчета базовой скорости распространения R0"""
    
    def test_base_rate_short_grass(self, physics_service):
        """Тест базовой скорости для short_grass"""
        r0 = physics_service.calculate_base_rate(FuelType.SHORT_GRASS)
        
        # R0 = (IR × ξ) / (ρb × ε × Qig)
        # R0 = (1500 × 0.4) / (0.2 × 0.716 × 514)
        # R0 = 600 / 73.6048 ≈ 8.15 м/с
        expected_numerator = 1500.0 * 0.4
        expected_denominator = 0.2 * EPSILON * QIG
        expected_r0 = expected_numerator / expected_denominator
        
        assert r0 > 0
        assert abs(r0 - expected_r0) < 0.001
        assert isinstance(r0, float)
    
    def test_base_rate_timber_understory(self, physics_service):
        """Тест базовой скорости для timber_understory"""
        r0 = physics_service.calculate_base_rate(FuelType.TIMBER_UNDERSTORY)
        
        # R0 = (2500 × 0.3) / (0.4 × 0.716 × 514)
        expected_numerator = 2500.0 * 0.3
        expected_denominator = 0.4 * EPSILON * QIG
        expected_r0 = expected_numerator / expected_denominator
        
        assert r0 > 0
        assert abs(r0 - expected_r0) < 0.001
    
    def test_base_rate_conifer_litter(self, physics_service):
        """Тест базовой скорости для conifer_litter"""
        r0 = physics_service.calculate_base_rate(FuelType.CONIFER_LITTER)
        
        # R0 = (3000 × 0.35) / (0.3 × 0.716 × 514)
        expected_numerator = 3000.0 * 0.35
        expected_denominator = 0.3 * EPSILON * QIG
        expected_r0 = expected_numerator / expected_denominator
        
        assert r0 > 0
        assert abs(r0 - expected_r0) < 0.001


class TestWindFactor:
    """Тесты расчета фактора ветра φw"""
    
    def test_wind_factor_zero_wind(self, physics_service):
        """Тест фактора ветра при нулевом ветре"""
        phi_w = physics_service.calculate_wind_factor(0.0)
        assert phi_w == 0.0
    
    def test_wind_factor_increases_with_speed(self, physics_service):
        """Тест увеличения фактора ветра со скоростью"""
        phi_w_low = physics_service.calculate_wind_factor(5.0)
        phi_w_high = physics_service.calculate_wind_factor(15.0)
        
        assert phi_w_low > 0
        assert phi_w_high > phi_w_low
    
    def test_wind_factor_different_fuels(self, physics_service):
        """Тест фактора ветра для разных типов топлива"""
        phi_w_grass = physics_service.calculate_wind_factor(10.0, FuelType.SHORT_GRASS)
        phi_w_timber = physics_service.calculate_wind_factor(10.0, FuelType.TIMBER_UNDERSTORY)
        phi_w_conifer = physics_service.calculate_wind_factor(10.0, FuelType.CONIFER_LITTER)
        
        # Все должны быть положительными
        assert phi_w_grass > 0
        assert phi_w_timber > 0
        assert phi_w_conifer > 0
        
        # Разные значения из-за разных σ/δ
        assert phi_w_grass != phi_w_timber or phi_w_timber != phi_w_conifer


class TestSlopeFactor:
    """Тесты расчета фактора склона φs"""
    
    def test_slope_factor_zero_slope(self, physics_service):
        """Тест фактора склона при нулевом уклоне"""
        phi_s = physics_service.calculate_slope_factor(0.0)
        # tan(0) = 0, поэтому φs должен быть 0
        assert phi_s == 0.0
    
    def test_slope_factor_increases_with_angle(self, physics_service):
        """Тест увеличения фактора склона с углом"""
        phi_s_10 = physics_service.calculate_slope_factor(10.0)
        phi_s_20 = physics_service.calculate_slope_factor(20.0)
        phi_s_30 = physics_service.calculate_slope_factor(30.0)
        
        assert phi_s_10 > 0
        assert phi_s_20 > phi_s_10
        assert phi_s_30 > phi_s_20
    
    def test_slope_factor_aspect_multipliers(self, physics_service):
        """Тест множителей аспекта склона"""
        # S (south) имеет максимальный множитель 1.2
        phi_s_south = physics_service.calculate_slope_factor(15.0, aspect='S')
        phi_s_north = physics_service.calculate_slope_factor(15.0, aspect='N')
        
        assert phi_s_south > phi_s_north
        assert abs(phi_s_south / phi_s_north - 1.2 / 0.8) < 0.01
    
    def test_slope_formula_verification(self, physics_service):
        """Проверка формулы φs = 5.275 × β^-0.3 × (tan(slope))^2 × aspect_multiplier"""
        slope_deg = 15.0
        aspect = 'S'
        
        phi_s = physics_service.calculate_slope_factor(slope_deg, aspect)
        
        slope_rad = math.radians(slope_deg)
        tan_slope = math.tan(slope_rad)
        aspect_mult = ASPECT_MULTIPLIERS[aspect]
        
        expected_phi_s = 5.275 * (BETA ** -0.3) * (tan_slope ** 2) * aspect_mult
        
        assert abs(phi_s - expected_phi_s) < 0.001


class TestEllipseGeometry:
    """Тесты геометрии эллипса распространения"""
    
    def test_ellipse_major_minor_ratio(self, physics_service):
        """Тест соотношения осей эллипса (major = 1.5×distance, minor = 0.8×distance)"""
        ellipse = physics_service.predict_spread_polygon(
            center_lat=55.0,
            center_lon=82.0,
            hours=24,
            wind_speed_ms=10.0,
            wind_direction_deg=90.0,
            slope_deg=0.0,
            fuel_type=FuelType.SHORT_GRASS
        )
        
        distance = ellipse.spread_distance_m
        
        # Проверка соотношений из спецификации
        assert abs(ellipse.major_axis_m - 1.5 * distance) < 0.01
        assert abs(ellipse.minor_axis_m - 0.8 * distance) < 0.01
    
    def test_ellipse_geojson_structure(self, physics_service):
        """Тест структуры GeoJSON эллипса"""
        ellipse = physics_service.predict_spread_polygon(
            center_lat=55.0,
            center_lon=82.0,
            hours=12,
            wind_speed_ms=5.0,
            wind_direction_deg=45.0
        )
        
        geojson = ellipse.to_geojson()
        
        assert geojson["type"] == "Feature"
        assert geojson["geometry"]["type"] == "Polygon"
        assert len(geojson["geometry"]["coordinates"][0]) > 3  # Замкнутый полигон
        
        # Проверка свойств
        props = geojson["properties"]
        assert props["center_lat"] == 55.0
        assert props["center_lon"] == 82.0
        assert props["wind_direction_deg"] == 45.0
        assert "confidence_low" in props
        assert "confidence_high" in props
        assert props["model"] == "rothermel_1972"
    
    def test_ellipse_center_preservation(self, physics_service):
        """Тест сохранения центра эллипса"""
        test_lats = [0.0, 45.0, -30.0, 89.0]
        test_lons = [0.0, 90.0, -120.0, 179.0]
        
        for lat in test_lats:
            for lon in test_lons:
                ellipse = physics_service.predict_spread_polygon(
                    center_lat=lat,
                    center_lon=lon,
                    hours=6,
                    wind_speed_ms=3.0,
                    wind_direction_deg=0.0
                )
                
                assert ellipse.center_lat == lat
                assert ellipse.center_lon == lon


class TestFullPrediction:
    """Тесты полного прогнозирования распространения"""
    
    def test_full_prediction_scenario(self, physics_service):
        """Тест полного сценария прогнозирования"""
        result = physics_service.simulate_fire_scenario(
            center_lat=55.0,
            center_lon=82.0,
            hours=24,
            weather={
                "wind_speed": 10.0,
                "wind_direction": 90.0,
                "temp": 30.0,
                "humidity": 20.0
            },
            terrain={
                "slope": 10.0,
                "aspect": "E"
            },
            fuel_type=FuelType.CONIFER_LITTER
        )
        
        # Проверка структуры результата
        assert "center" in result
        assert "simulation_hours" in result
        assert "weather" in result
        assert "terrain" in result
        assert "fuel_type" in result
        assert "spread_rate_ms" in result
        assert "base_rate_r0" in result
        assert "spread_distance_m" in result
        assert "ellipse" in result
        assert "confidence_interval" in result
        assert "timestamp" in result
        
        # Проверка значений
        assert result["simulation_hours"] == 24
        assert result["fuel_type"] == "conifer_litter"
        assert result["spread_rate_ms"] > 0
        assert result["spread_distance_m"] > 0
        
        # Ellipse должен быть валидным GeoJSON
        ellipse = result["ellipse"]
        assert ellipse["type"] == "Feature"
        assert ellipse["geometry"]["type"] == "Polygon"
    
    def test_prediction_time_scaling(self, physics_service):
        """Тест масштабирования прогноза со временем"""
        distances = []
        for hours in [6, 12, 24, 48]:
            ellipse = physics_service.predict_spread_polygon(
                center_lat=55.0,
                center_lon=82.0,
                hours=hours,
                wind_speed_ms=5.0,
                wind_direction_deg=0.0
            )
            distances.append(ellipse.spread_distance_m)
        
        # Расстояние должно расти линейно со временем
        # d(12) ≈ 2 × d(6), d(24) ≈ 4 × d(6), d(48) ≈ 8 × d(6)
        d6 = distances[0]
        assert d6 > 0
        
        for i, (hours, dist) in enumerate(zip([6, 12, 24, 48], distances)):
            expected_ratio = hours / 6
            actual_ratio = dist / d6
            assert abs(actual_ratio - expected_ratio) < 0.01
    
    def test_wind_speed_impact(self, physics_service):
        """Тест влияния скорости ветра на распространение"""
        distances = []
        for wind_speed in [0.0, 5.0, 10.0, 15.0]:
            ellipse = physics_service.predict_spread_polygon(
                center_lat=55.0,
                center_lon=82.0,
                hours=12,
                wind_speed_ms=wind_speed,
                wind_direction_deg=90.0
            )
            distances.append(ellipse.spread_distance_m)
        
        # С ростом ветра расстояние должно увеличиваться
        for i in range(1, len(distances)):
            assert distances[i] >= distances[i-1]
    
    def test_slope_impact(self, physics_service):
        """Тест влияния уклона на распространение"""
        distances_flat = physics_service.predict_spread_polygon(
            center_lat=55.0,
            center_lon=82.0,
            hours=12,
            wind_speed_ms=5.0,
            wind_direction_deg=0.0,
            slope_deg=0.0
        ).spread_distance_m
        
        distances_slope = physics_service.predict_spread_polygon(
            center_lat=55.0,
            center_lon=82.0,
            hours=12,
            wind_speed_ms=5.0,
            wind_direction_deg=0.0,
            slope_deg=15.0,
            aspect='S'
        ).spread_distance_m
        
        # Уклон должен увеличивать скорость распространения
        assert distances_slope > distances_flat


class TestAspectFromWindDirection:
    """Тесты определения аспекта по направлению ветра"""
    
    def test_aspect_cardinal_directions(self, physics_service):
        """Тест кардинальных направлений"""
        assert physics_service.get_aspect_from_wind_direction(0.0) == 'N'
        assert physics_service.get_aspect_from_wind_direction(90.0) == 'E'
        assert physics_service.get_aspect_from_wind_direction(180.0) == 'S'
        assert physics_service.get_aspect_from_wind_direction(270.0) == 'W'
    
    def test_aspect_intercardinal_directions(self, physics_service):
        """Тест промежуточных направлений"""
        assert physics_service.get_aspect_from_wind_direction(45.0) == 'NE'
        assert physics_service.get_aspect_from_wind_direction(135.0) == 'SE'
        assert physics_service.get_aspect_from_wind_direction(225.0) == 'SW'
        assert physics_service.get_aspect_from_wind_direction(315.0) == 'NW'
    
    def test_aspect_normalization(self, physics_service):
        """Тест нормализации углов"""
        assert physics_service.get_aspect_from_wind_direction(360.0) == 'N'
        assert physics_service.get_aspect_from_wind_direction(405.0) == 'NE'
        assert physics_service.get_aspect_from_wind_direction(-45.0) == 'NW'


class TestFuelModels:
    """Тесты моделей топлива"""
    
    def test_fuel_models_exist(self):
        """Тест существования всех моделей топлива"""
        assert FuelType.SHORT_GRASS in FUEL_MODELS
        assert FuelType.TIMBER_UNDERSTORY in FUEL_MODELS
        assert FuelType.CONIFER_LITTER in FUEL_MODELS
    
    def test_fuel_model_parameters(self):
        """Тест параметров моделей топлива"""
        grass = FUEL_MODELS[FuelType.SHORT_GRASS]
        assert grass.ir == 1500.0
        assert grass.xi == 0.4
        assert grass.rho_b == 0.2
        assert grass.sigma == 3500.0
        assert grass.delta == 0.3
        
        timber = FUEL_MODELS[FuelType.TIMBER_UNDERSTORY]
        assert timber.ir == 2500.0
        assert timber.xi == 0.3
        assert timber.rho_b == 0.4
        
        conifer = FUEL_MODELS[FuelType.CONIFER_LITTER]
        assert conifer.ir == 3000.0
        assert conifer.xi == 0.35
        assert conifer.rho_b == 0.3
    
    def test_get_fuel_model_default(self, physics_service):
        """Тест получения модели топлива по умолчанию"""
        model = physics_service.get_fuel_model()
        assert model.name == "short_grass"
    
    def test_get_fuel_model_explicit(self, physics_service):
        """Тест явного указания модели топлива"""
        model = physics_service.get_fuel_model(FuelType.CONIFER_LITTER)
        assert model.name == "conifer_litter"
        assert model.ir == 3000.0
