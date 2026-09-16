"""
Физическая модель распространения огня (Rothermel, 1972)

Модель описывает скорость распространения фронта пожара:
R = R0 × (1 + φw + φs)

где:
- R0 = (IR × ξ) / (ρb × ε × Qig) — базовая скорость распространения
- φw = C × (3.281 × U)^B × (σ/δ)^-E — фактор ветра
- φs = 5.275 × β^-0.3 × (tan(slope))^2 × aspect_multiplier — фактор склона

Параметры:
- IR = reaction_intensity (кВт/м²)
- ξ = propagating_flux_ratio
- ρb = bulk_density (кг/м³)
- ε = effective_heating_number (= 0.716)
- Qig = heat_of_ignition (кДж/кг, ~= 514)
- U = wind_speed (м/с → футы/мин: ×196.85)
- σ = surface_area_ratio (м²/м³)
- δ = fuel_bed_depth (м)
- β = 0.5 (коэффициент упаковки)
- aspect_multiplier = {'N':0.8,'NE':0.9,'E':1.0,'SE':1.1,'S':1.2,'SW':1.15,'W':1.0,'NW':0.85}
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, scale
import geojson


logger = logging.getLogger(__name__)


class FuelType(str, Enum):
    """Типы топлива с параметрами из спецификации"""
    SHORT_GRASS = "short_grass"
    TIMBER_UNDERSTORY = "timber_understory"
    CONIFER_LITTER = "conifer_litter"


@dataclass
class FuelModel:
    """Модель топлива с параметрами для уравнения Ротермеля"""
    name: str
    ir: float  # reaction_intensity (кВт/м²)
    xi: float  # propagating_flux_ratio
    rho_b: float  # bulk_density (кг/м³)
    sigma: float  # surface_area_ratio (м²/м³)
    delta: float  # fuel_bed_depth (м)


# Параметры моделей топлива из спецификации
FUEL_MODELS: Dict[FuelType, FuelModel] = {
    FuelType.SHORT_GRASS: FuelModel(
        name="short_grass",
        ir=1500.0,
        xi=0.4,
        rho_b=0.2,
        sigma=3500.0,
        delta=0.3
    ),
    FuelType.TIMBER_UNDERSTORY: FuelModel(
        name="timber_understory",
        ir=2500.0,
        xi=0.3,
        rho_b=0.4,
        sigma=2000.0,
        delta=0.6
    ),
    FuelType.CONIFER_LITTER: FuelModel(
        name="conifer_litter",
        ir=3000.0,
        xi=0.35,
        rho_b=0.3,
        sigma=2500.0,
        delta=0.5
    ),
}

# Константы модели
EPSILON = 0.716  # effective_heating_number
QIG = 514.0  # heat_of_ignition (кДж/кг)
BETA = 0.5  # коэффициент упаковки

# Коэффициенты для фактора ветра (из калибровки Rothermel)
C_WIND = 0.044
B_WIND = 1.2
E_WIND = 0.6

# Множители для аспекта склона
ASPECT_MULTIPLIERS: Dict[str, float] = {
    'N': 0.8,
    'NE': 0.9,
    'E': 1.0,
    'SE': 1.1,
    'S': 1.2,
    'SW': 1.15,
    'W': 1.0,
    'NW': 0.85,
}


@dataclass
class SpreadEllipse:
    """Эллипс распространения пожара"""
    center_lat: float
    center_lon: float
    major_axis_m: float  # большая полуось (м)
    minor_axis_m: float  # малая полуось (м)
    wind_direction_deg: float  # направление ветра (градусы)
    spread_distance_m: float  # расстояние распространения (м)
    confidence_interval: Tuple[float, float]  # доверительный интервал
    
    def to_geojson(self) -> Dict[str, Any]:
        """Конвертировать эллипс в GeoJSON Polygon"""
        # Создаем эллипс через аппроксимацию полигоном
        num_points = 64
        angles = [2 * math.pi * i / num_points for i in range(num_points)]
        
        # Полуоси эллипса
        a = self.major_axis_m  # большая полуось
        b = self.minor_axis_m  # малая полуось
        
        # Генерируем точки эллипса в локальных координатах
        local_points = []
        for angle in angles:
            x = a * math.cos(angle)
            y = b * math.sin(angle)
            local_points.append((x, y))
        
        # Поворот эллипса по направлению ветра
        wind_rad = math.radians(self.wind_direction_deg)
        cos_w = math.cos(wind_rad)
        sin_w = math.sin(wind_rad)
        
        rotated_points = []
        for x, y in local_points:
            xr = x * cos_w - y * sin_w
            yr = x * sin_w + y * cos_w
            rotated_points.append((xr, yr))
        
        # Смещение к центру (в метрах от центра)
        # Для простоты считаем что 1 градус ≈ 111000 м
        meters_per_deg = 111000.0
        global_points = []
        for x, y in rotated_points:
            lon = self.center_lon + x / meters_per_deg
            lat = self.center_lat + y / meters_per_deg
            global_points.append([lon, lat])
        
        # Замыкаем полигон
        global_points.append(global_points[0])
        
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [global_points]
            },
            "properties": {
                "center_lat": self.center_lat,
                "center_lon": self.center_lon,
                "major_axis_m": self.major_axis_m,
                "minor_axis_m": self.minor_axis_m,
                "wind_direction_deg": self.wind_direction_deg,
                "spread_distance_m": self.spread_distance_m,
                "confidence_low": self.confidence_interval[0],
                "confidence_high": self.confidence_interval[1],
                "model": "rothermel_1972"
            }
        }


class FirePhysicsService:
    """
    Сервис физической модели распространения огня на основе уравнения Ротермеля (1972)
    
    R = R0 × (1 + φw + φs)
    
    где:
    - R0 — базовая скорость распространения (без ветра и склона)
    - φw — фактор ветра
    - φs — фактор склона
    """
    
    def __init__(self, default_fuel_type: FuelType = FuelType.SHORT_GRASS):
        """
        Инициализировать сервис физики пожара
        
        Args:
            default_fuel_type: Тип топлива по умолчанию
        """
        self.default_fuel_type = default_fuel_type
        logger.info(f"FirePhysicsService initialized with default fuel: {default_fuel_type.value}")
    
    def get_fuel_model(self, fuel_type: Optional[FuelType] = None) -> FuelModel:
        """
        Получить модель топлива
        
        Args:
            fuel_type: Тип топлива (если None, используется default)
            
        Returns:
            FuelModel с параметрами
        """
        ft = fuel_type or self.default_fuel_type
        return FUEL_MODELS[ft]
    
    def calculate_base_rate(
        self,
        fuel_type: Optional[FuelType] = None,
        epsilon: float = EPSILON,
        qig: float = QIG
    ) -> float:
        """
        Рассчитать базовую скорость распространения R0 (без ветра и склона)
        
        R0 = (IR × ξ) / (ρb × ε × Qig)
        
        Args:
            fuel_type: Тип топлива
            epsilon: effective_heating_number
            qig: heat_of_ignition (кДж/кг)
            
        Returns:
            R0 в м/с
        """
        fuel = self.get_fuel_model(fuel_type)
        
        # R0 = (IR × ξ) / (ρb × ε × Qig)
        numerator = fuel.ir * fuel.xi
        denominator = fuel.rho_b * epsilon * qig
        
        if denominator <= 0:
            logger.warning("Invalid denominator in R0 calculation, returning 0")
            return 0.0
        
        r0 = numerator / denominator
        
        logger.debug(
            f"Base rate R0 calculated: {r0:.6f} m/s "
            f"(fuel={fuel.name}, IR={fuel.ir}, xi={fuel.xi}, rho_b={fuel.rho_b})"
        )
        
        return r0
    
    def calculate_wind_factor(
        self,
        wind_speed_ms: float,
        fuel_type: Optional[FuelType] = None,
        c: float = C_WIND,
        b: float = B_WIND,
        e: float = E_WIND
    ) -> float:
        """
        Рассчитать фактор ветра φw
        
        φw = C × (3.281 × U)^B × (σ/δ)^-E
        
        где U переводится из м/с в футы/мин (×196.85)
        
        Args:
            wind_speed_ms: Скорость ветра (м/с)
            fuel_type: Тип топлива
            c, b, e: Калибровочные коэффициенты
            
        Returns:
            φw (безразмерный множитель)
        """
        fuel = self.get_fuel_model(fuel_type)
        
        # Конвертация ветра: м/с → футы/мин
        # 1 м/с = 196.85 фут/мин
        u_fpm = wind_speed_ms * 196.85
        
        # Отношение поверхности к объёму
        sigma_delta_ratio = fuel.sigma / fuel.delta
        
        # φw = C × (3.281 × U)^B × (σ/δ)^-E
        # Примечание: 3.281 уже учтено в конвертации, используем напрямую U в футах/мин
        wind_term = c * (u_fpm ** b)
        fuel_term = sigma_delta_ratio ** (-e)
        
        phi_w = wind_term * fuel_term
        
        logger.debug(
            f"Wind factor φw calculated: {phi_w:.4f} "
            f"(wind={wind_speed_ms} m/s, σ/δ={sigma_delta_ratio:.2f})"
        )
        
        return max(0.0, phi_w)
    
    def calculate_slope_factor(
        self,
        slope_deg: float,
        aspect: str = 'S',
        beta: float = BETA
    ) -> float:
        """
        Рассчитать фактор склона φs
        
        φs = 5.275 × β^-0.3 × (tan(slope))^2 × aspect_multiplier
        
        Args:
            slope_deg: Уклон склона (градусы)
            aspect: Аспект склона (N, NE, E, SE, S, SW, W, NW)
            beta: Коэффициент упаковки
            
        Returns:
            φs (безразмерный множитель)
        """
        # Конвертация угла в радианы
        slope_rad = math.radians(slope_deg)
        
        # tan(slope)
        tan_slope = math.tan(slope_rad)
        
        # aspect_multiplier
        aspect_mult = ASPECT_MULTIPLIERS.get(aspect.upper(), 1.0)
        
        # φs = 5.275 × β^-0.3 × (tan(slope))^2 × aspect_multiplier
        beta_term = beta ** (-0.3)
        slope_term = tan_slope ** 2
        
        phi_s = 5.275 * beta_term * slope_term * aspect_mult
        
        logger.debug(
            f"Slope factor φs calculated: {phi_s:.4f} "
            f"(slope={slope_deg}°, aspect={aspect}, multiplier={aspect_mult})"
        )
        
        return max(0.0, phi_s)
    
    def calculate_spread_rate(
        self,
        wind_speed_ms: float,
        slope_deg: float,
        fuel_type: Optional[FuelType] = None,
        aspect: str = 'S'
    ) -> float:
        """
        Рассчитать полную скорость распространения огня R
        
        R = R0 × (1 + φw + φs)
        
        Args:
            wind_speed_ms: Скорость ветра (м/с)
            slope_deg: Уклон склона (градусы)
            fuel_type: Тип топлива
            aspect: Аспект склона
            
        Returns:
            R в м/с
        """
        r0 = self.calculate_base_rate(fuel_type)
        phi_w = self.calculate_wind_factor(wind_speed_ms, fuel_type)
        phi_s = self.calculate_slope_factor(slope_deg, aspect)
        
        r = r0 * (1.0 + phi_w + phi_s)
        
        logger.info(
            f"Spread rate R calculated: {r:.6f} m/s "
            f"(R0={r0:.6f}, φw={phi_w:.4f}, φs={phi_s:.4f})"
        )
        
        return r
    
    def predict_spread_distance(
        self,
        hours: int,
        wind_speed_ms: float,
        slope_deg: float,
        fuel_type: Optional[FuelType] = None,
        aspect: str = 'S'
    ) -> float:
        """
        Предсказать расстояние распространения огня за заданное время
        
        Args:
            hours: Время прогнозирования (часы)
            wind_speed_ms: Скорость ветра (м/с)
            slope_deg: Уклон склона (градусы)
            fuel_type: Тип топлива
            aspect: Аспект склона
            
        Returns:
            Расстояние распространения (метры)
        """
        spread_rate = self.calculate_spread_rate(
            wind_speed_ms, slope_deg, fuel_type, aspect
        )
        
        # Конвертация часов в секунды
        seconds = hours * 3600
        
        distance = spread_rate * seconds
        
        logger.info(
            f"Predicted spread distance: {distance:.2f} m "
            f"over {hours} hours at {spread_rate:.6f} m/s"
        )
        
        return distance
    
    def predict_spread_polygon(
        self,
        center_lat: float,
        center_lon: float,
        hours: int,
        wind_speed_ms: float,
        wind_direction_deg: float,
        slope_deg: float = 0.0,
        fuel_type: Optional[FuelType] = None,
        aspect: str = 'S'
    ) -> SpreadEllipse:
        """
        Сгенерировать эллипс распространения пожара
        
        Эллипс вытянут по направлению ветра:
        - major_axis = 1.5 × distance
        - minor_axis = 0.8 × distance
        
        Args:
            center_lat: Широта центра (градусы)
            center_lon: Долгота центра (градусы)
            hours: Время прогнозирования (часы)
            wind_speed_ms: Скорость ветра (м/с)
            wind_direction_deg: Направление ветра (градусы, 0=N, 90=E)
            slope_deg: Уклон склона (градусы)
            fuel_type: Тип топлива
            aspect: Аспект склона
            
        Returns:
            SpreadEllipse с геометрией и параметрами
        """
        # Расчет расстояния распространения
        distance = self.predict_spread_distance(
            hours, wind_speed_ms, slope_deg, fuel_type, aspect
        )
        
        # Параметры эллипса из спецификации
        major_axis = 1.5 * distance
        minor_axis = 0.8 * distance
        
        # Доверительный интервал (эмпирический, ±20%)
        confidence_low = 0.8 * distance
        confidence_high = 1.2 * distance
        
        ellipse = SpreadEllipse(
            center_lat=center_lat,
            center_lon=center_lon,
            major_axis_m=major_axis,
            minor_axis_m=minor_axis,
            wind_direction_deg=wind_direction_deg,
            spread_distance_m=distance,
            confidence_interval=(confidence_low, confidence_high)
        )
        
        logger.info(
            f"Spread ellipse generated: center=({center_lat}, {center_lon}), "
            f"distance={distance:.2f}m, major={major_axis:.2f}m, minor={minor_axis:.2f}m"
        )
        
        return ellipse
    
    def get_aspect_from_wind_direction(self, wind_direction_deg: float) -> str:
        """
        Определить аспект склона по направлению ветра
        
        Args:
            wind_direction_deg: Направление ветра (градусы, 0=N, 90=E)
            
        Returns:
            Строка аспекта (N, NE, E, SE, S, SW, W, NW)
        """
        # Нормализация угла [0, 360)
        wind_direction_deg = wind_direction_deg % 360
        
        # Сектора по 45 градусов
        sectors = [
            (22.5, 'N'),
            (67.5, 'NE'),
            (112.5, 'E'),
            (157.5, 'SE'),
            (202.5, 'S'),
            (247.5, 'SW'),
            (292.5, 'W'),
            (337.5, 'NW'),
            (360, 'N'),
        ]
        
        for threshold, aspect in sectors:
            if wind_direction_deg < threshold:
                return aspect
        
        return 'N'
    
    def simulate_fire_scenario(
        self,
        center_lat: float,
        center_lon: float,
        hours: int,
        weather: Dict[str, Any],
        terrain: Dict[str, Any],
        fuel_type: Optional[FuelType] = None
    ) -> Dict[str, Any]:
        """
        Симулировать сценарий распространения пожара
        
        Args:
            center_lat: Широта центра
            center_lon: Долгота центра
            hours: Время симуляции (часы)
            weather: Погодные данные {wind_speed, wind_direction, temp, humidity}
            terrain: Данные рельефа {slope, aspect}
            fuel_type: Тип топлива
            
        Returns:
            Dict с результатами симуляции
        """
        wind_speed = weather.get('wind_speed', 0.0)
        wind_direction = weather.get('wind_direction', 0.0)
        slope = terrain.get('slope', 0.0)
        aspect = terrain.get('aspect', self.get_aspect_from_wind_direction(wind_direction))
        
        # Расчет скорости распространения
        spread_rate = self.calculate_spread_rate(
            wind_speed, slope, fuel_type, aspect
        )
        
        # Прогноз эллипса
        ellipse = self.predict_spread_polygon(
            center_lat, center_lon, hours,
            wind_speed, wind_direction, slope, fuel_type, aspect
        )
        
        # Базовая скорость
        r0 = self.calculate_base_rate(fuel_type)
        
        return {
            "center": {"lat": center_lat, "lon": center_lon},
            "simulation_hours": hours,
            "weather": weather,
            "terrain": terrain,
            "fuel_type": fuel_type.value if fuel_type else self.default_fuel_type.value,
            "spread_rate_ms": spread_rate,
            "base_rate_r0": r0,
            "spread_distance_m": ellipse.spread_distance_m,
            "ellipse": ellipse.to_geojson(),
            "confidence_interval": ellipse.confidence_interval,
            "timestamp": datetime.utcnow().isoformat()
        }
