"""
Модель клеточных автоматов для симуляции распространения пожара

Сетка 100×100м клетки
Состояния: 0=не горит, 1=горит, 2=сгорело, 3=вода, 4=дорога
Вероятность возгорания соседа: p = p_base + p_wind + p_fuel
Шаг симуляции: 15 минут
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import IntEnum
import math


logger = logging.getLogger(__name__)


class CellState(IntEnum):
    """Состояния клетки в модели клеточного автомата"""
    NOT_BURNING = 0  # Не горит
    BURNING = 1      # Горит
    BURNED = 2       # Сгорело
    WATER = 3        # Вода (препятствие)
    ROAD = 4         # Дорога (препятствие)


@dataclass
class FireCellularAutomataConfig:
    """Конфигурация модели клеточных автоматов"""
    cell_size_m: float = 100.0  # Размер клетки (м)
    dt_minutes: float = 15.0    # Шаг времени (минуты)
    p_base: float = 0.3         # Базовая вероятность возгорания
    p_wind_max: float = 0.4     # Максимальный вклад ветра
    p_fuel_max: float = 0.3     # Максимальный вклад топлива
    
    # Параметры влияния ветра
    wind_alignment_power: float = 2.0  # Степень влияния направления
    
    # Коэффициенты для разных типов топлива
    fuel_factors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.fuel_factors is None:
            self.fuel_factors = {
                'short_grass': 1.0,
                'conifer_litter': 0.9,
                'timber_understory': 0.8,
                'deciduous': 0.5,
                'bare': 0.2,
                'water': 0.0,
            }


class FireCellularAutomata:
    """
    Модель клеточных автоматов для симуляции распространения лесного пожара
    
    Использует вероятностный подход с учетом:
    - Направления и скорости ветра
    - Типа топлива в каждой клетке
    - Препятствий (вода, дороги)
    """
    
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (100, 100),
        config: Optional[FireCellularAutomataConfig] = None
    ):
        """
        Инициализировать модель клеточных автоматов
        
        Args:
            grid_shape: Форма сетки (rows, cols)
            config: Конфигурация модели
        """
        self.rows, self.cols = grid_shape
        self.config = config or FireCellularAutomataConfig()
        
        # Инициализация сетки состояний (все не горят)
        self.grid: np.ndarray = np.zeros(grid_shape, dtype=np.uint8)
        
        # Сетка типа топлива (по умолчанию short_grass)
        self.fuel_grid: np.ndarray = np.zeros(grid_shape, dtype=np.uint8)
        
        # Массив направлений ветра (угол в радианах для каждой клетки)
        self.wind_direction: np.ndarray = np.zeros(grid_shape, dtype=np.float32)
        
        # Скорость ветра (м/с)
        self.wind_speed: float = 0.0
        
        logger.info(
            f"FireCellularAutomata initialized: {self.rows}x{self.cols} cells, "
            f"cell_size={self.config.cell_size_m}m"
        )
    
    def set_fuel_type(self, row_start: int, col_start: int, 
                      row_end: int, col_end: int, fuel_code: int):
        """
        Установить тип топлива в области
        
        Args:
            row_start, col_start: Начальные индексы
            row_end, col_end: Конечные индексы
            fuel_code: Код типа топлива
        """
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(self.rows, row_end)
        col_end = min(self.cols, col_end)
        
        self.fuel_grid[row_start:row_end, col_start:col_end] = fuel_code
        logger.debug(f"Set fuel type {fuel_code} in region [{row_start}:{row_end}, {col_start}:{col_end}]")
    
    def set_wind(self, speed_ms: float, direction_deg: float):
        """
        Установить параметры ветра
        
        Args:
            speed_ms: Скорость ветра (м/с)
            direction_deg: Направление ветра (градусы, 0=N, 90=E)
        """
        self.wind_speed = max(0.0, speed_ms)
        direction_rad = math.radians(direction_deg % 360)
        self.wind_direction[:, :] = direction_rad
        logger.info(f"Wind set: {speed_ms} m/s at {direction_deg}°")
    
    def set_obstacle(self, row_start: int, col_start: int,
                     row_end: int, col_end: int, obstacle_type: CellState):
        """
        Установить препятствие (вода или дорога)
        
        Args:
            row_start, col_start: Начальные индексы
            row_end, col_end: Конечные индексы
            obstacle_type: Тип препятствия (WATER или ROAD)
        """
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(self.rows, row_end)
        col_end = min(self.cols, col_end)
        
        self.grid[row_start:row_end, col_start:col_end] = obstacle_type.value
        logger.debug(f"Set obstacle {obstacle_type.name} in region")
    
    def ignite_cell(self, row: int, col: int):
        """
        Поджечь клетку
        
        Args:
            row: Индекс строки
            col: Индекс столбца
        """
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self.grid[row, col] not in [CellState.WATER.value, CellState.ROAD.value]:
                self.grid[row, col] = CellState.BURNING.value
                logger.debug(f"Ignited cell at ({row}, {col})")
    
    def ignite_region(self, row_start: int, col_start: int,
                      row_end: int, col_end: int):
        """
        Поджечь область клеток
        
        Args:
            row_start, col_start: Начальные индексы
            row_end, col_end: Конечные индексы
        """
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(self.rows, row_end)
        col_end = min(self.cols, col_end)
        
        mask = ~np.isin(self.grid[row_start:row_end, col_start:col_end], 
                        [CellState.WATER.value, CellState.ROAD.value])
        self.grid[row_start:row_end, col_start:col_end][mask] = CellState.BURNING.value
        logger.info(f"Ignited region [{row_start}:{row_end}, {col_start}:{col_end}]")
    
    def _get_fuel_factor(self, fuel_code: int) -> float:
        """
        Получить коэффициент топлива
        
        Args:
            fuel_code: Код типа топлива
            
        Returns:
            Коэффициент от 0 до 1
        """
        fuel_names = ['short_grass', 'conifer_litter', 'timber_understory', 
                      'deciduous', 'bare', 'water']
        
        if 0 <= fuel_code < len(fuel_names):
            fuel_name = fuel_names[fuel_code]
            return self.config.fuel_factors.get(fuel_name, 0.5)
        
        return 0.5  # По умолчанию
    
    def _calculate_ignition_probability(
        self,
        row: int,
        col: int,
        neighbor_row: int,
        neighbor_col: int
    ) -> float:
        """
        Рассчитать вероятность возгорания соседней клетки
        
        p = p_base + p_wind + p_fuel
        
        Args:
            row, col: Координаты горящей клетки
            neighbor_row, neighbor_col: Координаты соседней клетки
            
        Returns:
            Вероятность возгорания [0, 1]
        """
        # Базовая вероятность
        p = self.config.p_base
        
        # Фактор ветра
        if self.wind_speed > 0:
            # Вектор от горячей клетки к соседу
            drow = neighbor_row - row
            dcol = neighbor_col - col
            
            # Нормализация
            dist = math.sqrt(drow**2 + dcol**2)
            if dist > 0:
                drow /= dist
                dcol /= dist
            
            # Вектор ветра
            wind_angle = float(self.wind_direction[row, col])
            wind_row = math.sin(wind_angle)  # sin для row (север-юг)
            wind_col = math.cos(wind_angle)  # cos для col (восток-запад)
            
            # Скалярное произведение (alignment)
            alignment = drow * wind_row + dcol * wind_col
            
            # Вклад ветра зависит от alignment и скорости
            # alignment=1 означает полное совпадение направления
            wind_contribution = self.config.p_wind_max * (
                (max(0, alignment) ** self.config.wind_alignment_power) *
                min(1.0, self.wind_speed / 15.0)  # Нормализация по 15 м/с
            )
            p += wind_contribution
        
        # Фактор топлива
        fuel_code = int(self.fuel_grid[neighbor_row, neighbor_col])
        fuel_factor = self._get_fuel_factor(fuel_code)
        fuel_contribution = self.config.p_fuel_max * fuel_factor
        p += fuel_contribution
        
        # Ограничение [0, 1]
        return np.clip(p, 0.0, 1.0)
    
    def _step(self) -> int:
        """
        Выполнить один шаг симуляции
        
        Returns:
            Количество новых возгораний
        """
        new_burns = 0
        burning_cells = np.argwhere(self.grid == CellState.BURNING.value)
        
        if len(burning_cells) == 0:
            return 0
        
        # Копия сетки для атомарного обновления
        new_grid = self.grid.copy()
        
        # Соседи Мура (8 направлений)
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for row, col in burning_cells:
            for dr, dc in neighbors:
                nr, nc = row + dr, col + dc
                
                # Проверка границ
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                
                # Пропускаем если уже горит или сгорело
                if self.grid[nr, nc] in [CellState.BURNING.value, CellState.BURNED.value]:
                    continue
                
                # Препятствия не горят
                if self.grid[nr, nc] in [CellState.WATER.value, CellState.ROAD.value]:
                    continue
                
                # Расчет вероятности возгорания
                p_ignite = self._calculate_ignition_probability(row, col, nr, nc)
                
                # Стохастическая проверка
                if np.random.random() < p_ignite:
                    new_grid[nr, nc] = CellState.BURNING.value
                    new_burns += 1
            
            # Горящая клетка переходит в сгоревшую
            new_grid[row, col] = CellState.BURNED.value
        
        self.grid = new_grid
        return new_burns
    
    def simulate(self, hours: float, dt_minutes: Optional[float] = None) -> np.ndarray:
        """
        Запустить симуляцию на заданное время
        
        Args:
            hours: Длительность симуляции (часы)
            dt_minutes: Шаг времени (минуты), по умолчанию из config
            
        Returns:
            Финальное состояние сетки
        """
        dt = dt_minutes if dt_minutes is not None else self.config.dt_minutes
        
        # Количество шагов
        total_minutes = hours * 60
        num_steps = int(total_minutes / dt)
        
        logger.info(f"Starting simulation: {hours}h, {num_steps} steps, dt={dt}min")
        
        for step in range(num_steps):
            new_burns = self._step()
            
            if new_burns == 0 and step > 0:
                # Пожар потух
                logger.info(f"Fire extinguished at step {step}/{num_steps}")
                break
            
            if step % 10 == 0:
                logger.debug(f"Step {step}/{num_steps}: {new_burns} new burns")
        
        logger.info(f"Simulation complete: {hours}h simulated")
        return self.grid
    
    def get_burned_area_ha(self) -> float:
        """
        Рассчитать площадь выгоревшей территории
        
        Returns:
            Площадь в гектарах
        """
        # Клетки которые горят или сгорели
        burned_mask = np.isin(self.grid, [CellState.BURNING.value, CellState.BURNED.value])
        num_burned_cells = np.sum(burned_mask)
        
        # Площадь одной клетки (м²)
        cell_area_m2 = self.config.cell_size_m ** 2
        
        # Общая площадь (м² → га)
        area_ha = (num_burned_cells * cell_area_m2) / 10000.0
        
        logger.debug(f"Burned area: {num_burned_cells} cells = {area_ha:.2f} ha")
        return area_ha
    
    def get_fire_statistics(self) -> Dict[str, Any]:
        """
        Получить статистику пожара
        
        Returns:
            Dict со статистикой
        """
        unique, counts = np.unique(self.grid, return_counts=True)
        
        stats = {
            'total_cells': self.rows * self.cols,
            'not_burning': 0,
            'burning': 0,
            'burned': 0,
            'water': 0,
            'road': 0,
            'burned_area_ha': self.get_burned_area_ha(),
        }
        
        for state, count in zip(unique, counts):
            state_name = CellState(state).name.lower()
            stats[state_name] = int(count)
        
        return stats
    
    def get_grid_copy(self) -> np.ndarray:
        """
        Получить копию текущей сетки
        
        Returns:
            Копия сетки состояний
        """
        return self.grid.copy()
    
    def reset(self):
        """Сбросить сетку в начальное состояние"""
        self.grid[:, :] = CellState.NOT_BURNING.value
        logger.debug("Grid reset")
    
    def to_geojson(
        self,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        cell_size_deg: float = 0.001
    ) -> Dict[str, Any]:
        """
        Конвертировать сетку в GeoJSON
        
        Args:
            origin_lat: Широта левого нижнего угла
            origin_lon: Долгота левого нижнего угла
            cell_size_deg: Размер клетки в градусах
            
        Returns:
            GeoJSON FeatureCollection
        """
        features = []
        
        # Группировка по состояниям для оптимизации
        for state_val in [CellState.BURNING.value, CellState.BURNED.value]:
            cells = np.argwhere(self.grid == state_val)
            
            if len(cells) == 0:
                continue
            
            state_name = CellState(state_val).name.lower()
            
            for row, col in cells:
                # Конвертация координат
                lat = origin_lat + (self.rows - row) * cell_size_deg
                lon = origin_lon + col * cell_size_deg
                
                # Полигон клетки
                coords = [
                    [lon - cell_size_deg/2, lat - cell_size_deg/2],
                    [lon + cell_size_deg/2, lat - cell_size_deg/2],
                    [lon + cell_size_deg/2, lat + cell_size_deg/2],
                    [lon - cell_size_deg/2, lat + cell_size_deg/2],
                    [lon - cell_size_deg/2, lat - cell_size_deg/2],
                ]
                
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {
                        "state": state_name,
                        "row": int(row),
                        "col": int(col),
                        "fuel_code": int(self.fuel_grid[row, col])
                    }
                })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
