"""
API endpoints для прогнозирования распространения пожара
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from app.services.fire_physics import FirePhysicsService, FuelType
from app.services.fire_cellular_automata import FireCellularAutomata
from app.services.bayesian_fire_risk import BayesianFireRiskEstimator, BayesianFireRisk, RiskLevel

router = APIRouter(prefix="/prediction", tags=["Prediction"])

# === Pydantic Schemas ===

class PredictSpreadRequest(BaseModel):
    """Запрос на прогноз распространения"""
    center_lat: float = Field(..., description="Широта центра пожара")
    center_lon: float = Field(..., description="Долгота центра пожара")
    hours: int = Field(..., ge=1, le=72, description="Время прогноза (часы)")
    wind_speed_ms: float = Field(..., ge=0, description="Скорость ветра (м/с)")
    wind_direction_deg: float = Field(..., ge=0, lt=360, description="Направление ветра (градусы)")
    slope_deg: float = Field(0.0, ge=0, le=90, description="Уклон склона (градусы)")
    fuel_type: str = Field("conifer_litter", description="Тип топлива")
    aspect: str = Field("S", description="Аспект склона (N/NE/E/SE/S/SW/W/NW)")

class AssessRiskRequest(BaseModel):
    """Запрос на оценку риска"""
    temperature: float = Field(..., description="Температура воздуха (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Влажность (%)")
    wind_speed: float = Field(..., ge=0, description="Скорость ветра (м/с)")
    fuel_type: str = Field(..., description="Тип топлива")
    slope_deg: float = Field(0.0, ge=0, le=90, description="Уклон склона")

class SimulateCellularRequest(BaseModel):
    """Запрос на симуляцию клеточных автоматов"""
    grid_rows: int = Field(100, ge=10, le=500)
    grid_cols: int = Field(100, ge=10, le=500)
    ignition_row: int = Field(50, ge=0)
    ignition_col: int = Field(50, ge=0)
    wind_speed: float = Field(0.0, ge=0)
    wind_direction: float = Field(0.0, ge=0, lt=360)
    hours: float = Field(24.0, ge=0.25, le=72)


# === Endpoints ===

@router.post("/spread-rothermel")
async def predict_spread_rothermel(request: PredictSpreadRequest):
    """
    Прогноз распространения по модели Ротермеля (1972)
    
    Формула: R = R0 × (1 + φw + φs)
    
    Возвращает GeoJSON эллипса распространения с доверительным интервалом.
    """
    try:
        fuel_type_enum = FuelType(request.fuel_type)
    except ValueError:
        fuel_type_enum = FuelType.CONIFER_LITTER
    
    physics = FirePhysicsService(default_fuel_type=fuel_type_enum)
    
    result = physics.simulate_fire_scenario(
        center_lat=request.center_lat,
        center_lon=request.center_lon,
        hours=request.hours,
        weather={
            "wind_speed": request.wind_speed_ms,
            "wind_direction": request.wind_direction_deg,
            "temp": 25.0,  # дефолт
            "humidity": 40.0
        },
        terrain={
            "slope": request.slope_deg,
            "aspect": request.aspect
        },
        fuel_type=fuel_type_enum
    )
    
    return {
        "model": "Rothermel 1972",
        "formula": "R = R0 × (1 + φw + φs)",
        **result
    }

@router.post("/spread-cellular")
async def simulate_spread_cellular(request: SimulateCellularRequest):
    """
    Симуляция распространения через клеточные автоматы
    
    Формула: p(ignition) = p_base + p_wind + p_fuel
    
    Возвращает финальную сетку состояний и статистику.
    """
    ca = FireCellularAutomata(grid_shape=(request.grid_rows, request.grid_cols))
    
    # Установка точки возгорания
    ca.set_ignition_point(request.ignition_row, request.ignition_col)
    
    # Установка ветра
    ca.set_wind(request.wind_speed, request.wind_direction)
    
    # Запуск симуляции
    final_grid = ca.simulate(hours=request.hours)
    
    # Статистика
    import numpy as np
    burned_cells = int(np.sum(final_grid == 2))  # CellState.BURNED
    cell_area_ha = (ca.cell_size_m ** 2) / 10000
    burned_area_ha = burned_cells * cell_area_ha
    
    return {
        "model": "Cellular Automata",
        "formula": "p = p_base + p_wind + p_fuel",
        "grid_shape": list(final_grid.shape),
        "burned_cells": burned_cells,
        "burned_area_ha": round(burned_area_ha, 2),
        "simulation_hours": request.hours,
        "cell_size_m": ca.cell_size_m,
        "grid_encoded": final_grid.tolist()  # Для визуализации
    }

@router.post("/assess-risk")
async def assess_fire_risk(request: AssessRiskRequest):
    """
    Байесовская оценка риска пожара
    
    Формула: P(fire | evidence) = P(evidence | fire) × P(fire) / P(evidence)
    
    Возвращает вероятность, доверительный интервал, уровень риска.
    """
    estimator = BayesianFireRiskEstimator(default_prior=0.10)
    
    risk = estimator.estimate_risk(
        temperature=request.temperature,
        humidity=request.humidity,
        wind_speed=request.wind_speed,
        fuel_type=request.fuel_type,
        slope_deg=request.slope_deg
    )
    
    return {
        "model": "Bayesian Risk Estimation",
        "formula": "P(fire | evidence) = P(evidence | fire) × P(fire) / P(evidence)",
        **risk.to_dict()
    }

@router.post("/update-risk")
async def update_risk_with_observation(
    current_probability: float = Query(..., ge=0, le=1),
    fire_observed: bool = Query(..., description="Был ли пожар?"),
    learning_rate: float = Query(0.1, ge=0, le=1)
):
    """
    Обновить оценку риска по новому наблюдению (байесовское обновление)
    """
    estimator = BayesianFireRiskEstimator()
    
    # Определяем текущий уровень риска
    if current_probability < 0.25:
        risk_level = RiskLevel.LOW
    elif current_probability < 0.50:
        risk_level = RiskLevel.MODERATE
    elif current_probability < 0.75:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.EXTREME
    
    current_risk = BayesianFireRisk(
        probability=current_probability,
        confidence_interval=(max(0, current_probability - 0.1), min(1, current_probability + 0.1)),
        risk_level=risk_level,
        evidence={}
    )
    
    updated = estimator.update_with_observation(
        current_risk=current_risk,
        observation=fire_observed,
        learning_rate=learning_rate
    )
    
    return {
        "previous_probability": current_probability,
        "fire_observed": fire_observed,
        **updated.to_dict()
    }
