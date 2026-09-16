# Wildfire Nexus Core

Хакатонный сервис мониторинга лесных пожаров на основе спутниковых данных ДЗЗ.

## Возможности

- 🔥 Детекция очагов горения (MODIS/VIIRS)
- 🗺️ Картирование гарей (Sentinel-2 NBR/dNBR)
- 📊 Кластеризация точек в события
- 🚨 Telegram-алерты с приоритизацией
- 📈 Метрики качества детекции
- 🔮 Прогнозирование распространения огня (Rothermel + Cellular Automata)
- 🎯 Байесовская оценка риска пожара

---

## Scientific Approach

Wildfire Nexus Core использует математически обоснованные модели для прогнозирования и оценки рисков:

### 1. Физическая модель Rothermel (1972)

Расчет скорости распространения фронта пожара:

```
R = R₀ × (1 + φw + φs)

где:
- R₀ = (IR × ξ) / (ρb × ε × Qig) — базовая скорость
- φw — фактор ветра (зависит от скорости и направления)
- φs — фактор склона (зависит от уклона и аспекта)
```

**Применение:** Прогноз эллипса распространения на 6/12/24/48/72 часов.

### 2. Клеточные автоматы

Вероятностная симуляция распространения на сетке 100×100м:

```
p(ignition) = p_base + p_wind + p_fuel

Состояния: не горит → горит → сгорело
Шаг времени: 15 минут
```

**Применение:** Детальная симуляция динамики пожара с учетом препятствий.

### 3. Байесовский риск

Оценка вероятности возникновения пожара:

```
P(fire | evidence) = P(evidence | fire) × P(fire) / P(evidence)

Факторы: температура, влажность, ветер, тип топлива, уклон
```

**Применение:** Карты риска в реальном времени с доверительными интервалами.

### 4. Адаптивные пороги severity

Классификация тяжести пожара по dNBR с учетом типа растительности:

| Тип | Low | Moderate | High |
|-----|-----|----------|------|
| conifer | 0.10-0.27 | 0.27-0.44 | >0.44 |
| deciduous | 0.15-0.35 | 0.35-0.55 | >0.55 |
| grass | 0.20-0.40 | 0.40-0.60 | >0.60 |

**Применение:** Точная оценка ущерба для разных экосистем.

Подробнее см. [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## Быстрый старт

### Требования

- Docker 24+
- Docker Compose 2.20+

### Запуск

```bash
git clone https://github.com/vvseweedno/-fire_cosmo.git
cd -fire_cosmo
cp .env.example .env
docker compose up --build
```

Откройте http://localhost:8000

## API Endpoints

### Fire Detection & Events

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка здоровья |
| GET | `/api/v1/fires` | Список очагов (GeoJSON) |
| POST | `/api/v1/fires/analyze` | Анализ региона |
| GET | `/api/v1/events` | Список событий |
| GET | `/api/v1/events/{id}` | Детали события |
| POST | `/api/v1/fires/{id}/predict-spread` | Прогноз распространения |
| POST | `/api/v1/fires/{id}/assess-risk` | Оценка риска |
| GET | `/api/v1/fires/{id}/severity` | Классификация severity |

### Mathematical Prediction Models

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/api/v1/prediction/spread-rothermel` | Прогноз по модели Ротермеля |
| POST | `/api/v1/prediction/spread-cellular` | Симуляция клеточных автоматов |
| POST | `/api/v1/prediction/assess-risk` | Байесовская оценка риска |
| POST | `/api/v1/prediction/update-risk` | Обновление риска по наблюдению |

---

## Usage Examples

### 1. Прогноз распространения (Rothermel)

```bash
curl -X POST http://localhost:8000/api/v1/prediction/spread-rothermel \
  -H "Content-Type: application/json" \
  -d '{
    "center_lat": 55.75,
    "center_lon": 82.50,
    "hours": 24,
    "wind_speed_ms": 10.0,
    "wind_direction_deg": 90.0,
    "slope_deg": 15.0,
    "fuel_type": "conifer_litter",
    "aspect": "S"
  }'
```

**Ответ:** GeoJSON эллипса с доверительным интервалом.

### 2. Оценка риска (Bayesian)

```bash
curl -X POST http://localhost:8000/api/v1/prediction/assess-risk \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 30.0,
    "humidity": 25.0,
    "wind_speed": 8.0,
    "fuel_type": "short_grass",
    "slope_deg": 10.0
  }'
```

**Ответ:**
```json
{
  "probability": 0.73,
  "confidence_interval": [0.58, 0.88],
  "risk_level": "high",
  "evidence": {
    "temp_factor": 0.75,
    "humidity_factor": 0.75,
    "wind_factor": 0.53,
    "fuel_factor": 0.90,
    "slope_factor": 0.33
  }
}
```

### 3. Симуляция клеточных автоматов

```bash
curl -X POST http://localhost:8000/api/v1/prediction/spread-cellular \
  -H "Content-Type: application/json" \
  -d '{
    "grid_rows": 100,
    "grid_cols": 100,
    "ignition_row": 50,
    "ignition_col": 50,
    "wind_speed": 10.0,
    "wind_direction": 90.0,
    "hours": 12.0
  }'
```

**Ответ:** Финальная сетка состояний + статистика выгоревшей площади.

## Архитектура

См. [ARCHITECTURE.md](./ARCHITECTURE.md)

## Тесты

```bash
pytest tests/ -v --cov=app --cov-report=html
```

## Ограничения

- Задержка данных FIRMS: 3-4 часа
- Требуется MAP_KEY для реальных данных (иначе фикстуры)

## Лицензия

MIT
