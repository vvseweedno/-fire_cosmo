# Архитектура Wildfire Nexus Core

## Диаграмма потоков данных

```
[MODIS/VIIRS] → [FireDetectionService] → [FalsePositiveFilter] → [FireClusteringService]
                                                                         ↓
[Sentinel-2] → [BurnedAreaMapper] → [EventReport] ← [FireEvent]
                                                                         ↓
                                                                [TelegramAlert]
                                                                [LeafletMap]
```

## Слои

- **adapters/** — загрузка данных из внешних API
- **services/** — бизнес-логика (детекция, фильтрация, кластеризация)
- **api/** — REST endpoints
- **core/** — схемы данных и конфигурация

## Ключевые алгоритмы

- **Кластеризация:** DBSCAN-подобный с пространственно-временным окном (3 км, 24 ч)
- **NBR/dNBR:** (B08-B12)/(B08+B12) для оценки степени поражения
- **Фильтрация:** whitelist промзон + проверка confidence + термические пороги

---

## Mathematical Models

### 1. Rothermel Fire Spread Model (1972)

Физическая модель распространения огня:

```
R = R₀ × (1 + φw + φs)

где:
- R₀ = (IR × ξ) / (ρb × ε × Qig) — базовая скорость распространения (м/с)
- φw = C × (3.281 × U)^B × (σ/δ)^-E — фактор ветра
- φs = 5.275 × β^-0.3 × (tan(slope))² × aspect_multiplier — фактор склона

Параметры топлива:
- IR: reaction intensity (кВт/м²)
- ξ: propagating flux ratio
- ρb: bulk density (кг/м³)
- σ: surface area to volume ratio (м²/м³)
- δ: fuel bed depth (м)
- β: packing ratio (0.5)

Типы топлива:
- short_grass: IR=1500, ξ=0.4, ρb=0.2, σ=3500, δ=0.3
- timber_understory: IR=2500, ξ=0.3, ρb=0.4, σ=2000, δ=0.6
- conifer_litter: IR=3000, ξ=0.35, ρb=0.3, σ=2500, δ=0.5
```

**Реализация:** `app/services/fire_physics.py`

### 2. Cellular Automata Fire Simulation

Вероятностная модель распространения на сетке:

```
Состояния клетки: 0=не горит, 1=горит, 2=сгорело, 3=вода, 4=дорога
Вероятность возгорания: p = p_base + p_wind + p_fuel

где:
- p_base = 0.3 (базовая вероятность)
- p_wind = f(alignment, wind_speed) ∈ [0, 0.4]
- p_fuel = f(fuel_type) ∈ [0, 0.3]

Шаг времени: dt = 15 минут
Размер клетки: 100×100 м
```

**Реализация:** `app/services/fire_cellular_automata.py`

### 3. Bayesian Fire Risk Assessment

Байесовская оценка вероятности пожара:

```
P(fire | evidence) = P(evidence | fire) × P(fire) / P(evidence)

Факторы likelihood:
- temp_factor = clip((temp - 15) / 20, 0, 1)
- humidity_factor = 1 - clip(humidity / 100, 0, 1)
- wind_factor = clip(wind_speed / 15, 0, 1)
- fuel_factor = {short_grass: 0.9, conifer_litter: 0.8, ...}
- slope_factor = clip(slope_deg / 30, 0, 1)

Веса факторов:
- temperature: 25%, humidity: 25%, wind: 20%, fuel: 20%, slope: 10%

Уровни риска:
- low: [0.0, 0.25)
- moderate: [0.25, 0.50)
- high: [0.50, 0.75)
- extreme: [0.75, 1.0]
```

**Реализация:** `app/services/bayesian_fire_risk.py`

### 4. Adaptive Severity Thresholds

Адаптивная классификация тяжести пожара по dNBR:

| Тип растительности | Unburned | Low | Moderate | High |
|-------------------|----------|-----|----------|------|
| conifer | (-1.0, 0.10) | (0.10, 0.27) | (0.27, 0.44) | (0.44, 1.0) |
| deciduous | (-1.0, 0.15) | (0.15, 0.35) | (0.35, 0.55) | (0.55, 1.0) |
| grass | (-1.0, 0.20) | (0.20, 0.40) | (0.40, 0.60) | (0.60, 1.0) |
| mixed | (-1.0, 0.12) | (0.12, 0.30) | (0.30, 0.50) | (0.50, 1.0) |

**Реализация:** `app/services/burned_area_mapper.py`

### 5. Multispectral Indices

Индексы для анализа пожаров:

```
NBR = (B08 - B12) / (B08 + B12)
SAVI = ((B08 - B04) / (B08 + B04 + L)) × (1 + L), L=0.5
NDWI = (B03 - B08) / (B03 + B08)
dNBR = NBR_pre - NBR_post
RdNBR = dNBR / sqrt(|NBR_pre| + ε)

Fire Risk Index = vegetation_dryness×30 + soil_dryness×25 + fuel_factor×25 + history×20
```

**Реализация:** `app/services/multispectral_indices.py`
