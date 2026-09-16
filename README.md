# Wildfire Nexus Core

**Хакатонный сервис ДЗЗ для поиска очагов горения и картирования гарей**

## 📋 Соответствие задаче хакатона

Сервис реализует двухэтапный анализ:

1. **Поиск очагов горения** по тепловым каналам (MODIS, VIIRS) с отсевом ложных срабатываний
2. **Картирование гарей** по снимкам Sentinel-2 с оценкой степени поражения леса

## 🏗 Архитектура

```
app/
├── main.py                 # FastAPI приложение
├── api/
│   └── routes/
│       ├── fires.py        # Эндпоинты поиска пожаров
│       ├── events.py       # Эндпоинты событий
│       ├── report.py       # Генерация отчетов
│       └── health.py       # Health check
├── core/
│   ├── config.py           # Настройки
│   └── schemas.py          # Pydantic схемы
├── services/
│   ├── fire_detection.py   # Детекция очагов
│   ├── false_positive_filter.py  # Фильтр ложных
│   ├── fire_clustering.py  # Кластеризация в события
│   └── burned_area_mapper.py     # Картирование гарей (NBR/dNBR)
├── adapters/
│   ├── modis_adapter.py    # MODIS данные
│   ├── viirs_adapter.py    # VIIRS данные
│   └── sentinel2_adapter.py # Sentinel-2 STAC
└── static/
    └── index.html          # Веб-карта (Leaflet)
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
# Через uv (рекомендуется)
uv pip install -e ".[dev]"

# Или через pip
pip install -e ".[dev]"
```

### Запуск сервиса

```bash
# Копирование .env
cp .env.example .env

# Запуск
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Или откройте http://localhost:8000 для веб-интерфейса.

## 🔌 API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка здоровья |
| GET | `/api/v1/fires` | Список очагов (GeoJSON) |
| POST | `/api/v1/fires/analyze` | Анализ региона |
| GET | `/api/v1/events` | Список событий |
| GET | `/api/v1/events/{id}` | Детали события |
| GET | `/api/v1/events/{id}/burned.geojson` | Полигон гари |
| GET | `/api/v1/events/{id}/report` | JSON отчет |
| GET | `/api/v1/report/{id}` | HTML отчет |

## 📊 Источники данных

| Источник | Тип | Разрешение |
|----------|-----|------------|
| NASA FIRMS | MODIS | 1 км |
| NASA FIRMS | VIIRS | 375 м |
| Sentinel-2 L2A | Multispectral | 10-20 м |

## 🔧 Переменные окружения

См. `.env.example`:
- `FIRMS_MAP_KEY` - API ключ NASA FIRMS
- `OFFLINE_MODE` - Режим работы с фикстурами
- `SENTINEL2_STAC_API` - URL STAC API

## 🧪 Тесты

```bash
pytest tests/ -v
```

## ⚠️ Ограничения

1. При отсутствии API ключей используются фикстурные данные
2. Landsat адаптер требует дополнительной реализации
3. Задержка данных FIRMS ~3-4 часа

---

**Команда хакатона | 2024**
