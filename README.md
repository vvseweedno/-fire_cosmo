# Wildfire Nexus Core

Хакатонный сервис мониторинга лесных пожаров на основе спутниковых данных ДЗЗ.

## Возможности

- 🔥 Детекция очагов горения (MODIS/VIIRS)
- 🗺️ Картирование гарей (Sentinel-2 NBR/dNBR)
- 📊 Кластеризация точек в события
- 🚨 Telegram-алерты с приоритизацией
- 📈 Метрики качества детекции

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

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка здоровья |
| GET | `/api/v1/fires` | Список очагов (GeoJSON) |
| POST | `/api/v1/fires/analyze` | Анализ региона |
| GET | `/api/v1/events` | Список событий |
| GET | `/api/v1/events/{id}` | Детали события |

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
