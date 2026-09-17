# Wildfire Nexus — two-stage wildfire monitoring

Хакатонный веб-сервис для кейса **«Оперативный мониторинг лесных пожаров»**:

1. активные очаги: MODIS / VIIRS (NASA FIRMS) → нормализация → фильтрация → пространственно-временные события;
2. последствия: Sentinel-2 L2A → B08/B12 + SCL → NBR → dNBR → классы поражения → GeoJSON + площадь в гектарах.

Главный принцип ветки: **никаких вымышленных scene IDs, площадей или «успешных» результатов при отсутствии данных**. Offline-режим — явно маркированная фикстура; online-режим сохраняет STAC provenance реальных сцен и raster assets.

## Почему это соответствует кейсу

Официальное описание Красноярского КосмоХакатона требует двух этапов: поиск очагов по MODIS/VIIRS/Landsat с отсевом ложных срабатываний и картирование гарей по Sentinel-2 с оценкой поражения леса. Результат — веб-сервис/API с картой и площадью гари. На мастер-классе организаторов используется та же логика: NASA FIRMS → dNBR Sentinel-2 → гектары.

Ссылки:
- https://xn--80aa2abijcbdyq6a.xn--p1ai/krasnoyarsk/
- https://firms.modaps.eosdis.nasa.gov/api/area/
- https://earth-search.aws.element84.com/v1
- https://fire.trainhub.eumetsat.int/docs/figure5678_Sentinel-2.html

## Что реально работает

### Этап 1 — очаги

- `POST /api/v1/fires/analyze` получает и нормализует термоточки.
- Offline: воспроизводимые MODIS/VIIRS fixtures вокруг Красноярска.
- Online: NASA FIRMS Area API при `FIRMS_MAP_KEY`.
- Фильтр сейчас честно ограничен проверками координат, confidence, thermal/FRP и опциональным whitelist промышленных источников.
- Пространственно-временная кластеризация сохраняет события для следующих API-вызовов.

### Этап 2 — гарь

**Offline (`OFFLINE_MODE=true`)**
- локальная `dNBR.json` фикстура;
- площадь считается по пикселям классов low/moderate/high;
- provenance явно сообщает, что это demo fixture.

**Online (`OFFLINE_MODE=false`)**
- публичный STAC Sentinel-2 L2A;
- сохраняются scene ID, datetime, cloud cover, tile, item URL и B08/B12/SCL asset URLs;
- пара до/после выбирается с предпочтением одного MGRS tile;
- B12 (~20 m) используется как общая сетка; B08 перепроецируется на неё;
- SCL маскирует no-data, defective, cloud shadow, cloud, cirrus, snow/ice;
- `dNBR = NBR_pre - NBR_post`;
- GeoJSON строится из реального severity raster и переводится в EPSG:4326;
- площадь = число affected pixels × площадь пикселя в проекции Sentinel-2;
- если корректной пары нет или raster processing не удался, событие получает `burn_mapping_unavailable`, а не фальшивый результат.

## Быстрый старт — гарантированное демо

```bash
cp .env.example .env
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте `http://localhost:8000`, bbox демо: `91.80,55.85,92.20,56.15`.

Docker:

```bash
docker compose up --build
```

## Online mode с реальными данными

В `.env`:

```env
OFFLINE_MODE=false
FIRMS_MAP_KEY=<your NASA FIRMS MAP_KEY>
SENTINEL2_STAC_API=https://earth-search.aws.element84.com/v1
MAX_CLOUD_COVER=20
PRE_IMAGE_DAYS_BEFORE=90
POST_IMAGE_DAYS_AFTER=30
```

Запрос:

```bash
curl -X POST http://localhost:8000/api/v1/fires/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "bbox": [91.8, 55.85, 92.2, 56.15],
    "start_date": "2026-07-15",
    "end_date": "2026-07-16",
    "sensors": ["MODIS", "VIIRS"]
  }'
```

Для каждого события ответ показывает `status`, площадь, severity summary и processing provenance. Полный отчёт: `GET /api/v1/events/{event_id}/report`.

## Проверка

```bash
pytest -q
```

Добавлены тесты на STAC asset mapping, provenance, float-safe NBR, SCL mask и severity thresholds. Workflow `.github/workflows/ci.yml` предназначен для воспроизводимого запуска на Python 3.11.

> Последний подтверждённый baseline **до** нового real-Sentinel path: 155 tests passed. После текущих изменений нельзя считать этот результат новым CI-доказательством, пока workflow фактически не выполнен.

## Что показывать жюри

1. **Один сквозной event:** FIRMS point → причины прохождения фильтра → event → две Sentinel-2 сцены → dNBR → контур → гектары.
2. **Trust card:** scene IDs, даты, облачность, asset URLs, grid, cloud mask, valid/affected pixels.
3. **Fail honestly:** отсутствие подходящей сцены отображается как `burn_mapping_unavailable`.
4. **Одинаковые цифры:** площадь в API рассчитывается из того же raster mask, из которого строится GeoJSON.
5. **Метрики, а не заявления:** precision/recall/F1 для фильтра термоточек; IoU/Dice + area error для гарей на независимой разметке.

## Что ещё нужно закрыть до защиты

- Подготовить 2–3 **реальных** кейса (желательно Сибирь) и сохранить точные scene IDs/даты.
- Добавить независимую reference-разметку гарей и посчитать IoU/Dice/ошибку площади.
- Для формулировки «поражение леса» добавить лесную маску/land-cover: текущая severity относится к наблюдаемой территории, а не доказанно только к лесу.
- Валидировать эвристику false-positive filter на negative controls; не называть все удалённые точки ложными без разметки.
- Landsat остаётся резервным/экспериментальным адаптером; не заявлять полноценную глобальную поддержку, пока thermal raster path не завершён.

## Архитектурное преимущество для защиты

**Проверяемость результата.** У каждого mapped event есть цепочка происхождения данных, а система различает три состояния: обнаружен очаг, гарь картирована, картирование пока недоступно. Это позволяет отвечать на главный вопрос жюри: *«почему вашим гектарам можно доверять?»*
