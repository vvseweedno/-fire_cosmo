# Соответствие конкурсному заданию

Этот документ разделяет три уровня фактов:

1. **публично подтверждённый контракт кейса**;
2. **фактически доступные материалы организатора**;
3. **внутренний competition/evaluation harness репозитория**.

Такое разделение не позволяет выдать рабочую гипотезу за требование организатора.

## 1. Публично подтверждённый контракт кейса

Кейс «Оперативный мониторинг лесных пожаров» требует сервис на данных ДЗЗ,
работающий в два этапа.

### Этап 1 — Active Fire

Публично названы тепловые наблюдения:

- MODIS;
- VIIRS;
- Landsat.

Цель — поиск очагов горения с отсевом ложных срабатываний.

Текущий реализованный physics core глубже всего поддерживает VIIRS I-band path:
I4/I5 thermal contrast, local/context anomalies, optional I1/I2/I3 context и
metric-gated hard-negative suppression.

MODIS/Landsat не должны изображаться как уже реализованные адаптеры, пока
соответствующий ingestion path не существует и не протестирован.

### Этап 2 — Burned Area / Damage

Публично указан Sentinel-2.

Цель — картирование гарей и оценка степени поражения леса.

Текущий core использует Sentinel-2 pre/post physics features (NBR, dNBR, RBR,
RdNBR и др.). Sentinel-1/SAR, land cover и terrain являются дополнительными
исследовательскими источниками и должны оставаться optional/fallback, пока их
выигрыш не измерен.

### Ожидаемый продукт

Публичный результат:

- веб-сервис или API;
- карта очагов/гарей;
- площадь гари в гектарах.

Площадь в гектарах нельзя вычислять из одного количества пикселей без
корректной геопривязки и площади пикселя.

Корректный термин по задержке: near-real-time после появления нового
спутникового наблюдения, а не непрерывное real-time наблюдение.

## 2. Фактически доступный пакет организатора

На 2026-09-18 в подключённом наборе `Мониторинг DATA/fire-aoi` наблюдается
AOI-пакет, а не готовый train/test chip dataset.

GeoJSON содержит:

- `aoi` — мониторинговую границу;
- `utm_32637` — рекомендованную полосу UTM 37N;
- `utm_32638` — рекомендованную полосу UTM 38N.

Свойства `aoi`, явно записанные организатором:

- Нижнее Поволжье и Подонье;
- EPSG:4326;
- сезоны 2019–2025;
- месяцы 04–10;
- рекомендуемые EPSG:32637 / EPSG:32638;
- заявленная площадь 435273 км².

README пакета прямо указывает, что геометрии закрытых private-test блоков не
публикуются. Репозиторий не должен пытаться их реконструировать.

Строгая локальная проверка:

```bash
python scripts/inspect_aoi.py \
  --geojson /path/to/fire_monitoring_aoi.geojson \
  --output artifacts/aoi_audit.json
```

## 3. Рабочая competition objective

В internal evaluation harness реализована композиция:

```
Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity
```

Она является целевой функцией текущего competition spec и всего metric-gated
model-selection контура.

Но публичная страница кейса и доступный AOI-пакет сами по себе эту формулу не
подтверждают. Поэтому release evidence должен отдельно зафиксировать
организаторский источник scoring rules, прежде чем формула будет называться
публично подтверждённой официальной метрикой.

Никакая локальная OOF цифра не является обещанием результата жюри/private test.

## 4. Внутренний labelled evaluation harness

Если организаторы предоставят размеченные chips/targets либо команда создаст
легальный labelled validation corpus, репозиторий поддерживает:

### AF

- binary pixel target;
- VIIRS I4/I5 mandatory for current VIIRS path;
- optional I1/I2/I3, land cover, valid mask and explicit context;
- exact threshold optimisation;
- hard-negative candidate tournament.

### BS

Internal severity representation:

- 0 — unburned;
- 1 — low;
- 2 — moderate;
- 3 — high.

Implemented physics path:

- Sentinel-2 pre/post;
- NBR/dNBR/RBR/RdNBR;
- cloud-aware masks;
- optional Sentinel-1/SAR;
- optional land-cover thresholds.

These exact class IDs remain an internal target contract until the organiser's
label semantics are explicitly verified.

## 5. Submission/RLE harness

`wildfire/rle.py` and `wildfire/submission.py` implement a strict
template-driven competition submission path.

This remains useful if a machine-scored submission format is supplied, but the
currently observed AOI package does not by itself prove that this is the final
organizer delivery contract.

## 6. Safety boundaries

Competition/private-test work must not:

- reconstruct hidden block geometry;
- infer private-test answers from coordinates/dates;
- query FIRMS/VNP14/MOD14 or other answer-producing products to manufacture
  hidden labels when prohibited;
- fabricate Sentinel scene IDs, timestamps, coordinates or hectares;
- silently guess raster band order;
- call an internal working metric “official” without source evidence.

External products and literature may be used for architecture research,
operational demos, and legally allowed public-data validation when clearly
separated from hidden-test answer generation.

## 7. Engineering principles retained

- float-safe raster math;
- contextual VIIRS physics;
- B8A/B12-consistent burn-index path;
- cloud-aware masking;
- strict channel identity;
- reproducibility;
- CI/Docker;
- dataset fingerprints;
- explicit leakage audits;
- error-diversity analysis;
- fail loudly instead of fabricating output.
