# Соответствие конкурсному заданию

Этот документ фиксирует, под что именно оптимизирована ветка `ready-prototype`.

## 1. Active Fire (AF)

Цель: pixel-wise бинарная сегментация природного активного горения.

Используемые организаторами входы, под которые подготовлен reader/interface:

- VIIRS I1, I2, I3, I4, I5;
- land cover;
- DEM;
- геометрия наблюдения;
- метеоконтекст;
- valid mask.

Ключевой риск — экстремальный class imbalance. Поэтому accuracy не является целевой
метрикой; baseline и будущие модели оцениваются pixel F1.

В текущем deterministic baseline:
- I4 — основной thermal signal;
- I5 — thermal background;
- spatial anomaly усиливает локальные горячие пиксели;
- I3 используется как слабый контекстный штраф;
- built-up/water/snow land-cover classes используются только как priors.

Это baseline, а не заявление о достигнутой конкурсной точности.

## 2. Burn Severity (BS)

Цель: pixel-wise маска:
- 0 — no burn;
- 1 — low;
- 2 — moderate;
- 3 — high.

Интерфейс ожидает:
- Sentinel-2 pre/post: B2/B3/B4/B8A/B11/B12/SCL;
- Sentinel-1 VV/VH pre/post;
- DEM/slope/aspect;
- land cover.

Baseline использует:
- NBR = (B8A - B12) / (B8A + B12);
- dNBR = NBR_pre - NBR_post;
- SCL для исключения invalid/cloud/shadow/water/snow;
- land-cover-aware severity thresholds;
- небольшой SAR-support только как модификатор, а не источник burn сам по себе.

## 3. Метрика

В коде реализована композиция:

```
Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity
```

Метрики считаются по pixel masks. Отсутствующий в pred и GT severity class не должен
искусственно увеличивать mIoU.

## 4. Submission contract

`wildfire/rle.py`:
- row-major;
- 1-based starts;
- `start length`;
- пустой класс → пустая строка;
- строгая проверка диапазонов и пересечений.

`wildfire/submission.py`:
- AF: class_id=1;
- BS: class_id=1,2,3;
- для каждого chip/class обязательна строка.

Точный ожидаемый count проверяется после того, как reader увидит официальный test layout.

## 5. Запреты, которые соблюдает архитектура

Competition inference не обращается к:
- FIRMS;
- MOD14/MYD14/VNP14;
- готовым burned-area products;
- геокодированию test chips;
- поиску даты/исходной сцены test chip.

Исторические внешние данные можно исследовать отдельно только если это разрешено
регламентом и не используется для восстановления private test.

## 6. Что специально не перенесено из старых веток

- fake HTML reports;
- synthetic data, называемые real validation;
- Rothermel/Cellular Automata/Bayesian как центральная архитектура;
- FIRMS-first detection как конкурсный AF inference;
- hardcoded burned polygons/areas;
- выдуманные Sentinel scene IDs;
- tracked pycache.

## 7. Что перенесено как инженерные принципы

- float-safe raster math;
- B8A/B12 согласованный burn-index path;
- SCL-aware masking;
- reproducibility;
- CI/Docker;
- auditability;
- явное разделение real validation и demo;
- fail loudly instead of fabricating output.
