# Ready Prototype — КосмоХакатон 2026

Чистый competition-first прототип для кейса **«Мониторинг природных пожаров»**.

Эта ветка намеренно собрана заново. Старый FIRMS-first проект, прогноз распространения,
Rothermel/CA/Bayesian-модули, синтетическая «real validation» и захардкоженные отчёты сюда
не перенесены.

## Что решаем

### AF — Active Fire
Вход: многоканальный VIIRS-чип (I1..I5) + контекстные слои.
Выход: бинарная pixel-wise маска природного активного горения.

### BS — Burn Severity
Вход: Sentinel-2 pre/post, Sentinel-1 pre/post, SCL, DEM/рельеф и land cover.
Выход: классы 0/1/2/3 = no burn / low / moderate / high severity.

Основной конкурсный score:

```
0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity
```

## Что уже готово до выдачи датасета

- robust channel discovery для `.npy`, `.npz`, GeoTIFF;
- быстрый domain baseline для AF;
- быстрый domain baseline для BS с B8A/B12, dNBR, SCL и land-cover thresholds;
- dataset-level pixel F1 / burn IoU / severity mIoU / composite score;
- streaming evaluator с per-chip метриками и latency;
- dataset profiler с channel presence/shapes/dtypes/ranges/class counts;
- RLE row-major, 1-based;
- генерация submission с обязательными class rows;
- строгий validator submission;
- dataset inspector;
- unit tests;
- FastAPI demo layer;
- CI: ruff + compile + pytest + API smoke + Docker build;
- воспроизводимый CLI.

## Быстрый старт

```bash
python -m pip install -e ".[dev]"
pytest -q

# 1. Сначала понять фактический layout выданных данных.
python scripts/inspect_dataset.py --data-dir /path/to/train

# 2. Получить компактный EDA-профиль и class balance.
python scripts/profile_dataset.py \
  --data-dir /path/to/train \
  --output outputs/dataset_profile.json

# 3. Посчитать реальный baseline на размеченных train/validation chips.
python scripts/evaluate_baseline.py \
  --data-dir /path/to/validation \
  --output outputs/baseline_metrics.json

# 4. Финальный competition inference.
python inference.py \
  --data-dir /path/to/test \
  --output submission.csv

# 5. Структурная проверка submission/RLE перед сдачей.
python scripts/validate_submission.py \
  --submission submission.csv \
  --data-dir /path/to/test
```

## Принцип оценки

Мы не называем synthetic fixtures «реальной валидацией».
`evaluate_baseline.py` считает метрики только по фактически найденным `TARGET`-маскам и
сохраняет агрегированные confusion counts, class-wise IoU, per-chip ошибки и latency.

Если в официальном датасете target/channel names отличаются, сначала обновляется reader
под фактическую схему, и только после этого фиксируется baseline.

## Важное ограничение

До получения реальных файлов мы не притворяемся, что знаем точную структуру каталога.
Reader сделан tolerant к распространённым именам каналов. После появления официального
датасета первым шагом должен быть запуск `inspect_dataset.py`, после чего aliases/layout
фиксируются под фактический формат.

Competition inference не должен использовать FIRMS, готовые fire/burned-area products,
восстановление координат/дат test chips или поиск исходных сцен private test.

## Стратегия после получения данных

1. Зафиксировать фактический layout и официальный target format.
2. Построить dataset profile и проверить class imbalance / invalid pixels.
3. Зафиксировать train/validation split без leakage.
4. Посчитать B0 baseline и сохранить JSON с метриками и latency.
5. Делать EDA FP/FN и severity confusion.
6. Улучшать только через измеримые ablations на неизменном split.
7. ML-модели добавлять поверх текущего интерфейса, не ломая `inference.py` и submission.
8. Сервис — демонстрационный слой; competition inference остаётся отдельным и быстрым.

См. `docs/TASK_ALIGNMENT.md` и `docs/EXPERIMENTS.md`.
