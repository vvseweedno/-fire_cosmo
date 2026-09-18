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
- pixel F1 / burn IoU / severity mIoU / composite score;
- RLE row-major, 1-based;
- генерация submission с обязательными class rows;
- строгий validator submission;
- dataset inspector;
- unit tests;
- воспроизводимый CLI.

## Быстрый старт

```bash
python -m pip install -e ".[dev]"
pytest -q

python scripts/inspect_dataset.py --data-dir /path/to/train

python inference.py \
  --data-dir /path/to/test \
  --output submission.csv

python scripts/validate_submission.py \
  --submission submission.csv \
  --data-dir /path/to/test
```

## Важное ограничение

До получения реальных файлов мы не притворяемся, что знаем точную структуру каталога.
Reader сделан tolerant к распространённым именам каналов. После появления официального
датасета первым шагом должен быть запуск `inspect_dataset.py`, после чего aliases/layout
фиксируются под фактический формат.

## Стратегия после получения данных

1. Зафиксировать официальный train/validation split без leakage.
2. Посчитать baseline score раздельно для AF и BS.
3. Сделать EDA ошибок, class imbalance, cloud/valid fraction, land-cover distribution.
4. Улучшать только через измеримые ablations.
5. ML-модели добавлять поверх этого интерфейса, не ломая `inference.py` и submission.
6. Сервис — демонстрационный слой; competition inference остаётся отдельным и быстрым.

См. `docs/TASK_ALIGNMENT.md` и `docs/EXPERIMENTS.md`.
