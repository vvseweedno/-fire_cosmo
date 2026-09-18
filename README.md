# Ready Prototype — КосмоХакатон 2026

Чистый competition-first прототип для кейса **«Мониторинг природных пожаров»**.

Ветка собрана заново под фактическое конкурсное задание. Старый FIRMS-first проект,
прогноз распространения, Rothermel/CA/Bayesian-модули, синтетическая «real validation»
и захардкоженные отчёты сюда не перенесены.

## Конкурсное ядро

**AF — Active Fire:** VIIRS I1..I5 + контекст → бинарная pixel-wise mask природного
активного горения.

**BS — Burn Severity:** Sentinel-2 pre/post + Sentinel-1 + SCL + terrain + land cover →
классы 0/1/2/3 = no burn / low / moderate / high.

Официальная композиция:

```
Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity
```

Evaluator использует micro-aggregation по общему пулу пикселей. Для severity-класса,
отсутствующего одновременно в GT и prediction, IoU = 1 согласно постановке.

## Что уже готово

- tolerant discovery отдельных `.npy`, `.npz`, GeoTIFF channel files;
- raw layout probe для неизвестных/stacked файлов;
- полный набор официальных channel aliases, включая S2 B5/B6/B7 и AF context;
- deterministic AF baseline;
- dNBR/SCL/land-cover-aware BS baseline;
- versioned JSON model config;
- dataset profiler;
- official metric evaluator + per-chip metrics + latency;
- deterministic group-aware split по `fire_event_id`;
- настоящий `train.py` для AF threshold calibration только на train partition;
- strict RLE: row-major, 1-based, no overlap/touching runs;
- template-driven submission по `sample_submission.csv`;
- strict submission validator;
- обязательный `inference.py`;
- FastAPI demo layer;
- Docker;
- CI: Ruff + compile + pytest + API smoke + Docker build.

## Рабочий цикл после получения официального train

```bash
python -m pip install -e ".[dev]"

# 1. Ничего не угадываем — сначала видим реальный layout.
python scripts/inspect_dataset.py \
  --data-dir /path/to/train \
  --output outputs/layout.json

# 2. После адаптации reader под фактический layout — EDA/profile.
python scripts/profile_dataset.py \
  --data-dir /path/to/train \
  --output outputs/dataset_profile.json

# 3. Один раз фиксируем split. fire_event_id не пересекает train/validation.
python scripts/make_split.py \
  --meta /path/to/train/meta.csv \
  --output splits/seed42.json \
  --validation-fraction 0.2 \
  --seed 42

# 4. Обучаем/калибруем только на train partition.
python train.py \
  --data-dir /path/to/train \
  --split-manifest splits/seed42.json \
  --output artifacts/baseline_config.json

# 5. Честно оцениваем untouched validation.
python scripts/evaluate_baseline.py \
  --data-dir /path/to/train \
  --split-manifest splits/seed42.json \
  --partition validation \
  --model-config artifacts/baseline_config.json \
  --output outputs/baseline_validation.json

# 6. Финальный inference. sample_submission.csv и meta.csv — источник истины.
python inference.py \
  --data-dir /path/to/test \
  --model-config artifacts/baseline_config.json \
  --output submission.csv

# 7. Перед сдачей обязательно.
python scripts/validate_submission.py \
  --submission submission.csv \
  --data-dir /path/to/test
```

## Правила честной разработки

Synthetic fixtures не называются real validation. Private/public test не участвует в
обучении, threshold tuning или ручной разметке. Test location/date не восстанавливаются.
FIRMS и готовые fire/burned-area products не используются для получения test answers.

Облачность не удаляется из официальной оценки. `VALID_MASK` в evaluator доступен только
как явно включаемая исследовательская ablation; официальный baseline считает все target
pixels.

Точный layout официального архива пока не зашит в код. Если файлы окажутся stacked,
`inspect_dataset.py` покажет shape/dtype/NPZ keys, после чего `wildfire/io.py`
адаптируется под фактический формат с regression fixture.

## Приоритеты

Сначала: корректный loader → B0 Score → EDA ошибок → измеримые ablations → компактная
segmentation model. Только после роста F1/IoU/mIoU усиливается сервисный слой.

См. `docs/TASK_ALIGNMENT.md` и `docs/EXPERIMENTS.md`.
