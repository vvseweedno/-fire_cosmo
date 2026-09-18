# Ready Prototype — КосмоХакатон 2026

Competition-first решение для кейса «Мониторинг природных пожаров».

Ветка собрана заново под фактическую конкурсную метрику. Старый FIRMS-first проект, прогноз распространения, псевдо-real validation и декоративные математические модули в competition core не перенесены.

## Целевая функция

Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity

AF: VIIRS I1..I5 + контекст → бинарная pixel-wise mask.
BS: Sentinel-2 pre/post + Sentinel-1 + SCL + terrain + land cover → 0/1/2/3 severity.

## Что готово

- официальный micro-metric evaluator;
- strict template-driven RLE/submission;
- raw archive inspector и dataset profiler;
- полный channel schema;
- deterministic AF/BS physics baseline;
- exact AF threshold search по micro-F1;
- ordered BS threshold search напрямую по 0.35*IoU_burn + 0.30*mIoU_severity;
- NBR/dNBR, RBR, RdNBR, NDVI/NDMI, NBR2, MIRBI, BAIS2, temporal/SAR deltas;
- leakage-safe split по fire_event_id;
- deterministic balanced group OOF folds;
- versioned model config;
- reproducible train.py / inference.py;
- CI: Ruff + compile + pytest + API smoke + Docker.

## Основной workflow после получения train

1. scripts/inspect_dataset.py — подтвердить физический layout и scaling.
2. scripts/profile_dataset.py — class balance/ranges/nodata.
3. scripts/make_split.py — быстрый fixed holdout.
4. scripts/make_folds.py — 5 leakage-safe group OOF folds.
5. train.py — metric-aware calibration/training только на train.
6. evaluate_baseline.py — untouched validation.
7. pooled OOF model selection and ensemble calibration.
8. inference.py → submission.csv.
9. validate_submission.py перед сдачей.

## Главный принцип

Ни одна научная идея не считается улучшением до положительного delta Score_OOF на неизменных leakage-safe folds.
Private/public test не используется для training/tuning/manual labeling. Координаты и даты private test не восстанавливаются. FIRMS и готовые fire/burned-area products не используются для test answers.

Точный layout/масштаб официального архива не угадывается заранее: сначала inspector, потом адаптация reader с regression fixture.

## Accuracy roadmap

- docs/RESEARCH_2026.md — научные источники → проверяемые гипотезы;
- docs/ACCURACY_ARCHITECTURE.md — архитектура максимизации Score;
- docs/EXPERIMENTS.md — журнал ablations;
- docs/TASK_ALIGNMENT.md — соответствие постановке.
