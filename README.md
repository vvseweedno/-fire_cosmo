# Ready Prototype — КосмоХакатон 2026

Competition-first решение для кейса «Мониторинг природных пожаров».

Ветка строится вокруг официальной метрики и принципа: новая идея остаётся в
competition core только после честного leakage-safe измерения.

## Целевая функция

Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity

AF: VIIRS I1..I5 + контекст → binary fire mask.

BS: Sentinel-2 pre/post + Sentinel-1 + SCL + terrain + land cover →
0/1/2/3 burn severity.

## Текущий accuracy-core

- официальный micro-metric evaluator;
- strict template-driven RLE/submission;
- raw archive inspector и dataset profiler;
- exact AF threshold optimization;
- ordered BS threshold optimization напрямую по weighted competition subscore;
- NBR/dNBR, RBR, RdNBR, NDVI/NDMI, NBR2, MIRBI, BAIS2, temporal/SAR deltas;
- leakage-safe grouping по fire_event_id;
- balanced group OOF folds;
- pooled OOF calibration для финального deployment;
- cross-fitted OOF evaluation для честного сравнения архитектур;
- cloud-aware optical/SAR fallback с baseline-anchor cloud_sar_weight=0;
- event-level paired bootstrap для A/B ablations;
- monotonic greedy convex OOF ensemble search;
- versioned model config;
- reproducible train.py / inference.py;
- CI: Ruff + compile + pytest + API smoke + Docker.

## Что математически гарантировано конструкцией

На одном и том же OOF-пуле:

1. Cloud/SAR calibration не обязана ухудшать baseline, потому что в candidate
   set всегда присутствует cloud_sar_weight=0 — старое поведение.

2. Greedy ensemble не обязан ухудшать текущий ансамбль, потому что на каждом
   шаге alpha=1 сохраняет его неизменным.

3. Cross-fitted OOF не использует labels текущего holdout fold для настройки
   его thresholds.

Эти свойства относятся к оптимизационному/валидационному протоколу. Они не
являются обещанием hidden-test результата.

## Что остаётся гипотезой до official train

- learned cloud-aware optical/SAR/context router;
- bi-temporal attention U-Net;
- ordinal severity head P(y>=1), P(y>=2), P(y>=3);
- Lovasz/Dice/Focal/connectivity losses;
- EMA weights;
- Prithvi-EO-2.0 / LoRA / freeze-then-unfreeze;
- hard-negative mining;
- TTA;
- quantization.

Каждый пункт обязан пройти cross-fitted OOF и event-level bootstrap.

## Рабочий цикл после получения official train

1. scripts/inspect_dataset.py — layout, scaling, dtype, channel ordering.
2. scripts/profile_dataset.py — class balance, ranges, nodata, SCL/cloud stats.
3. scripts/make_folds.py — fixed leakage-safe group folds.
4. Train candidate model on each fold.
5. Save OOF score/logit maps.
6. scripts/evaluate_crossfit_oof.py — основной model-comparison Score.
7. scripts/compare_oof_experiments.py — paired fire-event bootstrap.
8. scripts/optimize_oof_ensemble.py — только для прошедших моделей.
9. Pooled OOF calibration — финальные thresholds/weights.
10. Freeze everything.
11. inference.py → submission.csv.
12. scripts/validate_submission.py.
13. Full-test latency benchmark.

## Исследовательские ориентиры

- NASA VIIRS contextual active-fire detection;
- Sentinel-1 + Sentinel-2 cloud-robust fusion;
- bi-temporal burned-area segmentation;
- connectivity-aware burn-scar segmentation;
- Prithvi-EO-2.0 wildfire scar / burn intensity benchmarks.

См.:
- docs/RESEARCH_2026.md
- docs/ACCURACY_ARCHITECTURE.md
- docs/CLOUD_AWARE_FUSION.md
- docs/ENSEMBLE_OPTIMIZATION.md
- docs/VALIDATION_PROTOCOL.md
- docs/ZIP_MATH_TRANSFER.md
- docs/ACCURACY_GUARANTEES.md
