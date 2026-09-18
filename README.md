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
- raw archive inspector, strict dataset preflight и dataset profiler;
- metadata-safe split-band / multiband GeoTIFF / NPZ channel loading без угадывания порядка каналов;
- exact AF threshold optimization;
- ordered BS threshold optimization напрямую по weighted competition subscore;
- metric-safe land-cover-specific BS threshold refinement с global-score fallback;
- NBR/dNBR, RBR, RdNBR, NDVI/NDMI, NBR2, MIRBI, BAIS2, temporal/SAR deltas;
- leakage-safe grouping по fire_event_id;
- balanced group OOF folds;
- reproducible baseline OOF score generation;
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

## Финальный one-command gate

После того как organiser train/test лежат локально, весь девятишаговый протокол можно
прогнать одной командой:

```bash
python scripts/finalize_competition.py \
  --train-dir /path/to/train \
  --test-dir /path/to/test \
  --work-dir final_run
```

По умолчанию финальный gate требует organiser-provided event/group id для каждого
train-чипа. Это сделано намеренно: chip-id fallback нельзя честно называть полностью
leakage-safe. Если официальный metadata действительно не содержит группировки,
диагностический прогон возможен с `--allow-chip-fallback`, но freeze manifest явно
сохранит ослабленную гарантию.

Gate выполняет: train/test preflight, fixed folds, cross-fit метрики, SAR/spectral/
land-cover candidates, baseline-preserving promotion, два независимых повторных
cross-fit прогона, два inference прогона, strict submission validation и SHA256
freeze manifest. Финальный кандидат выбирается только по cross-fitted official Score.

## Metric-max workflow на official train

Ниже — путь, который должен пройти финальный конфиг. Он не использует
геопривязку/даты private test и не подключает готовые продукты пожаров.

```bash
python scripts/preflight_dataset.py --data-dir /path/to/train --deep
python scripts/inspect_dataset.py --data-dir /path/to/train --output outputs/inspect.json
python scripts/profile_dataset.py --data-dir /path/to/train --output outputs/profile.json

python scripts/make_folds.py \
  --meta /path/to/train/meta.csv \
  --output splits/folds_seed42.json \
  --folds 5 --seed 42

python scripts/generate_baseline_oof.py \
  --data-dir /path/to/train \
  --fold-manifest splits/folds_seed42.json \
  --output-dir outputs/oof_baseline

python scripts/evaluate_crossfit_oof.py \
  --oof-dir outputs/oof_baseline \
  --fold-manifest splits/folds_seed42.json \
  --report outputs/crossfit_oof_report.json \
  --deployment-config artifacts/crossfit_deployment_config.json \
  --bs-max-candidates 128 --bs-passes 4

python inference.py \
  --data-dir /path/to/test \
  --model-config artifacts/crossfit_deployment_config.json \
  --output submission.csv

python scripts/validate_submission.py \
  --data-dir /path/to/test \
  --submission submission.csv
```

### Multiband safety

Для stacked GeoTIFF порядок каналов не угадывается по позиции. Каналы должны быть
однозначно названы через band descriptions/tags либо через sidecar
`*.bands.json` / `*.channels.json`. Неоднозначный multiband raster вызывает
ошибку вместо тихого чтения band 1.

### Что значит «максимум по метрике»

40/40 в разделе метрики получает решение с лучшим private Score среди валидных
решений после линейной нормализации организаторов. Репозиторий поэтому
оптимизирует не «баллы из 40» напрямую, а официальный Score и одновременно
защищает валидность submission. Ни одна локальная/OOF цифра не объявляется
гарантией hidden-test результата.

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
