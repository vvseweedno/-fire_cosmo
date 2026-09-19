# Ready Prototype — КосмоХакатон 2026

Competition-first решение для кейса «Оперативный мониторинг лесных пожаров».

Публичный контракт кейса — двухэтапный сервис: тепловые ДЗЗ-наблюдения
(MODIS/VIIRS/Landsat) для поиска активного горения с отсевом ложных
срабатываний, затем Sentinel-2 для картирования гарей и оценки поражения.
Результат — веб-сервис/API с картой и площадью гари.

Ветка сохраняет metric-driven principle: новая идея остаётся в accuracy core
только после честного leakage-safe измерения.

## Рабочая evaluation objective

Текущий competition spec репозитория использует:

Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity

Эта формула остаётся основной внутренней целью model-selection, но публичная
страница кейса и доступный AOI-пакет сами по себе её не подтверждают. Перед
публичным названием формулы «официальной» источник scoring rules должен быть
зафиксирован в release evidence.

AF: реализованный VIIRS I1..I5 contextual core → binary fire mask.

BS: реализованный Sentinel-2 pre/post physics core + optional
Sentinel-1/SCL/terrain/land-cover context → internal 0/1/2/3 severity mask.

## Фактически доступный organizer AOI

В подключённом пакете `Мониторинг DATA/fire-aoi` обнаружен AOI для
самостоятельной работы со спутниковыми наблюдениями, а не готовый размеченный
train/test chip dataset. Строгая проверка локальной копии:

```bash
python scripts/inspect_aoi.py --geojson fire_monitoring_aoi.geojson \
  --output artifacts/aoi_audit.json
```

Private-test block geometry намеренно отсутствует и не реконструируется.

## Текущий accuracy-core

- единый canonical evaluator для рабочей composite metric;
- strict template-driven RLE/submission;
- raw archive inspector, strict dataset preflight и dataset profiler;
- metadata-safe split-band / multiband GeoTIFF / NPZ channel loading без угадывания порядка каналов;
- exact AF threshold optimization;
- metric-gated AF hard-negative candidates, включая optional persistent-heat recurrence prior;
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
- deterministic dataset fingerprint + explicit leakage audit;
- machine-readable experiment registry;
- strict organizer/public AOI parser + truthful operational sensor capabilities;
- CRS-safe burned-area hectares for projected metric rasters;
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

## Что остаётся гипотезой до размеченной official/organizer validation

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

## Рабочий цикл после получения размеченных organizer/evaluation данных

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

## Финальный release gate

Для финальной сборки при наличии подтверждённых labelled train/test запускается один воспроизводимый
pipeline:

```bash
python scripts/finalize_competition.py \
  --train-dir /path/to/train \
  --test-dir /path/to/test \
  --work-dir final_run
```

Он выполняет deep preflight официальных train/test, фиксированные organiser-group
folds, cross-fitted `F1_AF` / `IoU_burn` / `mIoU_severity`, независимо
проверяет BASE-anchored AF и BS ensembles, оставляет только измеренное улучшение,
строит event-level paired bootstrap на fold-holdout predictions, дважды повторяет
fit, дважды запускает inference, дважды валидирует submission и требует
одинаковый SHA256 результата.

Сам `finalize_competition.py` намеренно **не имеет права объявлять результат
PROVEN**. После него должен существовать `final_run/artifacts/release_evidence.json`,
а окончательный статус вычисляется отдельным verifier:

```bash
python scripts/verify_proven_release.py \
  --evidence final_run/artifacts/release_evidence.json \
  --workflow-run-id <GITHUB_ACTIONS_RUN_ID> \
  --workflow-sha <EXACT_SOURCE_COMMIT_SHA> \
  --workflow-conclusion success \
  --output final_run/artifacts/final_validation.json
```

Инженерный контракт закрытия:

```text
workflow_conclusion = success
AND workflow_sha = source_commit_sha
AND workflow_run_id is attached
AND labelled-data preflight = pass
AND strict organiser event grouping = pass
AND explicit leakage audit = PASS
AND cross-fit metrics exist
AND final Score > baseline Score + epsilon
AND bootstrap/stability acceptable
AND repro run 1 ~= repro run 2
AND submission validator = pass
AND submission run 1 SHA256 == submission run 2 SHA256
```

Только если **каждый** пункт истинен, `final_validation.json` получает
`"status": "PASS"` и `"proven": true`. В противном случае статус — FAIL.

По умолчанию final pipeline требует organiser-provided event/group id для каждого
train-chip. Опция `--allow-chip-fallback` оставлена только как диагностический
режим: она может позволить сделать рабочий run, но strict release verifier
запрещает статус PROVEN при `chip_fallbacks > 0`.

Финальные артефакты:
- `final_run/artifacts/final_model_config.json`;
- `final_run/submission.csv`;
- `final_run/bootstrap_final_vs_baseline.json`;
- `final_run/artifacts/freeze_manifest.json`;
- `final_run/artifacts/data_audit.json`;
- `final_run/artifacts/leakage_audit.json`;
- `final_run/artifacts/experiments/<experiment_id>.json`;
- `final_run/artifacts/release_evidence.json`;
- `final_run/artifacts/final_validation.json`.

Короткий Makefile-путь для последней проверки:

```bash
make preflight-train TRAIN_DIR=/path/to/train WORK_DIR=final_run
make preflight-test TEST_DIR=/path/to/test WORK_DIR=final_run
make smoke-inference TEST_DIR=/path/to/test MODEL_CONFIG=final_run/artifacts/final_model_config.json WORK_DIR=final_run
make release-gate EVIDENCE=final_run/artifacts/release_evidence.json WORK_DIR=final_run
```

## Metric-max workflow на подтверждённых labelled data

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

# Дополнительный BS search: BASE остаётся legal fallback.
python scripts/generate_bs_candidate_oof.py \
  --data-dir /path/to/train \
  --fold-manifest splits/folds_seed42.json \
  --model-config configs/baseline.json \
  --output-root outputs/oof_bs_candidates

python scripts/evaluate_crossfit_bs_candidates.py \
  --candidate-root outputs/oof_bs_candidates \
  --fold-manifest splits/folds_seed42.json \
  --base-config configs/baseline.json \
  --output outputs/crossfit_bs_candidate_ensemble.json

# Важно: cross-fitted candidate comparison получает только frozen baseline
# config. Pooled OOF thresholds нельзя подавать обратно в fold-wise evaluation,
# иначе holdout labels попадут в optimizer initialization.

# Выполнять promotion только если promotion_allowed=true.
python scripts/optimize_bs_candidate_ensemble.py \
  --candidate-root outputs/oof_bs_candidates \
  --base-config artifacts/crossfit_deployment_config.json \
  --output-config artifacts/final_metric_config.json \
  --output-report outputs/bs_ensemble_report.json

python inference.py \
  --data-dir /path/to/test \
  --model-config artifacts/final_metric_config.json \
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

### Что значит «максимум по рабочей метрике»

При наличии подтверждённого machine-scored evaluation репозиторий оптимизирует
рабочий composite Score, а не визуальную привлекательность модели. Пока
организаторский scoring contract не приложен к release evidence, никакая
локальная/OOF цифра не объявляется официальным баллом или гарантией результата
жюри/private test.

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
- docs/EXPERIMENT_PROTOCOL.md
- docs/ERROR_ANALYSIS.md
- docs/OFFICIAL_DATA_CONTRACT.md
- docs/ORGANIZER_DATA_EVIDENCE.md
- docs/OPERATIONAL_ARCHITECTURE.md
- docs/ZIP_MATH_TRANSFER.md
- docs/ACCURACY_GUARANTEES.md
- docs/ARCHITECTURE.md

## Official role-based data adapter

The loader accepts the official role-based GeoTIFF layout directly. The
filename identifies the chip and observation role; fixed physical channel
orders are used only for the documented stacks:

```text
AF_<chip>_VIIRS_I1-I5.tif
AF_<chip>_AUX.tif
BS_<chip>_Sentinel-2_pre.tif
BS_<chip>_Sentinel-2_post.tif
BS_<chip>_Sentinel-1_pre.tif
BS_<chip>_Sentinel-1_post.tif
BS_<chip>_AUX.tif
<chip>_mask.tif / <chip>_target.tif
```

VIIRS I1-I5 and Sentinel-1 VV/VH use their documented order. Sentinel-2
supports the compact B8A/B12 fixture and the nine- or ten-band optical stacks;
an unlabelled stack with another band count fails preflight. Ambiguous auxiliary
multiband rasters still require band descriptions or an explicit sidecar.

Run the hard checks before any model experiment:

```bash
python scripts/preflight_dataset.py --data-dir /path/to/train --mode train --deep
python scripts/preflight_dataset.py --data-dir /path/to/test --mode test --deep
python inference.py --data-dir /path/to/test --output submission.csv
python scripts/validate_submission.py --data-dir /path/to/test --submission submission.csv
```

The validator checks exact template pair order, UTF-8 CSV structure, canonical
row-major 1-based RLE, bounds, round-trip encoding, and mutually exclusive BS
classes. Inference errors fail loudly with the chip id and source files.

## Runtime and readiness evidence

Measure the real production command, including process startup, model loading,
preprocessing, prediction, RLE, and CSV close:

```bash
python scripts/benchmark_inference.py \
  --data-dir /path/to/test \
  --output-dir final_run/artifacts/benchmark_inference \
  --output final_run/artifacts/benchmark_inference.json
python scripts/readiness_scorecard.py \
  --work-dir final_run \
  --output final_run/artifacts/readiness_scorecard.json
python scripts/reproduce_final.py \
  --work-dir final_run \
  --output release/final_manifest.json
```

The scorecard reports `NOT READY` while labelled evidence, exact-source CI,
submission hashes, or runtime measurements are absent. It does not convert
historical metrics into release evidence.

## Offline service demo

The service runs without external tiles or network data and reads the synthetic
catalog in `service/demo_results.geojson` by default. Set
`WILDFIRE_RESULTS_GEOJSON` to an explicit result catalog for a real run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/summary
curl 'http://localhost:8000/api/query?bbox=37.5,55.7,37.8,55.9&start_date=2026-07-01&end_date=2026-07-31'
curl -OJ http://localhost:8000/api/export/geojson
```

`POST /api/query` accepts a polygon or bbox and a date interval. `/map` is a
dependency-free offline map view. Area values are accepted only with explicit
projected-raster pixel metadata (`pixel_count` and `pixel_area_m2`).

Research, presentation, defense, compliance, and license sources live in:
`docs/REPORT.md`, `docs/PRESENTATION.md`, `docs/DEFENSE_QA.md`,
`docs/DATA_COMPLIANCE.md`, and `THIRD_PARTY_LICENSES.md`.
