# Experiment log

Никаких «улучшили модель» без одинакового split и численного результата.

## До получения данных

Готовы:
- dataset discovery;
- RLE/submission;
- dataset-level competition metrics;
- deterministic AF baseline;
- deterministic BS baseline;
- streaming baseline evaluator;
- dataset profiler;
- per-chip metrics + latency;
- CI/Docker.

## Первый час после получения train

1. Запустить `scripts/inspect_dataset.py`.
2. Запустить `scripts/profile_dataset.py`.
3. Зафиксировать реальные channel names/layout.
4. Проверить shapes/dtypes/nodata/ranges и class counts.
5. Найти официальный target layout.
6. Проверить, есть ли event/group metadata для split без leakage.
7. Зафиксировать split один раз и не менять его между ablations.
8. Запустить `scripts/evaluate_baseline.py` на validation.
9. Сохранить `outputs/baseline_metrics.json` как B0.

## Что считаем доказательством улучшения

Для каждого изменения должны быть:
- одинаковый validation split;
- F1 AF;
- IoU burn;
- mIoU severity;
- composite Score;
- latency;
- per-chip regressions;
- краткое объяснение, почему компонент оставлен или удалён.

## EDA

AF:
- fire pixels / all valid pixels;
- positive/negative chips;
- I4/I5 distributions по target;
- I4-I5 или нормализованный contrast;
- land-cover distribution fire vs non-fire;
- FP по built-up/water/bare;
- FN по слабым/маленьким очагам;
- worst chips по F1.

BS:
- class balance 0/1/2/3;
- dNBR distribution по severity и land-cover;
- SCL invalid/cloud fraction;
- pre/post seasonal shift;
- SAR delta by severity;
- slope/aspect distribution ошибок;
- worst chips по burn IoU и severity mIoU.

## Обязательные ablations

| ID | Изменение | F1 AF | IoU burn | mIoU sev | Score | Latency | Решение |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | deterministic baseline | PENDING | PENDING | PENDING | PENDING | PENDING | baseline candidate |
| A1 | AF without land-cover prior | PENDING | — | — | PENDING | PENDING | pending labelled train |
| A2 | AF I4/I5 + spatial context | PENDING | — | — | PENDING | PENDING | pending labelled train |
| A3 | AF threshold calibration | PENDING | — | — | PENDING | PENDING | pending labelled train |
| B1 | global dNBR thresholds | — | PENDING | PENDING | PENDING | PENDING | pending labelled train |
| B2 | land-cover-aware thresholds | — | PENDING | PENDING | PENDING | PENDING | pending labelled train |
| B3 | + Sentinel-1 support | — | PENDING | PENDING | PENDING | PENDING | pending labelled train |
| B4 | + terrain context | — | PENDING | PENDING | PENDING | PENDING | pending labelled train |
| M1 | compact segmentation model | PENDING | PENDING | PENDING | PENDING | PENDING | hypothesis only |

## ML direction после baseline

AF:
- compact U-Net/FPN-like segmentation;
- focal/Tversky/Dice-aware loss;
- hard-negative mining на industrial/built-up hotspots;
- threshold calibration по validation F1;
- oversampling positive chips только внутри train.

BS:
- multi-branch pre/post optical features;
- explicit dNBR/RdNBR as channels;
- SCL mask;
- land-cover embedding/one-hot;
- optional SAR branch;
- terrain features только после ablation;
- weighted CE + Dice/Tversky;
- class-wise postprocessing только если validation supports it.

## Leakage checklist

Перед каждым экспериментом:
- test chips не участвуют в tuning;
- private test location/date не восстанавливаются;
- запрещённые готовые fire/burned-area products не используются для test answer;
- validation split не меняется ради улучшения цифры;
- preprocessing fitted only on train;
- threshold calibration only on validation, не на test.

## Правило остановки

Компонент остаётся только если:
- даёт reproducible gain;
- не использует prohibited test information;
- укладывается в inference time;
- его можно объяснить в отчёте.
