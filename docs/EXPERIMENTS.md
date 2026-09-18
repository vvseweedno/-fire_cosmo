# Experiment log

Никаких «улучшили модель» без одинакового split и численного результата.

## До получения данных

Готовы:
- dataset discovery;
- RLE/submission;
- metrics;
- deterministic AF baseline;
- deterministic BS baseline;
- CI.

## Первый час после получения train

1. Запустить `scripts/inspect_dataset.py`.
2. Зафиксировать реальные channel names/layout.
3. Проверить shapes/dtypes/nodata/ranges.
4. Найти официальный target layout.
5. Сделать deterministic split по chip/event группам без spatial/temporal leakage.
6. Получить baseline score.

## EDA

AF:
- fire pixels / all valid pixels;
- positive/negative chips;
- I4/I5 distributions по target;
- I4-I5 или нормализованный contrast;
- land-cover distribution fire vs non-fire;
- FP по built-up/water/bare;
- FN по слабым/маленьким очагам.

BS:
- class balance 0/1/2/3;
- dNBR distribution по severity и land-cover;
- SCL invalid/cloud fraction;
- pre/post seasonal shift;
- SAR delta by severity;
- slope/aspect distribution ошибок.

## Обязательные ablations

| ID | Изменение | F1 AF | IoU burn | mIoU sev | Score | Latency | Решение |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | deterministic baseline | TBD | TBD | TBD | TBD | TBD | baseline |
| A1 | AF without land-cover prior | TBD | — | — | — | TBD | TBD |
| A2 | AF I4/I5 + spatial context | TBD | — | — | — | TBD | TBD |
| B1 | global dNBR thresholds | — | TBD | TBD | — | TBD | TBD |
| B2 | land-cover-aware thresholds | — | TBD | TBD | — | TBD | TBD |
| B3 | + Sentinel-1 support | — | TBD | TBD | — | TBD | TBD |
| M1 | compact segmentation model | TBD | TBD | TBD | TBD | TBD | TBD |

## ML direction после baseline

AF:
- compact U-Net/FPN-like segmentation;
- focal/Tversky/Dice-aware loss;
- hard-negative mining на industrial/built-up hotspots;
- threshold calibration по validation F1.

BS:
- multi-branch pre/post optical features;
- explicit dNBR/RdNBR as channels;
- SCL mask;
- land-cover embedding/one-hot;
- optional SAR branch;
- weighted CE + Dice/Tversky;
- class-wise threshold/postprocessing only if validation supports it.

## Правило остановки

Компонент остаётся только если:
- даёт reproducible gain;
- не использует prohibited test information;
- укладывается в inference time;
- его можно объяснить в отчёте.
