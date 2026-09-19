# Error Analysis Protocol

## Purpose

Accuracy work must start from measured failure modes, not visual preference or
model complexity. Error analysis is performed only on leakage-safe training-side
or cross-fitted predictions.

## Active Fire

For every baseline/candidate comparison record:

- TP
- FP
- FN
- TN
- precision
- recall
- F1

When organiser land-cover context is available, report false positives and false
negatives by land-cover class.

Priority AF failure modes to inspect:

- isolated noisy thermal pixels;
- built/industrial contexts;
- hot bare soil;
- water/snow/coast boundaries;
- sunglint-like I3 response;
- weak/sub-pixel fires;
- small connected fire structures.

The `HARD_NEG_CONTEXT` candidate is deliberately metric-gated. Its spatial
support and hard-negative penalties are not unconditional rules; the AF
cross-fit optimizer may assign it zero weight.

## Burn Severity

Record:

- burned/unburned TP/FP/FN;
- 4x4 severity confusion matrix;
- IoU for severity classes 1/2/3;
- mIoU severity.

Where metadata are available, compare results by event and later extend the
report by cloudiness, land cover, burned-area size, and sensor availability.

## CLI

Example for selected AF cross-fit predictions:

```bash
python scripts/analyze_errors.py \
  --prediction-dir final_run/repro_run_1/crossfit_predictions/AF/ensemble \
  --task AF \
  --meta-csv /path/train/meta.csv \
  --output artifacts/error_analysis/af_selected.json
```

Example for BS:

```bash
python scripts/analyze_errors.py \
  --prediction-dir final_run/repro_run_1/crossfit_predictions/BS/ensemble \
  --task BS \
  --meta-csv /path/train/meta.csv \
  --output artifacts/error_analysis/bs_selected.json
```

The final competition pipeline now generates baseline/selected reports
automatically.

## Candidate diversity

Use prediction diversity only to decide what may be worth ensembling. It is not
an independent promotion criterion.

```bash
python scripts/analyze_candidate_diversity.py \
  --candidate BASE=path/to/baseline_predictions \
  --candidate ENSEMBLE=path/to/ensemble_predictions \
  --meta-csv /path/train/meta.csv \
  --output artifacts/candidate_diversity.json
```

Reported pair statistics include:

- prediction correlation;
- disagreement rate;
- error overlap;
- error symmetric difference;
- errors unique to each model;
- event-level complementary wins when metadata are supplied.

Final promotion still requires improvement in the canonical official metric
under leakage-safe cross-fit.
