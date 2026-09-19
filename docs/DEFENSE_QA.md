# Defense Q&A

## Why micro averaging?

The configured case metric pools TP/FP/FN or intersections/unions over all
corresponding pixels before computing each metric. This prevents a tiny chip
from receiving the same weight as a large chip and is implemented by
`wildfire.evaluation.CompetitionEvaluator`.

## Why not a single dNBR threshold?

Spectral change shifts with land cover, phenology, clouds, and observation
conditions. The system exposes dNBR/RdNBR and land-cover-conditioned thresholds;
the final choice remains cross-fit and evidence-gated.

## Why Sentinel-1?

SAR can provide all-weather structural change context when optical imagery is
clouded. It is not automatically promoted: the full weighted Score and event
bootstrap must improve without leakage.

## How are false positives handled?

Thermal contrast, local robust context, validity, geometry, land cover, and
candidate hard-negative features target industrial heat, built-up areas, water
glint, and saturated pixels. Test coordinates and dates are never reconstructed.

## How is leakage prevented?

`fire_event_id` is the primary group boundary. Fold manifests and the leakage
audit verify that event groups do not cross validation folds. Production
preprocessing is fitted only on allowed training data.

## Why this model choice?

The deterministic physics/GBDT-compatible core is fast, explainable, and
available without heavyweight model dependencies. Neural candidates are
hypotheses until they beat it on the same cross-fitted official objective.

## Why these thresholds?

Thresholds are selected on calibration data disjoint from the evaluated
holdout. The deployment config is frozen only after the cross-fit and
reproducibility gates pass.

## How is runtime measured?

`scripts/benchmark_inference.py` launches the exact production command in a
cold process and measures until the output CSV is closed. It records every run,
SHA256, and the official time tier.

## What are the limitations?

No hidden-test metric is claimed without the real organiser package. Expected
errors include small fires, harvested fields, clouds, and adjacent severity
classes.

## What about test-data rules?

The code does not use answer-producing global fire products, reverse-engineer
private test geometry/date, or manually label test data. External resources
must be registered in `docs/DATA_COMPLIANCE.md` before use.
