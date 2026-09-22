# Defense Presentation Source

Status: **ready as a source outline; final metric slide is PENDING evidence**.

## Slide 1 — Problem and impact

Two-stage monitoring: thermal active-fire detection followed by Sentinel-2
burned-area and severity mapping. The product is an operational API and map,
not a screenshot-only prototype.

## Slide 2 — Official objective

The configured working objective is `0.35 F1_AF + 0.35 IoU_burn + 0.30 mIoU_sev`.
The scorer is micro-pooled over pixels. Organizer confirmation is attached to
release evidence before this formula is called publicly verified.

## Slide 3 — Data and EDA

Show the dataset fingerprint, event-group coverage, AF imbalance, BS class
balance, cloud fraction, and land-cover distributions generated from the real
organiser package. Current checkout: **PENDING**.

## Slide 4 — Architecture

Official role-based raster adapter -> strict preflight -> grouped cross-fit ->
metric-gated deployment config -> loud-fail inference -> exact submission
validator. A separate FastAPI service exposes results and GeoJSON.

## Slide 5 — Active fire

VIIRS I1-I5, thermal contrast, robust local context, validity, and optional
auxiliary context. The objective is F1 on rare active-fire pixels, not accuracy.

## Slide 6 — Burned area and severity

Sentinel-2 pre/post indices, cloud/SCL handling, optional Sentinel-1 and terrain,
and land-cover-conditioned severity thresholds. SAR is retained only after a
full cross-fitted ablation.

## Slide 7 — Experiments

Show the registry table with fixed folds, seed, git SHA, data fingerprint,
runtime, bootstrap result, and promotion state. Values are **PENDING** until
the labelled package is run.

## Slide 8 — Metrics and runtime

`F1_AF: PENDING` · `IoU_burn: PENDING` · `mIoU_sev: PENDING` · `Score: PENDING` ·
`runtime tier: PENDING`. Never replace these labels with historical estimates.

## Slide 9 — Service demo

Offline demo catalog, spatial-temporal query, AF points, severity contours,
area summary in hectares, GeoJSON export, and a dependency-free map endpoint.

## Slide 10 — Failure modes and limits

Small-fire misses, industrial heat, harvested fields, adjacent severity
confusion, clouds, and missing organiser evidence are shown explicitly.

## Slide 11 — Reproducibility and compliance

Pinned command surface, deterministic inference, strict submission round-trip,
CI, event leakage audit, data-compliance policy, and license register.

## Slide 12 — Conclusion

The final decision is evidence-driven: promote only reproducible Score gains,
and report `NOT READY` while hard release gates remain pending.
