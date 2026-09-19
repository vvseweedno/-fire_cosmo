# Final Research Report

Status: **PENDING labelled organiser train/test evidence**.

This report is deliberately evidence-first. The repository currently contains
the canonical evaluator and reproducible experiment protocol, but no labelled
organiser chip package is present in this checkout. Metrics marked `PENDING`
must be populated from generated artifacts, never typed in by hand.

## 1. EDA

The required EDA is generated after `scripts/preflight_dataset.py` succeeds.
It must report, separately for AF and BS:

- pixel and chip class balance;
- valid, cloud, and SCL fractions;
- thermal/context distributions and positive-negative separation;
- dNBR and severity distributions by land-cover class;
- Sentinel-1 delta statistics where those channels exist;
- event and geographic grouping coverage.

Each figure or table must end with an `OBSERVATION` and a `DECISION` grounded
in the same dataset fingerprint.

## 2. Baseline

The deterministic baseline is the VIIRS I1-I5 contextual AF score and the
Sentinel-2 pre/post physics BS score in `wildfire.baselines`. Its thresholds
and weights are versioned in `configs/baseline.json`.

| Candidate | F1_AF | IoU_burn | mIoU_sev | Score | Status |
|---|---:|---:|---:|---:|---|
| deterministic baseline | PENDING | PENDING | PENDING | PENDING | no labelled package attached |

Historical numbers from older branches are `HISTORICAL / UNVERIFIED` and are
not evidence for this release.

## 3. Leakage-safe experiments

All candidates use the same organiser `fire_event_id` grouped folds. The
cross-fitted evaluator pools pixel counts before calculating F1/IoU, and all
thresholds, ensemble weights, and post-processing parameters are selected
without labels from the evaluated holdout.

The machine-readable source of truth is the experiment registry under
`final_run/artifacts/experiments/`.

| Candidate | F1_AF | IoU_burn | mIoU_sev | Score | Delta | Runtime | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | PENDING | PENDING | PENDING | PENDING | — | PENDING | awaiting data |
| AF candidate ensemble | PENDING | — | — | PENDING | PENDING | PENDING | promotion gated |
| BS candidate ensemble | — | PENDING | PENDING | PENDING | PENDING | PENDING | promotion gated |

No candidate is promoted solely because it is more complex or uses more
channels. The promotion rule is implemented by the finalizer and requires a
reproducible cross-fitted gain or a documented robustness/runtime advantage.

## 4. Engineering and domain reasoning

- AF is extremely imbalanced, so threshold selection targets pooled pixel F1.
- I4/I5 contrast, local robust context, saturation, geometry, and land-cover
  features address thermal anomalies and industrial false positives.
- BS uses pre/post optical change features, cloud/SCL validity, optional SAR,
  terrain, and land-cover-conditioned severity thresholds.
- Harvested fields, phenology, dry grass, water, shadow, and cloud artefacts
  remain explicit error categories rather than being hidden by a global score.
- Sentinel-1 is promoted only when its leakage-safe ablation improves the full
  configured Score.

## 5. Conclusions

**MEASURED:** canonical micro-pooled scoring, strict IO contracts, grouped
fold infrastructure, RLE validation, and offline service demo.

**PENDING:** organiser-labelled metrics, ablation deltas, full inference wall
time, and final hidden-test performance.

The current blockers are the absence of the labelled organiser package and an
attached exact-source CI run for a final release. The repository therefore
must remain `NOT READY` until those artifacts exist.
