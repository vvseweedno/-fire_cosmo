# Experiment Protocol

## Objective

Every modelling idea is treated as a candidate, not as an upgrade.

The working selection objective is the configured composite score:

```text
Score = 0.35 * F1_AF + 0.35 * IoU_burn + 0.30 * mIoU_severity
```

This formula is the repository's configured competition objective. Public
organizer confirmation must be attached separately before it is described as a
publicly verified official scoring rule.

## Frozen reference

Before candidate search on labelled data:

1. validate the dataset contract;
2. fingerprint the dataset;
3. require explicit event/group IDs for strict mode;
4. create one deterministic group fold manifest;
5. run the existing physics baseline;
6. store its OOF predictions and aggregate metrics.

That exact tuple becomes the reference:

```text
(source SHA, dataset fingerprint, fold SHA, baseline config)
```

Do not silently regenerate folds while comparing candidates.

## Search discipline

Each phase has a finite search space defined before looking at holdout results.

Current AF candidate family includes:

- BASE;
- contextual VIIRS physics features;
- HARD_NEG_CONTEXT;
- optional HARD_NEG_CONTEXT_PERSISTENT when an explicit aligned recurrence
  prior exists.

The ensemble optimizer must always be able to return BASE unchanged.

New candidates are rejected when:

- delta <= epsilon;
- gain is driven by one fold/event only;
- paired event bootstrap is inconclusive;
- latency/memory cost is disproportionate;
- reproducibility degrades;
- the candidate requires guessed metadata or hidden-test leakage.

## Leakage boundary

Fold split happens by organizer event/group **before** any sampling, calibration,
threshold fitting, ensemble fitting or checkpoint selection.

The explicit leakage audit checks:

- event/group coverage;
- train/validation overlap;
- event appearing in more than one validation fold;
- duplicate validation assignment;
- same physical channel source reused across chip IDs.

A strict run cannot proceed if this audit fails.

## Error analysis

Before adding complexity, inspect OOF failures with:

```bash
python scripts/analyze_errors.py ...
python scripts/analyze_candidate_diversity.py ...
```

Candidate diversity is a search aid, not a promotion criterion.

## Statistics

For important promotion:

- primary decision: cross-fitted configured Score;
- require delta Score > epsilon;
- require event-level paired bootstrap;
- target at least 1000 bootstrap replicates, normally 2000;
- require lower 95% CI(delta) > 0;
- require P(delta > 0) >= 0.95;
- bootstrap aggregate scores must agree with the canonical cross-fit evaluator.

If the number of independent events is too small, record the limitation rather
than inventing statistical certainty.

## Experiment registry

Machine-readable records live at:

```text
artifacts/experiments/<experiment_id>.json
```

The schema includes:

- experiment id and UTC timestamp;
- source git SHA;
- dataset fingerprint;
- task/candidate/version/configuration;
- seed and fold-manifest SHA;
- train/holdout event fields;
- per-fold and aggregate metrics;
- score/reference/delta;
- runtime and peak RAM;
- GPU evidence/status;
- bootstrap result;
- lifecycle state;
- promotion/rejection reason.

The finalizer writes its aggregate final candidate as `CANDIDATE`, not
`FINAL`, because exact-source CI still has to be attached and verified.

## Lifecycle

Allowed states:

```text
EXPERIMENTAL -> CANDIDATE -> PROMOTED -> FINAL
                         \-> REJECTED
```

No manual flag may silently turn a failed candidate into FINAL.

## Exact-source release proof

`PROVEN` requires an actual GitHub Actions run bound to the release source:

```text
source_commit_sha == workflow_sha
AND workflow_conclusion == "success"
AND workflow_run_id exists
```

A JSON field such as `ci.green=true` is not evidence.

After the final pipeline, attach CI explicitly:

```bash
python scripts/verify_proven_release.py \
  --evidence final_run/artifacts/release_evidence.json \
  --workflow-run-id <RUN_ID> \
  --workflow-sha <EXACT_SOURCE_SHA> \
  --workflow-conclusion success \
  --output final_run/artifacts/final_validation.json
```

The verifier still checks every other gate; CI attachment cannot override a
failed metric, leakage, bootstrap, reproducibility or submission check.
