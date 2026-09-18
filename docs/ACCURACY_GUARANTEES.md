# Accuracy guarantees vs research hypotheses

This file prevents one category error: calling a promising idea "proven" before
it has been measured on the official data.

## Guarantees implemented in code

### Baseline-preserving cloud/SAR search

The cloud/SAR candidate set contains the exact previous behavior:

cloud_sar_weight = 0

Therefore introducing the fallback expands the search space without deleting
the old solution. On the same calibration objective, the optimizer can always
return to the baseline candidate.

### Monotonic greedy convex ensemble search

Every ensemble step searches:

z_new = alpha * z_current + (1-alpha) * z_candidate

with alpha=1 included.

Thus z_current remains a legal candidate at every step. The selected OOF
objective is non-decreasing by construction.

### Cross-fitted threshold evaluation

For fold k, AF/BS thresholds are calibrated using only the other folds. Fold-k
labels are not used to choose fold-k thresholds.

This reduces selection optimism relative to fitting and reporting thresholds on
the same pooled OOF labels.

### Fire-event cluster bootstrap

Ablation uncertainty is resampled by fire_event_id groups, not individual
pixels. This better respects within-fire correlation than a naive pixel
bootstrap.


### BASE-anchored burn-index candidate ensemble

The deployable BS candidate search always contains `BASE`, which is exactly the
current cloud/SAR-aware dNBR score. Candidate weights are convex and the pooled
optimizer starts from the best legal single candidate, so adding the candidate
search cannot make the selected pooled OOF objective worse than the best
candidate available in that search.

More importantly, promotion is measured separately by
`scripts/evaluate_crossfit_bs_candidates.py`: for fold k, candidate weights,
global severity thresholds and land-cover thresholds are fitted only on the
other folds. The report exposes `promotion_allowed` only when the aggregate
cross-fitted BS contribution improves over the BASE-only path.

Because AF is unchanged in this comparison, the reported BS subscore delta is
also the delta in the full competition Score on the same cross-fitted records.

## Strong evidence, but not a guarantee on this dataset

The following are research-backed candidates:

- Sentinel-1 + Sentinel-2 fusion for cloud/all-weather burned-area mapping;
- cloud-aware mixture-of-experts routing;
- bi-temporal segmentation;
- connectivity-aware burn-scar learning;
- Lovasz-style IoU surrogate losses;
- Earth-observation pretrained encoders such as Prithvi-EO-2.0;
- EMA checkpoints.

They remain hypotheses until cross-fitted OOF proves a positive total Score
delta on the hackathon data.

## Promotion rule

A component can enter the final competition stack only when all applicable
conditions hold:

1. Cross-fitted OOF total Score improves or remains neutral while materially
   improving a required non-score criterion such as latency.
2. No leakage or hidden-test information is used.
3. Event-level paired bootstrap does not show the gain to be obviously unstable.
4. Full-test latency remains inside the competition budget.
5. The result is reproducible from committed code/config.

## Hidden-test honesty

No architecture can be mathematically guaranteed to improve an unseen hidden
test distribution from training data alone.

What this repository can guarantee is narrower and useful:

- old solutions remain available when the search space expands;
- validation parameters are not tuned on the fold they score;
- ensemble search is monotonic on its objective;
- uncertainty is measured at the fire-event level;
- every hidden-test decision is frozen before inference.
