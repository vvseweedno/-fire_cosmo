# Accuracy architecture

The accuracy stack is designed around the official score, not feature count.

## AF path

VIIRS I1..I5 + context → contextual physics features → segmentation model → probability map → exact pooled-OOF F1 threshold → binary AF mask.

Primary error loop: rank OOF false positives/negatives, stratify by land cover/geometry/weather/context, then hard-negative mine only after error evidence exists.

## BS path

S2 pre/post + S1 pre/post + terrain + land cover → explicit change/physics channels → bi-temporal encoder → shared representation → burn head + ordered severity head.

Preferred severity hypothesis:
P(y >= 1), P(y >= 2), P(y >= 3), with monotonic ordering.
P(y >= 1) is also the natural burn/no-burn signal, aligning architecture with IoU_burn and mIoU_severity.
This is rejected if a 4-class softmax wins on identical OOF folds.

## Candidate BS objective

L = a * L_burn + b * L_severity + c * L_boundary

Candidate pieces: focal/BCE, Dice, Lovasz, weighted CE, ordinal cumulative loss, connectivity/boundary auxiliary loss.
Loss weights are model-selection parameters, not aesthetic choices.

## Calibration layer

AF: exact threshold sweep over pooled OOF scores.
BS deterministic baseline: ordered threshold coordinate search directly maximizing 0.35*IoU_burn + 0.30*mIoU_severity.
Neural BS: calibrate burn/severity decisions on pooled OOF probabilities.

## Ensemble layer

Ensemble calibrated logits/probabilities, never already-thresholded masks.
For candidate models z(x) = sum_m w_m*z_m(x), with w_m >= 0 and sum w_m = 1.
Weights are optimized on pooled OOF score and frozen before test inference.
A model joins the ensemble only when it adds complementary error correction.

## Speed constraint

Accuracy is primary, but every model-selection row records end-to-end inference latency.
Large foundation models remain challengers until their OOF Score gain justifies runtime.

## Stop rule

Remove a component if it does not improve pooled OOF Score, lowers total Score, creates leakage risk, is unstable across folds, costs disproportionate inference time, or is not reproducible.
