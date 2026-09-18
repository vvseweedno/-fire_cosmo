# Validation protocol: selection vs deployment calibration

There are now two different OOF operations and they must not be confused.

## 1. Cross-fitted OOF — primary architecture comparison

For fold k:
- calibrate AF threshold and BS ordered thresholds using OOF predictions from all
  other folds;
- apply those parameters to fold k;
- never use fold-k labels to tune fold-k decision thresholds.

After every fold is predicted this way, pool all cross-fitted predictions and
compute the official micro metrics once.

Use this score when comparing architectures, features, losses and training
strategies.

## 2. Pooled OOF calibration — deployment parameters

After an architecture is selected, use all OOF predictions to fit the final AF
threshold, BS thresholds, cloud/SAR fallback weight and ensemble weights.

These parameters are then frozen for hidden-test inference.

The score measured on those same pooled OOF labels after fitting is a selection
objective, not the primary unbiased validation estimate.

## Why this matters

Threshold search and ensemble-weight search are themselves forms of model
selection. Reporting the maximum on the same labels used to choose parameters
creates optimism.

Cross-fitting reduces that optimism while preserving all training examples for
the final deployment calibration.
