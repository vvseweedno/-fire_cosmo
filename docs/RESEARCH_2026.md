# Research-backed accuracy roadmap — September 2026

This document separates published evidence, our hypothesis, and the experiment required to keep a component.
No paper result is treated as our expected score gain.

## Active Fire: contextual VIIRS first

NASA's 375 m VIIRS active-fire product uses all five I-bands and a multispectral contextual algorithm.
I4/I5 are central thermal signals; the product also uses local background statistics and explicit handling of water/cloud/glint and abnormal I4 radiometry.

Sources:
- https://modaps.modaps.eosdis.nasa.gov/services/about/products/viirs-land-c2-nrt/VNP14IMG_NRT.html
- https://modaps.modaps.eosdis.nasa.gov/services/about/products/viirs-land-c1/VNP14IMG.html

Repository implementation:
- I4/I5 robust z-scores;
- I4-I5 thermal contrast;
- normalized thermal contrast;
- 3/5/9-pixel contextual anomalies;
- optional I3 contextual feature.

Hypothesis: contextual thermal features improve AF-F1, especially hard negatives.
Acceptance: keep only features with positive pooled group-OOF delta in F1_AF.

## Burned area: bi-temporal modelling is the primary neural baseline

BiAU-Net (2024) uses pre/post Sentinel-2, attention, and a tailored loss to address class imbalance, mixed burn boundaries and generalization.
Source: https://doi.org/10.1016/j.jag.2024.104034

Hypothesis: a compact bi-temporal encoder with explicit pre/post deltas outperforms a single-date U-Net on IoU_burn.
Required ablation: post only; pre+post; pre+post+physics/delta channels; same folds and training budget.

## Connectivity-aware learning is a high-priority BS loss experiment

CA-BAFormer (2026) explicitly models semantic connectivity of burn scars and reports improved compact burned-area segmentation versus its uni-temporal U-Net baseline.
Source: https://doi.org/10.1016/j.jag.2026.105199

Hypothesis: an auxiliary connectivity/boundary objective reduces holes and fragmented burn masks.
Acceptance: positive pooled OOF IoU_burn delta with acceptable inference cost.

## Loss must target the leaderboard metric

Lovasz-Softmax is a tractable surrogate for optimizing Jaccard/IoU rather than only per-pixel cross entropy.
Source: https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html

Planned BS loss ablations:
1. CE
2. CE + Dice
3. CE + Lovasz
4. CE + Lovasz + connectivity/boundary auxiliary objective
5. ordinal severity objective combined with burn-mask loss

AF loss ablations: BCE; focal; Dice+focal; Dice+focal+deep supervision.
Probability thresholds are calibrated from train/OOF predictions; 0.5 is not assumed optimal.

## Physics feature bank

Implemented candidate features: NBR/dNBR, RBR, RdNBR, NDVI/dNDVI, NDMI/dNDMI, NBR2/dNBR2, MIRBI/delta MIRBI, BAIS2/dBAIS2, per-band post-pre deltas, Sentinel-1 VV/VH deltas.

References:
- USGS burn severity glossary: https://burnseverity.cr.usgs.gov/glossary
- BAIS2: https://doi.org/10.3390/ecrs-2-05177

BAIS2 assumes surface-reflectance-like values. Official archive scaling must be verified before it is used.
Every physics feature is an input candidate, not a manually weighted truth.

## Foundation-model benchmark

NASA/IBM Prithvi-EO-2.0 publishes downstream examples for Wildfire Scar Detection and Burn Scar Intensity.
Official repo: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
Relevant configs: configs/firescars.yaml and configs/burnintensity.yaml.

Decision: benchmark Prithvi only after the real loader and OOF pipeline work.
A 300M/600M backbone is not the default because competition inference speed is part of the objective.

## Validation mathematics

Model selection uses group OOF:
- fire_event_id never crosses train/validation;
- every chip is validation exactly once;
- OOF probabilities/scores are concatenated;
- official micro metrics are computed once on pooled OOF pixels;
- thresholds and ensemble weights are tuned only on pooled OOF predictions.

Do not average fold F1/IoU values for final model selection when the official metric pools pixels.

## Accuracy experiment order

1. Verify official archive layout/scaling.
2. Deterministic B0 with exact AF threshold and learned ordered BS thresholds.
3. Five group-OOF folds.
4. Physics feature ablations.
5. Compact bi-temporal attention U-Net.
6. AF Dice+Focal/deep-supervision ablation.
7. BS Lovasz/Dice/CE ablation.
8. Ordinal severity head.
9. Connectivity/boundary auxiliary objective.
10. Prithvi-EO-2.0 benchmark.
11. Hard-negative mining from OOF AF false positives.
12. Logit ensemble only for models with complementary OOF errors.

Every experiment records F1_AF, IoU_burn, mIoU_severity, total Score, latency, parameters, peak memory and delta versus baseline.
