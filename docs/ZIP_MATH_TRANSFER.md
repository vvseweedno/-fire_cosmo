# Transferable mathematics from all files.zip

The archive is mostly language-model research. We do not copy architectures that do not match the hackathon task.

## Accepted transfers

### 1. Paired bootstrap from elia_omega_v6_academic_fixed.py
The archive contains paired ablation analysis with bootstrap confidence intervals.
We adapt this idea to the hackathon using fire_event_id as the resampling unit, not individual pixels.
Reason: adjacent pixels and chips from the same fire are correlated; pixel bootstrap would dramatically overstate certainty.

Implementation:
- wildfire/statistics.py
- scripts/compare_oof_experiments.py

Decision rule:
- delta Score > 0 is necessary but not sufficient;
- if the group-bootstrap 95% interval is entirely above zero, the gain is materially more credible;
- inconclusive gains stay experimental and do not automatically enter the final ensemble.

### 2. Model EMA from Seraphim / Elia training code
The archive repeatedly keeps an exponential moving average of parameters.
For future segmentation training we will compare raw vs EMA validation logits.
EMA adds no inference-architecture complexity because the EMA weights replace the raw weights at export.

### 3. Router / Mixture-of-Experts from CognitiveCore and Seraphim
The transferable idea is conditional expert weighting, not the language-model experts themselves.

Remote-sensing reinterpretation:
- optical expert: Sentinel-2 spectral/change features;
- SAR expert: Sentinel-1 VV/VH change and structure;
- context expert: terrain, land cover, observation quality;
- gate: SCL/cloud state plus learned multimodal context.

Candidate fusion:
z = g_opt*z_opt + g_sar*z_sar + g_ctx*z_ctx
with g = softmax(router(features)) and sum(g)=1.

This is especially relevant because cloudy pixels remain scored in the task. A learned gate can shift weight toward SAR when optical information is unreliable instead of forcing cloudy pixels to class 0.

Recent external support for gated optical-SAR fusion:
- MGFNet, Int. J. Applied Earth Observation and Geoinformation (2024), DOI 10.1016/j.jag.2024.104241
- CloudSeg, ISPRS JPRS (2024), DOI 10.1016/j.isprsjprs.2024.06.001
- SOLSTM, IEEE GRSL (2025), DOI 10.1109/LGRS.2025.3535524

### 4. Automatic micro-batch calibration
elia_omega_v6 benchmarks candidate micro-batches and chooses the fastest non-OOM configuration.
We will reuse the policy when neural training exists: maximize GPU utilization without changing the global batch or experiment protocol.

### 5. Quantization only after accuracy is locked
The archive contains INT6 quantization work.
For this hackathon it belongs only to the speed phase after the final OOF model is chosen.
Any quantized export must pass a measured delta-Score regression threshold before use.

## Rejected transfers

- complex-valued byte embeddings: no demonstrated relevance to multispectral segmentation;
- Ouroboros recurrence: unnecessary for two-date pixel segmentation until an ablation proves otherwise;
- LRU/state-space language memory: sequence optimization does not map cleanly to this spatial task;
- Hyperfield/resonance-energy modules: no defensible connection to the competition metric;
- cognitive/persona loops: unrelated to AF/BS prediction.

## High-priority neural experiment derived from the archive

Cloud-aware gated multimodal BS model:
1. independent optical and SAR encoders;
2. SCL/quality-aware router;
3. context branch for terrain/land cover;
4. shared burn head and ordinal severity head;
5. train with Dice/Lovasz/ordinal losses;
6. evaluate via pooled group OOF;
7. compare with paired fire-event bootstrap.

The router stays only if it improves total pooled OOF Score and its bootstrap comparison is stable enough to justify added complexity.
