# Cloud-aware multimodal routing

This deterministic layer is the bridge between the old physics baseline and the
future learned optical/SAR router.

## Why

The competition scores cloudy pixels; masking every cloudy optical pixel to
class 0 creates forced false negatives when burn labels exist there.

Sentinel-1 is an independent all-weather observation source, so the baseline now
contains an optional SAR fallback.

## Mathematical safety property

The calibration search explicitly includes:

cloud_sar_weight = 0

That setting reproduces the previous behavior: cloud/shadow optical pixels stay
masked.

Therefore, on the exact same calibration/OOF pixel pool, the optimizer's best
objective cannot be lower than the baseline candidate merely because the SAR
fallback was introduced. This is an optimization-set guarantee, not a
generalization guarantee on unseen test data.

## Routing rule

Clear optical pixel:

score = dNBR + w_clear * z(SAR change)

Cloud/shadow optical pixel with SAR:

score = w_cloud * z(SAR change)

Cloud/shadow optical pixel without SAR:

not predicted by this deterministic baseline

The sign of w_cloud is learned from OOF candidates; it is not hard-coded because
SAR backscatter response depends on ecosystem and acquisition conditions.

## Next neural experiment

Replace the scalar route with a learned gate:

g = softmax(router(SCL, optical features, SAR features, terrain/context))

F = g_opt * F_opt + g_sar * F_sar + g_ctx * F_ctx

Keep only if pooled group-OOF Score improves and event-level paired bootstrap
supports the gain.
