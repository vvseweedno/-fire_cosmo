# Monotonic OOF ensemble search

Final submissions may blend multiple complementary models, but only after every
candidate has produced aligned leakage-safe OOF score maps.

## AF

For model scores z_m(x), search convex blends:

z(x) = sum_m w_m z_m(x),  w_m >= 0,  sum_m w_m = 1

Every trial receives a fresh exact micro-F1 threshold calibration.

## BS

The same convex blending is followed by fresh ordered severity threshold
calibration against:

0.35 * IoU_burn + 0.30 * mIoU_severity

## Monotonicity property on OOF

The algorithm starts from the best single model.

At every greedy step it searches:

z_new = alpha * z_current + (1-alpha) * z_candidate

and the alpha grid includes alpha=1.

Therefore the current solution is always a legal candidate at the next step.
The selected OOF objective cannot decrease by construction.

This is not a promise about hidden-test generalization. The selected ensemble
must still pass fire-event paired bootstrap and latency checks.

## Practical rule

A model joins the final ensemble only when:
- pooled OOF Score improves;
- event-level bootstrap does not show an obviously unstable gain;
- inference latency remains inside the competition budget.
