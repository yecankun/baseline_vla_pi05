# Frozen Gated Adapter: Pre-Cast Versus Post-Cast Residual

Predeclared 2026-09-21, before inference. User authorized this single-factor
forward comparison. No optimization, checkpoint selection, gate rescaling,
temporal-encoder change, world loss, sensor/label change, or hardware.

## Fixed source and scope

Reuse the action-only step100 adapter in `real10_pi05_gated_action_v1`, the
original Elite3000/Piper1000 pair, and the exact 30-context panel/noise from the
completed `real10_pi05_gated_history_audit_v1`. All model weights are frozen.
Each of the 10 original training episodes supplies 2 hold + 1 feed contexts.
This is an enriched training diagnostic, not held-out validation, a new split,
a prevalence-weighted dataset score, or a model-selection set.

Let s be the native FP32 state-projection output, r the unchanged FP32 history
residual, and Q the existing BF16 cast. Compare only the addition location:

- `postcast`: Q(Q(s) + Q(r)), the unchanged production hook.
- `precast`: Q(s + r), an explicit opt-in hook on state_proj output.

The original checkpoint loader, normalization, state32/action32 interface,
prefix length/masks, current inputs, inference noise and signed gate are fixed.
The FP32 dtype must be observed at runtime; do not silently promote an already
BF16 projection and describe it as recovering its lost precision. The final
token remains BF16. No model-wide precision conversion is allowed.

## Necessary checks and comparisons

1. Reproduce saved original and post-cast outputs on all 30 contexts.
2. With the residual multiplier set to zero (without changing saved weights),
   the new path must reproduce the full original action tensor, output packet,
   and fixed-time action loss exactly on all 30 contexts.
3. Transparently capture the projection before either injection, the actual
   final prefix, masks and token counts. Verify each path's arithmetic and that
   only the existing last state token changes. Disabling/closing hooks restores
   original inference; all parameters stay frozen and the adapter is unchanged.
4. Repeat the previous fixed history diagnostics in each path: true history,
   past contents repeated from current, and past contents swapped. Keep current
   observation, timing slots and validity masks fixed. These are diagnostics,
   not candidate training data or a search for the best history manipulation.

## Metrics and interpretation

Report actual update e = float32(final token) - float32(Q(s)): changed-coordinate
fraction, RMS, relative L2 error versus r, and direction cosine. Separately
report error of the **whole combined token** versus FP32 s+r. These measures
are not interchangeable: one final cast can improve combined-token rounding
while increasing the incremental update's magnitude or distortion. Coordinate
counts and amplitude ratios are not retained-information percentages.

Report xyz action MAE (mm), fixed-time flow loss, per-axis/task/episode metrics,
paired action drift, and unchanged Piper probabilities. For each precision path,
report repeat/swap sensitivity before quantization and in the final action.
Show all predeclared conditions, not only a favorable subset. No statistical
significance, generalization, contact/tactile, rotation or real-system claim.

Runtime is expected below 2 minutes based on the previous 48-second audit;
no throughput superiority is claimed from this shared-machine run. Save raw
JSON and compact captured vectors, then make a Chinese figure from actual data.
Use the project local venv for plotting and remote project2026-pi for inference.

## Output, recovery and decision

New output: `simulation_output/real10_pi05_gated_precast_audit_v1/`. Refuse to
overwrite an existing directory. On failure, preserve its diagnostics and use
a named new directory only after confirming the old job stopped. There is no
training-resume command and no new checkpoint is written.

Zero-identity or reference-reproduction failure stops interpretation. Otherwise,
report whether numerical injection changes and whether the fixed train panel
also improves; neither automatically authorizes adoption or training. Keep the
original onsite model. Any temporal-encoding or learning experiment is a separate
next decision, not an automatic extension of this audit.
