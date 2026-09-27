# Real10 Parameter-Matched Adjacent-Change Encoding

Predeclared 2026-09-21 before execution. User approved testing explicit motion
changes after the frozen pre/post-cast diagnostic. This is a bounded action-only
representation experiment, not a world-loss run, deployment or hardware test.

## Hypothesis and exactly matched control

Compare the original absolute-history encoder with explicit adjacent changes
plus current content. Both use the original **post-cast** gated state-residual
hook, not the recently audited pre-cast option. This keeps the numerical
connection fixed and permits reproduction of the previous action-only result.
Do not simultaneously tune precision, gate scaling, model size or loss.

The motion encoder reuses the same projections, slot/view embeddings, attention,
queries and output projection; no parameters are added. For three timestamps:

- Visual slots: `[z1-z0, z2-z1, z2]`, where z is the existing per-frame/per-view
  normalized frozen PI05 visual feature.
- State slots: `[s1-s0, s2-s1, s2]`, using existing checkpoint-normalized state32.
- Time slots: `[t1-t0, t2-t1, 0]` in seconds, through the existing age projection.
- Mask each change unless both endpoints are valid; the current slot is valid.

Do not normalize the difference vectors to unit scale or divide by dt: noisy
low-visibility changes must not be automatically amplified into large features.
These are observable representation differences, not measured velocities, distal
tip motion, tactile estimates or exact contact. Output remains the canonical
`elite_tcp_delta_6d + piper_intent_id`; state layout, images, labels and controls
are unchanged. The same two internal tokens can feed the existing world head,
but that head is frozen/unused here. Persist the new semantic version in each
checkpoint: matching state_dict shapes do not make the two encoders equivalent.

## Fixed data and budget

Reuse the saved **untrained initial_state** from the source
`real10_pi05_joint_comparison_v1/paired_checkpoint.pt`, then add gate=0 to both
arms. Do not initialize from the trained step100 adapter. Load identical numeric
parameters into both representations, including the unused frozen world head.

Reuse the complete 2719-record/10-episode index and saved 100-step order from
`real10_pi05_gated_action_v1` (98 hold, 2 feed; not a full epoch). Each arm gets
100 updates, batch1, seed123, AdamW lr1e-4/wd0 and L2 clip1 over gate+temporal
parameters. Use exactly matched per-step native noise/time and the same fixed
final step100; no best-checkpoint/seed/learning-rate search or feed oversampling.
Freeze the original PI05 and Piper state_only head. Estimate the complete job
below 3 minutes using prior 56-70-second measurements; run on remote project2026-pi.

All 10 episodes were already used by the base model. The fixed panel contains
30 contexts, 2 hold + 1 feed per episode, and is **training-only**, not held-out
validation or a population-weighted dataset score. No new data collection or
post-hoc clean split is implied by this experiment.

## Necessary checks and reporting

1. Both arms have identical parameter names/shapes/counts and identical initial
   parameter values. Gate0 reproduces full original actions, packets and flow
   loss on all30 contexts; existing original predictions reproduce exactly.
2. Real reset/one-past contexts check pair masking and padded-content invariance.
   Repeating current content makes both explicit visual/state changes zero,
   preserving their time slots. Future targets stay outside context computation.
3. First update must reach the gate while temporal gradient is zero as expected;
   subsequent updates reach temporal parameters. Base/Piper/world remain frozen.
4. The concurrently rerun absolute-history control must reproduce its prior
   step100 parameters, predictions and training losses. Failure is a reproduction
   issue, not evidence for the new representation; stop interpretation.
5. Report original and both final-arm xyz MAE (mm), fixed-time flow loss,
   task/axis/episode errors, gate/gradient traces, residual scale, runtime and
   memory. Piper probabilities must stay unchanged. Evaluate true history,
   repeat-current and swap-past content with the previous fixed panel/noise and
   unchanged current inputs/timestamps/masks. Compare both shared-token and
   action sensitivity; nonzero sensitivity is not necessarily beneficial.
6. Save/reload each final adapter and verify one prediction; disabled hook
   restores original behavior. No real deployment or full-epoch claim.

The primary question is whether the representation helps under this fixed small
action-only budget. Swapping a near-static history need not change a correct
action; do not optimize merely to force order sensitivity. If scores improve,
call them a train-panel signal requiring broader, independently designed
evaluation, not world-model innovation evidence or stable generalization.
If they regress, do not add auxiliary losses, rescale gates or search variants
against this panel. Keep the original onsite model regardless of this result.

## Outputs and recovery

New output: `simulation_output/real10_pi05_motion_comparison_v1/`. Preserve all
source checkpoints and old results. Save paired recovery state at0/25/50/75/100;
resume the same protocol only after confirming no old process is running.
Completed reports are immutable. A failure before the first checkpoint keeps
its diagnostics; use an explicit new directory after fixing the cause.

Final artifacts: protocol, identity/input checks, training logs, all panel
predictions, report and versioned `deployable=false` adapters. Return only small
JSONs; keep checkpoints on the remote machine. Create a Chinese PNG/SVG with a
data manifest using the project local venv, not a new research environment.
Record implementation, verification and user visual acceptance separately.
