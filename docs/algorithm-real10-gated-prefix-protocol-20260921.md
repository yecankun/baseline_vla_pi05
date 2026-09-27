# Real10 Identity-Initialized Temporal Residual: Action-Only Diagnostic

Predeclared before execution, 2026-09-21. No deployment or world-loss training.

## Hypothesis and one connection change

The previous randomly appended prefix already perturbed the trained policy before
optimization. Test an identity-initialized connection before another joint run.
Reuse the same three-observation, two-view history encoder and its two internal
2048D tokens. Add `tanh(g) * mean(history_tokens)` to the existing trained state
token, with scalar `g=0` initially. No extra policy attention positions or masks;
zeroing newly appended tokens would not provide this guarantee.

This is a representation residual, not a post-hoc action correction. The shared
temporal representation can still condition the existing future predictor, but
the predictor is frozen/unused in this run. Original PI05 and Piper stay frozen.
Shared external observation/action semantics, labels, ROI and normalization stay
unchanged. No future target, contact truth, or new sensor estimate enters policy.

## Fixed data, references and optimization

Read the completed `real10_pi05_joint_comparison_v1` protocol/checkpoint read-only.
Reuse exactly its saved **untrained initial_state** for temporal/world parameters,
not either trained adapter. Add only the zero scalar gate. Reuse all 2719 source
transition indices, the same 30 diagnostic contexts, and the same 100-step order.
Keep seed123, batch1, AdamW lr1e-4/wd0, per-step native flow noise/time and temporal
gradient clipping L2=1. The scalar gate is part of that policy parameter block.
Do not change sampling to enrich feed: preserve the previous 98 hold / 2 feed
short-budget sequence. Fixed final step100; no best-checkpoint/model selection.

The clean paired references are original PI05 vs gated initialization vs gated
action-only after100 steps, all with identical per-context inference noise.
The old appended-token action-only result is a **historical same-protocol
connection reference**, not a concurrent replicate or evidence about joint loss.
Verify current original predictions reproduce the saved original predictions.

All ten episodes were already used to train the base. The 30-context panel is
training-only, feed-enriched (2 hold + 1 feed per episode), not a validation split
or a population-weighted success estimate. No seed/learning-rate/gate search.

## Minimal execution checks and measurements

- On all30 contexts: gate-zero normalized actions and fixed-time action loss
  equal original PI05 exactly; no prefix-length/mask changes, Piper unchanged.
- At the first action update: finite nonzero scalar-gate gradient, zero shared
  temporal gradient (expected because gate=0). After the gate opens, check that
  the action gradient reaches the temporal module; do not misreport the initial
  zero temporal gradient as a broken connection.
- Record each update's action loss, gate, shared and total gradient norms.
  Freeze/verify original PI05, Piper and unused world predictor. Check one final
  adapter save/reload and disabled-hook action equality.
- Report initial/final xyz MAE in mm, task/per-episode differences, fixed-time
  action loss, policy drift, gate magnitude, residual RMS, runtime, GPU memory.
  Finite optimization and recovered identity are engineering results, not proof
  of convergence, generalization, tactile capability or real-system success.
- Save recovery state at0/25/50/75/100, resume only the same fixed protocol after
  confirming the previous process stopped. Completed outputs are immutable.

## Decision boundaries

Keep the onsite base regardless of this small experiment. If initialization
identity fails, stop before training and preserve evidence. If action-only short
optimization regresses, do not add future loss or search weights against this
panel. If it remains stable or improves, this only supports a later matched
gated action-only vs gated joint experiment. The previous appended-token lambda
is not transferable: the new gate changes policy gradients, which are zero with
respect to temporal parameters at initialization. A later protocol must address
that warm-start/calibration issue explicitly, without claiming a benefit now.
