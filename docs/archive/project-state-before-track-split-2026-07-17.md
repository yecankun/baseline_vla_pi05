# Project State

Last updated: 2026-07-17

This file is the short current-state entry point. Historical experiment detail
was moved to `docs/archive/project-state-before-entry-slimming-2026-07-13.md`.
Use archived notes only when a task needs old metrics, old commands, or the
reason behind a compressed conclusion.

## Current Mainline

Highest priority:

```text
simulation realism and real-system correspondence > BC rollout success
```

The small BC model is a loop-closure/regression check only. A simulation rollout
is not considered solved if it depends on hidden MuJoCo truth, oracle wall/tip
state, route-relative correction, exact contact feedback, or a controller that
cannot plausibly exist on the real system.

Current research route:

- MuJoCo-first, not Isaac-first.
- Two-track execution after the 2026-07-13 group meeting:
  continue real data collection/alignment while starting VLA-style algorithm
  feasibility tests using existing real/sim data.
- Initial algorithm target: a small multimodal/VLA-style model, preferably 3B
  or smaller when feasible, with 7B as the upper early-test bound.
- Practical VLA baseline: openpi/pi0.5-style first, with pi0.7-style ideas used
  only as design inspiration until a reproducible implementation path is
  available.
- Accepted algorithm design: openpi/pi0.5-style reproducible baseline plus
  pi0.7-style context conditioning, starting with image-derived tactile/contact
  fields as the extra context.
- Mainline collector: `simulation.collect_formal_tip_line_guidance`.
- Guidewire assumption: tip-centric / hard-elastic guidewire with continuous
  line-shaped rendering.
- Default formal start range:
  `--start-fraction-min 0.42 --start-fraction-max 0.54`.
- Formal checks should use `--formal-data` and `--formal-eval` so oracle
  mechanisms fail closed.

Legacy/diagnostic routes:

- `simulation.collect_tip_guided_wire`
- `simulation.collect_mujoco_physical_guidance`
- Older oracle/tip-anchor experts and centerline correction experiments

## Current Interface

Target VLA-facing contract:

```text
observation + task instruction -> Elite TCP delta + Piper discrete intent
```

Observation should stay close to real availability:

- side/top images;
- Elite pose/state;
- Piper state/action label;
- task/branch target information;
- image-derived tactile/contact estimator outputs, such as estimated contact
  flag, confidence, and image distance, produced by visual collision/distance
  analysis rather than exact simulator contact truth;
- optional additional estimator outputs only if they can be produced from real
  sensors.

Controller-owned execution:

- Elite TCP delta -> IK -> joint execution.
- Piper intent is hold/feed/retract, with hold/feed as the senior-compatible
  minimum interface.
- Controller owns timing, cooldown, limits, safety checks, and executed-command
  logs.

The external compatibility schema may still appear as
`piper_feed_elite_joint`, but it is not the preferred policy target.

## Current Valid References

Use these as current anchors:

- `AGENTS.md`: compact repo entry and guardrails.
- `.codex/skills/project-2026-guidewire-mujoco/SKILL.md`: Codex workflow rules.
- `docs/handoff.md`: execution-focused handoff.
- `docs/weekly-meeting-log.md`: append-only factual meeting log.
- `docs/commands.md`: current copyable commands.
- `docs/real-alignment-20260707-calibration-audit.md`: real capture alignment
  and hardware-calibration audit.
- `docs/archive/`: historical detail and old entry snapshots.

Important data interpretation:

- `branchs/` is senior-provided real image/Elite magnetic-arm motion reference.
  It is not guidewire path truth and should not be used as centerline
  supervision.
- July 2026 onsite real captures are calibration/hardware-diagnostic data, not
  clean expert demonstrations unless explicitly reclassified later.

## Real-System Status

Real collection infrastructure:

- Current real-data entrypoint: `data/collect/collect_real_shadow_pilot.py`.
- The JSONL/image/pose recording path can produce synchronized records when the
  hardware is behaving.
- Video reconstruction for real captures is available at
  `tools/render_real_shadow_video.py`.

Hardware status:

- CAN/Piper bring-up must be checked first when `can0` is stopped, missing, or
  erroring.
- Removing the physical S-bend helped but did not fully solve guidewire
  delivery.
- Supervisor feedback: continue trying the original collection method; if it
  still fails mechanically, try a vessel model without the S-bend.
- The UDP feeder protocol works, but repeated onsite trials were unreliable.
  Treat it as fallback/debug, not the mainline data source.
- Clean real expert collection is currently limited by physical guidewire
  delivery reliability, not mainly by the dataset format.

## Simulation Status

Current simulator work has shifted from proving BC rollouts to improving
real-system correspondence:

- The guidewire visual route was adjusted toward a realistic entry/outlet path,
  including the S-bend-like entrance shape.
- The guidewire should visually start near the real feeder/Piper outlet rather
  than appear from the vessel branch.
- Background, lighting, camera visibility, and line thickness were tuned toward
  real side/top captures.
- Elite magnetic guidance now targets a front-up position relative to the
  guidewire tip or registered route point. This preserves the real-system
  intuition that the magnet should pull forward-upward, not merely upward from
  a point directly above the guidewire head.
- Branchs-like visual rendering helped compare appearance/statistics, but did
  not by itself solve sim-to-real transfer.

Registered-geometry wallguard:

- Acceptable only as a conservative Elite-only controller/safety layer.
- Strong Piper hold/away correction is rejected as a formal mainline mechanism
  unless a real observable and real controller path are defined.

Isaac Sim:

- Remains Plan C / renderer-spike backup.
- Do not migrate physics/control there by default; migration cost is high and
  the current bottleneck is still semantic and real-system alignment.

## Current Model Judgment

Do not continue optimizing the small BC model as the main research route.
Do start VLA-style algorithm feasibility work in parallel with data collection
and simulator alignment.

Initial algorithm scope:

- small multimodal/VLA-style model first;
- prefer 3B or smaller when feasible, 7B as the early-test upper bound;
- use openpi/pi0.5-style as the first practical pi-family baseline;
- add pi0.7-style context conditioning, with tactile/contact estimator outputs
  as the first extra context channel;
- include explicit tactile-like inputs derived from image collision/distance
  estimation;
- do not use exact MuJoCo contact/wall truth as a policy input.

Current algorithm feasibility status:

- The front-up pi-style pack has trained successfully on both local smoke
  code and the Ubuntu 4090 machine.
- A first OpenPI-compatible 32D state/action adapter exists. It is a framework
  compatibility layer only; the project-preferred target remains Elite
  TCP-delta regression plus Piper intent classification.
- The first 4090 mixed-head OpenPI-compatible smoke run reached best reported
  epoch 7/8 with `val_piper_acc=0.685`, `val_action_mae=0.3050`, and
  `val_elite_mae=0.3321`.
- Tactile/contact ablations currently produce metrics similar to the full
  context run. The interface supports tactile conditioning, but this synthetic
  tactile signal has not yet shown a clear performance contribution.
- A tactile-signal audit found why: `estimated_contact_flag` is constant zero
  in the current pack, and the varying confidence/distance fields are weakly
  correlated with Piper feed labels and Elite movement scale. Current tactile
  fields should be treated as interface placeholders until contact-rich or
  real-image-estimator labels exist.
- A posthoc contact-rich control pack with about `15%` positive contact-like
  flags still did not show a clear full-context advantage over
  `drop_tactile_all`. The weak tactile effect is therefore not only an all-zero
  binary-label artifact; the current action target/dataset likely does not make
  this distance-like tactile signal important.
- A small LeRobotDataset export smoke has passed on the Ubuntu 4090 machine.
  The exporter preserves `action_32` as framework compatibility and keeps
  `elite_tcp_delta_6d` plus `piper_intent_id` as explicit project targets.
- A LeRobotDataset-loader smoke model has also trained successfully on the
  full exported dataset, verifying the official dataset loader path before
  moving to OpenPI model code.
- The first official LeRobot PI05 adapter probes have passed on the Ubuntu
  4090 machine. The preprocess probe verified LeRobotDataset fields,
  task/state tokenization, and CUDA tensor placement; the forward/backward
  probe with `paligemma_variant=gemma_2b` and
  `action_expert_variant=gemma_300m` computed `loss=1.4252` and gradients on
  the project data. This is an interface feasibility result only, not a
  real-system validation result.
- A short PI05 adapter-training smoke on shuffled samples did not show clear
  validation improvement over 40 steps (`initial_val_loss=1.4671`,
  `final_val_loss=1.5115`), but the fixed-noise micro-batch overfit diagnostic
  passed: one repeated training batch dropped from `train_loss=1.2840` to
  `6.55e-05` over 80 steps, while the small record-level validation subset
  moved from `1.4953` to `1.1758`. Current conclusion: the PI05 optimizer and
  data/model wiring are functional; broader policy-quality claims require
  longer, less noisy training/evaluation.
- A longer normal PI05 adapter-training smoke with episode-level split and
  fixed evaluation seed showed stable validation loss reduction over 300 steps:
  `initial_val_loss=1.4810`, `final_val_loss=0.9837`, train/val samples
  `1649/399`, and validation checkpoints decreasing from `1.4622` at step 25
  to `0.9837` at step 300. Current conclusion: the official PI05 adapter is
  trainable on the current synthetic LeRobot dataset; this remains sim-data
  algorithm feasibility, not real-system validation.
- A checkpointed 1000-step PI05 adapter run on the Ubuntu 4090 machine saved
  `final_policy.pt` and continued the validation-loss reduction:
  `initial_val_loss=1.4696`, `final_val_loss=0.3158`, with validation checkpoints
  decreasing from `1.3084` at step 100 to `0.3158` at step 1000. Current
  conclusion: the PI05 training path can optimize substantially on the current
  synthetic LeRobot export. The next evidence needed is held-out open-loop
  action prediction, not only diffusion training loss.
- Full held-out open-loop sampling from that checkpoint is weaker than the
  loss curve suggests: over `616` validation samples, `piper_acc=0.5032` versus
  a majority baseline of `0.5292`, and Elite TCP-delta MAE has mean `0.6283`
  with median `0.2451`. Current conclusion: the checkpoint learns something
  under the PI05 training loss, but the current `action_32` sampling/decoding
  path is not yet a convincing mixed-action policy. Diagnose inference-step
  count, stochastic sampling variance, per-dimension errors, and whether Piper
  should remain a separate classification head instead of a one-hot
  compatibility slice.
- Increasing PI05 sampling to `--num-inference-steps 50` did not materially
  fix open-loop action quality: over the same `616` validation samples,
  `piper_acc=0.5081` versus majority baseline `0.5292`, and Elite TCP-delta
  MAE stayed about the same (`mean=0.6296`, `median=0.2472`). Current
  conclusion: the weak Piper result is unlikely to be just too few diffusion
  inference steps. The next algorithm-side fix should prioritize the project
  contract directly: Elite continuous action plus Piper explicit classifier,
  rather than treating Piper as a continuous one-hot slice inside `action_32`.
- A source-level probe of the installed LeRobot `0.4.4` PI05 implementation
  found that `PI05Policy.forward()` and `predict_action_chunk()` consume images
  and language tokens but do not consume `observation.state`. Therefore the
  current official PI05 runs do not yet condition Elite actions on robot state
  or tactile/contact fields, even though those fields are present in the
  LeRobotDataset export.
- A first project mixed-head path now reuses the frozen 1000-step PI05
  checkpoint for Elite and trains an explicit Piper classifier from shared
  PI05 image/language prefix embeddings plus normalized `observation.state`.
  Remote 2-step train and 2-record held-out smokes passed, including a small
  head-only checkpoint, but their single-class validation subset is not a
  quality result. The next evidence needed is a full episode-held-out run with
  class coverage and comparison against the `0.5292` Piper majority baseline.
- The 1000-step frozen-head run was a negative result. The selected training
  records contained Piper classes `[0, 1660, 1820]`; the 128-record validation
  slice contained `[0, 60, 68]`. Final Piper accuracy was `0.53125`, exactly
  the validation majority baseline, balanced accuracy stayed `0.5`, and every
  final prediction was class 2. Intermediate evaluations only switched between
  predicting every sample as class 1 or every sample as class 2. Do not run the
  full mixed-head open-loop evaluator on this checkpoint.
- The current 1000-step PI05 checkpoint is an architecture/loss feasibility
  checkpoint, not a pretrained PI05 fine-tune. The training path constructs
  `PI05Policy(config)` directly, does not call `from_pretrained`, and freezes
  the randomly initialized PaliGemma path with `train_expert_only=True`. The
  remote cache does not currently contain `lerobot/pi05_base`. Loading and
  verifying a reproducible pretrained PI05 base now precedes further
  mixed-head tuning or state/tactile injection.

Use BC to answer narrow questions:

- can the observation/action schema close a loop at all;
- do output scales look reasonable;
- does a changed dataset or controller break obvious behavior;
- does sim-to-real shadow evaluation expose distribution mismatch.

Do not treat BC/VLA success on current sim data as proof of simulator realism
or real policy readiness.

## Current Blockers

- Real guidewire delivery remains physically unreliable.
- Existing onsite data is useful for calibration and diagnostics, but not yet a
  clean large expert dataset.
- Sim-to-real mismatch remains visible in image domain, guidewire delivery
  scale, and hardware dynamics.
- Any formal expert must avoid simulation-only privileged signals.

## Next Direction

Default next direction:

1. Use current real captures as calibration/alignment anchors and limited
   algorithm smoke-test data, not as a clean large expert dataset.
2. Continue the original real collection route when onsite; if the physical
   guidewire still cannot pass reliably, test a no-S-bend vessel model.
3. Keep improving the MuJoCo formal simulator toward real anchors:
   guidewire entry geometry, camera domain, Piper feed event scale, Elite
   motion scale, and observable contact/tip estimators.
4. Continue the VLA-style feasibility track by adding an explicit
   `lerobot/pi05_base` pretrained-loading path and verifying which weights are
   actually loaded before further training. Only after that should the Piper
   `state_only` versus `prefix_state` feature ablation and state/tactile token
   injection be tested. Do not evaluate the collapsed v1 mixed-head checkpoint
   further. Add real-domain calibration anchors before any sim-to-real claim.
5. Generate formal synthetic data only after the expert/controller path passes
   real-observable validity.
6. Use small BC only as a regression check on the resulting data/control loop.
7. Return onsite when the physical guidewire delivery path or fixture has a
   concrete improvement to test.

## Where Details Live

- Old state details:
  `docs/archive/project-state-before-entry-slimming-2026-07-13.md`
- Old handoff details:
  `docs/archive/handoff-before-entry-slimming-2026-07-13.md`
- Commands:
  `docs/commands.md`
- Data/action semantics:
  `docs/data-and-action-schema.md`
- VLA/control contract:
  `docs/vla-target-interface.md`, `docs/control-layer-contract.md`,
  `docs/vla-pi-baseline-plan.md`
- Real collection:
  `docs/real-data-collection-checklist.md`,
  `docs/real-alignment-20260707-calibration-audit.md`
- Formal validity:
  `docs/simulation-expert-validity-audit.md`,
  `docs/real-observable-interface-audit.md`
