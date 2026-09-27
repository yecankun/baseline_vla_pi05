---
name: project-2026-guidewire-mujoco
description: Project-specific workflow guardrails for the inherited Project 2026 guidewire intervention repository. Use when working in D:\PycharmProjects\project_2026 or when the user asks about MuJoCo guidewire simulation, dual-arm data collection, Piper/Elite action schema, behavior-cloning training, rollout videos, Elite jump diagnostics, wall/contact metrics, or handoff documentation for this project.
---

# Project 2026 Guidewire MuJoCo

## First Step

When this skill triggers, inspect the local project docs before making non-trivial changes:

```text
AGENTS.md
docs/project-state.md
docs/decision-log.md
docs/commands.md
docs/handoff.md
docs/data-and-action-schema.md
docs/vla-target-interface.md
docs/control-layer-contract.md
docs/real-observable-interface-audit.md
docs/estimator-tactile-interface-spec.md
docs/visual-tip-contact-estimator-plan.md
docs/simulation-assumptions.md
docs/troubleshooting.md
docs/experiment-registry.md
docs/open-questions.md
docs/weekly-meeting-log.md
```

Read only the docs relevant to the request when context is tight, but always read
`AGENTS.md` and `docs/project-state.md` for collection, training, rollout, or
diagnostic work.

## Core Guardrails

- Treat simulator realism and real-system correspondence as the highest
  priority. A small BC model completing rollouts in simulation is only a
  regression check, not the project goal.
- Treat `docs/vla-target-interface.md` as the canonical target
  observation/action contract before starting more small-BC tuning. The current
  VLA-facing direction is `observation + task instruction -> Elite TCP delta +
  Piper discrete command`; BC experiments should test that interface rather
  than inventing a temporary action space just to pass MuJoCo rollout.
- Treat `docs/control-layer-contract.md` as the canonical policy/controller
  split. The current decision is hybrid: policy predicts Elite TCP delta and
  Piper intent, while the controller owns IK, timing, limits, cooldown, safety,
  and executed-command logs.
- Before accepting a data-generation expert, controller, force, safety rule, or
  rollout intervention, ask whether the same mechanism could plausibly exist on
  the real system using real observations, real robot control, and reasonable
  calibration.
- Mechanisms that depend on simulation-only privileged state, exact tip
  position, exact vessel frame, exact wall distance/contact, hidden centerline
  correction, or oracle tip-relative anchors may be used for diagnostics, but
  must not be treated as formal data-generation behavior unless a matching real
  sensing/control path is explicitly defined.
- Treat the active mainline as the MuJoCo physical-guidance simulation, not the
  older 2D/centerline environments.
- Keep the current action mode as `piper_feed_elite_joint` unless the user
  explicitly decides to change it.
- Interpret current simulator `piper_feed` as a scalar feed/retract command
  near the vessel entrance, but do not treat it as real-action aligned. In the
  route-plan expert, negative simulator `piper_feed` can be an internal finite
  feeder-stroke reset/repositioning action rather than true guidewire
  retraction; current wire integration clamps negative Piper command to zero.
  Do not automatically map negative `piper_feed` to real `retract`. Senior's
  real Piper inference code uses binary stop/advance semantics:
  `0=stop/hold`, `1=piper.step_forward(...)`. A retract action is still a
  plausible and likely useful real-control capability, but it must be explicitly
  defined in the real Piper interface and data labels before formal sim data is
  treated as real-action aligned.
- Do not map one real `piper.step_forward(pause_time=0.8)` call to one MuJoCo
  `env.step`. For current formal tip-line collection, use the Piper primitive
  controller (`--piper-primitive-steps`, `--piper-primitive-feed-value`) and
  event semantics when `--piper-primitive-steps > 1`: sparse
  `piper_step_command=1` labels mark real-style feed events, while
  `controller_state.piper_executed_feed` records the multi-step simulator
  execution during the busy primitive.
- Interpret `elite_joints` as six Elite joint targets. Do not silently convert
  the compatibility/execution schema to magnetic-target commands. For the
  VLA-facing learning target, prefer Elite TCP delta followed by IK, as
  documented in `docs/vla-target-interface.md`.
- Treat senior data under `branchs/` as rough real-scene/Elite trajectory
  reference, not guidewire centerline supervision.
- For the current tip-centric route, preserve the hard-elastic guidewire
  assumption: the magnetically guided head is the primary controlled/contact
  state, while formal images should render a continuous line-shaped guidewire
  rather than the older ball/segment-looking visual. Do not return to full
  flexible-body guidewire physics as the default next step unless the user
  intentionally changes direction.
- Use `simulation.collect_formal_tip_line_guidance` for new formal mainline
  collection. The older `simulation.collect_tip_guided_wire` and
  `simulation.collect_mujoco_physical_guidance` entrypoints are legacy/
  diagnostic only; do not call them for new mainline data unless explicitly
  reproducing or comparing an older route.
- For current formal tip-line collection, default to an earlier guidewire start
  range (`--start-fraction-min 0.42 --start-fraction-max 0.54`). The older
  `0.58-0.70` range puts the red guidewire head too close to the
  bifurcation/branch area at episode start.
- Hide debug markers and path tubes for formal data collection.
- Use `--formal-data` for collection commands and `--formal-eval` for rollout
  commands when checking formal sim-to-real validity. These flags fail closed:
  current oracle experts and rollout interventions such as `--elite-tip-anchor`
  are rejected rather than silently treated as valid data.
- Do not default to `piper_feed_negative_loss_weight=4.0`; it amplified Elite
  joint-output instability in the current diagnostics.
- Prefer `.\\.venv\\Scripts\\python.exe` for project commands on Windows.
- Do not directly start long-running data collection, training, full rollout, or
  video-rendering jobs unless the user explicitly asks. Provide copyable
  commands for the user to run, then ask for resulting logs/metrics/files.

## Standard Workflow

For new data or model work:

1. Confirm current docs and output naming in `docs/experiment-registry.md`.
2. Provide the command to collect formal mainline data with
   `simulation.collect_formal_tip_line_guidance` unless explicitly asked to run
   it. Use legacy collectors only for explicit comparisons/regressions.
3. Inspect manifest success rate, rejected episodes, wall/contact metrics, and a
   few videos if available.
4. Provide the command to train with `simulation.train_dual_arm_baseline` unless
   explicitly asked to run it.
5. Provide the command to roll out with `simulation.eval_mujoco_guided_wire_rollout`
   unless explicitly asked to run it.
6. Run `tools/analyze_mujoco_elite_rollout.py`.
7. Judge results using both metrics and video. Do not trust either alone.
8. Update project docs only when the work changes durable project state.

## Documentation Update Policy

Do not update all docs every turn. This wastes context and creates unnecessary
churn. Update docs only when there is a durable change such as:

- a new key experiment result or negative result;
- a changed next-step recommendation;
- a new recurring command, script, or diagnostic workflow;
- a decision that affects action semantics, data strategy, simulator realism,
  or handoff instructions.

When docs do need updates, edit only the relevant files. Usually this means one
or two targeted docs, not the full documentation set. Keep purely exploratory
analysis in the chat unless it changes the project handoff or future workflow.

## Weekly Meeting Fact Log

Use `docs/weekly-meeting-log.md` as the durable source for group-meeting
reporting. This file is an append-only fact log, not a polished weekly summary.

Append to it when there is durable reporting value, such as:

- a meaningful implementation, collection, training, rollout, or diagnostic
  milestone;
- a new finding that changes how previous work should be interpreted;
- a failed experiment that creates a reusable negative conclusion;
- a project-direction, priority, or next-step change;
- the start or end of onsite real-system work;
- a reusable command, workflow, checklist, hardware setup, or recovery process;
- supervisor or group-meeting feedback that changes the plan;
- the preparation step before a weekly group meeting.

Write short factual entries with evidence paths and tags such as
`[simulation]`, `[real-system]`, `[data]`, `[model]`, `[decision]`, and
`[blocker]`. Do not write broad summaries there; generate meeting summaries
from the accumulated facts when the user asks.

Do not update it for ordinary trial-and-error, small parameter tweaks,
temporary ideas, or work that does not affect future reporting or decisions.

## Git And GitHub Workflow

- Treat GitHub as the home for code, durable docs, configs, and lightweight
  reference assets, not for `simulation_output/`, `branchs/`, local
  environments, or routine model/video artifacts.
- Use `main` directly for small low-risk changes such as docs, commands,
  comments, or localized fixes that do not change the agreed simulation route or
  action semantics.
- Prefer a branch for environment-dynamics changes, action-schema changes,
  expert/control logic changes, rollout execution-layer changes, parallel
  experiments, or refactors that may need comparison before merge.
- Good branch prefixes in this repo are `feat/`, `fix/`, `exp/`, and `docs/`.
- Treat GitHub primarily as a stable rollback/sync point, not a place for every
  small local tweak. Local commits can be finer grained; pushes should mark
  meaningful milestones.
- Push to GitHub after important updates such as a confirmed new mainline
  entrypoint, changed formal-data validity rules, a durable project-direction
  decision, or a validated dataset/model/control milestone. Also push before a
  long collection/training run only when the run depends on a confirmed code
  state worth preserving as a rollback point.
- Do not push immediately for small parameter tweaks, smoke-only experiments,
  wording-only edits, or intermediate trial-and-error changes. Batch them into
  the next meaningful milestone commit/push once the result is accepted.
- Avoid noisy micro-commits and avoid pushing local/generated artifacts by
  accident. Check `git status --short` before commit/push when the working tree
  may include experiment debris.

## Diagnostics Flow

When the user reports Elite flashing, robot jump, guidewire wall contact, rollout
failure, or suspicious success:

1. Run or ask the user to run `tools/analyze_mujoco_elite_rollout.py`.
2. Separate policy output spikes from execution/rendering artifacts using:
   - `elite_target_joint_step_linf`
   - `elite_executed_joint_step_linf`
   - `elite_target_to_executed_joint_l2`
   - `elite_tool_step`
   - `magnetic_step`
   - `tip_to_elite_tool`
   - `tip_to_magnetic`
3. Treat small negative `min_segment_distance_to_wall` as a warning signal, not
   automatic failure. Check magnitude and video.
4. When judging realism, compare visible `elirobot_pose` / `magnetic_pose`
   against `simulation_output/real_control_bandwidth`, not only against
   rollout success. Step p95, acceleration p95, jerk p95, direction reversals,
   and visible tool drift all matter.
5. If a weighted model jumps more than an unweighted model, suspect training
   strategy before blaming MuJoCo itself.

## Current Recommended Direction

The next agreed direction is:

1. Start from `docs/simulation-expert-validity-audit.md` before scaling data or
   optimizing another BC rollout. The immediate priority is to distinguish
   real-implementable expert/control mechanisms from simulation-only oracle
   mechanisms.
2. Use `docs/real-observable-interface-audit.md` before another formal training
   run. Policy inputs must be direct real signals, explicit real controller
   state, or estimator/tactile signals with a defined real acquisition path;
   exact MuJoCo tip/contact/wall/route-frame fields remain diagnostic.
3. Use `docs/estimator-tactile-interface-spec.md` and
   `tools/audit_observation_provenance.py` before adding guidewire-tip/contact
   fields to any formal policy input. Current working assumption: real
   contact/tactile-like information is produced by a visual image-distance or
   contour-vs-vessel-mask contact estimator from image input, not necessarily by
   a neural network and not by a dedicated tactile sensor. Estimator fields
   require provenance; `full_sim_state` remains diagnostic because it consumes
   privileged MuJoCo truth.
4. Use `docs/visual-tip-contact-estimator-plan.md` for the current
   no-new-hardware perception route: rule-based visual distance estimation is
   the first auditable baseline/fallback, while a learned visual estimator is
   the higher-capacity VLA path if it preserves the same output contract and
   provenance.
5. Continue from the current MuJoCo physical-guidance and tip-centric outputs,
   but treat `--elite-tip-anchor` and similar tip-relative controllers as
   diagnostics unless the real system will implement equivalent tip estimation
   and Elite low-level control.
6. Use `branchs/branch1` and `branchs/branch2` as real Elite/magnetic-arm motion
   anchors, not guidewire trajectory labels.
7. Tune visible Elite motion against the real pose step scale, roughly 5 mm p95,
   using `docs/sim-vs-real-alignment.md` and
   `tools/compare_sim_real_alignment.py`.
8. Use `simulation_output/real_control_bandwidth` as the real control-bandwidth
   anchor. Current real pose statistics are roughly: position-step p95
   `5.07 mm/frame`, acceleration p95 `2.98 mm/frame^2`, jerk p95
   `2.84 mm/frame^3`, and direction-reversal median `0.0`.
9. Do not blindly increase `--elite-smoothness-weight` or tighten low-pass
   smoothing. Smooth4 improved the right branch but worsened the left branch,
   and `rate010_smooth025_r1` only slightly improved video despite matching the
   step-scale metric.
10. Next, prefer a rollout bandwidth comparison diagnostic and then a realistic
   execution/control layer with joint velocity/acceleration/jerk limits while
   preserving the external `piper_feed_elite_joint` schema.
11. If model-side training changes are needed, avoid naive unconstrained
   action-delta training; use an absolute anchor or bounded/rate-limited target
   formulation only if the anchor is framed as a real-implementable controller
   or kept diagnostic.
12. Begin resolving the Piper semantic gap: senior real data has binary
   stop/advance labels (`0=hold`, `1=step_forward`), while MuJoCo currently uses
   continuous normalized simulator-side `piper_feed`. Do not remove retract just
   to match the old BC interface, but also do not infer real retract from
   simulator feeder reset. Define a real-implementable hold/feed/retract or
   signed-step Piper interface if retract is part of the intended real
   controller.
13. Before another mainline BC rollout, decide whether Piper is directly
    predicted by the future VLA as `hold/feed/retract`, handled by an explicit
    real controller/state machine, or represented by a hybrid policy intent plus
    safety/timing controller. The current decision is the hybrid policy intent
    plus controller route. Do not ask a small image BC model to infer a hidden
    feed/hold schedule that is absent from its observations.
14. New collection code writes `piper_step_command` alongside `piper_feed` for
    `piper_feed_elite_joint` samples: `-1=retract`, `0=hold`, `1=feed`.
    Route-plan experts should use low-frequency `feed/hold` scheduling via
    `--piper-command-period` and `--piper-command-width`, not alternating
    feed/reset every sample. Low-frequency scheduling must still keep a
    realistic feed duty cycle; `period=125,width=5` made labels sparse but
    starved physical feed and failed all smoke attempts, while
    `period=250,width=125` finished before any hold phase and degraded contact.
    Binary labels and execution magnitude are separate: route-plan
    `piper_step_command=1` should emit `piper_feed_command = --piper-cmd`, not
    always full `1.0`. If hold phases are introduced, also check whether route-
    plan progress continues advancing while Piper is holding; use
    `--route-plan-command-phase-lock` as a diagnostic/real-plausible scheduler
    candidate before scaling, but remember that it compensates by duty cycle, so
    the nominal route-plan step may need to be reduced. Existing BC code still
    reads `piper_feed`; use the signed-step label, feed/hold duty cycle,
    execution magnitude, transition cadence, contact, and `tip_to_magnetic` for
    real-action-alignment diagnostics before changing training.
15. Current clean signed-Piper route-plan setting is:
    `--route-plan-step 0.15 --piper-cmd 0.70 --piper-command-period 40
    --piper-command-width 20 --route-plan-command-phase-lock`. The 2-episode
    v8 smoke and 8-episode v8 stress check both stayed `formal_valid`, reached
    env success, had zero contact, complete images, positive wall clearance, and
    real feed/hold labels. Treat this as the current formal expert setting to
    scale before any BC training.
16. For the registered-geometry wall-safety controller, use the accepted
    Elite-only form as the current candidate: do not hold Piper and do not add
    active away-wall TCP motion by default; only remove the predicted Elite TCP
    delta component toward the estimated wall normal. The stronger variant that
    held Piper and added repeated away-wall correction broke tip-magnet coupling
    and caused rollout failure, so keep it rejected unless intentionally
    re-tested as a diagnostic.
17. For sim-to-real work, start with shadow-mode validation before commanding
    real robots. Use `tools/real_shadow_policy_adapter.py` to run the current
    checkpoint on real-style side/top images, Elite TCP 6D pose, Piper state,
    and estimator fields; record both raw policy action and guarded
    controller action. Judge model ability from raw outputs and intervention
    rate, not only from the guarded system result.
18. For new real pilot capture, prefer
    `data/collect/collect_real_shadow_pilot.py` over the older hard-coded
    `data_collect_mode1.py` / `data_collect_mode2.py` scripts. The new
    collector keeps the useful senior signals but writes the current
    `records.jsonl` shadow schema directly. It is log-only for Piper by
    default and does not move or count Piper steps in that mode. If Piper is
    moved by an operator or another program, use `--external-piper-control`;
    only use `--enable-piper-control` after the physical setup and supervision
    are explicitly confirmed. For senior mode1-style capture, use
    `--piper-label-source auto_periodic` with the real feed period/count so the
    collector records periodic Piper feed events without manual key presses.
    After any real pilot, run `tools/validate_real_shadow_pilot.py` before
    shadow inference.
19. Senior `branchs/` data can be used for real-data diagnostics, but it is not
    a complete current VLA/sim-to-real input because it has one camera and lacks
    estimated 3D tip/registered-geometry fields. Use
    `tools/build_branchs_training_manifest.py` for a branchs-native
    senior-like baseline before drawing conclusions about transfer.
20. Single-camera schema matching is not sufficient by itself: the sim-trained
    single-camera senior-like checkpoint still predicted Piper feed for every
    `branchs` shadow frame, while the branchs-native checkpoint learned the same
    real labels. Treat the remaining gap as visual/domain distribution plus
    real label/control timing mismatch, not merely as a missing second camera.
21. When sim-trained checkpoints fail on `branchs`, run
    `tools/audit_sim_real_visual_domain.py` before another BC rollout-tuning
    loop. The first audit showed bright clean synthetic side images versus
    darker green-background real camera frames, so rendering/camera/domain
    alignment or aligned real-data collection is the next lever.
22. Current option A remains MuJoCo-first, and `--render-domain-preset
    branchs_like_v1` is the accepted branchs-like render preset. It improves
    side-camera luminance/saturation/edge statistics and was visually accepted,
    but the branchs-like single-camera senior-like checkpoint still collapses
    to all-feed on `branchs` transfer. Treat rendering alignment as useful but
    insufficient by itself. The next lever is real-observation coverage and
    real/sim data alignment: synchronized real side/top images if available,
    real Elite TCP pose, Piper controller state, target/task metadata, and
    estimator/contact fields with explicit provenance. Remaining visual gaps are
    mainly glass-vessel reflections, real-camera blur, fixture/crop/occlusion,
    and Piper label/control timing.
23. Keep Isaac Sim or a remote-workstation renderer-only spike as backup option
    C. Do not start a full simulator migration unless MuJoCo branchs-like
    rendering/camera alignment clearly hits a ceiling and the user explicitly
    chooses that route.
    If the user chooses Plan C, assume Isaac Sim is on a separate host unless
    told otherwise. Use `tools/export_isaac_renderer_spike.py` locally to export
    a portable renderer-only package with assets, reference images, samples, and
    `isaac_build_stage.py`; run Isaac only on the remote/other host to build USD
    stages. Do not require local Isaac installation and do not migrate physics,
    control, collection, or training in this spike.

Do not jump directly to OpenVLA or Isaac Sim unless the user intentionally
changes the research plan.
