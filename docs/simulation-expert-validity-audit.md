# Simulation Expert Validity Audit

Last updated: 2026-06-23

## Highest Priority Rule

The project goal is not to make a small BC model look good in simulation. The
goal is to build a simulator that can generate useful data because large-scale
real collection is difficult.

Therefore, any expert policy, controller, force, safety rule, or rollout
intervention must be judged by this standard first:

```text
Can this mechanism plausibly exist in the real system, using real observations,
real robot control, and reasonable calibration?
```

If the answer is no, the mechanism may still be useful for diagnostics or
oracle studies, but it must not be treated as validated formal data-generation
behavior.

## Audit Scope

This audit covers the current MuJoCo expert/data path:

- `simulation.collect_mujoco_physical_guidance`
- `simulation.collect_tip_guided_wire`
- `simulation.mujoco_guided_wire_env.MuJoCoMagneticGuideExpert`
- `simulation.tip_guided_wire_env.TipGuidedWireEnv`
- rollout-only `--elite-tip-anchor` in
  `simulation.eval_mujoco_guided_wire_rollout`

## Current Findings

### Acceptable Real-System Anchors

These are defensible as real-system constraints or measurements:

- The real vessel mesh and route are acceptable as a known task geometry proxy.
- The magnetic point is attached to the Elite end tool. This matches the user's
  clarification of the real system and should remain the active semantics.
- Elite commands are represented as six joint targets. This preserves the
  current sim-to-real preference better than direct magnetic target commands.
- Piper is a local feed/retract command near the entrance, not a mobile
  guidewire-tip controller.
- Real `branchs` data should anchor visible Elite/magnetic-arm motion scale,
  smoothness, and Piper semantic gaps. It must not be used as guidewire path
  truth.
- Joint velocity, acceleration, or jerk limits are acceptable only when they are
  tied to real robot bandwidth measurements or hardware limits.

### Plausible But Needs Validation

These approximations may be usable, but should be marked as assumptions:

- The tip-centric abstraction is plausible because the real guidewire is
  relatively hard and elastic, and the main observed safety event is whether the
  guided head contacts the vessel. It still needs real-image validation.
- Using a known vessel centerline/route for simulator setup is acceptable, but
  using it as hidden online control information is only valid if the real system
  would also have a registered vessel model and guidewire-tip estimate.
- Contact or wall-distance signals can support diagnostics and tactile-style
  future data. They should not be silently used as policy inputs or expert
  feedback unless the real system has an equivalent sensing path.

### Privileged Or Oracle Mechanisms

These mechanisms currently use information that is available in simulation but
not automatically available in the real system:

- `MuJoCoMagneticGuideExpert` computes a desired Elite target from exact
  `env.tip`, exact path progress, local path frame, centerline lookahead,
  radius, and exact tactile/contact state.
- `TipMagneticGuideExpert` inherits the same expert behavior for the
  tip-centric environment.
- `TipGuidedWireEnv._advance_tip` directly uses local vessel frame, exact
  magnetic-to-tip vector, exact wall contact, and explicit centering/contact
  relief terms to update the tip.
- Rollout `--elite-tip-anchor` computes an expert-style tip-relative Elite
  anchor from exact tip position, local frame, lookahead route point, correction
  toward the local center, and IK. It then clips the learned Elite joint target
  around that anchor.
- `piper_feed_phase_guard`, `tactile_safety`, and scripted Piper feed variants
  use exact simulated contact, wall distance, insertion, or phase state. They
  are useful diagnostics but are only real-valid if the same sensing/controller
  exists on hardware.

## Interpretation Of The Anchor Result

`baseline_tip_guided_wire_v1_elitedelta_anchor0020_rollout_r1` is the best
current diagnostic rollout for the tip-centric + Elite-delta line:

```text
left/right success: True / True
target_jump_p95: about 0.0159 / 0.0130
tip_to_magnetic median: about 20.6 / 24.9 mm
wall penetration: none
video review: acceptable
```

This is useful because it shows that magnetic-tool-to-tip drift is a dominant
failure mode for the delta model.

It is not, by itself, proof that the simulator is ready for formal data
generation. The anchor uses a privileged tip-relative expert prior unless the
real system is explicitly designed to estimate the guidewire tip and run the
same low-level Elite controller.

## Current Go / No-Go Judgment

Current status:

```text
go for diagnostics: yes
go for formal data-generation mainline: not yet
```

Reason:

- The tip-centric expert data is clean and valuable as a controlled abstraction.
- The BC loop now closes, and anchor diagnostics identify a real failure mode.
- However, the expert and anchor logic still rely on privileged simulator state.
- Formal synthetic data should not be scaled until the expert/control stack is
  rewritten or explicitly framed as a real-implementable controller.

## Required Next Step

Before scaling data collection, define one of these two routes:

1. Real-implementable controller route:
   - Use camera or sensor-derived guidewire-tip estimation.
   - Use registered vessel geometry if available.
   - Convert the current tip-relative anchor/expert into a declared low-level
     Elite controller.
   - Record this controller as part of the real system architecture, not as a
     hidden simulation force.

2. Pure simulation physics route:
   - Remove rollout-only or expert-only privileged corrections from the formal
     data-generation path.
   - Keep only physical magnetic influence, Piper feed, wall interaction, and
     robot kinematics.
   - Accept that BC rollout quality may be worse until the model and data are
     improved.

Until one route is chosen, use anchor and similar mechanisms only for
diagnostics and ablations.

## Implemented Guardrails

The current scripts now expose explicit validity switches:

```text
simulation.collect_mujoco_physical_guidance --formal-data
simulation.collect_tip_guided_wire --formal-data
simulation.eval_mujoco_guided_wire_rollout --formal-eval
```

Current behavior:

- `--formal-data` rejects the current MuJoCo/tip-centric oracle experts because
  they use exact simulated tip state, vessel frame, lookahead, contact/wall
  signals, and IK.
- Collection without `--formal-data` still works for diagnostics and ablations,
  but writes `validity_audit` into `manifest.json`.
- `--formal-eval` rejects rollout interventions such as `--elite-tip-anchor`,
  `--tactile-safety`, `--scripted-piper-feed`, `--piper-feed-phase-guard`, and
  visible debug path/tool markers.
- Rollout without `--formal-eval` still works for diagnostics and writes
  `validity_audit` into each rollout `meta.json`.

This makes Route 2 operational: formal paths fail closed, while diagnostic
oracle tools remain available with explicit labeling.

## Artifact Audit Tool

Use this tool to audit existing datasets and rollouts:

```powershell
.\.venv\Scripts\python.exe -B tools\audit_simulation_validity.py `
  simulation_output\tip_guided_wire_dataset_v1 `
  simulation_output\baseline_tip_guided_wire_v1_elitedelta_anchor0020_rollout_r1 `
  --out docs\_simulation_validity_audit_current
```

The tool classifies artifacts as:

- `formal_valid`: no known oracle/diagnostic mechanism was detected by the
  recorded metadata.
- `diagnostic_oracle`: useful for diagnostics or ablations, not formal
  data-generation evidence.
- `unknown_legacy`: old artifact lacks enough metadata; treat as not
  formal-valid until rerun or manually reviewed.

Current audit output:

```text
docs/_simulation_validity_audit_current/simulation_validity_audit.md
docs/_simulation_validity_audit_current/simulation_validity_audit.json
```

Current result:

- `simulation_output/tip_guided_wire_dataset_v1`: `diagnostic_oracle`
- `simulation_output/mujoco_physical_feed_action_dataset_magnet_attached_envsuccess_v1`: `diagnostic_oracle`
- `simulation_output/baseline_tip_guided_wire_v1_elitedelta_anchor0020_rollout_r1`: `diagnostic_oracle`
- `simulation_output/baseline_tip_guided_wire_v1_elitedelta_rollout_nolimit_r1`: `unknown_legacy`

## Route-Plan Formal Probe

The collection scripts now support:

```text
--expert-mode oracle|route_plan
```

`oracle` is the previous expert and remains diagnostic only.

`route_plan` is a Route 2 formal-validity probe. It avoids exact tip position
and exact wall/contact feedback. It uses:

- pre-registered vessel route;
- scheduled route progress;
- Piper insertion state;
- Elite IK to reach a planned route point.

This makes `--formal-data --expert-mode route_plan` pass the validity guard.
It does not mean the expert is good enough. The first tiny implementation smoke
wrote:

```text
simulation_output/_formal_route_plan_tip_smoke
status: formal_valid
accepted_episodes: 0/1
steps: 20
progress_gain: 5.52
max_contact: 0.000
```

Interpretation: the formal-validity path now runs and is auditable, but the
route-plan expert still needs a longer smoke before it can be judged as a useful
data generator.

A longer tip-centric formal route-plan smoke then wrote:

```text
simulation_output/formal_route_plan_tip_smoke_v1
status: formal_valid
accepted_episodes: 4/4
samples: 150
left/right env_success: 2/2 and 2/2
steps median: 183.5
final_progress median: 122.13, target_progress 131/132
distance_to_target median: 0.04197 m
min_tip_distance_to_wall min: 0.000546 m
stall_fraction: 0.0
tip_to_magnetic median range: 0.050-0.059 m
contact_p95 range: 0.568-0.601
```

Interpretation: route-plan became a promising formal-valid baseline because it
can reach `env_success` without exact tip/contact oracle feedback, but this
first parameter setting was not good enough to scale: contact was sustained and
high, and the magnetic/Elite tool sat about `5-7 cm` from the guidewire tip.

A small route-plan sweep then tested slower route advance and smaller Elite
lead:

```text
formal_route_plan_tip_sweep_step020_ahead010:
  status: formal_valid
  accepted_episodes: 4/4
  env_success: 4/4
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.001548 m
  tip_to_magnetic median range: 0.0036-0.0047 m
  tip_to_magnetic p95 range: 0.0061-0.0106 m
  steps median: 197.5

formal_route_plan_tip_sweep_step020_ahead006:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.002541 m
  tip_to_magnetic median range: 0.0043-0.0054 m
  steps median: 200.5

formal_route_plan_tip_sweep_step024_ahead006:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.001494 m
  tip_to_magnetic median range: 0.0055-0.0059 m
  steps median: 185.5

formal_route_plan_tip_sweep_step024_ahead010:
  contact_p95: 0.0
  max_contact_strength: 0.008
  min_tip_distance_to_wall min: 0.001375 m
  tip_to_magnetic median range: 0.0094-0.0098 m
  steps median: 183.5
```

Current judgment: `--route-plan-step 0.20 --elite-ahead 0.010` is the best
balanced formal route-plan candidate. It preserves success, removes contact in
the smoke, and keeps the magnetic/Elite tool close to the tip without relying
on exact tip/contact oracle feedback.

The moderate dataset with this setting has now been collected:

```text
dataset: simulation_output/formal_route_plan_tip_dataset_step020_ahead010_v1
status: formal_valid
accepted episodes: 40/40
left/right: 20/20
rejected attempts: 0
samples: 1360
image refs: 2720/2720 exist
env_success: 40/40
steps median/p95/max: 164.5 / 205 / 212
min_tip_distance_to_wall min/median: 1.386 / 1.906 mm
max_contact_strength max: 0.0
contact_p95 max: 0.0
stall_fraction max: 0.0
slow_fraction max: 0.0
tip_to_magnetic median: 3.47 mm
tip_to_magnetic episode-p95 median/max: 6.66 / 11.08 mm
sample tip_to_magnetic p95/max: 8.55 / 13.26 mm
elite action-step episode-p95 median/max: 0.01577 / 0.01701
```

This passes the current formal expert-data quality gate for a small BC
loop-closure check. The remaining caveat is not formal validity, but whether a
small model can learn the route-plan expert without reintroducing Elite target
jitter or magnet/tip drift.

The current dedicated expert-data report is:

```text
docs/formal-expert-dataset-validity-report.md
```

It freezes the small BC result as a diagnostic baseline and recommends shifting
the next default step back to expert-data coverage, simulator realism, and
sim-to-real correspondence rather than further BC rollout tuning.
