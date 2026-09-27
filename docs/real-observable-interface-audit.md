# Real Observable Interface Audit

Last updated: 2026-06-26

## Purpose

The current boundary is no longer "can a small BC model complete a rollout."
The boundary is:

```text
Can a future policy receive this signal on the real system, and can the
controller execute this action on the real system?
```

This document turns that boundary into a short checklist for future simulator
data, BC interface tests, and VLA-facing dataset design.

Highest priority:

```text
real-observable policy inputs + real-executable actions > small-BC rollout success
```

## Current Evidence

The latest clean control-state/TCP-delta dataset is:

```text
simulation_output/formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta
```

It validates the data/logging path:

```text
accepted/env_success: 40/40
samples: 1796
missing images/controller_state/elite_tcp_delta: 0
contact_p95/max: 0.0 / 0.0
piper controller pairs: hold->hold 884, feed->feed 912
elite controller types: tcp_delta 1756, reset 40
```

The matching small BC checkpoint did not become a useful policy:

```text
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v12_controlstate_tcpdelta_seniorlike_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v12_controlstate_tcpdelta_seniorlike_pipercls_open_loop_diag
piper_sign_mismatch_fraction: 0.361
elite_linf median/p95: ~0.930 / 1.567 in TCP pose-delta metric space
expert/predicted Elite target-step p95: ~7.11 / 7.63
```

Interpretation:

```text
The interface/logging path is now useful.
The small BC model is not the main lever.
Do not keep tuning BC rollout performance as the default next step.
```

## Formal Policy Input Tiers

### Tier 0: Direct Real Signals

These are allowed for formal policy inputs now, assuming synchronization and
calibration are handled:

| Signal | Current sim field | Real path | Judgment |
| --- | --- | --- | --- |
| side/top images | `frames/side`, `frames/top` | real cameras | allowed |
| task instruction | `task`, `instruction` | operator/task command | allowed |
| Elite TCP 6D pose | `elite_tcp_pose_6d`, `robot_state.elite_tcp_pose_6d` | robot controller/FK | allowed |
| Elite joints | `robot_state.elite_joints` | robot controller | allowed as state/log |
| Piper feed count | `piper_step` | senior code already tracks it | allowed |
| Piper insertion/actuator state | `piper_insertion_length`, Piper joints | actuator or controller state | allowed if real mapping is defined |

The current closest implemented schema is:

```text
--observation-schema senior_piper_real_like
```

It is not sufficient to make the small BC strong, but it is the best current
real-input contract test:

```text
images + Elite TCP 6D pose + Piper step + task id
```

### Tier 1: Real Controller State

These are allowed only if they are emitted by the real control layer, not
reconstructed from simulator truth:

| Signal | Current sim/log field | Requirement |
| --- | --- | --- |
| last Piper intent | `controller_state.piper_intent` | real controller logs requested intent |
| executed Piper command | `controller_state.piper_executed_command` | real controller logs accepted command |
| Piper motion state | `controller_state.piper_motion_state` | real controller has idle/feed/retract/cooldown state |
| Piper step count/insertion | `controller_state.piper_step_count`, `controller_state.piper_insertion_length` | real hardware/controller exposes it |
| last Elite requested/executed TCP | `controller_state.elite_requested_tcp_pose_6d`, `controller_state.elite_executed_tcp_pose_6d` | real controller logs request and execution |
| IK or limit status | `controller_state.elite_target_limited` | real controller can report rejection/clipping |

Important rule:

```text
Controller state may be a policy input only if the real controller exposes the
same state online. Otherwise it is a diagnostic log only.
```

### Tier 2: Estimator Or Sensor Signals

These are promising for the future VLA/tactile direction, but need a real
acquisition path before they become formal inputs:

| Signal | Current sim field | Required real path |
| --- | --- | --- |
| guidewire tip position | `tip_pos` | vision estimator or tracking system |
| guidewire heading | `heading` | vision estimator |
| contact flag/strength | `contact_flag`, `contact_strength` | validated visual image-distance/contact estimator, force proxy, or tactile sensor if later available |
| contact direction/normal | `contact_direction`, `contact_normal` | visual/tactile/geometry estimator; high risk unless explicitly supported |
| wall safety margin | `distance_to_wall`, `segment_min_distance_to_wall` | registered vessel model + estimated tip |
| route progress | `path_progress` | registered route + estimated tip |
| local route frame/radius | `path_tangent`, `local_radius`, `lateral_offset` | registered route + estimated tip/progress |

If these fields are used in synthetic training, the dataset must label them as
estimator-level signals. Do not pass exact MuJoCo truth under a real-sounding
name.

### Tier 3: Simulation-Only Privileged Signals

These are diagnostic only unless promoted through a real estimator or real
controller design:

```text
exact MuJoCo tip position
exact MuJoCo wall distance/contact
exact route-frame progress from simulator truth
exact local vessel frame at the true tip
oracle tip-relative Elite anchor
hidden route-plan phase that is not real controller state
rollout-only tactile safety or scripted Piper interventions
```

They may be useful for debugging why an expert, controller, or rollout failed.
They must not silently enter formal policy inputs.

## Formal Action Contract

Current VLA-facing action target:

```text
Elite: elite_tcp_delta_6d
Piper: hold | feed | retract intent
```

Execution belongs to the controller:

```text
Elite TCP delta -> IK -> joint command -> limits/safety -> executed joints
Piper intent -> bounded real feed/hold/retract primitive -> executed command
```

Current compatibility action mode remains:

```text
piper_feed_elite_joint
```

Do not remove compatibility fields such as `elite_joints` or `piper_feed`, but
do not mistake them for the future semantic policy target.

## Pre-Training Gate

Before training another model intended as formal sim-to-real evidence, answer:

1. Does every policy input fall under Tier 0 or explicitly approved Tier 1/2?
2. If Tier 2 fields are used, what real sensor/estimator produces them?
3. Are exact MuJoCo tip/contact/wall/route-frame fields excluded from the
   policy input?
4. Is Piper action labeled as discrete intent, not a simulator feeder reset?
5. Is Elite action labeled as TCP delta, with IK and safety handled by the
   controller?
6. Are requested policy labels separated from executed controller logs?
7. Does `--formal-data` or `--formal-eval` fail closed if the run depends on a
   privileged oracle mechanism?

If the answer is unclear, keep the run diagnostic rather than formal.

## Recommended Next Work

Do not run another default small-BC rollout from the v12 checkpoint.

Next mainline work should choose and implement one of these real-observable
routes:

1. **Real-direct route:** keep policy inputs to images, Elite TCP pose, Piper
   state, task, and real controller state. Use BC only as a weak interface
   regression test.
2. **Estimator route:** define a real guidewire-tip/contact estimator first,
   then expose estimated tip/contact/route fields to both real and synthetic
   data.
3. **Visual-distance contact route:** define the image/mask distance signal that
   the real system can actually provide, then make simulation emit the same
   binary flag, optional pixel distance, confidence, latency, and error profile.

The strongest next research step is likely the estimator/tactile route, because
the current real-direct small BC has already shown that images plus robot state
alone are not enough for this small model to learn the full mapping reliably.
