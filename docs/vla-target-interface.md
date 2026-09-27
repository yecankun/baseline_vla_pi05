# VLA Target Interface

Last updated: 2026-07-13

## Purpose

This document defines the target observation/action contract that future VLA
models should use. The small BC baseline is only a miniature interface test for
this contract. It should not invent a temporary action space that cannot map
back to the real robots.

Highest priority:

```text
real-system executable interface > small-BC rollout success
```

## Target System Shape

The intended future policy is a multimodal policy for dual-arm guidewire
intervention:

```text
observation + task instruction -> Elite TCP delta + Piper discrete command
```

The policy should decide high-level executable commands. Robot-specific details
such as inverse kinematics, velocity limits, safety checks, and command timing
belong in the execution/controller layer.

The control-layer split is now defined in:

```text
docs/control-layer-contract.md
```

Decision: use a hybrid policy/controller contract. The policy predicts Elite TCP
delta and Piper discrete intent; the controller owns IK, motion limits, command
timing, hardware safety, and executed-command logging.

## Initial Algorithm Choice

The first VLA-style experiments should use small models, not a large frontier
model:

- preferred starting range: 3B parameters or smaller, if tooling and hardware
  allow;
- upper early-test bound: about 7B parameters;
- training style: feasibility-first fine-tuning or adapter tuning on the
  intended observation/action interface;
- evaluation goal: prove that the interface and data pipeline can support a
  multimodal policy, not prove real-system deployment readiness.

The current practical baseline is an openpi/pi0.5-style model. The accepted
design improvement is pi0.7-style context conditioning: keep the reproducible
open baseline, then add project-specific context channels such as
image-derived tactile/contact estimates.

Detailed baseline plan:

```text
docs/vla-pi-baseline-plan.md
```

The model should consume explicit tactile-like contact inputs generated from
images. In the current hardware setting this means visual collision/distance
analysis of the guidewire head against the vessel/wall mask, not a physical
tactile sensor and not exact MuJoCo contact truth.

## Target Observations

Core observations that should be available on the real system:

- side/top camera images;
- current Elite TCP 6D pose: `[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]`;
- Piper state such as feed count, insertion length, or joint/actuator state;
- task instruction such as `enter left branch` or `enter right branch`;
- tactile-like contact fields from visual image-distance/collision analysis,
  such as `estimated_contact_flag`, `contact_estimator_confidence`, and
  `estimated_image_distance_px`;
- optional controller state such as current command phase, if the real
  controller actually uses one.

Senior thesis evidence confirms the minimum autonomous-system observation set:

```text
operative image + magnetic guidance arm 6D pose + accumulated Piper feed count
+ target branch / target position
```

Estimator-level observations are allowed only after their real acquisition path
is defined:

- guidewire-tip position or heading from vision;
- guidewire-head contact signal from the real visual distance/contact estimator;
- distance-to-wall or vessel safety margin;
- registered vessel route/progress derived from estimated tip pose.

The senior thesis describes a real coordinate path for promoting image
observations into estimator fields:

```text
2D guidewire-head localization + depth camera
-> camera-frame 3D tip coordinate
-> hand-eye / least-squares registration
-> robot-base or virtual-environment coordinate
```

Therefore, estimated 3D tip and registered route/branch state are plausible
future policy inputs only when produced by this kind of real estimator pipeline,
not by direct simulator truth.

The following must remain diagnostic unless a real sensing or registration
pipeline is explicitly implemented:

- exact MuJoCo tip position;
- exact wall/contact state;
- exact route-frame progress from simulator truth;
- oracle tip-relative anchor/controller information.

## Target Actions

### Elite

The target learned Elite action is:

```text
elite_tcp_delta_6d
```

Current preferred implementation:

```text
policy predicts TCP translation delta
controller keeps TCP orientation unchanged
IK converts target TCP pose to Elite joints
robot executes move_joint
```

This matches the inherited real-code pattern more closely than predicting
absolute Elite joint targets from TCP-pose observations. The dataset and rollout
logs may still retain `elite_joints` for execution, diagnostics, and
compatibility, but those joints are not the preferred semantic learning target
for the VLA path.

### Piper

The target learned or commanded Piper action should be discrete:

```text
hold
feed
retract
```

Minimum senior-compatible subset:

```text
hold/feed
```

The inherited senior Piper BC uses binary labels:

```text
0 = hold
1 = piper.step_forward(...)
```

The senior thesis autonomous-system interface also uses a binary Piper action:

```text
0 = stationary / hold
1 = execute one forward feed
```

Retract remains a reasonable future capability, but it must be implemented as a
real Piper command and labeled as true rollback/retraction. It must not be
inferred from simulator-only feeder reset strokes.

## Execution Layer

The VLA should not directly own every low-level robot detail.

Elite execution layer:

```text
current TCP pose + predicted TCP delta
-> target TCP pose
-> inverse kinematics
-> joint command
-> velocity/acceleration/jerk or hardware safety limits
```

Piper execution layer:

```text
hold    -> no Piper motion
feed    -> piper.step_forward(...) or equivalent forward step
retract -> explicit real rollback command, if implemented
```

If Piper behavior is better represented as a deterministic scheduler or a
safety state machine, that controller state should be explicit. Do not ask the
image model to infer a hidden schedule that is not present in its observations.

## Small-Model Role

The small BC model is useful only as a contract/regression test:

- can the selected observations support the target actions in open-loop?
- does `elite_tcp_delta_6d` stay close to expert scale and smoothness?
- are Piper labels learnable from real-available inputs?
- do diagnostics fail before expensive rollout/video runs?

The small model should not be tuned simply to complete a MuJoCo rollout if that
requires a fake action semantic, hidden oracle state, privileged contact/wall
feedback, or a controller that cannot exist on the real system.

## Current Evidence

Paper-aligned interface audit:

```text
docs/senior-thesis-interface-audit.md
```

The thesis confirms that the inherited real autonomous interface is:

```text
image + Elite/magnetic-arm 6D pose + Piper feed count + target
-> Elite/magnetic-arm 6D pose delta + Piper hold/feed
```

This strengthens the current VLA-facing contract and narrows the senior-
compatible Piper subset to hold/feed unless a real retract primitive is added.

The v10 TCP-delta data path is the current reference for Elite action semantics:

```text
dataset: simulation_output/formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta
elite_action_representation: tcp_delta
```

Open-loop diagnostics showed that TCP-delta brings the Elite target scale closer
to the expert behavior than absolute joint prediction from senior-like TCP pose
inputs.

Piper remains unresolved. The v10 senior-like and phase-state small BC tests
did not make scheduled feed/hold labels reliable:

```text
senior_piper_real_like mismatch ~= 0.397
senior_piper_real_like_with_phase mismatch ~= 0.371
```

Interpretation:

```text
Do not keep optimizing small BC to infer scheduled Piper feed/hold.
Define Piper as an explicit real controller/policy interface first.
```

## Current Control-Layer Decision

The Piper control contract is hybrid:

```text
policy predicts: hold/feed/retract intent
controller executes: bounded real feed/hold/retract primitive
```

The controller may delay, reject, or clip policy intent because of hardware busy
state, cooldown, insertion limits, IK failure, or safety checks. Those decisions
must be logged and, when useful, exposed as real controller state in the next
observation.

Whichever simulator expert is used, formal data should record the same
policy/controller split that the real system can execute. Do not train the
policy to infer hidden Piper schedule phases from images.
