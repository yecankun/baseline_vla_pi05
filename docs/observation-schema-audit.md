# Observation Schema Audit

Last updated: 2026-06-25

## Purpose

The current question is not whether the small BC baseline has enough inputs to
fit a rollout. The main question is whether the simulator observation schema can
match the real system closely enough for synthetic data to be useful.

The current small BC model is not image-only. It consumes:

- side camera image;
- top camera image;
- a structured state vector built in `simulation/train_dual_arm_baseline.py`.

The state vector currently includes many fields that are easy to obtain in
MuJoCo but not automatically available on the real system. Formal training
should separate real-available inputs from estimator-dependent and
simulation-privileged inputs.

## Real Piper Baseline

The inherited real Piper inference path in
`intervention/bc_piper/inference.py` calls:

```python
small_action = predict(image, pose, piper_step)
```

and then executes:

```python
if small_action == 1:
    piper.step_forward(pause_time=0.8)
    piper_step += 1
```

Therefore, a real-aligned Piper policy may output a binary `0/1` command, but
its inputs should be real-available observations such as camera image, Elite
pose, and Piper step count. The policy should not be forced to infer a hidden
simulator-only scheduler phase from images alone.

The current evidence indicates that senior `pose` is not six Elite joint angles.
It is most likely Elite TCP 6D pose:

```text
[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
```

`branchs/branch1/path/pose1.txt` starts with millimeter-scale first-three values
and near-constant radian-scale last-three values. The collection code writes
`ec.current_pose`, and Elite motion is executed by calling
`ec.get_inverse_kinematic(pose=...)` before `move_joint(...)`. See
`docs/real-observation-schema-alignment.md`.

## Current Baseline Inputs

The baseline state vector currently uses:

```text
tip_pos
heading
target_pos
contact_normal
distance_to_wall
contact_strength
contact_flag
path_progress
piper_step
piper_insertion_length
elirobot_pose
lateral_offset
path_tangent
local_radius
piper_joints
elite_joints
step_norm
task_left
task_right
```

Together with side/top images, this is a strong synthetic state input, but it
mixes real robot state, estimated perception state, route-plan metadata, tactile
signals, and exact simulator internals.

## Field Audit

| Field | Current meaning | Real-system status | Formal-training judgment |
| --- | --- | --- | --- |
| `frames/side`, `frames/top` | Rendered camera images | Real cameras exist, calibration/domain still needs validation | Allowed |
| `piper_step` | Piper executed feed count / progress count | Directly present in senior code | Allowed |
| `piper_insertion_length` | Piper insertion or actuator travel estimate | Plausible from actuator/joint state | Allowed if mapped to real Piper state |
| `piper_joints` | Piper joint state | Real robot state should be readable | Allowed if synchronized |
| `elite_joints` | Elite joint state | Real robot state should be readable | Allowed |
| `elirobot_pose` | Elite tool or pose estimate | Real Elite pose is available through robot FK/calibration | Allowed if frame is calibrated |
| `task_left`, `task_right` | Branch/task id | Real task command is known | Allowed |
| `step_norm` | Episode clock normalized by max steps | Controller-side clock is available, but semantics must match real loop | Allowed only if defined as real controller time/step |
| `target_pos` | Current target point | Can be real if from registered route/task plan | Allowed only as registered-route metadata, not exact tip feedback |
| `tip_pos` | Exact guidewire tip position | Needs camera/perception estimator in real system | Needs estimator; not allowed as raw simulator truth |
| `heading` | Exact guidewire tip heading | Needs image/perception estimator | Needs estimator; not allowed as raw simulator truth |
| `contact_flag` | Exact contact state | Needs tactile/contact sensor or estimator | Needs sensor/estimator |
| `contact_strength` | Exact contact magnitude | Needs tactile/contact sensor or estimator | Needs sensor/estimator |
| `contact_normal` | Exact wall/contact normal | Needs tactile/contact plus geometry estimator | Needs sensor/estimator; high risk |
| `distance_to_wall` | Exact tip-wall distance | Needs registered vessel + tip estimate | Needs estimator; otherwise privileged |
| `path_progress` | Exact nearest route progress | Needs registered route + tip estimate | Privileged unless derived from real tip estimator |
| `path_tangent` | Local route tangent at progress | Needs registered route + progress estimate | Privileged unless derived from real route/progress estimator |
| `local_radius` | Local vessel radius | Needs registered vessel model and progress estimate | Privileged unless derived from registered model |
| `lateral_offset` | Tip offset in local vessel frame | Needs registered route + exact/estimated tip | Privileged unless derived from real tip estimator |

## Recommended Formal Observation Levels

Use explicit observation levels instead of a single mixed state vector.

### Level A: Real-Robot Direct

This is closest to the senior Piper path and should be the first formal-safe
baseline:

```text
side/top image
Elite pose or Elite joints
Piper step count
Piper insertion or Piper joint state
task id
controller time/step if defined
```

This level does not use exact guidewire tip, exact wall distance, or exact route
progress.

### Level B: Real-Robot With Estimators

This level is allowed only after the estimator or sensor path is explicitly
defined:

```text
estimated tip position
estimated tip heading
estimated contact/tactile signal
estimated distance-to-wall or safety margin
registered-route target/progress derived from the estimated tip
```

If these are used in synthetic training, the dataset should record that they are
estimator-level fields, not raw simulator truth.

### Level C: Simulation Oracle

These fields should stay diagnostic unless a real equivalent is implemented:

```text
exact tip_pos
exact heading
exact contact normal/strength
exact distance_to_wall
exact path_progress
exact local route frame
exact lateral_offset
```

They are useful for debugging expert quality and simulator behavior, but they
should not silently enter formal policy inputs.

## Implications For The Current BC Results

The signed-Piper BC failures should not be read as proof that the expert data is
invalid. They show that the current small model is trained on a mixed
observation schema and still fails to learn the control mapping robustly.

For Piper specifically:

- The model may output `0=hold` and `1=feed`; this matches the senior code.
- The inputs should include real controller state such as `pose` and
  `piper_step`.
- If `period/width/phase-lock` is part of the controller design, it should be
  explicit controller state or label-generation logic, not a hidden simulator
  phase that the model must infer from one frame.

For tactile/contact:

- Tactile-like fields are important for the future VLA/tactile direction.
- They should be introduced as sensor/estimator fields with a real acquisition
  plan, not as exact MuJoCo wall/contact truth.

## Required Code Direction

The training/evaluation path now supports an explicit observation-schema switch:

```text
--observation-schema full_sim_state
--observation-schema real_direct
--observation-schema senior_piper_real_like
```

Planned later extensions:

```text
--observation-schema real_direct_plus_estimated_tip
--observation-schema real_direct_plus_tactile
```

Initial implemented target:

```text
real_direct = images + Elite pose/joints + Piper step/insertion/joints + task id
senior_piper_real_like = images + Elite TCP 6D pose + Piper step + task id
```

Keep `full_sim_state` for diagnostics and legacy comparison, but do not use it
as the default formal sim-to-real training input.

Important nuance: `real_direct` is a real-direct approximation, not an exact
replica of the senior Piper input. The senior interface appears to consume a 6D
Elite TCP pose, while current `real_direct` uses simulated `elirobot_pose` as a
3D tool point plus robot joint states.

The simulator now writes `elite_tcp_pose_6d` and
`robot_state.elite_tcp_pose_6d`. For old datasets without that field,
`senior_piper_real_like` falls back to `elirobot_pose * 1000 + zero
orientation`, so it remains trainable on prior manifests but should be treated
as a compatibility diagnostic rather than full 6D evidence.

The action side now has a matching real-style option:

```text
--elite-action-representation tcp_delta
```

New collection code writes `action.elite_tcp_pose_6d` and
`action.elite_tcp_delta_6d`. The current labels use the target TCP translation
from the expert IK result and keep the current TCP orientation, mirroring the
inherited Elite inference path that applies only translation deltas before
calling inverse kinematics.

## TCP-6D Validation Result

The first dataset with real `elite_tcp_pose_6d` fields is:

```text
simulation_output/formal_route_plan_tip_signedpiper_dataset_v9_tcp6d
```

It validates the collection side of the schema: all samples contain
`elite_tcp_pose_6d` in both `state` and `robot_state`, the dataset remains
`formal_valid`, accepts `40/40` episodes, has complete image references
(`3614/3614`), keeps contact p95 at `0.0`, and preserves the signed-Piper
feed/hold distribution (`951/856`).

The corresponding small BC checkpoint is:

```text
simulation_output/baseline_formal_route_plan_tip_signedpiper_v9_tcp6d_seniorlike_pipercls
```

Open-loop diagnostics are poor:

```text
piper_sign_mismatch_fraction: 0.371
piper predicted labels: hold=1452, feed=355
elite_linf median/p95: ~0.133 / 0.318
predicted Elite target-step p95: ~0.169
expert Elite target-step p95: ~0.022
predicted Elite delta-from-current p95: ~0.315
expert Elite delta-from-current p95: ~0.0084
```

This result should be interpreted as a target-representation mismatch, not as a
failure of the formal expert data. `senior_piper_real_like` gives the model TCP
pose, but the current action target is an absolute six-joint Elite command.
The inherited real Elite code appears to use TCP pose/delta followed by IK.
Therefore the next real-aligned action experiment should consider an explicit
TCP-delta/IK control path or a real-implementable controller state, rather than
continuing to train TCP-pose observations directly against absolute joint
targets.

## Initial Validation Result

The first `real_direct` training test used the signed-Piper v8 formal dataset:

```text
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v8_real_direct_pipercls
manifest: simulation_output/formal_route_plan_tip_signedpiper_dataset_v8/manifest.json
observation_schema: real_direct
state_dim: 22
piper_head: step_classification
```

Open-loop diagnostic result:

```text
piper_sign_mismatch_fraction: 0.3818
piper feed/hold accuracy: ~61.8%
predicted labels: hold=1364, feed=443
confusion: hold->feed 91/856, feed->hold 599/951
elite_linf median/p95: ~0.122 / 0.196
predicted Elite target-step p95: ~0.141
expert Elite target-step p95: ~0.022
predicted Elite delta-from-current p95: ~0.193
expert Elite delta-from-current p95: ~0.0084
```

This is not a usable policy checkpoint, but it is a useful schema result.
Compared with the full 44-dimensional simulator-state Piper-classification
checkpoint, the real-direct model did not meaningfully lose Piper accuracy or
Elite target quality. Therefore the current small BC failure should not be
explained mainly as "the model needed privileged simulator fields." The stronger
conclusion is that the current baseline architecture/action representation is
not reliable enough, and the project should next align the exact real input
semantics before adding estimator/tactile fields.

## Open Validation Items

- Confirm the exact frame convention and normalization of senior `pose`; current
  evidence says it is Elite TCP 6D pose, likely
  `[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]`, not six joint angles.
- Confirm whether real Piper insertion length or only `piper_step` is available.
- Define the future tactile input: binary contact, scalar pressure, contact
  direction, or richer tactile image/signal.
- Decide whether route geometry will be available online through registration,
  and if so which route-derived fields are allowed.
- Audit real image/domain mismatch before scaling synthetic image training.
