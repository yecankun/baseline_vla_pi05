# Data and Action Schema

Last updated: 2026-06-25

## Current Action Mode

The current preferred action mode is:

```text
piper_feed_elite_joint
```

This is the current dataset/execution compatibility mode. The target
VLA-facing policy contract is documented in:

```text
docs/vla-target-interface.md
docs/control-layer-contract.md
```

In short, the future policy target is `Elite TCP delta + Piper discrete command`;
`elite_joints` remain an execution/diagnostic representation after IK.
The current control-layer decision is hybrid: the policy predicts Elite TCP
delta and Piper intent, while the controller owns IK, motion limits, Piper step
timing, cooldown, bounds, and executed-command logging.

Action structure:

```json
{
  "piper_feed": 0.0,
  "piper_step_command": 0,
  "piper_command_label": "hold",
  "elite_joints": {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0
  },
  "elite_tcp_pose_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "elite_tcp_delta_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Meaning:

- `piper_feed`: current real guidewire-level scalar command target for
  collection/training. In the current route-plan expert this is `0=hold` or
  `1=feed`.
- `piper_sim_feed`: optional diagnostic simulator-side feeder stroke amount. A
  negative value may be an internal feeder reset/repositioning stroke in the
  simulator, not necessarily real guidewire retraction.
- `piper_step_command`: real-implementable signed-step label:
  `-1=retract`, `0=hold`, `1=feed`.
- `piper_command_label`: string version of `piper_step_command`.
- `elite_joints`: six target joint values for the Elite magnetic guidance arm.
- `elite_tcp_pose_6d`: real-style Elite TCP target pose
  `[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]`.
- `elite_tcp_delta_6d`: target TCP pose minus current TCP pose. Current
  route-plan labels use translation delta and keep orientation unchanged,
  matching the inherited Elite inference behavior.

The environment still executes through Elite joints internally, but formal
training can now use `--elite-action-representation tcp_delta` so the learned
Elite action matches the senior-style TCP-delta + IK interface.

## Why Piper Is a Scalar

Piper is the guidewire feeding robot. In the real setup, it stays near the vessel
entrance and performs local feed steps. It should not follow the guidewire head
inside the vessel.

The scalar `piper_feed` captures the real command-level effect intended for
learning:

- positive value: feed forward;
- negative value: true retraction only if the expert explicitly emits a real
  retract command;
- near zero: hold.

New formal collection code also writes `piper_step_command` and
`piper_command_label` beside `piper_feed`. The continuous value remains useful
for current MuJoCo execution and existing BC scripts, while the signed-step
label is the candidate real-interface target. For route-plan data, the
simulator-side feeder stroke is recorded separately as `piper_sim_feed`.

Important semantic rule:

- Do not automatically interpret every negative simulator feeder stroke as real
  guidewire retract.
- In the current route-plan expert, periodic negative simulator feeder stroke
  mainly resets the finite simulator feeder travel. Those diagnostic values
  should stay in `piper_sim_feed`, while `piper_feed` and
  `piper_step_command` should represent the real guidewire-level command.
- Retract remains part of the candidate real interface, but it must represent a
  true rollback/retract command rather than a simulator-only actuator reset.

This is not yet real-action aligned. Senior's Piper inference code in
`intervention/bc_piper/inference.py` uses binary actions:

```text
0: stop / hold
1: advance one Piper step via piper.step_forward(...)
```

The real inference loop only calls `piper.step_forward(pause_time=0.8)` when
`small_action == 1`; action `0` performs no Piper movement in that loop. There
is no negative/retract action in this senior BC interface.

This does not make retract invalid as a project goal. A real guidewire system
likely needs a rollback/retract capability. The action-schema gap is that
retract must be defined as a real Piper command and labeled in real-compatible
data, rather than appearing only as a simulator-side continuous negative value.

## Why Elite Has Both TCP Delta And Joints

Elite carries the magnetic guidance tool. The real robot interface and senior's
code expose an important two-layer control structure:

```text
policy/control semantics: Elite TCP pose or TCP delta
execution layer: inverse kinematics -> Elite joint command
```

Senior real code reads `ec.current_pose`, predicts a large-arm action, adds the
translation components to the current TCP pose, keeps orientation, and then
calls `ec.get_inverse_kinematic(pose=...)` before `move_joint(...)`.

Therefore, the real-aligned training target is now:

```text
--elite-action-representation tcp_delta
```

`elite_joints` remain in the dataset and rollout logs for execution,
compatibility, and diagnostics. They should not be treated as the preferred
semantic learning target when using `senior_piper_real_like` observations.

## Observation/State Fields

Each sample contains side/top images and a structured state. Important fields:

```text
tip_pos
heading
target_pos
contact_flag
contact_direction
contact_normal
contact_strength
distance_to_wall
segment_min_distance_to_wall
path_progress
piper_step
piper_insertion_length
boundary_projection_count
elirobot_pose
magnetic_pose
lateral_offset
path_tangent
local_radius
robot_state
```

The baseline model consumes:

- `frames/side/*.png`
- `frames/top/*.png`
- structured state vector built by `simulation/train_dual_arm_baseline.py`

New samples also carry `state.controller_state` as an audit/logging field. It
records the hybrid control-layer split:

```text
piper_intent
piper_executed_command
piper_requested_feed
piper_executed_feed
piper_motion_state
piper_step_count
piper_insertion_length
elite_action_type
elite_tcp_delta_6d
elite_requested_tcp_pose_6d
elite_executed_tcp_pose_6d
elite_requested_joints
elite_executed_joints
elite_target_limited
```

This field is not part of the current default small-BC input vector. Use it to
audit policy intent versus controller execution, and promote only real-available
subfields into an observation schema after an explicit decision.

Important caveat: the current structured vector is a mixed full-simulator state,
not a strictly real-system observation schema. It includes robot state and
Piper step information that can exist on hardware, but it also includes exact
tip, route-frame, wall-distance, and contact fields that require real sensors or
estimators before they can be used in formal sim-to-real training.

Current audit:

```text
docs/observation-schema-audit.md
```

Before more formal model training, prefer an explicit observation schema such as
`real_direct` over the legacy full sim state.

## Image Views

Current formal datasets use two camera views:

```text
frames/side/
frames/top/
```

These correspond to real-system side and top camera intuition, not necessarily
perfect real calibration.

Debug markers and path tubes should be hidden for formal data collection:

```text
--hide-tool-markers
--hide-path-tubes
```

## Senior Real Data

Files under `branchs/` are important but semantically delicate.

Current understanding:

- They include real pose trajectories, real image frames, camera position rows,
  and Piper labels.
- The `path/pose*.txt` rows are most likely Elite TCP 6D pose, not six joint
  angles: `[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]`. The first three values
  are millimeter-scale Cartesian position; the last three values are
  near-constant radian-scale orientation. The collection code writes
  `ec.current_pose` and then uses `get_inverse_kinematic(pose=...)` before
  `move_joint(...)`.
- The pose trajectories record Elite/magnetic-arm TCP motion, not the true
  guidewire path.
- Piper labels are binary `0/1`: `0` means stop/hold, `1` means advance one
  Piper step.
- The current simulator action uses a continuous normalized `piper_feed` scalar,
  including negative retract values. This is a confirmed sim-to-real action
  semantic gap, not merely an unknown label mapping. Retract may be retained, but
  it needs a real-interface definition such as hold/feed/retract or signed
  forward/backward step.
- They were collected in a real scene that is not exactly aligned with the
  current simulation.

Use for:

- scene scale and rough alignment;
- understanding task geometry;
- possible later sim-to-real comparison.
- movement-scale and smoothness sanity checks.

Avoid:

- using them as guidewire centerline labels;
- forcing the simulated guidewire to match them;
- treating them as the simulator's `elite_reference_path`.

Inspection script:

```text
tools/inspect_branch_real_data.py
tools/compare_sim_real_alignment.py
```

Detailed real observation alignment:

```text
docs/real-observation-schema-alignment.md
```

## Dataset Manifest

The manifest stores:

- dataset metadata;
- accepted episodes;
- rejected episodes;
- sample list;
- action mode;
- robot joint names;
- collection settings.

A sample points to image files and stores the corresponding state/action pair.

If `FileNotFoundError` appears during training, check whether all manifest image
paths exist. Earlier scripts had partial-episode issues when frames were missing.
