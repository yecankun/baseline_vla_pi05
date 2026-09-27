# Real Observation Schema Alignment

Last updated: 2026-06-25

## Purpose

This note clarifies what the inherited real-system `pose` input means and how it
should map to simulation observations.

The main point:

```text
senior real `pose` is most likely Elite TCP 6D pose, not six joint angles
```

This matters because the formal simulator policy input should match real
available signals. The learned action may still output Elite joint targets, but
the observation side should not silently replace senior `pose` with unrelated
joint-state semantics.

## Evidence

Example from `branchs/branch1/path/pose1.txt`:

```text
[-336.181452, 251.656322, 321.989323, 3.066174, 0.036473, 0.068426]
```

Across all current `branchs/*/path/pose*.txt` files:

```text
xyz min/max: roughly [-367, -53, 322] to [-234, 252, 377]
xyz span: roughly [133, 305, 55]
rot min/max: roughly [3.066126, 0.036455, 0.068400] to [3.066205, 0.036499, 0.068438]
rot span: roughly [0.000079, 0.000043, 0.000038]
```

The first three values are millimeter-scale Cartesian position. The last three
values are near-constant radian-scale orientation. This is consistent with a TCP
pose `[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]`, not six same-kind joint
angles.

The collection and inference code supports this interpretation:

- `data/collect/data_collect_mode1.py` and `data_collect_mode2.py` write
  `ec.current_pose` into `pose*.txt`.
- They move the Elite arm by calling
  `ec.move_joint(target_joint=ec.get_inverse_kinematic(pose=target_pose), ...)`.
  This means the saved `pose` is passed into inverse kinematics before joint
  motion.
- `intervention/bc_elirobot/inference.py` predicts a large-arm action, adds
  only the first three values to `cur_pose`, keeps the last three orientation
  values unchanged, then calls `get_inverse_kinematic(pose=new_pose)`.
- `intervention/bc_piper/inference.py` passes `ec.current_pose` directly into
  `predict(image, pose, piper_step)`.

The local repo does not contain the external `elite.EC` implementation, so this
cannot be proven from the class definition here. However, the value scale and
all call sites point to TCP pose.

## Senior Real Input Contract

The senior Piper policy input is:

```text
image
pose: Elite TCP 6D pose, likely [x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]
piper_step: discrete Piper feed-count state
task_id: implicit or explicit branch id
```

The senior Piper output is:

```text
0 = hold
1 = piper.step_forward(...)
```

The senior Elite policy uses the same `pose` input and predicts a 6D TCP-pose
delta; in inference only the first three translation components are applied
before inverse kinematics.

## Current Sim Input Gap

The current implemented `real_direct` simulator schema is useful, but it is not
identical to the senior real input contract.

Current `real_direct` includes:

```text
piper_step
piper_insertion_length
elirobot_pose        # currently a 3D tool point in sim coordinates
piper_joints
elite_joints
step_norm
task_left/task_right
```

Senior real input includes a 6D Elite TCP pose, not just a 3D tool point and not
necessarily raw Elite joints. Therefore `real_direct` should be treated as a
real-direct approximation, not a faithful senior-interface replica.

## Implemented Code Direction

A more specific observation schema has been implemented:

```text
--observation-schema senior_piper_real_like
```

Target fields:

```text
side image or matched real camera image
Elite TCP 6D pose in the same unit convention as senior data
piper_step
task id if training both branches together
```

The simulator now exposes:

```text
elite_tcp_pose_6d
robot_state.elite_tcp_pose_6d
```

The first three components are millimeter-scale TCP xyz, and the last three are
roll/pitch/yaw radians from the simulated Elite tool transform. For legacy
datasets that do not contain this field, training falls back to
`elirobot_pose * 1000 + [0, 0, 0]` orientation. This fallback is for
compatibility only; newly collected data should contain the real 6D field.

Current state dimensions:

```text
full_sim_state: 44
real_direct: 22
senior_piper_real_like: 9
```
