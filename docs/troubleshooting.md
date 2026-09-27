# Troubleshooting

Last updated: 2026-06-17

## Elite Arm Appears to Flash or Jump

First check whether the jump is in policy target, executed joints, or rendering.

Run:

```powershell
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py `
  simulation_output\your_rollout_dir `
  --no-plots
```

Inspect:

```text
elite_target_joint_step_linf
elite_executed_joint_step_linf
elite_target_to_executed_joint_l2
elite_tool_step
magnetic_step
tip_to_elite_tool
tip_to_magnetic
```

Interpretation:

- Large `elite_target_joint_step_linf`: policy output is spiky.
- Large `elite_executed_joint_step_linf`: environment/control execution is
  moving too abruptly.
- Small `magnetic_step` but large `elite_tool_step`: visible robot motion is
  larger than effective magnetic motion.
- Small `tip_to_magnetic` but large `tip_to_elite_tool`: magnetic effective
  point tracks the tip, but visible Elite tool is offset.

## Rollout Succeeds Numerically but Video Looks Wrong

Do not accept success metrics alone.

Check:

- rollout video;
- `min_segment_distance_to_wall`;
- `max_contact_strength`;
- Elite diagnostics;
- whether debug markers/path tubes are visible;
- whether camera view hides important geometry behind robot bodies.

## `min_segment_distance_to_wall` Is Negative

Small negatives are possible in the current approximation. Use magnitude and
video judgment.

Concerning signs:

- large negative values;
- persistent wall hugging;
- visible segment crossing vessel wall;
- high contact strength for long periods.

## Guidewire Looks Like Two Lines

Possible causes:

- debug/reference path tubes are visible;
- previous trajectory traces are being rendered;
- wire rendering has duplicated geometry;
- video overlays include both current wire and reference/debug line.

For formal collection, use:

```text
--hide-path-tubes
--hide-tool-markers
```

## Training Fails with Missing Image File

Likely cause:

- manifest contains samples whose frame files were not written;
- partial failed episode remained in the dataset.

Check:

```text
manifest.json
episode_*/frames/side/
episode_*/frames/top/
```

Formal collection should avoid keeping failed/incomplete samples unless
explicitly intended.

## OpenGL `glutInit` Error on Windows

Symptom:

```text
OpenGL.error.NullFunctionError: Attempt to call an undefined function glutInit
```

Meaning:

Python package `PyOpenGL` is installed, but the system GLUT/freeglut runtime is
missing or not visible on PATH.

This mainly affects the old `utils/interface/opengl_interface.py`, not the
current MuJoCo mainline.

## `from stl import mesh` Cannot Find `mesh`

Use the `numpy-stl` package, not a different package named `stl`.

Expected import:

```python
from stl import mesh
```

If IDE still complains after installing, verify the active interpreter is the
project `.venv`.

## Rollout Gets Stuck at Branch

Possible causes:

- guidewire too soft or underdamped;
- magnetic guidance too weak;
- Piper feed phase not suitable;
- policy enters out-of-distribution state;
- path/route target or branch direction mismatch.

Useful checks:

- progress logs every 100 steps;
- `contact_strength`;
- `min_segment_distance_to_wall`;
- video from overview/top/side;
- compare left and right separately.

