# VLA Elite Real-Pose Reference Audit

Date: 2026-07-17

## Scope And Eligibility

Audited real pilot:

```text
collected_data/real_pilot_20260707_left_linked_elite_piper_matched_001/records.jsonl
```

This episode failed and is not accepted as real training data. Its Elite
trajectory is used only as motion-reference evidence for deciding whether the
first VLA baseline needs an orientation output.

The audit does not reinterpret the Piper labels, contact fields, guidewire
path, or episode outcome. It does not make this dataset training-eligible.

Generated evidence:

```text
simulation_output/real_pilot_20260707_left_linked_elite_piper_matched_001_elite_pose_audit.json
```

Reproduce locally:

```powershell
.\.venv\Scripts\python.exe tools\audit_real_pilot_elite_pose.py `
  collected_data\real_pilot_20260707_left_linked_elite_piper_matched_001 `
  --path-file path\path1_pose.txt `
  --out simulation_output\real_pilot_20260707_left_linked_elite_piper_matched_001_elite_pose_audit.json
```

## Source Semantics

The pilot manifest defines Elite pose as:

```text
[x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]
```

It also records:

```text
zero_orientation_delta: true
Elite path mode: automatic execution
Elite path file: path/path1_pose.txt
```

The path file has 20 rows and exactly three columns. It defines XYZ waypoint
positions only. The collector keeps the initial Elite orientation when forming
the requested six-dimensional TCP targets.

## Position Motion

The 300 records cover `59.81 s`. The executed TCP position spans:

```text
X: 56.37 mm
Y: 195.22 mm
Z: 17.94 mm
```

There are 13 recorded executed-position updates above `0.01 mm`. Twelve are
normal path advances. The path-file target spacing is tightly bounded:

```text
minimum: 17.0963 mm
median:  17.3168 mm
maximum: 17.6504 mm
```

The first update is a `69.2541 mm` reposition from the initial robot pose to
path index 0. It is an initialization/reset move, not a representative policy
control step. The run reaches path indices 0 through 12 before the failed
episode ends.

The approximately `17.3 mm` value is a waypoint/event-scale motion reference.
It is not a per-camera-frame policy target.

## Orientation Motion

All 275 non-null requested TCP targets have exactly the same RPY. Requested
orientation span is zero on all three axes.

Executed orientation variation across the complete pilot is also negligible:

```text
full span by RPY axis: 0.00221 / 0.00110 / 0.00173 degrees
maximum absolute drift from initial: 0.00162 / 0.00110 / 0.00095 degrees
maximum consecutive rotation norm: 0.00229 degrees
position-update events above 0.01 degrees rotation: 0 / 13
```

This is consistent with numerical/servo measurement variation around a fixed
orientation, not an intentional rotation trajectory.

The recorded `reference_action.elite_tcp_delta_6d` likewise has zero rotation
for all 300 records. It has 13 non-zero translation events corresponding to
the path updates.

## Synchronization Limitation

This pilot is not a reliable per-frame image/pose alignment source:

```text
image-pose lag median: 33.03 ms
image-pose lag P95: 1994.88 ms
image-pose lag maximum: 3983.74 ms
Elite pose marked stale: 144 / 300 records (48%)
validator warnings above 100 ms: 144
```

The discrete executed-pose updates therefore reflect the collector's pose
refresh timing, not instantaneous 69 mm or 17 mm robot motion between adjacent
camera frames. This is another reason not to train from this failed pilot.

## Algorithm Decision

The first bounded VLA baseline should predict:

```text
Elite TCP translation delta 3D + Piper binary hold/feed intent
```

Elite orientation remains controller-owned and fixed to the configured magnet
orientation. Do not add synthetic rotation labels merely to fill a six-
dimensional tensor.

For compatibility, `elite_tcp_delta_6d` may remain in datasets and adapters,
with dimensions 3:6 fixed to zero. Training loss, model selection, and primary
evaluation must use translation dimensions 0:3. Aggregate six-dimensional
improvement must not be used as the main Elite result.

This decision is limited to the present fixed-magnet real setup. It should be
reopened only if later real control intentionally commands orientation changes
and those commands have trustworthy executed-pose evidence.

`estimated_contact_flag` remains outside the current baseline scope. No
contact-label work is requested by this audit.
