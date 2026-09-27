# Real Data Collection Checklist

Last updated: 2026-07-04

## Purpose

The next useful real-system step is not another small-BC rollout. It is a
minimal real-data capture that can support sim-to-real shadow evaluation.

Target:

```text
real observations + real controller/action logs -> shadow policy prediction
```

The dataset does not need to be large at first. It must be synchronized,
auditable, and aligned with the current VLA-facing interface:

```text
input:  images + Elite TCP pose + Piper state + task/target + estimator fields
output: Elite TCP delta + Piper hold/feed/retract intent
```

## Minimum Pilot Scope

Start with a small calibration/pilot capture before collecting a large dataset:

```text
tasks: left and right
paths per task: 2-3
frames per path: 100-300 usable synchronized frames
operator mode: supervised/manual or senior policy is acceptable
robot command mode: log-only/shadow-safe first; no new learned policy commands
```

Success criterion for this pilot:

```text
The records can be converted into a JSONL shadow input and run through
tools/real_shadow_policy_adapter.py without missing-image, missing-pose, or
schema failures.
```

## Required Per-Frame Signals

These fields are the minimum for a real-aligned shadow dataset:

| Field | Required | Source | Notes |
| --- | --- | --- | --- |
| `timestamp` | yes | capture process | Use one monotonic clock if possible. |
| `task` | yes | operator/script | `left` or `right`; avoid inferring from folder name only. |
| `step` | yes | capture process | Monotonic frame index after synchronization. |
| `side_image` | yes | side camera | Save raw frame or minimally compressed PNG. |
| `top_image` | strongly preferred | top camera | If unavailable, explicitly duplicate side and mark it. |
| `elite_tcp_pose_6d` | yes | Elite controller/FK | `[x_mm, y_mm, z_mm, rx_rad, ry_rad, rz_rad]`. |
| `piper_step` | yes | Piper controller | Accumulated feed-step count or senior-compatible state. |
| `piper_motion_state` | preferred | Piper controller | `idle/feed/retract/cooldown` if available. |
| `target/task metadata` | yes | operator/script | Target branch or target position/instruction. |
| `reference_action.piper_step_command` | yes | operator/senior policy/controller | `0=hold`, `1=feed`; `-1=retract` only if real retract is defined. |
| `reference_action.elite_tcp_delta_6d` | preferred | controller log | Next target pose minus current pose; translation-first is acceptable. |
| `executed command logs` | preferred | controller | Requested vs executed Elite/Piper commands should be separated. |

Important:

```text
Do not label real data using simulated guidewire centerline, MuJoCo route
progress, or branchs pose interpreted as guidewire path truth.
```

## Estimator / Contact Fields

These fields are not strictly required for the first pilot, but they are the
right target if the future VLA should use tactile/contact-like information
without adding hardware.

| Field | Required for pilot | Source | Notes |
| --- | --- | --- | --- |
| `estimated_contact_flag` | optional | visual estimator | Binary contact-like flag from image distance or learned model. |
| `estimated_image_distance_px` | optional | visual estimator | Prefer continuous distance plus confidence over only binary labels. |
| `contact_estimator_confidence` | optional | visual estimator | Required if estimator output may be missing or occluded. |
| `estimated_tip_pos_3d` | optional | RGB-D/depth/calibration pipeline | Use only if calibration and transform are known. |
| `tip_estimator_confidence` | optional | estimator | Record low confidence instead of fabricating a value. |
| `estimated_wall_margin` | optional | registered geometry estimator | Requires registered vessel geometry and estimated tip. |
| `route_estimator_confidence` | optional | registered route estimator | Required if route-derived fields are used. |

For missing estimator fields, prefer explicit null/low-confidence records:

```json
{
  "estimated_tip_pos_3d": null,
  "tip_estimator_visible": false,
  "tip_estimator_confidence": 0.0,
  "estimated_contact_flag": null,
  "contact_estimator_confidence": 0.0
}
```

## Recommended JSONL Record

Use this structure for real shadow input. It matches
`tools/real_shadow_policy_adapter.py`.

```json
{
  "timestamp": 0.0,
  "task": "left",
  "step": 0,
  "side_image": "frames/side/000000.png",
  "top_image": "frames/top/000000.png",
  "state": {
    "elite_tcp_pose_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "piper_step": 0.0,
    "piper_insertion_length": null,
    "controller_state": {
      "piper_intent": "hold",
      "piper_executed_command": "hold",
      "piper_motion_state": "idle",
      "elite_requested_tcp_pose_6d": null,
      "elite_executed_tcp_pose_6d": null,
      "elite_target_limited": false
    },
    "estimated_tip_pos_3d": null,
    "estimated_tip_heading_3d": null,
    "tip_estimator_confidence": 0.0,
    "estimated_contact_flag": null,
    "estimated_image_distance_px": null,
    "contact_estimator_confidence": 0.0,
    "estimated_wall_margin": null,
    "route_estimator_confidence": 0.0
  },
  "reference_action": {
    "piper_step_command": 0,
    "piper_command_label": "hold",
    "elite_tcp_delta_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "elite_tcp_pose_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "input_quality": {
    "side_visible": true,
    "top_visible": true,
    "tip_visible": null,
    "notes": ""
  }
}
```

## Directory Layout

Recommended pilot layout:

```text
real_pilot_YYYYMMDD/
  manifest.json
  records.jsonl
  calibration/
    camera_intrinsics_side.json
    camera_intrinsics_top.json
    camera_extrinsics.json
    robot_camera_transform.json
  frames/
    side/
      000000.png
    top/
      000000.png
  logs/
    elite_tcp_pose.csv
    piper_state.csv
    controller_commands.csv
  notes.md
```

`manifest.json` should include:

```json
{
  "mode": "real_shadow_pilot",
  "tasks": ["left", "right"],
  "camera_setup": "side_top",
  "pose_semantics": "Elite TCP 6D pose [xyz_mm, rpy_rad]",
  "piper_semantics": "0=hold, 1=feed; retract only if explicitly implemented",
  "synchronization": "timestamp nearest-neighbor or hardware trigger",
  "operator_mode": "manual_or_senior_policy",
  "known_missing_fields": []
}
```

## Synchronization Requirements

The most important quality issue is synchronization, not dataset size.

Minimum acceptable synchronization:

```text
For every image pair, record the nearest Elite TCP pose, Piper state, and
reference action with timestamp deltas.
```

Record these diagnostics:

```text
side_top_time_delta_ms
image_pose_time_delta_ms
image_piper_time_delta_ms
image_action_time_delta_ms
```

Reject or flag frames when:

```text
image_pose_time_delta_ms > 100
side_top_time_delta_ms > 100
image is blurred/black/missing
Elite TCP pose is missing
Piper label is missing
task label is missing
```

The exact threshold can be tightened later, but the pilot must log the deltas.

## Conversion / Shadow Gate

Before using a real pilot dataset as evidence, it should pass these checks:

1. All image paths in `records.jsonl` exist.
2. Every record has `state.elite_tcp_pose_6d` with six numeric values.
3. Every record has `state.piper_step`.
4. Every record has `task` and `step`.
5. Every record has `reference_action.piper_step_command` if comparing labels.
6. If `top_image` is duplicated from side, this is recorded in `manifest.json`.
7. Missing estimator fields are null/low-confidence, not silently zeroed.
8. The dataset can run through `tools/real_shadow_policy_adapter.py`.

## Collection Script

Use the new pilot collector instead of the older hard-coded
`data_collect_mode1.py` / `data_collect_mode2.py` when collecting new real
shadow data:

```text
data/collect/collect_real_shadow_pilot.py
```

For the lab-side command sequence, use:

```text
docs/real-collection-lab-runbook.md
```

It keeps the useful parts of the senior scripts:

```text
save image frames
read Elite current_pose
track Piper step count
record Piper hold/feed/retract intent
```

But it writes the new pilot layout directly:

```text
manifest.json
records.jsonl
frames/side/*.png
frames/top/*.png
logs/elite_tcp_pose.csv
logs/piper_state.csv
logs/controller_commands.csv
```

By default it is log-only for Piper. It records keyboard labels but does not
call Piper hardware and does not advance `piper_step`. If Piper is controlled
by an operator or another program outside this collector, add
`--external-piper-control` so feed/retract labels are counted as externally
executed Piper steps. If the collector itself should command Piper, add
`--enable-piper-control` after confirming the physical setup and supervision.

For senior mode1-style data, use `--piper-label-source auto_periodic` with
`--piper-auto-feed-period 0.8` and `--piper-auto-feed-count` to record periodic
Piper feed events without requiring manual key presses every feed cycle. For
mode2-style data, keep `--piper-label-source keyboard` and record
`--elite-control-source senior_script` or `human` according to who is moving
Elite.

For mode1-style captures, add `--stop-after-auto-feed-count` when the intended
trial length is exactly the configured number of Piper feed primitives. After
each run, inspect `summary.json` and confirm `final_piper_step`,
`auto_feed_count`, `piper_command_counts`, and `robot_command_mode` match the
physical trial.

Local mock smoke:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out simulation_output\_smoke_real_shadow_pilot_mock `
  --task left `
  --side-source mock `
  --top-source duplicate_side `
  --pose-source mock `
  --max-frames 3 `
  --no-preview `
  --sample-period 0.01
```

Lab-side log-only pilot template:

```bash
python data/collect/collect_real_shadow_pilot.py \
  --out real_pilot_YYYYMMDD_left_001 \
  --task left \
  --side-source realsense \
  --side-camera-id 1 \
  --top-source opencv \
  --top-camera-id 0 \
  --pose-source elite \
  --elite-ip 192.168.137.200 \
  --sample-period 0.2 \
  --max-frames 300
```

Keyboard labels during preview:

```text
0 or h: hold
1 or f: feed
r or b: retract label
q or esc: stop
```

If the real Piper should be commanded by the collection script, add
`--enable-piper-control`. Do this only after confirming the physical setup and
operator supervision.

If the real Piper is moved by the operator or by senior's original control code
while this script only records, add `--external-piper-control` instead.

Shadow-mode command template:

```powershell
.\.venv\Scripts\python.exe tools\real_shadow_policy_adapter.py `
  --checkpoint simulation_output\some_checkpoint\best_model.pt `
  --input-jsonl real_pilot_YYYYMMDD\records.jsonl `
  --image-base-dir real_pilot_YYYYMMDD `
  --out simulation_output\real_pilot_YYYYMMDD_shadow_predictions.jsonl `
  --cpu
```

Judge the result by:

```text
raw policy Piper mismatch
raw Elite TCP-delta scale and direction
wallguard/intervention rate if estimator fields are present
failure cases grouped by image quality, task, and Piper state
```

Do not judge only by a guarded controller outcome.

Immediate post-collection validation:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_YYYYMMDD_left_001 `
  --check-readable-images
```

This checks image paths/readability, Elite TCP pose length, Piper labels,
timestamp deltas, duplicated top-camera metadata, and basic label/step counts.

## What The Current `branchs/` Data Lacks

The existing `branchs/` data remains useful, but it is not enough for the next
real-aligned VLA interface by itself:

```text
has: real image frames, Elite TCP 6D pose, Piper 0/1 labels
lacks: true synchronized side/top views, estimator/contact fields, explicit
       controller requested/executed logs, calibration/provenance metadata
```

Therefore `branchs/` is a diagnostic baseline, not a decisive sim-to-real
benchmark.
