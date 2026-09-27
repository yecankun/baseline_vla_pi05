# Real Alignment Calibration Audit - 2026-07-07

## Purpose

This audit classifies the 2026-07-07 real-system captures for sim-to-real
calibration. These datasets are not high-quality BC/VLA expert demonstrations.
Their current value is to constrain MuJoCo camera/render/control scales and to
document real physical limitations.

Core interpretation:

```text
real calibration/pilot data -> infer real visual/control scales -> align simulator -> generate scalable synthetic data
```

## Inputs

Local root:

```text
collected_data/
```

Audit outputs:

```text
docs/_real_alignment_20260707_audit/summary.json
docs/_real_alignment_20260707_audit/events.csv
docs/_real_alignment_20260707_audit/contact_sheets/
```

Generated with:

```powershell
.\.venv\Scripts\python.exe tools\audit_real_alignment_calibration.py `
  --root collected_data `
  --out docs\_real_alignment_20260707_audit
```

## Dataset Classification

| Dataset | Records | Events | Use |
|---|---:|---:|---|
| `real_align_20260707_left_elite_step_calib_001` | 203 | 12 | Use for Elite TCP/path-step scale. One busy-skip event should be excluded. |
| `real_align_20260707_right_elite_step_calib_001` | 199 | 16 | Use for Elite TCP/path-step scale. |
| `real_align_20260707_left_piper_burst_calib_001` | 211 | 8 | Use as Piper command-response evidence, but not as a precise visual displacement calibration yet. |
| `real_align_20260707_right_piper_burst_calib_001` | 192 | 8 | Use as Piper command-response evidence, but not as a precise visual displacement calibration yet. |
| `real_align_20260707_left_manual_coordination_001` | 460 | 36 | Use only as qualitative coordination evidence. Do not train on it as expert data. |
| `real_align_20260707_physical_limit_stuck_evidence_001` | 63 | 3 | Use as physical limitation/stuck evidence, not as normal feed calibration. |

Synchronization is acceptable for the alignment datasets. The
`image_pose_time_delta_ms_p95` values are roughly `17-33 ms`, and the side/top
camera deltas are roughly `4-31 ms` depending on dataset. This is good enough
for coarse control/camera alignment.

The older `real_pilot_20260707_left_linked_elite_piper_matched_001` still has
large pose lag and should not be used as the primary synchronized reference.
Use `real_pilot_20260707_left_linked_separate_elite_conn_001` if a synchronized
left coordination reference is needed.

## Elite Step Findings

The Elite-only calibration data is the cleanest quantitative anchor.

Event-window statistics from `events.csv`:

```text
left Elite next_submitted:
  n = 11 usable events
  TCP delta L2 median ~= 17.48 mm
  TCP delta Linf median ~= 16.86 mm
  side absdiff mean median ~= 4.60
  top absdiff mean median ~= 5.07

right Elite next_submitted:
  n = 16 usable events
  TCP delta L2 median ~= 14.45 mm
  TCP delta Linf median ~= 13.93 mm
  side absdiff mean median ~= 4.91
  top absdiff mean median ~= 5.24
```

Important caveat:

```text
These are event-window movement magnitudes, not per-frame policy targets.
```

The window uses the current audit defaults of two frames before the key event
and ten frames after it. The values are still useful because they measure the
real visual and TCP scale of one operator-triggered Elite path step in the
current setup.

Calibration implication:

- MuJoCo Elite path/TCP commands should be compared against a real event-window
  step scale of roughly `14-18 mm`, depending on branch and path segment.
- For policy/execution bandwidth, do not directly use this as a per-frame
  delta. Convert it through the real command duration and collection frame
  rate.
- Use side/top rendered image differences and contact sheets to check visual
  motion scale, not only numeric TCP deltas.

## Piper Burst Findings

The Piper-only calibration data confirms command logging and physical response,
but the current camera view does not provide a clean automatic whole-frame
displacement estimate. This is expected: Piper stays near the entrance and only
pushes the guidewire forward, so the only meaningful quantity is visible
guidewire advancement along the vessel/entry direction, not robot motion or
global image displacement.

Event-window statistics:

```text
left Piper burst 1:
  n = 3
  Elite TCP delta = 0
  side absdiff mean median ~= 3.40
  top absdiff mean median ~= 1.69

left Piper burst 2:
  n = 3
  Elite TCP delta = 0
  side absdiff mean median ~= 2.65
  top absdiff mean median ~= 1.63

left Piper burst 3:
  n = 2
  Elite TCP delta = 0
  side absdiff mean median ~= 3.57
  top absdiff mean median ~= 1.60

right Piper burst 1/2/3:
  n = 3/2/3
  Elite TCP delta = 0
  side absdiff mean median ~= 1.69-2.09
  top absdiff mean median ~= 1.81-1.93
```

Visual inspection of representative contact sheets shows that these global
image differences are dominated by weak scene/camera/lighting changes and do
not isolate the guidewire tip or inserted wire length. Therefore:

```text
Do not infer a precise real Piper feed distance from the current global image
diff statistics.
```

For Piper-only data, future measurement must focus on:

```text
guidewire forward progress = change in visible red tip / black wire endpoint /
inserted visible wire length along the vessel entry direction
```

It should not use full-image absdiff as the calibration target, because the
robot is intentionally stationary and the visible wire can be partially
occluded, stuck, or too subtle for global image metrics.

Use the manual annotation helper for this step:

```powershell
.\.venv\Scripts\python.exe tools\annotate_piper_forward_progress.py `
  --root collected_data `
  --events docs\_real_alignment_20260707_audit\events.csv `
  --out docs\_real_alignment_20260707_piper_forward_annotations\annotations.jsonl `
  --camera side `
  --max-window-width 1800 `
  --max-window-height 1000
```

Controls:

```text
left click in the left pane: before guidewire endpoint/red tip
left click in the right pane: after guidewire endpoint/red tip
in blink view, left click labels the currently displayed before/after frame
x: set forward axis by clicking axis start then forward direction
h: hide/show forward axis overlay
o: toggle split/blink view
b: switch blink view between before and after
1: save usable
2: save uncertain
3: save stuck
4: save not_visible
n: skip
q or Esc: quit
```

The blink view displays only one full-size frame at a time and `b` switches
between before and after, which helps judge subtle guidewire motion without
blend/difference artifacts. Clicking is allowed in blink view: the point is
assigned to before or after according to the currently displayed frame. The tool
uses an autosized OpenCV window and scales the image itself to preserve aspect
ratio; avoid relying on OS fullscreen stretching. Adjust
`--max-window-width` and `--max-window-height` if the window is too large or too
small. The tool also writes a CSV next to the JSONL. `forward_px` is only
meaningful after setting or auto-filling the forward axis for the current
camera; otherwise use `dx_px`, `dy_px`, and `euclidean_px` for manual review
only. If points were annotated before setting an axis, run:

```powershell
.\.venv\Scripts\python.exe tools\annotate_piper_forward_progress.py `
  --out docs\_real_alignment_20260707_piper_forward_annotations\annotations.jsonl `
  --autofill-forward-axis
```

Current manual annotation pass before explicit weak-motion filtering:

```text
annotations: 16 side-camera Piper-only events
quality: usable 14, stuck 2
primary metric: euclidean_px from before/after guidewire endpoint clicks
usable euclidean_px median: ~16.0 px
usable euclidean_px range: ~7.0-27.0 px
```

Per-dataset/burst medians from the first pass:

```text
left burst 1: n=2, median ~= 11.2 px
left burst 2: n=3, median ~= 17.0 px
left burst 3: n=2, median ~= 8.1 px
right burst 1: n=2, median ~= 17.2 px
right burst 2: n=2, median ~= 18.0 px
right burst 3: n=3, median ~= 25.2 px
```

Interpretation:

```text
The labels are good enough to show order-of-magnitude visible forward progress
of roughly 10-25 px for many successful feed windows, with two stuck windows.
They are not yet enough to infer a clean linear burst_count -> distance law
because the sample count is small, left burst-3 remains weak, and the metric is
observed endpoint displacement rather than a route-centered forward projection.
```

Filtered summary command:

```powershell
.\.venv\Scripts\python.exe tools\summarize_piper_forward_annotations.py `
  --out docs\_real_alignment_20260707_piper_forward_annotations\summary_filtered.json `
  --exclude real_align_20260707_left_piper_burst_calib_001:6 `
  --exclude real_align_20260707_left_piper_burst_calib_001:7
```

Filtering rule:

```text
Include quality=usable.
Exclude quality=stuck automatically.
Additionally exclude left burst-3 events 6 and 7 because their visible endpoint
movement is only ~7.0 and ~9.2 px, lower than left burst-1/2 and consistent
with the observed physical blocking/weak-push failure mode.
```

Filtered Piper scale reference:

```text
included usable events after filtering: 12
overall euclidean_px median: ~17.2 px
overall euclidean_px p25/p75: ~14.1 / 20.0 px

burst 1: n=4, median ~= 14.0 px
burst 2: n=5, median ~= 17.5 px
burst 3: n=3, median ~= 25.2 px
```

Interpretation:

```text
After removing explicit physical-blocking/weak-motion outliers, the remaining
annotation suggests a rough monotonic trend: larger Piper burst generally
produces larger visible guidewire endpoint displacement. Because n is tiny and
the left branch is physically less stable, use these numbers as a first
simulator-scale target band rather than a fitted law.
```

The physical-limit dataset has even smaller visible motion:

```text
physical-limit feed events:
  n = 3
  Elite TCP delta = 0
  side absdiff mean median ~= 1.86
  top absdiff mean median ~= 1.79
```

This supports the onsite observation that real guidewire progression can stall
or become hard to see under physical fixture/vessel constraints.

Calibration implication:

- Keep the Piper feed primitive in the real-aligned interface as
  `hold/feed` intent plus executed-command logs.
- Do not tune MuJoCo Piper feed scale solely from this Piper burst audit.
- Do not interpret one real `piper.step_forward(pause_time=0.8)` call as one
  MuJoCo `env.step`. A real Piper feed primitive is a short-duration actuator
  execution; in the current simulator, comparable visible side-camera red-tip
  motion requires several consecutive simulated feed steps.
- The next useful step is a local guidewire/tip tracker or manual annotation on
  the Piper-only windows, focused on red-tip/black-wire forward progress along
  the vessel/entry direction instead of whole-frame difference.
- Treat physical-limit windows as constraint evidence: the simulator should not
  assume every Piper feed command always produces forward guidewire progress.

## MuJoCo Piper Primitive Probe

Diagnostic tool:

```powershell
.\.venv\Scripts\python.exe tools\probe_sim_piper_forward_scale.py `
  --out simulation_output\sim_piper_forward_probe_realprimitive_v1 `
  --tasks left right `
  --piper-advection-scales "0.11" `
  --piper-cmds "0.6 1.0" `
  --primitive-steps "4 6 8" `
  --bursts "1 2 3" `
  --render-width 640 `
  --render-height 480 `
  --after-settle-steps 0 `
  --settle-steps 0
```

The tool detects the rendered red guidewire head by HSV color threshold and
writes contact sheets plus `measurements.csv` / `summary.json`. Visual checks
confirmed the yellow detection marker follows the red head on both branches.

Key probe result at 640x480 side view:

```text
cmd=0.6, primitive_steps=4:
  left  burst1/2/3 ~= 7.6 / 14.2 / 21.2 px
  right burst1/2/3 ~= 6.8 / 13.6 / 20.2 px

cmd=0.6, primitive_steps=8:
  left  burst1/2/3 ~= 14.2 / 28.2 / 35.1 px
  right burst1/2/3 ~= 13.6 / 26.2 / 32.5 px

real filtered annotation target:
  burst1/2/3 medians ~= 14.0 / 17.5 / 25.2 px
```

Interpretation:

```text
The order-of-magnitude match comes from mapping one real Piper feed primitive
to multiple simulated feed steps, not from simply increasing
piper_advection_scale. A primitive of roughly 6-8 simulated feed steps matches
real burst1, while 4 simulated feed steps better matches later bursts. This is
consistent with real elastic storage, slip, friction, and partial physical
blocking: repeated feed commands need not add visible red-tip displacement
linearly.
```

Implemented simulator semantics:

```text
The formal tip-line environment now supports a real-style Piper primitive
controller. The policy/expert emits a discrete feed/hold/retract intent, while
the controller executes one feed intent over `--piper-primitive-steps` simulator
steps using `--piper-primitive-feed-value`.

For formal mainline collection, `--piper-primitive-steps > 1` automatically
switches the route-plan Piper schedule to event semantics: `piper_step_command=1`
marks a new real-style `step_forward` event, while subsequent frames can remain
policy hold but still show `controller_state.piper_executed_feed > 0` during
the busy primitive.
```

Initial smoke:

```text
simulation_output/_smoke_formal_piper_primitive_event_period10_v1
task: left
result: success=True, steps=165
settings:
  --piper-command-period 10
  --piper-command-width 1
  --piper-primitive-steps 8
  --piper-primitive-feed-value 0.6
interpretation: event-style Piper primitive closes the short-loop smoke, but
contact/visual quality still needs normal video review before scaling.
```

Implementation caveat:

```text
Changing piper_advection_scale alone changes the logged Piper insertion length
but does not reliably change red-tip progression in the current environment,
because the tip/route progression is still dominated by normalized Piper
command and guidewire stabilization/anchor logic. Treat piper_advection_scale
as an insertion-joint scale, not as a direct visible-tip calibration knob.
```

## Manual Coordination Findings

`real_align_20260707_left_manual_coordination_001` contains both Elite path
steps and Piper feed bursts. It is useful because it records how the real system
looks under attempted coordination, but it should not be treated as expert data.

Reasons:

- The operator could not precisely know the mechanical step length of both
  robots while pressing keys.
- Piper and Elite primitive scales are not matched.
- Some Piper feed commands are visually weak or physically limited.
- Coordination data mixes intentional control, timing mismatch, and fixture
  resistance.

Use it for:

- qualitative sim-real visual comparison;
- checking whether MuJoCo reproduces lag, wall proximity, or hard-to-progress
  regions;
- sanity-checking whether a proposed controller produces real-plausible motion.

Do not use it for:

- supervised policy targets;
- claiming a real expert demonstration;
- validating BC/VLA task success.

## Immediate Simulator Actions

1. Add a comparison command that renders MuJoCo event-window contact sheets for
   the same branch/task/camera layout as the real alignment sheets.
2. Tune Elite path/TCP event-window scale first, because it is the cleanest
   quantitative real anchor.
3. Keep Piper feed scale conservative until a tip/local-wire tracker or manual
   annotation provides a better displacement estimate.
4. Add stochastic or state-dependent Piper effectiveness in simulation only
   after it is framed as real physical insertion resistance, not as hidden
   oracle correction.
5. Use the physical-limit capture as negative evidence: real Piper feed can
   execute while visible guidewire progress remains small.

## Current Decision

The next mainline should be:

```text
real alignment audit -> Elite scale matching -> Piper local tracking/manual
annotation -> MuJoCo parameter update -> small sim validation set
```

Do not start a new BC training round until the real-to-sim calibration changes
are implemented and visually checked.
