# Commands

Last updated: 2026-07-17

This file keeps copyable commands for the data/simulation/real-collection
track and retains older non-VLA commands for provenance. Current algorithm
innovation commands live in `docs/algorithm-track-commands.md`; the pre-split
Pi-Style/OpenPI/PI05 command block is archived under `docs/archive/`.

## Conventions

Run commands from the repository root:

```powershell
cd D:\PycharmProjects\project_2026
```

Use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Current mainline defaults:

```text
collector = simulation.collect_formal_tip_line_guidance
guidewire = tip-centric / hard-elastic
wire visual = continuous line
action = Piper signed feed/hold intent + Elite TCP delta / IK execution
```

The older collection entrypoints are now legacy/diagnostic only:

```text
simulation.collect_tip_guided_wire
simulation.collect_mujoco_physical_guidance
```

Do not use legacy collection scripts for new mainline data unless the task is
explicitly a comparison or regression against an older route.

## Visual Route Debugging

Adjust the visual guidewire route:

```powershell
.\.venv\Scripts\python.exe tools\adjust_wire_visual_outlet.py `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --task left `
  --camera side `
  --start-fraction 0.58
```

Use this when the rendered guidewire exits from the wrong side or the tail path
does not follow the intended vessel-entry route. The tool edits
`wire_visual_piper_exit_point`, `wire_visual_entry_point`,
`wire_visual_entry_progress`, and optional `wire_visual_route_points` in the
route config; it does not change expert control, policy labels, or rollout
semantics.

When `wire_visual_route_points` are present, the tip-centric renderer treats
them as a fixed visual entry prefix. The final route point is used as the
attach marker for where the renderer should resume the dynamic route-to-tip
tail; earlier route points are smoothed with centripetal Catmull-Rom
interpolation. Optional `wire_visual_dynamic_route_points` then shape only the
post-attach visual segment directly before it reaches the current tip. These
controls are visual-only.

Controls:

```text
e: cycle active point (outlet -> entry -> route points)
n: add a manual route point between entry and tip
g: add a dynamic route point after the prefix attach marker
x: delete active route point
j/l: move x, i/k: move y, u/o: move z
[/]: change movement step
1/2/3: side/top/overview camera
h: show/hide robots
s: save, q: quit
```

## Current Mainline Collection

Small visual/logic pilot:

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m simulation.collect_formal_tip_line_guidance `
  --out simulation_output\formal_tip_line_visual_route_v1_small_check `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 3 `
  --start-fraction-min 0.42 `
  --start-fraction-max 0.54 `
  --max-steps 700 `
  --sample-every 10 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-segments 240 `
  --formal-data `
  --progress-log-every 100
```

Current accepted visual-route baseline:

```text
simulation_output\formal_tip_line_visual_route_v1_small_fix1
accepted_episodes: 20
attempts: 20
samples: 567
left/right success: 10/10 and 10/10
max_contact_strength: 0.0
contact_p95 max: 0.0
min_tip_distance_to_wall: ~1.386 mm
visual review: accepted by user
```

Scale from the accepted visual route:

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m simulation.collect_formal_tip_line_guidance `
  --out simulation_output\formal_tip_line_visual_route_v1_dataset `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --start-fraction-min 0.42 `
  --start-fraction-max 0.54 `
  --max-steps 700 `
  --sample-every 10 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-segments 240 `
  --formal-data `
  --progress-log-every 100
```

This mainline collector hides tool markers and path tubes by construction. It
uses the current fixed formal settings:

```text
route_plan_step=0.15
start_fraction_min=0.42
start_fraction_max=0.54
elite_target_semantics=front_up_magnetic_target
elite_ahead=0.010
elite_height_offset≈0.00924 with the current vessel_scale=0.077
piper_cmd=0.70
piper_command_period=40
piper_command_width=20
route_plan_command_phase_lock=true
wire_visual_mode=line
wire_visual_radius=0.0008
wire_visual_rgb="0.02 0.02 0.018"
wire_tip_visual_rgb="0.78 0.04 0.02"
wire_tip_visual_segments=8
wire_tip_visual_radius_scale=2.2
wire_tip_marker_radius=0.0020
wire_tip_marker_alpha=0.95
wire_visual_offset=(0.0, 0.0, 0.0)
wire_segments=240
tip_tail_decay_segments=18
```

Formal visual-distance estimator collection with the accepted side-camera
coverage config and fixed rendered-image estimator:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_formal_tip_line_guidance `
  --camera-config simulation\camera_configs\mujoco_camera_side_coverage_v1.json `
  --out simulation_output\formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --visual-distance-estimator `
  --progress-log-every 100
```

Do not use `formal_tip_line_black_red_head_sidecam_v1_dataset_v2` estimator
fields for training. That dataset collected successfully, but visual review
found that the older projection estimator could draw wall markers at visibly
wrong top-camera locations. The fixed estimator measures the rendered image and
emits missing/low-confidence estimates when the red guidewire head is occluded
or too close to the image boundary.

Current result:

```text
simulation_output\formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4495
missing image refs: 0
provenance audit: real_direct_plus_visual_contact passes
visual check: docs\_formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1_visual_check
visual review: accepted by user
side estimator visible: 3915/4495
top estimator visible: 0/4495
estimated_contact_flag: 0/4495
estimated_image_distance_px p50/p95: ~6.54 / 7.79 px
piper_step_command feed/hold: 2355/2140
```

Treat this as the current accepted formal visual-distance estimator candidate.
Use continuous `estimated_image_distance_px` plus
`contact_estimator_confidence`; the all-zero binary contact flag is not yet a
strong training signal.

Paper-aligned estimated 3D tip collection adds a synthetic calibrated RGB-D
estimator on top of the accepted rendered-image red-tip detection. This emits
`estimated_tip_pos_3d`, `estimated_tip_heading_3d`,
`tip_estimator_visible`, `tip_estimator_confidence`, and
`tip_estimator_latency_steps` with observation provenance. The implementation
uses red-tip image visibility plus a declared synthetic calibration/noise
wrapper; it does not expose clean MuJoCo `tip_pos` as a policy input.

Use the manually adjusted top-oblique camera config when collecting the next
estimated-tip dataset. It keeps the accepted side camera unchanged, while the
top camera looks diagonally downward from a less occluded angle so the red
guidewire head remains visible.

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_formal_tip_line_guidance `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --out simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --visual-distance-estimator `
  --estimated-tip-3d `
  --progress-log-every 100
```

Audit a collected estimated-tip dataset before training:

```powershell
.\.venv\Scripts\python.exe tools\audit_observation_provenance.py `
  simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1 `
  --observation-schema real_direct_plus_estimated_tip_contact `
  --sample-limit 200 `
  --out docs\_formal_tip_line_estimated_tip_3d_top_manual_dataset_v1_audit
```

Use this before any new model training that consumes visual-distance/contact
estimator fields. After collection, inspect:

```text
manifest success and rejected attempts
missing image refs
side/top estimator visibility counts
estimated_image_distance_px and contact_estimator_confidence
estimated_contact_flag distribution
contact/wall/tip_to_magnetic metrics
sample side/top frames
provenance audit with real_direct_plus_visual_contact
```

## Git And GitHub Workflow

This repository is for code, durable docs, configs, and lightweight reference
assets. Do not use GitHub as the storage place for `simulation_output/`,
`branchs/`, local virtual environments, or model/video artifacts.

Use `main` directly when:

- the change is small and low-risk, such as docs, commands, comments, or a
  localized bug fix;
- the change only clarifies current project state or handoff information;
- you have already checked that it does not change action semantics or the
  agreed simulation route.

Use a branch first when:

- changing environment dynamics, action semantics, expert logic, rollout
  execution limits, or dataset acceptance rules;
- running parallel ideas that may be discarded, such as alternative control
  formulations or refactors;
- the change will need side-by-side comparison before merge;
- the user may want to keep the current `main` as the stable reference.

Recommended branch names:

```text
feat/<topic>
fix/<topic>
exp/<topic>
docs/<topic>
```

GitHub is mainly a stable rollback/sync point, not a destination for every small
local tweak. Local commits can be finer grained; pushes should mark meaningful
milestones.

Good push points:

- after a confirmed new mainline entrypoint;
- after changing formal data validity rules;
- after a durable project-direction decision or accepted experiment conclusion;
- after a validated dataset/model/control milestone;
- before a long training or collection run only when the run depends on a
  confirmed code state worth preserving as a rollback point;
- before broad cleanup, renaming, or other repo-wide edits;
- at the end of a work session only if the current state is stable enough for a
  future conversation or another machine to resume.

Do not push immediately for small parameter tweaks, smoke-only experiments,
wording-only edits, or intermediate trial-and-error changes. Batch them into the
next meaningful milestone commit/push once the result is accepted.

Typical small-update flow on `main`:

```powershell
git status
git add <changed files>
git commit -m "docs: update handoff after accel-limit review"
git push
```

Typical experiment flow on a branch:

```powershell
git switch -c exp/magnet-attached-envsuccess-v2
git status
git add <changed files>
git commit -m "exp: adjust corrected-semantics collection acceptance"
git push -u origin exp/magnet-attached-envsuccess-v2
```

Before pushing, check that you are not accidentally including large or local
files:

```powershell
git status --short
git diff --stat
```

If the work changed durable project direction, update and commit these together
with the code when appropriate:

```text
docs/project-state.md
docs/handoff.md
docs/experiment-registry.md
docs/decision-log.md
```

## Smoke-Test Data Collection

Use this before a long collection run:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_mujoco_physical_guidance `
  --out simulation_output\mujoco_physical_feed_action_dataset_v2_smoke `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 3 `
  --seed 8601 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --progress-deltas 14 20 28 40 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 80
```

## Diagnostic Oracle Data Collection

Current MuJoCo expert collection is diagnostic/oracle data, not validated formal
sim-to-real data. The expert uses simulation-only state. Do not scale it as
formal data unless the expert/control stack is revised to satisfy
`docs/simulation-expert-validity-audit.md`.

Current diagnostic dataset command:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_mujoco_physical_guidance `
  --out simulation_output\mujoco_physical_feed_action_dataset_v2 `
  --tasks left right `
  --episodes-per-task 40 `
  --max-attempts-per-episode 4 `
  --seed 8601 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --progress-deltas 14 20 28 40 `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 80
```

After collection, inspect:

```text
simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json
```

Check accepted/rejected episodes, sample count, `min_segment_distance_to_wall`,
contact, and progress gain.

Formal validity guard:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_mujoco_physical_guidance `
  --formal-data
```

This intentionally fails with the current oracle expert. A successful
`--formal-data` run should only become possible after the expert/control stack
uses real-implementable inputs.

## Tip-Centric Guidewire Data Collection

Use this for the parallel tip-centric abstraction. It keeps the external
`piper_feed_elite_joint` action schema and corrected Elite-attached magnetic
semantics, but collects from `simulation.tip_guided_wire_env` instead of the
full polyline physical guidewire.

Current tip-centric collection is also diagnostic/oracle data because the expert
uses exact simulated tip, vessel frame, lookahead, wall/contact, and IK.

Short script smoke:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\tip_guided_wire_dataset_smoke_v1 `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 2 `
  --seed 9801 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 120
```

If the smoke is clean, collect a moderate dataset:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\tip_guided_wire_dataset_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9801 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 700 `
  --sample-every 2 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --progress-log-every 120
```

After collection, inspect:

```text
simulation_output\tip_guided_wire_dataset_v1\manifest.json
```

Check accepted/rejected episodes, sample count, `env_success`,
`stall_fraction`, `min_tip_distance_to_wall`, `max_contact_strength`, and
`tip_to_magnetic_median/p95`.

Formal validity guard:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --formal-data
```

This intentionally fails with the current tip-centric oracle expert.

Formal route-plan probe:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_smoke_v1 `
  --tasks left right `
  --episodes-per-task 2 `
  --max-attempts-per-episode 2 `
  --seed 9901 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.62 `
  --success-mode env_success `
  --max-steps 500 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --progress-log-every 100 `
  --keep-failures
```

This is not expected to match the oracle expert immediately. Judge it by
progress gain, wall/contact behavior, `tip_to_magnetic`, and whether any
episodes reach `env_success`.

Current result:

```text
simulation_output\formal_route_plan_tip_smoke_v1
formal_valid: yes
accepted episodes: 4/4
left/right env_success: 2/2 and 2/2
samples: 150
steps median: 183.5
stall_fraction: 0.0
min_tip_distance_to_wall min: 0.000546 m
tip_to_magnetic median: ~50-59 mm
contact_p95: ~0.568-0.601
```

This proves route-plan is a formal-valid baseline candidate, but it is not
clean enough to scale. Next, test smaller planned Elite lead and slower route
advance before training:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$variants = @(
  @{name='step024_ahead010'; step='0.24'; ahead='0.010'},
  @{name='step020_ahead010'; step='0.20'; ahead='0.010'},
  @{name='step024_ahead006'; step='0.24'; ahead='0.006'},
  @{name='step020_ahead006'; step='0.20'; ahead='0.006'}
)
foreach ($v in $variants) {
  .\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
    --out "simulation_output\formal_route_plan_tip_sweep_$($v.name)" `
    --tasks left right `
    --episodes-per-task 2 `
    --max-attempts-per-episode 2 `
    --seed 9911 `
    --start-fraction-min 0.58 `
    --start-fraction-max 0.62 `
    --success-mode env_success `
    --max-steps 500 `
    --sample-every 5 `
    --image-size 224 `
    --render-width 960 `
    --render-height 720 `
    --robot-visual-mode kinematic `
    --action-mode piper_feed_elite_joint `
    --hide-tool-markers `
    --hide-path-tubes `
    --expert-mode route_plan `
    --formal-data `
    --route-plan-step $v.step `
    --elite-ahead $v.ahead `
    --progress-log-every 100 `
    --keep-failures
}
```

Judge the sweep by keeping `env_success` while reducing `contact_p95` and
`tip_to_magnetic_median/p95`.

Current sweep result:

```text
all variants: formal_valid and env_success 4/4

step020_ahead010:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.001548 m
  tip_to_magnetic median: ~3.6-4.7 mm
  tip_to_magnetic p95: ~6.1-10.6 mm
  steps median: 197.5

step020_ahead006:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.002541 m
  tip_to_magnetic median: ~4.3-5.4 mm
  steps median: 200.5

step024_ahead006:
  contact_p95: 0.0
  max_contact_strength: 0.0
  min_tip_distance_to_wall min: 0.001494 m
  tip_to_magnetic median: ~5.5-5.9 mm
  steps median: 185.5

step024_ahead010:
  contact_p95: 0.0
  max_contact_strength: 0.008
  min_tip_distance_to_wall min: 0.001375 m
  tip_to_magnetic median: ~9.4-9.8 mm
  steps median: 183.5
```

Recommended moderate formal route-plan collection:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9921 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 600 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --route-plan-step 0.20 `
  --elite-ahead 0.010 `
  --progress-log-every 100 `
  --keep-failures
```

Moderate collection for the current real-action-aligned signed-Piper route-plan
expert. This uses the v8 settings: lower-frequency feed/hold labels, explicit
feed magnitude, and route-plan progress locked to feed phases with a reduced
nominal route step.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_signedpiper_dataset_v8 `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9961 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 600 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --route-plan-step 0.24 `
  --elite-ahead 0.010 `
  --piper-cmd 0.70 `
  --piper-command-period 40 `
  --piper-command-width 20 `
  --route-plan-command-phase-lock `
  --progress-log-every 100 `
  --keep-failures
```

Elite front-up guidance audit:

```powershell
.\.venv\Scripts\python.exe -B tools\audit_elite_frontup_guidance.py `
  simulation_output\formal_tip_line_frontup_magnet_esttip_reggeom_dataset_v1 `
  --out simulation_output\formal_tip_line_frontup_magnet_esttip_reggeom_dataset_v1\elite_frontup_audit.json
```

Interpretation:

- `actual_forward_m` is `(magnetic_pose - tip_pos) dot path_tangent`; it should
  usually be positive if the Elite-attached magnet is actually in front of the
  guidewire head.
- `desired_forward_m` is available for datasets collected after the diagnostic
  fields were added and checks the route-plan expert's original desired target.
- `formal_tip_line_frontup_magnet_esttip_reggeom_dataset_v1` should not be used
  as the accepted front-up source: its audit found episode median actual
  forward projection around `-2.9 cm`, with about 94-95% negative samples. The
  cause was route-plan progress lag from `--route-plan-step 0.15`.
- A short `--route-plan-step 0.24` smoke under
  `simulation_output\_smoke_frontup_routeplan_step024` restored positive
  desired and actual forward projections for both branches.

Previous result to avoid repeating:

```text
formal_route_plan_tip_signedpiper_smoke_v3 used period=125,width=5. It passed
the formal-data guard and produced sparse labels, but failed all 4 attempts
because the feed duty cycle was too low: accepted_episodes=0/4,
piper_step_command counts feed/hold=20/460, transition_fraction ~=0.0795,
tip_to_magnetic median about 149-168 mm, and contact max about 0.51-0.55.

formal_route_plan_tip_signedpiper_smoke_v4 used period=250,width=125. It passed
formal/env success 2/2, but both short episodes finished before the first hold
phase, so every saved command was feed. It also used full feed magnitude and
degraded expert quality: contact p95 about 0.54-0.59 and tip_to_magnetic median
about 40-42 mm, far worse than the clean v2 millimeter-scale, zero-contact
expert.

formal_route_plan_tip_signedpiper_smoke_v5 used piper_cmd=0.70 and
period=40,width=20. It passed formal/env success 2/2 and produced real feed/hold
labels (feed/hold=53/48, transition_fraction ~=0.24), but quality was still not
clean enough: left contact_p95 ~=0.594, right contact_p95 ~=0.00088, and
tip_to_magnetic median stayed around 20-21 mm. This supports a route-plan/Piper
phase mismatch hypothesis rather than a pure label-frequency issue.

formal_route_plan_tip_signedpiper_smoke_v6 added route-plan-command-phase-lock
with route_plan_step=0.20. It kept formal/env success 2/2 and the same feed/hold
labels, but quality worsened: contact_p95 max ~=0.605 and tip_to_magnetic median
rose to about 29-32 mm. The likely issue is that phase-lock compensated by duty
cycle, so the feed-phase route step became 0.40.

formal_route_plan_tip_signedpiper_smoke_v8 used route_plan_step=0.15 with the
same phase-lock and Piper cadence. It restored clean expert quality:
formal/env success 2/2, zero contact, complete images, feed/hold=55/52, and
tip_to_magnetic median about 2.2-2.8 mm with p95 about 7.5-7.8 mm. A small
stress run, formal_route_plan_tip_signedpiper_smoke_v8_stress4, accepted 8/8
episodes with zero contact, complete images, feed/hold=192/176, and sample
tip_to_magnetic p95 about 7.7 mm. This is the current signed-Piper formal
expert setting to scale.
```

Current moderate signed-Piper dataset result:

```text
simulation_output\formal_route_plan_tip_signedpiper_dataset_v8
formal_valid: yes
accepted/env_success: 40/40
rejected attempts: 0
samples: 1807
image refs: 3614/3614
contact_p95 max: 0.0
sample contact p95: 0.0
tip_to_magnetic episode-p95 median: ~7.6-7.7 mm
sample tip_to_magnetic p95: ~7.7 mm
piper_step_command feed/hold: 951/856
piper_feed values: 0.7 / 0.0
transition_fraction median: ~0.244
```

Legacy continuous-Piper training command. This was useful as a first diagnostic,
but the resulting rollout failed left/right because `piper_step_command`
feed/hold was regressed into mid-valued continuous `piper_feed` outputs rather
than preserved as a discrete real command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600
```

Piper-classification training command. It keeps the external
`piper_feed_elite_joint` rollout action but trains Piper as a three-class
`piper_step_command` head (`-1=retract, 0=hold, 1=feed`) and maps feed back to
`piper_feed=0.7` during rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Result: the model trained and saved correctly, but open-loop diagnostics showed
this is not yet a usable signed-Piper baseline. Piper feed/hold accuracy was
only about `61.8%` (`hold->feed 289/856`, `feed->hold 402/951`), and Elite
absolute target quality was poor (`elite_linf` median/p95 about `0.122/0.206`,
predicted target-step p95 about `0.112` versus expert `0.022`). Do not keep
rerunning this exact training as the next default.

Real-direct observation-schema diagnostic. This removes exact simulator-only
state such as `tip_pos`, `path_progress`, wall distance, local route frame, and
exact contact from the structured policy input. Use this as an observation
alignment check, not as proof that the small BC should become the main path:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8_real_direct_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema real_direct `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Senior-Piper-like observation-schema diagnostic. This is closer to the inherited
real Piper input contract: image plus Elite TCP 6D pose, Piper step, and task id.
The simulator now writes `elite_tcp_pose_6d`; old datasets fall back to
`elirobot_pose * 1000 + zero orientation`, so prefer newly collected data for a
true 6D comparison when available:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v8\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v8_seniorlike_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema senior_piper_real_like `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Expected data-schema check:

```text
Each piper_feed_elite_joint action should keep piper_feed and additionally
write piper_step_command (-1=retract, 0=hold, 1=feed) plus piper_command_label.
This preserves current continuous execution while making the candidate real
Piper command label explicit.
```

Current formal route-plan dataset result:

```text
simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1
formal_valid: yes
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

Train a small route-plan baseline:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600
```

Current training result:

```text
model: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1
dataset: simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1
samples: 1360
action mode: piper_feed_elite_joint
elite action representation: absolute
elite_smoothness_weight: 0.0
best val_loss: 0.001086 at epoch 8
epoch 8 val_piper_loss: 0.000423
epoch 8 val_elite_loss: 0.000664
```

After training, run a no-video formal rollout in the same tip environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100
```

Do not add diagnostic rollout helpers such as `--piper-feed-phase-guard`,
`--elite-tip-anchor`, scripted Piper, or tactile safety to a formal rollout.

Current formal rollout result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_rollout_formal_r1
formal_valid: yes
left/right success: True / True
left/right steps: 310 / 281
max_contact_strength: ~0.605 / 0.574
contact median: ~0.365 / 0.403
stall fraction: ~0.332 / 0.374
slow fraction: ~0.452 / 0.480
tip_to_magnetic median: ~25 / 19 mm
tip_to_magnetic p95: ~92 / 127 mm
elite target step p95: ~0.093 / 0.063
```

This is a negative loop-closure result despite numerical success. Do not render
video for this checkpoint. Run an open-loop diagnostic first:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out docs\_baseline_formal_route_plan_tip_step020_ahead010_v1_open_loop_diag `
  --batch-size 64
```

Current open-loop result:

```text
piper_abs median/p95: ~0.026 / 0.049
piper sign mismatch: 0.0
elite_linf median/p95: ~0.051 / 0.088
expert Elite target-step p95: ~0.0156
predicted Elite target-step p95: ~0.113
expert Elite delta-from-current p95: ~0.0062
predicted Elite delta-from-current p95: ~0.0872
```

Next model-side check: train the same formal route-plan dataset with Elite
delta targets, then diagnose open-loop before rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --elite-action-representation delta
```

After training, diagnose before rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --out docs\_baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_open_loop_diag `
  --batch-size 64
```

Current Elite-delta open-loop result:

```text
model: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta
best val_loss: 0.000147 at epoch 8
piper_abs median/p95: ~0.011 / 0.027
piper sign mismatch: 0.0
elite_linf median/p95: ~0.0136 / 0.0212
expert Elite target-step p95: ~0.0156
predicted Elite target-step p95: ~0.0312
expert Elite delta-from-current p95: ~0.0062
predicted Elite delta-from-current p95: ~0.0219
```

The no-video formal rollout has been run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100
```

Current result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_rollout_formal_r1
formal_valid: yes
left/right success: False / False
left/right steps: 600 / 600
distance_to_target: ~0.0617 / 0.0773 m
max_contact_strength: ~0.606 / 0.552
contact median: ~0.590 / 0.525
slow-progress fraction: ~0.683 / 0.700
tip_to_magnetic median: ~309 / 434 mm
tip_to_magnetic p95: ~503 / 657 mm
```

This is a negative closed-loop result. Do not render video and do not continue
naive delta as the main model formulation.

Route-plan-anchor check:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0020_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100 `
  --elite-route-plan-anchor `
  --elite-route-plan-anchor-step 0.20 `
  --elite-route-plan-anchor-ahead 0.010 `
  --elite-route-plan-anchor-joint-limit 0.020
```

This is intended as a formal-safe bounded-target check because the anchor uses a
registered route plan and scheduled progress, not current tip/contact/wall
oracle feedback. Keep `--formal-eval`; if the validity guard rejects it, treat
that as a bug in the audit logic rather than bypassing the guard.

Current `0.020` result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0020_rollout_formal_r1
formal_valid: yes
metadata elite_route_plan_anchor_joint_limit: 0.020
left/right success: True / True
left/right steps: 277 / 276
max_contact_strength: ~0.606 / 0.574
tip_to_magnetic median: ~34 / 32 mm
tip_to_magnetic p95: ~50 / 54 mm
magnetic/Elite pose step p95: ~4.1 / 3.7 mm
progress zero fraction: ~33% / 34%
progress slow fraction: ~42% / 44%
```

This is a real improvement over the unanchored Elite-delta rollout, but it is
not physically clean enough to render video or treat as solved. This directory
has been overwritten once, so future comparisons should use a distinct output
directory and trust `meta.json` over the directory name. The next no-video check
should tighten the route-plan anchor joint window:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta\best_model.pt `
  --out simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0010_rollout_formal_r1 `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 600 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100 `
  --elite-route-plan-anchor `
  --elite-route-plan-anchor-step 0.20 `
  --elite-route-plan-anchor-ahead 0.010 `
  --elite-route-plan-anchor-joint-limit 0.010
```

Current `0.010` result:

```text
rollout: simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0010_rollout_formal_r1
formal_valid: yes
metadata elite_route_plan_anchor_joint_limit: 0.010
left/right success: True / True
left/right steps: 292 / 284
max_contact_strength: ~0.606 / 0.573
tip_to_magnetic median: ~34 / 29 mm
tip_to_magnetic p95: ~57 / 57 mm
contact p95: ~0.602 / 0.565
elite_target_jump_p95: ~0.0168 / 0.0154
executed_joint_jump_p95: ~0.0077 / 0.0069
magnetic/Elite pose step p95: ~3.5 / 2.6 mm
```

Interpretation:

```text
The 0.010 anchor improves command smoothness but not physical quality.
Do not render video or keep shrinking the anchor joint window as the default
next step. Diagnose schedule/Piper/tip progress mismatch first.
```

Diagnose route-plan anchor phase against expert data:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_route_plan_anchor_phase.py `
  --dataset simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1 `
  --rollout simulation_output\baseline_formal_route_plan_tip_step020_ahead010_v1_elitedelta_routeanchor0010_rollout_formal_r1 `
  --out docs\_route_plan_anchor_phase_diag
```

Current diagnostic result:

```text
expert plan_minus_actual median/p95:
  left  -2.567 / -0.428
  right -2.467 / -0.428
rollout plan_minus_actual median/p95:
  left   4.808 / 11.640
  right  3.354 / 10.987
rollout corr(plan_lead, contact):
  left  0.820
  right 0.866
```

Interpretation: the scheduled route-plan anchor is out of phase with actual
tip/Piper progress during BC rollout. In the expert data, planned progress is
slightly behind actual tip progress at sampled states; in rollout, planned
progress runs far ahead and contact rises with that lead.

Current moderate dataset result:

```text
simulation_output\tip_guided_wire_dataset_v1
accepted episodes: 40/40
left/right: 20/20
rejected: 0
samples: 3231
image refs: 6462/6462 exist
env_success: 40/40
wall penetration: none in episode/sample checks
stall_fraction: 0.0 on all episodes
```

Train a small baseline on this dataset:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\tip_guided_wire_dataset_v1\manifest.json `
  --out simulation_output\baseline_tip_guided_wire_v1 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700
```

Roll it out in the same tip-centric environment, without video first:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_tip_guided_wire_v1\best_model.pt `
  --out simulation_output\baseline_tip_guided_wire_v1_rollout `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard
```

Do not omit `--env-type tip` for this rollout. The default rollout environment
is still the full polyline guidewire for backward compatibility.

Formal rollout validity guard:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --formal-eval `
  --checkpoint simulation_output\baseline_tip_guided_wire_v1\best_model.pt `
  --out simulation_output\formal_eval_probe `
  --env-type tip `
  --tasks left right `
  --guidance-mode physical
```

`--formal-eval` rejects diagnostic/oracle rollout interventions such as
`--elite-tip-anchor`, `--tactile-safety`, `--scripted-piper-feed`, and
`--piper-feed-phase-guard`.

## Simulation Validity Audit

Audit existing datasets and rollouts before treating them as mainline evidence:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\audit_simulation_validity.py `
  simulation_output\tip_guided_wire_dataset_v1 `
  simulation_output\mujoco_physical_feed_action_dataset_magnet_attached_envsuccess_v1 `
  simulation_output\baseline_tip_guided_wire_v1_elitedelta_anchor0020_rollout_r1 `
  --out docs\_simulation_validity_audit_current
```

Interpretation:

```text
formal_valid: no known oracle/diagnostic mechanism detected
diagnostic_oracle: useful for diagnostics, not formal data-generation evidence
unknown_legacy: insufficient metadata; do not treat as formal without review
```

Compare formal route-plan expert datasets against real `branchs` motion/Piper
anchors without involving BC rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\compare_formal_expert_real_alignment.py `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_v1\manifest.json `
  --manifest simulation_output\formal_route_plan_tip_dataset_step020_ahead010_stress_wide_v1\manifest.json `
  --out docs\_formal_expert_real_alignment
```

Interpretation:

```text
Use real branchs pose only as Elite/magnetic-arm motion anchor, not guidewire truth.
The current report finds the largest confirmed gap in Piper semantics:
real binary 0=stop/hold, 1=step_forward labels versus continuous simulated
positive/negative piper_feed. This does not mean retract should be removed; it
means retract needs an explicit real Piper command and label.
```

Diagnose the trained tip-centric model on expert states before another rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_tip_guided_wire_v1\best_model.pt `
  --manifest simulation_output\tip_guided_wire_dataset_v1\manifest.json `
  --out simulation_output\baseline_tip_guided_wire_v1_open_loop_diag `
  --batch-size 64
```

Current diagnostic result:

```text
piper_abs_error median/p95: ~0.019 / 0.059
piper_sign_mismatch_fraction: 0.0
elite_linf_error median/p95: ~0.051 / 0.071
expert Elite target step p95: ~0.020
predicted Elite target step p95: ~0.049
```

Do not render another video if this diagnostic still shows large Elite target
error and target-step jitter on expert states.

## Baseline Training

Train first without high Piper negative weighting:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700
```

If Piper rollback is insufficient, try a conservative weighted variant:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_piperw2 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --piper-feed-negative-loss-weight 2.0
```

Avoid defaulting to:

```text
--piper-feed-negative-loss-weight 4.0
```

because it previously amplified Elite joint-output instability.

## Baseline Training With Elite Smoothness

Use this when closed-loop rollout shows systematic Elite joint target jitter.
This keeps the external action schema as `piper_feed_elite_joint` and adds an
adjacent-sample Elite joint delta loss during training:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_smooth4 `
  --epochs 8 `
  --batch-size 32 `
  --temporal-batch-size 16 `
  --image-size 224 `
  --max-steps 700 `
  --elite-smoothness-weight 4.0
```

After training, compare rollout target jitter against:

```text
baseline_mujoco_physical_feed_action_v2_rollout:
left/right target_jump_p95 ~= 0.0549 / 0.0529

expert labels in mujoco_physical_feed_action_dataset_v2:
elite joint step p95 ~= 0.00561
```

## Baseline Training With Elite Delta Actions

Use this after rollout-time rate limiting reduces flashes but video still shows
Elite jitter. This keeps the external rollout action schema as
`piper_feed_elite_joint`, but trains the model to predict Elite joint deltas from
the current Elite joint state:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --elite-action-representation delta
```

Roll it out first without video:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard
```

## Rollout With Video

Use this after training:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

For a different checkpoint, change only:

```text
--checkpoint
--out
```

unless the experiment intentionally changes environment parameters.

## Rollout With Elite Acceleration Limit

Use this to test a real-bandwidth execution layer without retraining. It keeps
the external action schema as `piper_feed_elite_joint`, treats the model Elite
output as the desired six-joint target, and limits how fast the executed Elite
joint delta can change between environment steps:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1 `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.002
```

If it preserves success but jerk remains high, try a tighter value:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel001_r1 `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.001
```

If the logged tool pose is smooth but the video still shows mechanical-arm
twitching, keep the `0.010` velocity limit and `0.002` acceleration limit, then
add a jerk limit:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_jerk001_r1 `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.002 `
  --elite-joint-jerk-limit 0.001
```

Start without video. If it preserves success and lowers executed joint jerk,
render a video with the same limits.

Then compare against real bandwidth:

```powershell
.\.venv\Scripts\python.exe -B tools\compare_rollout_control_bandwidth.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1 `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel001_r1 `
  --real-summary simulation_output\real_control_bandwidth\real_control_bandwidth_summary.json `
  --out simulation_output\rollout_control_bandwidth_accel_limit
```

Judge this first without video. Only render video after success, drift, step,
acceleration, jerk, and direction-reversal metrics look promising.

## Elite Jump Diagnostics

Run after rollout:

```powershell
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  --no-plots
```

Key outputs:

```text
elite_diagnostics_summary_all.json
left\elite_diagnostics_summary.json
right\elite_diagnostics_summary.json
```

Key metrics:

```text
elite_target_joint_step_linf
elite_executed_joint_step_linf
elite_target_to_executed_joint_l2
elite_tool_step
magnetic_step
tip_to_elite_tool
tip_to_magnetic
```

## Real Branch Data Inspection

Use this to summarize the real data under `branchs/`:

```powershell
.\.venv\Scripts\python.exe tools\inspect_branch_real_data.py `
  --root branchs `
  --out simulation_output\real_branch_data_inspection
```

Outputs:

```text
real_branch_summary.json
real_branch_paths.json
real_branch_paths.csv
```

Use this as a reality check for MuJoCo tuning. Current real-data summary:

```text
real paths: 24
pose/image frames: 4039 / 4039
position_step_p95: roughly 5 mm
rotation_step_linf_p95: roughly 1.1e-5
piper labels: binary 0/1
```

## Real Control Bandwidth Inspection

Use this to summarize real Elite/magnetic-arm motion bandwidth from `branchs/`.
This is the preferred anchor for deciding whether a rollout-time rate limit or
control layer is physically plausible:

```powershell
.\.venv\Scripts\python.exe tools\inspect_real_control_bandwidth.py `
  --root branchs `
  --out simulation_output\real_control_bandwidth
```

Outputs:

```text
real_control_bandwidth.md
real_control_bandwidth_summary.json
real_control_bandwidth_paths.json
real_control_bandwidth_paths.csv
```

Current summary:

```text
paths: 24
pose frames: 4039
position step p95 median/max: 5.072 / 5.564 mm/frame
position acceleration p95 median: 2.982 mm/frame^2
position jerk p95 median: 2.835 mm/frame^3
direction reversal fraction median: 0.0000
zero step fraction median: 0.4131
piper labels: {"0": 1849, "1": 2190}
```

If the real sampling rate becomes known, re-run with `--fps`:

```powershell
.\.venv\Scripts\python.exe tools\inspect_real_control_bandwidth.py `
  --root branchs `
  --out simulation_output\real_control_bandwidth_fps30 `
  --fps 30
```

## Rollout-vs-Real Control Bandwidth Comparison

Use this after one or more rollouts exist. It compares simulated
`elirobot_pose`, `magnetic_pose`, and sim-only `tip_pos` against the real
Elite/magnetic-arm bandwidth anchor:

```powershell
.\.venv\Scripts\python.exe -B tools\compare_rollout_control_bandwidth.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_r1 `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_smooth025_r1 `
  simulation_output\baseline_mujoco_physical_feed_action_v2_elitedelta_rollout `
  --real-summary simulation_output\real_control_bandwidth\real_control_bandwidth_summary.json `
  --out simulation_output\rollout_control_bandwidth_comparison
```

Outputs:

```text
rollout_control_bandwidth_comparison.json
rollout_control_bandwidth_comparison.csv
rollout_control_bandwidth_comparison.md
```

If the current environment refuses writes under `simulation_output`, use a
temporary writable output such as:

```powershell
--out docs\_rollout_control_bandwidth_check
```

Initial check result:

```text
v2 rollout: elirobot step_p95 10.065 mm, accel_p95 10.507, jerk_p95 16.144
rate010_r1: elirobot step_p95 5.777 mm, accel_p95 6.013, jerk_p95 7.947
rate010_smooth025_r1: elirobot step_p95 4.369 mm, accel_p95 4.439, jerk_p95 5.766
elitedelta rollout: elirobot step_p95 11.021 mm, accel_p95 7.171, jerk_p95 10.068
real anchor: step_p95 5.072 mm, accel_p95 2.982, jerk_p95 2.835
```

Interpretation:

`rate010_smooth025_r1` matches the real step scale reasonably well, but its jerk
and direction reversals are still high. This explains why the video can still
look jittery even when step p95 is close to real.

## Elite Visual Jitter Source Diagnosis

Use this when numeric bandwidth metrics look good but the video still appears
jittery. It separates policy target jitter, executed joint jerk, logged tool
pose smoothness, pose consistency, and coarse video frame-difference spikes:

```powershell
.\.venv\Scripts\python.exe -B tools\analyze_elite_visual_jitter_source.py `
  simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1_video `
  --out docs\_elite_visual_jitter_source_accel002_video_check
```

Outputs:

```text
elite_visual_jitter_source.json
elite_visual_jitter_source.csv
elite_visual_jitter_source.md
```

Current `rate010_accel002_r1_video` diagnosis:

```text
left:  target_step_p95=0.05348, executed_accel_p95=0.00200,
       executed_jerk_p95=0.00400, tool_jerk_p95=2.039 mm
right: target_step_p95=0.04561, executed_accel_p95=0.00200,
       executed_jerk_p95=0.00400, tool_jerk_p95=2.024 mm
diagnosis: policy_target_jitter_remains, executed_joint_jerk_high,
           logged_tool_pose_smooth
```

Interpretation:

If the video still jitters here, do not treat it as a simple tool-position
bandwidth problem. The logged tool pose is already smooth; remaining jitter is
more likely from policy target jumps, joint-chain motion, or robot visual/link
mapping.

## Diagnostic Video With Robot Visuals Hidden

Use this only to isolate whether the visible robot meshes are the source of the
video jitter. Keep `--robot-visual-mode kinematic` so policy side/top camera
inputs stay in-distribution; `--diagnostic-video-hide-robots` hides robot geoms
only in the saved video frames. Add `--diagnostic-video-overlay` when the
magnetic point should remain visible after robot geoms are hidden:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v2\best_model.pt `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_diag_hide_robots_overlay_video `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --elite-joint-rate-limit 0.010 `
  --elite-joint-accel-limit 0.002 `
  --robot-visual-mode kinematic `
  --video `
  --video-camera overview `
  --render-every 2 `
  --diagnostic-video-hide-robots `
  --diagnostic-video-overlay
```

Do not use `--robot-visual-mode none` for this diagnostic. That also changes
policy inputs and caused the model to fail with sustained Piper rollback.

## Sim-vs-Real Alignment Table

Use this after real-data inspection and after a representative MuJoCo dataset or
rollout exists:

```powershell
.\.venv\Scripts\python.exe tools\compare_sim_real_alignment.py `
  --real-paths simulation_output\real_branch_data_inspection\real_branch_paths.json `
  --sim-manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --rollout simulation_output\baseline_mujoco_physical_feed_action_v2_rollout `
  --out simulation_output\sim_vs_real_alignment
```

Outputs:

```text
simulation_output\sim_vs_real_alignment\sim_vs_real_alignment.json
simulation_output\sim_vs_real_alignment\sim_vs_real_alignment.md
docs\sim-vs-real-alignment.md
```

Important semantic constraint:

```text
Real branch pose = Elite/magnetic-arm trajectory, not guidewire trajectory.
Compare it with simulated elirobot_pose/magnetic_pose, not with tip_pos as real ground truth.
```

## Controlled Checkpoint Comparison

Use this when comparing two models in the same environment. Keep all parameters
identical except `--checkpoint` and `--out`.

Unweighted reference:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v1\best_model.pt `
  --out simulation_output\compare_unweighted_newenv_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

Weighted reference:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_weighted_v1\best_model.pt `
  --out simulation_output\compare_weighted_newenv_rollout `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

Then diagnose both:

```powershell
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py simulation_output\compare_unweighted_newenv_rollout --no-plots
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py simulation_output\compare_weighted_newenv_rollout --no-plots
```

## TCP-Delta Elite Control Path

Use this path after the TCP-delta action implementation. It matches the
inherited Elite control semantics more closely: model predicts Elite TCP delta,
rollout converts TCP target to joints through IK.

Collect a fresh formal dataset with TCP action labels:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.collect_tip_guided_wire `
  --out simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta `
  --tasks left right `
  --episodes-per-task 20 `
  --max-attempts-per-episode 3 `
  --seed 9971 `
  --start-fraction-min 0.58 `
  --start-fraction-max 0.70 `
  --success-mode env_success `
  --max-steps 600 `
  --sample-every 5 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --robot-visual-mode kinematic `
  --action-mode piper_feed_elite_joint `
  --hide-tool-markers `
  --hide-path-tubes `
  --expert-mode route_plan `
  --formal-data `
  --route-plan-step 0.24 `
  --elite-ahead 0.010 `
  --piper-cmd 0.70 `
  --piper-command-period 40 `
  --piper-command-width 20 `
  --route-plan-command-phase-lock `
  --progress-log-every 100 `
  --keep-failures
```

Train the senior-like TCP-delta baseline:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema senior_piper_real_like `
  --elite-action-representation tcp_delta `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Run open-loop diagnostics before any rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls_open_loop_diag `
  --batch-size 64
```

Current v10 open-loop interpretation:

```text
piper_sign_mismatch_fraction ~= 0.397
elite_linf median/p95 ~= 0.947 / 1.676 in TCP pose-delta metric space
expert/predicted Elite target-step p95 ~= 7.20 / 7.01
```

Elite TCP-delta scale is now close to the expert target scale, so do not
immediately roll out or tune Elite smoothing. Piper remains weak, and the error
tracks the scheduled route-plan command phase: `step % 40` is feed for the
first half of the cycle and hold for the second half, but
`senior_piper_real_like` does not expose that controller phase. This is a label
/ observation-closure issue, not evidence that more BC rollout tuning is the
right next step.

Diagnostic phase-state training. This keeps the same formal dataset and
real-style TCP-delta Elite action, but adds explicit controller phase
`sin/cos(step modulo 40)` to the senior-like state. It is a diagnostic for
scheduled Piper labels, not a new oracle simulator input:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 600 `
  --observation-schema senior_piper_real_like_with_phase `
  --piper-command-period-for-state 40 `
  --elite-action-representation tcp_delta `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Then diagnose before any rollout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls\best_model.pt `
  --manifest simulation_output\formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta\manifest.json `
  --out simulation_output\baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls_open_loop_diag `
  --batch-size 64
```

Current phase-state diagnostic result:

```text
piper_sign_mismatch_fraction ~= 0.371
elite_linf median/p95 ~= 0.910 / 1.577 in TCP pose-delta metric space
expert/predicted Elite target-step p95 ~= 7.20 / 7.62
```

Interpretation: adding explicit controller phase only slightly improves Piper
classification (`0.397 -> 0.371`) and does not justify rollout as the next
default. Treat this as evidence that scheduled Piper feed/hold should become an
explicit real controller/state-machine decision, or be kept outside the BC
target, rather than another small-BC tuning target.

## Estimated Tip / Contact Interface Training

Use this only after the dataset passes the provenance audit for
`real_direct_plus_estimated_tip_contact`. This is a small interface check for
estimated 3D tip/contact observations plus real-style Elite TCP delta and Piper
step classification. Diagnose open-loop before any rollout.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1\manifest.json `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --observation-schema real_direct_plus_estimated_tip_contact `
  --elite-action-representation tcp_delta `
  --piper-head step_classification `
  --piper-step-feed-value 0.7
```

Then run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\formal_tip_line_estimated_tip_3d_top_manual_dataset_v1\manifest.json `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_open_loop_diag `
  --batch-size 64 `
  --image-size 224 `
  --max-steps 700
```

If the open-loop diagnostic is acceptable, rollout must use the same camera and
formal guidewire visual semantics as the estimated-tip dataset. The checkpoint
schema enables the policy estimator automatically.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls\best_model.pt `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_r1 `
  --tasks left right `
  --env-type tip `
  --guidance-mode physical `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-visual-mode line `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 5 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --policy-estimator-mode auto `
  --policy-every 5 `
  --progress-log-every 100
```

If rollout reaches the target by nearly continuous Piper feeding and shows
visible contact, test the real-style Piper step controller. This keeps the
policy output as feed/hold intent but executes at most one Piper primitive per
cooldown window, matching the senior `piper.step_forward(...)` style more
closely than continuous cached feed.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls\best_model.pt `
  --out simulation_output\baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_piperctrl_r1 `
  --tasks left right `
  --env-type tip `
  --guidance-mode physical `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-visual-mode line `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 5 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --policy-estimator-mode auto `
  --policy-every 5 `
  --piper-step-controller `
  --piper-step-cooldown 10 `
  --progress-log-every 100
```

## Observation Provenance Audit

Use this before treating a dataset/model run as formal sim-to-real evidence.
It checks the selected `--observation-schema`, not merely which diagnostic
fields are stored in samples. Storing MuJoCo truth for diagnostics is allowed;
feeding it to the policy is not.

Current v12 senior-like schema should pass:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\audit_observation_provenance.py `
  simulation_output\formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta `
  --observation-schema senior_piper_real_like `
  --sample-limit 50 `
  --out docs\_observation_provenance_v12_seniorlike_check
```

Legacy/full simulator state should fail this formal gate because it consumes
privileged simulator fields:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B tools\audit_observation_provenance.py `
  simulation_output\formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta `
  --observation-schema full_sim_state `
  --sample-limit 50 `
  --out docs\_observation_provenance_v12_fullsim_check
```

If future data uses estimator/tactile fields, the manifest should include
`observation_provenance`; see:

```text
docs/estimator-tactile-interface-spec.md
docs/real-observable-interface-audit.md
```

## Current Registered-Geometry Wallguard Rollout

Use this after training a
`real_direct_plus_estimated_tip_registered_geometry` checkpoint. This is the
currently accepted controller candidate: it does not hold Piper and does not
add active away-wall TCP motion. It only removes the predicted Elite TCP-delta
component that points toward the estimated wall normal.

Do not reuse the rejected strong wallguard setting that held Piper and added
active away-wall correction; it broke tip-magnet coupling and caused the left
branch to fail.

```powershell
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls\best_model.pt `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_r1 `
  --tasks left right `
  --env-type tip `
  --guidance-mode physical `
  --route-config simulation\routes\vessel_0422_wire_route_v1.json `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --wire-visual-mode line `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 8 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --wire-segments 240 `
  --policy-estimator-mode auto `
  --policy-every 5 `
  --registered-geometry-wall-safety-guard `
  --registered-geometry-wall-risk-threshold 0.50 `
  --registered-geometry-wall-margin-threshold-m 0.0015 `
  --registered-geometry-wall-pull-threshold-m 0.0025 `
  --no-registered-geometry-wall-hold-piper `
  --registered-geometry-wall-away-delta-mm 0.0 `
  --registered-geometry-wall-max-correction-mm 0.4 `
  --formal-eval `
  --progress-log-every 100
```

## Real Pilot Data Collection

Use this when collecting a small real-system pilot for shadow-mode validation.
This replaces the old hard-coded `data_collect_mode1.py` /
`data_collect_mode2.py` path for new real pilot captures.

The collector is safe by default: it records operator Piper labels, images,
Elite TCP pose, Piper state, and controller logs, but it does not command Piper
and does not advance `piper_step` unless a real execution mode is selected.

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

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260704_left_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300
```

In this pure log-only mode, the guidewire will not move unless the operator or
another program controls Piper outside this script. For a real moving pilot
where Piper is controlled externally, add:

```powershell
--external-piper-control
```

This still does not call Piper hardware from the collector, but it treats the
keyboard feed/retract labels as externally executed commands for `piper_step`
accounting.

Replacement UDP feeder device mode:

```text
transport: UDP JSON
default target: 192.168.5.22:8888
forward/backward semantics: one packet = one feeder step
nominal feed scale: 12 mm/step, onsite observed range about 11-13 mm/step
status: experimental fallback only; repeated onsite trials were not good enough
        for current mainline data collection
```

The collector records this as `feeder_device_control_enabled`, not as the old
PiperRobot path. `piper_step` remains the integer feed/retract step count, while
`piper_insertion_length` is the nominal millimeter estimate from
`--feeder-step-mm`.

As of 2026-07-10, do not use this UDP feeder mode as the default real-data
collection route. It is kept for protocol/debug reuse, but the device showed
enough practical delivery problems in repeated onsite tests that data collected
with it should be treated as hardware-diagnostic or fallback data, not clean
expert demonstration data.

If the real entry S-shaped bend has been removed, re-test feed-only insertion
before permanently rejecting Piper or the feeder. The current hypothesis is
that the earlier push failure may have been dominated by the real mechanical
S-bend blockage rather than by actuator control alone. Treat this as a smoke
test first: watch whether the guidewire advances smoothly, then validate the
records before using the data.

Standalone one-step feeder probe on Ubuntu:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device \
  --action forward \
  --host 192.168.5.22 \
  --port 8888 \
  --execute
```

Standalone five-step feeder probe:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device \
  --action forward \
  --count 5 \
  --interval-s 1.0 \
  --host 192.168.5.22 \
  --port 8888 \
  --execute
```

Lab-side UDP-feeder collection template on Ubuntu, using the current observed
camera IDs from the 2026-07-07 setup (`10=side`, `4=top`; re-check if cameras
were unplugged):

```bash
python data/collect/collect_real_shadow_pilot.py \
  --out real_pilot_YYYYMMDD_left_udp_feeder_001 \
  --task left \
  --side-source opencv \
  --side-camera-id 10 \
  --top-source opencv \
  --top-camera-id 4 \
  --pose-source elite \
  --elite-ip 192.168.5.66 \
  --warmup-frames 30 \
  --sample-period 0.2 \
  --max-frames 300 \
  --collection-mode mode1_auto_piper_manual_elite \
  --piper-label-source auto_periodic \
  --piper-auto-feed-period 0.8 \
  --piper-auto-feed-count 14 \
  --stop-after-auto-feed-count \
  --enable-feeder-device-control \
  --feeder-host 192.168.5.22 \
  --feeder-port 8888 \
  --feeder-step-mm 12.0 \
  --piper-control-source feeder_device \
  --elite-control-source human \
  --session-note "UDP feeder: auto periodic feed, human/manual Elite guidance"
```

Validate after collection:

```bash
python tools/validate_real_shadow_pilot.py \
  real_pilot_YYYYMMDD_left_udp_feeder_001 \
  --check-readable-images
```

Keyboard labels during preview:

```text
0 or h: hold
1 or f: feed
r or b: retract label
q or esc: stop
```

Output layout:

```text
manifest.json
records.jsonl
frames/side/*.png
frames/top/*.png
logs/elite_tcp_pose.csv
logs/piper_state.csv
logs/controller_commands.csv
summary.json
```

Validate a collected pilot immediately:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_20260704_left_001 `
  --check-readable-images
```

Render collected PNG frames back into a video:

```powershell
.\.venv\Scripts\python.exe tools\render_real_shadow_video.py `
  collected_data\real_align_20260707_left_elite_step_calib_001 `
  --camera hstack
```

The real collection format stores frame images under `frames/side/*.png` and
`frames/top/*.png`, with paths and timestamps listed in `records.jsonl`.
`--camera hstack` renders side and top views side by side. Use `--camera side`
or `--camera top` for a single view. The script infers FPS from record
timestamps unless `--fps` is provided.

Before collecting motion data, run a short camera/Elite-pose smoke that does
not move Piper:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_smoke_left_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 30 `
  --collection-mode shadow_static
```

Then validate:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_smoke_left_001 `
  --check-readable-images
```

Mode1-style pilot: Piper advances periodically while the operator controls
Elite/magnetic guidance. This is the closest replacement for senior
`data_collect_mode1.py`, but it writes the current `records.jsonl` schema.
Only use this after physical setup/supervision is confirmed, because the
collector commands Piper:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260704_left_mode1_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300 `
  --collection-mode mode1_auto_piper_manual_elite `
  --piper-label-source auto_periodic `
  --piper-auto-feed-period 0.8 `
  --piper-auto-feed-count 14 `
  --stop-after-auto-feed-count `
  --enable-piper-control `
  --piper-control-source collector `
  --elite-control-source human `
  --session-note "mode1: collector periodic Piper feed, human/manual Elite guidance"
```

Mode2-style pilot: Elite is moved by the operator or a senior-style external
path script while this collector commands Piper from keyboard labels. This is
the closest replacement for senior `data_collect_mode2.py` when using the new
schema:

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260704_left_mode2_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300 `
  --collection-mode mode2_auto_elite_manual_piper `
  --piper-label-source keyboard `
  --enable-piper-control `
  --piper-control-source collector `
  --elite-control-source senior_script `
  --session-note "mode2: external/senior Elite path, keyboard Piper feed labels"
```

If another program controls Piper and the collector should only record, use:

```powershell
--external-piper-control
```

With external Piper control, also set `--piper-control-source human` or
`--piper-control-source senior_script` so the dataset records who actually
executed the motion.

## Real-System Shadow-Mode Adapter

Use this before commanding real robots. It runs the current dual-camera/state
policy on real-system-style image/state records and writes raw policy outputs
plus the conservative Elite-only wallguard output. It never calls Elite or Piper
hardware APIs.

To convert the senior `branchs/` data into this format:

```powershell
.\.venv\Scripts\python.exe tools\build_branchs_shadow_input.py `
  --branch-root branchs `
  --out simulation_output\branchs_shadow_input_visualcontact_full.jsonl `
  --visual-contact-estimator
```

This conversion is diagnostic only. `branchs/` contains one camera, Elite 6D
pose, and Piper 0/1 labels, but no top camera, estimated 3D tip, or registered
wall-geometry estimator fields. The converter therefore duplicates the single
image as both side/top input and writes low-confidence estimator placeholders.
With `--visual-contact-estimator`, it also runs a simplified HSV red-tip plus
image-edge-distance estimator that fills `estimated_contact_flag`,
`contact_estimator_confidence`, and `estimated_image_distance_px`. This is a
temporary 2D contact/tactile-like signal, not an estimated 3D tip replacement.

Input JSONL schema, one record per frame:

```json
{
  "task": "left",
  "step": 0,
  "side_image": "path/to/side.png",
  "top_image": "path/to/top.png",
  "state": {
    "elite_tcp_pose_6d": [0, 0, 0, 0, 0, 0],
    "piper_step": 0,
    "piper_insertion_length": 0.0,
    "estimated_tip_pos_3d": [0, 0, 0],
    "estimated_tip_heading_3d": [0, 0, 0],
    "tip_estimator_confidence": 0.0,
    "tip_estimator_visible": 0,
    "estimated_contact_flag": 0,
    "contact_estimator_confidence": 0.0,
    "estimated_image_distance_px": 0.0,
    "estimated_wall_margin": 0.0,
    "estimated_wall_margin_fraction": 0.0,
    "route_estimator_confidence": 0.0,
    "estimated_magnet_wall_pull": 0.0,
    "estimated_wall_side_risk": 0.0,
    "estimated_wall_normal_3d": [0, 0, 0],
    "estimated_route_tangent_3d": [0, 0, 0]
  }
}
```

Run shadow inference:

```powershell
.\.venv\Scripts\python.exe tools\real_shadow_policy_adapter.py `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls\best_model.pt `
  --input-jsonl path\to\real_shadow_input.jsonl `
  --image-base-dir . `
  --out simulation_output\real_shadow_predictions.jsonl
```

The output records:

```text
raw_policy_action          # model output before safety filtering
shadow_controller_action   # action after Elite-only wallguard
wallguard                  # active/corrected/reason/correction metadata
```

Use `raw_policy_action` to judge model ability and
`shadow_controller_action` to judge real-system execution behavior. A high
wallguard intervention rate means the model is not yet reliable, even if the
guarded action looks safer.

## Branchs-Native Real-Data Baseline

Use this to test whether the model/training pipeline can learn the senior real
data format itself. This is not a final VLA model and not a complete
sim-to-real interface; it is a real-data diagnostic baseline.

Build the manifest:

```powershell
.\.venv\Scripts\python.exe tools\build_branchs_training_manifest.py `
  --branch-root branchs `
  --out simulation_output\branchs_training_manifest_full.json
```

The generated labels are:

```text
input:  duplicated single camera image + Elite TCP 6D pose + Piper step + task
output: Elite TCP delta from pose[t+1] - pose[t] + Piper hold/feed label
```

Train:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\baseline_branchs_native_seniorlike_tcpdelta_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --observation-schema senior_piper_real_like `
  --elite-action-representation tcp_delta `
  --piper-head step_classification
```

Diagnose before any rollout or real control:

```powershell
.\.venv\Scripts\python.exe tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_branchs_native_seniorlike_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\baseline_branchs_native_seniorlike_tcpdelta_pipercls_open_loop_diag `
  --batch-size 32 `
  --image-size 224
```

## Single-Camera Senior-Like Sim Alignment

Use this to test whether a sim-trained model still fails on `branchs` after
matching the single-camera senior-like input shape more closely. This is a
schema/domain diagnostic, not a rollout-tuning step.

Build a sim manifest that duplicates one selected camera as both side/top:

```powershell
.\.venv\Scripts\python.exe tools\build_single_camera_seniorlike_manifest.py `
  --manifest simulation_output\formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1\manifest.json `
  --out simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --camera side
```

Train:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --observation-schema senior_piper_real_like `
  --elite-action-representation tcp_delta `
  --piper-head step_classification
```

Diagnose on sim same-schema data:

```powershell
.\.venv\Scripts\python.exe tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls_open_loop_diag `
  --batch-size 32 `
  --image-size 224
```

Diagnose the same sim-trained checkpoint on senior real `branchs` data:

```powershell
.\.venv\Scripts\python.exe tools\diagnose_bc_open_loop.py `
  --checkpoint simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls\best_model.pt `
  --manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls_on_branchs_open_loop_diag `
  --batch-size 32 `
  --image-size 224
```

Current result:

```text
sim same-schema open-loop:
  Piper mismatch 21.17%, Elite TCP-delta linf median/p95 0.44/1.69
branchs open-loop:
  Piper mismatch 45.73%, all hold frames predicted as feed,
  Elite TCP-delta linf median/p95 5.32/9.71
branchs shadow:
  feed=4039, hold=0, mismatch=1849/4039=45.78%
```

Interpretation:

```text
Single-camera schema matching alone does not close the sim-to-real gap.
Continue toward visual/domain alignment or real-data collection rather than
treating the missing second camera as the only blocker.
```

## Sim/Real Visual-Domain Audit

Use this after a sim-to-real shadow or branchs open-loop mismatch to check
whether the image distributions are already visibly different before tuning the
policy.

```powershell
.\.venv\Scripts\python.exe tools\audit_sim_real_visual_domain.py `
  --sim-manifest simulation_output\formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json `
  --real-manifest simulation_output\branchs_training_manifest_full.json `
  --out docs\_sim_real_visual_domain_audit_singlecam_v1 `
  --camera side `
  --samples-per-task 12
```

Current result:

```text
sample pairs: 24
missing inputs: 0
luminance_mean: sim 184.38 vs real 120.49
saturation_mean: sim 0.278 vs real 0.388
bright_ratio: sim 0.324 vs real 0.002
edge_density: sim 0.034 vs real 0.026
contact sheet: docs/_sim_real_visual_domain_audit_singlecam_v1/contact_sheet.png
report: docs/_sim_real_visual_domain_audit_singlecam_v1/report.md
```

Manual read:

```text
The sim side images are bright, clean, pale-rendered scenes with translucent
vessel geometry and synthetic tabletop/background. The real branchs images are
darker, green-background physical camera frames with real robot/fixture
appearance, stronger crop/perspective, and real occlusion. This is an obvious
visual-domain gap and supports fixing rendering/domain alignment or collecting
new aligned real data before another BC rollout-tuning loop.
```

## Branchs-Like MuJoCo Render Preset

Use this for the current A route: keep MuJoCo, but render formal images closer
to the senior `branchs` side-camera distribution before trying more model
tuning. This is a renderer/domain diagnostic, not a policy-quality result.

Short smoke:

```powershell
.\.venv\Scripts\python.exe -m simulation.collect_formal_tip_line_guidance `
  --out simulation_output\_smoke_branchs_like_render_preset_v1 `
  --tasks left right `
  --episodes-per-task 1 `
  --max-attempts-per-episode 1 `
  --max-steps 320 `
  --sample-every 20 `
  --progress-log-every 100 `
  --image-size 224 `
  --render-width 960 `
  --render-height 720 `
  --camera-config simulation\camera_configs\mujoco_camera_top_manual_v1.json `
  --render-domain-preset branchs_like_v1 `
  --wire-segments 240 `
  --wire-visual-radius 0.0008 `
  --wire-visual-rgb "0.02 0.02 0.018" `
  --wire-tip-visual-rgb "0.78 0.04 0.02" `
  --wire-tip-visual-segments 8 `
  --wire-tip-visual-radius-scale 2.2 `
  --wire-tip-marker-radius 0.0020 `
  --wire-tip-marker-alpha 0.95 `
  --formal-data
```

Audit against `branchs`:

```powershell
.\.venv\Scripts\python.exe tools\audit_sim_real_visual_domain.py `
  --sim-manifest simulation_output\_smoke_branchs_like_render_preset_v1\manifest.json `
  --real-manifest simulation_output\branchs_training_manifest_full.json `
  --out docs\_sim_real_visual_domain_audit_branchs_like_v1 `
  --camera side `
  --samples-per-task 11
```

Current smoke/audit result:

```text
left/right env_success: 2/2
contact_p95/max: 0.0 / 0.0
luminance_mean: sim 118.60 vs real 120.76
saturation_mean: sim 0.407 vs real 0.392
bright_ratio: sim 0.000 vs real 0.003
edge_density: sim 0.027 vs real 0.026
contact sheet: docs/_sim_real_visual_domain_audit_branchs_like_v1/contact_sheet.png
```

User visual review:

```text
Accepted as usable for the next step. Remaining major visual gaps are real
glass-vessel reflections and real-camera blur; physical fixture/crop/occlusion
differences also remain. Treat this as good enough to collect a small
branchs-like dataset before training, not as final visual realism.
```

Plan C renderer-only Isaac spike:

```text
Current machine does not need Isaac Sim installed. Export a portable package
locally, copy it to the Isaac host, and build USD stages there. This is not a
physics/control migration.
```

Export package locally:

```powershell
.\.venv\Scripts\python.exe tools\export_isaac_renderer_spike.py `
  --sim-manifest simulation_output\formal_tip_line_branchs_like_v1_full\manifest.json `
  --real-manifest simulation_output\branchs_training_manifest_full.json `
  --out simulation_output\isaac_renderer_spike_branchs_like_v1_pkg `
  --camera side `
  --samples-per-task 3 `
  --copy-reference-images
```

Copy the whole output directory to the Isaac host:

```text
simulation_output\isaac_renderer_spike_branchs_like_v1_pkg
```

Build USD stages on the Isaac host from inside the copied package:

```powershell
<ISAAC_SIM_ROOT>\python.bat isaac_build_stage.py --package package_manifest.json --headless
```

Linux equivalent:

```bash
<ISAAC_SIM_ROOT>/python.sh isaac_build_stage.py --package package_manifest.json --headless
```

Expected package contents:

```text
package_manifest.json
samples.jsonl
README.md
isaac_build_stage.py
assets/
reference_images/
```

Expected Isaac-side output:

```text
isaac_assets/vessel_0422.usd
isaac_stages/<frame_id>.usd
```

Acceptance criterion:

```text
First open the generated USD files in Isaac and visually compare them against
reference_images/sim and reference_images/real. Only add automated Isaac
rendering after the USD scene loads correctly on that Isaac version.
```

## Algorithm / VLA Commands

Current algorithm commands live in:

```text
docs/algorithm-track-commands.md
```

The pre-split Pi-Style/OpenPI/PI05 command history is preserved at:

```text
docs/archive/algorithm-command-history-before-track-split-2026-07-17.md
```

## Documentation Updates After Experiments

After meaningful data/simulation/real-collection experiments, update:

```text
docs/data-track-handoff.md
docs/experiment-registry.md
```

Algorithm experiments update:

```text
docs/algorithm-track-handoff.md
docs/algorithm-track-commands.md  # only for reusable command changes
```

Update shared `docs/project-state.md` or `docs/handoff.md` only when the
cross-track interface or routing changes. Group-meeting facts from either track
go into the shared append-only `docs/weekly-meeting-log.md`.

If a decision changes, update:

```text
docs/decision-log.md
```

If a new recurring failure is discovered, update:

```text
docs/troubleshooting.md
```
