# Project State

Last updated: 2026-07-07

## Current Mainline

The active project route is still MuJoCo-first simulation for dual-arm
guidewire intervention. The goal is not to make a small BC model look good in
simulation; the goal is to make synthetic data generation and control semantics
close enough to the real system to be useful.

Highest priority:

```text
simulation realism and real-system correspondence > BC rollout success
```

The small BC model is only a loop-closure diagnostic. A rollout is not accepted
as solved if it depends on simulation-only oracle state, hidden centerline/tip
correction, exact wall/contact feedback, or a controller that cannot plausibly
exist on the real setup.

Current formal-data guardrails:

```text
simulation.collect_mujoco_physical_guidance --formal-data
simulation.collect_tip_guided_wire --formal-data
simulation.eval_mujoco_guided_wire_rollout --formal-eval
```

The current preferred external action schema remains:

```text
piper_feed_elite_joint
```

Meaning:

- Piper command is a guidewire-level feed/hold/retract scalar or signed-step
  label near the vessel entrance.
- Elite execution still happens through joints, but the real-aligned policy
  target is now TCP delta followed by IK.
- Senior real `branchs` data is an Elite/magnetic-arm and image reference, not
  guidewire path ground truth.

The target VLA-facing interface is documented separately:

```text
docs/vla-target-interface.md
docs/control-layer-contract.md
```

Short version:

```text
observation + task instruction -> Elite TCP delta + Piper discrete command
```

The current small BC baseline is only a miniature test of this contract, not an
independent endpoint.

## Current Environment

Primary environment:

```text
simulation/mujoco_guided_wire_env.py
```

Important assets/configs:

```text
utils/interface/model/0422.stl
simulation/routes/vessel_0422_wire_route_v1.json
simulation_output/robot_scene_mvp/scene_config.json
simulation_output/mujoco_camera_config.json
simulation/camera_configs/mujoco_camera_side_coverage_v1.json
```

Current effective assumptions:

- The magnetic point is attached to the Elite end tool.
- Formal route-plan experts may use a registered route and scheduled progress,
  but not exact tip/contact/wall oracle feedback.
- Formal tip-line collection now samples earlier guidewire starts by default:
  `--start-fraction-min 0.42 --start-fraction-max 0.54`. The previous
  `0.58-0.70` range placed the red tip too close to the bifurcation/branch
  area for current visual review.
- The tip-centric route now uses the hard-elastic guidewire assumption:
  the controlled/observable head is primary, while the rendered guidewire is a
  continuous line-shaped tail rather than the older ball/segment-looking visual.
- New formal data collection should use
  `simulation.collect_formal_tip_line_guidance`; the older
  `simulation.collect_tip_guided_wire` and
  `simulation.collect_mujoco_physical_guidance` entrypoints are legacy/
  diagnostic only.
- Debug markers and path tubes are hidden by construction in the mainline
  collector.
- Real control bandwidth is anchored by `simulation_output/real_control_bandwidth`
  rather than by rollout success alone.

Current line-visual smoke:

```text
simulation_output/_smoke_tip_line_wire_formal_hidden
task: left
success: true
steps: 208
samples: 5
min_tip_wall: ~1.375 mm
max_contact_strength: ~0.008
visual check: side/top frames show continuous wire and no debug marker/path-tube leakage
```

Current mainline-collector smoke:

```text
simulation_output/_smoke_formal_tip_line_mainline_entry
collector: simulation.collect_formal_tip_line_guidance
task: left
success: true
steps: 208
samples: 5
min_tip_wall: ~1.779 mm
max_contact_strength: 0.0
```

Current accepted formal guidewire visual:

```text
wire_visual_radius: 0.0008
wire_visual_rgb: black tail
wire_tip_visual_rgb: red head
wire_tip_visual_segments: 8
wire_tip_visual_radius_scale: 2.2
wire_tip_marker_radius: 0.0020
wire_segments: 240
wire_visual_offset: 0 0 0
visual review: accepted by user after comparison with real side/top images
```

Current accepted visual-route baseline:

```text
dataset: simulation_output/formal_tip_line_visual_route_v1_small_fix1
camera_config: simulation/camera_configs/mujoco_camera_top_manual_v1.json
route: Piper outlet -> vessel entry -> fixed S-bend visual prefix -> dynamic in-vessel route -> red tip
accepted_episodes: 20
attempts: 20
samples: 567
left/right success: 10/10 and 10/10
max_contact_strength: 0.0
contact_p95 max: 0.0
min_tip_distance_to_wall: ~1.386 mm
max_tip_to_magnetic: ~8.429 mm
visual review: accepted after fixing right-branch red-tip jump and visual tail overshoot
```

Interpretation:

```text
This is the current line-guidewire visual baseline to scale from. The rendered
guidewire now uses a fixed user-tuned entry/S-bend prefix and route-progress
filtered dynamic points so future right-branch points cannot appear ahead of
the current tip. Continue from this route instead of returning to the older
ball/segment wire visual or the previous misaligned wall/route projection.
```

Current image-distance estimator status:

```text
config: simulation/camera_configs/mujoco_camera_side_coverage_v1.json
accepted visual smoke: docs/_smoke_image_estimator_wall_fix_v3_visual_check
estimator method: rendered_image_red_tip_to_local_edge_distance
source image: the just-rendered side/top camera frames, not 3D cross-section projection
visible rule: if the red tip is occluded or too close to the image edge, emit missing/low-confidence instead of a wall point
```

Interpretation:

```text
The earlier sidecam-v1 dataset revealed a more serious estimator semantic bug:
the projected wall point could fly to a visibly wrong location, especially in
the top camera. The fixed estimator now computes the tip/wall distance from the
rendered image and is intentionally conservative. Current smoke shows the side
view gives reasonable red-tip/local-edge measurements, while the top view often
marks the tip as unavailable because Elite occludes the red head. This is
preferable to fabricating a precise-looking but wrong top-camera wall point.
```

## Current Reference Data

Real-system alignment/calibration audit:

```text
docs/real-alignment-20260707-calibration-audit.md
source data: collected_data/real_align_20260707_*
audit output: docs/_real_alignment_20260707_audit
```

Interpretation:

```text
The 2026-07-07 real captures are calibration/alignment data, not expert
demonstrations. Elite-only step data is the cleanest quantitative anchor:
event-window TCP movement is roughly 14-18 mm depending on branch. Piper-only
burst data confirms command execution and physical response, but it should only
be measured as guidewire forward progress along the vessel/entry direction.
Whole-frame visual differences are too weak/noisy for precise feed-distance
calibration without local guidewire/tip tracking or manual annotation.
The first MuJoCo Piper primitive probe shows that real `step_forward(0.8s)`
should not be mapped to one simulator step: comparable visible red-tip motion
requires roughly several simulated feed steps, and repeated real feed commands
are not expected to add linearly because of friction, slip, elastic storage,
and physical blocking.

Replacement UDP feeder device status:

```text
interface: UDP JSON to 192.168.5.22:8888
collector mode: --enable-feeder-device-control
nominal measured scale: about 11-13 mm per forward command, using 12 mm/step
current decision: fallback/debug only, not the real-data mainline
```

Repeated onsite trials showed that the replacement feeder device still has
enough practical delivery/stability problems that it should not be used as the
default source of clean real expert data. Keep the adapter and commands for
future hardware debugging or rollback, but do not build the next data strategy
around this device unless the mechanical delivery issues are fixed.

Updated onsite result after fixture change:

```text
previous symptom: Piper / feeder could not reliably push the guidewire forward
fixture change: the real S-shaped bend has been removed
result: feed-only control improved enough to prove CAN/Piper control works, but
        guidewire delivery still remains unreliable for clean data collection
current decision: the S-shaped bend was one blocking factor, not the only
                  limiting factor; pause onsite collection and wait for
                  supervisor / hardware-path guidance
```

This does not change the accepted MuJoCo visual route, which still contains a
user-tuned S-bend visual prefix for the simulated vessel-entry appearance. The
new conclusion is about the real onsite fixture/mechanical path: removing the
real S-shaped bend helps isolate the issue, but it does not by itself make the
Piper/feeder route reliable enough for clean real expert data. Treat the latest
onsite runs as hardware/fixture diagnostics, not as final demonstration data.

Implemented Piper primitive controller:

```text
Policy/expert Piper output remains a discrete real-style intent:
  hold/feed/retract

Simulator execution can now map one feed intent to a short controller primitive:
  --piper-primitive-steps N
  --piper-primitive-feed-value V

When `--piper-primitive-steps > 1`, the formal route-plan collector uses event
semantics for Piper: a sparse `piper_step_command=1` starts one real-style feed
primitive, while `controller_state.piper_executed_feed` logs the continuous
simulator execution during the busy window. This keeps policy labels closer to
real `piper.step_forward(...)` calls instead of treating every MuJoCo frame in
a feed window as a separate real command.
```
```

Clean signed-Piper formal expert setting:

```text
--route-plan-step 0.15
--piper-cmd 0.70
--piper-command-period 40
--piper-command-width 20
--route-plan-command-phase-lock
```

Reference datasets:

```text
simulation_output/formal_route_plan_tip_signedpiper_dataset_v8
simulation_output/formal_route_plan_tip_signedpiper_dataset_v9_tcp6d
simulation_output/formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta
```

Both are clean formal expert datasets:

```text
accepted/env_success: 40/40
samples: 1807
image refs: 3614/3614
formal_valid: true
piper_step_command feed/hold: 951/856
piper_feed values: 0.7 / 0.0
transition_fraction median: ~0.244
contact_p95 max: 0.0
tip_to_magnetic episode p95 median: ~7.6-7.7 mm
```

The v9 dataset additionally records real TCP-6D fields:

```text
state.elite_tcp_pose_6d
state.robot_state.elite_tcp_pose_6d
format: xyz_mm + rpy_rad
```

The v12 dataset is the current control-state/TCP-delta reference:

```text
accepted/env_success: 40/40
samples: 1796
missing images/controller_state/elite_tcp_delta: 0
contact_p95/max: 0.0 / 0.0
tip_to_magnetic median/p95/max: ~2.10 / 7.66 / 8.43 mm
piper controller pairs: hold->hold 884, feed->feed 912
elite controller types: tcp_delta 1756, reset 40
```

The first visual-distance estimator dataset exposed a projection-validity bug
and must not be used as a formal reference:

```text
dataset: simulation_output/formal_tip_visual_distance_estimator_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4328
missing images: 0
formal_valid: true
visual estimator: synthetic_visual_distance_estimator, all_visible rule
estimated_contact_flag: 99/4328 = 2.29%
estimated_image_distance_px p50/p95: ~2.61 / 6.04 px
piper_step_command feed/hold: 2283/2045
piper_feed values: 0.7 / 0.0
provenance audit: real_direct_plus_visual_contact passes
audit: docs/_visual_distance_estimator_dataset_v1_audit
visual check: docs/_visual_distance_estimator_dataset_v1_visual_check
```

Interpretation:

```text
This dataset is valid as a bug-finding diagnostic only. The visual audit showed
that side-camera projected tip/wall pixels were often outside the rendered image,
but the estimator still counted the side view as visible because it only checked
depth. All visual contact flags in this dataset were affected by side off-frame
projection. Do not train on this estimator dataset.
```

The estimator has been patched so projected pixels are valid only when they have
positive depth and fall inside the rendered image. Smoke validation after the
patch:

```text
dataset: simulation_output/_smoke_visual_distance_estimator_frameclip
task: right
episodes/env_success: 1/1
samples: 104
side in-frame samples: 25/104
top in-frame samples: 104/104
estimated_contact_flag: 0/104
visual check: docs/_visual_distance_estimator_frameclip_smoke_visual_check
```

Interpretation:

```text
The frame-clipped estimator no longer treats off-frame side projections as
visible. A new formal visual-distance estimator dataset must be collected before
using estimator fields for training or VLA-interface tests.
```

The patched formal visual-distance estimator dataset has been collected and
audited:

```text
dataset: simulation_output/formal_tip_visual_distance_estimator_dataset_v2_frameclip
episodes/env_success: 40/40
rejected: 0
samples: 4328
missing images: 0
formal_valid: true
provenance audit: real_direct_plus_visual_contact passes
visual check: docs/_visual_distance_estimator_dataset_v2_frameclip_visual_check
side in-frame samples: 608/4328
top in-frame samples: 4328/4328
estimated_contact_flag: 0/4328
estimated_image_distance_px p50/p95: ~0.51 / 3.02 px
piper_step_command feed/hold: 2283/2045
```

Interpretation:

```text
The frame-clipping bug is fixed: no off-frame sample is incorrectly flagged.
However, the current side camera covers the estimator point in only about 14%
of samples, so the strict all_visible binary contact flag is too conservative
and becomes always zero. Treat estimated_image_distance_px plus confidence as
the useful structured visual signal for now; do not rely on the binary flag
until camera coverage or the contact rule is redesigned and visually accepted.
```

The first black-tail/red-head sidecam-v1 formal dataset should not be used for
estimator training:

```text
dataset: simulation_output/formal_tip_line_black_red_head_sidecam_v1_dataset_v2
episodes/env_success: 40/40
samples: 4495
image refs: complete
expert/control trajectory: useful diagnostic
estimator fields: invalid for formal training
visual audit issue: top-camera wall markers can be visibly wrong because the
old estimator used projected 3D vessel cross-sections instead of image-derived
visible edges
```

The replacement estimator smoke is:

```text
smoke: simulation_output/_smoke_image_estimator_wall_fix_v3
visual check: docs/_smoke_image_estimator_wall_fix_v3_visual_check
samples: 24
side estimator pixels: 24/24
top estimator pixels: 0/24
contact flags: 0/24
visual review: accepted by user
```

Interpretation:

```text
The fixed rule estimator is now visually honest but mostly side-view-only under
the current top-camera occlusion. For the next formal dataset, use the continuous
estimated_image_distance_px plus contact_estimator_confidence as a candidate
side-view visual signal, and treat missing top-view estimates as low confidence
rather than a failure.
```

The accepted wall-fix formal visual-distance estimator dataset is:

```text
dataset: simulation_output/formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4495
missing image refs: 0
formal_valid: true
provenance audit: real_direct_plus_visual_contact passes
visual check: docs/_formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1_visual_check
visual review: accepted by user
estimator method: rendered_image_red_tip_to_local_edge_distance
side estimator visible: 3915/4495
top estimator visible: 0/4495
estimated_contact_flag: 0/4495
estimated_image_distance_px p50/p95: ~6.54 / 7.79 px
contact_p95/max: 0.0 / ~0.00053
tip_to_magnetic episode-p95 median: ~7.68 mm
piper_step_command feed/hold: 2355/2140
```

Interpretation:

```text
This is the current accepted formal visual-distance estimator candidate. It
fixes the old wrong-wall-marker issue by measuring the rendered camera image
rather than projecting 3D vessel cross-sections. Under the current camera and
Elite occlusion, it is effectively a side-view distance signal: top-view
estimates are absent because the red guidewire head is occluded. Treat
estimated_image_distance_px plus contact_estimator_confidence as the useful
policy-facing candidate signal. Do not treat the all-zero binary contact flag
as a strong contact/no-contact supervision signal yet.
```

Paper-aligned estimated 3D tip support has been implemented in the formal
collector:

```text
flag: --estimated-tip-3d
source: synthetic_rgbd_tip_estimator
method: rendered red-tip visibility + declared synthetic calibration/noise
        wrapper
fields: estimated_tip_pos_3d, estimated_tip_heading_3d,
        tip_estimator_visible, tip_estimator_confidence,
        tip_estimator_latency_steps
frame: world, treated as the current registered simulator frame
```

Smoke validation:

```text
dataset: simulation_output/_smoke_estimated_tip_3d_v1
task: left
env_success: true
samples: 21
3D tip visible/present: 17/21
source camera: side=17, missing=4
estimated-tip 3D error vs diagnostic truth: median/p95/max
  ~1.90 / 4.34 / 4.61 mm
provenance audit: real_direct_plus_estimated_tip_contact passes
audit: docs/_smoke_estimated_tip_3d_v1_audit
```

Interpretation:

```text
This is the first implementation of the thesis-style estimated 3D tip route.
The first version is a conservative synthetic calibration wrapper: image-space
red-head visibility gates whether the estimator returns a noisy 3D tip/heading
field in the registered simulator frame. It is still a synthetic estimator, not
a real sensor, so future formal datasets should inspect visibility, confidence,
and visual quality before using these fields for training.
```

The first full estimated-3D-tip formal dataset has been collected and audited:

```text
dataset: simulation_output/formal_tip_line_estimated_tip_3d_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4495
image refs: 8990/8990 present
provenance audit: real_direct_plus_estimated_tip_contact passes
audit: docs/_formal_tip_line_estimated_tip_3d_dataset_v1_audit
visual check: docs/_formal_tip_line_estimated_tip_3d_dataset_v1_visual_check
3D tip visible/present: 3915/4495 = ~87.1%
source camera: side=3915, top=0, missing=580
estimated-tip 3D error vs diagnostic truth: median/p95/max
  ~2.34 / 4.22 / 6.84 mm
heading dot vs diagnostic truth: median/p05/min
  ~0.999 / 0.996 / 0.989
estimated_image_distance_px p50/p95: ~6.54 / 7.79 px
estimated_contact_flag: 0/4495
piper_step_command feed/hold: 2355/2140
```

Interpretation:

```text
This dataset is valid as the current formal estimated-tip interface candidate.
The visual spot check did not show the previous wrong-wall-marker failure: side
view tip/wall overlays are local to the red head, while top-view estimator
fields remain missing because the red head is occluded. Treat this as a
side-view-gated synthetic 3D tip estimate, not a completed real RGB-D
deprojection model. Before model training, decide whether the current 87% side
visibility is sufficient or whether top-camera/occlusion coverage should be
improved first.
```

The manually tuned top-oblique estimated-tip dataset supersedes the previous
side-gated camera setting:

```text
dataset: simulation_output/formal_tip_line_estimated_tip_3d_top_manual_dataset_v1
camera_config: simulation/camera_configs/mujoco_camera_top_manual_v1.json
episodes/env_success: 40/40
rejected: 0
samples: 4495
image refs: 8990/8990 present
provenance audit: real_direct_plus_estimated_tip_contact passes
audit: docs/_formal_tip_line_estimated_tip_3d_top_manual_dataset_v1_audit
visual check: docs/_formal_tip_line_estimated_tip_3d_top_manual_dataset_v1_visual_check
top estimator visible: 4495/4495 = 100%
side estimator visible: 3915/4495 = ~87.1%
estimated 3D tip present: 4495/4495 = 100%
estimated_tip_pixel_source: top=4495
estimated-tip 3D error vs diagnostic truth: median/p95/max
  ~2.33 / 4.22 / 6.84 mm
heading dot vs diagnostic truth: median/p05/min
  ~0.999 / 0.996 / 0.989
estimated_image_distance_px p50/p95/max: ~6.47 / 7.51 / 19.59 px
estimated_contact_flag: 0/4495
piper_step_command feed/hold: 2355/2140
```

Interpretation:

```text
This is now the preferred formal estimated-tip dataset. Manual top camera
tuning fixed the top-view occlusion problem without changing the accepted side
camera. Visual spot checks show local top tip/wall overlays rather than the old
wrong-wall-marker failure. The binary contact flag is still all zero, so treat
estimated_tip_pos_3d, estimated_tip_heading_3d, estimated_image_distance_px, and
confidence fields as the useful estimator interface candidates.
```

## Compressed Experiment History

Older smoothing/rate-limit experiments established the following and should not
be repeated as the default path:

- Naive higher Piper negative weighting amplified Elite target jitter.
- Naive Elite smoothness loss improved one branch while worsening another.
- Rollout-time velocity/acceleration/jerk limits can make logged Elite motion
  closer to real bandwidth, but video-level twitching and policy target jitter
  remained.
- Naive Elite delta targets improved some open-loop smoothness metrics but
  accumulated large closed-loop magnet/tip drift.
- Tip-centric guidewire abstraction closed the data/training/rollout loop, but
  the small BC model still produced sustained contact or large `tip_to_magnetic`
  drift. Treat this path as diagnostic unless a matching real tip-estimation
  and low-level Elite controller is defined.

Current interpretation:

```text
Expert-data quality is not the immediate bottleneck.
The bottleneck is real-aligned observation/action representation and the
small BC model's ability to learn the Elite/Piper control mapping.
```

Branchs-native real-data baseline:

```text
manifest: simulation_output/branchs_training_manifest_full.json
model: simulation_output/baseline_branchs_native_seniorlike_tcpdelta_pipercls
diagnostic: simulation_output/baseline_branchs_native_seniorlike_tcpdelta_pipercls_open_loop_diag
samples: 4015
observation_schema: senior_piper_real_like
action: Elite TCP delta + Piper step classification
```

Open-loop result:

```text
Piper mismatch overall: 5.75%
left/right Piper mismatch: 6.18% / 5.15%
Elite TCP-delta linf error median/p95: ~0.50 / 2.08
```

Interpretation:

```text
The current model/training pipeline can learn the senior real-data Piper labels
when trained directly on branchs. Therefore the earlier all-feed result from
running the sim-trained registered-geometry checkpoint on branchs is best
understood as a sim-to-real/input-schema mismatch, not as proof that the model
class cannot learn real data.
```

Single-camera senior-like sim alignment experiment:

```text
sim manifest:
  simulation_output/formal_tip_line_visual_route_v1_singlecam_seniorlike_side_full.json
source:
  simulation_output/formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1
model:
  simulation_output/baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls
sim open-loop diagnostic:
  simulation_output/baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls_open_loop_diag
branchs open-loop diagnostic:
  simulation_output/baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls_on_branchs_open_loop_diag
branchs shadow:
  simulation_output/branchs_shadow_predictions_singlecam_side_seniorlike_full.jsonl
```

Result:

```text
sim same-schema open-loop:
  Piper mismatch: 21.17%
  Elite TCP-delta linf error median/p95: ~0.44 / 1.69

branchs open-loop with the same sim-trained checkpoint:
  Piper mismatch: 45.73%, all hold frames predicted as feed
  Elite TCP-delta linf error median/p95: ~5.32 / 9.71

branchs shadow with the same checkpoint:
  records: 4039
  predicted Piper: feed=4039, hold=0
  mismatch: 1849/4039 = 45.78%
```

Interpretation:

```text
Duplicating one simulated side camera as side/top is not enough to close the
sim-to-real gap. It improves the input-schema comparison but the sim-trained
model still collapses to all-feed on branchs, while the branchs-native model
learns the same senior-like real labels well. Treat the remaining gap as
visual/domain distribution plus real label/control timing mismatch, not merely
as a single-vs-dual-camera schema issue.
```

Sim/real visual-domain audit:

```text
tool:
  tools/audit_sim_real_visual_domain.py
report:
  docs/_sim_real_visual_domain_audit_singlecam_v1/report.md
contact sheet:
  docs/_sim_real_visual_domain_audit_singlecam_v1/contact_sheet.png
sample pairs:
  24, left/right balanced, no missing images
```

Key measured differences:

```text
luminance_mean: sim 184.38 vs real 120.49
saturation_mean: sim 0.278 vs real 0.388
bright_ratio: sim 0.324 vs real 0.002
edge_density: sim 0.034 vs real 0.026
```

Visual interpretation:

```text
The current sim images are bright, clean, pale-rendered scenes with translucent
vessel geometry and synthetic tabletop/background. The real branchs images are
darker physical camera frames with green lab background, real robot/fixture
appearance, stronger crop/perspective, and real occlusion. The domain gap is
visible before policy rollout, so the next work should prioritize rendering/
camera/domain alignment or new aligned real data rather than more small-BC
rollout tuning.
```

Branchs-like MuJoCo render preset:

```text
preset: --render-domain-preset branchs_like_v1
smoke:
  simulation_output/_smoke_branchs_like_render_preset_v1
audit:
  docs/_sim_real_visual_domain_audit_branchs_like_v1
sample pairs: 22 side-camera pairs, left/right balanced
left/right env_success: 2/2
contact_p95/max: 0.0 / 0.0
```

Key measured differences after the preset:

```text
luminance_mean: sim 118.60 vs real 120.76
saturation_mean: sim 0.407 vs real 0.392
bright_ratio: sim 0.000 vs real 0.003
edge_density: sim 0.027 vs real 0.026
```

Interpretation:

```text
This is the current accepted first MuJoCo branchs-like renderer preset. It
substantially fixes the obvious brightness/saturation/overexposed-background
gap from the first audit while preserving the accepted tip-line guidewire route
and zero contact in the smoke. User visual review accepted it as usable for the
next step, while noting that the remaining visible gap is still large and is
mainly concentrated in real glass-vessel reflections and real-camera blur. The
real frames also still contain physical fixtures, crop/perspective differences,
and real occlusion that MuJoCo does not yet reproduce. Continue with a small
branchs-like dataset before training, not with BC rollout tuning.
```

Accepted branchs-like small dataset:

```text
dataset:
  simulation_output/formal_tip_line_branchs_like_v1_small
render preset:
  branchs_like_v1
episodes:
  accepted 8/8, rejected 0, left/right 4/4
samples:
  936, side/top image files complete
contact:
  contact_p95 max 0.0, max_contact_strength max ~0.00053
wall clearance:
  min_tip_distance_to_wall min/median ~= 1.39 / 1.60 mm
tip/magnet:
  tip_to_magnetic_p95 median ~= 7.68 mm
piper labels:
  feed=486, hold=450
estimator fields:
  estimated-tip 3D and registered-geometry fields present for 936/936 samples
provenance audit:
  docs/_formal_tip_line_branchs_like_v1_small_provenance_audit
visual-domain audit:
  docs/_sim_real_visual_domain_audit_branchs_like_v1_small
```

Small-dataset visual-domain metrics:

```text
luminance_mean: sim 118.53 vs real 120.49
saturation_mean: sim 0.408 vs real 0.388
bright_ratio: sim 0.000 vs real 0.002
edge_density: sim 0.027 vs real 0.026
```

Interpretation:

```text
The accepted render preset remains stable when scaled from the 2-episode smoke
to an 8-episode small dataset. Dataset quality is good enough to scale before
training. Remaining visual gap is still mainly glass reflection and real-camera
blur, not a reason to resume BC rollout tuning yet.
```

Accepted branchs-like full dataset:

```text
dataset:
  simulation_output/formal_tip_line_branchs_like_v1_full
render preset:
  branchs_like_v1
episodes:
  accepted 40/40, rejected 0, left/right 20/20
samples:
  4495, side/top image files complete
contact:
  contact_p95 max 0.0, max_contact_strength max ~0.00053
wall clearance:
  min_tip_distance_to_wall min/median ~= 1.39 / 1.81 mm
tip/magnet:
  tip_to_magnetic_p95 median ~= 7.68 mm
piper labels:
  feed=2355, hold=2140
estimated tip:
  visible 4495/4495, pixel source top=4495
estimator fields:
  estimated-tip 3D and registered-geometry fields present for 4495/4495 samples
provenance audit:
  docs/_formal_tip_line_branchs_like_v1_full_provenance_audit
visual-domain audit:
  docs/_sim_real_visual_domain_audit_branchs_like_v1_full
```

Full-dataset visual-domain metrics:

```text
luminance_mean: sim 118.52 vs real 120.49
saturation_mean: sim 0.408 vs real 0.388
bright_ratio: sim 0.000 vs real 0.002
edge_density: sim 0.027 vs real 0.026
```

Interpretation:

```text
This is the current accepted branchs-like full formal dataset and supersedes
the small version for training. It preserves the accepted visual preset, full
estimated-tip/registered-geometry coverage, explicit Piper feed/hold labels,
and clean contact/wall metrics. Remaining sim-to-real image gap is still mainly
glass-vessel reflection, real-camera blur, fixtures, crop/perspective, and
occlusion.
```

Branchs-like full registered-geometry BC baseline:

```text
dataset:
  simulation_output/formal_tip_line_branchs_like_v1_full
model:
  simulation_output/baseline_formal_tip_line_branchs_like_v1_reggeom_tcpdelta_pipercls
diagnostic:
  simulation_output/baseline_formal_tip_line_branchs_like_v1_reggeom_tcpdelta_pipercls_open_loop_diag
observation_schema:
  real_direct_plus_estimated_tip_registered_geometry
elite_action_representation:
  tcp_delta
piper_head:
  step_classification
```

Open-loop result:

```text
samples: 4495
piper_sign_mismatch_fraction: 21.20%
left/right Piper mismatch: 23.87% / 18.43%
piper confusion:
  hold->hold 1833, hold->feed 307
  feed->hold 646, feed->feed 1709
elite_linf_error median/p95/max: ~0.515 / 1.707 / 4.732
predicted/expert Elite target-step p95: ~3.715 / 2.995
predicted/expert Elite delta-from-current p95: ~2.575 / 2.811
```

Interpretation:

```text
The branchs-like render preset did not make the small reggeom BC model solve
Piper timing. Piper errors remain concentrated around the scheduled feed/hold
phase, especially step_mod=0. Elite TCP-delta error is not catastrophic, but
predicted target-step p95 is higher than expert, so a closed-loop rollout may
still show jump/contact risk. Do not treat this as a completed sim-to-real
policy. The next transfer-relevant comparison should use a branchs-compatible
single-camera/senior-like schema on the new branchs-like render data, otherwise
the model still consumes estimator fields that `branchs` does not have.
```

Branchs-like full single-camera senior-like transfer check:

```text
sim manifest:
  simulation_output/formal_tip_line_branchs_like_v1_singlecam_seniorlike_side_full.json
model:
  simulation_output/baseline_formal_tip_line_branchs_like_v1_singlecam_side_seniorlike_tcpdelta_pipercls
sim open-loop diagnostic:
  simulation_output/baseline_formal_tip_line_branchs_like_v1_singlecam_side_seniorlike_tcpdelta_pipercls_open_loop_diag
branchs open-loop diagnostic:
  simulation_output/baseline_formal_tip_line_branchs_like_v1_singlecam_side_seniorlike_tcpdelta_pipercls_on_branchs_open_loop_diag
observation_schema:
  senior_piper_real_like
elite_action_representation:
  tcp_delta
piper_head:
  step_classification
```

Open-loop result:

```text
same-schema sim:
  samples: 4495
  piper_sign_mismatch_fraction: 20.24%
  piper confusion:
    hold->hold 1632, hold->feed 508
    feed->hold 402, feed->feed 1953
  elite_linf_error median/p95/max: ~0.496 / 1.557 / 4.055
  predicted/expert Elite target-step p95: ~3.350 / 2.995

branchs transfer:
  samples: 4015
  piper_sign_mismatch_fraction: 45.73%
  piper confusion:
    hold->feed 1836
    feed->feed 2179
  elite_linf_error median/p95/max: ~3.614 / 8.102 / 10.347
  predicted/expert Elite delta-from-current p95: ~3.644 / 4.826
```

Interpretation:

```text
The branchs-like visual preset improved side-camera image statistics but did
not solve sim-to-branchs transfer. The single-camera branchs-compatible model
still predicts Piper feed for every real branchs frame, so the remaining gap is
not only dual-camera/schema mismatch. Treat this as evidence that MuJoCo
branchs-like rendering is useful but insufficient by itself; the next lever
should be real-observation coverage and real/sim data alignment, especially
glass-vessel reflections, blur, fixture/crop/occlusion, and Piper label/control
timing. Do not keep optimizing this small BC checkpoint as the main route.
```

## Latest Model Diagnostics

Legacy continuous-Piper training on signed-Piper v8 failed because feed/hold
labels were trained as continuous regression; rollout Piper output collapsed to
a mid-valued command instead of discrete `0.0/0.7`.

Piper-classification training fixed the invalid mid-valued output format but
did not produce a good policy. Both full-state and real-direct models stayed
around `~62%` feed/hold accuracy and had poor Elite absolute-joint targets.

TCP-6D senior-like absolute-joint run:

```text
dataset: simulation_output/formal_route_plan_tip_signedpiper_dataset_v9_tcp6d
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v9_tcp6d_seniorlike_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v9_tcp6d_seniorlike_pipercls_open_loop_diag
```

Open-loop result:

```text
piper_sign_mismatch_fraction: 0.371
piper predicted labels: hold=1452, feed=355
elite_linf median/p95: ~0.133 / 0.318
predicted Elite target-step p95: ~0.169
expert Elite target-step p95: ~0.022
predicted Elite delta-from-current p95: ~0.315
expert Elite delta-from-current p95: ~0.0084
```

Conclusion:

```text
Do not roll out this checkpoint.
Do not repeat senior_piper_real_like + absolute Elite joint targets as the next
training path.
```

This result does not invalidate the v9 expert data. It shows that
`image + Elite TCP 6D pose + piper_step + task id` is a poor match for an
absolute six-joint Elite target. Senior's real Elite-style control appears to
move in TCP pose space and then use IK, so the next real-aligned experiment
should consider TCP-delta/IK or an explicit real controller state.

Latest TCP-delta senior-like runs:

```text
dataset: simulation_output/formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls_open_loop_diag
```

Open-loop result:

```text
observation_schema: senior_piper_real_like
elite_action_representation: tcp_delta
piper_sign_mismatch_fraction: 0.397
elite_linf median/p95: ~0.947 / 1.676 in TCP pose-delta metric space
expert/predicted Elite target-step p95: ~7.20 / 7.01
```

Adding explicit scheduled-controller phase produced only a small improvement:

```text
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls_open_loop_diag
observation_schema: senior_piper_real_like_with_phase
state_dim: 11
piper_sign_mismatch_fraction: 0.371
elite_linf median/p95: ~0.910 / 1.577
expert/predicted Elite target-step p95: ~7.20 / 7.62
```

Conclusion:

```text
Do not roll out the v10 senior-like or phase checkpoint as the next default.
Do not keep optimizing the small BC model to learn scheduled Piper feed/hold.
```

Latest control-state TCP-delta run:

```text
dataset: simulation_output/formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v12_controlstate_tcpdelta_seniorlike_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v12_controlstate_tcpdelta_seniorlike_pipercls_open_loop_diag
```

Open-loop result:

```text
observation_schema: senior_piper_real_like
elite_action_representation: tcp_delta
piper_head: step_classification
samples: 1796
piper_sign_mismatch_fraction: 0.361
elite_linf median/p95: ~0.930 / 1.567 in TCP pose-delta metric space
expert/predicted Elite target-step p95: ~7.11 / 7.63
expert/predicted Elite delta-from-current p95: ~2.82 / 3.12
```

Conclusion:

```text
Do not roll out this checkpoint as the next default.
The v12 data/control-state logging path is valid, but the small BC model still
does not learn the real-aligned Piper/Elite mapping well enough to justify more
rollout tuning.
```

The TCP-delta action representation is the right Elite semantic direction, but
the Piper result shows that the scheduled `feed/hold` label is better treated as
an explicit real controller/scheduler decision than as something a tiny BC model
should infer from images plus state. The next step should move Piper command
semantics into the expert/controller design, or into a real-interface state
machine, before further BC rollout testing.

Estimated-tip interface rollout and coupling-guard diagnostic:

```text
dataset: simulation_output/formal_tip_line_estimated_tip_3d_top_manual_dataset_v1
model: simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls
baseline rollout: simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_r1
estimated-tip coupling guard rollout:
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_esttip_guard_r1
```

The baseline rollout reached both targets but showed obvious contact. The
Piper-step cooldown variants showed that throttling feed alone trades contact
for slow progress and tip/magnet drift. A later estimated-tip coupling guard
used only estimator-like `estimated_tip_pos_3d` plus current Elite TCP pose to
hold Piper and bias Elite back near the estimated tip when they drifted apart.
It also reached both targets:

```text
left:  success=True, steps=180, max_contact=0.570, contact_p95=0.540,
       tip_to_magnetic median/p95/max ~= 17.8 / 34.8 / 35.4 mm,
       guard active 5/180 actions
right: success=True, steps=152, max_contact=0.573, contact_p95=0.567,
       tip_to_magnetic median/p95/max ~= 8.9 / 21.6 / 23.0 mm,
       guard active 0/152 actions
```

Conclusion:

```text
The current high-contact failure is not primarily an estimated-tip/Elite
coupling failure. During the high-contact periods the magnet is already close
to the guidewire head, especially on the right branch. The all-zero binary
contact flag and nearly constant estimated_image_distance_px also do not yet
provide a reliable contact-safety signal. Do not keep tuning Piper cooldown or
estimated-tip coupling guards as the main fix.
```

Wall-contact direction diagnosis:

```text
tool: tools/diagnose_rollout_wall_contact.py
comparison:
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_esttip_guard_r1
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_r1
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_piperctrl_cd3_r1
output: docs/_wall_contact_diagnostics_esttip_guard_compare
```

The diagnostic projects the magnetic vector from tip to magnet onto the
contact normal. Positive `magnet_wall_pull` means the magnetic tool is pulling
the tip toward the wall/contact side. In the high-contact samples,
`magnet_wall_pull` is consistently positive and correlates with contact, while
`estimated_image_distance_px` has weak or negative correlation with exact
MuJoCo contact.

```text
guard right: corr(contact, magnet_wall_pull) ~= 0.775,
             corr(contact, estimated_image_distance_px) ~= -0.192
base right:  corr(contact, magnet_wall_pull) ~= 0.787,
             corr(contact, estimated_image_distance_px) ~= -0.248
cd3 right:   corr(contact, magnet_wall_pull) ~= 0.652,
             corr(contact, estimated_image_distance_px) ~= -0.532
```

Interpretation:

```text
The next mainline issue is not only Piper timing or tip/magnet distance. It is
that the learned Elite TCP direction can keep the magnet close to the tip while
placing it on the wall side, and the current image-distance estimator does not
reliably report this as danger. Next work should improve wall-margin/contact
estimation and/or constrain Elite guidance direction using real-observable
estimated tip + registered vessel/wall geometry.
```

Registered-geometry estimator integration:

```text
fields:
  estimated_wall_margin
  estimated_wall_margin_fraction
  estimated_route_progress / estimated_route_index
  route_estimator_confidence
  estimated_magnet_wall_pull
  estimated_wall_side_risk
  estimated_wall_normal_3d
  estimated_route_tangent_3d
schema:
  real_direct_plus_estimated_tip_registered_geometry
smoke rollout:
  simulation_output/_smoke_registered_geometry_rollout_v1
provenance audit:
  docs/_smoke_registered_geometry_estimator_v1_registered_schema_audit
minimal train smoke:
  simulation_output/_smoke_registered_geometry_schema_train_v1
full dataset:
  simulation_output/formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1
trained checkpoint:
  simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls
baseline rollout:
  simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_r1
accepted wall-safety rollout:
  simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_video_r1
```

Interpretation:

```text
The registered-geometry estimator chain is now technically closed for short
validation: collection can emit the fields, rollout can auto-enable them for
the new schema, the provenance audit treats them as estimator fields rather
than privileged MuJoCo truth, and the training code can build a 32D state vector
and complete a 1-epoch smoke run. The full registered-geometry dataset,
training run, no-video rollout, and side-video rollout have now also completed.
The registered-geometry fields are useful as a controller safety signal, but
only in a conservative Elite-only form.
```

Registered-geometry wall-safety controller result:

```text
baseline rollout:
  left  success=True, steps=228, max_contact=0.0592, contact_p95~=0.0546
  right success=True, steps=214, max_contact~=0.0006, contact_p95=0.0

bad wallguard:
  output: simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_r1
  result: left failed, right succeeded slowly
  cause: Piper was held too often and active away-wall correction pulled the
         magnetic tool away from the guidewire tip
  diagnostic: left guard active 550/700 actions; tip_mag median ~=98.5 mm

accepted candidate:
  output: simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_video_r1
  controller: no Piper hold, no active away-wall correction, only remove the
              Elite TCP-delta component toward the estimated wall normal
  result:
    left  success=True, steps=242, max_contact=0.0571, contact_p95~=0.0483
    right success=True, steps=210, max_contact~=0.0005, contact_p95=0.0
  user visual review: occasional wall contact remains but is acceptable
```

Conclusion:

```text
Do not use registered-geometry wall safety as a strong override. In particular,
do not hold Piper or add repeated away-wall TCP corrections by default. The
current acceptable controller candidate is Elite-only: keep Piper policy output
unchanged, preserve tip-magnet coupling, and only remove the predicted Elite
TCP-delta component that points toward the estimated wall side.
```

## Implemented Control-Semantics Update

The first real-aligned Elite action path has been implemented:

```text
--elite-action-representation tcp_delta
```

Collection now writes:

```text
action.elite_tcp_pose_6d
action.elite_tcp_delta_6d
```

Training can learn TCP 6D deltas from current Elite TCP pose. Rollout converts
the predicted TCP delta back into an Elite joint command through MuJoCo IK and
still logs `elite_joints` for diagnostics. This matches the inherited
senior-style control pattern more closely than predicting absolute six-joint
targets from TCP-pose observations.

Current implementation detail:

```text
Elite TCP delta labels are translation-first and keep current orientation,
matching intervention/bc_elirobot/inference.py.
```

## Current Next Step

Stop the current BC tuning loop. The MuJoCo branchs-like visual/domain preset
is accepted as a useful renderer improvement, but the latest single-camera
senior-like transfer check still collapses to all-feed on `branchs`. The active
next mainline task is therefore not another BC rollout tweak; it is improving
real-observation coverage and real/sim data alignment enough to make a
meaningful sim-to-real shadow test.

1. Keep `formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta`
   as the current clean formal control-state/TCP-delta reference.
2. Treat `--elite-action-representation tcp_delta` as the preferred Elite
   learning target when using senior-like TCP-pose observations.
3. Do not roll out v10/v12 senior-like checkpoints unless a later diagnostic
   materially improves open-loop Piper and Elite behavior.
4. Use the hybrid contract: policy predicts Piper `hold/feed/retract` intent;
   the controller owns real step timing, cooldown, bounds, and executed-command
   logging.
5. Keep deterministic Piper scheduler phase out of the learned target unless it
   is represented as explicit real controller state.
6. Keep BC as an open-loop/closed-loop diagnostic. Do not treat BC rollout
   success as the project objective.
7. Do not use `formal_tip_visual_distance_estimator_dataset_v1` for training.
   Its visual audit exposed an off-frame projection bug in the rule-based
   estimator.
8. Do not use `formal_tip_line_black_red_head_sidecam_v1_dataset_v2` estimator
   fields for training. Its visual audit found visibly wrong wall points from
   the older 3D projection estimator. Use
   `formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1` as the current
   accepted replacement candidate.
9. Use `docs/estimator-tactile-interface-spec.md` to define any future
   guidewire-tip/contact fields before they enter a formal policy input. For
   contact, model the real path as a visual image-distance/contact estimator
   unless new hardware evidence says otherwise.
10. Use `docs/visual-tip-contact-estimator-plan.md` as the current perception
   route: first keep the rule-based visual distance estimator as an auditable
   baseline/fallback, then allow a learned visual estimator to replace or
   enhance it without changing the policy-facing output contract.
11. Use `tools/audit_observation_provenance.py` before formal training. Current
    sanity check: `senior_piper_real_like` passes on v12, while `full_sim_state`
    fails because it consumes exact tip/contact/wall/route-frame fields.
12. Use `real_direct_plus_estimated_tip_registered_geometry` as the current
    estimator-interface candidate. The full dataset, training run, rollout, and
    visual review support the conservative Elite-only wall-safety controller.
    Avoid the stronger Piper-hold / active-away correction variant because it
    breaks tip-magnet coupling and can make contact worse.
13. Use `--render-domain-preset branchs_like_v1` as the current accepted MuJoCo
    branchs-like render diagnostic. The contact sheet in
    `docs/_sim_real_visual_domain_audit_branchs_like_v1` has been visually
    accepted, and the full dataset remains clean. However, the single-camera
    senior-like model trained on this dataset still predicts feed for all
    `branchs` frames, so rendering statistics alone are insufficient. Keep
    Isaac Sim or a remote-workstation renderer-only spike as backup option C,
    not the default route.
14. Before another sim-trained policy comparison, prefer collecting or building
    real-aligned inputs that match the target VLA interface more closely:
    synchronized real side/top images if available, real Elite TCP pose, Piper
    controller state, target/task metadata, and estimator/contact fields with
    explicit provenance. If only the current `branchs` data is available, treat
    it as a shadow diagnostic with known missing inputs rather than a decisive
    sim-to-real benchmark.
15. Use `docs/real-data-collection-checklist.md` as the next real pilot capture
    target. The minimum pilot should produce a `records.jsonl` compatible with
    `tools/real_shadow_policy_adapter.py`, including image paths, Elite TCP 6D
    pose, Piper step/state, task metadata, reference Piper labels, optional
    estimator fields with explicit null/low-confidence values, and timestamp
    synchronization diagnostics. Use
    `data/collect/collect_real_shadow_pilot.py` as the current collection
    entrypoint; it defaults to log-only Piper and requires
    `--enable-piper-control` before calling Piper hardware.

Useful current docs:

```text
docs/senior-thesis-interface-audit.md
docs/handoff.md
docs/data-and-action-schema.md
docs/observation-schema-audit.md
docs/real-observation-schema-alignment.md
docs/real-observable-interface-audit.md
docs/estimator-tactile-interface-spec.md
docs/visual-tip-contact-estimator-plan.md
docs/simulation-expert-validity-audit.md
docs/sim-vs-real-alignment.md
docs/vla-target-interface.md
docs/control-layer-contract.md
```

## Paper-Aligned Interface Evidence

A targeted read of the local ignored senior thesis PDF is now recorded in:

```text
docs/senior-thesis-interface-audit.md
```

The thesis confirms that the inherited autonomous system used:

```text
policy input:
  operative image
  magnetic guidance arm 6D pose
  accumulated Piper feed count
  target branch / target position

policy output:
  magnetic guidance arm 6D pose delta
  Piper binary hold/feed decision
```

This directly supports the current project interface:

```text
Elite: TCP delta -> IK -> joints
Piper: senior-compatible hold/feed intent
```

The thesis also shows that the real perception stack is not only a binary
contact flag. It includes red guidewire-head localization, vessel/wall
segmentation, image-distance collision detection, depth-camera coordinate
recovery, and coordinate transforms into robot or virtual-environment frames.

Updated implication:

```text
The next mainline estimator step should extend the accepted wallfix
image-distance candidate toward estimated 3D tip / registered route fields with
provenance and confidence. Do not resume BC rollout tuning as the main lever.
```
