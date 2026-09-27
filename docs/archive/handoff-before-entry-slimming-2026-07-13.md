# Handoff

Last updated: 2026-07-07

## Current Status

The project is an inherited real-camera / real-robot guidewire intervention
repository. The current track is a MuJoCo simulation pipeline used to generate
synthetic data because large real data collection is difficult.

Highest priority:

```text
simulation realism and real-system correspondence > BC rollout success
```

BC models are diagnostic loop-closure checks only. Do not optimize a small BC
rollout at the cost of simulator/real mismatch.

Current action schema:

```text
piper_feed_elite_joint
```

Roles:

- Piper stays near the vessel entrance and feeds/holds/retracts the guidewire.
- Elite carries the magnetic tool. Execution uses joints, but the real-aligned
  learned action path is now TCP delta followed by IK.
- The real magnet is attached to the Elite end tool.

Target VLA-facing contract:

```text
observation + task instruction -> Elite TCP delta + Piper discrete command
```

Canonical note:

```text
docs/senior-thesis-interface-audit.md
docs/vla-target-interface.md
docs/control-layer-contract.md
```

The small BC baseline is only a miniature interface/regression test for this
contract. Do not let it define a different temporary action space just to make a
simulation rollout succeed.

Paper-backed real interface evidence:

```text
senior thesis chapter 4 policy input:
  operative image + Elite/magnetic-arm 6D pose + Piper feed count + target

senior thesis chapter 4 policy output:
  Elite/magnetic-arm 6D pose delta + Piper binary hold/feed decision
```

This confirms that the current TCP-delta/IK Elite target and senior-compatible
Piper hold/feed subset are the correct real-system-aligned starting point.

Senior real data rule:

```text
branchs/* pose data = Elite/magnetic-arm TCP trajectory, not guidewire path.
```

Current guidewire realism direction:

```text
tip-centric / hard-elastic guidewire assumption + continuous line-shaped visual guidewire
```

Do not return to full flexible-body wire physics as the default next step. The
current hypothesis is that the real guidewire is hard and elastic enough that
keeping the magnetically guided head off the wall is the first-order task
constraint, while the rendered training image should look like a continuous
wire rather than a chain of small balls/segments.

Current formal mainline collection entrypoint:

```text
simulation.collect_formal_tip_line_guidance
```

Default formal start range:

```text
--start-fraction-min 0.42
--start-fraction-max 0.54
```

This supersedes the older `0.58-0.70` range, which placed the red guidewire
head too close to the bifurcation/branch area at the beginning of an episode.

The older collection entrypoints are legacy/diagnostic only:

```text
simulation.collect_tip_guided_wire
simulation.collect_mujoco_physical_guidance
```

Do not use legacy scripts for new mainline data unless the task is explicitly
to reproduce or compare an older route.

Latest visual smoke / accepted route baseline:

```text
simulation_output/_smoke_tip_line_wire_formal_hidden
left env_success: true
steps: 208
samples: 5
wire visual: line, radius 0.0020, segments 64, tail decay 18
formal visual leakage check: no colored debug marker/path-tube leakage in side/top frames

simulation_output/_smoke_formal_tip_line_mainline_entry
left env_success: true
steps: 208
samples: 5
min_tip_wall: ~1.779 mm
max_contact_strength: 0.0

simulation_output/formal_tip_line_visual_route_v1_small_fix1
camera_config: simulation/camera_configs/mujoco_camera_top_manual_v1.json
route: Piper outlet -> vessel entry -> fixed S-bend prefix -> dynamic in-vessel route -> red tip
accepted_episodes: 20
attempts: 20
samples: 567
left/right success: 10/10 and 10/10
max_contact_strength: 0.0
contact_p95 max: 0.0
min_tip_distance_to_wall: ~1.386 mm
max_tip_to_magnetic: ~8.429 mm
visual review: accepted by user
```

After user visual review against real side/top images, the line guidewire was
still visibly thicker than the real thin guidewire at `0.0014`. The current
mainline default has been thinned to `--wire-visual-radius 0.0008`.
The real guidewire appearance is black along the tail with a wider red head, so
the formal collector now renders a black tail plus an 8-segment red tip at 2.2x
the base visual radius, plus a small red tip marker attached to the simulated
guidewire tip for stable camera visibility. A short offset probe showed that a
simple z-offset does not materially improve red-head visibility and can make
the wire look like it is floating, so the default visual offset remains zero.
The accepted route uses a fixed user-tuned entry/S-bend prefix followed by
route-progress-filtered dynamic points. This fixed the right-branch failure
where the red head appeared in the middle while the black tail extended past it.

Latest camera/estimator status:

```text
config: simulation/camera_configs/mujoco_camera_side_coverage_v1.json
fixed estimator smoke: simulation_output/_smoke_image_estimator_wall_fix_v3
visual check: docs/_smoke_image_estimator_wall_fix_v3_visual_check
method: rendered_image_red_tip_to_local_edge_distance
side estimator pixels: 24/24
top estimator pixels: 0/24
visual review: accepted by user
```

Use this camera config and fixed rendered-image estimator for the next formal
visual-distance estimator collection. The older projection estimator is not
acceptable: it could draw top-camera wall markers at visibly wrong locations
because it projected a 3D cross-section instead of measuring the rendered image.
Under the current camera/Elite occlusion, the rule estimator is mostly side-view
only; top-view missing estimates should be treated as low confidence, not filled
with synthetic wall points.

## Current Reference Artifacts

Real-system alignment/calibration audit:

```text
docs/real-alignment-20260707-calibration-audit.md
source data: collected_data/real_align_20260707_*
audit output: docs/_real_alignment_20260707_audit
```

Current decision:

```text
Use the 2026-07-07 real captures to calibrate MuJoCo, not to train a policy.
Elite-only events provide a usable real event-window step scale of roughly
14-18 mm. Piper-only events should only be interpreted as guidewire forward
progress along the vessel/entry direction; they are not yet precise
feed-distance labels because visible wire motion is weak, partially occluded,
and can be physically limited. Use local tip/wire tracking or manual annotation
before tuning Piper feed scale.
The first MuJoCo Piper primitive probe indicates that one real
`piper.step_forward(pause_time=0.8)` call should be represented as a short
multi-step simulator feed primitive, not as one `env.step`; repeated feed
commands can be visibly sublinear because of friction, slip, elastic storage,
and physical blocking.

Current implementation:

```text
simulation.collect_formal_tip_line_guidance now supports:
  --piper-primitive-steps
  --piper-primitive-feed-value
  --piper-primitive-retract-value
  --piper-command-as-event
  --piper-command-period
  --piper-command-width

With `--piper-primitive-steps > 1`, event semantics are enabled automatically:
`piper_step_command=1` starts one real-style Piper feed primitive, and
`controller_state.piper_executed_feed` records the simulator controller's
multi-step execution. Use this instead of interpreting one real feed call as a
single MuJoCo step.
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

Current formal expert datasets:

```text
simulation_output/formal_route_plan_tip_signedpiper_dataset_v8
simulation_output/formal_route_plan_tip_signedpiper_dataset_v9_tcp6d
simulation_output/formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta
```

Shared quality:

```text
formal_valid: true
accepted/env_success: 40/40
rejected: 0
samples: 1807
image refs: 3614/3614
contact_p95 max: 0.0
tip_to_magnetic p95 median: ~7.6-7.7 mm
piper_step_command feed/hold: 951/856
piper_feed values: 0.7 / 0.0
transition_fraction median: ~0.244
```

The v9 dataset adds true TCP-6D fields for senior-like observation tests:

```text
state.elite_tcp_pose_6d
state.robot_state.elite_tcp_pose_6d
format: xyz_mm + rpy_rad
```

The v12 dataset is the latest clean control-state/TCP-delta reference:

```text
accepted/env_success: 40/40
samples: 1796
missing images/controller_state/elite_tcp_delta: 0
contact_p95/max: 0.0 / 0.0
tip_to_magnetic median/p95/max: ~2.10 / 7.66 / 8.43 mm
piper controller pairs: hold->hold 884, feed->feed 912
elite controller types: tcp_delta 1756, reset 40
```

The first visual-distance estimator dataset is not a clean formal reference:

```text
dataset: simulation_output/formal_tip_visual_distance_estimator_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4328
missing images: 0
formal_valid: true
estimated_contact_flag: 99/4328 = 2.29%
estimated_image_distance_px p50/p95: ~2.61 / 6.04 px
piper_step_command feed/hold: 2283/2045
provenance audit: real_direct_plus_visual_contact passes
audit: docs/_visual_distance_estimator_dataset_v1_audit
visual check: docs/_visual_distance_estimator_dataset_v1_visual_check
```

Visual inspection found a real estimator bug: side-camera projected tip/wall
pixels were often outside the rendered frame, but the estimator still treated
the side view as visible because it only checked depth. All contact flags in
this dataset were affected by side off-frame projection. Do not train on
`formal_tip_visual_distance_estimator_dataset_v1`.

The estimator has been patched to require both positive depth and in-frame pixel
coordinates. Smoke result:

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

Next visual-estimator step: recollect the formal dataset with the frame-clipped
estimator, then rerun the visual audit before accepting it as trainable data.

Patched formal dataset:

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
The off-frame projection bug is fixed. The remaining issue is estimator
coverage/semantics: the side camera sees valid estimator points in only about
14% of samples, so strict all_visible makes the binary contact flag always
zero. Treat continuous image distance and confidence as the candidate structured
visual signal for user review; do not rely on the binary contact flag until the
camera coverage or rule is redesigned.
```

Black-tail/red-head sidecam-v1 formal dataset:

```text
dataset: simulation_output/formal_tip_line_black_red_head_sidecam_v1_dataset_v2
episodes/env_success: 40/40
samples: 4495
image refs: complete
expert/control trajectory: useful diagnostic
estimator fields: invalid for formal training
reason: visual review found the wall marker could fly to an obviously wrong
top-camera location because the estimator used 3D cross-section projection
instead of image-derived visible edges
```

Accepted wallfix replacement:

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
piper_step_command feed/hold: 2355/2140
```

Interpretation:

```text
This is the current accepted formal visual-distance estimator candidate. The
fixed estimator measures the rendered camera image and no longer draws
projected 3D wall markers at visibly wrong locations. It is mostly side-view
only under the current top-camera/Elite occlusion. Use continuous
estimated_image_distance_px plus contact_estimator_confidence as the useful
candidate signal; do not rely on estimated_contact_flag yet because it is
all zero in this dataset.
```

Estimated 3D tip dataset:

```text
dataset: simulation_output/formal_tip_line_estimated_tip_3d_dataset_v1
episodes/env_success: 40/40
rejected: 0
samples: 4495
image refs: 8990/8990 present
provenance audit: real_direct_plus_estimated_tip_contact passes
visual check: docs/_formal_tip_line_estimated_tip_3d_dataset_v1_visual_check
3D tip visible/present: 3915/4495 = ~87.1%
source camera: side=3915, top=0, missing=580
estimated-tip 3D error vs diagnostic truth: median/p95/max
  ~2.34 / 4.22 / 6.84 mm
estimated_contact_flag: 0/4495
piper_step_command feed/hold: 2355/2140
```

Interpretation:

```text
This is the current formal estimated-tip interface candidate. It remains a
side-view-gated synthetic calibrated estimator: it is useful for testing the
policy-facing estimated-tip contract, but it should not be mistaken for a
completed real RGB-D deprojection implementation. Top-view estimator fields are
still missing under current Elite/red-head occlusion.
```

Preferred manually tuned top-camera estimated 3D tip dataset:

```text
dataset: simulation_output/formal_tip_line_estimated_tip_3d_top_manual_dataset_v1
camera_config: simulation/camera_configs/mujoco_camera_top_manual_v1.json
episodes/env_success: 40/40
rejected: 0
samples: 4495
image refs: 8990/8990 present
provenance audit: real_direct_plus_estimated_tip_contact passes
visual check: docs/_formal_tip_line_estimated_tip_3d_top_manual_dataset_v1_visual_check
top estimator visible: 4495/4495 = 100%
side estimator visible: 3915/4495 = ~87.1%
estimated 3D tip present: 4495/4495 = 100%
estimated_tip_pixel_source: top=4495
estimated-tip 3D error vs diagnostic truth: median/p95/max
  ~2.33 / 4.22 / 6.84 mm
estimated_contact_flag: 0/4495
piper_step_command feed/hold: 2355/2140
```

Interpretation:

```text
This supersedes the earlier side-gated estimated-tip dataset as the current
formal estimated-tip interface candidate. The manually tuned top camera keeps
the side camera unchanged but gives complete top red-head visibility. Continue
to treat binary contact as weak because it is all zero.
```

## Current Model Result

Earlier absolute-joint senior-like model:

```text
simulation_output/baseline_formal_route_plan_tip_signedpiper_v9_tcp6d_seniorlike_pipercls
```

Open-loop diagnostic:

```text
simulation_output/baseline_formal_route_plan_tip_signedpiper_v9_tcp6d_seniorlike_pipercls_open_loop_diag
```

Result:

```text
observation_schema: senior_piper_real_like
piper_head: step_classification
state_dim: 9
piper_sign_mismatch_fraction: 0.371
piper predicted labels: hold=1452, feed=355
elite_linf median/p95: ~0.133 / 0.318
predicted Elite target-step p95: ~0.169
expert Elite target-step p95: ~0.022
predicted Elite delta-from-current p95: ~0.315
expert Elite delta-from-current p95: ~0.0084
```

Judgment:

```text
Do not roll out this checkpoint.
Do not retrain this exact senior_piper_real_like + absolute-joint target setup.
```

This is not an expert-data failure. The v9 data is clean. The problem is that
senior-like input gives Elite TCP 6D pose, while the current target asks the
model to output absolute six-joint Elite commands. Senior's real Elite-style
code appears to move in TCP pose space and then run IK, so the next real-aligned
control experiment should consider TCP delta plus IK, or add enough real
controller state for six-joint targets to be identifiable.

Latest TCP-delta model:

```text
dataset: simulation_output/formal_route_plan_tip_signedpiper_dataset_v10_tcpdelta
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_pipercls_open_loop_diag
```

Result:

```text
observation_schema: senior_piper_real_like
elite_action_representation: tcp_delta
piper_sign_mismatch_fraction: 0.397
elite_linf median/p95: ~0.947 / 1.676 in TCP pose-delta metric space
expert/predicted Elite target-step p95: ~7.20 / 7.01
```

The diagnostic phase-state variant was:

```text
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v10_tcpdelta_seniorlike_phase_pipercls_open_loop_diag
observation_schema: senior_piper_real_like_with_phase
state_dim: 11
piper_sign_mismatch_fraction: 0.371
elite_linf median/p95: ~0.910 / 1.577
expert/predicted Elite target-step p95: ~7.20 / 7.62
```

Judgment:

```text
Do not roll out either v10 senior-like checkpoint as the next default.
Do not keep trying to make the small BC model infer scheduled Piper feed/hold.
```

TCP-delta is still the preferred Elite semantic direction, but Piper scheduling
should be made explicit as controller logic/state or moved out of the BC target.

Latest control-state TCP-delta model:

```text
dataset: simulation_output/formal_route_plan_tip_signedpiper_dataset_v12_controlstate_tcpdelta
model: simulation_output/baseline_formal_route_plan_tip_signedpiper_v12_controlstate_tcpdelta_seniorlike_pipercls
diagnostic: simulation_output/baseline_formal_route_plan_tip_signedpiper_v12_controlstate_tcpdelta_seniorlike_pipercls_open_loop_diag
```

Result:

```text
observation_schema: senior_piper_real_like
elite_action_representation: tcp_delta
piper_head: step_classification
piper_sign_mismatch_fraction: 0.361
elite_linf median/p95: ~0.930 / 1.567 in TCP pose-delta metric space
expert/predicted Elite target-step p95: ~7.11 / 7.63
expert/predicted Elite delta-from-current p95: ~2.82 / 3.12
```

Judgment:

```text
Do not roll out this checkpoint as the next default.
The v12 data/control-state path is clean, but the small BC baseline is still
not learning the real-aligned Piper/Elite mapping well enough.
```

## Historical Conclusions

The following old experiment families are now background context, not active
next steps:

- Piper negative weighting: helped rollback labels but amplified Elite target
  instability.
- Smoothness loss / low-pass smoothing: sometimes improved one metric or one
  branch, but did not solve video-level twitching or target jitter.
- Rollout rate/accel/jerk limits: useful as real-bandwidth diagnostics, but
  not a model-side fix.
- Naive Elite delta action: improved open-loop smoothness metrics but caused
  closed-loop magnet/tip drift.
- Tip-centric guidewire abstraction: useful diagnostic route, but BC still
  suffered contact/drift unless a real tip-estimation and control path is
  defined.

Retained high-level conclusion:

```text
Formal expert data is currently cleaner than the small BC policies trained on it.
The next work should align observation/action semantics with the real controller,
not tune BC rollout appearance.
```

Latest estimated-tip rollout diagnostic:

```text
model: simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls
baseline rollout: simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_r1
piper cooldown variants:
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_piperctrl_r1
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_piperctrl_cd3_r1
estimated-tip coupling guard:
  simulation_output/baseline_formal_tip_line_estimated_tip_3d_top_manual_v1_esttip_tcpdelta_pipercls_rollout_esttip_guard_r1
```

Key result:

```text
The estimated-tip coupling guard is a real-observable diagnostic controller
because it uses estimated_tip_pos_3d and current Elite TCP pose, not exact
MuJoCo contact/wall truth. It still did not solve the contact issue.

left:  success=True, steps=180, max_contact=0.570, contact_p95=0.540,
       tip_to_magnetic median/p95 ~= 17.8 / 34.8 mm, guard active 5/180
right: success=True, steps=152, max_contact=0.573, contact_p95=0.567,
       tip_to_magnetic median/p95 ~= 8.9 / 21.6 mm, guard active 0/152
```

Interpretation:

```text
The current high-contact rollout behavior is not mainly caused by the magnetic
tool losing the estimated guidewire head. Contact appears while the magnet is
already close to the tip. Do not continue cooldown-only or tip-coupling-only
tuning as the next mainline. The next useful work is to improve the contact /
wall-margin estimator and the physical/contact semantics, or to inspect whether
the learned Elite TCP direction is steering the tip too close to the wall even
when coupling is good.
```

Wall-contact direction diagnosis has been added:

```text
tool: tools/diagnose_rollout_wall_contact.py
output: docs/_wall_contact_diagnostics_esttip_guard_compare
```

This diagnostic projects the magnetic vector from guidewire tip to magnet onto
the contact normal. Positive `magnet_wall_pull` means the magnet is on the
wall/contact side of the tip. Current comparison shows high-contact samples
have positive `magnet_wall_pull`, while `estimated_image_distance_px` does not
reliably track exact contact:

```text
guard right: corr(contact, magnet_wall_pull) ~= 0.775,
             corr(contact, estimated_image_distance_px) ~= -0.192
base right:  corr(contact, magnet_wall_pull) ~= 0.787,
             corr(contact, estimated_image_distance_px) ~= -0.248
cd3 right:   corr(contact, magnet_wall_pull) ~= 0.652,
             corr(contact, estimated_image_distance_px) ~= -0.532
```

Next implication:

```text
The policy/controller needs either a better wall-margin/contact estimator or a
real-observable registered-geometry constraint that prevents Elite from guiding
the tip from the wall side. Do not treat success with close tip/magnet distance
as sufficient.
```

Registered-geometry estimator status:

```text
new schema: real_direct_plus_estimated_tip_registered_geometry
collection smoke: simulation_output/_smoke_registered_geometry_estimator_v1
rollout smoke: simulation_output/_smoke_registered_geometry_rollout_v1
audit: docs/_smoke_registered_geometry_estimator_v1_registered_schema_audit
train smoke: simulation_output/_smoke_registered_geometry_schema_train_v1
full dataset: simulation_output/formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1
trained checkpoint: simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls
baseline rollout: simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_r1
accepted wallguard rollout:
  simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_video_r1
```

The chain is now technically wired: the collector can emit registered-route
wall-margin and wall-side-risk estimator fields, rollout auto-enables the
registered estimator for the new schema, the provenance audit passes, and a
1-epoch smoke training run completes. A full dataset and checkpoint have now
also completed. The registered fields are currently useful as a conservative
controller signal, not as a strong override.

Registered-geometry wallguard result:

```text
Rejected variant:
  output: simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_r1
  result: left failed, right succeeded slowly
  cause: frequent Piper hold plus repeated away-wall TCP correction broke
         tip-magnet coupling; left tip_mag median grew to ~=98.5 mm

Accepted candidate:
  output: simulation_output/baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_video_r1
  controller: no Piper hold, no active away-wall correction; only remove the
              Elite TCP-delta component toward the estimated wall normal
  result: left/right both succeeded, right remained clean, left still has
          occasional contact but the user judged it acceptable on video
```

## Implemented Control-Semantics Update

The first TCP-delta action path has been implemented:

```text
--elite-action-representation tcp_delta
```

New collection output fields:

```text
action.elite_tcp_pose_6d
action.elite_tcp_delta_6d
```

Rollout now interprets a `tcp_delta` checkpoint as:

```text
current Elite TCP pose + predicted TCP delta -> MuJoCo IK -> Elite joints
```

The dataset and rollout logs still retain `elite_joints` for execution
diagnostics, but the semantic learning target can now match the inherited
real-code pattern: predict a TCP translation delta, keep orientation, then use
IK before `move_joint`.

## Next Action

Recommended next validation task:

```text
Stop the current BC-tuning loop. Use v12 as the boundary check showing that the
interface/data path is clean but the small model is not the main project lever.
```

Current estimator/contact interface task:

```text
docs/senior-thesis-interface-audit.md
docs/estimator-tactile-interface-spec.md
docs/visual-tip-contact-estimator-plan.md
tools/audit_observation_provenance.py
```

The spec defines candidate real guidewire-tip/contact fields such as
`estimated_tip_pos_3d`, `estimated_contact_flag`, and
`estimated_contact_strength`, plus required provenance metadata. The audit tool
checks whether a chosen `--observation-schema` consumes direct/controller
signals, estimator fields with provenance, or privileged MuJoCo truth.

Senior clarified that the real contact/tactile-like signal was implemented
simply as image-distance/collision logic, not as a neural network or dedicated
tactile sensor. Current code evidence is in `utils/camera/hsv_locate.py`,
`utils/camera/utils_camera.py`, `utils/camera/utils_cap.py`, and
`project_main.py`; the imported `frunet.collision_detect*` implementation files
are missing from this checkout.

The current VLA-facing perception route is to keep that simple visual distance
logic as a baseline/fallback, but not as the final ceiling. A learned visual
tip/contact estimator can be introduced later if it uses the same output
contract: estimated tip/contact/distance/confidence fields with provenance.

The senior thesis strengthens this route and adds an important next step: real
perception also included guidewire-head localization, depth-camera coordinate
recovery, and coordinate transforms into robot/virtual-environment frames. The
next estimator interface should move beyond pixel distance toward estimated 3D
tip and registered route/branch state, while keeping provenance and confidence.

Immediate handoff:

```text
The next mainline step is a small real-system pilot capture for shadow-mode
validation. Use data/collect/collect_real_shadow_pilot.py, which is inspired by
the senior data_collect_mode1/2 scripts but writes the current records.jsonl
schema directly. Keep the first run log-only for Piper unless the physical
setup and operator supervision explicitly justify --enable-piper-control.
The replacement UDP feeder device path exists in the collector, but repeated
onsite trials were not good enough for clean data collection; treat
--enable-feeder-device-control as fallback/debug only, not the current real-data
mainline. The real entry S-shaped bend has now been removed and feed-only smoke
improved enough to confirm the CAN/Piper control path can work, but delivery is
still not reliable enough for clean data collection. Treat the S-bend as one
blocking factor, not the only one. The next onsite move should wait for
supervisor / hardware-path guidance rather than continuing trial-and-error
collection.
Do not jump back to BC rollout tuning as the main task. Keep Isaac Sim or a
remote-workstation renderer-only spike as backup option C if MuJoCo visual
alignment clearly hits a ceiling, not as the default route.
```

Sanity check already run:

```text
v12 + senior_piper_real_like: passes provenance gate
v12 + full_sim_state: fails, because it consumes exact tip/contact/wall/route-frame fields
formal_tip_visual_distance_estimator_dataset_v1 + real_direct_plus_visual_contact: provenance passes, but visual audit fails because off-frame side projections were counted as visible
formal_tip_visual_distance_estimator_dataset_v2_frameclip + real_direct_plus_visual_contact: provenance passes and frame clipping works, but all_visible binary contact is always zero because side-camera coverage is low
formal_tip_line_black_red_head_sidecam_v1_dataset_v2: collection succeeds but estimator fields are invalid because wall markers can be visually wrong
_smoke_image_estimator_wall_fix_v3: fixed rendered-image estimator is visually accepted; side view works, top view is often missing because Elite occludes the red head
formal_tip_line_black_red_head_sidecam_v1_wallfix_dataset_v1 + real_direct_plus_visual_contact: provenance passes, user visual review accepts the wallfix samples, side estimates are available for 3915/4495 samples, top estimates are missing under current Elite occlusion
senior thesis interface audit: confirms image + Elite 6D pose + Piper count -> Elite 6D pose delta + Piper hold/feed, and supports a visual estimator route with tip localization, coordinate transforms, collision/distance, and controller safety checks
_smoke_registered_geometry_estimator_v1 + real_direct_plus_estimated_tip_registered_geometry: provenance passes; fields are estimator-provenance, not privileged schema inputs
_smoke_registered_geometry_rollout_v1: policy-time registered geometry fields are written for execution steps; final terminal frame may omit estimator fields
_smoke_registered_geometry_schema_train_v1: 1-epoch smoke training completes with 32D registered-geometry state
formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1: full formal registered-geometry dataset collected and audited; schema is real_direct_plus_estimated_tip_registered_geometry
baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_r1: both sides succeed but left has mild contact/visual wall proximity
baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_r1: rejected; strong Piper hold/away correction breaks tip-magnet coupling
baseline_formal_tip_line_visual_route_v1_esttip_reggeom_tcpdelta_pipercls_rollout_wallguard_eliteonly_video_r1: accepted candidate; both sides succeed and user visual review accepts occasional contact
tools/real_shadow_policy_adapter.py: sim-to-real shadow-mode adapter; reads real-style image/state JSONL, writes raw_policy_action, shadow_controller_action, and wallguard metadata, and never commands robots
tools/build_branchs_shadow_input.py: converts senior branchs data to shadow JSONL; diagnostic only because branchs has one camera and no estimated-tip/registered-geometry fields
branchs_shadow_input_smoke + current reggeom checkpoint: 240-record smoke runs end-to-end, but current sim-trained checkpoint predicts Piper feed for all 240 frames and Elite TCP delta magnitude is about 14-16 mm, showing a clear sim-to-real/input-schema mismatch rather than a ready real policy
branchs_shadow_input_visualcontact_smoke: simplified HSV red-tip plus image-edge-distance estimator detects red tip in 234/240 frames and flags 104/234 as contact-like, but the current reggeom checkpoint still predicts Piper feed for all 240 frames; this contact signal can temporarily fill visual-contact fields, but it is not an estimated 3D tip substitute
branchs_shadow_predictions_visualcontact_full: full 4039-record shadow run still predicts Piper feed for every frame; mismatch is 1043/2359 on left and 806/1680 on right, while Elite TCP delta norm is roughly 13-15 mm. The temporary image-edge contact estimator is also over-sensitive, flagging 2902/4033 detected frames as contact-like. Treat this as evidence of real-input/schema mismatch, not a ready sim-to-real policy.
tools/build_branchs_training_manifest.py: converts branchs into a train_dual_arm_baseline manifest with duplicated single-camera images, Elite TCP 6D pose + Piper step state, Elite TCP delta labels from pose[t+1]-pose[t], and Piper 0/1 labels
branchs_training_manifest_full: 4015 samples, feed=2179, hold=1836; use it for a branchs-native senior-like real-data baseline before claiming anything about sim-to-real transfer
baseline_branchs_native_seniorlike_tcpdelta_pipercls: branchs-native real-data baseline trained successfully. Open-loop piper mismatch is 5.75% overall (left 6.18%, right 5.15%), so the model/training pipeline can learn the senior real-data Piper labels. Elite TCP-delta error is about median/p95 0.50/2.08 in TCP delta metric space, with right branch slightly worse. This supports interpreting the previous all-feed branchs shadow result as a sim-to-real/input-schema mismatch of the sim-trained reggeom checkpoint.
tools/build_single_camera_seniorlike_manifest.py: converts a sim manifest to a branchs-like single-camera senior_piper_real_like manifest by duplicating one chosen sim camera as side/top while preserving Elite TCP-delta and Piper hold/feed labels
baseline_formal_tip_line_visual_route_v1_singlecam_side_seniorlike_tcpdelta_pipercls: sim single-camera senior-like model trained on formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1. Same-schema sim open-loop improved versus earlier senior-like attempts (Piper mismatch 21.17%, Elite TCP-delta linf median/p95 0.44/1.69), but branchs transfer still collapsed to all-feed. On branchs open-loop, Piper mismatch is 45.73% and Elite TCP-delta linf median/p95 is 5.32/9.71; branchs shadow predicts feed for all 4039 records. This rules out "single vs dual camera schema" as the only cause and points to visual/domain distribution plus real label/control timing mismatch.
tools/audit_sim_real_visual_domain.py: compares sim and branchs image samples and writes a contact sheet plus simple image statistics. The first audit, docs/_sim_real_visual_domain_audit_singlecam_v1, found an obvious visual-domain gap: sim luminance_mean 184.38 vs real 120.49, sim bright_ratio 0.324 vs real 0.002, and the contact sheet shows bright clean synthetic rendering versus darker green-background real camera frames with real robot/fixture appearance. Treat rendering/camera/domain alignment or aligned real-data collection as the next lever before another BC rollout-tuning loop.
branchs_like_v1 render preset: opt-in MuJoCo render/domain preset enabled by --render-domain-preset branchs_like_v1. Smoke output simulation_output/_smoke_branchs_like_render_preset_v1 succeeds left/right with zero contact. Audit docs/_sim_real_visual_domain_audit_branchs_like_v1 improves side-camera statistics to luminance_mean 118.60 vs real 120.76, saturation_mean 0.407 vs real 0.392, bright_ratio 0.000 vs real 0.003, and edge_density 0.027 vs real 0.026. User visual review accepted it as good enough for the next step, while noting the remaining gap is mainly real glass-vessel reflections and real-camera blur. Physical fixtures, crop/perspective, and occlusion are also not fully reproduced.
formal_tip_line_branchs_like_v1_small: accepted 8/8, rejected 0, samples 936, left/right 4/4, side/top image files complete, contact_p95 max 0.0, max_contact_strength max about 0.00053, min_tip_distance_to_wall min about 1.39 mm, tip_to_magnetic_p95 median about 7.68 mm, Piper feed/hold 486/450, estimated-tip 3D and registered-geometry fields present for all samples. Provenance audit docs/_formal_tip_line_branchs_like_v1_small_provenance_audit passes for real_direct_plus_estimated_tip_registered_geometry; visual-domain audit docs/_sim_real_visual_domain_audit_branchs_like_v1_small remains close to branchs side-camera statistics. Next scale this preset before training.
formal_tip_line_branchs_like_v1_full: current accepted branchs-like full training candidate. accepted 40/40, rejected 0, samples 4495, left/right 20/20, side/top images complete, contact_p95 max 0.0, max_contact_strength max about 0.00053, min_tip_distance_to_wall min/median about 1.39/1.81 mm, tip_to_magnetic_p95 median about 7.68 mm, Piper feed/hold 2355/2140, estimated-tip visible 4495/4495 with top pixel source, registered-geometry fields complete. Provenance audit docs/_formal_tip_line_branchs_like_v1_full_provenance_audit passes; visual-domain audit docs/_sim_real_visual_domain_audit_branchs_like_v1_full remains stable. Use this for the next training command.
baseline_formal_tip_line_branchs_like_v1_reggeom_tcpdelta_pipercls: trained on the accepted branchs-like full dataset with real_direct_plus_estimated_tip_registered_geometry, TCP-delta Elite action, and Piper step classification. Open-loop diagnostic shows Piper mismatch 21.20% overall (left 23.87%, right 18.43%), with feed->hold errors 646 and hold->feed errors 307. Elite TCP-delta linf error median/p95 is about 0.515/1.707, and predicted Elite target-step p95 is 3.715 vs expert 2.995. Do not treat it as a completed sim-to-real policy; use it only as a baseline/diagnostic unless a later rollout is explicitly needed.
baseline_formal_tip_line_branchs_like_v1_singlecam_side_seniorlike_tcpdelta_pipercls: trained on the accepted branchs-like full dataset after converting it to branchs-compatible single-camera senior_piper_real_like form. Same-schema sim open-loop is slightly better than the earlier visual_route single-camera run (Piper mismatch 20.24%, Elite TCP-delta linf median/p95 0.496/1.557), but branchs transfer still collapses to all-feed: Piper mismatch 45.73%, hold->feed 1836, feed->feed 2179, and Elite TCP-delta linf median/p95 3.614/8.102. This shows branchs_like_v1 rendering is useful but insufficient by itself; the next lever is real-observation coverage and real/sim data alignment, not another small-BC rollout tweak.
tools/export_isaac_renderer_spike.py: Plan C renderer-only bridge for the user's separate Isaac Sim host. It exports a portable package with vessel STL, route/camera/scene configs, selected sim/real reference images, samples.jsonl, README.md, and isaac_build_stage.py. Current package is simulation_output/isaac_renderer_spike_branchs_like_v1_pkg with 6 side-camera samples and no missing images. Run the generated isaac_build_stage.py on the Isaac host to convert the STL and build per-frame USD stages. This is not a MuJoCo physics/control migration.
docs/real-data-collection-checklist.md: current next real-system pilot target. It defines the minimum real shadow dataset: synchronized side/top images, Elite TCP 6D pose, Piper step/state, task/target metadata, reference Piper labels, optional estimator/contact fields with explicit null/low-confidence values, timestamp-delta diagnostics, and a records.jsonl layout compatible with tools/real_shadow_policy_adapter.py. Use this before asking for another sim-trained policy comparison.
data/collect/collect_real_shadow_pilot.py: current real pilot collection entrypoint. Mock smoke at simulation_output/_smoke_real_shadow_pilot_mock produced 3 records, side/top frames, Elite pose log, Piper state log, command log, manifest, and summary. The script defaults to log-only Piper and requires --enable-piper-control before it calls Piper hardware.
hardware/feeder_device + --enable-feeder-device-control: UDP JSON adapter for the replacement guidewire feeder. Protocol probing and integration are kept for fallback/debug, but repeated onsite trials showed the device is not currently reliable enough to be the main real-data collection route.
real fixture S-bend removal: latest onsite test shows the physical entry S-shaped bend was only one cause of guidewire push blockage. Removing it helped confirm CAN/Piper control can execute, but guidewire delivery remains unreliable for clean data collection. Pause onsite collection and wait for supervisor / hardware-path guidance before deciding whether to use Piper, the UDP feeder, or manual/external labels.
```

Before scaling either option:

- keep `--formal-data` / `--formal-eval` for formal checks;
- keep Piper feed/hold labels explicit;
- do not use exact tip/contact/wall oracle fields as policy inputs unless a real
  estimator/sensor path is defined;
- judge any model first with open-loop diagnostics before rollout.
- policy output should be Elite TCP delta plus Piper intent;
- controller output should be the executed Elite/Piper command after IK,
  timing, cooldown, bounds, and safety checks;
- if Piper remains scheduled by route-plan/controller logic, expose that as real
  controller state or keep it outside the BC target instead of asking images to
  infer the hidden schedule.
- next mainline work should improve real-observation coverage, real/sim data
  alignment, controller semantics, and physical realism rather than tuning
  another small BC rollout.
- define estimator/tactile fields before using guidewire-tip/contact signals as
  policy inputs.

## Start Here

A new agent should read:

```text
AGENTS.md
docs/project-state.md
docs/senior-thesis-interface-audit.md
docs/data-and-action-schema.md
docs/observation-schema-audit.md
docs/real-observation-schema-alignment.md
docs/real-observable-interface-audit.md
docs/real-data-collection-checklist.md
docs/estimator-tactile-interface-spec.md
docs/visual-tip-contact-estimator-plan.md
docs/simulation-expert-validity-audit.md
docs/vla-target-interface.md
docs/control-layer-contract.md
docs/commands.md
```

Key paths:

```text
simulation/mujoco_guided_wire_env.py
simulation/train_dual_arm_baseline.py
simulation/eval_mujoco_guided_wire_rollout.py
simulation/collect_tip_guided_wire.py
tools/diagnose_bc_open_loop.py
tools/audit_observation_provenance.py
```

## Do Not Do Next

- Do not treat real `branchs` pose as guidewire trajectory.
- Do not jump to OpenVLA or a full Isaac Sim physics/control migration. The
  current Isaac route is only Plan C renderer-only staging on the user's
  separate Isaac host.
- Do not repeat arbitrary smoothing as the main fix.
- Do not roll out the v9 senior-like checkpoint.
- Do not roll out the v10 senior-like or phase checkpoint as the next default.
- Do not roll out the v12 control-state TCP-delta checkpoint as the next
  default unless a new diagnostic materially improves open-loop behavior.
- Do not change away from `piper_feed_elite_joint` casually; if Elite TCP
  delta/IK is introduced, document why it better matches the real controller.

## Workflow Reminder

The user should run long data collection, training, full rollout, and
video-rendered evaluation jobs manually by default. Provide exact commands and
inspect artifacts afterward.
