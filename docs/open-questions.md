# Open Questions

Last updated: 2026-06-30

These are unresolved research or engineering questions. Do not treat them as
simple implementation bugs unless new evidence narrows them down.

## Guidewire Physical Fidelity

Question:

How close does the simplified MuJoCo guidewire model need to be before it is
useful for sim-to-real?

Current state:

- The guidewire is modeled as a simplified polyline with stiffness, damping,
  magnetic influence, wall interaction, and curvature smoothing.
- It is not a full FEM or high-fidelity flexible-body model.
- Some visually acceptable rollouts still show small negative wall clearance.
- A parallel tip-centric prototype now exists:
  `simulation/tip_guided_wire_env.py`. It models the magnetically guided
  guidewire head as the primary state and uses a short visual tail rather than
  full flexible-wire dynamics.

Need:

- More systematic clearance/contact diagnostics.
- Compare polyline versus tip-centric expert data quality: success rate, stalled
  progress, tip-to-magnetic distance, contact strength, and wall clearance.
- Real-data comparison once real videos/trajectories are available.

## Contact/Tactile Signal Meaning

Question:

What tactile information should be exposed to the model?

Current state:

- Senior clarified that real "tactile/contact" information was implemented
  simply, without a neural network: image input -> guidewire-head contour or
  keypoint -> distance/collision against a vessel wall mask.
- The original lab setup may not have a dedicated tactile/force sensor.
- Contact-like feedback in simulation is currently derived from geometric
  distance/contact against the vessel wall.
- Current checkout evidence is caller-side code in `utils/camera/hsv_locate.py`,
  `utils/camera/utils_camera.py`, `utils/camera/utils_cap.py`, and
  `project_main.py`.
- `frunet.collision_detect` and `frunet.collision_detect2` are imported by that
  code, but their module files are not present in the current checkout.

Need:

- Recover or inspect the missing `frunet` collision detector implementation if
  available in another copy of the project.
- Confirm the real visual distance estimator's inputs, threshold, output labels,
  optional pixel distance/confidence score, latency, and failure cases.
- Start with `estimated_contact_flag` plus `contact_estimator_confidence` as
  the likely formal signal, and optionally add `estimated_image_distance_px` if
  it is actually available.
- Do not use exact MuJoCo `contact_strength`, `contact_normal`, or
  `distance_to_wall` as policy input unless wrapped as a visual-estimator-like
  signal with provenance.

## Elite Visual Jump Source

Question:

How much of the Elite visual jump is model output, control execution, kinematic
mapping, or rendering?

Current evidence:

- Expert labels are smooth.
- Weighted policy outputs are much less smooth than unweighted policy outputs.
- Magnetic effective point can track the guidewire tip while the visible robot
  tool appears far or jumpy.
- `simulation_output/real_control_bandwidth` shows real Elite/magnetic-arm pose
  bandwidth: position-step p95 median about `5.07 mm/frame`, acceleration p95
  median about `2.98 mm/frame^2`, jerk p95 median about `2.84 mm/frame^3`, and
  direction-reversal median `0.0`.
- `rate010_smooth025_r1` can match the real step scale numerically, but video
  still shows visible jitter, so step size alone is not enough.

Need:

- A rollout bandwidth comparison that includes step, acceleration, jerk,
  direction reversals, and hold-frame behavior.
- A realistic Elite execution/control layer grounded in real bandwidth, rather
  than arbitrary smoothing.
- Continued diagnostics using `tools/analyze_mujoco_elite_rollout.py` and
  `tools/compare_sim_real_alignment.py`.

## Piper Semantics

Question:

How should the simulator expose Piper actions so they are real-implementable,
while keeping the useful retract capability?

Current state:

- Real `branchs` data has Piper labels `0/1`.
- `intervention/bc_piper/inference.py` confirms the real inference semantics:
  `0` means stop/hold, and `1` calls `piper.step_forward(...)` to advance one
  Piper step.
- MuJoCo expert/policy uses continuous normalized `piper_feed`, including
  negative retraction values.
- The project should not remove retract just to match the senior two-action BC
  interface; retract is likely needed for a practical real controller.

Need:

- Decide whether the real-aligned action should be discrete
  `hold/feed/retract`, a signed step command, or another actuator-level command
  that the real Piper can execute.
- Preserve the senior semantics as a known baseline (`0=hold`, `1=feed`) while
  explicitly adding retract if the real controller will support it.
- Avoid claiming sim-to-real action alignment until this is resolved.

## Correct Clearance Target

Question:

Should the guidewire be encouraged to stay centered, or is occasional wall
contact acceptable/necessary?

Current state:

- Earlier centerline constraints were too artificial.
- Removing centerline force improved physical honesty but made wall proximity
  more visible.
- Real guidewires may contact vessel walls, but persistent wall-hugging is not
  desirable.

Need:

- Define acceptable clearance/contact thresholds.
- Compare against real operation if available.

## Registered Route Centerline And Entrance Shape

Question:

How should the registered in-vessel route centerline be rebuilt so the visual
guidewire tail matches the real vessel geometry, especially near the entrance?

Current state:

- The old image picker based on MuJoCo surface selection was unreliable and
  often selected table/vessel surface points rather than route-center points.
- The newer native MuJoCo route-marker picker removes the custom projection
  mismatch by rendering route-center candidates as actual scene geoms.
- Visual inspection of those native markers suggests the registered route is
  still biased toward the vessel wall and does not yet look like a robust
  centerline.
- The entrance region should include an S-shaped bend: the correct path is not
  a straight segment from Piper outlet to a downstream interior point.
- User side-view annotation on 2026-06-30 showed that the generated visual wire
  can exit from the wrong side; this is an outlet/entry-route semantics problem,
  not a BC-model or rollout-tuning issue.

Need:

- Rebuild or correct the entrance route as
  `Piper outlet -> vessel entry center -> S-bend route center -> red tip`.
- Calibrate `wire_visual_piper_exit_point` as a free-space visual outlet, not
  by automatically taking a Piper link midpoint.
- Convert any clicked vessel-wall evidence into corresponding route-center
  points before using it for visual guidewire rendering.
- Visually validate the route with both robots hidden and with Piper visible,
  because the Piper outlet and vessel entry are separate semantic points.

## Simulator Roadmap

Question:

When should the project move from MuJoCo to Isaac Sim or another simulator?

Current state:

- MuJoCo is useful for fast iteration and current modeling questions.
- Isaac Sim could help with richer robot/camera rendering later.
- It is not yet clear that Isaac Sim will solve guidewire realism.

Need:

- First stabilize MuJoCo collection/training/rollout.
- Then decide whether the next bottleneck is rendering, robot integration, or
  guidewire physics.

## Real Dataset Integration

Question:

How should senior's real data be used?

Current state:

- It is valuable for understanding robot behavior and scene scale.
- It should not be directly treated as simulated guidewire path labels.

Need:

- Inventory the real dataset formats.
- Identify which signals are robot joints, camera images, magnetic tool pose,
  and any inferred guidewire state.
