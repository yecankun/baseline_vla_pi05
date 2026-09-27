# Simulation Assumptions

Last updated: 2026-07-13

This document describes simplifications that are intentional at the current
stage. Do not treat every mismatch with reality as a bug.

## Highest Priority

The highest priority is making the simulator and data-generation semantics as
close to the real system as possible. A small BC model succeeding in simulation
is only a regression check.

Do not accept a mechanism as part of formal data generation only because it
improves rollout video or success rate. First ask whether the same mechanism
could exist on the real system with real observations, real robot control, and
reasonable calibration.

Dedicated audit:

```text
docs/simulation-expert-validity-audit.md
```

## Guidewire Model

The current guidewire is a simplified polyline-like physical object, not a full
continuum robot, FEM rod, or high-fidelity flexible body.

It currently uses approximations for:

- stiffness;
- damping;
- curvature smoothing;
- wall interaction;
- magnetic influence;
- Piper advection/feed.

The goal is stable task-level data generation, not publication-grade guidewire
mechanics yet.

There is now also a tip-centric / hard-elastic prototype:

```text
simulation/tip_guided_wire_env.py
```

This prototype treats the magnetically guided guidewire head as the primary
state and uses a continuous line-shaped visual guidewire tail instead of full
flexible-body/polyline dynamics. It keeps the real vessel route, robot
kinematics, Piper feed semantics, and corrected Elite-attached magnetic point.
Use it as a parallel experiment to test whether task-level data generation is
better served by modeling the observable/control-relevant guidewire tip under a
hard elastic wire assumption rather than trying to reproduce every trailing-wire
deformation.

The current justification is not only engineering convenience. The real
guidewire material has been confirmed to be relatively hard and elastic, closer
to a springy wire than a very soft cable. Under that assumption, controlling the
head so it does not touch the vessel wall is a reasonable first-order proxy for
avoiding problematic trailing-wire contact.

For formal tip-centric image data, the current visual default is a MuJoCo
line-like capsule chain rather than the older ball/segment-looking wire:

```text
--wire-visual-mode line
--wire-visual-radius 0.0008
--wire-visual-rgb "0.02 0.02 0.018"
--wire-tip-visual-rgb "0.78 0.04 0.02"
--wire-tip-visual-segments 5
--wire-tip-visual-radius-scale 2.2
--wire-tip-marker-radius 0.0020
--wire-tip-marker-alpha 0.95
--wire-visual-offset 0.0 0.0 0.0
--wire-segments 64
--tip-tail-decay-segments 18
```

This visual split matches the real setup more closely: most of the guidewire is
black and thin, while the head is red and visually wider. These parameters only
change rendering; guidewire/tip dynamics, wall/contact metrics, expert control
semantics, and policy labels are unchanged. Keep the default visual offset at
zero unless a camera-calibrated comparison shows that a real guidewire-depth
offset is needed.

This is a visual-realism choice, not a new hidden controller. It preserves
MuJoCo occlusion and should be used with hidden debug markers/path tubes for
formal image data.

### Visual Tail Route Geometry

The visual guidewire tail should follow this physical routing semantics:

```text
Piper outlet -> vessel entry center -> in-vessel route center -> red tip
```

The vessel entrance is not a straight line. User visual inspection indicates
that the real/model entrance region has an S-shaped bend, so the rendered
guidewire tail should preserve that bend rather than connecting the Piper
outlet to an interior point with one or two straight segments.

User annotation on 2026-06-30 further confirmed that the current rendered route
can use the wrong outlet/exit side in the side view. Treat
`wire_visual_piper_exit_point` as a manually calibrated visual outlet, not an
automatic Piper link midpoint, until it is visually accepted against the real
Piper outlet and vessel entrance.

Clicked vessel-wall/surface points are only selection evidence. They should not
be used directly as the guidewire path. Any picked entrance/via location should
be converted to a route-center point before it drives the visual tail.

The current native MuJoCo route-marker picker fixed the earlier custom
world-to-screen projection mismatch, but visual inspection still suggests that
parts of the registered route center are biased toward the vessel wall. Treat
the current `wire_visual_via_points` in
`simulation/routes/vessel_0422_wire_route_v1.json` as provisional until the
entrance centerline and S-bend have been re-picked or rebuilt and visually
accepted.

## Vessel Wall Interaction

The vessel mesh is real, but collision/contact is still approximated.

Important metric:

```text
min_segment_distance_to_wall
```

Small negative values can occur. They mean some segment is slightly outside or
inside the boundary approximation. Interpret them together with video and
contact metrics.

## Magnetic Guidance

The magnetic field/control model is simplified.

The current corrected semantics are:

- the magnetic point is attached to the Elite end tool;
- `magnetic_pose` should coincide with `elite_pose` / `elite_tool_world`;
- Elite's role is magnetic guidance through that physical end tool, not a
  separate hidden effective point near the guidewire tip.
- The Elite magnetic target should be in front of and above the guidewire head
  or registered route point, not directly above it. A directly overhead magnet
  mostly pulls upward; the real guiding setup needs a forward-upward attraction
  component so Piper feed can advance the guidewire. The current MuJoCo experts
  therefore target:

```text
route/tip anchor + path-forward offset + world-up height offset
```

`magnetic_pose` still records the actual Elite-attached magnetic point after
IK/execution, not a hidden proxy point.

Front-up must be checked on the executed Elite-attached magnet, not only on the
target formula. If route-plan progress is slower than guidewire progress, the
target can be front-up relative to an old registered route point while the
actual magnet is behind the current guidewire head. Audit this with
`tools/audit_elite_frontup_guidance.py`; current formal collection defaults use
`--route-plan-step 0.24` to match the observed simulator guidewire progress
scale.

Older datasets and checkpoints collected before 2026-06-19 used a different
effective-magnetic-point semantics. Do not compare those old rollouts as if the
action labels still mean the same thing.

## No Artificial Centerline Force

Earlier versions effectively constrained the guidewire toward a reference path
or centerline. The current physical mode does not use such a center force.

This is intentional because real guidewire motion should be driven by:

- magnetic force;
- Piper feeding;
- wall interaction;
- gravity;
- wire stiffness/damping.

Without centerline force, control becomes harder and wall contact becomes more
visible.

## Piper Role

Piper is not a mobile guidewire-tip controller. It stays near the entry region
and performs local feed steps.

Current MuJoCo `piper_feed` still allows continuous feed/retract values. This is
a simulator-side simplification, not the confirmed real Piper interface.
Senior's `intervention/bc_piper/inference.py` uses binary stop/advance actions:
`0=stop/hold`, `1=piper.step_forward(...)`.

Retract should remain on the table as a practical real-control capability, but
it needs a real Piper command path and corresponding labels.

If Piper appears to follow the guidewire through the vessel, that is likely an
incorrect control/visualization assumption.

## Cameras and Rendering

The current scene includes material, table, lighting, and two cameras for
dataset generation. These are practical approximations, not final real camera
calibration.

Formal training data should not include debug path tubes or tool markers.

## What Counts as Good Enough Now

For the current stage, a rollout is useful if:

- left/right tasks can complete in closed loop;
- guidewire behavior is visually plausible;
- wall penetration/contact is not extreme;
- Elite joint target and executed joint jumps are not systematically large;
- data can be collected and trained without obvious visual leakage.

These are necessary checks, not sufficient proof of realism. A rollout that
succeeds because of privileged expert state, hidden centerline control,
simulation-only contact feedback, or an oracle tip-relative anchor should be
treated as diagnostic until the corresponding real controller or sensor path is
defined.

This is not yet the final sim-to-real environment.
