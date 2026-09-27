# Estimator And Tactile Interface Spec

Last updated: 2026-07-13

## Purpose

The next mainline step is to define which guidewire-tip and contact signals can
exist on the real system before using their MuJoCo equivalents in formal
training.

The current decision is:

```text
Do not keep optimizing the small BC rollout.
Define the real estimator/tactile interface first.
```

This document specifies the candidate interface that future VLA-facing data
should use. It is intentionally stricter than the simulator state: exact MuJoCo
tip, wall, contact, and route-frame values remain diagnostic unless they are
converted into an estimator-like signal with declared provenance.

The design route for rule-based versus learned visual estimators is in:

```text
docs/visual-tip-contact-estimator-plan.md
```

## Current Boundary

Direct real observations are already defined:

```text
side/top images
Elite TCP 6D pose
Piper step/insertion/controller state
task instruction
```

The missing useful observations are:

```text
guidewire tip estimate
guidewire heading estimate
contact/tactile estimate
wall safety margin estimate
registered-route progress estimate
```

These are Tier 2 fields in `docs/real-observable-interface-audit.md`: allowed
only after a real acquisition path or estimator contract is defined.

Current working assumption from the real system:

```text
"tactile/contact" is a visual image-distance/contact estimate: camera image ->
guidewire-head contour/keypoint -> distance/collision against a vessel wall mask.
```

Treat this as a visual distance/contact estimator unless the real hardware later
exposes a dedicated tactile or force sensor. It should not be assumed to be a
neural network; senior's clarification was that this part was implemented
simply, by computing an image distance.

Current algorithm decision:

```text
The first small VLA-style model should receive this image-derived
contact/tactile signal as an explicit input field.
```

This is the tactile channel for the current project stage. It means
image-derived contact estimation, not hardware force/tactile sensing and not
direct MuJoCo contact truth.

Current repo evidence:

```text
utils/camera/hsv_locate.py
  detect_red(), detect_red2()
  contours_intersect_or_min_distance(...)

utils/camera/utils_camera.py
  Camera.predict_pos() -> CollisionDetector.simple_analyze(...)

utils/camera/utils_cap.py
  CaptureUtil.show_frame_detect() -> CollisionDetector.simple_analyze(...)

project_main.py
  camera_main() -> collision_detector.simple_analyze(...) -> global collision
```

Important caveat:

```text
frunet.collision_detect and frunet.collision_detect2 are imported by the real
camera code, but the corresponding module files are not present in the current
checkout. The current local evidence is therefore the caller-side contour/mask
pipeline and imports, not the full detector implementation.
```

## Candidate Real Estimator Outputs

### Guidewire Tip

Policy-facing field names:

```text
estimated_tip_pos_3d
estimated_tip_heading_3d
tip_estimator_confidence
tip_estimator_visible
tip_estimator_latency_steps
```

Required semantics:

- `estimated_tip_pos_3d`: estimated guidewire head position in a calibrated
  robot/vessel frame, meters.
- `estimated_tip_heading_3d`: estimated local head direction, unit vector when
  available.
- `tip_estimator_confidence`: scalar in `[0, 1]`; low confidence means the
  policy/controller should rely more on safe hold/slow-feed behavior.
- `tip_estimator_visible`: boolean or `0/1`; false when the tip is occluded or
  segmentation fails.
- `tip_estimator_latency_steps`: integer delay between image acquisition and
  estimator output.

Real acquisition path:

```text
side/top camera images -> guidewire segmentation/keypoint detection ->
multi-view or calibrated projection -> estimated tip pose
```

Simulation generation rule:

```text
Do not expose state.tip_pos directly as estimated_tip_pos_3d.
Generate estimated_tip_pos_3d through a declared estimator model or synthetic
noise/latency/dropout wrapper that mimics the expected real estimator.
```

Minimum first implementation:

```text
estimated_tip_pos_3d = tip_pos + calibrated noise + optional latency/dropout
estimated_tip_heading_3d = heading + angular noise + optional invalid flag
```

This is still a simulated estimator, but it is honest about uncertainty and can
be audited separately from exact MuJoCo truth.

### Contact Or Tactile

Policy-facing field names:

```text
estimated_contact_flag
estimated_contact_strength
estimated_contact_direction_2d
contact_estimator_confidence
contact_source
```

Required semantics:

- `estimated_contact_flag`: binary likely-contact signal.
- `estimated_contact_strength`: optional scalar in `[0, 1]`, interpreted as
  estimator confidence or severity proxy, not exact MuJoCo wall penetration.
- `estimated_contact_direction_2d`: optional coarse local direction such as
  left/right or up/down, only if the real visual distance estimator supports it.
- `contact_estimator_confidence`: scalar in `[0, 1]`.
- `contact_source`: one of `visual_distance_estimator`, `force_proxy`,
  `tactile_sensor`, `synthetic_visual_distance_estimator`, or `unknown`.

Likely real acquisition path:

```text
side/top camera image(s) -> guidewire-head contour/keypoint ->
vessel-wall mask distance/collision threshold ->
estimated_contact_flag + optional confidence/distance
```

Recommended first real-compatible choice:

```text
estimated_contact_flag
contact_estimator_confidence
estimated_image_distance_px
contact_source="visual_distance_estimator"
```

For the rule-based synthetic baseline, prefer a conservative binary flag:

```text
estimated_image_distance_px = continuous side/top projected distance signal
estimated_contact_flag = 1 only when both visible side/top distances pass the
                         contact threshold
```

This avoids treating a single-view silhouette overlap as true contact.

Avoid making contact strength, normal, or direction a first formal dependency
unless the real image-distance estimator explicitly outputs them. Directional
contact is tempting but high-risk because exact MuJoCo `contact_normal` is much
cleaner than any likely real signal.

Simulation generation rule:

```text
Do not expose contact_flag/contact_strength/contact_normal as exact truth under
estimated_* names. Apply image-distance-style thresholding, pixel/segmentation
noise, false positives/negatives, confidence calibration, dropout, and optional
delay, and record contact_source="synthetic_visual_distance_estimator".
```

### Wall Safety Margin And Route Progress

Policy-facing field names:

```text
estimated_wall_margin
estimated_route_progress
route_estimator_confidence
registered_route_id
```

Required real path:

```text
registered vessel model + estimated tip pose -> route progress / wall margin
```

Formal rule:

These fields are allowed only if both conditions hold:

1. the vessel route/mesh can be registered to the real camera/robot frame;
2. guidewire tip estimate exists in the same frame.

If either condition is missing, keep route progress and wall margin out of the
policy input. Use them only for offline diagnostics.

## Proposed Observation Schema

Future schema name:

```text
real_direct_plus_estimated_tip_contact
```

Policy inputs:

```text
side/top images
Elite TCP 6D pose
Piper step/insertion/controller state
task id/instruction
estimated_tip_pos_3d
estimated_tip_heading_3d
tip_estimator_confidence
tip_estimator_visible
estimated_contact_flag
contact_estimator_confidence
estimated_image_distance_px
contact_source
```

Optional later additions:

```text
estimated_contact_strength
estimated_contact_direction_2d
estimated_wall_margin
estimated_route_progress
route_estimator_confidence
```

Do not include:

```text
exact tip_pos
exact heading
exact contact_normal
exact distance_to_wall
exact path_progress
exact lateral_offset
exact path_tangent
exact local_radius
```

unless they are explicitly wrapped as estimated fields with provenance.

## Required Provenance Metadata

Any dataset that contains estimated fields should record:

```json
{
  "observation_provenance": {
    "estimated_tip_pos_3d": {
      "tier": "estimator",
      "source": "synthetic_estimator",
      "truth_field": "tip_pos",
      "noise_model": "gaussian_xyz_m",
      "latency_steps": 1,
      "dropout_probability": 0.05,
      "formal_policy_input_allowed": true
    }
  }
}
```

Minimum required keys:

```text
tier
source
formal_policy_input_allowed
```

If `source` is `synthetic_estimator`, also record:

```text
truth_field
noise_model
latency_steps
dropout_probability
```

## Pre-Implementation Checklist

Before coding the schema, decide:

1. What frame will `estimated_tip_pos_3d` use?
2. What noise scale is realistic for the side/top camera setup?
3. Should the first estimator be purely synthetic, image-derived, or both?
4. What exactly does the real visual distance/contact estimator output: binary
   flag only, pixel distance, confidence, severity proxy, or direction?
5. Will route/wall margin be available online, or kept offline only?

Recommended first implementation:

```text
synthetic estimator wrapper with noise + latency + dropout
visual-distance-style binary contact flag + optional pixel distance/confidence
no contact direction
no route progress as policy input
audit provenance before training
```

This is the smallest step that moves toward VLA/tactile realism without
pretending MuJoCo truth is a real sensor.
