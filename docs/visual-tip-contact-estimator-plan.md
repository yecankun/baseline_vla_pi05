# Visual Tip And Contact Estimator Plan

Last updated: 2026-06-26

## Purpose

The next VLA-facing perception step should use the existing side/top cameras
more effectively without adding new hardware.

The target is not a dedicated tactile sensor. The target is a real-observable
visual estimator:

```text
side/top camera images -> guidewire tip/contact/distance/confidence estimates
```

These estimates can then be used by a future VLA policy together with raw
images, Elite TCP pose, Piper/controller state, and task instruction.

The senior thesis interface audit adds an important constraint:

```text
docs/senior-thesis-interface-audit.md
```

The real system did not stop at a contact bit. It combined guidewire-head
localization, depth-camera coordinate recovery, coordinate transforms, and
virtual-environment display. The estimator route should therefore be staged
from image-space distance to estimated 3D tip and registered route/branch state.

## Design Position

The inherited real code appears to use a simple visual distance/contact logic:

```text
red guidewire-head contour/keypoint + vessel wall mask -> image distance/contact
```

This is a useful baseline, but it should not limit the final design. For VLA
work, the better route is a staged estimator layer:

```text
rule-based visual distance estimator as baseline/fallback
+ learned visual estimator as the higher-capacity path
+ stable policy-facing output contract
```

The key point is that both implementations should expose the same output
schema. The VLA/controller should not care whether the estimate came from HSV
thresholding, geometric distance, or a neural network, as long as provenance,
confidence, latency, and failure modes are recorded.

## Recommended Output Contract

Minimum first contract:

```text
estimated_tip_pixel_side
estimated_tip_pixel_top
estimated_tip_visible
estimated_image_distance_px
estimated_contact_flag
contact_estimator_confidence
tip_estimator_confidence
contact_source
estimator_latency_steps
```

Optional later contract:

```text
estimated_tip_pos_3d
estimated_tip_heading_3d
estimated_contact_strength
estimated_contact_direction_2d
estimated_wall_margin
registered_route_id
estimated_route_progress
estimated_branch_state
route_estimator_confidence
```

Do not expose exact MuJoCo `tip_pos`, `contact_normal`, `contact_strength`,
`distance_to_wall`, or `path_progress` as policy inputs unless they are wrapped
as estimated fields with declared provenance.

## Rule-Based Baseline

Candidate implementation:

```text
side/top image
-> guidewire-head segmentation or color thresholding
-> contour/keypoint selection
-> vessel mask registration
-> nearest image-space wall distance
-> conservative binary contact threshold + confidence heuristic
```

Advantages:

- uses only currently available cameras;
- easy to implement and debug;
- matches the senior code direction closely;
- provides a deterministic fallback for safety/control;
- easy to reproduce in simulation with synthetic masks and projection noise.

Limitations:

- sensitive to lighting, color, glare, occlusion, and camera calibration;
- depends heavily on vessel-mask quality;
- may fail when the guidewire head is visually ambiguous;
- produces a thin signal that does not capture local visual context well.
- a single-view silhouette distance can look small even when 3D clearance is
  positive, so the binary contact flag should be stricter than the continuous
  distance signal.

Use this as:

```text
baseline estimator
fallback safety signal
source of weak labels / pseudo labels for later learned estimator work
```

Do not treat it as the final perception ceiling.

## Learned Visual Estimator

Candidate implementation:

```text
side/top image pair
-> lightweight segmentation/keypoint/contact network
-> tip keypoint + visibility + wall distance/contact + confidence
```

Possible heads:

```text
guidewire head heatmap
guidewire local direction
vessel wall / lumen mask
contact probability
image-space distance regression
visibility / uncertainty
```

Advantages:

- still uses only existing cameras;
- can be more robust to lighting, noise, color changes, and partial occlusion;
- can learn local context around the guidewire head rather than only thresholded
  colors;
- gives the future VLA a richer, structured perception layer while preserving
  raw images as input.

Risks:

- needs labeled or pseudo-labeled real/sim data;
- may overfit to synthetic rendering if trained only in simulation;
- confidence calibration matters, especially if the controller uses it for
  safety;
- a black-box contact classifier is harder to debug than a distance threshold.

Use this as:

```text
main future VLA perception path
drop-in replacement or enhancer for the rule-based estimator
```

## Simulation Alignment

The simulator should not directly feed exact contact truth to the policy.

For formal synthetic data, generate estimator-like outputs:

```text
synthetic_visual_distance_estimator:
  truth source: MuJoCo tip/wall geometry or rendered masks
  output: estimated_image_distance_px, estimated_contact_flag, confidence
  corruption: pixel noise, threshold noise, dropout, latency, false positives,
              false negatives, calibration bias
```

This keeps the synthetic signal close to what the real system can provide.

Paper-aligned next simulation target:

```text
rendered side/top images + synthetic depth/calibration metadata
-> estimator-like 2D tip detections
-> estimated 3D tip in camera/world/robot frame
-> registered route or branch-state estimate
```

These fields must be emitted as estimates with provenance, confidence, dropout,
latency, and coordinate-frame metadata. Exact MuJoCo tip/wall/route-frame truth
may be used to generate noisy synthetic labels, but should not be exposed as
formal policy input.

Current rule-based baseline:

```text
estimated_image_distance_px = median side/top projected image distance
estimated_contact_flag = 1 only when both visible side/top distances are below
                         the contact threshold
```

This makes the continuous distance the main VLA-facing signal and keeps the
binary flag as a conservative auxiliary contact indicator.

Recommended provenance example:

```json
{
  "estimated_contact_flag": {
    "tier": "estimator",
    "source": "synthetic_visual_distance_estimator",
    "truth_field": "tip_wall_distance",
    "noise_model": "pixel_threshold_with_dropout",
    "latency_steps": 1,
    "dropout_probability": 0.05,
    "formal_policy_input_allowed": true
  }
}
```

## VLA Interface Recommendation

The VLA should receive both raw observations and structured estimates:

```text
raw side/top images
Elite TCP 6D pose
Piper/controller state
task instruction
estimated tip/contact/distance/confidence fields
```

The structured fields should help the policy avoid learning all contact
semantics from pixels alone. Raw images should remain available so the model can
still use visual context that the estimator misses.

The controller should use confidence conservatively:

```text
low tip/contact confidence -> slow feed, hold, or request safer Elite motion
high contact probability -> reduce feed or trigger safe correction
```

This must be designed as a real-executable controller behavior, not as a
simulation-only oracle intervention.

## Validation Plan

Phase 1: rule-based estimator baseline

- recover or reimplement the current contour/mask distance pipeline;
- log contact flag, image distance, visibility, and confidence;
- compare side/top outputs on real videos if available;
- implement the same output fields in simulation with provenance.

Phase 2: paper-aligned 3D tip / coordinate wrapper

- convert side/top or RGB-D detections into estimated 3D guidewire-tip fields;
- attach camera/robot/virtual-frame metadata to each estimate;
- expose registration confidence and estimator failure modes;
- keep exact simulator coordinates diagnostic unless wrapped as estimated
  fields.

Phase 3: synthetic estimator wrapper

- add noise, latency, dropout, and threshold uncertainty;
- audit with `tools/audit_observation_provenance.py`;
- verify formal data does not consume exact MuJoCo contact/wall truth directly.

Phase 4: learned estimator prototype

- train on simulated images plus pseudo labels from geometry/masks;
- fine-tune or calibrate with real images if labels can be produced;
- compare against the rule-based estimator on real videos and simulation.

Phase 5: VLA/small-model interface test

- keep the same estimator output contract;
- test whether adding estimated tip/contact fields improves open-loop action
  plausibility before any expensive rollout;
- judge the result by real-observable validity first, not by rollout success
  alone.

## Current Recommendation

Use the rule-based image-distance estimator as the first implementation because
it is real-available and auditable. Treat it as a baseline and fallback, not the
final VLA perception solution.

After the accepted wallfix dataset, the next mainline estimator work should
extend the output contract toward paper-aligned estimated 3D tip and
route/branch state. Do not rely on the all-zero binary contact flag as the main
learning signal, and do not return to projection-based wall markers just to make
top-view fields non-missing.

The likely stronger contribution is a learned visual estimator that keeps the
same policy-facing output contract:

```text
same cameras, no added hardware, better tip/contact/distance estimates
```

This gives the project a credible innovation path while preserving the central
sim-to-real rule:

```text
real-observable estimator outputs > exact MuJoCo contact truth
```
