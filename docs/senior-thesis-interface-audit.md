# Senior Thesis Interface Audit

Last updated: 2026-06-28

## Purpose

This note records the project-relevant conclusions from a targeted read of the
local ignored thesis PDF:

```text
paper/硕士论文.pdf
title: 面向血管介入手术的半自主与自主磁驱动机器人系统研究
```

The goal is not to summarize the thesis. The goal is to extract real-system
interface evidence that should constrain simulation data, estimator design, and
future VLA/small-BC interface tests.

Highest priority:

```text
paper-backed real interface > inferred BC rollout convenience
```

## Real Autonomous System Interface

The autonomous dual-arm system described in thesis chapter 4 is organized as:

```text
perception -> decision -> execution -> feedback/monitoring
```

The execution roles match the current repository interpretation:

- Elite / EC66 carries the external permanent magnet and guides the magnetic
  guidewire head.
- Piper feeds the guidewire near the entry point through a gripper/step
  mechanism.

The policy input is explicitly multimodal:

```text
I_t: operative image
q_t: magnetic guidance arm 6D pose
c_t: accumulated Piper feed count
target branch / target position
```

The policy output is:

```text
a_L: 6D pose delta for the magnetic guidance arm
a_S: binary Piper action, 0=hold/stationary, 1=forward feed once
```

This matches the inherited code in `intervention/dual_bc`:

```text
image + pose + count + task_id -> large 6D regression + small Bernoulli action
```

and matches the real execution pattern:

```text
current Elite TCP pose + predicted translation delta
-> IK
-> move_joint(...)

Piper action 1
-> piper.step_forward(...)
```

## Consequences For Current Mainline

The thesis strengthens several existing project decisions:

1. `branchs/*/path/pose*.txt` should be treated as Elite/magnetic-arm TCP
   motion, not guidewire path ground truth.
2. Elite's real-aligned learning target should remain TCP pose delta followed
   by IK, not absolute joint regression as the semantic policy target.
3. Piper's senior-compatible formal action subset is `hold/feed`. A true
   retract action remains useful, but it is not present in the thesis BC
   interface and must be added as a real controller capability before it becomes
   a formal learned label.
4. The current `piper_feed_elite_joint` compatibility schema may remain for
   execution/logging, but formal learning should prefer:

```text
Elite: elite_tcp_delta_6d
Piper: piper_step_command in {0, 1} for senior-compatible hold/feed
```

## Estimator And Contact Evidence

The thesis chapter 3 perception stack is:

```text
vessel mask / wall segmentation
red magnetic guidewire-head localization
side/top visual distance or collision detection
optional tactile feedback generated from visual collision state
```

This validates the current no-new-hardware route:

```text
side/top image -> estimated tip/contact/distance/confidence
```

However, the thesis also goes beyond the current pixel-distance-only simulator
signal. It describes a real coordinate pipeline:

```text
2D guidewire-head localization
+ depth camera
-> camera-frame 3D tip coordinate
-> hand-eye / least-squares registration
-> robot-base or virtual-environment coordinate
```

Therefore, the next estimator route should not stop at
`estimated_image_distance_px`. The next formal estimator interface should move
toward estimated 3D tip and route-aware fields, with provenance and confidence:

```text
estimated_tip_pos_3d
estimated_tip_visible
estimated_tip_to_wall_px_or_mm
estimated_contact_flag
estimated_branch_or_route_state
registered_route_id
route_estimator_confidence
```

Exact MuJoCo `tip_pos`, wall distance, contact normal, or route progress must
remain diagnostic unless wrapped as estimator outputs with declared provenance
and realistic noise/failure modes.

## Controller And Safety Evidence

The thesis online autonomous flow includes more than BC inference:

- thresholding the Piper feed probability before executing a feed step;
- limiting or correcting the Elite target pose for safety;
- checking target arrival from estimated guidewire-tip position;
- detecting wrong-branch entry from vessel topology;
- stopping feed and triggering pullback/correction when wrong-branch entry is
  detected;
- emergency stop behavior for repeated safety failures.

This supports the current hybrid policy/controller split. The policy should not
own every safety and timing detail. The controller should own IK, motion limits,
Piper primitive timing, accepted/rejected command logs, and safety fallback.

The current simulator should treat wrong-branch detection and pullback as a
future controller/estimator capability, not as an oracle intervention unless it
uses a real-observable estimated tip and registered route.

## Timing Evidence

The thesis timing table indicates that neural inference is not the dominant
online bottleneck. The reported decision inference time is millisecond-scale,
while physical execution dominates the cycle time:

```text
decision inference: about 2.86 ms
Elite motion: about 1105.68 ms
Piper feed motion when executed: about 2007.47 ms
```

Implication:

```text
rollout/runtime slowness should be analyzed mostly as execution/controller
timing and rendering/simulation stepping, not as small-model inference cost.
```

## Recommended Next Work

Do not restart a BC rollout tuning loop from this thesis reading. The paper
supports the current decision to focus on real-observable estimator and
controller semantics.

Recommended next engineering direction:

1. Define a paper-aligned estimator output contract that extends the accepted
   wallfix image-distance candidate toward estimated 3D tip and registered
   route state.
2. Update simulation formal data so estimator fields can be emitted with
   provenance, confidence, dropout, and optional coordinate-frame metadata.
3. Keep the VLA/small-BC policy contract:

```text
raw side/top images
+ Elite TCP 6D pose
+ Piper feed count/controller state
+ estimated tip/contact/route fields
+ task instruction
-> Elite TCP delta
+ Piper hold/feed intent
```

4. Treat retract/pullback as controller-level safety/correction until a real
   retract primitive and corresponding labels are defined.

