# Decision Log

Last updated: 2026-06-26

This document records high-impact decisions and why they were made. It should be
updated when the project changes action semantics, data strategy, or simulator
direction.

## Use Simulation Before Large VLA Training

Decision:

Build a simulation/data/control pipeline before attempting OpenVLA-style large
model training.

Reasoning:

- Real-robot data collection is expensive and requires lab access.
- Current available real data is limited.
- A large VLA model is unlikely to help until action semantics, observations,
  and task success criteria are reliable.
- Simulation enables controlled ablations and fast rollout diagnostics.

Rejected alternative:

Directly replacing the original baseline with OpenVLA-like architecture.

Reason:

The bottleneck is currently data and environment fidelity, not model capacity.

## Define The VLA Interface Before More Small-Model Tuning

Decision:

Use the small BC baseline only as a miniature test of the future VLA
observation/action contract.

Target contract:

```text
observation + task instruction -> Elite TCP delta + Piper discrete command
```

Reasoning:

- The project ultimately needs a VLA-style policy, not a bespoke small BC model
  with temporary simulation-only action semantics.
- Small BC rollouts are useful only if they test the same real-executable
  interface that the future VLA should use.
- The v10 TCP-delta experiments support using Elite TCP delta plus IK as the
  real-aligned Elite action direction.
- The v10 Piper experiments show that scheduled feed/hold labels are not
  reliably learned by the small model and should be made an explicit
  controller/policy interface decision.

Implication:

- Keep `piper_feed_elite_joint` as the compatibility/execution schema.
- Treat `elite_tcp_delta_6d` as the preferred Elite learning target when using
  senior-like TCP-pose observations.
- Define Piper as a real executable discrete command space:
  `hold/feed/retract`, with senior-compatible `hold/feed` as the minimum.
- Do not continue small-BC tuning unless the experiment tests this target
  interface.

Current interface note:

```text
docs/vla-target-interface.md
docs/control-layer-contract.md
```

## Use A Hybrid Policy/Controller Control Layer

Decision:

Use a hybrid control-layer contract for the VLA-facing path.

```text
policy:     Elite TCP delta + Piper hold/feed/retract intent
controller: IK, robot limits, Piper timing, cooldown, safety, executed command
```

Reasoning:

- Pure VLA low-level control would ask the model to learn robot timing and
  hardware safety from images, which is a poor real-system contract.
- Pure scripted Piper scheduling would remove useful policy responsibility for
  when to feed, hold, or retract based on visual/contact context.
- The inherited senior code already suggests this split: Elite action is a TCP
  pose delta followed by IK, while Piper execution is a concrete
  `step_forward(...)` primitive.

Implication:

- Train/evaluate policy labels as `elite_tcp_delta_6d` and Piper intent.
- Log executed Elite joints and Piper commands as controller outputs.
- Do not train the image model to infer hidden feed/hold schedule phases.
- Retract remains in the target interface only after a true real rollback
  command and labels exist.

## Prioritize Simulation Realism Over BC Rollout Success

Decision:

The highest execution standard is simulator realism and real-system
correspondence, not making a small BC model complete rollouts in simulation.

Reasoning:

- The project uses simulation because large-scale real data collection is
  difficult.
- Synthetic data is only useful if the simulator and expert/control semantics
  correspond to mechanisms that can exist on the real system.
- A BC rollout can be made to look better with privileged simulator state,
  expert forces, hidden centerline corrections, or tip-relative oracle
  controllers, but such data may harm sim-to-real transfer.

Implication:

- Treat BC rollout success as a regression check, not the main objective.
- Before scaling data, audit whether the expert policy uses real-observable
  inputs or simulation-only privileged state.
- Any controller such as `--elite-tip-anchor` is valid for formal data only if
  the real system will implement an equivalent guidewire-tip estimation and
  Elite low-level controller.

Current audit:

```text
docs/simulation-expert-validity-audit.md
```

## Use MuJoCo Before Isaac Sim

Decision:

Continue developing the lightweight MuJoCo environment before moving to Isaac Sim.

Reasoning:

- MuJoCo is faster to iterate.
- Current questions are mostly control semantics, data quality, and simplified
  guidewire behavior.
- Isaac Sim may be useful later for richer rendering/robotics integration, but
  it will not automatically solve guidewire modeling.

Rejected alternative:

Move immediately to Isaac Sim.

Reason:

The earlier Isaac Sim attempt already struggled with the guidewire. Moving now
would add simulator complexity before the task abstraction is stable.

## Keep Action Schema as `piper_feed_elite_joint`

Decision:

Current action format is:

```text
piper_feed + elite_joints
```

Where:

- `piper_feed` is currently one normalized simulator-side scalar feed/retract
  command.
- `elite_joints` are six Elite joint targets.

Reasoning:

- Piper's real role is repeated local feeding near the entrance, not moving with
  the guidewire tip.
- The current `piper_feed` scalar is not yet aligned with senior's real Piper
  BC interface. `intervention/bc_piper/inference.py` uses binary actions:
  `0=stop/hold`, `1=piper.step_forward(...)`, with no negative/retract action.
  Retract is still a reasonable target capability; it just needs an explicit
  real command/interface before being treated as real-aligned formal data.
- The senior's original models/data appear to use joint-level robot commands.
- Keeping Elite as six joint targets preserves a clearer sim-to-real path.

Rejected alternative:

Command Elite by end-effector or magnetic target position.

Reason:

It may make simulation control easier, but it changes the action interface away
from the likely real robot control format.

## Separate Piper Feeder Reset From Real Retract

Decision:

Keep the candidate real Piper label space as:

```text
-1 = retract
 0 = hold
+1 = feed
```

but do not label every negative simulator `piper_feed` as `retract`.

Reasoning:

- The current simulator uses a finite Piper feeder stroke. Route-plan expert
  data can contain periodic negative `piper_feed` values that reset or reposition
  that simulated feeder stroke.
- The current wire integration clamps negative Piper command to zero, so those
  negative strokes do not represent true guidewire pullback.
- Treating simulator feeder reset as real retract would create a misleading
  sim-to-real action label. The smoke run
  `simulation_output/formal_route_plan_tip_signedpiper_smoke_v1` exposed this:
  it passed formal/environment metrics, but its initial labels alternated
  feed/retract on nearly every saved sample.

Implication:

- `piper_feed` remains the simulator-side continuous execution value for
  backward compatibility.
- `piper_step_command` is the real guidewire-level command label. Current
  route-plan negative reset strokes should map to `0=hold` unless an expert
  explicitly emits true retract.
- Retract is still allowed in the future real interface, but it must be defined
  and labeled as actual rollback/retraction, not inferred from simulator reset.

## Treat Senior Trajectories as Alignment Reference, Not Guidewire Path

Decision:

Data under `branchs/` should not be used as direct guidewire path supervision.

Reasoning:

- The trajectory is likely the Elite/magnetic-arm path.
- The magnetic tool moves above/near the guidewire, not exactly on the guidewire.
- The current simulation scene is manually calibrated and does not perfectly
  match the real setup.

Use it for:

- rough vessel/robot placement sanity checks;
- understanding real task geometry.

Do not use it for:

- forcing the guidewire centerline;
- defining `elite_reference_path` in the simulated environment.

## Hide Debug Visuals in Formal Data

Decision:

Formal datasets should hide tool markers and path tubes.

Reasoning:

- Markers and reference paths are debugging aids.
- If visible in training images, the model may learn visual shortcuts that do
  not exist in real camera images.

Practical collection flags:

```text
--hide-tool-markers
--hide-path-tubes
```

## Do Not Blindly Reuse High Piper Negative Weight

Decision:

Do not reuse `piper_feed_negative_loss_weight=4.0` as a default.

Reasoning:

- It improved Piper rollback.
- It made Elite joint output less smooth in closed-loop rollout.
- It produced much larger `elite_target_joint_step_linf` p95 values.

Preferred next options:

- Train unweighted on newly collected data first.
- If Piper rollback is insufficient, try lower weight such as `2.0`.
- Add Elite temporal smoothness loss before using high Piper-specific weighting.

## Anchor Elite Smoothing to Real Control Bandwidth

Decision:

Do not continue improving Elite video jitter by repeatedly adding arbitrary
smoothness weights or low-pass filters. Future smoothing, rate limits, and
execution interpolation should be justified as approximations of real
Elite/magnetic-arm control bandwidth.

Current real anchor:

```text
simulation_output/real_control_bandwidth
position step p95 median: 5.072 mm/frame
position acceleration p95 median: 2.982 mm/frame^2
position jerk p95 median: 2.835 mm/frame^3
direction reversal fraction median: 0.0000
```

Reasoning:

- The project needs simulation data that can transfer back to real hardware.
- A smooth-looking controller can be harmful if it represents behavior the real
  robot would not execute.
- `rate010_smooth025_r1` matched the real step scale numerically but still
  looked jittery, which means step-size p95 alone is not sufficient.
- Naive delta training reduced target-jump p95 but allowed unacceptable visible
  Elite tool drift.

Preferred next options:

- Add sim rollout bandwidth diagnostics against the real report.
- Then implement a joint velocity/acceleration/jerk-limited execution layer
  while preserving the external `piper_feed_elite_joint` schema.

## Test Tip-Centric Guidewire Modeling

Decision:

Add and evaluate a parallel tip-centric guidewire model before investing more
effort in full polyline guidewire physics.

Reasoning:

- The real guidewire material has been confirmed to be relatively hard and
  elastic, closer to a springy wire than to a highly floppy cable.
- In the real task, the most important observable/contact event is whether the
  magnetically guided head touches the vessel wall.
- If the head stays inside the vessel and does not collide, the stiffer trailing
  wire is much less likely to become the dominant contact source.
- Previous attempts to tune full polyline stiffness, damping, curvature, and
  wall interaction consumed substantial effort and still produced unrealistic
  contact and stalled progress.

Implementation:

```text
simulation/tip_guided_wire_env.py
tools/smoke_tip_guided_wire.py
```

Status:

The prototype is a parallel route, not yet a replacement for the polyline
environment. Compare expert data quality first, then decide whether to connect
it to formal data collection and small-BC regression tests.
