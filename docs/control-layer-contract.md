# Control Layer Contract

Last updated: 2026-06-26

## Decision

Use a hybrid policy/controller split:

```text
policy output:
  Elite TCP delta + Piper discrete intent

controller responsibility:
  IK, command timing, velocity/acceleration/jerk limits, step execution,
  hardware safety, and command rejection/fallback
```

This is the mainline control contract for future VLA-facing simulation data and
small-BC interface tests.

## Why Hybrid

Rejected extremes:

- Pure VLA low-level control: asks the model to learn robot timing, step
  cadence, and hardware safety from images. That is not a good simulation-to-real
  contract.
- Pure scripted Piper scheduling: removes the model's ability to decide whether
  guidewire feeding should pause or retract based on visual/contact context.

The hybrid split keeps the learned policy responsible for intervention decisions
while keeping robot mechanics in a real-implementable controller.

## Policy Interface

Target policy input:

```text
side/top images
Elite TCP 6D pose
Piper state
task instruction
optional real controller state
optional real estimator/tactile signals
```

Target policy output:

```text
elite_tcp_delta_6d
piper_intent
```

Where:

```text
elite_tcp_delta_6d = [dx_mm, dy_mm, dz_mm, drx_rad, dry_rad, drz_rad]
piper_intent       = hold | feed | retract
```

Current Elite labels are translation-first: the policy predicts translation
delta and the controller keeps orientation unchanged unless a later real-control
need justifies orientation deltas.

Minimum senior-compatible Piper subset:

```text
hold | feed
```

Retract remains in the target interface, but it is formal only after a true
real rollback command and labels exist. Do not infer retract from simulator
feeder reset.

## Elite Controller

The Elite execution layer owns:

```text
current TCP pose + elite_tcp_delta_6d
-> target TCP pose
-> inverse kinematics
-> joint target
-> velocity/acceleration/jerk and hardware limits
-> move_joint or equivalent real command
```

The policy should not output absolute joint targets as the preferred semantic
action when the observation is senior-style Elite TCP pose. `elite_joints` may
remain in logs and datasets as the execution/diagnostic representation after
IK.

Allowed controller safeguards:

- cap TCP translation delta;
- keep orientation fixed by default;
- reject IK failures and hold last safe target;
- apply joint velocity/acceleration/jerk limits grounded in real robot behavior;
- log both requested target and executed target.

Formal training data should clearly separate:

```text
policy label: action.elite_tcp_delta_6d
execution log: action.elite_joints or executed Elite joints
```

## Piper Controller

The Piper execution layer owns hardware timing.

Policy intent:

```text
hold    -> request no guidewire feed motion
feed    -> request one forward step or a bounded forward feed primitive
retract -> request one rollback step or bounded retract primitive, if available
```

Controller state machine:

```text
idle
feeding
retracting
cooldown
blocked_or_fault
```

The controller may ignore or delay an intent when hardware is busy, in cooldown,
at insertion bounds, or blocked by safety checks. In that case it should expose
the rejection/fallback state to logs and, when useful, to the next policy
observation.

Required logged fields:

```text
piper_intent
piper_executed_command
piper_motion_state
piper_step_count or insertion_length
piper_busy / cooldown state
```

Current simulator compatibility fields:

```text
piper_step_command = -1 | 0 | 1
piper_command_label = retract | hold | feed
piper_feed = continuous execution magnitude used by MuJoCo compatibility code
```

Formal interpretation:

- `piper_step_command` / `piper_command_label` are the policy-level intent
  labels.
- `piper_feed` is an execution magnitude for simulator compatibility.
- Periodic simulator feeder reset must not become `retract` unless it is true
  guidewire rollback.

## Controller State As Observation

Controller state may be policy input only if the real system can provide the
same state.

Allowed examples:

- current Elite TCP pose;
- Piper step count;
- insertion length or actuator position;
- whether Piper is idle, feeding, retracting, or cooling down;
- last executed Piper command;
- task id/instruction.

Not allowed as formal policy inputs unless a real estimator/sensor path exists:

- exact MuJoCo guidewire-tip position;
- exact vessel wall distance;
- exact contact/wall penetration;
- exact route-frame progress from simulator truth;
- hidden route-plan phase that does not exist in the real controller.

If a deterministic Piper scheduler is used, its phase must be an explicit real
controller state. Otherwise it must stay outside the learned target instead of
being a hidden label pattern the image model has to guess.

## Dataset Rule

Formal synthetic data should record the same policy/controller split:

```text
action.elite_tcp_delta_6d       # policy label
action.piper_step_command       # policy intent label
action.piper_command_label      # readable intent
action.piper_feed               # simulator execution magnitude
controller.*                    # executed command/state, when available
```

Small BC/VLA-interface tests should train on the policy labels and use
controller logs for diagnostics, not as privileged inputs.

## Next Engineering Step

Implement or verify a simulator-side controller adapter that enforces this
contract:

1. accept `elite_tcp_delta_6d` and `piper_step_command`;
2. execute Elite through TCP delta -> IK -> joint limits;
3. execute Piper through discrete intent -> bounded feed/hold/retract primitive;
4. log requested intent, executed command, and controller state;
5. fail `--formal-data` / `--formal-eval` if privileged simulator-only state is
   required by the policy.
