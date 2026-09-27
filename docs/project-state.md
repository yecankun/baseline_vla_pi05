# Project State

Last updated: 2026-07-17

This file is the shared project-state entry point. It intentionally contains
only cross-track decisions, interface contracts, and routing. Detailed status
belongs in one of the two track handoffs:

```text
algorithm / VLA: docs/algorithm-track-handoff.md
data / simulation / real collection: docs/data-track-handoff.md
```

The pre-split mixed snapshot is preserved at
`docs/archive/project-state-before-track-split-2026-07-17.md`.

## Highest Priority

```text
simulation realism and real-system correspondence > BC rollout success
```

The project now runs as two parallel work tracks:

- Data track: simulation correspondence, synthetic-data validity, real
  collection, calibration, estimator provenance, and hardware-facing evidence.
- Algorithm track: OpenPI/PI05/LeRobot integration, pretrained loading,
  mixed action heads, context conditioning, and open-loop policy evaluation.

Algorithm work may proceed before simulator realism is perfect, but remains
feasibility/prototyping until stronger real data and alignment evidence exist.

## Shared Interface

Target policy contract:

```text
observation + task instruction -> Elite TCP delta + Piper discrete intent
```

Expected policy observations:

- side/top images;
- Elite pose/state;
- Piper state/history;
- task instruction;
- image-derived tactile/contact fields with confidence and provenance.

Controller-owned execution:

- Elite TCP delta -> IK -> joints;
- Piper intent -> hold/feed/retract primitive;
- controller owns timing, cooldown, bounds, safety, and executed-command logs.

Compatibility fields such as `state_32`, `action_32`, and
`piper_feed_elite_joint` may remain in adapters and datasets. They do not
replace the preferred project contract. Keep `elite_tcp_delta_6d` and
`piper_intent_id` explicit.

## Track Ownership

Algorithm track owns:

- model architecture and pretrained checkpoints;
- PI05/OpenPI/LeRobot adapters;
- algorithm losses and action heads;
- state/tactile model conditioning;
- open-loop model evaluation.

Data track owns:

- simulator and formal collector behavior;
- real collection and hardware procedures;
- dataset acceptance and label provenance;
- sim-to-real calibration and visual correspondence;
- estimator/contact field production.

Neither track may silently redefine the shared observation/action schema. A
schema change requires evidence, updates to both track handoffs, and a shared
contract update here.

## Current Shared Status

Algorithm track:

- PI05 architecture and loss wiring run on the Ubuntu 4090.
- Current checkpoints are feasibility artifacts, not real-system validation.
- The next blocker is verified loading of a reproducible pretrained
  `lerobot/pi05_base`; see `docs/algorithm-track-handoff.md`.

Data track:

- MuJoCo correspondence and real collection continue independently.
- Current onsite captures remain calibration/hardware-diagnostic data unless
  explicitly reclassified.
- Physical guidewire delivery remains the real collection bottleneck; see
  `docs/data-track-handoff.md`.

## Shared Stable Decisions

- MuJoCo remains the main simulator; Isaac Sim is a backup renderer/spike path.
- Formal data must not depend on hidden simulator truth without a matching real
  observable/control path.
- `branchs/` is real image and Elite/magnetic-arm motion reference, not
  guidewire trajectory or centerline truth.
- Tactile/contact policy inputs must be image/estimator derived, confidence
  aware, and provenance explicit.
- Small BC and current VLA results are interface/algorithm diagnostics, not
  proof of real readiness.
- Long collection, training, rollout, and full evaluation jobs are normally
  user-run from copyable commands.

## Documentation Routing

Shared entry and contract documents:

```text
AGENTS.md
.codex/skills/project-2026-guidewire-mujoco/SKILL.md
docs/project-state.md
docs/handoff.md
```

Choose exactly one working handoff after the shared entry:

```text
docs/algorithm-track-handoff.md
docs/data-track-handoff.md
```

The group-meeting fact log is intentionally shared:

```text
docs/weekly-meeting-log.md
```

Algorithm-only reusable commands belong in
`docs/algorithm-track-commands.md`. The older `docs/commands.md` remains the
data/general command catalog and historical aggregate; algorithm agents should
not read it broadly.

Historical detail:

```text
docs/archive/
```
