# Shared Handoff Router

Last updated: 2026-07-17

This file routes a new agent to one work track. It intentionally does not carry
simulation, collection, hardware, or PI05 experiment detail. The pre-split
mixed handoff is preserved at
`docs/archive/handoff-before-track-split-2026-07-17.md`.

## Select One Track

Algorithm innovation / VLA:

```text
docs/algorithm-track-handoff.md
docs/algorithm-track-commands.md
docs/vla-pi-baseline-plan.md
```

Use this track for OpenPI, PI05, LeRobot, pretrained checkpoints, mixed action
heads, state/tactile conditioning, and open-loop policy evaluation. Do not
modify simulation collectors, real collection, hardware procedures, or dataset
semantics.

Data / simulation / real collection:

```text
docs/data-track-handoff.md
docs/commands.md
```

Use this track for simulator correspondence, formal synthetic data, estimator
provenance, real collection, calibration, and hardware-facing diagnostics. Do
not modify PI05/OpenPI model architecture or algorithm experiment conclusions.

Do not load both detailed handoffs by default. Read the other track only when
a concrete shared-interface issue requires coordination.

## Shared Contract

```text
observation + task instruction -> Elite TCP delta + Piper discrete intent
```

Keep the following explicit across both tracks:

```text
side/top images
Elite pose/state
Piper state/history
task instruction
image-derived tactile/contact fields with confidence/provenance
elite_tcp_delta_6d
piper_intent_id
```

Controller execution remains outside the policy. Compatibility tensors are not
permission to redefine robot controls.

## Shared Guardrails

- Simulation realism and real-system correspondence remain higher priority than
  BC/VLA rollout appearance.
- Algorithm results remain feasibility until real-system evidence improves.
- Data collectors and labels are owned by the data track.
- Model architecture and losses are owned by the algorithm track.
- Cross-track schema changes must update both handoffs and this shared router.
- Long jobs are normally handed to the user as copyable commands.

## Machine Split

```text
Windows local: D:\PycharmProjects\project_2026
Windows Python: .\.venv\Scripts\python.exe
Ubuntu 4090 SSH alias: project4090
Ubuntu project: /home/zsw/project_2026
Ubuntu conda env: project2026-pi
```

LeRobot/OpenPI execution belongs on the Ubuntu 4090. Local simulation and data
inspection remain on Windows unless the user explicitly changes the setup.

## Documentation Ownership

- Algorithm durable state: `docs/algorithm-track-handoff.md`
- Algorithm commands: `docs/algorithm-track-commands.md`
- Data durable state: `docs/data-track-handoff.md`
- Shared interface/routing: `docs/project-state.md`, `docs/handoff.md`
- Shared group-meeting facts: `docs/weekly-meeting-log.md`

Do not copy track-specific experiment histories into the shared router. Update
the smallest owned document, and use the weekly log only for durable facts with
reporting value.
