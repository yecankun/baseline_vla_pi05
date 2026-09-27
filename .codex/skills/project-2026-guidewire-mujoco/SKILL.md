---
name: project-2026-guidewire-mujoco
description: Stable workflow and evidence guardrails for the Project 2026 guidewire intervention repository. Use for work in D:\PycharmProjects\project_2026 involving simulation, synthetic or real data, Elite/guidewire feed control, contact estimation, policy interfaces, rollout diagnostics, hardware collection, or project handoff documentation.
---

# Project 2026 Guidewire Workflow

## Role Of This Skill

Treat this file as a stable operating contract, not a project-state snapshot.
Keep experiment results, current parameter values, model choices, active
entrypoints, and temporary next steps in the track handoffs and command docs.

Follow the user's current direction and the repository's `AGENTS.md` when an
older statement here has become obsolete. Update this skill when a durable
workflow principle changes instead of silently accumulating exceptions.

## Start With One Track

Read the shared routing set before non-trivial work:

```text
AGENTS.md
docs/project-state.md
docs/handoff.md
```

Select one primary detailed handoff, or use shared maintenance / read-only
review for tasks spanning the workflow:

```text
algorithm / VLA:                  docs/algorithm-track-handoff.md
data / simulation / collection:  docs/data-track-handoff.md
SOFA / robot co-simulation:      docs/sofa-robot-track-handoff.md
```

Do not load both detailed tracks in full by default. Read relevant sections
of either track for shared maintenance or necessary dependency inspection.
Cross-track edits require authorization for that scope, which need not be
requested again when already given. Use `rg` for narrow historical retrieval.

Open deeper documents only when the task requires them:

- Algorithm commands: `docs/algorithm-track-commands.md`
- Data and simulation commands: `docs/commands.md`
- SOFA / robot commands and implementation history:
  `docs/sofa-robot-track-commands.md`, `docs/sofa-robot-track-progress.md`
- Shared schema and controller contracts: `docs/data-and-action-schema.md`,
  `docs/vla-target-interface.md`, `docs/control-layer-contract.md`
- Real collection and calibration: `docs/real-data-collection-checklist.md`,
  `docs/real-alignment-20260707-calibration-audit.md`
- Estimator and contact provenance: `docs/estimator-tactile-interface-spec.md`,
  `docs/visual-tip-contact-estimator-plan.md`
- Formal validity: `docs/real-observable-interface-audit.md`,
  `docs/simulation-expert-validity-audit.md`
- Historical snapshots: `docs/archive/`

## Delegation And Maintenance Boundaries

When using subagents, assign one bounded responsibility with explicit read/write
and do-not-touch paths. Necessary cross-track reads are allowed; cross-track
writes must fit the user's authorized scope. Preserve concurrent changes.

Choose model and reasoning effort in proportion to the task. For continuous
work using the same model and effort, inheritance and a full-history fork are
allowed. Override settings only when a different configuration is useful or
requested; use a tool-supported combination and `fork_turns=none` or a positive
bounded number, with a self-contained task prompt. Follow the tool's fork and
model constraints and any explicit user model choice. Reassess settings when
reusing an agent, without recreating it solely to restate unchanged settings.

For each invoked subagent, report its canonical task name, actual model and
effort when known, fork mode, and final status or concise outcome, including
failures and interruptions. Record inherited settings when available; otherwise
say inherited / unverified rather than guessing an exact value.

For cleanup recovery, protected inputs, and action-specific stop conditions,
use [Git Hygiene](../../../AGENTS.md#git-hygiene).

## Track Ownership

Use [Track Ownership](../../../AGENTS.md#track-ownership) for responsibilities
and [Cross-Track Change Protocol](../../../docs/project-state.md#cross-track-change-protocol)
for shared observation/action changes.

## Stable Policy And Control Boundary

Use the authoritative [Shared Policy Interface](../../../docs/project-state.md#shared-policy-interface)
and [Controller Boundary](../../../docs/project-state.md#controller-boundary).

## Observability And Provenance

Use [Observation And Target Provenance](../../../docs/project-state.md#observation-and-target-provenance)
for formal-data and deployable policy/expert/collector input restrictions.

For every estimated field used online, record its source, confidence, timing,
validity, and whether policy input is allowed. Keep diagnostic targets separate
from policy samples. Treat `branchs/` as real camera and Elite-motion reference,
not guidewire-path truth.

Experts and collectors may evolve before precise real statistical calibration
exists. Their formal-data or deployable decisions follow the shared provenance
contract, including the diagnostic-oracle exception. Use uncalibrated
response variants as coverage-oriented diagnostic scenarios, not as claimed
real probabilities or measured physical bounds. Real evidence is needed for
formal sim-to-real claims, but it is not a prerequisite for diagnostic expert
iteration. Diagnostic oracles follow the explicit isolation and authorization
boundary in [Expert And Scenario Evolution](../../../docs/project-state.md#expert-and-scenario-evolution);
their privileged behavior is not formal expert data.

## Diversity Integrity

Apply the dataset-independence contract in
[Expert And Scenario Evolution](../../../docs/project-state.md#expert-and-scenario-evolution).
The following detailed inspection requirements remain here:

- record `source_trajectory_id`, scenario family, sampled parameters, and
  calibration status;
- audit action transitions, trajectory overlap/shared suffixes, contact
  location and timing, and meaningful visual differences;
- reject or discount repeated trajectories instead of multiplying their
  statistical weight.

Different seeds, tiny render noise, camera augmentation, or different
truncations of one deterministic route do not establish behavior diversity.
Avoid hidden randomized expert styles that map the same observation to
conflicting actions; behavior differences must follow observable state or
explicit context.

## Evidence Standard

Apply [Execution And Verification](../../../AGENTS.md#execution-and-verification)
for core-function-first development, minimal risk-proportionate checks, visual
artifact requirements, the three visual acceptance states, and completion while
user acceptance is pending.

Generate user-facing figures in Chinese by default, including titles, labels,
legends, annotations, captions, and conclusion text for visual review and group
meetings. Preserve literal technical identifiers such as schema keys, action
fields, filenames, paths, commands, and gate or stage IDs when translation
would reduce traceability. Use a Chinese-capable font, inspect the rendered
artifact for missing glyphs, clipping, and overlap, and provide the user a
directly viewable image or video link. Use another language only when the user
explicitly requests it or an external submission format requires it.

Separate raw policy or expert behavior from guarded controller behavior. A
successful simulation rollout, diagnostic oracle, or aggregate estimator score
does not establish formal training readiness or real-system validation.

## Working Procedure

1. Determine the primary track or shared-maintenance scope and current handoff
   state from the request and files. Ask only when a material ambiguity remains.
2. Inspect existing code, tools, manifests, and narrow historical evidence.
3. Make the smallest change that tests the current hypothesis.
4. Apply the five-minute runtime and existing-authorization rule in `AGENTS.md`.
   Complete entrypoints, parameters, output/recovery instructions, and necessary
   short checks before handing user-run jobs over.
5. Inspect and deliver the evidence appropriate to the change under AGENTS.md;
   record implementation, verification, and user visual acceptance separately.
6. Update only the documents that own the durable conclusion.

Use the local interpreter and runtime rules in
[Execution And Verification](../../../AGENTS.md#execution-and-verification).

Before remote execution, synchronize the relevant final changed version with
sequential `scp` and confirm the intended remote entrypoint/version with a
lightweight readback. Per-file hashes are not required by default; follow the
minimal-verification rule in `AGENTS.md` when an integrity check is needed.
Do not synchronize merely after each keystroke or intermediate edit. If any
single file or prepared archive is larger than 100 MB, give exact source and
destination paths and let the user transfer it; do not split it to evade this
boundary. Transfer size and the five-minute computation rule are separate.

## Documentation Ownership

Use [Documentation Ownership](../../../AGENTS.md#documentation-ownership).
This skill contains reusable procedures, not experiment history.

## Hardware And Git Caution

For onsite work, verify the active hardware backend and connection before
diagnosing application code. Treat unstable delivery trials as hardware or
calibration evidence unless explicitly accepted as clean demonstrations.

For commits, synchronization milestones, and excluded artifacts, use
[Git Hygiene](../../../AGENTS.md#git-hygiene).
