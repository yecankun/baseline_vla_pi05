# Project Instructions

This repository combines guidewire simulation, real-camera perception, robot
control, synthetic and real data collection, contact estimation, and policy
experiments. Keep this file limited to durable repository-wide rules. Put
current experiments, parameters, model choices, and next steps in the track
handoffs.

## Rule Ownership

For the consolidated rules, edit the owning text rather than its references:

- Execution, track responsibilities, documentation, and Git: this file.
- Shared policy/controller semantics, diagnostic-oracle isolation, and dataset
  independence: `docs/project-state.md`.
- Workflow, delegation, and detailed inspection procedures: the project skill.
- Track selection and links: `docs/handoff.md`.
- Current experiments and commands: the owning track handoff and command file.

This index does not settle differing clauses that remain in place. Their
quoted sources and proposed resolutions are in
[Instruction review decisions](docs/instruction-consolidation-review-20260910.md).
That review records decisions and the remaining D4 proposal; it is not an
additional execution instruction or approval gate.

## Source Of Truth

For non-trivial work, read:

```text
AGENTS.md
.codex/skills/project-2026-guidewire-mujoco/SKILL.md
docs/project-state.md
docs/handoff.md
```

Then select one primary track, or shared maintenance / read-only review:

```text
algorithm / VLA:                  docs/algorithm-track-handoff.md
data / simulation / collection:  docs/data-track-handoff.md
SOFA / robot co-simulation:      docs/sofa-robot-track-handoff.md
```

Use the matching command document. For shared maintenance, read only the
relevant sections of either track. Necessary cross-track dependency inspection
is read-only by default; user-authorized cross-track edits stay within the
named scope and preserve concurrent work. Search narrowly for historical
evidence; old snapshots belong under `docs/archive/`.

The user's current direction supersedes obsolete project assumptions. When a
durable rule changes, update the owning document instead of silently adding an
exception.

## Project Priority

```text
simulation realism and real-system correspondence > rollout appearance
```

This priority does not require waiting for perfect real calibration before
running diagnostic simulation or algorithm experiments. Experts, collectors,
and policies may be prototyped with declared assumptions. Do not present their
results as formal training readiness or real-system validation without the
required provenance and evidence.

## Track Ownership

Algorithm track owns:

- model architecture, losses, adapters, checkpoints, and training;
- policy conditioning and open-loop or rollout evaluation;
- algorithm-only experiment conclusions and commands.

Data track owns:

- simulation environments, experts, collectors, and scenario generation;
- dataset acceptance, split integrity, label and estimator provenance;
- sim-to-real alignment, real collection, and hardware procedures.

SOFA / robot co-simulation track owns its independent SOFA guidewire backend,
MuJoCo robot bridge, fresh entrance initialization, and their diagnostic
contracts, verification, and commands. Historical finite-tip experiments and
formal data acceptance remain in the data track. Record every implementation
milestone and its validation or failure in the append-only SOFA track progress
log, with input and code provenance and outstanding work.

No track may silently redefine the shared observation/action interface.
Coordinate cross-track changes explicitly and update the shared contracts.
Existing user authorization for the same scope remains valid; do not request
it again merely because work crosses a track boundary.

## Shared Interface

The authoritative policy and controller contract is in
[Shared Policy Interface](docs/project-state.md#shared-policy-interface) and
[Controller Boundary](docs/project-state.md#controller-boundary).

## Observability And Data Integrity

Formal-data and deployable policy/expert/collector input restrictions follow
[Observation And Target Provenance](docs/project-state.md#observation-and-target-provenance).

Diagnostic oracle inputs, output isolation, and authorization are defined in
[Expert And Scenario Evolution](docs/project-state.md#expert-and-scenario-evolution).

Estimated online fields require explicit source, confidence, timing, validity,
and policy-allowance metadata. Keep policy samples separate from diagnostic
targets. Treat `branchs/` as real image and Elite-motion reference, not
guidewire-path truth.

Real statistical bounds are not a prerequisite for diagnostic expert or
collector improvements. Uncalibrated variations must be framed as
coverage-oriented scenario families, not measured real distributions or event
probabilities. Real evidence later constrains plausibility and any formal
sim-to-real claim.

Dataset independence and split integrity follow
[Expert And Scenario Evolution](docs/project-state.md#expert-and-scenario-evolution).
The project skill retains the detailed diversity inspection checklist.

## Repository Boundaries

- `simulation/`: simulation, experts, collectors, training, and rollout code.
- `data/collect/`: real collection code.
- `tools/`: diagnostics, validation, calibration, conversion, and visualization.
- `hardware/`: hardware adapters and probes.
- `intervention/`: inherited model code.
- `robot_assets/` and `utils/interface/model/`: robot and vessel assets.
- `branchs/`: senior-provided real reference data with limited semantics.
- `collected_data/`: local real captures; not clean expert data unless accepted.
- `simulation_output/`: generated artifacts; not durable documentation.

Preserve raw external or inherited code when practical. Integrate through
adapters rather than silently rewriting its semantics.

## Execution And Verification

以当前任务的核心功能交付和端到端跑通为优先，避免与当前风险无关的过度防御性编程、
兼容层和复杂兜底。验证保持最小必要范围，优先复用已有检查；尽量减少 smoke test，
不为每个文件或每次小改动重复新增、运行测试。SHA-256 仅用于确有必要的完整性、
版本一致性或明确要求的工件身份校验，不例行逐文件计算。实机安全、关键动作单位与
限幅检查、破坏性操作保护仍须保留。该原则不改变执行授权和结果表述的证据要求。

Use `.\.venv\Scripts\python.exe` for local Windows commands unless the active
handoff specifies another machine or environment.

Estimate runtime from workload, prior timings, and the execution environment.
A computation or evaluation expected to exceed five minutes is user-run by
default: provide exact commands unless the user has already authorized the
agent to execute that task. Execute necessary checks expected to take at most
five minutes within the current authorized scope. Estimate one complete logical
job, not split commands; reading, implementation, and reporting time do not
count toward this runtime threshold. If a run unexpectedly exceeds five
minutes, report progress and use its safe stop/resume mechanism when needed;
do not terminate solely because the estimate was exceeded. Explicit user-run
assignments and separate hardware, training, or formal-data gates still apply.

Before handing a run to the user, finish the runnable entrypoint, parameters,
output paths, recovery instructions, and applicable short checks. After the
run, inspect its logs, manifests, metrics, and representative visual evidence.

Choose verification to match the change. Changes to generated behavior or
appearance require a directly viewable representative image, comparison sheet,
or video. Documentation-only changes and schema-only fixes do not require an
unrelated render or full episode. Report zero-step or short-window evidence as
such; schedule full episodes when the behavioral risk requires them, under the
runtime rule above. Inspect relevant metrics, provenance, execution, and failures.

Record visual status as `not_viewed`, `viewed_not_accepted`, or `accepted`, with
the user's feedback and the artifact/scope it applies to. Viewing alone, agent
review, and automated gates do not establish acceptance. Mark `accepted` only
when the user explicitly approves that result; a criticism remains
`viewed_not_accepted`. Acceptance of one artifact does not accept later variants.

Track implementation completion, verification completion, and user visual
acceptance separately. Pending visual acceptance blocks only actions explicitly
dependent on it, not authorized checks, artifact organization, commands, or
factual documentation updates. Finish those deliverables before handing off
the remaining acceptance or user-run step; do not claim that step is complete.

Use available fail-closed formal and provenance audits. A successful simulation
episode, diagnostic oracle, aggregate estimator score, or guarded rollout is
not sufficient evidence by itself.

## Documentation Ownership

- Algorithm state: `docs/algorithm-track-handoff.md`
- Algorithm commands: `docs/algorithm-track-commands.md`
- Data/simulation/real state: `docs/data-track-handoff.md`
- Data and simulation commands: `docs/commands.md`
- SOFA / robot state: `docs/sofa-robot-track-handoff.md`
- SOFA / robot commands: `docs/sofa-robot-track-commands.md`
- SOFA / robot implementation history: `docs/sofa-robot-track-progress.md`
- Shared interface and routing: `docs/project-state.md`, `docs/handoff.md`
- Group-meeting facts: `docs/weekly-meeting-log.md`

Update the smallest necessary set of owning documents for durable changes.
Shared-interface changes follow `docs/project-state.md` / Cross-Track Change
Protocol, including affected contracts and handoffs; record only the interface
effect in the other track. The weekly log is
append-only factual evidence, not a polished summary. Do not copy one track's
history into the other track's handoff, and do not turn routing documents into
experiment logs.

## Git Hygiene

Use GitHub as a stable rollback and synchronization point, not for every trial.
Commit or push meaningful code, contract, and documentation milestones. Avoid
committing `simulation_output/`, raw captures, local environments, large data,
or experiment debris.

Before project cleanup, inventory the intended targets and establish a
verifiable recovery point: either reachable Git content covering the targets or
a complete backup of those targets with verified contents and recovery paths.
Preserve untracked recovery inputs as well. Record the current next step and
inventory import/document/contract references before removal. Do not classify
code as unused from experiment number, age, or filename alone. If a proposed
removal requires broad behavior refactoring or unavailable regression evidence,
stop that removal/refactoring and continue safe inventory and reporting. Archive
dated documentation before slimming an active entry document. Do not use `git clean`, `git reset --hard`, or
age-based bulk deletion as a cleanup shortcut. A generated or untracked file
referenced by the current handoff, manifest, contract, or checkpoint remains
protected. Track-local cleanup must not alter the other track's documents,
training artifacts, checkpoints, or concurrent uncommitted work without
explicit user or owning-track authorization.
