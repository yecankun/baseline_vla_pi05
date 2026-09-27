# VLA Baseline And Robustness Benchmark Plan

Last updated: 2026-09-13

## Current Execution Order (2026-09-13)

Baseline results first; world-model research is paused by the user's current
direction. The candidate below remains a future hypothesis, not an active run.

1. Reproduce the trained official Diffusion Policy checkpoint on native Push-T
   after pinned acquisition, exact normalization migration, strict load and a
   short rollout. Random-policy interface smokes do not fill this result row.
2. Prepare BC/ACT on the same Push-T task and evaluation initial conditions.
   Separate public pretrained reproduction from controlled same-data/split/
   budget training. The official DP checkpoint lacks a pinned training-dataset
   revision; shared evaluation alone does not control its training provenance.
3. Keep LIBERO-Spatial and the existing PI05 results in a separate table. Add
   another model only with compatible observable inputs and trained weights;
   do not silently trim state or synthesize a missing camera for SmolVLA.
4. Once clean results are reliable, run frozen paired corruption tests; only
   then resume world-model work against the resulting baseline and failure
   cases. This does not authorize inspecting R0 or modifying project datasets.

Every scored run needs a frozen checkpoint, normalization, task/init schedule,
horizon, inference randomness, action handling and clean metric definition.
Report episode counts and uncertainty, plus any protocol deviations. Different
domains are not a single architecture leaderboard. Concrete acquisition and
execution gates are in the algorithm handoff and command document.

## Decision

Use a two-axis experiment instead of waiting for the project simulator to
become a formal benchmark:

```text
architecture ladder: weak BC -> ACT -> Diffusion Policy -> SmolVLA -> PI0.5
evaluation domains:  project D0/R0 diagnostics + clean/noisy public benchmarks
```

OpenVLA-OFT is excluded by the current user decision. The algorithm innovation
candidate remains PI0.5 candidate generation plus the action-conditioned
short-horizon world model in `docs/vla-action-effect-world-model-plan.md`.

The public benchmark axis answers whether an implementation is executable and
whether its visual robustness mechanism works in a reproducible environment.
It does not establish guidewire capability. D0/R0 answer project relevance but
remain diagnostic until their owning data gates pass.

`D0` and `R0` below are algorithm-benchmark aliases from the current
discussion. They must not be confused with the data track's D0-D5 physical
cable integration stages.

## Baseline Ladder

1. Temporal and constant controls: previous action, previous Piper label, and
   training-set mean.
2. Small BC: reproduce the senior-style lightweight BC architecture on the
   same accepted data and split as the other project models. The historical
   senior dataset and its branch-only success criterion remain excluded from
   current training, normalization, validation, and model selection.
3. ACT: lightweight action-chunking baseline.
4. Diffusion Policy: established continuous action-generation baseline.
5. SmolVLA: smaller pretrained language-conditioned VLA.
6. PI0.5: large pretrained VLA baseline using the already pinned checkpoint.
7. PI0.5 plus action-effect world model: project innovation candidate, tested
   only after the plain ladder and data gates are stable.

The historical senior BC result may be shown as context, but it is not a fair
row in the controlled table unless the architecture is rerun on the current
accepted split. No historical image is mixed into current data.

## Evaluation Domains

### Project D0

Use the existing 1005-record stride-5 simulation view with its fixed 34/6
episode split. Report the existing project metrics: Elite translation MAE,
transition/steady MAE, Piper accuracy/balanced accuracy, transition/steady
accuracy, and hold/feed confusion matrix. D0 remains a diagnostic translation
plus binary hold/feed benchmark, not a formal simulator benchmark.

### Project R0

Reserve the current ten real episodes for a future data-audited low-data
diagnostic. Do not open, score, normalize from, or train on those records in
this architecture gate. When the data track accepts an interface, keep whole
episodes and left/right balance in the fold split. R0 will be offline evidence;
it cannot provide closed-loop success or real-system validation by itself.

### Push-T

Use Push-T as the cheapest public closed-loop pipeline check. It has one image,
a two-dimensional state, and a native two-dimensional continuous action. Its
single/constant instruction means it tests visual continuous control and
robustness plumbing, not language grounding.

### LIBERO-Spatial

Use LIBERO-Spatial after Push-T. In LeRobot 0.4.4 it exposes two image views,
an eight-dimensional state, a native seven-dimensional continuous action, and
task instructions. Start with one task for reset/preprocess/action/step smoke,
then freeze task ids, episodes, seeds, control mode, and maximum horizon before
any comparison.

ACT and Diffusion do not natively consume language. On the project profiles
they may use the already observable task indicators in `state_32`; on LIBERO,
their controlled comparison must be task-specific rather than silently adding
a language encoder. SmolVLA and PI0.5 retain native task text.

External benchmarks keep their native action spaces. Their actions must not be
reinterpreted as `elite_tcp_delta_6d + piper_intent_id`, and raw scores from
different domains must not be compared as if they shared one scale.

## Visual Robustness Protocol

Train and select checkpoints on the clean split only. For every clean test
episode, replay the same task id, initial-condition seed, horizon, and policy
checkpoint under evaluation-only image corruption:

```text
gaussian_sensor_noise: severity 1/2/3
low_contrast:          severity 1/2/3
blur:                  severity 1/2/3
occlusion:             severity 1/2/3
```

Corruption seeds are derived from episode/frame/view/corruption/severity plus
a fixed global seed. Multi-view images receive deterministic per-view draws.
Do not change actions, proprioception, labels, environment geometry, or
dynamics. Do not count noisy copies as new training records or use noisy test
scores for checkpoint selection.

Report clean score, corrupted score, absolute degradation, and relative
degradation. For closed-loop public benchmarks the primary metric is success
rate. For D0/R0 use the existing project action metrics and report error
increase or classification-score decrease. These corruptions are coverage
tests, not measured models of vessel oxidation.

## Architecture Gates

### Gate B0: Contract And Configuration

Passed on Windows and the Ubuntu 4090:

- four fixed profiles: `project_d0`, `project_r0`, `pusht`, and
  `libero_spatial`;
- project nine-dimensional compatibility action round-trips to the preferred
  explicit output `elite_tcp_delta_6d + piper_intent_id`;
- public profiles retain native two- and seven-dimensional actions;
- paired deterministic noise keeps image shape and geometry unchanged;
- exact tip/contact/wall/route fields are rejected from policy observations;
- LeRobot 0.4.4 imports ACT, Diffusion, SmolVLA, and PI05 and accepts all four
  feature/action configurations.

### Gate B1: Zero-Step Model Execution

Partially closed without any dataset or optimizer step:

- ACT passed finite synthetic forward/backward on `project_d0` and `pusht`;
- Diffusion passed finite synthetic forward/backward on `project_d0` and
  `pusht`;
- a reduced random SmolVLA passed finite synthetic forward/backward on
  `project_d0` after using its native 512x512 internal resize;
- the full pinned PI0.5 path had already passed the real dual-view feature
  probe with loaded-parameter coverage `1.0`, latent shape `[1,2,2048]`, active
  action shape `[1,1,1,9]`, and zero optimizer steps.

The SmolVLA processor emitted an upstream deprecation warning about
`preprocessor.json`; it did not affect the current forward/backward result.
No SmolVLA pretrained weights were loaded by this smoke.

### Gate B2: Public Environment Runtime

Closed for the selected Push-T and LIBERO task-0 runtime profiles on
2026-09-11. The 4090 has the pinned LeRobot `0.4.4`
Push-T/LIBERO extras, and `pip check`, CUDA, package imports, all four policy
imports/configs, and five zero-training lightweight executions pass. Evidence
is `simulation_output/lerobot_baseline_architecture_probe_v7.json`.

Push-T passes reset, raw and processed observation keys,
`[0,512]^2` native action bounds, RGB render, one finite step, and exact
same-seed state/image replay. LIBERO-Spatial task 0 passes two-camera and nested
raw state extraction, two processed image tensors plus an eight-dimensional
state and task text, native relative `[-1,1]^7` action bounds, RGB render, one
finite step, and exact same-seed replay of all raw observation leaves. Its
effective assets path is the explicit project directory, not a user cache.

These are bounded runtime-interface results only. No demonstrations, public
dataset, pretrained policy/checkpoint, guidewire record, training loop, or
optimizer step was used, so Gate B2 does not provide a policy score. An
upstream path behavior found during a rejected intermediate run created a
duplicate LIBERO asset cache; the accepted v3 and combined runs pin and verify
the explicit project path without downloading assets.

The evaluation-only corruption wrapper is also runtime-integrated at the raw
observation/preprocessing boundary. A paired Gaussian-noise severity-2 smoke
completed 20 fixed-action transitions on Push-T and LIBERO task 0. Across both
paths, only declared image leaves changed; independently reset source
observations, non-image leaves, processed state/task, native actions, rewards,
and termination flags remained exact. This closes the noise-plumbing
precondition but is not a policy robustness result because no policy was
loaded. Dependency-light tests separately cover all four corruption types.

### Gate B3: One-Batch/One-Episode Policy Integration

Closed for ACT, Diffusion Policy, reduced SmolVLA, and pinned PI0.5 on Push-T,
and for the language-conditioned reduced SmolVLA and pinned PI0.5 slice on
LIBERO-Spatial task 0. The random policies and the full pretrained PI0.5
consumed real environment batches through the official LeRobot processors and
completed 20 transitions without an out-of-bounds action. The
Diffusion run additionally verified its two-observation history and
four-action queue. SmolVLA verified its constant Push-T instruction through
newline handling and tokenization, its two-action queue, and fully offline
construction from cached config/tokenizer/processor files with no model-weight
file present. PI0.5 resolved the pinned 14.47 GB checkpoint at the declared
revision with full unique-parameter coverage, encoded a constant task plus 32
state bins, and verified its two-action queue. On LIBERO, both VLA paths
preserved the native task description, processed two native 256x256 camera
tensors plus the eight-dimensional state, and returned native relative 7D
actions in `[-1,1]`. The benchmark feature declaration is 224x224; model-native
visual preprocessing performs the actual resize, so the reports retain both
sizes rather than mislabelling the environment output. None of the smokes
loaded demonstrations or used a backward/optimizer step. Their normalization
values are explicit semantic/bounds-derived processor inputs, not dataset
statistics. These are architecture-integration results, not baseline scores;
random SmolVLA and base PI0.5 rollouts must not be scored. Score-bearing
training/evaluation remains a separate explicitly approved stage under the
five-minute rule.

### Gate B4: Frozen Score-Bearing LIBERO Protocol

The pre-acquisition contract is frozen in
`docs/libero-spatial-score-protocol-v1.json`. This gate does not authorize a
checkpoint or dataset download, training, or evaluation.

Gate B4a is a score-pipeline preflight on LIBERO-Spatial task `0`: ten episodes,
initialization-state indices `0..9`, episode seeds `1000..1009`, batch size `1`,
`max_parallel_tasks=1`, relative control, and a `280`-step horizon. Report its
episode success rate, but do not compare this one-task result with the published
ten-task Spatial aggregate. Gate B4b applies the same per-task initialization,
seed, and horizon contract to task ids `0..9` for `100` total episodes and
reports per-task success plus the suite macro average.

The primary no-training route is the exact task-finetuned
`lerobot/pi05_libero_finetuned` revision
`8e174154ef5f6c60a8da12ae99c303d8963138c1`. Its config matches the runtime
contract: two `256x256` camera inputs, state dimension `8`, and action dimension
`7`. The direct `lerobot/smolvla_libero` route is rejected because its pinned
config declares state dimension `6` and three cameras, so a direct score would
not preserve the same environment input contract. A future SmolVLA comparison
must instead start from pinned `lerobot/smolvla_base` and build a current
dataset-defined config from pinned `lerobot/libero`; this requires a separate
training authorization.

The audit passed `52/52` local static/Hub checks and `60/60` remote
static/Hub/runtime checks without downloading a large blob. The remote checks
also verified the installed package versions, ten-task suite, task-0 asset
hashes, 50 available init states, and the `280`-step limit. No policy,
demonstration, environment episode, optimizer, or score was involved. All
selected resources exceed `100 MB`, so acquisition remains user-owned under
the project transfer rule.

## Current Evidence

```text
simulation_output/vla_benchmark_contract_smoke_v1.json
simulation_output/vla_benchmark_contract_smoke_remote_v1.json
simulation_output/lerobot_baseline_architecture_probe_v7.json
simulation_output/lerobot_public_env_runtime_smoke_pusht_v1.json
simulation_output/lerobot_public_env_runtime_smoke_libero_v3.json
simulation_output/lerobot_public_env_runtime_smoke_all_v1.json
simulation_output/vla_visual_noise_wrapper_smoke_remote_v2.json
simulation_output/lerobot_public_noise_pair_smoke_all_v3.json
simulation_output/act_pusht_policy_integration_v1/report.json
simulation_output/diffusion_pusht_policy_integration_v1/report.json
simulation_output/smolvla_pusht_policy_integration_v1/report.json
simulation_output/pi05_pusht_policy_integration_v1/report.json
simulation_output/smolvla_libero_policy_integration_v1/report.json
simulation_output/pi05_libero_policy_integration_v1/report.json
docs/libero-spatial-score-protocol-v1.json
simulation_output/libero_score_protocol_audit_local_v1.json
simulation_output/libero_score_protocol_audit_remote_v1.json
simulation_output/pi05_action_effect_feature_interface_probe_v1.json
```

The installed PI0.5 path additionally requires the five official OpenPI
Transformers replacement files on top of `transformers==4.53.2`. A dependency
reinstall that removed them was rejected with AdaRMS key mismatches; the final
environment matches clean OpenPI commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`, and the integration smoke audits
all five file hashes before loading the checkpoint. Reinstalling Transformers
must therefore be followed by reapplying and verifying those files.

The final architecture probe reports LeRobot `0.4.4`, available Push-T and
LIBERO imports, four of four policy imports/config matrices passing, five
finite lightweight execution probes, no execution failures, and
`optimizer_steps=0`.

## External References Checked

- LeRobot 0.4.4 policies and simulation index:
  `https://huggingface.co/docs/lerobot/v0.4.4/index`
- ACT in LeRobot 0.4.4:
  `https://huggingface.co/docs/lerobot/v0.4.4/act`
- SmolVLA in LeRobot 0.4.4:
  `https://huggingface.co/docs/lerobot/v0.4.4/smolvla`
- OpenPI PyTorch setup and required Transformers replacement files:
  `https://github.com/Physical-Intelligence/openpi#pytorch-support`
- LIBERO interface and evaluation in LeRobot 0.4.4:
  `https://huggingface.co/docs/lerobot/v0.4.4/libero`
- Official LeRobot Push-T dataset:
  `https://huggingface.co/datasets/lerobot/pusht`
- Diffusion Policy project and paper:
  `https://diffusion-policy.cs.columbia.edu/`

## Stop Conditions

- Do not install or train OpenVLA-OFT for this ladder.
- Do not inspect the new real records during the architecture gate.
- Do not mix the senior historical dataset with current real data.
- Do not use a public benchmark score as evidence of guidewire or real-system
  success.
- Do not use noisy evaluation for training or model selection in the first
  controlled comparison.
- Do not modify collectors, simulation experts, or data-track documents from
  this plan.
