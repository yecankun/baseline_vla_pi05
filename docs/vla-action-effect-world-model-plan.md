# PI0.5 Action-Effect World Model Plan

Status: pre-training implementation contract, 2026-09-02

This document belongs to the algorithm/VLA track. It defines the smallest
world-model innovation that can be implemented and tested before the new
simulation pack is ready. It does not accept a data pack, certify training
readiness, or report model performance.

## Research Question

Both left and right tasks require active Elite magnetic guidance. The physical
system nevertheless has a passive left bias: feeding without effective Elite
guidance can still enter the left branch. The historical senior dataset used a
branch-only success definition, so a successful left outcome did not prove
that the model learned the intended guidance mechanism.

The current research question is therefore:

> Can a PI0.5 action prior, combined with an action-conditioned latent world
> model trained primarily in simulation and adapted with a small current-real
> dataset, select actions that produce observable magnetic-guidance effects
> under oxidation-related low visibility rather than exploiting passive-left
> shortcut success?

Keep two evaluation concepts separate:

```text
legacy_branch_success = entered the requested branch
guidance_valid_success = entered the requested branch AND valid Elite guidance was engaged
```

The second metric is a proposed mechanism-aware evaluation target. Its
real-data labeling and acceptance procedure remain data-track work.

## Data Boundary

Training sources are limited to:

1. an accepted future simulation source with complete episode/family
   provenance;
2. the current ten real episodes, five left and five right, after a later
   data-track audit.

The user's current statement is that the new ten episodes have credible Elite
tracking/action execution and primarily suffer from visibility loss caused by
vessel oxidation. This is a planning assumption, not yet an algorithm-side
data audit result.

The historical senior real dataset is excluded from training, validation,
normalization, class weights, early stopping, and model selection because its
single-view observation, camera geometry, action quality, and branch-only
success semantics do not match the current source. It remains historical BC
context only.

Do not interpret repeated frames or augmentations as independent data. Every
window and derived view remains attached to its complete episode and source
trajectory.

## Shared Input And Output Contract

Policy-time input remains observable:

```text
recent PI0.5 Side/Top visual latents
recent state_32 values and validity
left/right task instruction
candidate action chunks
```

Candidate and executed action semantics remain:

```text
elite_tcp_delta_6d + piper_intent_id
```

The nine active compatibility dimensions are six Elite delta values followed
by retract/hold/feed one-hot values. PI0.5 padding dimensions are not controls.

Exact simulator tip, contact, wall, route, centerline, and related privileged
truth are forbidden policy inputs. A future accepted simulation pack may use
them as separate offline auxiliary targets if provenance and masks are
explicit. They must never be copied into the observation batch.

Raw event id, offline sample role, source weight, and diagnostic target objects
are also forbidden policy inputs.

## Feature-Pack Interface

The world-model trainer consumes:

```text
manifest.json
arrays.npz
episode_split.json
```

`manifest.json` schema:

```text
project2026_action_effect_feature_pack_v1
```

Required arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `visual_latent` | `[N,V,D]` | frozen PI0.5 per-view mean-patch latent |
| `state_32` | `[N,32]` | observable state compatibility tensor |
| `elite_tcp_delta_6d` | `[N,6]` | explicit Elite target |
| `piper_intent_id` | `[N]` | retract=0, hold=1, feed=2 |
| `task_id` | `[N]` | left=0, right=1 |
| `episode_index` | `[N]` | complete episode identity |
| `frame_index` | `[N]` | ordered frame identity inside episode |
| `domain_id` | `[N]` | sim=0, current real=1 |

Optional masked arrays:

```text
visual_valid[N,V]
state_valid_32[N,32]
degraded_visual_latent[N,V,D]  # paired coverage view encoded by the same frozen PI0.5 extractor
degradation_pair_valid[N]      # true only when the clean/degraded pair is valid
guidance_effect_id[N]   # -1 unknown, 0 ineffective/unknown, 1 left, 2 right
branch_outcome_id[N]    # -1 unknown, 0 left, 1 right
invalid_feed_flag[N]    # -1 unknown, 0 valid, 1 invalid/repeated
```

The manifest must pin both the PI0.5 pretrained source and revision, declare extractor
`pi05_multiview_mean_patch_v1`, keep that backbone frozen during feature-pack
construction, exclude historical senior data, and state that exact truth is
absent from policy input. It must also provide one `episode_provenance` entry
per episode with `episode_instance_id`, `source_trajectory_id`,
`scenario_family_id`, task, and domain. Undeclared arrays are rejected so
diagnostic truth cannot be smuggled into the feature pack.

The manifest must declare the exact existing observable `state_32` layout:

```text
0:6    elite_tcp_pose_6d
6:14   base Piper/tactile compatibility slots
14:16  task one-hot
16:27  observable Piper event-state mapping already fixed by the E2 adapter
27:32  reserved
```

It must also include a streaming SHA256 for `arrays.npz`. The episode split is
bound to the complete manifest SHA256, so a split from another feature pack
fails closed even if episode indices happen to match.

`degraded_visual_latent` and `degradation_pair_valid` are an all-or-nothing
pair. If they are absent, the adapter copies the clean latent for shape
compatibility and sets the validity mask false, so no consistency loss is
silently fabricated. If present, both clean and degraded images must have been
encoded by the same pinned, frozen PI0.5 vision extractor.

`episode_split.json` uses
`project2026_action_effect_episode_split_v1`. Train and validation episode sets
must be complete, disjoint, and cover every feature-pack episode exactly once.
The validator fails closed if any source trajectory or scenario family crosses
train and validation.

State and Elite-action mean/std are computed from training episodes only.
State dimensions are estimated only from valid training values; unobserved or
near-constant dimensions use mean `0`/scale `1` as applicable. Missing states
become normalized zero and retain an explicit validity bit in the world-model
input. The same saved statistics must normalize inference states and candidate
Elite actions; Piper one-hot intent is never normalized. Historical senior
records cannot contribute to these statistics.

## Temporal Adapter

Initial pre-training defaults:

```text
context length: 4 observations
candidate horizon: 3 actions
target horizon: 3 next observations
```

For history ending at `t`:

```text
history: [t-3, t-2, t-1, t]
actions: [a_t, a_t+1, a_t+2]
targets: [o_t+1, o_t+2, o_t+3]
```

Windows may not cross episode, task, domain, or source-trajectory boundaries.
Frame indices must be unique and strictly increasing inside each episode.

## Model

PI0.5 remains the action prior. The added module is a lightweight decoder-free
action-conditioned latent dynamics model, not a replacement Piper classifier:

```text
visual latent history ─┐
state history ─────────┼─> context GRU ─> action GRU for each candidate chunk
task embedding ────────┘                         │
                                                 ├─ next visual latent
                                                 ├─ observable state delta
                                                 ├─ guidance-effect logits
                                                 ├─ branch-trend logits
                                                 ├─ invalid-feed logits
                                                 └─ effect uncertainty
```

Initial hidden width is 256. The model does not decode pixels. This keeps the
new trainable component small enough for simulation pretraining plus limited
real adaptation and makes the innovation depend on predicted action effects,
not merely a new output head.

## Oxidation-Visibility Coverage

`OxidationCoverageAugmenter` creates clean/degraded pairs without geometric
warp. Its coverage transformations include contrast compression,
non-uniform low-frequency haze, bounded color gain, and optional mild blur.

This is explicitly a coverage distribution, not a fitted model of vessel
oxidation. Parameters must not be described as measured real statistics until
the current real images have been audited. Clean and degraded views share the
same actions and dynamics targets. Consistency is applied to task-related
future latent/state/effect predictions, not raw pixels.

## Losses

The implemented world-model objective is:

```text
L_wm =
    1.00 * L_next_visual_latent
  + 0.25 * L_observable_state_delta
  + 0.50 * L_guidance_effect
  + 0.50 * L_branch_trend
  + 0.25 * L_invalid_feed
  + 0.10 * L_uncertainty
  + 0.25 * L_clean_degraded_consistency
```

Continuous targets use Smooth-L1. Missing targets use explicit validity masks
or label `-1`; they are never imputed as positive or negative. A separate
clean/degraded consistency loss aligns future latent, state, guidance, and
branch predictions.

These coefficients are pre-training defaults for a controlled first run, not
validated final hyperparameters.

## Candidate Reranking

For each PI0.5 candidate chunk:

```text
score =
    target-branch probability
  + 0.75 * target-direction guidance probability
  - 0.20 * uncertainty
  - 0.25 * invalid-feed probability
  - 0.01 * Elite translation magnitude
```

No-guidance counterfactuals can be evaluated diagnostically but are marked
policy-ineligible. At least one candidate must be policy-eligible. This keeps
passive-left behavior available for causal comparison without legalizing it as
the left-task policy.

## Sim/Real Training Protocol

Training is staged:

1. train the action-effect model on accepted simulation windows;
2. fit oxidation-coverage consistency with paired simulation views;
3. add current-real windows using frozen PI0.5 features and small world-model
   or residual adaptation;
4. evaluate PI0.5 candidates with and without world-model reranking;
5. consider policy distillation only after reranking has held-out benefit.

Initial mixed-window draw mass is:

```text
simulation: 75%
current real: 25%
```

Sampling is source-balanced, then episode-balanced, then window-balanced.
This prevents long or repetitive episodes from dominating. Baseline and
world-model experiments must use equal optimizer steps and fixed seeds.

For exactly five left and five right current-real episodes, use five-fold
episode validation: each fold holds out one left and one right episode. All
windows and augmentations derived from an episode remain in that fold.

## Controlled Experiment Ladder

Run only after the data and feature-pack gates pass:

1. existing PI0.5 simulation baseline;
2. direct sim/current-real mixture without world model;
3. simulation pretraining plus current-real small-parameter adaptation;
4. add oxidation-coverage consistency;
5. add world-model auxiliary losses but no reranking;
6. add action-effect candidate reranking;
7. remove no-guidance counterfactual supervision;
8. remove guidance-effect target;
9. remove uncertainty or current-real residual adaptation.

Report legacy branch success separately from guidance-valid success. Also
report left/right separately, passive-left shortcut rate, Piper transition and
steady accuracy, invalid/repeated feed rate, world-model one-step error, and
uncertainty calibration. Offline metrics do not replace guarded rollout and
visual evidence.

## Training Gates

### Gate A0: implementation contract

Passed locally and remotely on 2026-09-02. The synthetic smoke verifies:

- explicit action round trip and PI0.5 action-32 conversion;
- frozen multiview latent extraction interface;
- episode-safe temporal windows;
- finite forward/loss/backward;
- paired clean/degraded latent loading and masked consistency loss in the
  trainer;
- candidate reranking and no-guidance ineligibility;
- 75/25 source mass and five-fold current-real split;
- source/episode/window balancing inside each domain, training-only
  normalization, feature-array hashing, and split-to-manifest binding;
- privileged-field rejection;
- trainer `--dry-run` with `training_started=false`.

The installed PI0.5 integration probe also passed on one existing stride-5
simulation record: pinned pretrained coverage `1.0`, finite frozen Side/Top
latent `[1,2,2048]`, explicit active action `[1,1,1,9]`, and zero optimizer
steps. This validates the code path, not the future data pack.

### Gate A1: source acceptance

Pending data-track delivery. Requires accepted simulation provenance, current
real episode audit, image paths/arrays, representative user visual review, and
no historical senior records.

### Gate A2: real PI0.5 feature-pack smoke

Pending source delivery. Run frozen feature extraction on a small accepted
sim/current-real subset and validate manifest fingerprints, latent shapes,
episode windows, missing-field masks, and split integrity.

### Gate A3: training authorization

Pending. Start no optimization before A1 and A2 pass and the user approves the
first controlled command.

## Implemented Files

```text
tools/pi05_action_effect_world_model.py
tools/pi05_action_effect_dataset.py
tools/probe_pi05_action_effect_feature_interface.py
tools/train_pi05_action_effect_world_model.py
tools/test_pi05_action_effect_world_model_smoke.py
```
