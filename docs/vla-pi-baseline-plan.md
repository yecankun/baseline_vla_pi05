# Pi-Style VLA Baseline Plan

Last updated: 2026-07-13

## Decision

Use a pi-family-style VLA as the first serious algorithm baseline, but do not
promise exact reproduction of the latest Physical Intelligence model.

Practical starting point:

```text
openpi / pi0.5-style baseline
```

Accepted design:

```text
openpi/pi0.5-style reproducible base + pi0.7-style context conditioning
```

For this project, the first pi0.7-style context channel is the image-derived
tactile/contact estimator output.

Reason:

- Physical Intelligence has newer pi-family research posts, including pi0.7,
  but the practical open-source baseline path is currently openpi / pi0.5-style
  fine-tuning.
- `openpi` provides pi0, pi0-FAST, and pi0.5 model paths and checkpoints.
- LeRobot exposes a `pi05_base` policy interface with multi-view images,
  proprio/state, optional language instruction, and continuous actions.
- This project still needs custom changes for tactile/contact input and mixed
  continuous/discrete actions.
- The project-specific improvement over a plain pi0.5-style baseline is to
  expose tactile/contact estimates as explicit context rather than asking the
  model to infer contact only from raw pixels.

## External References Checked

- Physical Intelligence pi0.7 blog, 2026-04-16:
  `https://www.pi.website/blog/pi07`
- Physical Intelligence pi0.5 blog:
  `https://www.pi.website/blog/pi05`
- OpenPI repository:
  `https://github.com/Physical-Intelligence/openpi`
- LeRobot pi05_base model card:
  `https://huggingface.co/lerobot/pi05_base`

## Baseline Semantics

Target observation:

```text
side image
top image
Elite TCP 6D pose / state
Piper feed state or controller state
task instruction
image-derived tactile/contact fields
```

Tactile/contact fields should come from visual collision or distance analysis:

```text
estimated_contact_flag
contact_estimator_confidence
estimated_image_distance_px
contact_source
```

Do not feed exact MuJoCo contact, wall distance, contact normal, or route truth
as tactile input.

Target action:

```text
Elite TCP delta + Piper discrete intent
```

The pi0.5-style base interface is continuous-action oriented, so our baseline
needs one of these adaptations:

1. Mixed head: continuous Elite TCP-delta head plus discrete Piper intent head.
2. Continuous padded action: encode Piper intent as an action dimension for the
   first feasibility run, then split it into a real classifier later.

Prefer the mixed head when implementation cost is reasonable. Use continuous
padding only as a smoke-test shortcut and label it as such.

## First Experiment Scope

The first experiment should be feasibility-first:

- freeze most of the pretrained backbone when possible;
- use LoRA/adapters or a small trainable action head first;
- train on existing sim data plus available real calibration captures where
  labels are valid;
- evaluate open-loop before any closed-loop rollout;
- compare with:
  - current small BC;
  - pi-style baseline without tactile/contact fields;
  - pi-style baseline with tactile/contact fields treated as explicit
    pi0.7-style context.

Success criteria:

- dataset adapter works;
- model can ingest dual-view images, state, task, and tactile fields;
- predicted Elite TCP delta has reasonable scale;
- Piper intent is not degenerate;
- tactile/contact ablation changes behavior in the expected direction.

Non-goals:

- prove real-system deployment readiness;
- claim sim-to-real success;
- solve guidewire delivery hardware;
- reproduce pi0.7 or the full pi0.5 training recipe;
- claim that context conditioning has been validated before the ablation
  succeeds.

## Implementation Notes

Start from a dataset adapter before changing rollout control:

```text
records/manifest -> pi-style sample format
```

Minimum fields per sample:

```text
images: side/top
state: Elite TCP, Piper state, estimator/contact fields
language/task: left/right branch instruction
action: Elite TCP delta, Piper intent
```

Keep all normalizers and action scaling explicit. The pi-style code path is
sensitive to action normalization and robot embodiment mismatch.
