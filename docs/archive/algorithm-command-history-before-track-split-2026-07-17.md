# Algorithm Command History Before Track Split

Archived: 2026-07-17

This file preserves the algorithm/VLA command block mechanically removed from
`docs/commands.md` when the project documentation was split into algorithm and
data tracks. Current algorithm commands live in
`docs/algorithm-track-commands.md`.

## Pi-Style VLA Dataset Audit

Use this before writing or training the openpi/pi0.5-style baseline adapter. It
checks whether an existing sim manifest or real `records.jsonl` dataset can map
to the planned pi-style sample schema:

```text
images: side/top
state: Elite TCP + Piper state + estimated tactile/contact context
language: left/right branch instruction
action: Elite TCP delta + Piper intent
```

Audit a current sim dataset with estimator/contact fields:

```powershell
.\.venv\Scripts\python.exe tools\audit_pi_style_dataset.py `
  simulation_output\formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1 `
  --check-images `
  --out simulation_output\pi_style_audit_formal_tip_line_visual_route_v1_esttip_reggeom.json
```

Audit selected real captures:

```powershell
.\.venv\Scripts\python.exe tools\audit_pi_style_dataset.py `
  collected_data\real_pilot_20260710_left_short_after_s_bend_removed_001 `
  collected_data\real_pilot_20260707_left_linked_separate_elite_conn_001 `
  --check-images `
  --out simulation_output\pi_style_audit_selected_real.json
```

Quick scan all known sim/real datasets:

```powershell
.\.venv\Scripts\python.exe tools\audit_pi_style_dataset.py `
  --scan-defaults `
  --max-records 50 `
  --out simulation_output\pi_style_audit_scan_defaults_50.json
```

Current smoke result:

```text
simulation_output\formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1:
  candidate_for_pi_style_with_tactile_context

collected_data\real_pilot_20260710_left_short_after_s_bend_removed_001:
  candidate_for_pi_style_without_tactile_context_or_needs_contact_estimator_fill
  Elite TCP delta all-zero, so weak for Elite action learning

collected_data\real_pilot_20260707_left_linked_separate_elite_conn_001:
  candidate_for_pi_style_without_tactile_context_or_needs_contact_estimator_fill
```

Interpretation: current sim estimator datasets can seed the first
tactile-context pi-style adapter. Current real captures have the right outer
schema, but need image-distance/contact fields filled before they can test the
pi0.7-style context channel.

## Pi-Style Intermediate Export

Export a current sim estimator dataset into the project intermediate JSONL
format for the first openpi/pi0.5-style adapter. This is not yet a native
openpi or LeRobot dataset; it is the stable project boundary before writing
the framework-specific converter.

Smoke export with copied images and required tactile/contact context:

```powershell
.\.venv\Scripts\python.exe tools\export_pi_style_dataset.py `
  simulation_output\formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1 `
  --out simulation_output\pi_style_export_smoke_sim_tactile `
  --max-records 20 `
  --copy-images `
  --require-tactile-context
```

Full sim export command to run when ready:

```powershell
.\.venv\Scripts\python.exe tools\export_pi_style_dataset.py `
  simulation_output\formal_tip_line_visual_route_v1_esttip_reggeom_dataset_v1 `
  --out simulation_output\pi_style_formal_tip_line_visual_route_v1_esttip_reggeom `
  --copy-images `
  --require-tactile-context
```

Real-capture schema smoke export, without requiring tactile/contact fields:

```powershell
.\.venv\Scripts\python.exe tools\export_pi_style_dataset.py `
  collected_data\real_pilot_20260707_left_linked_separate_elite_conn_001 `
  --out simulation_output\pi_style_export_smoke_real_no_tactile `
  --max-records 20 `
  --copy-images
```

The exporter writes:

```text
manifest.json
samples.jsonl
images/side/...
images/top/...
```

`manifest.json` includes normalization hints for Elite TCP pose, Elite TCP
delta, Piper state, and tactile/contact fields. Keep these stats explicit
when adapting to openpi/LeRobot; do not silently rely on framework defaults.

Validate an export before training:

```powershell
.\.venv\Scripts\python.exe tools\validate_pi_style_export.py `
  simulation_output\pi_style_export_smoke_sim_tactile `
  --require-images-existing `
  --out simulation_output\pi_style_validate_smoke_sim_tactile.json
```

Optional quick visual contact sheet:

```powershell
.\.venv\Scripts\python.exe tools\validate_pi_style_export.py `
  simulation_output\pi_style_export_smoke_sim_tactile `
  --require-images-existing `
  --contact-sheet simulation_output\pi_style_export_smoke_sim_tactile_contact_sheet.png
```

Expected interpretation:

```text
Sim estimator export:
  should pass validation with tactile_signal_samples > 0.

Real capture export:
  may pass structure validation, but should warn about missing tactile/contact
  signal until the image-distance/contact estimator fill path is implemented.
```

Prepare a framework-neutral training pack from a validated export:

```powershell
.\.venv\Scripts\python.exe tools\prepare_pi_style_training_pack.py `
  simulation_output\pi_style_export_smoke_sim_tactile `
  --out simulation_output\pi_style_pack_smoke_sim_tactile `
  --require-tactile-context `
  --strict-images
```

For a full exported sim dataset:

```powershell
.\.venv\Scripts\python.exe tools\prepare_pi_style_training_pack.py `
  simulation_output\pi_style_formal_tip_line_visual_route_v1_esttip_reggeom `
  --out simulation_output\pi_style_pack_formal_tip_line_visual_route_v1_esttip_reggeom `
  --require-tactile-context `
  --strict-images
```

The training pack writes:

```text
manifest.json
index.jsonl
arrays.npz
```

It keeps image paths relative to the pi-style export directory and stores
array-ready state/action fields:

```text
observations:
  elite_tcp_pose_6d
  piper_state
  tactile_context
  tactile_context_valid
  task_one_hot

preferred action heads:
  elite_tcp_delta_6d: continuous
  piper_intent_id: classification

smoke-test shortcut:
  action_continuous_padded = elite_tcp_delta_6d + Piper intent one-hot
```

Train a small mixed-head pi-style pack smoke model. This is not the final
openpi/pi0.5 implementation; it verifies that the pi-style pack can train with
dual-view images, state, tactile context, Elite TCP-delta regression, and Piper
intent classification:

```powershell
$env:PYTHONUNBUFFERED='1'; .\.venv\Scripts\python.exe -B tools\train_pi_style_pack_smoke.py `
  simulation_output\pi_style_pack_frontup_step024_esttip_reggeom_v1 `
  --out simulation_output\pi_style_smoke_model_frontup_step024_esttip_reggeom_v1 `
  --epochs 8 `
  --batch-size 64 `
  --image-size 128
```

Package the same pi-style export/pack for Linux/Ubuntu 4090 training. This
normalizes JSONL image paths to POSIX-style separators and writes a bundle with
`pi_style_export/` plus `pi_style_pack/`:

```powershell
.\.venv\Scripts\python.exe -B tools\package_pi_style_4090_bundle.py `
  simulation_output\pi_style_frontup_step024_esttip_reggeom_v1 `
  simulation_output\pi_style_pack_frontup_step024_esttip_reggeom_v1 `
  --out simulation_output\pi_style_4090_bundle_frontup_step024_esttip_reggeom_v1 `
  --zip
```

On the Ubuntu 4090 machine, first run a one-epoch smoke check before adapting
to OpenPI/LeRobot:

```bash
python tools/train_pi_style_pack_smoke.py \
  simulation_output/pi_style_4090_bundle_frontup_step024_esttip_reggeom_v1/pi_style_pack \
  --out simulation_output/pi_style_4090_bundle_smoke_model \
  --epochs 1 \
  --batch-size 64 \
  --image-size 128
```

## OpenPI-Compatible Adapter Pack

After the pi-style smoke model trains on the 4090 machine, prepare the first
OpenPI-compatible adapter pack. This does not replace the project action
contract. It creates 32-dim state/action compatibility tensors while preserving
the preferred mixed target:

```text
Elite TCP delta regression + Piper intent classification
```

Create the compatibility pack:

```powershell
.\.venv\Scripts\python.exe -B tools\prepare_openpi_compat_pack.py `
  simulation_output\pi_style_pack_frontup_step024_esttip_reggeom_v1 `
  --out simulation_output\openpi_compat_pack_frontup_step024_esttip_reggeom_v1
```

Validate it before syncing to the 4090 machine:

```powershell
.\.venv\Scripts\python.exe -B tools\validate_openpi_compat_pack.py `
  simulation_output\openpi_compat_pack_frontup_step024_esttip_reggeom_v1 `
  --out simulation_output\openpi_compat_pack_frontup_step024_esttip_reggeom_v1_validate.json
```

For this adapter:

```text
state_32:
  dims 0:6   = Elite TCP pose 6D
  dims 6:8   = Piper state
  dims 8:11  = tactile/contact context
  dims 11:14 = tactile/contact validity mask
  dims 14:16 = task one-hot
  dims 16:32 = padded zeros

action_32:
  dims 0:6  = Elite TCP delta 6D
  dims 6:9  = Piper intent one-hot compatibility field
  dims 9:32 = padded zeros

Important: `action_32` is only a framework compatibility tensor. Do not
interpret padded zero dimensions as robot controls, and do not treat the Piper
one-hot compatibility field as the final preferred action head.
```

Train a small OpenPI-compatible mixed-head smoke model on the 4090 machine.
This is still a local feasibility adapter, not official OpenPI training:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/train_openpi_compat_smoke.py \
  simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1 \
  --out simulation_output/openpi_compat_smoke_model_frontup_step024_esttip_reggeom_v1 \
  --epochs 8 \
  --batch-size 64 \
  --image-size 128
```

The expected metric surface is:

```text
val_piper_acc: Piper intent classification accuracy
val_action_mae: active dimensions of the 32D compatibility action
val_elite_mae: dims 0:6 of action_32, equivalent to Elite TCP-delta MAE
```

Run the tactile/contact ablation on the same 4090 pack. The `full` run is the
reference; `drop_tactile_values` zeros `state_32` dims 8:11 before
normalization, while keeping the tactile-validity dims available:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/train_openpi_compat_smoke.py \
  simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1 \
  --out simulation_output/openpi_compat_smoke_model_frontup_step024_notactile_values_v1 \
  --epochs 8 \
  --batch-size 64 \
  --image-size 128 \
  --tactile-ablation drop_tactile_values
```

If the first ablation is inconclusive, run the stronger variant that also zeros
the tactile-validity dims 11:14:

```bash
python tools/train_openpi_compat_smoke.py \
  simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1 \
  --out simulation_output/openpi_compat_smoke_model_frontup_step024_notactile_all_v1 \
  --epochs 8 \
  --batch-size 64 \
  --image-size 128 \
  --tactile-ablation drop_tactile_all
```

Audit whether the current tactile/contact fields actually carry useful signal:

```powershell
.\.venv\Scripts\python.exe -B tools\audit_openpi_tactile_signal.py `
  simulation_output\openpi_compat_pack_frontup_step024_esttip_reggeom_v1 `
  --out simulation_output\openpi_tactile_signal_audit_frontup_step024_esttip_reggeom_v1.json
```

Interpretation guide:

```text
estimated_contact_flag unique_count=1:
  the contact flag is constant and cannot help supervised learning.

weak correlation with piper_feed_binary / elite_xyz_l2:
  current tactile values are likely interface placeholders rather than useful
  conditioning signals.
```

Create a contact-rich tactile-label variant for a control experiment. This
posthoc label uses `estimated_image_distance_px` only; it is not real contact
truth and should not be used as formal sim-to-real validation:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/relabel_openpi_tactile_threshold.py \
  simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1 \
  --out simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1 \
  --target-positive-rate 0.15 \
  --confidence-band-px 2.0

python tools/validate_openpi_compat_pack.py \
  simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1 \
  --out simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1_validate.json

python tools/audit_openpi_tactile_signal.py \
  simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1 \
  --out simulation_output/openpi_tactile_signal_audit_frontup_step024_contactrich_q15_v1.json
```

If the relabeled pack validates, compare full tactile input against dropped
tactile input:

```bash
python tools/train_openpi_compat_smoke.py \
  simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1 \
  --out simulation_output/openpi_compat_smoke_model_frontup_step024_contactrich_q15_full_v1 \
  --epochs 8 \
  --batch-size 64 \
  --image-size 128

python tools/train_openpi_compat_smoke.py \
  simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1 \
  --out simulation_output/openpi_compat_smoke_model_frontup_step024_contactrich_q15_dropall_v1 \
  --epochs 8 \
  --batch-size 64 \
  --image-size 128 \
  --tactile-ablation drop_tactile_all
```

## LeRobot / OpenPI Dataset Export

Use this only after the OpenPI-compatible pack validates. This step exports the
pack into an official `LeRobotDataset` directory for OpenPI-style integration.
It preserves project semantics by storing:

```text
action = action_32 compatibility tensor
elite_tcp_delta_6d = explicit Elite target
piper_intent_id = explicit Piper discrete target
```

First install LeRobot/OpenPI dependencies in an isolated 4090 environment. Do
not install them into the local MuJoCo `.venv`:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
pip install "lerobot[dataset]"
```

Run a tiny export smoke without video encoding first:

```bash
python tools/export_openpi_compat_to_lerobot.py \
  simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1 \
  --out simulation_output/lerobot_project2026_frontup_step024_smoke \
  --repo-id project2026/guidewire-openpi-compat-smoke \
  --image-size 224 \
  --no-videos \
  --max-episodes 2 \
  --max-frames-per-episode 16 \
  --force
```

If the smoke works, export the full dataset. Prefer `--no-videos` until the
official training path is confirmed; video encoding adds another failure mode:

```bash
python tools/export_openpi_compat_to_lerobot.py \
  simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1 \
  --out simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --image-size 224 \
  --no-videos \
  --force
```

The exporter writes:

```text
project2026_lerobot_export_manifest.json
```

Use that manifest to verify episode/frame counts and the action semantics before
starting any OpenPI training.

Before a real OpenPI policy run, verify that the exported LeRobotDataset can be
read by a mixed-head project smoke model. This tests the official LeRobot
dataset loader, episode-level train/val split, image tensors, `action_32`,
`elite_tcp_delta_6d`, and `piper_intent_id` fields.

Small loader/training smoke:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/train_lerobot_compat_smoke.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/lerobot_compat_smoke_model_frontup_step024_small \
  --epochs 1 \
  --batch-size 32 \
  --max-records 256
```

If the small run works, run the full smoke:

```bash
python tools/train_lerobot_compat_smoke.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/lerobot_compat_smoke_model_frontup_step024_v1 \
  --epochs 8 \
  --batch-size 64
```

This script still uses a small local policy. Its purpose is to verify the
LeRobotDataset interface before plugging into OpenPI/LeRobot model code.

## PI05 / OpenPI Policy Adapter Probe

After the LeRobotDataset loader smoke succeeds, run a real LeRobot `PI05Policy`
adapter probe on the Ubuntu 4090 machine. This is still an algorithm/interface
feasibility check, not real-system validation. The script keeps
`action_32` as a framework compatibility tensor and uses `chunk_size=1` because
the current dataset contains one supervised action per frame.

If PI05 tokenizer dependencies are missing, install them in the isolated 4090
environment, not in the local Windows MuJoCo `.venv`:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
pip install "lerobot[transformers-dep]"
```

Run a preprocessing-only probe first. This verifies LeRobotDataset fields,
state/action normalization, PI05 prompt/tokenizer preparation, and tensor
shapes without instantiating the policy weights:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi

python tools/probe_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_lerobot_adapter_probe_preprocess_frontup_step024_v1.json \
  --batch-size 1 \
  --max-records 16 \
  --preprocess-only
```

If preprocessing succeeds, run one PI05 forward/backward probe. This may
download model/tokenizer assets and use substantial GPU memory:

```bash
python tools/probe_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_lerobot_adapter_probe_forward_backward_frontup_step024_v2.json \
  --batch-size 1 \
  --max-records 16 \
  --paligemma-variant gemma_2b \
  --action-expert-variant gemma_300m \
  --dtype bfloat16 \
  --freeze-vision-encoder \
  --train-expert-only \
  --gradient-checkpointing \
  --backward
```

Interpretation:

```text
preprocess-only success:
  the official PI05 preprocessor can consume the project LeRobotDataset export.

forward/backward success:
  the official PI05Policy can compute loss and gradients on project data.

failure at dependency/tokenizer download:
  environment/model-cache issue, not a dataset-schema failure.

failure after processed_batch is formed:
  likely PI05 config, action chunk shape, dtype, or GPU-memory issue.

Known PI05 config constraint:
  use `--paligemma-variant gemma_2b` with `--action-expert-variant gemma_300m`
  for this probe. `paligemma-variant gemma_300m` mismatches the PaliGemma image
  embedding width and fails in `embed_prefix` with a `2048` versus `1024`
  tensor-size mismatch.
```

## PI05 Adapter Short Training Smoke

After the PI05 forward/backward probe succeeds, run a short adapter-training
smoke. This checks whether the official PI05 loss can optimize for a few steps
on the project LeRobotDataset export. It is still an algorithm/interface
feasibility check, not a policy-quality or real-system validation result.

The script defaults to the verified PI05 configuration:

```text
paligemma_variant=gemma_2b
action_expert_variant=gemma_300m
dtype=bfloat16
chunk_size=1
freeze_vision_encoder=True
train_expert_only=True
gradient_checkpointing=True
```

Run the first short smoke on the Ubuntu 4090 machine:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_lerobot_adapter_train_smoke_frontup_step024_v1 \
  --max-steps 40 \
  --batch-size 1 \
  --max-records 512 \
  --val-batches 8 \
  --eval-every 10 \
  --log-every 1
```

The script writes:

```text
train_log.json
summary.json
```

By default it does not save model weights because PI05 checkpoints are large.
If a short run looks useful and a checkpoint is needed, add
`--save-checkpoint` in a later run.

Run the checkpointed 1000-step PI05 adapter training job on the Ubuntu 4090
machine only. This writes a large `final_policy.pt` checkpoint:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_lerobot_adapter_train_ckpt_frontup_step024_v1 \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --val-batches 32 \
  --eval-every 100 \
  --log-every 10 \
  --eval-seed 20260716 \
  --save-checkpoint
```

If the 40-step smoke has noisy validation or no clear decrease, run a
micro-batch overfit diagnostic. This repeats one batch and fixes PI05
noise/time sampling, so it answers a narrower question: can the optimizer lower
the PI05 loss on an identical supervised batch?

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_lerobot_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_lerobot_adapter_overfit_smoke_frontup_step024_v1 \
  --max-steps 80 \
  --batch-size 1 \
  --max-records 128 \
  --val-batches 4 \
  --eval-every 10 \
  --log-every 1 \
  --overfit-first-batch \
  --fixed-train-loss-seed 20260716 \
  --eval-seed 20260716
```

Interpretation:

```text
initial_val_loss -> final_val_loss decreases:
  the official PI05 adapter is trainable on this data path.

loss is finite but does not decrease in 40 steps:
  the interface is still valid; run the micro-batch overfit diagnostic before
  drawing model-quality conclusions.

micro-batch overfit loss decreases:
  the optimizer/model path is functional, and the previous no-decrease result
  was likely caused by short-run noise, diffusion loss variance, small batch
  size, or insufficient training horizon.

micro-batch overfit loss does not decrease:
  inspect optimizer scope, LR, trainable parameter selection, and action
  normalization before attempting longer PI05 training.

OOM:
  keep batch-size 1, reduce val-batches, remove backward checkpoint saving, or
  make a smaller training subset. Do not move this heavy run to Windows.
```

## PI05 Held-Out Open-Loop Diagnostic

After a checkpointed PI05 run completes, evaluate whether sampled actions look
like held-out expert actions. This is still sim-data algorithm feasibility.
`action_32` remains a framework compatibility tensor:

```text
dims 0:6 = Elite TCP delta
dims 6:9 = Piper intent one-hot compatibility
dims 9:32 = padding
```

Run the full diagnostic on the Ubuntu 4090 machine:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/eval_pi05_lerobot_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --checkpoint simulation_output/pi05_lerobot_adapter_train_ckpt_frontup_step024_v1/final_policy.pt \
  --out simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_v1.json \
  --max-records 4096 \
  --max-val-batches 128 \
  --batch-size 1 \
  --eval-seed 20260717 \
  --save-samples 32 \
  --sample-mode random
```

The diagnostic reports normalized active-action error, raw dims `0:9` error,
Elite TCP-delta MAE for dims `0:6`, Piper intent argmax accuracy for dims
`6:9`, and a small set of prediction examples. Use it to decide whether the
checkpoint predicts plausible held-out actions before doing any rollout-like
test.

If the full diagnostic shows a gap between low PI05 training loss and weak
sampled actions, compare a larger diffusion inference-step count before drawing
a final conclusion about the checkpoint:

```bash
python tools/eval_pi05_lerobot_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --checkpoint simulation_output/pi05_lerobot_adapter_train_ckpt_frontup_step024_v1/final_policy.pt \
  --out simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_fullval_steps50_v1.json \
  --max-records 4096 \
  --max-val-batches 616 \
  --batch-size 1 \
  --eval-seed 20260717 \
  --save-samples 64 \
  --sample-mode random \
  --num-inference-steps 50
```

Current full-val result to beat:

```text
simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_fullval_v1.json
records=616
piper_acc=0.5032
piper_majority_baseline=0.5292
elite_tcp_delta_mae_mean=0.6283
elite_tcp_delta_mae_median=0.2451

simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_fullval_steps50_v1.json
num_inference_steps=50
records=616
piper_acc=0.5081
piper_majority_baseline=0.5292
elite_tcp_delta_mae_mean=0.6296
elite_tcp_delta_mae_median=0.2472
```

Interpretation: increasing PI05 inference steps to 50 did not materially fix
the weak Piper result. The next useful algorithm change is likely a mixed-head
contract implementation: Elite continuous regression plus Piper explicit
classification, rather than relying on Piper one-hot values inside
`action_32`.

## PI05 Mixed-Head Action Adapter

The installed LeRobot `0.4.4` PI05 implementation uses images and language in
its forward/sampling paths but does not consume `observation.state`. Inspect the
actual remote implementation before changing the adapter:

```powershell
scp tools\probe_pi05_feature_interface.py project4090:/home/zsw/project_2026/tools/
ssh project4090 "cd /home/zsw/project_2026 && /home/zsw/miniconda3/bin/conda run -n project2026-pi python tools/probe_pi05_feature_interface.py --out simulation_output/pi05_feature_interface_probe_v1.json --source-out simulation_output/pi05_modeling_source_v1.py"
```

Sync the mixed-head scripts after every local change:

```powershell
scp tools\train_pi05_mixed_head_adapter.py project4090:/home/zsw/project_2026/tools/
scp tools\eval_pi05_mixed_head_open_loop.py project4090:/home/zsw/project_2026/tools/
```

First train only the explicit Piper head while reusing the frozen 1000-step
PI05 checkpoint for Elite. This isolates the action-contract change and saves a
small head-only checkpoint instead of another 7 GB policy copy:

```bash
cd /home/zsw/project_2026
conda activate project2026-pi
export HF_HUB_OFFLINE=1

python tools/train_pi05_mixed_head_adapter.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --out simulation_output/pi05_mixed_head_adapter_train_frontup_step024_v1 \
  --init-policy-checkpoint simulation_output/pi05_lerobot_adapter_train_ckpt_frontup_step024_v1/final_policy.pt \
  --max-steps 1000 \
  --batch-size 1 \
  --max-records 4096 \
  --val-batches 128 \
  --eval-every 100 \
  --log-every 10 \
  --eval-seed 20260717 \
  --class-weighting balanced \
  --freeze-pi05 \
  --save-checkpoint
```

Evaluate Elite sampling and Piper classification as separate action heads on
the same held-out episode split:

```bash
python tools/eval_pi05_mixed_head_open_loop.py \
  --root simulation_output/lerobot_project2026_frontup_step024_v1 \
  --repo-id project2026/guidewire-openpi-compat-frontup-step024-v1 \
  --base-policy-checkpoint simulation_output/pi05_lerobot_adapter_train_ckpt_frontup_step024_v1/final_policy.pt \
  --mixed-head-checkpoint simulation_output/pi05_mixed_head_adapter_train_frontup_step024_v1/mixed_head_policy.pt \
  --out simulation_output/pi05_mixed_head_open_loop_frontup_step024_v1.json \
  --max-records 4096 \
  --max-val-batches 616 \
  --batch-size 1 \
  --eval-seed 20260717 \
  --save-samples 64
```

The first decision threshold is Piper held-out accuracy above the existing
`0.5292` majority baseline, with balanced accuracy and the confusion matrix
reported alongside it. Because the Elite checkpoint is frozen in this first
experiment, its open-loop result should remain comparable to the existing PI05
baseline. Passing this test closes the mixed action semantics only; it does not
yet inject state/tactile context into the Elite diffusion path.

Current v1 result: do not run the full evaluator above. The 1000-step
frozen-head run ended at `piper_acc=0.53125`, exactly equal to its validation
majority baseline, with `balanced_accuracy=0.5` and every validation prediction
in class 2. The base checkpoint was created with `PI05Policy(config)` rather
than `PI05Policy.from_pretrained(...)`, so its frozen PaliGemma prefix is not a
pretrained PI05 representation. Add and verify a `lerobot/pi05_base` loading
path before another long mixed-head run.

