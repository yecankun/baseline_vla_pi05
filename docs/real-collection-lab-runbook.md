# Real Collection Lab Runbook

Last updated: 2026-07-05

Purpose:

```text
Collect a small real pilot dataset before another sim-trained policy comparison.
The goal is synchronized real observations and expert/controller action logs,
not immediate learned-policy robot control.
```

## Rule

Do not start with learned policy control on the real robots.

Use this order:

```text
1. static smoke: cameras + Elite pose + records.jsonl schema
2. validate smoke output
3. moving pilot, mode1 or mode2
4. validate every pilot immediately
5. only then run shadow-mode inference
```

## Keyboard

During preview:

```text
0 or h: hold label
1 or f: feed label
r or b: retract label
q or esc: stop
```

## Static Smoke

This should not move Piper.

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260705_smoke_left_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 30 `
  --collection-mode shadow_static `
  --session-note "static smoke: no Piper motion"
```

Validate:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_20260705_smoke_left_001 `
  --check-readable-images
```

Accept the smoke only if:

```text
ok: true
records > 0
image paths readable
state.elite_tcp_pose_6d is present
image_pose_time_delta_ms is reasonable
```

## Mode1 Pilot

Meaning:

```text
Piper periodically feeds.
Human controls Elite/magnetic guidance.
```

Use this only after physical setup and supervision are confirmed because this
command controls Piper.

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260705_left_mode1_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300 `
  --warmup-frames 30 `
  --collection-mode mode1_auto_piper_manual_elite `
  --piper-label-source auto_periodic `
  --piper-auto-feed-period 0.8 `
  --piper-auto-feed-count 14 `
  --stop-after-auto-feed-count `
  --enable-piper-control `
  --piper-control-source collector `
  --elite-control-source human `
  --session-note "mode1: collector periodic Piper feed, human Elite guidance, camera warmup 30 frames"
```

Validate immediately:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_20260705_left_mode1_001 `
  --check-readable-images
```

Then repeat for right by changing:

```text
--out real_pilot_20260705_right_mode1_001
--task right
```

## Mode2 Pilot

Meaning:

```text
Elite is moved by a human or senior-style external path script.
Collector controls Piper from keyboard feed/hold/retract labels.
```

Use this only after physical setup and supervision are confirmed because this
command controls Piper.

```powershell
.\.venv\Scripts\python.exe data\collect\collect_real_shadow_pilot.py `
  --out real_pilot_20260705_left_mode2_001 `
  --task left `
  --side-source realsense `
  --side-camera-id 1 `
  --top-source opencv `
  --top-camera-id 0 `
  --pose-source elite `
  --elite-ip 192.168.137.200 `
  --sample-period 0.2 `
  --max-frames 300 `
  --warmup-frames 30 `
  --collection-mode mode2_auto_elite_manual_piper `
  --piper-label-source keyboard `
  --enable-piper-control `
  --piper-control-source collector `
  --elite-control-source senior_script `
  --session-note "mode2: external Elite path, keyboard Piper labels"
```

Validate immediately:

```powershell
.\.venv\Scripts\python.exe tools\validate_real_shadow_pilot.py `
  real_pilot_20260705_left_mode2_001 `
  --check-readable-images
```

If Elite is moved manually rather than by the senior path script, use:

```text
--elite-control-source human
```

## After Each Run

Open `summary.json` and check:

```text
records
final_piper_step
auto_feed_count
piper_command_counts
piper_executed_counts
robot_command_mode
piper_label_source
collection_mode
```

For mode1, `final_piper_step` and `auto_feed_count` should match the intended
number of feed primitives.

For mode2, `piper_command_counts` should match what the operator actually did.

## If Something Fails

Common fixes:

```text
camera open failure: swap --side-camera-id / --top-camera-id
top camera unavailable: use --top-source duplicate_side and record this limitation
Elite connection failure: verify --elite-ip and network
Piper should not move: remove --enable-piper-control
Piper moved outside collector: use --external-piper-control, not --enable-piper-control
```

Do not keep collecting large data if validation fails.
