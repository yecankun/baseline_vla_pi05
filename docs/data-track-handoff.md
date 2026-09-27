# Data Track Handoff

Last updated: 2026-07-18

This document is for the agent continuing the unresolved data problems:
simulation data quality and real data collection. Algorithm innovation and
PI05/OpenPI model changes may be handled by a separate agent. Keep those two
threads connected through the data/action interface, but do not let model
experiments redefine data semantics silently.

Algorithm state and commands live in `docs/algorithm-track-handoff.md` and
`docs/algorithm-track-commands.md`. Do not mirror algorithm experiment history
here; keep only the shared data/action interface needed by the data track.

## Project Priority

The project priority is still:

```text
simulation realism and real-system correspondence > BC rollout success
```

Treat all current VLA/OpenPI/PI05 results as algorithm feasibility only. They
are useful for checking that the interface can train, but they are not
real-system validation.

The current policy-facing target contract is:

```text
observation + task instruction -> Elite TCP delta + Piper discrete intent
```

Execution remains controller-owned:

- Elite TCP delta -> IK -> joint execution.
- `piper_intent_id` remains the compatibility name for guidewire feed intent;
  current real execution uses the replacement feed device.
- Controller owns timing, cooldown, bounds, safety checks, and executed-command
  logs.

The 32D `action_32` tensor is only a framework compatibility surface:

```text
action_32 dims 0:6  = Elite TCP delta
action_32 dims 6:9  = Piper intent one-hot compatibility slice
action_32 dims 9:32 = padding
```

Do not treat the Piper one-hot compatibility slice as the final preferred
action head. Algorithm evidence for that decision lives in
`docs/algorithm-track-handoff.md`.

## Machine Split

Use the Windows workstation for simulation, export, data inspection, docs, and
small smoke checks:

```text
local project root: D:\PycharmProjects\project_2026
local Python: .\.venv\Scripts\python.exe
```

The Ubuntu 4090 is the algorithm track's LeRobot/OpenPI execution host. The data
track may transfer an accepted export there when the two tracks coordinate, but
should not run or diagnose model training from this handoff:

```text
remote host: cn-hk-bgp-4.ofalias.net
remote ssh port: 27455
remote user: zsw
remote project root: /home/zsw/project_2026
remote conda env: project2026-pi
```

SSH from Windows:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 27455 zsw@cn-hk-bgp-4.ofalias.net
```

Copy one file to the 4090:

```powershell
scp -P 27455 -i $env:USERPROFILE\.ssh\id_ed25519 `
  path\to\local_file.py `
  zsw@cn-hk-bgp-4.ofalias.net:/home/zsw/project_2026/path/to/local_file.py
```

Copy one remote output back:

```powershell
scp -P 27455 -i $env:USERPROFILE\.ssh\id_ed25519 `
  zsw@cn-hk-bgp-4.ofalias.net:/home/zsw/project_2026/path/to/output.json `
  path\to\local_output.json
```

Current algorithm execution commands belong in
`docs/algorithm-track-commands.md`.

## Algorithm Interface Needed By Data Track

Detailed model status belongs in `docs/algorithm-track-handoff.md`. The data
track only needs to preserve the agreed export boundary:

- Keep exporting explicit `elite_tcp_delta_6d` and `piper_intent_id`.
- Do not hide Piper intent inside a continuous action-only interpretation.
- Keep side/top images, state, task, and estimator/tactile provenance explicit.
- Keep `action_32` only as an optional framework compatibility tensor.
- Report interface or provenance problems to both tracks instead of changing
  model targets inside the data pipeline.

## Simulation Data Line

Current simulation mainline:

```text
simulation/mujoco_guided_wire_env.py
simulation.collect_formal_tip_line_guidance
```

Current guidewire assumption:

```text
tip-centric / hard-elastic guidewire with continuous line-shaped rendering
```

Current formal start range:

```text
--start-fraction-min 0.42
--start-fraction-max 0.54
```

Current accepted simulation direction:

- MuJoCo-first, not Isaac-first.
- Elite magnetic target should be front-up relative to the guidewire tip or
  registered route point, not directly overhead.
- Guidewire visual route should begin near the feeder/Piper outlet and pass
  through the vessel entry, including the S-bend-like entrance geometry.
- Background, lighting, camera visibility, line thickness, and blur/reflection
  should be compared against real side/top captures.
- Formal data must not depend on hidden MuJoCo truth unless a matching real
  observable path exists.

Important existing artifacts:

```text
simulation_output/formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1
simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1
simulation_output/lerobot_project2026_frontup_step024_v1
docs/_formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1_audit/
```

Important simulation tools:

```text
tools/audit_elite_frontup_guidance.py
tools/audit_pi_style_dataset.py
tools/prepare_pi_style_training_pack.py
tools/prepare_openpi_compat_pack.py
tools/prepare_openpi_temporal_view.py
tools/validate_openpi_compat_pack.py
tools/export_openpi_compat_to_lerobot.py
tools/audit_openpi_tactile_signal.py
tools/relabel_openpi_tactile_threshold.py
tools/build_contact_supervision_sidecar.py
tools/train_contact_estimator_oof.py
```

Current simulation data tasks:

1. Keep improving correspondence to real captures:
   - guidewire entry geometry;
   - camera domain;
   - Piper feed event scale;
   - Elite motion scale;
   - observable contact/tip estimator fields.
2. Keep tactile/contact fields explicit and provenance-aware.
3. Do not use exact MuJoCo contact/wall/tip truth as policy input unless it is
   only a diagnostic and clearly marked as such.
4. Before new formal synthetic data is accepted, run formal/provenance audits.
5. Use BC or PI-style training only as an interface regression check after data
   changes.

Recommended simulation acceptance checks:

```text
formal validity passes
complete side/top images
front-up Elite target audit positive
reasonable tip-to-magnet coupling
reasonable wall/contact visual review
Piper intent distribution is not degenerate
observation provenance contains no hidden oracle-only fields
```

### Current Temporal Test View

The first 4936-record algorithm export passed schema checks but was too
temporally redundant for a clean learning comparison. The data-track follow-up
created a no-relabel, uniform-stride-5 view:

```text
simulation_output/openpi_compat_pack_frontup_step024_temporal_stride5_v1
docs/simulation-training-data-temporal-audit-20260717.md
```

Current evidence:

```text
1005 samples / 40 complete episodes
34 train episodes / 6 validation episodes, task-stratified
previous Piper label accuracy: 90.36% -> 51.09%
previous Elite translation MAE: 0.2517 -> 0.9601
source Piper transitions retained: 472 / 472
complete image validation: 2010 checked / 0 missing
image references: Windows/Linux portable forward-slash form
self-contained transfer size: 83.48 MB
```

This view is accepted for a bounded Elite-translation plus Piper hold/feed
algorithm test. It does not solve visual-domain diversity, all-zero Elite
rotation, missing Piper retract, missing positive contact examples, or the
teacher-student observability decision. Do not use it for a tactile-benefit or
three-class-control claim.

### Contact Estimator Supervision Layer

The current formal dataset now has a separate offline-only contact supervision
sidecar:

```text
simulation_output/contact_supervision_frontup_step024_v1
docs/contact-supervision-sidecar-audit-20260718.md
```

It contains 4936 image-pair references and 411 diagnostic MuJoCo contact
positives across five whole-episode folds. All 9872 side/top references pass.
Exact contact stays under `diagnostic_target` with
`policy_input_allowed=false`; it is not exported into `state_32`.

The old RGB edge-distance estimator has zero recall/F1 against the sidecar.
The replacement RGB-local estimator completed full five-fold, whole-episode
OOF evaluation on all 4936 records:

```text
output: simulation_output/contact_estimator_oof_full_v1
RGB-local AP / F1: 0.99138 / 0.95848
precision / recall: 0.93519 / 0.98297
tip-position-only AP / F1: 0.82569 / 0.68216
left AP / F1: 0.99308 / 0.96732
right AP / F1: 0.98365 / 0.87179
all five fold AP values: 0.97521-0.99972
```

All 59 truth contact runs were detected; onset delay median/P95/max was
`0/0/1` sampled frames. There were 35 frame errors, 68.6% within one frame of
a truth transition. This passes the simulation estimator feasibility gate and
allows only a separate diagnostic estimator-driven contact-probe expert. It
does not make the source data suitable for a tactile-benefit claim, because the
source expert still lacks a clean contact-conditioned action change.

Real OOD audit output:

```text
simulation_output/contact_estimator_real_pilot_ood_v1
300 failed-pilot records / 5-model ensemble
predicted positives at 0.5: 0
probability P95 / max: 4.16e-43 / 1.26e-33
real contact truth: unavailable
policy_input_allowed: false
```

This is saturated negative domain-transfer behavior, not real accuracy
evidence. The real pilot remains calibration-only. Do not publish an all-zero
real `estimated_contact_flag`; keep it invalid until a manually annotated real
calibration set supports detector validation and recalibration.

### Diagnostic Contact-Probe Collector

A separate estimator-driven probe collector now exists:

```text
simulation/collect_estimated_contact_probe.py
docs/estimated-contact-probe-audit-20260718.md
```

It preserves explicit `elite_tcp_delta_6d` and `piper_intent_id`, keeps Piper
on the original feed/hold schedule, and only removes the Elite component toward
the planned wall side after an estimated-contact confirmation delay. It rejects
`--formal-data` and does not modify the mainline route-plan expert.

The left `2.0x`-radius smoke generated 11 exact-contact steps and 45 Elite
response steps, but the source-trained estimator had `TP/FP/FN = 0/41/11` on
that new trajectory. Initial right `n1` smokes produced no exact contact. A
dedicated geometry audit then found a usable right `n2/-1` route:

```text
start 0.42: radius fraction 0.95, 27/60 contact steps
start 0.48: radius fraction 1.10, 24/60 contact steps
start 0.54: radius fraction 0.95, 5/60 contact steps
```

The actual right dual-camera collector confirmed 27 exact-contact steps in 100
steps at `0.42/n2/-1/0.95`, but estimator recall was still zero (`FN=27`). The
top view was occluded during a reviewed contact frame. Therefore the collector
is currently useful only for generating additional offline contact
supervision. Its estimator fields are marked
`policy_input_allowed=false`, and the dataset must not yet be handed to the
algorithm track as contact-conditioned expert data.

The first long left supervision bucket is complete:

```text
simulation_output/probe_left_start042_v1
3 episodes / 540 samples / 0 missing images
exact contact: 153
estimator TP/FP/FN per episode: 4/40/47, 4/41/47, 4/41/47
```

This confirms the collector can produce dense offline contact supervision, but
also confirms that augmented estimator OOF retraining is required before any
policy-facing export.

The right 0.42 bucket is complete as well:

```text
simulation_output/probe_right_start042_v1
3 episodes / 540 samples / 0 missing images
exact contact: 81
estimator TP/FP/FN per episode: 0/0/27 for all three episodes
```

It is a hard-positive supervision bucket only; the current RGB estimator misses
all of its contact frames.

The right 0.48 bucket is complete:

```text
simulation_output/probe_right_start048_v1
3 episodes / 540 samples / 0 missing images
exact contact: 153
estimator TP/FP/FN per episode: 0/0/51 for all three episodes
```

This provides a denser right-side positive segment, but remains offline
supervision only.

The right 0.54 bucket is complete:

```text
simulation_output/probe_right_start054_v1
3 episodes / 540 samples / 0 missing images
exact contact: 15
estimator TP/FP/FN per episode: 0/0/5 for all three episodes
```

All four planned buckets are now ready for merge and augmented OOF training.

The merge and sidecar stages are complete:

```text
simulation_output/probe_supervision_merged_v1/manifest.json
simulation_output/contact_supervision_probe_augmented_v1/manifest.json
52 episodes / 7096 samples / 14192 checked images / 0 missing
exact-contact positives: 813 (left 530, right 283)
```

The next required artifact is `contact_estimator_oof_probe_augmented_v1`; keep
all estimator fields policy-disabled until its probe-held-out metrics pass.

The augmented OOF artifact now exists, but promotion is rejected:

```text
overall RGB AP/F1: 0.9917/0.9269
right 0.54 AP/F1: 0.5377/0.2500
right 0.54 precision/recall: 0.1429/1.0000
augmented real pilot positives: 0/300
```

The right 0.54 errors are temporally structured: truth is steps `44-48`, while
the model predicts `19-52` plus step `54` in every episode. Do not duplicate
that bucket or silently change the threshold. Define a separate risk flag or a
boundary-aware temporal estimator before another OOF/OOD run. Do not set
`policy_input_allowed=true` from the aggregate score.

The temporal audit is complete at
`simulation_output/contact_estimator_oof_probe_augmented_v1/temporal_risk_audit.json`.
It confirms right 0.54 behaves as an early risk warning (median lead 25 frames,
three pure risk runs), while the other probe buckets align with exact contact.
The next interface decision is whether to add a separate
`estimated_contact_risk_flag`; do not overload `estimated_contact_flag`.

The policy-safe diagnostic pack is now materialized at:

```text
simulation_output/contact_risk_diagnostic_pack_v1/manifest.json
simulation_output/contact_risk_diagnostic_pack_v1/samples.jsonl
simulation_output/contact_risk_diagnostic_pack_v1/diagnostic_targets.jsonl
```

It contains 7096 samples from 52 complete episodes, 14192 checked side/top
image references with no missing files, and 813 exact-contact positives in the
separate diagnostic target file. `samples.jsonl` contains only observable
state, images, the optional early-warning risk flag/probability, and the
explicit `elite_tcp_delta_6d + piper_intent_id` target. Legacy feed/hold labels
are normalized to `retract=0`, `hold=1`, `feed=2`; no action semantics changed.
Exact contact, wall distance, and risk conditioning all remain
`policy_input_allowed=false`. This pack is suitable for interface/diagnostic
inspection only, not for a tactile-benefit or real-system readiness claim.

## Real Collection Line

Current real data entrypoint:

```text
data/collect/collect_real_shadow_pilot.py
```

Current onsite execution environment, confirmed 2026-07-21:

```text
SSH: zsw@192.168.5.11
real collection root: /home/zsw/PycharmProjects/real_collection
real collection Python: /media/zsw/SSD1T/conda_piper/envs/sam3/bin/python
algorithm/LeRobot root: /home/zsw/project_2026
```

Run real collection from `real_collection`, whose interpreter contains the
Elite SDK. Keep `/home/zsw/project_2026` for the LeRobot/algorithm environment.
Synchronize the project-facing collector and hardware adapter into both roots
when those files change; do not assume the two interpreters are interchangeable.

The onsite operator runs collection directly on the lab computer. SSH is only
for file inspection and synchronization, not for keyboard control or preview.
The current linked-pilot workflow is fully manual: `n` advances Elite by one
preset trajectory point and `f` sends one forward feeder event. These actions
are independently timed by the operator; do not enable
`--piper-feed-on-elite-path-step`, Elite auto playback, or periodic auto feed.
The feeder backend is forward-only by default and blocks retract keys.
The linked pilot uses `--start-recording-on-first-action`: post-approach preview
frames are not saved until the first valid operator action, and that action is
preserved as sample `0`.

The checked `path1/path2` files do not match the latest observed Elite pose:
their first point is about `346 mm` away and their nearest checked point remains
more than `234 mm` away. Repeated collection now uses the explicit
`--elite-path-approach-start` startup phase: Elite moves directly to the first
selected path point before recording begins, the reset motion is excluded from
training records, and the first operator `n` command advances to the following
point. The initial pose/distance and final approach error remain logged. Without
this option, execution still fail-closes above
`--elite-path-max-start-distance-mm` (default `30 mm`).

The first automatic approach attempt
`real_pilot_20260721_180414_left_s_bend_manual_linked` was stopped before the
first path point because the collector used `move_joint` for the long reset,
which produced a twisted joint-space TCP path. It contains only a manifest and
is not data. The corrected startup primitive is Elite `move_line` with explicit
Cartesian speed semantics (`speed_type=0`); subsequent short trajectory steps
retain the inherited `move_joint` behavior.

Current real validation/render tools:

```text
tools/validate_real_shadow_pilot.py
tools/render_real_shadow_video.py
```

Known real system details:

```text
Elite IP used onsite: 192.168.5.66
Piper interface: can0
Previously observed camera IDs: side=10, top=4
Camera IDs can change after unplug/replug
Camera warmup should be about 20-30 frames
```

The current camera update binds side/top RealSense devices by SDK serial
`317222072584` / `317222071938`, requests `1920x1080@15` color plus
`1280x720@15` native depth, aligns depth to color, and stores raw `uint16` depth
PNG paths in optional `side_depth` / `top_depth` record fields. Runtime stream
profiles, intrinsics, depth-to-color extrinsics, and depth scale belong in the
manifest. Depth through the transparent vessel is auxiliary calibration data,
not an exact contact label.

Do not pass the udev `ID_SERIAL_SHORT` values `318123025778` / `318123027596`
to `pyrealsense2.config.enable_device`; those identify the UVC shell rather than
the RealSense SDK devices and produce `RuntimeError: No device connected`.

The corrected dual-camera RGB-D smoke is accepted at
`real_diag_20260721_192500_dual_rgbd_1080p_warmup100`: 100 unique warmup frames
followed by two records, side/top color `1920x1080`, aligned side/top depth
`1920x1080 uint16`, all four image files readable, and zero validator
errors/warnings. Background latest-frame capture reduced the side/top
host-arrival timestamp delta from the serial-read smoke's `57.18 ms` to
`25.75 ms`. After the real warmup, nonzero aligned-depth coverage was about
`81.3%` for side and `73.9-74.3%` for top. This validates the collection
interface and startup stabilization, not transparent-vessel depth accuracy or
contact inference.

Known real hardware status:

- The JSONL/image/pose recording path can produce synchronized records when the
  hardware behaves.
- Physical guidewire delivery remains the main blocker.
- Removing the physical S-bend helped, but did not fully solve delivery.
- The UDP feeder protocol works. After an onsite mechanical adjustment on
  2026-07-21, the replacement feed device completed a basic usable feed test.
  Stable S-bend delivery, repeatability, and synchronized capture are not yet
  verified, so it remains a diagnostic candidate rather than accepted expert
  collection hardware.
- The collector's Python `socket.sendto` path returned without error onsite but
  did not move the device, while the same JSON sent through Bash `/dev/udp`
  worked. Use the explicit `--feeder-udp-transport bash_dev_udp` diagnostic
  mode until the Python-socket discrepancy is resolved.
- The exact Bash `printf` transport completed a physically observed single-step
  movement in `real_diag_20260721_s_bend_feeder_printf_single_004`. Its 40
  records validated with zero errors/warnings; the executed log contains one
  `feed_feeder_bash_dev_udp_sent`, side/top sync P95 is 4.07 ms, and
  image/action sync P95 is 14.20 ms. This closes the single-step synchronized
  diagnostic gate, not the repeatability or expert-data gate.
- The first five-step repeatability run
  `real_diag_20260721_s_bend_feeder_printf_repeat5_005` validated 60 records
  with five `feed_feeder_bash_dev_udp_sent` events and no errors/warnings. All
  five commands produced physical forward movement, but one movement was
  visibly shorter. Side/top sync P95 was 30.33 ms and image/action sync P95 was
  33.99 ms. Repeated two-second-period trials ruled out command cooldown as the
  main cause: some initializations delivered normally, while others produced
  almost no effective progress. The current onsite interpretation is subtle
  initial guidewire-state variation causing local impingement in the vessel
  model, not proven feeder-output variability. The nominal `60 mm` summary is
  command-derived and is not measured displacement. After the substantial
  onsite device adjustment, the old `12 mm/step`, `11-13 mm` range, and all
  derived `60 mm` values are invalidated. New captures must leave insertion
  millimeters null. Fixed millimeters per feed are not a valid calibration
  target for the curved, initialization-sensitive vessel setup; physical
  progress must remain image/observation derived.
- Current July 2026 real captures are calibration/hardware-diagnostic data
  unless explicitly reclassified as clean demonstrations.
- `real_pilot_20260721_192749_left_s_bend_rgbd` is the first physically
  successful S-bend linked feeder-plus-Elite episode: 350 complete RGB-D
  records and nine executed feeder events. It remains a physical-success and
  visual-reference candidate rather than accepted training data because the
  shared Elite SDK connection blocked pose polling during `move_joint`;
  `148/350` records (`42.3%`) used stale poses and image-to-pose P95 was
  `1.76 s`. The next capture must use
  `--elite-path-separate-connection` and pass the pose-sync audit before more
  successful episodes are collected.
- `real_pilot_20260721_200628_left_s_bend_rgbd` is the first clean successful
  candidate under the corrected connection mode: 310 complete `1920x1080`
  RGB-D records, eight operator-confirmed feeder events, zero stale Elite
  poses, image/pose P95 `81.23 ms`, and image/action P95 `87.92 ms`. The
  validator found zero errors and five isolated synchronization warnings.
  Retain it for detailed visual/action audit before algorithm handoff.
- The immediately following `200813` and `201018` episodes were physically
  unsuccessful after seven and two logged feeder events, respectively, while
  their data interfaces and Elite pose synchronization remained valid. The
  initial onsite hypothesis was feeder thermal degradation. Later inspection
  found that the hot-melt-adhesive fixture had loosened after heating and could
  no longer clamp the guidewire consistently. Stop the current dynamic
  collection session rather than chase another success under changing hardware
  state; future sessions should begin cold and record powered-on/cooldown time.
- The final right-route reference is
  `real_pilot_20260721_203449_right_s_bend_rgbd`: 183 complete RGB-D records,
  zero stale Elite poses, and final Elite path index `16/19`. The operator
  confirmed that the guidewire passed the bifurcation before the last several
  feed commands stopped producing physical motion. Treat it as a valuable
  right-route prefix plus actuator-failure suffix, not a complete expert
  success. The full session classification is in
  `docs/real-collection-onsite-audit-20260721.md`.

Do first onsite:

1. Verify camera IDs and warmup.
2. Verify image synchronization.
3. Verify Elite connection and servo state.
4. Verify CAN/Piper bring-up before assuming Python-side failure.
5. Test guidewire physical delivery before collecting long data.
6. Stop early if the guidewire mechanically cannot pass.

Useful camera probe pattern:

```bash
python - <<'PY'
import cv2
for i in range(12):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    print(i, ok, None if frame is None else frame.shape)
    cap.release()
PY
```

Common CAN/Piper issue:

```text
can0 may be STOPPED, missing, or ERROR-ACTIVE after reboot/replug.
Bring up can0 before running collection with Piper control.
```

Use existing command notes in `docs/commands.md` for current real collection
templates. After every real collection, run validation immediately and render a
quick video if the dataset is intended for review.

Real dataset format:

```text
records.jsonl
frames/side/*.png
frames/top/*.png
manifest or summary json
Elite TCP pose/state
Piper state/action label
timestamps and timing diagnostics
```

Real data interpretation:

- `branchs/` is senior-provided real image and Elite/magnetic-arm motion
  reference data.
- Do not use `branchs/` as guidewire centerline or guidewire trajectory truth.
- Use current onsite captures mainly for calibration/alignment and failure
  analysis until the physical delivery path is reliable.

## Data Interface Checklist

For both simulation and real data, the preferred VLA-facing sample should
eventually expose:

```text
observation:
  side image
  top image
  Elite TCP pose/state
  Piper insertion/state/action history
  task / branch instruction
  image-derived tactile/contact fields with confidence/provenance

target:
  elite_tcp_delta_6d
  piper_intent_id

optional compatibility:
  state_32
  action_32
```

Keep these fields explicit:

```text
elite_tcp_delta_6d: continuous target
piper_intent_id: discrete target
estimated_contact_flag: image-derived or null/invalid
estimated_contact_risk_flag: optional early near-wall/contact warning; not a strict contact label
estimated_image_distance_px: image-derived or null/invalid
tactile/contact confidence: explicit confidence, not hidden truth
```

Do not silently create:

```text
centerline labels from visual wall clicks
guidewire path truth from branchs pose
real contact labels from MuJoCo exact contact
real retract labels unless real retract was commanded/logged
```

## Recommended Next Work For This Agent

Simulation track:

1. Use the audited right `n2/-1` geometry with explicit start buckets; do not
   collapse `0.42/0.48/0.54` to one global radius fraction.
2. Use the four bucket commands in `docs/commands.md` to create additional
   offline supervision, then merge the original formal source with probe
   manifests using `tools/merge_probe_manifests.py`.
3. Retrain with whole probe episodes held out and report source/probe metrics
   separately.
4. Require nonzero held-out probe-contact recall and temporal onset performance
   before setting probe estimator fields `policy_input_allowed=true` or sending
   contact-conditioned expert data to the algorithm track.
5. Keep the behavior Elite-only at contact. Do not add exact-contact feedback
   or strong Piper hold/retract recovery.
6. Re-audit the current front-up dataset against real visual anchors.
7. Identify which sim-to-real gaps are still visible:
   - glass reflection and blur;
   - guidewire thickness and brightness;
   - camera placement;
   - Piper feed event scale;
   - Elite motion scale;
   - contact/risk estimator realism.
8. Make small simulator/data-export changes only when they improve
   real-system correspondence.
9. Re-export packs only after a meaningful simulator/data-interface change.

Real collection track:

1. Stop feed-period tuning. Standardize and record the initial guidewire state:
   axial reference mark, entry angle, visible slack/curvature, and pre-feed
   side/top frames. Use a unique output directory for every trial.
2. Retain both clear-progress and locally blocked feeder-only trials. Compare
   their initial frames and suspected blockage location; treat `piper_step` as
   executed-command count, not physical insertion truth.
3. Do not spend further onsite time on a Piper-only comparison. Use the
   replacement feed device as the real feed backend and proceed to linked
   feeder-plus-Elite guidance trials after the initial-state procedure is
   stable.
4. For the linked pilot, let the operator independently trigger Elite next-point
   and feeder-forward events from the local preview. Use the unrecorded automatic
   Elite start approach on every episode; never auto-couple path steps and feed.
5. Keep every failed or partial run validated and video-rendered immediately;
   treat it as calibration/failure-analysis data, not clean expert data.

Cross-track:

1. Use real captures to calibrate simulation, not to claim policy success.
2. Keep algorithm handoff fields stable: `elite_tcp_delta_6d`,
   `piper_intent_id`, side/top images, and contact/tactile metadata.
3. Record durable findings in `docs/weekly-meeting-log.md`; do not bury them
   only in chat.

## Files To Read First

For a fresh data-track agent:

```text
AGENTS.md
.codex/skills/project-2026-guidewire-mujoco/SKILL.md
docs/project-state.md
docs/handoff.md
docs/data-track-handoff.md
docs/commands.md
docs/real-data-collection-checklist.md
docs/real-alignment-20260707-calibration-audit.md
docs/vla-target-interface.md
docs/control-layer-contract.md
```

Open archived docs only when a specific historical result is needed.
