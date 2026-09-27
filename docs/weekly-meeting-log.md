# Weekly Meeting Log

This document is an append-only fact log for weekly group meetings. It should
record durable facts that can later be used to generate a meeting summary, not
the summary itself.

## Writing Rules

- Write facts, not polished summaries.
- Prefer append-only entries. If an older fact becomes outdated, append a newer
  correction or superseding fact instead of rewriting history.
- Keep each fact short enough to be quoted or filtered later.
- Record evidence paths when available: dataset names, script names, output
  directories, validation reports, or important logs.
- Separate simulation/software facts from real-system/hardware facts.
- Do not record every command, parameter tweak, or intermediate debugging step
  unless it changes a future decision.

## Update Triggers

Append entries when one of the following happens:

- A meaningful implementation, collection, training, rollout, or diagnostic
  milestone completes.
- A new finding changes how previous results should be interpreted.
- A failed experiment creates a reusable negative conclusion.
- The project direction, priority, or next-step plan changes.
- Onsite real-system work starts or ends.
- A reusable command, workflow, checklist, hardware setup, or recovery process
  is established.
- Supervisor or group-meeting feedback changes the plan.
- Before each weekly group meeting, to consolidate facts from the current week.

Do not append for ordinary trial-and-error, small parameter changes, temporary
ideas, or work that does not affect future reporting or decisions.

## Entry Format

Use one dated section per meeting week. Keep the newest week at the top.

```markdown
## YYYY-MM-DD

### Facts

- [simulation] Fact. Evidence: path or command output.
- [real-system] Fact. Evidence: path or validation output.
- [data] Fact. Evidence: dataset or manifest.
- [model] Fact. Evidence: diagnostic directory or metric.
- [decision] Fact. Evidence: discussion, doc, or accepted result.
- [blocker] Fact. Evidence: observed failure or hardware status.

### Open Follow-Ups

- Follow-up item that remains unresolved.
```

## 2026-08-10

### Facts

- [simulation][decision] The native MuJoCo cable mainline now has a visually
  accepted real-aligned scene, complete registered vessel collision, a stable
  root-feed boundary, gravity/wall contact, locked Elite geometry, and
  left/right branch-local straight-seed placement. It replaces the earlier
  centerline-shaped visual wire as the physical-behavior research path, but it
  is not yet authorized for formal data generation. Evidence:
  `docs/data-track-handoff.md`, D1.12-D1.13 accepted audit chain.
- [simulation][decision] Branch-local feed experiments established that the
  cable can be advanced through the bifurcation without abnormal branch
  switching or vessel escape under bounded diagnostic controls. They also
  showed that gravity, wall contact, feed, and magnetic steering must be
  separated causally; visual success alone is insufficient. Evidence:
  D1.13m-D1.13o reports and reviewed videos under `simulation_output/`.
- [simulation][decision] The D1.13p fixed task-dependent branch-force proxy is
  rejected because its left magnetic direction is not physically tied to the
  visible Elite magnet or a magnetic field. Its magnitude and stop threshold
  must not be tuned. Evidence:
  `simulation/bifurcation_root_feed_lateral_successor_v2_formal_integration_D1_13p_visual_review_v1.json`.
- [simulation][decision] D1.14a replaced the rejected branch-force direction
  with the accepted Elite magnet trajectory and a direction-only point-dipole
  field geometry. The geometry sheet passed human review, but did not validate
  real field magnitude or dynamic steering. Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_14a_magnetic_pose_field_geometry_audit_v1/`.
- [simulation][decision] The one predeclared D1.14b torque-on/off dynamic suite
  fails closed at `20/25`. Left/right paired final-tip differences are only
  `0.099/0.743 mm`; right torque-on remains nearest the left branch, and
  torque-on left/right tip separation is only `0.738 mm`. All four runs remain
  finite and within the penetration gate, so this is valid negative behavior
  evidence rather than a solver failure. Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_14b_torque_dominant_short_dynamic_smoke_v1/`.
- [simulation][decision] The visually accepted D1.14c read-only temporal audit
  shows no sustained left shape response. On the right, contact-count
  divergence begins at step `2115` and sustained shape divergence begins only
  at step `2867`, supporting a delayed contact-coupled interpretation rather
  than direct consistent torque steering. Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_14c_readonly_temporal_causal_audit_v1/`,
  `simulation/bifurcation_root_feed_lateral_successor_v2_formal_integration_D1_14c_visual_review_v1.json`.
- [simulation][decision] D1.14d passes its read-only effective-moment audit
  `9/9`: torque sign is correct and adjacent torques do not materially cancel,
  but the proximal active-joint moment proxy reaches only `32.4%/60.5%` of the
  `M=EI*kappa` reference for left/right and the tip proxy only `12.2%/20.9%`.
  This identifies structural underdrive consistent with D1.14b, without
  proving a unique cause or validating a larger real torque. The Chinese
  figure passed user review. Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_14d_effective_moment_transfer_audit_v1/`,
  `simulation/bifurcation_root_feed_lateral_successor_v2_formal_integration_D1_14d_visual_review_v1.json`.
- [simulation][decision] The magnetic actuation semantics were corrected after
  the user clarified that only the red guidewire head is magnetically
  responsive and may be modeled as attractable soft-ferromagnetic metal such
  as iron. The following cable is passive. This supersedes the D1.14b
  distributed permanent-dipole interpretation: the next audit must be
  tip-only and use field-strength-gradient attraction, with optional
  head-tail-symmetric axis alignment for an elongated head. Evidence:
  `docs/data-track-handoff.md`, D1.15 model decision.
- [simulation][correction] A subsequently provided senior-thesis excerpt
  supersedes the soft-ferromagnetic attraction-only hypothesis. The inherited
  model uses an external point-dipole field and a terminal magnetic unit with
  equivalent fixed dipole moment, receiving both
  `F=grad(m_tip dot B_ext)` and `T=m_tip cross B_ext`. Only the red tip is
  magnetic; the following cable remains passive. Tip material, polarity,
  moment magnitude, and field magnitude are still unconfirmed. Evidence:
  user-provided thesis sections 2.3.2-2.3.3; `docs/data-track-handoff.md`,
  D1.15 evidence correction.
- [simulation][decision] D1.15a passes its compile/read-only ownership and
  relative-polarity audit `12/12`, pending visual review. The current red tip
  marker is a separate, noncolliding diagnostic body; magnetic actuation must
  instead belong only to `formal_cable_t3_physical_B_last`. Under the current
  external `+Z` convention, `m_tip=+tangent` repels at all six registered
  samples, while `m_tip=-tangent` attracts at all six with force-to-Elite angle
  `0.908-13.986 deg`, matching the supplied real behavior. This selects only a
  relative polarity class, not absolute N/S labels or magnitude. Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_15a_tip_only_dipole_ownership_polarity_audit_v1/`.
- [data][interface] The data/simulation track continues to preserve
  `observation + task instruction -> elite_tcp_delta_6d + piper_intent_id`.
  Exact tip/contact/wall/route truth remains diagnostic-only; the current
  physical-cable and magnetic results are not formal policy data.
- [real-system][blocker] Stable new real demonstrations remain blocked pending
  completion of the redesigned feed device. Existing onsite captures remain
  alignment, hardware, and partial-trajectory evidence rather than clean
  training demonstrations.

### Open Follow-Ups

- Confirm red-head physical-body ownership and audit both axial polarity
  hypotheses for the tip-only fixed equivalent dipole. Compute direction-only
  `grad(m_tip dot B_ext)` and `m_tip cross B_ext`, then obtain material,
  polarity, moment, and field-magnitude evidence before any dynamic scale.
- After feeder redesign, collect a small number of repeatable left/right real
  demonstrations and use them to constrain feed response and sim-to-real
  plausibility.
- Reopen expert diversity and formal data generation only after the physical
  cable actuation path has a real-observable, physically credible steering
  model; do not scale the current diagnostic magnetic route.

## 2026-07-13

### Facts

- [real-system] The real shadow collection path can write synchronized side/top
  image records, Elite TCP pose, Piper state/action labels, controller logs, and
  timing diagnostics. Evidence: `data/collect/collect_real_shadow_pilot.py`,
  `tools/validate_real_shadow_pilot.py`.
- [real-system] Real collection videos can be reconstructed from frame PNGs and
  `records.jsonl` timestamps. Evidence: `tools/render_real_shadow_video.py`,
  smoke output `simulation_output/_smoke_real_shadow_video.mp4`.
- [docs] Startup docs were slimmed so new agents read short entry files first
  and recover old detail from archive only when needed. Evidence: `AGENTS.md`,
  `.codex/skills/project-2026-guidewire-mujoco/SKILL.md`, `docs/archive/`.
- [real-system] Onsite validation showed the Python/CAN/Piper control chain can
  recover and execute a one-step Piper feed after `can0` is restored.
  Evidence: onsite Piper `step_forward(pause_time=0.8)` smoke.
- [real-system] `can0` can enter `STOPPED` or disappear after device/reboot
  events; CAN/Piper bring-up must be checked before collection. Evidence:
  observed `can state STOPPED`, `Cannot find device "can0"`, and later
  `ERROR-ACTIVE` recovery.
- [real-system] The UDP feeder device protocol was identified and tested:
  send UDP JSON to `192.168.5.22:8888` with actions `forward`, `backward`,
  `turn_left`, and `turn_right`. Evidence: `hardware/feeder_device/` adapter.
- [real-system] UDP feeder forward/backward commands work, with nominal measured
  motion around `11-13 mm` per step, but repeated trials were not reliable
  enough for clean mainline real expert data. Evidence: onsite feeder tests.
- [real-system] Removing the physical entry S-bend improved feed-only behavior
  but did not fully solve guidewire delivery. Evidence: post-removal onsite
  smoke; current conclusion is that the S-bend was one blocker, not the only
  blocker.
- [data] The latest onsite real captures should be treated as calibration and
  hardware-diagnostic data, not clean expert demonstrations. Evidence:
  `collected_data/`, `docs/real-alignment-20260707-calibration-audit.md`.
- [simulation] The MuJoCo mainline remains tip-centric / hard-elastic guidewire
  with continuous line-shaped rendering and real-aligned action semantics:
  Elite TCP delta plus Piper discrete intent. Evidence:
  `simulation/collect_formal_tip_line_guidance.py`,
  `docs/project-state.md`.
- [decision] Current priority remains simulation realism and real-system
  correspondence over small-BC rollout success. Evidence:
  `docs/project-state.md`, `docs/handoff.md`.
- [blocker] Clean real data collection is currently limited by physical
  guidewire delivery reliability, not by the JSONL recording format alone.
  Evidence: onsite Piper/feeder trials and validation results.
- [docs] `docs/project-state.md` and `docs/handoff.md` were slimmed into
  current-state entry documents; their long historical versions were archived
  under `docs/archive/`. Evidence:
  `docs/archive/project-state-before-entry-slimming-2026-07-13.md`,
  `docs/archive/handoff-before-entry-slimming-2026-07-13.md`.
- [decision] Supervisor feedback after the 2026-07-13 group meeting was to keep
  trying the original real collection method; if it still fails mechanically,
  try a vessel model without the S-bend. Evidence: user group-meeting report.
- [decision] The project should not spend all effort on simulator/collection
  refinement before algorithms; start VLA-style feasibility tests with existing
  real/sim data while continuing data collection and alignment. Evidence: user
  group-meeting report.
- [decision] Initial algorithm tests should use small multimodal/VLA-style
  models first, preferably 3B or smaller and at most around 7B for early
  feasibility. Inputs should include image-derived tactile/contact signals from
  visual collision or distance analysis. Evidence: user algorithm-scope
  decision.
- [decision] The first practical pi-family baseline is openpi/pi0.5-style,
  adapted for project-specific tactile/contact input and Elite/Piper action
  semantics. Newer pi-family work such as pi0.7 is design inspiration, not the
  first reproducible baseline target. Evidence: `docs/vla-pi-baseline-plan.md`.
- [decision] The accepted algorithm framing is openpi/pi0.5-style reproducible
  baseline plus pi0.7-style context conditioning; the first context channel is
  image-derived tactile/contact estimation. Evidence:
  `docs/vla-pi-baseline-plan.md`.
- [data] A pi-style dataset audit tool was added to check whether sim manifests
  and real `records.jsonl` captures can map to the planned openpi/pi0.5-style
  baseline schema. Evidence: `tools/audit_pi_style_dataset.py`,
  `simulation_output/pi_style_audit_smoke_sim.json`,
  `simulation_output/pi_style_audit_smoke_real.json`.
- [simulation] Elite magnetic guidance semantics were corrected before the
  next dataset export: the magnetic target should be front-up relative to the
  guidewire tip or registered route point, not directly overhead. Evidence:
  `simulation/mujoco_guided_wire_env.py`, `docs/simulation-assumptions.md`.
- [data] Front-up route-plan dataset
  `simulation_output/formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1`
  collected 40/40 successful episodes. `tools/audit_elite_frontup_guidance.py`
  reports positive actual Elite magnet forward projection for all episodes
  with median episode p50 about `+1.21 cm`, and observation provenance audit
  passes for `real_direct_plus_estimated_tip_registered_geometry`. Evidence:
  `simulation_output/formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1/elite_frontup_audit.json`,
  `docs/_formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1_audit/`.
- [model] The first pi-style training-pack smoke model trained on
  `simulation_output/pi_style_pack_frontup_step024_esttip_reggeom_v1` completed
  and learned above majority baseline on the validation split. Best epoch was
  7/8 with `val_piper_acc=0.661`, versus majority baseline `0.516`, and
  `val_elite_raw_mae=0.344 mm/rad averaged over six TCP-delta channels`.
  Evidence:
  `simulation_output/pi_style_smoke_model_frontup_step024_esttip_reggeom_v1/train_log.json`,
  `simulation_output/pi_style_smoke_model_frontup_step024_esttip_reggeom_v1/open_loop_pack_diag.json`.
- [model] Ubuntu 4090 training environment was brought up for pi-style smoke
  tests, and the transferred dataset bundle reproduced the local smoke-training
  scale. Evidence:
  `simulation_output/pi_style_4090_bundle_frontup_step024_esttip_reggeom_v1/`
  on the 4090 machine and user-reported 8-epoch smoke metrics.
- [model] An OpenPI-compatible 32D state/action adapter pack was implemented
  while preserving the project-preferred mixed target semantics. The 4090
  validation passed with `4936` samples, no non-finite arrays, and
  `missing_sampled_images=0`. Evidence:
  `tools/prepare_openpi_compat_pack.py`,
  `tools/validate_openpi_compat_pack.py`,
  `simulation_output/openpi_compat_pack_frontup_step024_esttip_reggeom_v1_validate.json`
  on the 4090 machine.
- [model] The first OpenPI-compatible mixed-head smoke model trained on the
  4090 machine. Best reported epoch was 7/8 with `val_loss=0.71343`,
  `val_piper_acc=0.685`, `val_action_mae=0.3050`, and
  `val_elite_mae=0.3321`. Evidence:
  `tools/train_openpi_compat_smoke.py`,
  `simulation_output/openpi_compat_smoke_model_frontup_step024_esttip_reggeom_v1/`
  on the 4090 machine.
- [model] Tactile/contact ablations on the OpenPI-compatible smoke path
  (`drop_tactile_values` and `drop_tactile_all`) produced metrics similar to
  the full-context run. Current conclusion: the present synthetic tactile
  fields are wired into the model interface, but their value has not yet been
  demonstrated by this dataset/model. Evidence:
  `tools/train_openpi_compat_smoke.py`,
  `simulation_output/openpi_compat_smoke_model_frontup_step024_notactile_values_v1/`,
  `simulation_output/openpi_compat_smoke_model_frontup_step024_notactile_all_v1/`
  on the 4090 machine.
- [model] Tactile-signal audit explains the weak ablation effect:
  `estimated_contact_flag` is constant zero over all `4936` samples; confidence
  and image-distance fields vary, but have weak linear correlation with Piper
  feed labels and Elite movement scale. Evidence:
  `tools/audit_openpi_tactile_signal.py`,
  `simulation_output/openpi_tactile_signal_audit_frontup_step024_esttip_reggeom_v1.json`.
- [model] A posthoc contact-rich control pack was generated by thresholding
  `estimated_image_distance_px` to about `15%` positives, but full tactile
  training and `drop_tactile_all` training still produced similar metrics.
  Current conclusion: the weak tactile effect is not only caused by the
  original all-zero contact flag; this action target/dataset does not yet make
  the distance-like tactile signal important. Evidence:
  `tools/relabel_openpi_tactile_threshold.py`,
  `simulation_output/openpi_compat_pack_frontup_step024_contactrich_q15_v1/`
  and corresponding 4090 full/drop training outputs.
- [model] The OpenPI-compatible pack was exported through the official
  LeRobotDataset API in a small smoke run on the 4090 machine. The exporter
  keeps `action_32` as a framework compatibility tensor and preserves
  `elite_tcp_delta_6d` plus `piper_intent_id` as explicit project targets.
  Evidence: `tools/export_openpi_compat_to_lerobot.py`,
  `simulation_output/lerobot_project2026_frontup_step024_smoke/` on the 4090
  machine.
- [model] A LeRobotDataset-loader smoke model trained successfully on the full
  exported LeRobot dataset. This verifies the official LeRobot loader path,
  episode-level train/val split, side/top image tensors, `action_32`,
  `elite_tcp_delta_6d`, and `piper_intent_id` fields before moving to OpenPI
  model code. Evidence: `tools/train_lerobot_compat_smoke.py`,
  `simulation_output/lerobot_compat_smoke_model_frontup_step024_v1/` on the
  4090 machine.
- [model] The official LeRobot PI05 adapter path reached forward/backward on
  the project LeRobot dataset. The successful probe used
  `paligemma_variant=gemma_2b`, `action_expert_variant=gemma_300m`,
  `dtype=bfloat16`, frozen vision encoder, expert-only training, and gradient
  checkpointing; it reported `loss=1.4252` and valid gradients. This verifies
  PI05 interface feasibility only, not real-system validation. Evidence:
  `tools/probe_pi05_lerobot_adapter.py`,
  `simulation_output/pi05_lerobot_adapter_probe_preprocess_frontup_step024_v1.json`,
  `simulation_output/pi05_lerobot_adapter_probe_forward_backward_frontup_step024_v2.json`
  on the 4090 machine.
- [model] PI05 adapter training passed a micro-batch overfit diagnostic. A
  normal 40-step shuffled smoke did not improve validation
  (`initial_val_loss=1.4671`, `final_val_loss=1.5115`), but a fixed-noise
  repeated-batch overfit run reduced train loss from `1.2840` to `6.55e-05`
  over 80 steps and changed the small validation subset from `1.4953` to
  `1.1758`. Current interpretation: the PI05 optimizer/data/model wiring is
  functional, while broader policy quality still needs longer and more stable
  evaluation. Evidence: `tools/train_pi05_lerobot_adapter.py`,
  `simulation_output/pi05_lerobot_adapter_train_smoke_frontup_step024_v1/`,
  `simulation_output/pi05_lerobot_adapter_overfit_smoke_frontup_step024_v1/`
  on the 4090 machine.
- [model] A 300-step normal PI05 adapter-training smoke with episode-level
  split and fixed evaluation seed showed stable validation loss reduction.
  The run used `1649` train samples and `399` validation samples; validation
  loss moved from `1.4810` initially to `0.9837` at step 300, with intermediate
  checkpoints generally decreasing. Current interpretation: the official PI05
  adapter is trainable on the current synthetic LeRobot dataset, still as
  sim-data algorithm feasibility only. Evidence:
  `tools/train_pi05_lerobot_adapter.py`,
  `simulation_output/pi05_lerobot_adapter_train_smoke_frontup_step024_v2/` on
  the 4090 machine.
- [model] A checkpointed 1000-step PI05 adapter run completed on the Ubuntu
  4090 machine and saved a `final_policy.pt` checkpoint. Validation loss
  decreased from `1.4696` initially to `0.3158` at step 1000; 100-step
  validation checkpoints were `1.3084`, `1.1948`, `1.0347`, `0.9168`,
  `0.7570`, `0.5315`, `0.4459`, `0.4046`, `0.3480`, and `0.3158`. Current
  interpretation: PI05 can optimize substantially on the current synthetic
  LeRobot export, but this remains sim-data feasibility and needs held-out
  open-loop action prediction diagnostics. Evidence:
  `tools/train_pi05_lerobot_adapter.py`,
  `simulation_output/pi05_lerobot_adapter_train_ckpt_frontup_step024_v1/` on
  the 4090 machine.
- [model] A held-out PI05 open-loop diagnostic script was added to compare
  predicted `action_32` against held-out expert actions while reporting active
  dims `0:9`, Elite TCP-delta MAE for dims `0:6`, and Piper intent argmax
  accuracy for dims `6:9`. A 4-batch smoke loaded the checkpoint and wrote a
  JSON diagnostic, but the smoke is not itself a stable metric. Evidence:
  `tools/eval_pi05_lerobot_open_loop.py`,
  `simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_smoke.json`
  on the 4090 machine.
- [model] Full held-out open-loop sampling from the checkpointed PI05 policy
  exposed a gap between training loss and decoded mixed-action quality. Over
  `616` validation samples, Piper argmax accuracy was `0.5032` versus a
  majority baseline of `0.5292`; Elite TCP-delta MAE had mean `0.6283` and
  median `0.2451`. Current interpretation: the PI05 training/loss path is
  functional, but the current `action_32` sampling/decoding path is not yet a
  convincing mixed-action policy. Evidence:
  `simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_fullval_v1.json`
  on the 4090 machine.
- [model] Increasing PI05 sampling to `--num-inference-steps 50` did not
  materially improve held-out open-loop quality. Over the same `616` validation
  samples, Piper argmax accuracy was `0.5081` versus majority baseline
  `0.5292`, and Elite TCP-delta MAE stayed similar (`mean=0.6296`,
  `median=0.2472`). Current interpretation: the weak mixed-action result is
  unlikely to be caused only by too few diffusion inference steps; the next
  algorithm-side test should use the project contract more directly, with
  Piper as an explicit classification head. Evidence:
  `simulation_output/pi05_lerobot_open_loop_ckpt_frontup_step024_fullval_steps50_v1.json`
  on the 4090 machine.
- [docs] A dedicated data-track handoff document was added to separate the
  unresolved simulation-data and real-collection work from the algorithm-agent
  PI05/mixed-head work. It records the Windows/4090 machine split, SSH and copy
  commands, current model-interface constraints, simulation data tasks, real
  collection tasks, and the shared data schema. Evidence:
  `docs/data-track-handoff.md`.
- [model] A source-level probe of the installed LeRobot `0.4.4` PI05
  implementation found that its training and sampling paths consume images and
  language but do not consume `observation.state`. The existing official PI05
  runs therefore have not yet exercised Elite robot-state or tactile/contact
  conditioning, despite those fields being present in the dataset export.
  Evidence: `tools/probe_pi05_feature_interface.py`,
  `simulation_output/pi05_feature_interface_probe_v1.json`, and
  `simulation_output/pi05_modeling_source_v1.py` on the 4090 machine.
- [model] The first explicit PI05 mixed-action path was added. It reuses the
  frozen 1000-step PI05 checkpoint for Elite and trains a lightweight Piper
  classifier from PI05 image/language prefix embeddings plus normalized state;
  held-out evaluation reports Elite sampling error and Piper classifier
  accuracy separately. Remote 2-step training and 2-record evaluation smokes
  passed and saved a `2.1 MB` head-only checkpoint, but the smoke validation
  subset contained one Piper class and is not a quality result. Evidence:
  `tools/train_pi05_mixed_head_adapter.py`,
  `tools/eval_pi05_mixed_head_open_loop.py`,
  `simulation_output/pi05_mixed_head_adapter_smoke_frontup_step024_v1/`, and
  `simulation_output/pi05_mixed_head_open_loop_smoke_frontup_step024_v1.json`
  on the 4090 machine.
- [model] The full 1000-step frozen PI05 mixed-head run failed to learn a
  discriminative Piper classifier. Train/validation class supports were
  `[0, 1660, 1820]` and `[0, 60, 68]`; final accuracy was `0.53125`, equal to
  the validation majority baseline, balanced accuracy remained `0.5`, and all
  validation records were predicted as class 2. Intermediate evaluations only
  switched between all-class-1 and all-class-2 predictions. Evidence:
  `simulation_output/pi05_mixed_head_adapter_train_frontup_step024_v1/` on the
  4090 machine.
- [decision] The current 1000-step PI05 checkpoint is reclassified as an
  architecture/loss feasibility checkpoint rather than a pretrained PI05
  fine-tune. The training script constructs `PI05Policy(config)` directly and
  freezes the randomly initialized PaliGemma path; the 4090 Hugging Face cache
  does not contain `lerobot/pi05_base`. Verify loading and provenance of a
  reproducible pretrained PI05 base before further mixed-head tuning or
  state/tactile injection.
- [docs] The project handoff was split along the two active work tracks.
  `docs/algorithm-track-handoff.md` and `docs/algorithm-track-commands.md` now
  own VLA/OpenPI/PI05 work; `docs/data-track-handoff.md` owns simulation data
  and real collection. `docs/project-state.md` and `docs/handoff.md` are shared
  routing/interface documents only. The append-only weekly meeting log remains
  shared across both tracks.
- [model] A fail-closed PI05 pretrained loader and dedicated probe were added.
  The loader records pinned revision, resolved files, architecture compatibility,
  loaded/missing/unexpected keys, shape mismatches, parameter coverage, and
  before/checkpoint/after fingerprints; existing PI05 probe/training entrypoints
  now require either a pretrained source or explicit `--allow-random-init`.
  Metadata verification passed for public repository `lerobot/pi05_base` at
  commit `7de663972b7817d2c4cf2d84c821153dfea772e9`; its single weight file is
  `14,467,165,872` bytes. Synthetic loading and project preprocess regression
  smokes passed on the 4090. The real weight file is not downloaded yet, so
  pretrained parameter coverage and forward/backward remain unverified.
  Evidence: `tools/pi05_pretrained_loader.py`,
  `tools/probe_pi05_pretrained_loading.py`,
  `tools/test_pi05_pretrained_loader_smoke.py`, and
  `simulation_output/pi05_pretrained_metadata_probe_v1.json` on the 4090.

### Open Follow-Ups

- Confirm the next hardware path with the supervisor before further onsite
  trial-and-error collection.
- Use current real captures for calibration/alignment analysis rather than
  training a policy directly.
- Continue local sim-to-real alignment while waiting for a more reliable real
  guidewire delivery setup.
- Treat tactile/contact context as an interface requirement, not yet an
  empirically demonstrated performance gain.
- To make tactile conditioning meaningful, change the task/data so contact
  risk influences the target action, or use real contact/risk labels rather
  than only posthoc image-distance thresholds.
- Add and verify a reproducible `lerobot/pi05_base` pretrained-loading path,
  including loaded/missing-key and checkpoint-provenance reporting.
- After pretrained loading is verified, compare `state_only` and balanced
  `prefix_state` Piper heads before injecting state/tactile tokens into Elite.
- Use `docs/data-track-handoff.md` as the entry point for the data-focused
  agent that continues simulation realism and real collection work.
- Add real-domain calibration anchors before claiming any sim-to-real transfer.

## 2026-07-17

### Facts

- [model] Pinned `lerobot/pi05_base` pretrained loading is verified on the 4090
  at revision `7de663972b7817d2c4cf2d84c821153dfea772e9`. The checkpoint omits
  one PaliGemma embedding because it is tied to the stored LM head; a dedicated
  probe confirmed identical shape and shared storage, and the fail-closed
  loader now accepts only this exact alias and verifies it after loading. The
  offline project-data probe reported `812/812` direct checkpoint-key matches,
  zero effective missing/unexpected/shape-mismatch keys, `1.0` unique-parameter
  coverage, loss `1.043866`, and 208 gradient tensors with nonzero aggregate
  gradient L2. This establishes checkpoint provenance and trainability, not
  policy quality or real-system validation. Evidence:
  `tools/pi05_pretrained_loader.py`,
  `tools/probe_pi05_tied_weights.py`,
  `simulation_output/pi05_tied_weights_probe_v1.json`, and
  `simulation_output/pi05_pretrained_load_forward_backward_v2.json` on the
  4090 machine.
- [model] A controlled pretrained Piper-head ablation path was implemented for
  `state_only` versus dimension-balanced `prefix_state`. The latter projects
  pooled PI05 prefix features and normalized state into equal 128D branches
  before fusion, replacing the earlier raw 2048D-plus-32D concatenation. Both
  modes now use fail-closed pinned pretrained initialization; mode/checkpoint
  smokes and two-step project-data integration smokes passed on the 4090. A
  comparison tool rejects mismatched split, seed, class counts, or pretrained
  provenance and reports majority baseline, balanced accuracy, predicted-class
  support, and confusion matrices. The small integration validation slice had
  one class, so no performance conclusion is recorded yet. Evidence:
  `tools/train_pi05_mixed_head_adapter.py`,
  `tools/compare_pi05_piper_head_runs.py`, and
  `simulation_output/pi05_piper_head_comparison_pretrained_smoke_v1.json` on
  the 4090 machine.
- [model] The full controlled pretrained Piper-head comparison rejected the
  dimension-balanced PI05 `prefix_state` fusion. With identical split, seeds,
  class weighting, and pretrained revision, `state_only` reached best accuracy
  `0.625` and balanced accuracy `0.6167` at step 100 with predicted support
  `[0,46,82]`, above the `0.53125` majority baseline. `prefix_state` reached
  best accuracy `0.5234` and balanced accuracy `0.5515` at step 800 with
  predicted support `[0,121,7]`; its accuracy stayed below majority and its
  best balanced accuracy was `0.0652` lower than `state_only`. Both heads
  degraded after their best point, so the trainer now saves a separate
  best-validation head checkpoint. Current algorithm recommendation is to
  retain early-stopped `state_only` and not continue this `prefix_state` design
  without a new hypothesis. Evidence:
  `simulation_output/pi05_piper_head_comparison_pretrained_frontup_step024_v1.json`
  on the 4090 machine.
- [model] The selected pretrained `state_only` Piper head was reproduced in a
  100-step run: accuracy, balanced accuracy, CE, and confusion matrix exactly
  matched the original long run's step-100 validation point. Separate best and
  final checkpoint files contain identical head weights at this step and carry
  correct selection metadata. The open-loop evaluator now fails closed on
  training/evaluation pretrained provenance mismatch; a one-record interface
  smoke passed with best checkpoint step 100, pinned revision, and coverage
  `1.0`. The one-record task metrics are not treated as performance evidence.
  Evidence:
  `simulation_output/pi05_piper_state_only_pretrained_frontup_step024_best_v1/`
  and `simulation_output/pi05_piper_state_only_pretrained_open_loop_smoke_v1.json`
  on the 4090 machine.
- [model] Full 616-record open-loop evaluation of the selected pretrained
  `state_only` mixed head produced Piper accuracy `0.5552`, balanced accuracy
  `0.5643`, and majority baseline `0.5292`; class-1/class-2 recalls were
  `0.7207/0.4080`, so the classifier is weak but non-degenerate. The
  unfine-tuned pretrained PI05 Elite output had 6D MAE mean `0.7360`, median
  `0.5146`, and P95 `2.6789`, with dim-1 MAE `1.8585`. This is worse in
  mean/median than the older random-base project-trained diagnostic and shows
  that project action adaptation is required before tactile injection or
  rollout-like evaluation. An Elite-only pretrained training path excluding
  Piper compatibility dims passed a two-step remote smoke and reduced
  validation loss from `3.1553` to `2.0839`. Evidence:
  `simulation_output/pi05_piper_state_only_pretrained_open_loop_fullval_v1.json`
  and `simulation_output/pi05_pretrained_elite_only_train_smoke_v1/` on the
  4090 machine.
- [model] The 1000-step pinned-pretrained Elite-only scan reduced validation
  loss from `2.4257` to its best value `0.5759` at the final step. Intermediate
  100-step checks fluctuated but none beat step 1000, so the selected checkpoint
  step is 1000. The scan excluded Piper compatibility dimensions and did not
  save the large policy. Validation reporting now aggregates per-dimension loss
  over all evaluated batches; a remote smoke verified that the six dimension
  means reproduce total loss. The next action is an exact step-1000 rerun that
  saves the approximately 7 GB policy checkpoint. Evidence:
  `simulation_output/pi05_pretrained_elite_only_train_scan_frontup_step024_v1/`
  on the 4090 machine.
- [model] The exact step-1000 Elite-only saving rerun reproduced every scan
  validation point with maximum difference `0` and saved a 7.47 GB policy
  checkpoint with SHA256
  `43c084e5b49d2a3019506305212b075db825d0e8ff2a9f31602c665ff31e98f9`.
  The evaluator now fail-closes on Elite-only loss scope and accepts the saved
  Elite policy with the selected step-100 `state_only` Piper head only when
  both derive from the same pinned pretrained source and revision. A one-record
  combined-policy smoke passed at pretrained coverage `1.0`; its task metrics
  are not performance evidence. The next algorithm action is the full
  616-record combined open-loop evaluation. Evidence:
  `simulation_output/pi05_pretrained_elite_only_train_ckpt_frontup_step024_v1/`
  and
  `simulation_output/pi05_pretrained_elite_only_state_head_open_loop_smoke_v1.json`
  on the 4090 machine.
- [model] Full 616-record combined evaluation preserved the selected
  `state_only` Piper metrics exactly and reduced aggregate Elite MAE
  mean/median/P95 from `0.7360/0.5146/2.6789` to
  `0.4664/0.0722/2.5340`. This aggregate gain is misleading for the current
  control target: per-dimension mean-MAE changes were `+12.8%`, `-4.6%`,
  `-1.3%`, `-90.1%`, `-94.8%`, and `-92.7%`. Most improvement came from
  rotation dims 3:6 whose targets are constant zero, while translation dim 0
  regressed and dims 1:3 changed little. The algorithm track therefore defers
  tactile injection and adds a translation-only dims-0:3 loss diagnostic. Its
  pinned-pretrained two-step remote smoke passed with active dims `3` and
  coverage `1.0`; the next action is a no-checkpoint 1000-step controlled scan.
  Evidence:
  `simulation_output/pi05_pretrained_elite_only_state_head_open_loop_fullval_v1.json`
  and `simulation_output/pi05_pretrained_elite_translation_only_train_smoke_v1/`
  on the 4090 machine.
- [model] The controlled 1000-step translation-only PI05 scan reduced its
  validation loss from `3.4264` to the best/final value `1.1215`, but did not
  beat the existing six-dimensional Elite-only run. At step 1000, translation
  per-dimension losses were `0.8601/1.1625/1.3420` versus the existing run's
  lower `0.7941/1.1561/1.3389`; translation mean was about `2.3%` worse.
  Therefore no new 7 GB checkpoint will be saved. The next algorithm
  hypothesis is explicit normalized-state conditioning in the Elite PI05
  diffusion path, which LeRobot `0.4.4` currently does not consume, before
  tactile injection. Evidence:
  `simulation_output/pi05_pretrained_elite_translation_only_train_scan_frontup_step024_v1/`
  on the 4090 machine.
- [data][model][decision] A read-only audit of the 4096-record PI05 training
  slice found valid exported schema but strong data/interface limitations:
  previous-action Elite translation MAE was `0.2487`, previous-label Piper
  accuracy was `0.9051`, Elite rotation targets were all zero, and the full
  4936-record export had no positive `estimated_contact_flag` examples. The
  current export remains an interface-smoke dataset; the next full
  state-conditioned PI05 ablation is paused pending whole-episode,
  transition-stratified, and observability/contact readiness checks. Evidence:
  `docs/vla-training-data-readiness-audit.md` and
  `simulation_output/pi05_training_data_audit_frontup_step024_v1.json`.
- [real-system][model][decision] A read-only Elite-pose audit of the failed
  `real_pilot_20260707_left_linked_elite_piper_matched_001` pilot found exactly
  constant requested RPY and at most `0.00221` degrees executed per-axis
  orientation span. Normal XYZ waypoint spacing had median `17.3168 mm`; the
  first `69.2541 mm` move was an initialization reposition. Because image-pose
  lag P95 was about `1.995 s` and 48% of pose records were stale, the episode
  remains reference-only and is not training data. The bounded VLA Elite target
  is now translation dims `0:3`, with fixed orientation owned by the
  controller. Evidence:
  `docs/vla-elite-real-pose-reference-audit-20260717.md` and
  `simulation_output/real_pilot_20260707_left_linked_elite_piper_matched_001_elite_pose_audit.json`.
- [data] A no-relabel, uniform-stride-5 view of the 4936-record simulation
  export retained all 40 episodes and all 472 Piper transitions while reducing
  the previous-label Piper baseline from `90.36%` to `51.09%` and increasing
  the previous-action Elite translation MAE from `0.2517` to `0.9601`. The
  1005-record view uses a task-stratified 34/6 whole-episode split and passed
  OpenPI validation with all 2010 side/top images checked and none missing. The
  self-contained Windows/Linux-portable transfer bundle is 83.48 MB. Evidence:
  `docs/simulation-training-data-temporal-audit-20260717.md`,
  `tools/prepare_openpi_temporal_view.py`, and
  `simulation_output/openpi_compat_pack_frontup_step024_temporal_stride5_v1/validation.json`.
- [decision] The stride-5 view is accepted only for a bounded Elite-translation
  plus Piper hold/feed algorithm test. It does not support tactile benefit,
  Elite rotation, Piper retract, broader scene coverage, or real-system claims;
  positive contact behavior still requires new estimator-derived collection.
  Evidence: `docs/simulation-training-data-temporal-audit-20260717.md`.
- [data] An offline-only contact supervision sidecar was built from the current
  4936-record formal simulation dataset. It contains 411 diagnostic contact
  positives, 4525 negatives, five task-stratified whole-episode folds with
  80-85 positives each, and 9872 verified side/top image references. Exact
  MuJoCo contact is marked `policy_input_allowed=false` and remains outside
  `state_32`. Evidence: `tools/build_contact_supervision_sidecar.py`,
  `simulation_output/contact_supervision_frontup_step024_v1/manifest.json`, and
  `docs/contact-supervision-sidecar-audit-20260718.md`.
- [data] A whole-episode OOF RGB contact-estimator baseline tool now extracts
  red-tip-local side/top features and compares them against a mandatory
  tip-position-only shortcut baseline. A 184-record balanced smoke completed
  all five folds, saved ten fold models, and produced complete audit-only OOF
  probabilities. Its high scores are not performance evidence because the
  position-only baseline already reached AP `0.9124`; the full 4936-record OOF
  audit is required before changing the expert. Evidence:
  `tools/train_contact_estimator_oof.py`,
  `simulation_output/contact_estimator_oof_smoke_v1/report.json`, and
  `docs/contact-supervision-sidecar-audit-20260718.md`.
- [data][decision] The current rendered-image edge-distance contact estimator
  has zero recall and zero F1 on the sidecar. Full-frame review shows that the
  contact geometry is only a few pixels around the red tip. The sidecar is
  accepted for estimator training/evaluation only; it does not establish
  tactile policy benefit because right-task contact is sparse and the source
  expert lacks a clean contact-conditioned action change. The next estimator
  must use whole-episode out-of-fold predictions, tip localization, and local
  wall/lumen features. Evidence:
  `docs/contact-supervision-sidecar-audit-20260718.md`.
- [data] The full 4936-record, five-fold whole-episode RGB-local contact
  estimator audit reached AP `0.99138`, F1 `0.95848`, precision `0.93519`, and
  recall `0.98297`, materially above the tip-position-only baseline AP/F1 of
  `0.82569/0.68216`. All five folds and both tasks remained non-collapsed; all
  59 truth contact runs were detected with onset delay P95 of zero sampled
  frames. This passes only the simulation feasibility gate for a separate
  diagnostic expert. Evidence:
  `simulation_output/contact_estimator_oof_full_v1/report.json`,
  `simulation_output/contact_estimator_oof_full_v1/oof_error_sheet.png`, and
  `docs/contact-supervision-sidecar-audit-20260718.md`.
- [real-system][data][blocker] The five-model simulation RGB estimator ensemble
  produced zero positives on all 300 frames of the failed July real pilot;
  probability P95/max was only `4.16e-43/1.26e-33`. Because the pilot has no
  contact truth, this is saturated negative OOD behavior rather than accuracy
  evidence. Real `estimated_contact_flag` remains invalid until manual real
  annotation and calibration. Evidence:
  `tools/audit_contact_estimator_real_pilot.py`,
  `simulation_output/contact_estimator_real_pilot_ood_v1/report.json`, and
  `docs/contact-supervision-sidecar-audit-20260718.md`.
- [simulation][data] A separate fail-closed diagnostic contact-probe collector
  preserved `elite_tcp_delta_6d + piper_intent_id`, left Piper feed/hold
  scheduling unchanged, and generated 11 exact-contact steps plus 45
  conservative Elite response steps in a 140-step left smoke. Evidence:
  `simulation/collect_estimated_contact_probe.py`,
  `simulation_output/_smoke_estimated_contact_probe_v4/manifest.json`, and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data][blocker][decision] The source-trained RGB estimator did not generalize
  to probe-induced contact: left-smoke TP/FP/FN was `0/41/11`, and two
  right-task direction smokes generated no exact contact. Probe estimator
  fields are therefore `policy_input_allowed=false`; the collector is accepted
  only for expanding offline supervision, not policy training. Evidence:
  `simulation_output/_smoke_estimated_contact_probe_v4/episode_left_0001_try_01/meta.json`,
  `simulation_output/_smoke_estimated_contact_probe_right_v2/episode_right_0001_try_01/meta.json`,
  and `docs/estimated-contact-probe-audit-20260718.md`.
- [simulation][data] A no-render, no-estimator right probe geometry audit found
  the effective wall direction to be `n2/-1`, not the previously tested `n1`.
  Mild start-bucket settings produced `27/60`, `24/60`, and `5/60` contact
  steps at start fractions `0.42/0.48/0.54` with radius fractions
  `0.95/1.10/0.95`; every audit command completed in under 50 seconds.
  Evidence: `tools/audit_right_probe_geometry.py`,
  `simulation_output/right_probe_geometry_audit_coarse_v1.json`, and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data][blocker] The actual dual-camera right probe collector confirmed 27
  exact-contact steps in 100 steps at `0.42/n2/-1/0.95`, but the RGB estimator
  predicted zero positives. Manual frame review found side-tip visibility and
  top-tip occlusion during contact. Right contact generation is now feasible,
  but probe-distribution estimator validation remains blocked. Evidence:
  `simulation_output/_smoke_estimated_contact_probe_right_n2neg095_v1/manifest.json`
  and `docs/estimated-contact-probe-audit-20260718.md`.
- [data] The first validated left probe supervision bucket completed with 3
  episodes and 540 samples at start fraction 0.42. It has 153 exact-contact
  steps, no missing side/top images, and per-episode estimator TP/FP/FN of
  `4/40/47`, `4/41/47`, and `4/41/47`. It is offline supervision only;
  `policy_input_allowed=false` remains enforced. Evidence:
  `simulation_output/probe_left_start042_v1/manifest.json` and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data][blocker] The right start-0.42 supervision bucket completed with 3
  episodes and 540 samples, including 81 exact-contact steps and no missing
  side/top images. The source RGB estimator missed all contacts in every
  episode (`TP/FP/FN=0/0/27`), so the bucket is hard-positive offline
  supervision only. Evidence:
  `simulation_output/probe_right_start042_v1/manifest.json` and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data] The right start-0.48 supervision bucket completed with 3 episodes and
  540 samples, including 153 exact-contact steps and no missing images. The
  source RGB estimator missed every contact frame (`TP/FP/FN=0/0/51` per
  episode). Evidence:
  `simulation_output/probe_right_start048_v1/manifest.json` and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data] The right start-0.54 supervision bucket completed with 3 episodes and
  540 samples, including 15 exact-contact steps and no missing side/top images.
  The source RGB estimator missed every contact frame (`TP/FP/FN=0/0/5` per
  episode). All four planned offline supervision buckets are now available.
  Evidence: `simulation_output/probe_right_start054_v1/manifest.json` and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data][blocker][decision] Augmented five-fold RGB OOF reached overall
  AP/F1 `0.9917/0.9269`, but the sparse right start-0.54 bucket had
  AP/F1 `0.5377/0.2500` with precision `0.1429` and 90 false positives. The
  augmented model also predicted zero positives on all 300 real-pilot frames.
  Promotion remains rejected; right 0.54 must be rebalanced or extended before
  another OOF/OOD run. Evidence:
  `simulation_output/contact_estimator_oof_probe_augmented_v1/report.json`,
  `simulation_output/contact_estimator_real_pilot_ood_probe_augmented_v1/report.json`,
  and `docs/estimated-contact-probe-audit-20260718.md`.
- [data][decision] The temporal risk audit found right start-0.54 risk runs
  lead exact contact by 25 frames at the median, with three pure risk runs;
  formal source, left 0.42, right 0.42, and right 0.48 had near-zero lead.
  This supports a possible separate risk-warning field but does not permit
  silently promoting or renaming `estimated_contact_flag`. Evidence:
  `tools/audit_contact_risk_temporal.py`,
  `simulation_output/contact_estimator_oof_probe_augmented_v1/temporal_risk_audit.json`,
  and `docs/estimated-contact-probe-audit-20260718.md`.
- [data][interface][decision] Added optional diagnostic observation fields
  `estimated_contact_risk_flag`, `estimated_contact_risk_probability`, and
  explicit risk semantics. The fields preserve the existing
  `elite_tcp_delta_6d + piper_intent_id` action interface and remain
  `contact_risk_policy_input_allowed=false`; they are not strict contact labels
  or real tactile validation. Evidence:
  `simulation/collect_estimated_contact_probe.py`,
  `simulation_output/_smoke_contact_risk_field_v1/manifest.json`,
  and `docs/project-state.md`.
- [data][decision] The right start-0.54 OOF errors are temporally structured:
  exact contact is steps `44-48`, while predictions run `19-52` plus step `54`
  in all three episodes. Threshold 0.50 gives precision/recall/F1
  `0.1429/1.0000/0.2500`; threshold 0.99 gives `0.5000/0.2000/0.2857`.
  The field currently behaves as an early risk warning, not a precise contact
  label. Do not duplicate the bucket or silently change semantics. Evidence:
  `simulation_output/contact_estimator_oof_probe_augmented_v1/report.json`,
  `simulation_output/contact_estimator_oof_probe_augmented_v1/oof_error_sheet.png`,
  and `docs/estimated-contact-probe-audit-20260718.md`.
- [data] The formal source and four probe buckets were merged into 52 whole
  episodes and 7096 samples; all 14192 side/top image references passed and the
  augmented sidecar contains 813 exact-contact positives (left 530, right 283).
  This is still offline supervision only; augmented RGB OOF evaluation is the
  next gate. Evidence:
  `simulation_output/probe_supervision_merged_v1/manifest.json`,
  `simulation_output/contact_supervision_probe_augmented_v1/manifest.json`, and
  `docs/estimated-contact-probe-audit-20260718.md`.
- [data][interface][decision] Built the policy-safe contact-risk diagnostic
  pack from augmented OOF predictions. It contains 7096 samples and 7096
  separate exact-contact/wall audit targets; all 14192 image references are
  valid. `samples.jsonl` preserves non-null `elite_tcp_delta_6d` and
  `piper_intent_id` (`retract=0`, `hold=1`, `feed=2`) while keeping exact
  contact and risk conditioning disabled. Evidence:
  `simulation_output/contact_risk_diagnostic_pack_v1/manifest.json` and
  `tools/build_contact_risk_diagnostic_pack.py`.
- [real-system][data][decision] Clarified that the replacement feeder and
  Piper mechanical arm are mutually exclusive guidewire feed sources. The next
  onsite gate keeps the physical S-bend and compares feeder-only versus
  Piper-only runs; Elite guidance is held stationary for the first comparison.
  There is no combined feed mode. These runs remain hardware-diagnostic until
  displacement, S-bend crossing, command execution, and image synchronization
  are verified. Evidence and commands:
  `docs/commands.md` under `S-Bend Component-Isolation Gate`.
- [real-system][data] On 2026-07-21, the replacement guidewire feed device was
  mechanically adjusted onsite and a basic feed test was reported usable. This
  clears the standalone functionality gate only; stable S-bend passage,
  repeatability, synchronized images, and executed-command logs still require
  a short feeder-only capture before the hardware can support expert data.
- [real-system][data][interface] Confirmed the onsite environment split:
  `/home/zsw/PycharmProjects/real_collection` uses the `sam3` interpreter with
  the Elite SDK, while `/home/zsw/project_2026` is the separate LeRobot root.
  The current collector, UDP feeder adapter, validator, and video renderer were
  synchronized into `real_collection`; compilation, Elite/OpenCV imports, and
  a non-hardware `bash_dev_udp` loopback smoke passed.
- [real-system][data] The exact Bash `printf` `/dev/udp` collector transport
  produced a physically observed single feeder movement on 2026-07-21. Artifact
  `real_diag_20260721_s_bend_feeder_printf_single_004` contains 40 records, one
  `feed_feeder_bash_dev_udp_sent`, zero validation errors/warnings, side/top
  sync P95 4.07 ms, and image/action sync P95 14.20 ms. Python `socket.sendto`
  and the initial Bash `cat` transport did not move the device. The next gate is
  five-step feeder-only repeatability with Piper disabled and Elite stationary.
- [real-system][data] Five-step feeder-only repeatability at a one-second period
  completed with physical forward movement on all five commands, but one step
  moved visibly less. Artifact
  `real_diag_20260721_s_bend_feeder_printf_repeat5_005` has 60 records, five
  `feed_feeder_bash_dev_udp_sent` events, and zero validation errors/warnings;
  side/top and image/action sync P95 were 30.33/33.99 ms. Nominal 60 mm is not
  measured displacement. The next isolation test changes only the feed period
  to two seconds.
- [real-system][data][decision] Follow-up repeats at a two-second feed period
  ruled out command cooldown as the main explanation for variable progress.
  Depending on the initial guidewire state, five commands could deliver
  normally or produce almost no effective progress. The onsite interpretation
  is local vessel-model impingement caused by subtle initial guidewire-state
  differences, not proven feeder-output variability. Future repeats must use
  unique output directories and record the initial axial reference, entry
  angle, slack/curvature, and pre-feed frames; `piper_step` remains command
  count rather than physical insertion truth.
- [real-system][data][decision] The feed device underwent a substantial onsite
  adjustment on 2026-07-21, invalidating the old `12 mm/step`, `11-13 mm`
  range, and all derived `60 mm` summaries in earlier diagnostic captures.
  Future collection records feed-command counts only and leaves insertion
  millimeters null until a new physical calibration is accepted. Historical
  `_004/_005/_006` millimeter fields must not be used for simulator alignment.
- [real-system][data][decision] Fixed millimeters per feed are no longer a
  calibration target: curved geometry, local impingement, and initial-state
  variation make one feed event non-equivalent to a reliable distance. New
  insertion-length fields remain null and progress must be image/observation
  derived. The adjusted replacement feed device is now the real feed backend;
  no further Piper-only comparison is planned. `piper_intent_id` remains the
  compatibility field for discrete guidewire feed intent, routed by the real
  controller to the feed device.
- [model][interface] The algorithm-side contact-risk adapter smoke validated
  all 7096 policy samples, 52 episodes, and 14192 image paths, with no exact
  contact/wall keys in policy observations. A matched 24-record risk-off/on
  pass through the existing adapter changed only state dims `8:14`; images,
  actions, and explicit Elite/Piper targets remained identical. No PI05 or
  mixed-head structure changed, and risk was not used as a label. The field
  remains `policy_input_allowed=false` and diagnostic-only. Evidence:
  `tools/probe_contact_risk_adapter_views.py` and
  `simulation_output/contact_risk_adapter_probe_v1.json`.
- [model][data][interface] The stride-5 candidate is now converted through the
  official LeRobot loader on the 4090 with 1005 frames, 40 complete episodes,
  and the authoritative 851/154-record, 34/6-episode split. PI05 training and
  open-loop paths now fail-close on that manifest, forbid `max_records`, and
  report Elite translation plus Piper transition/steady and temporal
  baselines. The fixed-split audit measured previous-label Piper accuracy
  `0.5135` and previous-action Elite MAE `0.9662`; state/image baselines showed
  learnable signal. Pinned-pretrained one-step Elite/Piper and two-record
  open-loop interface smokes passed. Evidence:
  `simulation_output/pi05_training_data_audit_temporal_stride5_v1.json`,
  `simulation_output/pi05_stride5_translation_split_smoke_v1/`, and
  `simulation_output/pi05_stride5_mixed_open_loop_interface_smoke_v1.json` on
  the 4090.
- [model] The fixed-split stride-5 PI05 Elite translation scan used all
  851/154 train/validation records and reduced validation diffusion loss from
  `2.9458` to its minimum `1.0316` at step 900; step 1000 was slightly worse at
  `1.0409`. Pretrained loading remained complete (`812/812` keys, no effective
  missing/unexpected/shape mismatches, unique-parameter coverage `1.0`). The
  two LeRobot vision patch-embedding warnings are unconditional informational
  messages, not missing-weight evidence for this run. Evidence:
  `simulation_output/pi05_stride5_translation_scan_v1/summary.json` and
  `train_log.json` on the 4090.
- [model] The selected step-900 stride-5 Elite translation checkpoint was
  reproduced exactly: final validation diffusion loss `1.0315715289`, pinned
  pretrained coverage `1.0`, and checkpoint SHA256
  `fc6b863f9f7ddb8a5a50d6337080e766fcae1618d4fe0c5816931bc5ec61f116`.
  The next gate is full 154-record fixed-split open-loop evaluation; no
  rollout or real-system claim follows from the checkpoint alone. Evidence:
  `simulation_output/pi05_stride5_translation_ckpt_step900_v1/summary.json`
  and `final_policy.pt` on the 4090.
- [model] Full fixed-split open-loop evaluation over all 154 validation records
  gave Elite translation MAE `0.7660` (median `0.4378`, P95 `2.5704`), better
  than the previous-action baseline `0.9662` but worse than train-mean
  `0.7057`, state-Ridge `0.5520`, and image-Ridge `0.5080`. Transition/steady
  Elite MAE was `0.4866/0.9260`. The translation-only checkpoint's untrained
  Piper compatibility output scored `0.3247` versus majority `0.5325`, so it
  is not a Piper comparison. Next algorithm gate: state-conditioned Elite
  translation ablation without tactile/contact inputs. Evidence:
  `simulation_output/pi05_stride5_translation_open_loop_fullval_step900_v1.json`.
- [model][interface] Implemented a state-conditioned PI05 Elite translation
  adapter that appends a projected 32D `observation.state` prefix token after
  pinned pretrained loading. Remote 2-step training, checkpoint round-trip,
  and fixed-split 1-record open-loop inference smokes passed; pretrained
  coverage stayed `1.0`. This is an adapter/interface milestone only. The
  next user-run job is a 1000-step state-conditioned scan without tactile or
  contact fields. Evidence:
  `simulation_output/pi05_stride5_state_conditioned_smoke_v1/summary.json` and
  `simulation_output/pi05_stride5_state_conditioned_open_loop_smoke_v1.json`.
- [model] The full 1000-step state-conditioned Elite translation scan selected
  step 900 with validation diffusion loss `1.035247`, slightly worse than the
  image/language-only step-900 loss `1.031572` (`+0.36%`). Dim-0 loss improved,
  while dims 1/2 worsened. A step-900 checkpoint and full raw-MAE open-loop
  comparison are required before accepting or rejecting the state prefix.
  Evidence: `simulation_output/pi05_stride5_state_conditioned_scan_v1/summary.json`
  and `train_log.json` on the 4090.
- [model] The selected state-conditioned step-900 checkpoint exactly
  reproduced validation loss `1.0352473862`, preserved pinned pretrained
  coverage `1.0` and `state_prefix_token_v1` metadata, and was saved with
  SHA256 `6d37fae648c2cf9a0d894c3f5a0b8da2e748e853932f406417537dabef45aae4`.
  The remaining acceptance gate is full 154-record raw open-loop comparison.
  Evidence: `simulation_output/pi05_stride5_state_conditioned_ckpt_step900_v1/`.
- [model] Final fixed-split state-conditioned open-loop evaluation improved
  Elite translation MAE mean/median/P95 from `0.7660/0.4378/2.5704` to
  `0.7478/0.4169/2.4515`; transition MAE improved `0.4866 -> 0.4433`, while
  steady MAE changed only `0.9260 -> 0.9192`. Mean gain was `2.37%`, still
  weaker than train-mean/state-Ridge/image-Ridge baselines. The state prefix is
  retained as the current Elite candidate; next gate is a separate stride-5
  Piper `state_only` head. Evidence:
  `simulation_output/pi05_stride5_state_conditioned_open_loop_fullval_step900_v1.json`.
- [model] The stride-5 Piper `state_only` head selected step 1000 with
  accuracy/balanced accuracy `0.5974/0.6194` versus majority `0.5325` and
  previous-label `0.5135`; transition/steady accuracy was `0.6111/0.5526`.
  A combined one-record smoke with the state-conditioned Elite checkpoint
  passed, including adapter restoration and provenance checks. The next gate
  is the full 154-record combined open-loop evaluation. Evidence:
  `simulation_output/pi05_stride5_piper_state_only_v1/summary.json` and
  `simulation_output/pi05_stride5_state_conditioned_piper_combined_smoke_v1.json`.
- [model] Final combined stride-5 evaluation over 154 held-out records passed
  fixed-split and provenance checks. Elite translation MAE was
  `0.7478/0.4169/2.4515` (mean/median/P95), transition/steady MAE
  `0.4433/0.9192`; Piper accuracy/balanced accuracy was `0.5974/0.6194`, with
  transition/steady accuracy `0.6111/0.5526`. This is the current mixed-action
  simulation feasibility baseline; contact-risk remains policy-disabled and no
  tactile claim follows. Evidence:
  `simulation_output/pi05_stride5_state_conditioned_piper_combined_fullval_v1.json`.
- [real-system] The linked S-bend pilot control was fixed as operator-triggered
  actions from the lab computer: Elite advances one preset trajectory point and
  the replacement feeder sends forward-only events, with no automatic coupling.
  SSH is used only for synchronization/inspection. A new fail-closed Elite
  start-pose guard defaults to `30 mm`; the checked path1/path2 starts were about
  `346 mm` from the latest observed pose and must not be executed until the
  intended start is restored or a matching path is selected. Evidence:
  `data/collect/collect_real_shadow_pilot.py`, `docs/commands.md`.
- [real-system] The repeated-episode reset procedure was changed from manual
  Elite placement to an explicit automatic start approach. Before recording,
  Elite moves from the current pose to the selected trajectory start; the reset
  motion is excluded from training records, while initial distance, requested
  IK target, completion status, and final error are logged. Manual `n`/`f`
  control begins after the approach, with no Elite/feed coupling. Evidence:
  `data/collect/collect_real_shadow_pilot.py`, `docs/commands.md`.
- [real-system] The first automatic Elite reset was stopped before reaching the
  path start because a long `move_joint` command produced a twisted joint-space
  TCP path. The artifact contains only `manifest.json` and is not data. Startup
  reset was corrected to blocking Elite `move_line` with `speed_type=0` and
  explicit millimeters-per-second speed semantics; short in-path steps retain
  inherited `move_joint`. Evidence:
  `collected_data/real_pilot_20260721_180414_left_s_bend_manual_linked`,
  `data/collect/collect_real_shadow_pilot.py`.
- [data] Manual linked collection now gates persistence on the first valid
  operator action. Cameras and pose remain live after Elite startup approach,
  but no frames or JSONL rows are saved until `n/j/f/0`-style input; the
  triggering action is retained as sample `0`. Evidence:
  `data/collect/collect_real_shadow_pilot.py`, `docs/commands.md`.
- [data] Dual-camera collection gained serial-bound RealSense RGB-D capture.
  Side/top color is requested at `1920x1080@15`; native depth at
  `1280x720@15` is aligned to color and stored as raw `uint16` PNG with
  timestamps, intrinsics, extrinsics, profiles, and depth scale. Depth remains
  auxiliary and is not promoted to contact truth. Evidence:
  `data/collect/collect_real_shadow_pilot.py`,
  `tools/validate_real_shadow_pilot.py`.
- [data] The onsite dual-camera RGB-D smoke passed after distinguishing
  RealSense SDK serials (`317222072584` / `317222071938`) from the incompatible
  udev `ID_SERIAL_SHORT` values. Background latest-frame capture and 100 unique
  warmup frames reduced the side/top timestamp delta from `57.18 ms` to
  `25.75 ms`; the accepted smoke has two `1920x1080` RGB pairs, two aligned
  `1920x1080 uint16` depth pairs, and zero validator errors/warnings. Nonzero
  depth coverage was about `81.3%` side and `73.9-74.3%` top. This validates
  capture and startup stabilization only, not transparent-vessel depth
  accuracy. Evidence:
  `collected_data/real_diag_20260721_192500_dual_rgbd_1080p_warmup100`,
  `data/collect/collect_real_shadow_pilot.py`.
- [real-system] The first physically successful linked S-bend RGB-D episode
  was collected with the replacement feeder and preset Elite path: 350 records
  and nine executed feeder events. Images and aligned depth are complete, but
  the episode is not yet accepted for training because the shared Elite SDK
  connection blocked pose polling during movement; `148/350` records (`42.3%`)
  used stale poses and image/pose P95 was `1.76 s`.
  The next collection uses a separate Elite command connection before scaling
  the episode count. Evidence:
  `collected_data/real_pilot_20260721_192749_left_s_bend_rgbd`,
  `tools/validate_real_shadow_pilot.py`.
- [data] The corrected separate-Elite-connection workflow produced the first
  clean physically successful S-bend candidate: 310 complete `1920x1080`
  RGB-D records, eight operator-confirmed feeder events, zero stale Elite
  poses, image/pose P95 `81.23 ms`, and zero validator errors. The next two
  attempts failed physically after seven and two logged feeder events despite
  valid data interfaces. Feeder overheating is an onsite hypothesis, not a
  measured conclusion; dynamic collection was stopped rather than mixing more
  episodes from a changing hardware state. Evidence:
  `collected_data/real_pilot_20260721_200628_left_s_bend_rgbd`,
  `collected_data/real_pilot_20260721_200813_left_s_bend_rgbd`,
  `collected_data/real_pilot_20260721_201018_left_s_bend_rgbd`.
- [real-system] The final right-route attempt reached Elite path index `16/19`
  and the operator confirmed that the guidewire passed the bifurcation before
  feeder motion stopped. The capture has 183 complete `1920x1080` RGB-D
  records, zero stale Elite poses, image/pose P95 `81.84 ms`, and zero validator
  errors. It is retained as a right-route valid prefix plus actuator-failure
  suffix, not as a complete expert success. After about four hours onsite, the
  feeder's hot-melt-adhesive fixture had loosened with heat and no longer
  clamped the wire consistently; dynamic collection ended pending a stable
  mechanical repair. Evidence:
  `collected_data/real_pilot_20260721_203449_right_s_bend_rgbd`,
  `docs/real-collection-onsite-audit-20260721.md`.
- [data] Offline replay confirmed that the right partial episode stopped making
  physical progress at approximately step `140`; the last effective feed was
  step `114`, and feed intents at `154/158/171/174/175/176/177` produced no
  motion. A conservative `0-139` right-route prefix selection was recorded with
  boundary uncertainty `135-145`. The full episode remains excluded from
  expert training. Evidence:
  `simulation_output/real_collection_review_20260721/right_203449_prefix_selection_v1.json`.
- [data] A direct audit compared the current 4936-sample formal simulation set
  with the full clean-left candidate and the conservative right `0-139` prefix.
  Elite translation target P95 is close (real/sim left `4.58/4.45 mm`, right
  `4.68/4.98 mm`), but the observation and temporal distributions are not:
  simulation uses idealized square `224x224` close views and about `52%` feed
  labels, while real data is reflective/cluttered `1920x1080` and only
  `2.6-2.9%` sparse feed events. Camera/FOV alignment and sparse real-style
  Piper event generation now precede another full data export. Evidence:
  `docs/real-sim-alignment-audit-20260722.md`,
  `simulation_output/real_sim_alignment_20260721_v1/report.json`.
- [data] Feeder semantics clarify the raw `52%` simulation-versus-`2.6-2.9%`
  real feed-label difference. The replacement feeder uses a much longer stroke
  than Piper. Grouped simulation labels contain `6-8` feed runs per episode
  (median `7`), already close to the eight one-shot events in the successful
  left capture, but each simulation run repeats `feed` for about ten saved
  records. Future simulation data will label only the long-stroke phase onset
  as `piper_intent_id=feed` and log subsequent movement as controller-owned
  execution; no fixed feed distance is inferred. Evidence:
  `docs/real-sim-alignment-audit-20260722.md`.
- [simulation][data] The existing formal collector's event mode passed a
  60-step long-stroke interface smoke with `piper_primitive_steps=8`. Policy
  `feed` targets occur only at steps `0/40`; sampled steps
  `2/4/6/8/42/44/46/48` target `hold` while the controller continues positive
  feed execution. This validates one-shot intent versus multi-step execution,
  not route success or fixed physical displacement. Evidence:
  `simulation_output/_smoke_feeder_longstroke_event_v1/piper_event_semantics_audit.json`,
  `tools/audit_piper_event_semantics.py`.
- [simulation][data] The first complete long-stroke event diagnostic rejected
  the eight-step, execution-value-0.7 primitive despite valid one-shot labels.
  Both branches stopped at 700 steps with progress `52.36`; tip-to-magnet
  median reached `135/150 mm` and max contact `0.594/0.553`. A corrected
  20-step, execution-value-1.0 smoke exactly matched the previous formal
  source over the first 60 steps (saved progress `17.68`; tip-to-magnet median
  `11.30` versus `11.18 mm`) while reducing feed targets from 20 records to two
  events. This restores simulator cumulative drive but does not assign a fixed
  real feed distance. Evidence:
  `simulation_output/formal_tip_line_feeder_longstroke_event_diag_v1/manifest.json`,
  `simulation_output/_smoke_feeder_longstroke_event_steps20_value10_v1/manifest.json`.
- [simulation][data] The corrected complete v2 long-stroke event diagnostic
  accepted both left/right episodes on their first attempts with 247 saved
  samples and 494 valid image references. Each branch has seven one-record
  `feed` events followed by controller-owned execution under `hold`; event and
  Elite front/up audits pass. Tip-to-magnet median is `12.82/13.39 mm`
  (left/right). This establishes the execution/event-semantics baseline, not a
  visually aligned training export; camera/FOV/domain alignment remains next.
  The right contact maximum `0.0807` still requires one exact-seed replay before
  it is classified as seed/start variation. Evidence:
  `simulation_output/formal_tip_line_feeder_longstroke_event_diag_v2/manifest.json`,
  `simulation_output/formal_tip_line_feeder_longstroke_event_diag_v2/piper_event_semantics_audit.json`.
- [simulation][data] An exact-seed right replay (`seed=20260747`, start
  fraction `0.48259370295500165`) under one-shot event semantics reproduced the
  old formal right anchor exactly: success in 246 steps, contact max/P95
  `0.005151112949607249/0.00025366467889398336`, and tip-to-magnet median/P95
  `13.40/17.69 mm`; the event audit passed. The higher contact in the central
  v2 right episode is seed/start variation rather than an event-mode timing
  regression. Long-stroke execution semantics are closed; camera/FOV/domain
  alignment is next. Evidence:
  `simulation_output/_diag_feeder_event_exact_old_right_seed_v1/manifest.json`,
  `simulation_output/_diag_feeder_event_exact_old_right_seed_v1/piper_event_semantics_audit.json`.
- [data][real-system] Onsite visual confirmation found the July 21 RGB-D raw
  camera names are reversed relative to physical placement. Raw `side_*` from
  RealSense `317222072584` is the physical top/high-oblique view; raw `top_*`
  from `317222071938` is the physical side/low-oblique view. The `200628` left
  and `203449` right anchors share this mapping. Raw RGB/depth/timestamp and
  calibration provenance remains internally consistent, so files are not
  renamed; consumers must swap channels at the semantic loading/export
  boundary, and future collection commands must swap the serial assignments.
  Evidence: both dataset manifests and paired-frame visual review.
- [simulation][data] Camera alignment showed camera pose alone is insufficient:
  the vessel requires one shared rigid scene pose, and the obsolete Piper robot
  visual must be hidden because the replacement feeder is the real feed
  backend. The simulator now applies `vessel_pose` consistently to the mesh,
  route, wire initialization, and targets, while Piper/Elite visibility is
  independent. An identity-pose hidden-Piper 60-step regression exactly matched
  the previous baseline (episode/saved progress `18.60/17.68`, tip-to-magnet
  median `11.275 mm`, contact max `0`), confirming the change is visual-only at
  identity. Evidence:
  `simulation_output/_smoke_realalign_identity_hidepiper_v1/manifest.json`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data] The interactive real/simulation alignment tool now previews
  vessel-pose sliders without rebuilding MuJoCo: it transforms only the vessel
  mesh and path-tube visuals during tuning, then performs one full coherent
  scene rebuild on apply/save. A bounded test updated and rendered 89 vessel
  geoms in `0.054 s`, with preview positions matching the rebuilt model within
  `1 µm`. Evidence: `tools/adjust_mujoco_camera.py`.
- [simulation][data] Vessel-size alignment is now interactive through a
  `vessel_scale_%` control. A `115%` test refreshed the visual preview in
  `0.48 s`, kept the world-locked camera unchanged after full rebuild, matched
  rebuilt vessel positions within `1 µm`, and selected a scale-keyed STL-derived
  OBJ so an older cached size cannot be reused. Formal route/contact regression
  remains required after the final real-aligned scale is accepted. Evidence:
  `tools/adjust_mujoco_camera.py`, `simulation/mujoco_guided_wire_env.py`.
- [simulation][data] Manual alignment showed that shared vessel pose/scale
  cannot be finalized from side or top alone because camera projection changes
  can mimic vessel translation and size. The alignment tool now has a joint
  `--camera both` mode with independent side/top camera controls, simultaneous
  four-panel or dual-overlay review, and one shared vessel control/save path.
  Evidence: `tools/adjust_mujoco_camera.py`, July 21 paired real references.
- [simulation][data] The joint side/top adjustment reached an operator-judged
  approximately `95%` camera/vessel match and was accepted as the current
  anchor despite plausible small physical-camera shifts. The remaining visual
  correction is Elite-specific: base translation/orientation plus a long
  flange-mounted magnet representation. The tool now previews and saves those
  fields jointly; the added magnet geom has zero mass and collision masks and
  does not alter the magnetic-force point or `elite_tcp_delta_6d +
  piper_intent_id`. Elite base changes still require bounded left/right
  route-contact regression because they affect FK/IK. Evidence:
  `tools/adjust_mujoco_camera.py`, `simulation/mujoco_guided_wire_env.py`.
- [simulation][data] Elite visual alignment now includes six URDF-limited joint
  sliders in addition to base and magnet controls. The selected radians are
  saved as `elite_joints` and used as the scene initial pose/IK seed; this does
  not redefine the policy target away from `elite_tcp_delta_6d`. A headless
  check confirmed all six MuJoCo joint `qpos` values exactly matched the
  requested values and both side/top renders changed. Evidence:
  `tools/adjust_mujoco_camera.py`, `simulation/mujoco_guided_wire_env.py`.
- [simulation][data] Elite alignment now includes a uniform
  `elite_visual_scale` for link meshes, joint-origin distances, and matching FK
  translation while preserving independent base and flange-magnet dimensions.
  Live-preview geometry matched a `1.15x` full rebuild within `0.81 µm`, and
  Elite tool-world position matched exactly. The `200628` real step-0 record
  provides a controller-requested six-joint baseline of
  `[125.398755, -101.242030, 114.437884, -100.594364, 94.034745, 31.307997]`
  degrees; it is not measured joint feedback. Evidence:
  `tools/adjust_mujoco_camera.py`, `simulation/mujoco_guided_wire_env.py`,
  `collected_data/real_pilot_20260721_200628_left_s_bend_rgbd/records.jsonl`.
- [simulation][data] Elite joint alignment controls were refined from integer
  degrees to `0.1 degree` steps after integer rounding was identified as a
  possible source of accumulating distal-link projection error. The overlay
  now reports authoritative signed joint angles with one decimal place; scene
  persistence remains radians in `elite_joints`. Evidence:
  `tools/adjust_mujoco_camera.py`.
- [simulation][data] An Elite model-variant audit confirmed the real-alignment
  scene loads standardized `ec66_description.urdf`, whose eight `ec66` STL
  meshes are byte-identical to the ROS source copy. Real `200628` step-0 FK
  comparison using the requested joints gives flange errors `ec66=0.482 mm`,
  `ec63=130.598 mm`, and `ec612=276.902 mm`; the current distal visual mismatch
  is not caused by choosing the wrong robot variant. Evidence:
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`,
  `simulation/mujoco_guided_wire_env.py`, and the real `records.jsonl`.
- [simulation][data] A second audit compared MuJoCo `ec66` body positions for
  `base_link/link1...link6/flan` with direct URDF FK under the real baseline
  joints. Maximum per-link position error was `0.00004 mm`, ruling out the
  ROS-to-MuJoCo joint-origin/axis conversion as the cause of distal visual
  misalignment. Remaining candidates are camera projection/base tilt,
  requested-vs-measured real joints, or link-visual registration. Evidence:
  `simulation/mujoco_guided_wire_env.py`, `robot_assets/standardized/elite_description/urdf/ec66_description.urdf`.
- [simulation][data] RealSense intrinsics identified a remaining camera-model
  mismatch after elevation tuning failed: MuJoCo free-camera FOV is the default
  `45 deg`, while the physical-top raw-side camera is `43.208 deg` vertical and
  `70.195 deg` horizontal. Recorded color distortion coefficients are zero;
  FOV should be fixed from intrinsics before more distance/elevation tuning.
  Evidence: `real_pilot_20260721_200628_left_s_bend_rgbd/manifest.json`,
  `simulation/mujoco_guided_wire_env.py`.
- [simulation][data] Per-camera `fovy_deg` is now applied by MuJoCo rendering
  and adjustable at `0.1 deg` resolution. The real-alignment config initializes
  Top/Side to measured RealSense vertical FOV `43.208/43.166 deg`; a headless
  `43.208 -> 70 deg` probe changed the rendered image with MAE `32.35`. Elite
  alignment is Top-only because the physical Side view contains too little
  Elite geometry for a useful constraint. Evidence:
  `simulation/mujoco_guided_wire_env.py`, `tools/adjust_mujoco_camera.py`,
  `simulation/camera_configs/mujoco_camera_realalign_20260721_v1.json`.
- [simulation][data] Elite visual diagnosis now supports preview-only colored,
  labeled screen-space markers projected from exact centers for joints 1-6 and
  the flange in the dual-camera alignment tool. The overlay remains visible
  over link housings and separates kinematic-center projection error from
  outer-shell or real-housing appearance mismatch without changing or saving
  scene parameters. Evidence:
  `tools/adjust_mujoco_camera.py`, `simulation/mujoco_guided_wire_env.py`.
- [simulation][data] Operator review found the joint-center overlay
  insufficient for real alignment because physical Top does not show the full
  Elite arm and contains no real center labels. It remains an internal
  kinematic-consistency diagnostic, not sim-to-real evidence.
- [simulation][data] Camera alignment now supports `0.1 deg` azimuth,
  elevation, and optical-axis roll, `0.1%` distance, `0.1 mm` lookat,
  `0.01 deg` FOV, and resolution-scaled principal-point offsets. July 21
  RealSense intrinsics initialize Top/Side FOV to `43.20848/43.16626 deg` and
  native `1920x1080` principal offsets to `(+6.317,+16.755)` and
  `(-21.900,+16.686) px`. Headless checks confirmed affine scaling, roll,
  formal-render application, and diagnostic-marker co-registration. Evidence:
  `simulation/mujoco_guided_wire_env.py`, `tools/adjust_mujoco_camera.py`,
  `simulation/camera_configs/mujoco_camera_realalign_20260721_v1.json`.
- [simulation][data] Dual-camera alignment now exposes direct `0.1 mm`
  camera-head world XYZ controls synchronized with the orbit representation.
  XYZ edits perform a true 3D translation while shifting lookat by the same
  delta, so view direction is preserved and scene parallax changes; orbit edits
  update XYZ in return. A headless Top probe requested `(+10,-5,+2) mm`, matched
  the camera-head displacement within floating-point tolerance, kept forward/up
  unchanged, and changed `55,833` pixels at `480x270`. The derived XYZ is not a
  second saved pose source. Evidence: `tools/adjust_mujoco_camera.py`.
- [simulation][data][blocker] Operator review rejected the latest manual
  camera/vessel alignment for formal use: Top RGB cannot constrain its
  line-of-sight axis or prove that the magnet is above the vessel. Camera XYZ
  was confirmed redundant with lookat translation; azimuth/elevation improved
  appearance but did not remove the 3D ambiguity. The current config remains a
  draft rather than a synthetic-data baseline.
- [data][real-system][blocker] A fail-closed frame-0 RGB-D relative-camera
  audit at `simulation_output/real_rgbd_registration_audit_20260723_v1` rejected
  all automatic routes. Physical Top/Side depth coverage was `80.98/73.36%`,
  but transparent-vessel depth contained holes/background leakage. SIFT/PnP
  produced only six inliers; FPFH had no consistent accepted transform; ICP
  from the manual camera initialization reached only `0.0303` fitness and
  drifted `0.158 m / 18.32 deg`. No transform may be written back. Next onsite
  calibration requires a rigid ChArUco/checkerboard target simultaneously
  visible in both cameras at multiple depths and tilts. Evidence:
  `tools/audit_real_rgbd_camera_registration.py`, `report.json` in the audit
  directory.

## 2026-07-23

- [decision][real-system] After reviewing the repeated delivery failures, the
  supervisor decided to redesign the replacement feeder comprehensively for
  stable clamping and thermal reliability. New real expert collection is paused
  until that redesign is complete; old overheating and hot-melt-fixture behavior
  must not be modeled as intended feeder semantics.
- [simulation][data] Real Elite TCP-path projection made manual dual-view
  adjustment materially easier. The saved Side/Top camera, vessel, EC66, and
  three-cylinder gray magnet scene is accepted as the current manual alignment
  baseline; ChArUco calibration is deferred, so this is not a measured camera
  calibration. Evidence: `tools/adjust_mujoco_camera.py`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`,
  `simulation/camera_configs/mujoco_camera_realalign_20260721_v1.json`.
- [simulation][data] The post-alignment left/right route-contact regression
  succeeded on the first attempt in `247/246` steps with gains `59.62/59.16`,
  zero stalls/boundary projections, and minimum tip-wall distances
  `1.674/1.804 mm`. This clears the bounded geometry/control regression, not
  formal training export. Evidence:
  `simulation_output/realalign_scene_route_contact_regression_v1`.
- [simulation][data][blocker] The same regression exposed that
  `render_camera_pair(size=224)` still saves both cameras as distorted
  `224x224` even when MuJoCo renders at `960x540`. The next data-track task is a
  backward-compatible 16:9 saved-observation interface before material tuning
  or another algorithm-facing export.
- [simulation][data] The square-output blocker is resolved in code with paired
  `--image-width/--image-height` collector options and backward-compatible
  legacy `--image-size`. Rectangular dimensions also propagate through the
  visual-distance and estimated-tip camera model. Smoke output
  `simulation_output/_smoke_formal_16x9_interface_v1` saved both cameras at
  `640x360` and declared aspect ratio `1.7777778`; one complete left/right
  regression remains before freezing the observation path.
- [simulation][data][blocker] Operator review superseded the numeric success of
  both `realalign_scene_route_contact_regression_v1` and
  `realalign_scene_route_contact_regression_16x9_v2`: Elite and guidewire
  visuals were grossly misregistered, so neither run clears geometry/control
  or formal-export readiness. The rectangular command also omitted
  `--lock-camera-world`, but locking alone did not fix the images.
- [simulation][data] The mismatch audit separated four faults: reset-time Elite
  IK overwrote saved joints, old wire points belonged to the zero-pose
  `0.077`-scale scene, their entry/control order backtracked, and Elite flange
  TCP was conflated with the magnetic offset. The real-aligned scene now fails
  closed without world-locked cameras, preserves saved joints, uses a
  registered entry-to-tip visual wire, and separates flange TCP from the
  `191 mm` distal magnetic point while preserving
  `elite_tcp_delta_6d + piper_intent_id`.
- [simulation][data] Bounded left/right 40-step smokes removed the initial
  `[-5.9,+20.4,-141.9] mm` false Elite target; the first TCP delta is now zero
  and later samples are millimetric, with no gross guidewire loop in internal
  frame review. Operator visual acceptance is still required before another
  full regression. Evidence:
  `simulation_output/_smoke_realalign_tcp_magnet_split_40step_v1`,
  `simulation_output/_smoke_realalign_tcp_magnet_split_right_40step_v1`.

## 2026-07-24

- [simulation][data] Closed the current real-aligned visual wire entry and
  centering regression by separating world-fixed outlet/entry overrides from
  vessel-scaled historical prefix points, then joining the prefix to the
  mesh-section-centered route. Operator adjustment plus internal Side/Top frame
  review found no entry jump, gross loop, coordinate break, or renewed wall
  hugging. The complete locked-camera left/right regression succeeded on the
  first attempt in `283/283` steps, with zero contact/boundary projections and
  minimum tip-wall distances `2.549/2.598 mm`. This is a bounded
  geometry/control milestone, not formal dataset or real-system validation.
  Evidence: `tools/adjust_realalign_wire_entry.py`,
  `simulation_output/realalign_split_entry_full_route_regression_v1`.
- [simulation][data] The accepted real-aligned scene also passed the sparse
  long-stroke event integration gate. Left/right both succeeded in `283` steps;
  each branch contains eight one-record feed targets and 134 hold targets, all
  `568` image references exist, and the event-semantics audit confirms that
  subsequent feed motion is controller-owned execution under hold targets.
  Exact and estimated contact remained zero, so this is neither a fixed-mm
  feeder calibration nor tactile validation. Evidence:
  `simulation_output/realalign_split_entry_longstroke_event_diag_v1`.
- [simulation][data] A semantically remapped visual-domain audit compared the
  accepted scene with both July 21 real anchors using simulation Side -> raw
  Top and simulation Top -> raw Side. Across 24 pairs per mapping, simulated
  luminance remained `206-207` versus real `106-119`, simulated bright-pixel
  fraction `55-56%` versus real `0-2%`, and real edge density was roughly
  `3-4.5x` higher. The next rendering priority is a reversible July 21 preset:
  amber vessel material first, then white/metal table, background, and
  exposure; color tuning alone cannot reproduce real lab clutter. Evidence:
  `simulation_output/realalign_visual_domain_audit_20260724_v1`.
- [simulation][data][decision] Accepted an axis-aligned future-capture visual
  baseline after operator review and a full integrated left/right regression.
  Camera `v2` fixes Top/Side azimuth to `180/0 deg` and roll to `0/0 deg`; the
  reversible lab preset uses an amber vessel, neutral EC66/magnet colors, a
  clean Top green curtain, and continuous table while omitting exposed room
  shell artifacts. Both branches succeeded in `283` steps with `142` samples
  each, `568/568` images, zero contact/boundary projections, minimum tip-wall
  `2.549/2.598 mm`, complete estimator fields, and a passing one-shot feeder
  event audit. Camera `v1` remains the reproduction path for old tilted July
  captures; `v2` requires the next real cameras to be physically aligned to
  the fixture axes. Evidence:
  `simulation/camera_configs/mujoco_camera_realalign_axis_aligned_v2.json`,
  `simulation_output/realalign_axis_v2_full_route_event_regression_v1`.
- [simulation][data] Fixed-start `0.54` axis-aligned coverage also passed for
  both branches in `220/217` steps with `438/438` images, zero
  contact/boundary projections, and a passing one-shot event audit. Top tip
  pixels were present on `218/219` samples and fused 3D tip fields on
  `219/219`. Evidence:
  `simulation_output/realalign_axis_v2_start054_full_route_event_regression_v1`.
- [simulation][data][blocker] Formal episode diversity is not guaranteed by
  changing only the seed: for fixed task and fixed `start_fraction`, reset
  initializes lateral offset to zero and the route-plan/controller are
  deterministic. Random start fractions produce different truncations but
  can share long identical route suffixes. A repeat-episode redundancy audit
  and source-trajectory grouping are required before choosing the formal
  episode count.
- [simulation][data] The first progress-aligned redundancy audit quantified the
  issue across the accepted `0.42/0.48/0.54` runs. No pair met the strict
  exact/near-exact trajectory threshold, but all six same-task cross-start
  pairs had high shared suffixes: `99-100%` of the shorter episode matched a
  route-progress sample in the other run and the images were highly similar.
  Tip/TCP differences and Piper phase prevent calling them exact duplicates.
  The behavior-diversity gate remains failed because no same-start/different-
  seed pair exists and fixed-start reset is deterministic. Evidence:
  `tools/audit_sim_trajectory_redundancy.py`,
  `simulation_output/realalign_axis_v2_multistart_redundancy_audit_v1`.
- [simulation][data][decision] Froze the design for later formal-data
  augmentation around the axis-aligned camera `v2` baseline. The plan separates
  episode-static camera installation offsets, bounded photometric transforms,
  and deferred temporally correlated jitter; requires pixel labels and
  effective intrinsics to follow geometric transforms; preserves
  `elite_tcp_delta_6d + piper_intent_id`; and keeps a canonical unaugmented
  evaluation subset. Augmentation remains unimplemented until fixed-start
  `0.48/0.54` coverage and new physically aligned real references are complete.
  Evidence: `docs/simulation-data-augmentation-plan.md`.
- [simulation][data] Fixed-start `0.48` axis-aligned coverage passed for both
  branches in `248/247` steps with `124` samples per branch, `496/496` images,
  zero contact/boundary projections, minimum tip-wall `2.549/2.598 mm`, and a
  passing one-shot event audit. One left frame lacked a Top tip pixel but used
  the valid Side view for the fused 3D estimate. Evidence:
  `simulation_output/realalign_axis_v2_start048_full_route_event_regression_v1`.
- [decision][simulation][real-system] The next physical experiment will retain
  the current vessel-support layout. Before formal synthetic generation, add a
  simplified visual-only support proxy using approximate white/translucent
  posts and base pads with collisions disabled. Do not model load-bearing
  mechanics or alter vessel/route physics; rerun dual-view estimator checks
  because support edges may affect image-derived tip/wall cues.
- [simulation][data][decision] Expert upgrade is now a separate formal-data
  readiness work item because camera augmentation cannot create behavior
  diversity. Keep the deterministic route-plan expert as the canonical
  regression baseline; develop an estimator-conditioned expert first as a
  diagnostic mode using real-observable tip/route, robot, controller, and
  confidence signals. Preserve `elite_tcp_delta_6d + piper_intent_id` and
  one-shot feed events; exact tip/contact/wall/progress truth, fixed feed-event
  distance, and contact-triggered retract remain excluded from formal expert
  input. Evidence: `docs/simulation-expert-upgrade-plan.md`,
  `simulation_output/realalign_axis_v2_multistart_redundancy_audit_v1`.
- [simulation][data] Implemented the diagnostic-only
  `estimated_tip_route_e1` expert without replacing the deterministic formal
  baseline. E1 consumes a strict estimator-field whitelist, uses full
  registered-route search without exact progress hints, applies confidence-
  gated bounded Elite feedback, and preserves the canonical one-shot Piper
  schedule. A 12-step left smoke produced `6/6` valid estimator updates, zero
  fallback steps, maximum observation age of one step, and a passing Piper
  event audit. Validity remains explicitly non-formal pending full-route,
  diversity, and real-calibration gates. Evidence:
  `simulation_output/_smoke_estimated_tip_route_e1_v1`,
  `docs/simulation-expert-upgrade-plan.md`.
- [simulation][data] The complete canonical E1 gate passed both branches in
  `283/283` steps with `568/568` images, zero contact/boundary projections,
  `142/142` valid estimator updates per branch, zero fallbacks, and a passing
  one-shot Piper event audit. Representative dual-view frames showed no new
  geometry regression. Against the canonical route-plan run, however, progress
  overlap remained `1.0`, tip RMSE was only `0.190/0.171 mm`, Piper agreement
  was `1.0`, and both task pairs retained high shared suffixes. E1 therefore
  passes observable-interface/physical closure but not behavior diversity or
  formal promotion. Evidence:
  `simulation_output/realalign_axis_v2_esttip_route_e1_fullroute_diag_v1`,
  `simulation_output/realalign_axis_v2_esttip_route_e1_vs_canonical_audit_v1`,
  `docs/simulation-expert-upgrade-plan.md`.
- [simulation][data] Added bounded diagnostic initial-state families at local
  route `n1 = -/+10%` radius while preserving exact zero-offset canonical
  reset. At start `0.42` this is about `-/+1.412 mm`. E1 now anchors the scene
  magnet offset to the registered route center and applies a `0.75` estimated
  lateral correction, rather than canceling the perturbation against the first
  estimated tip. Paired E1 diagnostics use common task-step estimator noise so
  differences are physical-state driven. Reset/response checks passed; a
  rendered positive 12-step smoke had `6/6` valid updates, zero fallback/contact,
  a passing one-shot event audit, and non-formal validity. Full signed
  left/right runs remain pending. Evidence:
  `simulation_output/_smoke_estimated_tip_route_e1_n1_pos10pct_v2`,
  `docs/simulation-expert-upgrade-plan.md`.
- [simulation][data][decision] The complete canonical/negative/positive E1
  initial-offset comparison rejected signed `n1` reset variation as a source of
  full-episode behavior diversity. All six episodes succeeded safely, but all
  `6/6` same-task pairs were near duplicates with high shared suffixes;
  canonical-to-signed tip RMSE was only `0.132-0.135 mm`, signed-to-signed RMSE
  `0.266-0.267 mm`, and Piper agreement `1.0`. The initial `1.41 mm` difference
  merged by roughly steps `8-14` and was visually negligible. Retain these only
  as short recovery diagnostics; do not enlarge them blindly or count their
  common suffixes as new demonstrations. Evidence:
  `simulation_output/realalign_axis_v2_e1_signed_initial_response_audit_v2/report.json`,
  `docs/simulation-expert-upgrade-plan.md`.
- [simulation][data] Added diagnostic feeder-response families that keep the
  one-shot `piper_intent_id` schedule unchanged and vary only controller-owned
  primitive duration. With requested duration `20`, weak/nominal/strong execute
  `16/20/24` steps; manifests prohibit fixed-distance interpretation and keep
  non-canonical families non-formal. A 22-step weak smoke confirmed the sole
  feed target at step 0, executed feed through sampled step 16, zero execution
  at steps 18/20, `11/11` valid estimator updates, zero fallback/contact, and a
  passing event audit. Full three-member left/right comparison is pending.
  Evidence: `simulation_output/_smoke_e1_feeder_weak80pct_v1`,
  `docs/simulation-expert-upgrade-plan.md`.
- [simulation][data][decision] The full canonical/weak/strong feeder-response
  comparison completed with all six left/right episodes successful. Nominal,
  weak, and strong finished in `283/283`, `334/334`, and `234/228` steps;
  minimum tip-wall distance was `2.548-2.598 mm`, with zero contact, boundary
  projection, and E1 fallback, and all event audits passed. Response variation
  persistently changed episode length and Elite actions, but all `6/6`
  same-task pairs retained high shared suffixes, so the behavior-diversity gate
  remains false. Treat the family as a controller-response robustness factor,
  not independent demonstrations, and calibrate it against the redesigned real
  feeder before formal randomization. Evidence:
  `simulation_output/realalign_axis_v2_e1_feeder_response_audit_v1/report.json`,
  `docs/simulation-expert-upgrade-plan.md`.
- [data][decision] Guidewire slack/curvature will not be used as a simulation
  diversity or augmentation variable. Collection uses the same guidewire and
  a standardized reset, so pre-feed Side/Top frames serve only as reset-quality
  checks. Future calibration of the diagnostic response family should use the
  redesigned feeder's execution-response evidence, not artificial slack
  categories. Evidence: `docs/simulation-expert-upgrade-plan.md`,
  `docs/simulation-data-augmentation-plan.md`.
- [data][decision] Clarified the same-wire reset constraint: guidewire material
  slack is not an independent diversity variable, but manual reset can cause
  small correlated differences in axial start, entry angle, and visible
  curvature. Future pre-feed Side/Top frames should measure this narrow range
  and reject large reset errors; simulation must not invent categorical slack
  variation. Evidence: `docs/simulation-expert-upgrade-plan.md`,
  `docs/simulation-data-augmentation-plan.md`.
- [simulation][data] Added a simplified render-only vessel-support proxy to the
  current real-aligned scene: a translucent white base plate and five white
  route-anchored post/pad assemblies approximated from real pilot `200628`
  step 0. Collision and physics effects are explicitly disabled, vessel pose is
  unchanged, and no policy state field is added. An 8-step E1 dual-view smoke
  had `4/4` valid estimator updates, zero fallback/contact/boundary events, and
  a passing feed-event audit. The support changed visual edge distance but did
  not change tip 3D estimates or create a false contact. Full left/right visual
  regression remains pending. Evidence:
  `simulation_output/_smoke_vessel_support_visual_v1`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data][decision] The complete vessel-support visual regression is
  accepted. Left/right both succeeded in `283/283` steps with `568/568` images,
  `142/142` valid estimator updates per branch, zero contact/boundary/fallback,
  and a passing feed-event audit. Against the no-support baseline, physical
  trajectories and actions were exactly unchanged (`tip RMSE = 0`, Elite MAE
  `= 0`, Piper agreement `= 1.0`) while normalized image MAE was `0.017-0.026`.
  Support edges changed visual distance by at most `3.98/7.59 px` for
  left/right but changed no contact flags or estimated tip positions. Reviewed
  start/middle/end dual-view frames showed no occlusion or geometry regression.
  Exact trajectory duplication is required for this render-only change and is
  not new behavior data. Evidence:
  `simulation_output/realalign_axis_v2_support_visual_fullroute_regression_v1`,
  `simulation_output/realalign_axis_v2_support_visual_vs_no_support_audit_v1/report.json`.
- [simulation][data][decision] Operator review found that several simplified
  support top pads do not visually meet the vessel surface, despite the earlier
  physics/estimator regression passing. Support placement is therefore a draft,
  not a finished visual baseline. Added a live Side/Top adjuster for per-anchor
  route fraction, world XY offset, and center-to-top height; saving marks the
  scene pending a new full regression. The rectangular top-pad shape remains a
  deferred visual proxy and has no contact-mechanics meaning. Evidence:
  `tools/adjust_vessel_support_visual.py`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data][decision] After manual support alignment, removed original
  support 2 and retained stable support ids `1/3/4/5`. Added an explicit table
  top offset and set it to the vessel scene-bounds minimum Z; this raises the
  table by about `4.533 mm` from the old automatic default and shortens all
  retained post cylinders by the same amount without moving their adjusted top
  pads. The new four-support raised-table scene supersedes the earlier
  five-support visual regression and remains pending a new full-route check.
  Evidence:
  `simulation_output/_smoke_vessel_support_adjuster_v1/preview_table_at_vessel_min_supports_1345.png`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data] Added a signed live table-height control to the vessel
  support adjuster. The value is table-top offset from vessel minimum Z;
  positive raises and negative lowers the table over `-50..+50 mm`. Only the
  table body, support base plate, and post bottoms move. Piper/Elite mounts,
  robot bodies, vessel pose, and adjusted top pads remain fixed so prior robot
  alignment is preserved. A non-saving `+10 mm` preview confirmed the intended
  isolation. Evidence:
  `simulation_output/_smoke_vessel_support_adjuster_v1/preview_table_plus10mm_no_mount_move.png`,
  `tools/adjust_vessel_support_visual.py`.
- [simulation][data][decision] The operator saved table-top offset `+43.3 mm`
  because the visually relevant vessel bottom differs from the STL AABB
  minimum. Verification found that environment rebuild still indirectly moved
  Piper/Elite mount visuals by deriving mount Z from table height, despite the
  live adjuster keeping them fixed. Decoupled mount construction with a fixed
  reference offset of `-4.532764698 mm` from scene minimum; corrected mount
  world positions are Piper `[-0.28, -0.137, -0.059213]` and Elite
  `[-0.451, 0.193, -0.059213]` while the table remains at `+43.3 mm`. A new
  visual confirmation and full-route regression remain pending. Evidence:
  `simulation_output/_smoke_vessel_support_adjuster_v1/preview_operator_plus43p3mm_fixed_mounts.png`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data][decision] The saved `+43.3 mm` table exposed a rendered-tip
  visibility failure: transparent amber-vessel compositing shifted tip HSV hue
  from `7-8` to `13-14`, so the old red-mask upper bound `10` caused `4/4`
  `tip_not_visible` fallbacks. Rendering a purer red tip did not solve the
  compositing shift. Added scene-specific `visual_tip_red_hue_max=14` while
  retaining default `10` elsewhere and recording the value in provenance. The
  repeated 8-step E1 smoke restored `4/4` valid updates with zero fallback or
  estimated contact, image distance `8.97-12.39 px`, and a passing event audit.
  Full left/right regression remains pending. Evidence:
  `simulation_output/_smoke_vessel_support_operator43p3_hue14_v1`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data][decision] Replaced the four accepted support top-pad box
  proxies (`1/3/4/5`) with operator-approved conformal render-only meshes. The
  generator samples the transformed vessel underside, fills local gaps up to
  `8 mm` with `0.4 mm` overlap, trims unused rectangular cells, and retains only
  a small connection above each post. Collision, physics, vessel pose, and
  policy state remain unchanged. An 8-step E1 smoke retained `4/4` valid visual
  updates, zero fallback/contact/boundary, `8/8` images, image distance
  `8.97-12.39 px`, and a passing event audit; versus the box-pad smoke, tip
  RMSE and Elite MAE were `0` and Piper agreement was `1.0`. Full left/right
  regression remains pending. Evidence:
  `simulation_output/_smoke_vessel_support_conformal_set_v1`,
  `simulation/assets/vessel_support_realalign_20260721/`,
  `tools/generate_vessel_support_conformal_prototype.py`.
- [simulation][data][interface] Renumbered the four retained visual supports
  contiguously after deletion of the original support 2. The mapping is old
  `1/3/4/5` to current `1/2/3/4` (`1->1, 3->2, 4->3, 5->4`); mesh asset names
  were changed consistently. This is display/provenance metadata only and does
  not change anchor geometry, rendering, collision settings, observations, or
  the `elite_tcp_delta_6d + piper_intent_id` interface. Historical manifests
  retain their original ids. Evidence:
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`,
  `simulation/assets/vessel_support_realalign_20260721/`.
- [simulation][data][decision] Accepted the complete conformal-support visual
  regression. Left/right both succeeded in `283/283` steps with `142/142`
  valid estimator updates per branch, `568/568` images, zero fallback/contact/
  boundary events, minimum visual distance `5.249 px`, and a passing Piper
  event audit. Against the box-pad full-route baseline, tip RMSE and Elite
  translation/rotation MAE were `0`, Piper agreement was `1.0`, and reviewed
  images changed without occlusion or false-edge regressions. Exact trajectory
  duplication is expected because support geometry remains render-only. The
  run manifest predates the id remap and records `1/3/4/5`; a post-remap
  `1/2/3/4` scene preview loaded the renamed byte-identical assets with fixed
  robot mounts, so numbering alone does not require another route run.
  Evidence:
  `simulation_output/realalign_axis_v2_support_conformal_fullroute_regression_v1`,
  `simulation_output/realalign_axis_v2_support_conformal_fullroute_regression_v1/keyframe_sheet_side_top.png`,
  `simulation/scene_configs/mujoco_scene_realalign_20260721_v1.json`.
- [simulation][data][decision] Froze the final real-aligned scene and E1
  interface with a repeatable aggregate provenance audit. All `10/10` checks
  pass: four conformal supports are render-only and absent from state; all
  `284` actions contain finite `elite_tcp_delta_6d`; Piper intent counts are
  `feed=16`, `hold=268` with valid ids and separate executed-command logs; the
  E1 runtime whitelist exactly matches its manifest, has no exact-truth
  overlap, and all eight estimator inputs have provenance. Full-route, event,
  visual, and render-only trajectory gates also pass. Status is explicitly
  `passed_diagnostic_freeze`, while the complete artifact remains
  `diagnostic_oracle`, `formal_data_allowed=false`, and
  `formal_training_ready=false` because behavior diversity and post-redesign
  real calibration remain open. Evidence:
  `docs/_final_scene_interface_provenance_audit_20260725/`,
  `tools/audit_final_scene_interface.py`,
  `tools/audit_observation_provenance.py`.

## 2026-07-25

- [data][estimator][decision] Rebuilt the 7096-record contact supervision pack
  with source/family provenance and conservative same-task split grouping:
  `52` raw episodes become `44` source trajectories and only `2` valid
  left/right families. The honest unweighted family-held-out RGB estimator is
  weak (`AP/F1=0.1825/0.2922` overall; `0.0875/0.1443` on right), has poor
  calibration (`ECE=0.2518`), and produces many pure temporal false runs. On
  the unlabeled July real pilot, its two fold models disagree at opposite
  extremes on `300/300` records, leaving `0/300` valid ensemble predictions.
  `estimated_contact_risk_*` remains `policy_input_allowed=false`; do not tune
  weights/thresholds or deploy the fold ensemble. Next data action is
  `contact_probe_v2` with distinct contact geometry and hard negatives.
  Evidence: `simulation_output/contact_estimator_oof_family_v4_unweighted_v1`,
  `simulation_output/contact_estimator_real_pilot_ood_family_v4_unweighted_v2`.

## 2026-07-26

- [simulation][data][estimator] Accepted the model-independent
  `contact_probe_v2` raw diagnostic pilot after correcting magnetic-point IK,
  profiling the tip lateral cap, scaling it by declared severity, and reducing
  only the two long-clear caps. All `16/16` episodes complete, with `1280`
  samples, `2560/2560` images, coverage across both tasks, three route regions,
  both probe axes/signs, four severity classes and four temporal shapes, and no
  contact outside declared windows, failures, provenance issues, or online
  exact-truth use. Full episodes share route suffixes, so the accepted
  estimator view keeps `524` active-window samples and groups `16` sources into
  six conservative `task x route_region` families. Fixed-fold integrity passes
  with zero cross-family/fold near-duplicate or shared-image pairs. Evidence:
  `simulation_output/contact_probe_v2_pilot_v3`,
  `simulation_output/contact_probe_v2_pilot_v3_active`,
  `simulation_output/contact_supervision_probe_v2_active_v1`.
- [data][estimator][decision] Six-family OOF on the active probe view improves
  RGB AP/F1 to `0.8345/0.7182` (`0.7817/0.6667` left,
  `0.8882/0.7602` right), but calibration remains imperfect
  (`Brier/ECE=0.1715/0.1506`), the weakest family has AP/F1
  `0.4119/0.4151`, and temporal audit finds `14` pure persistent risk runs.
  Corrected-camera-role real OOD leaves `268/310` left and `132/140` right
  predictions within the disagreement limit, but zero valid positives and no
  real contact ground truth. This reduces the previous fold-collapse problem
  without establishing real accuracy; `estimated_contact_risk_*` remains
  `policy_input_allowed=false`. Evidence:
  `simulation_output/contact_estimator_oof_probe_v2_active_v1`,
  `simulation_output/contact_estimator_real_ood_probe_v2_left_v1`,
  `simulation_output/contact_estimator_real_ood_probe_v2_right0_139_v1`.
- [simulation][data][expert][decision] Implemented the diagnostic E2 observable
  feed-event loop with separate requested/accepted/executed state and per-event
  A/B/C/D response coverage. The revised two-sequence `v2` pilot completes all
  four left/right episodes (`788` samples, `1576/1576` images) and passes its
  interface, provenance, event-response, and event-timing gates. Full-episode
  redundancy remains negative: both ABCD/BACD same-task pairs have full
  progress overlap, high shared suffixes, tip RMSE about `0.20 mm`, Elite MAE
  about `0.29 mm`, and `96.9%` Piper-label agreement. E2 is not training-ready;
  response-order permutations will not be scaled, and the next expert change
  must create observable spatial recovery differences. Evidence:
  `simulation_output/realalign_axis_v2_e2_two_sequence_pilot_audit_v2/`.
- [simulation][data][expert][decision] The E2 spatial-recovery pilot clears the
  small diagnostic diversity gate without changing
  `elite_tcp_delta_6d + piper_intent_id`. Four successful CABD/DABC left/right
  episodes contain `836` samples and `1672/1672` valid images; event/interface
  and provenance audits pass. Same-task full episodes are no longer near
  duplicates (tip RMSE `1.841/1.723 mm`, Elite translation MAE
  `1.526/1.404 mm`), and the dedicated recovery-window gate passes. Both pairs
  still share the common route suffix and remain grouped as two conservative
  task families. This accepts the diagnostic expert behavior, not a formal
  training pack; fixed split and sample-weighting design remain open. Evidence:
  `simulation_output/realalign_axis_v2_e2_spatial_pilot_audit_v1/`.
- [data][expert][packaging][decision] Built and fail-closed validated the E2
  spatial diagnostic overlay: `836` aligned rows from four complete source
  trajectories, two conservative same-task families, and `1672/1672` valid
  images pass all `12` packaging checks. Feed/recovery decisions are upweighted,
  execution holds are downweighted, and every source contributes equal total
  weight. Policy samples preserve `elite_tcp_delta_6d + piper_intent_id` and
  exclude exact simulator truth, which remains in a diagnostic sidecar. Both
  families are fixed to `overlay_train`; no internal validation split is claimed
  because their route suffixes overlap. Status is
  `diagnostic_overlay_allowed=true`, `policy_training_ready=false`, and
  `formal_data_allowed=false`; one external-validation algorithm ablation is
  required before any scale-up. Evidence:
  `simulation_output/e2_spatial_candidate_pack_v1/`.

## 2026-07-27

- [algorithm][data][E2][decision] The strict fresh paired PI05 comparison
  rejects scaling the current CABD/DABC diagnostic overlay. With identical
  851/154 base split, seeds, final steps, validation, and base-only Piper class
  weights, the overlay worsens Elite translation MAE from `0.7733` to `0.9085`
  (`+17.49%`), including transition/steady regressions of `+35.04%/+9.55%`.
  Piper improves only slightly (`accuracy +0.0065`, balanced accuracy
  `+0.0120`), with no transition-accuracy gain. Stop expanding or tuning the
  current CABD/DABC families and retain the fresh paired baseline. This is not
  a rejection of future genuinely independent geometry/recovery families and
  does not establish rotation, contact/tactile, formal-readiness, or
  real-system claims. Evidence:
  `simulation_output/pi05_e2_weighted_overlay_comparison_v1.json`.
- [data][algorithm-interface][E2][decision] The old stride-5 base and E2 overlay
  have a scene/camera/absolute-pose domain break, but the accepted current-scene
  E1 left/right regression provides a valid matched nominal bridge. The
  current-scene audit finds `409/836` supported image+TCP matches (`237/172`
  left/right) and exact same-seed reset pose/action agreement. Supported
  `clear_response` pairs have `0%` Elite direction conflict, versus
  `20.93%/27.08%/27.78%` for execution-hold/observe-effect/recovery roles. E2
  records observable event state, but the negative algorithm experiment did not
  consume it; event-boundary results remain uninterpretable until the existing
  fields are mapped. E3 may use task-normalized residual bands only as
  diagnostic coverage gates and should start near a `10%` overlay dose.
  Evidence: `simulation_output/current_scene_nominal_e2_bridge_audit_v1/`.
- [simulation][data][expert][E3a][decision] The complete E3a bridge pilot passes
  structural and observable event-state packaging but is rejected for behavior
  diversity. Its `4350` policy rows cover `12` successful episodes,
  `8700/8700` valid images, four whole families, both tasks, all four event
  statuses, and clean `150` request / `144` next-step accepted / `6`
  terminal-censored alignment. However, mild and moderate Elite residual
  medians are effectively identical on both tasks, all `12/12` same-start
  condition pairs are near duplicates, and all `30/30` same-task pairs retain
  high shared suffixes. Reset-only 5%/10% offsets do not create an observable
  recovery bridge; E3a is not an algorithm overlay and cannot be scaled or
  repaired by weighting. The next pilot must first gate one matched
  nominal/mild/moderate triplet using a persistent observable perturbation.
  Evidence: `simulation_output/e3a_geometry_recovery_bridge_pack_v1/`,
  `simulation_output/e3a_geometry_recovery_bridge_redundancy_v1/`.

## 2026-08-01

- [simulation][data][MuJoCo][decision] The D0 formal physical-cable integration
  design and Chinese group-meeting view were visually accepted, then the
  separately predeclared D1 default-off composite compile check passed `18/18`.
  The unchanged `tip_line` baseline and opt-in `mujoco_cable_v2` candidate both
  compile in the formal scene; the candidate contains one feed-slide DOF, `87`
  internal cable DOF, one visible physical-wire owner, hidden contact walls,
  and a bounded feed actuator. No `MjData`, renderer, `mj_step`, camera render,
  dynamics, or data generation occurred. D2 initialization/render and all later
  stages remain locked pending visual review. Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_compile_v1/`.
- [simulation][data][MuJoCo][decision] The Chinese D1 composite-compile summary
  was visually accepted. A separate hash-bound receipt closes D1 while leaving
  the original report immutable. This permits only predeclaring D2 no-step
  initialization and dual-camera rendering; it does not yet authorize
  `MjData`, rendering, dynamics, or data generation. Evidence:
  `simulation/bifurcation_root_feed_lateral_successor_v2_formal_integration_D1_compile_visual_review_v1.json`.
- [simulation][data][MuJoCo][decision] The authorized D2 static initialization
  and aligned Side/Top render diagnostic is rejected at `17/21` gates without
  calling `mj_step` or advancing time. The new physical cable initializer does
  not inherit the accepted split-prefix/manual bridge, and its `56.07 mm`
  feed-zero span is shorter than the `82.46/80.81 mm` required by the declared
  left/right start-0.42 cases. D3 dynamics and data generation remain locked;
  a separately predeclared D1.1 path-ownership and cable-length revision is
  required. The Chinese diagnostic remains pending user visual confirmation.
  Evidence:
  `simulation_output/mujoco_cable_formal_integration_D2_initialization_render_v1/`.
- [simulation][data][MuJoCo][decision] The D2 failure interpretation was
  visually accepted, and the bounded D1.1a reset-curve audit passes `15/15`
  without `MjData`, rendering, dynamics, or backend changes. The accepted full
  reset path is about `506-598 mm`; preserving the current link spacing would
  require `263-311` dynamic bodies and `786-930` internal DOF. Its length also
  varies by `91.35/84.38 mm` across left/right starts `0.42-0.54`. Therefore a
  `58 mm` length-scalar repair is rejected: formal physical integration must
  first define upstream slack ownership and a feed-through boundary. Candidate
  topology and D2/D3 remain locked pending visual review of the Chinese audit.
  Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_1_reset_curve_audit_v1/`.

## 2026-08-02

- [simulation][data][MuJoCo][decision] D1.2c static coverage audit confirms the
  current formal start080 model has no complete vessel collision backend: the
  local capsule proxy covers only about `12%` of active cable arclength and
  mass, while the full `vessel_mesh` is render-only. The user therefore
  reclassified the prior `50 MPa`, `5 GPa`, and `500 MPa` runs as invalid
  material trials; they remain diagnostic evidence but consume `0/3` valid
  trials and cannot rank `bend` values. A new dynamic series may start only
  after a complete collision backend passes static and visual acceptance.
  Evidence:
  `simulation_output/mujoco_cable_formal_integration_D1_2c_complete_collision_coverage_static_audit_v1/`,
  `simulation/bifurcation_root_feed_lateral_successor_v2_formal_integration_D1_2c_material_trial_validity_adjudication_v1.json`.

## 2026-09-05

- [simulation][data][decision] E2es从E2er后继续磁头约束因果筛查：正常弯曲/扭转材料下，
  只关闭磁头7个weld，A复用原基线，B从200步smoke严格续算至2000步（累计0.5s）。
  零步24/24，smoke与完整诊断均37/37；零动力学恢复/形变审计5/5。末0.1s平均尖端
  上移A/B为0.078208/0.077804mm，磁头内部形变几乎一致；本次不支持额外weld约束
  是上移主因，也不能排除材料/壁面支撑。初态、reset、正式数据、policy权限继续关闭，
  中文图待用户目测。证据：`simulation_output/finite_tip_E2es_head_weld_knockout_full_v1/report.json`
  和同目录`head_deformation_audit.json`、`E2es_磁头约束关闭位移对照_中文.png`。
- [simulation][data][decision] 用户完成E2et正常配置2s静置（复用2000步、新增6000步），
  并明确拒绝持续上移的结果。数值/结构/取景12/12不覆盖物理视觉拒绝；尖端末态上移
  0.783239mm，末0.1s趋势+0.616281mm/s。17个保存状态的零动力学复核确认2s磁头
  质心上升0.804843mm、全导丝质心下降0.406128mm，根因仍未定位。冻结同配置延时，
  初始化/reset/正式数据/policy权限保持关闭。证据：
  `simulation/finite_tip_E2et_normal_hold_two_seconds_visual_review_v1.json`及
  `simulation_output/finite_tip_E2et_saved_motion_audit_v1/report.json`。
- [simulation][data][experiment] 用户完成E2eu单变量阻尼对照：A复用E2et 8000步，B从
  同一原始零速初态，仅将导丝阻尼0.04改为0.004，运行8000步/2s。自动12/12，独立
  hash/恢复/硬限审计通过；A/B末态尖端上移0.783239/2.862196mm，B晚期0.1/0.5s
  斜率为-0.399362/-0.391592mm/s。B磁头质心上升3.219697mm、全导丝质心下降
  2.535361mm；降低阻尼未消除2s净上移，晚期回落不能证明贴壁或稳定。未将0.004
  接受为新基线，不继续阻尼扫描，视觉裁决仍待用户。证据：
  `simulation_output/finite_tip_E2eu_tenth_damping_pair_v1/report.json`和
  `simulation_output/finite_tip_E2eu_saved_motion_audit_v1/report.json`；agent新动力学0步。
- [simulation][data][diagnostic] 按用户关于近头部贴壁支撑的假设，E2ev恢复A17/B21
  个已有状态，零动力学6/6。2s上游管壁反力Fz合计A/B为+1.598361/+2.299874mN；
  A磁头末端255的管壁反力为-0.229824mN，B未检出磁头承载性管壁接触。上游支撑
  与磁头间仍有16–17个ball joint；A2相邻231/232端帽接触点近重合（0.335nm）、
  同法线且两条均承载，B2的229/230相距0.0707865mm。该发现定位了接触离散与
  支撑传力候选，尚不证明错误双算或上移根因。未新增动态试验、未改物理参数，正式
  权限关闭、用户视觉裁决未变。证据：
  `simulation_output/finite_tip_E2ev_near_head_support_audit_v1/report.json`及同目录中文图。
- [simulation][data][implementation] 用户同意E2ew相邻端帽接触去重对照；已实现仅B
  生效的native mjcb_control规则及续跑入口，A复用E2et、B拟从同t0运行2s，两组阻尼
  均0.04。只处理224–232相邻端帽、同wall、点距≤1µm、法线差≤0.1°且支持中心
  位于共享端点的接触，保留更深者。零步预检19/19、纯选择测试20项通过；470模型
  数组完全相同，no-op与原生forward及两种规则插入时机的结果逐位一致。A2保存
  状态命中两对、EFC78→70，仅用于静态方法审计。B动力学未运行，2s仍由用户执行，
  接触去重不是已验证物理修复。证据：
  `simulation/finite_tip_E2ew_adjacent_endcap_dedup_pair_contract_v1.json`及
  `simulation_output/finite_tip_E2ew_adjacent_endcap_dedup_pair_v1_preflight/report.json`。
- [simulation][data][verification] E2ew续跑/回调/异常零积分测试16/16通过；覆盖非零
  qvel/warmstart恢复、非法输入与hash边界、Ctrl+C、step异常注入后的failed_closed。
  真实mj_step/mj_step2/mj_implicit调用均0。B2s仍待用户运行。证据：
  `tools/test_finite_tip_E2ew_runtime.py`。
- [simulation][data][experiment] 用户完成E2ew 8000步/2s，自动13/13。1581个积分步
  触发1958条端帽去重记录，A/B末态尖端上移0.783239/0.783512mm；末0.1s均值
  0.726570/0.710179mm，斜率+0.616281/+0.698848mm/s。干预改变了细部轨迹但
  未消除持续上移，不能将近重合端帽接触视为已确认主因；上游支撑传力仍待定位。
  纯轨迹只读审计5/5，未新增动力学、未改正式权限、未代替用户作视觉裁决。证据：
  `simulation_output/finite_tip_E2ew_trace_outcome_audit_v1/report.json`及同目录中文图。
- [simulation][data][verification] E2ew独立恢复/求解审计12/12：21个checkpoint及20个
  保存post-state去重列表吻合，46个原始输入hash不变。1.6s同态231/232原法向力
  0.706545/0.103731mN，去重后0/0.796965mN，证明承载重新分配、干预实际生效；
  快照力不用于位移归因，末态无命中不能代表全程无干预。新增积分0步。证据：
  `simulation_output/finite_tip_E2ew_checkpoint_execution_audit_v1/report.json`。
- [simulation][data][diagnostic] 用户同意E2ex支撑至磁头的运动/传力审计；恢复A17/B21
  已有状态，新增积分0步。运动6/6、独立几何审查通过：232材料近端升高A/B
  0.786688/0.789242mm，磁头尖端升高0.783239/0.783512mm，但248轴线仰角
  变化-0.214775/-0.219206度，全链COM下降0.406128/0.400294mm。全链约205
  起由下沉转为升高；224–248区间232局部转角最大约0.0759度，不能指定为唯一
  转轴。载荷11/11、266截面Newton–Euler闭合：J248末态向上传力
  0.351729/0.394933mN，弹性矩仍朝低头，正净抬头矩由更大阻尼投影构成；
  阻尼对相对运动做功非正。尚未确定具体接触来源或位移因果；下一步定位205–232
  的接触迁移与运动传递。未改参数/正式权限，用户视觉待定。证据：
  `simulation_output/finite_tip_E2ex_support_motion_audit_v1/report.json`、
  `simulation_output/finite_tip_E2ex_load_transfer_audit_v1/report.json`及同目录中文图。
- [simulation][data][diagnostic] 用户批准并完成E2ey零积分全链接触位置审计：A17/B21
  已有状态、396条几何接触、12/12；97个来源hash和各组470个模型数组保持。
  逐条胶囊表面/轴线重建最大残差3.18e-17m，独立方法/数值审查通过。A2s材料
  228/232近端升高0.718955/0.786688mm，同wall1456/1489加权承载位置变化
  -0.148448/+0.237741mm，不能把材料上移等同支点上移。224–232在A17/B20
  个保存状态承载；B1.9s仍+1.383930mN，2s无接触但上游截面仍向上传力。
  下一项建议为固定wall1456/1457/1489的局部支撑移除诊断，尚未实现/运行；
  删除接触不是物理修复，正式权限与用户视觉裁决未变。证据：
  `simulation_output/finite_tip_E2ey_contact_migration_audit_v1/report.json`、
  `location_summary.json`及同目录7张中文图。
- [simulation][data][implementation] 用户批准E2ez固定wall1456/1457/1489局部支撑
  移除诊断，runner/contract/callback/测试及中文范围图已落地。A复用E2et，B从原始
  零速完整初态开始；仅排除选定wall与skin43–255的活动约束，保留几何及原1.5mm
  全穿透停止限值。最终runner生产预检22/22、470模型数组A/B一致；t0/A2分别
  移除3/4接触，剩余EFC身份及几何保持，独立分阶段求解逐位一致。零积分测试
  23/23（一次prepare及最终字段纯fixture补验），异常/恢复/权限/hash拒绝路径
  核验；全部8积分API真实调用0。尚未启动B的8000步/2s，命令交由用户执行。
  下降不构成物理修复或根因确认，正式权限关闭、用户视觉待确认。证据：
  `simulation_output/finite_tip_E2ez_local_support_knockout_pair_v1_preflight/report.json`、
  `simulation_output/finite_tip_E2ez_local_support_knockout_tests_v1/report.json`及
  `simulation_output/finite_tip_E2ez_support_scope_v1/E2ez_局部支撑移除范围_中文.png`。

## 2026-09-06

- [simulation][data][experiment] 用户完成E2ez 8000步/2s，自动17/17。三处固定墙体
  的pre-step接触移除60348条实际生效；A/B尖端上移0.783239/0.356954mm，末
  0.1s均值0.726570/0.335072mm、斜率+0.616281/+0.264604mm/s，仍未消除上升。
  source232近端升高0.786688/0.280632mm，B相对A额外世界X偏移-0.766316mm。
  邻墙1424/1490分别在0.00475/0.7625s首次承载，从0.935s起局部190–247每个
  post-state样本有至少一处非目标支撑；单墙均断续，不能指定稳定支点。晚窗局部
  平均Fz+1.371465mN，两墙+0.643344/+0.728122mN，磁头平均-0.254934mN。
  被移除区域最大穿透0.445831mm，未触发1.5mm硬限；不能把位移改善比例当支撑
  因果份额或物理修复。下一步用已有状态查看225–236三维几何/传力，不扩大删墙。
- [simulation][data][verification] E2ez独立JSON/NPZ审计20/20：76个输入hash不变，
  A8000步重建、B8000行/20分块、21个checkpoint与报告算术核对，未prepare或
  积分。补充24输入hash纯JSON替代承载/材料位移分析及中文4面板图，agent已看，
  用户视觉仍待确认；原始报告未改、正式权限关闭。证据：
  `simulation_output/finite_tip_E2ez_trace_outcome_audit_v1/report.json`、
  `simulation_output/finite_tip_E2ez_replacement_support_analysis_v1/report.json`及
  `E2ez_位移与替代支撑_中文.png`。
- [simulation][data][diagnostic] 用户批准并完成E2fa已有状态三维几何/传力审计：
  A8/B10、15/15，8积分API调用0，80来源hash和各470模型数组保持。233几何接触
  （70被排除零反力）、1080最短轴距、324截面；6D闭合最大9.46e-16N/
  2.19e-17Nm。独立纯几何14/14+JSON16/16，198个相邻截面单段平衡残差≤
  1.49e-16N。B1.9s235/1490接触Fz+.998031mN，J235→236跃升+.957403；
  B2s226/1424接触+1.294628mN，J226→227跃升+1.249691，差为重力/惯性。
  承载在段内与端帽间迁移，不能称固定整段；B2s J248上传+.432657mN但累计上移
  仍小于A，瞬时力不代表位移份额。恢复态最大穿透.419087mm，全程.445831峰值
  在无checkpoint的3147步未恢复。4中文图已agent看，用户视觉待定、正式权限关闭。
  结束同类快照审计；下一项建议原完整接触A上的局部分辨率敏感性对照，需先验证
  物理/状态等效，尚未实现或运行。证据：
  `simulation_output/finite_tip_E2fa_replacement_geometry_audit_v1/report.json`、
  `segment_force_balance.json`、`visual_summary.json`及同目录4张中文图。

- 2026-09-06：完成E2fb原完整接触A的224–236局部二分零步预检。v1遗漏
  compiled sameframe=INERTIA导致3.3026e-7m胶囊偏移，失败报告及源码保留；
  v2修复后XML16/16、实际模型28/28、独立JSON/NPZ50/50+AST7/7通过。
  213体/212球关节→226体/225球关节，原中心线/节点误差≤5.55e-17m，
  mass/COM/全惯量重组成立，新关节锁零后的质量矩阵差1.69e-21。13段均匀
  capsule解析惯量补证通过，原材料邻接exclude展开保持，原墙/根/7磁头weld
  和EI/GJ保持。新mid与J225–237阻尼从.04按半长补偿为.08，局部同轴线性
  串联指标不变；但原折角初始弹性力矩严格约翻倍（7.836–8.629→15.672–
  17.258µNm），新mid≈0。机械等效门槛失败，未写/运行2s动力学入口，8积分
  API调用0；60原来源hash前后保持。离散应变二次型38.180404→39.005464µJ，
  不称MuJoCo总能量或严格有限转动势能。下一步需跨分辨率一致的参考应变/初态
  构造，不能据当前B归因长刚段支撑。中文图已查看，用户视觉未确认；根因、
  物理验收及正式权限关闭。证据：
  `simulation_output/finite_tip_E2fb_local_refinement_preflight_v2/report.json`
  （sha256 b201eda4526337bfebb334b6d8a8aef54ede7dc86a5cc269099bd758bb564866）、
  `xml_audit.json`、`visual_and_inertia_supplement.json`及`E2fb_局部二分零步预检_中文.png`。

- 2026-09-06：完成E2fc同材料应变初态构造和用户2s运行器，agent动态积分0。
  原224–236 cell转角θ以两个θ/2分配，参考构型保持，材料κ一致；λ五点仅作
  运动学核验，不是平衡装载。preflight_v2模型18/18通过，应变差≤9.06e-14rad/m，
  初始力矩模长均7.836–8.629µNm，不再翻倍。预测并实际复现头平移
  (-.008571,-.062674,+.074590)mm，模.097802mm，原方向/head内部weld保持。
  离散应变二次型两组均38.180404µJ，不称严格势能。A初始墙接触9条/B6条，
  A228/229→1456和232→1489三条在B消失；最大穿透.011962→.097482mm。
  新最大穿透source255→1660初始法向力0，不能说头支撑增强。101来源hash保持，
  三张中文图已查看，用户视觉待确认。A复用E2et8000步，B待用户运行；自身t0
  与共同A参考两种位移并列，每400步checkpoint，可中断续算/仅重试渲染。
  硬失败保存独立证据且禁止自动续算。诊断含几何积分/接触划分/新增DOF联合
  差异，长刚段根因、物理改善和正式权限均未确认。证据：
  `simulation_output/finite_tip_E2fc_strain_initialization_preflight_v2/report.json`
  （sha256 7610ba40029b0365e1eac490781940df4c8353ee7e00a6734f965f01dd264a95）、
  `visual_summary.json`及`tools/run_finite_tip_E2fc_matched_strain_pair.py --resume`。

- 2026-09-06：用户完成E2fc B8000步/2s；数值完成后继承渲染器因相对mesh路径
  `utils/interface/model/0422.stl`在项目外工作目录失败。新增独立恢复入口
  `tools/recover_finite_tip_E2fc_render.py`，原绑定源码/合同不变。实际从
  `C:\Users\Silence`恢复三张中文图，8积分API调用0，20trace+21checkpoint的
  41文件及全部保护源hash前后不变，report数值字段保持，progress仅改completed。
  A/B自身t0最终上移+.783239/+.659227mm，响应差-.124012mm；B计入初始
  Z+.074590mm后相对共同A0为+.733817mm。最后.1s斜率+.616281/+.444086mm/s，
  均仍向上；动态最大穿透.048759/.096804mm，B初态.097482mm另列。
  数值完成不代表上翘已修复，初始接触变化使长刚段根因仍未单独确认。
  三图已agent查看，用户视觉待确认，未启动新动力学或训练。证据：
  `simulation_output/finite_tip_E2fc_matched_strain_pair_v1/report.json`及
  `render_recovery/20260906T054036466198Z/recovery.json`（passed=true）；旧报错
  与恢复前report/progress保留。物理验收、初始化和正式权限仍关闭。
- 同日独立纯JSON/NPZ数值复核通过：B20/20 trace及21/21 checkpoint完整且hash
  一致，A按E2ep2000步+E2et6000步重建，37项A绑定源hash一致。所有汇总重算
  与报告吻合，未import/call MuJoCo。证据：
  `simulation_output/finite_tip_E2fc_render_recovery_audit_v1/report.json`。
- 2026-09-06：用户要求在当前末态追加1–2次feed观察是否下降。实现E2fd从
  E2fc B8000步完整末态恢复原局部X slide/affine PD；仅新增q/v=0，旧形状、
  速度、warmstart、细分材料、接触和7-weld保持。实际完整迁移23/23，误差0；
  adapter4/4、独立纯Python调度/断点对抗29/29，准备及渲染禁止8积分API。
  同模型同起点hold/feed2各1.6s，feed2在0–.2s/.4–.6s分别参考推进1.931832mm，
  第二次后保持1s。现有有限链短程推进，不新增入口材料，显式记录新增跨度；
  不是现实feed事件距离定义或完整连续后端。当前未执行新动力学，动态由用户
  运行；下降不作为PASS门槛，也不单独排除墙体支撑根因。固定初态裁切避免
  跟随tip掩盖运动；当前入口preflight_v4，早期裁切/像素门槛报告及源码保留。
  证据：`simulation_output/finite_tip_E2fd_feed_reactivation_preflight_v4/report.json`、
  `visual_summary.json`、`simulation_output/finite_tip_E2fd_runner_tests_v1/report.json`
  及`tools/run_finite_tip_E2fd_two_feed_continuation.py --resume`。物理/视觉待确认。

- 2026-09-06：用户完成E2fd hold/feed2各6400步/1.6s。最终出图误用E2et
  固定根与统一.04阻尼检查，当前feed slide和细分阻尼使该旧检查失败。
  新增`tools/recover_finite_tip_E2fd_render.py`，只回读已验证状态，保留原绑定
  源码/合同。实际恢复34个状态、4张中文PNG与17帧同步MP4；17/17帧可解码，
  66数值文件及226保护源hash不变，progress/完整末态/模型数组不变，8积分API
  调用0。证据：`simulation_output/finite_tip_E2fd_two_feed_continuation_v1/`
  下`render_recovery/20260906T072708695535Z/recovery.json`（passed=true）。
  feed2相对本次起点在第一次结束-.393502mm、第二次结束-.197643mm，保持1s后
  +.911239mm；同时hold +1.273898mm，最终差-.362659mm。实际总推进3.863631mm。
  feed2最低-.478892mm；最后.1s hold/feed2斜率+1.382234/+1.066917mm/s，仍上升。
  说明短推进暂时压低尖端但未消除后续上翘，不能据此排除/确认特定管壁根因。
  没有新增入口材料或新增动力学；曲线及同步末帧已agent查看，用户视觉与物理
  验收仍待确认。当前图/视频位于`renders/20260906T072734898031Z/`。

- 同日E2fd独立复核：恢复入口静态/AST41/41；纯JSON/NPZ审计确认两组6400步
  及全部chunk/checkpoint hash、同初态、汇总/事件和逐接触near-head wall Fz
  一致，均未import/call MuJoCo。报告为
  `simulation_output/finite_tip_E2fd_render_recovery_tests_v1/ast_audit.json`和
  `simulation_output/finite_tip_E2fd_render_recovery_audit_v1/report.json`。

- 2026-09-06：用户要求20mm续推前优化计算及物理送丝速率。E2fe测得旧400步
  72.274s，mj_step48.368s+步后forward19.594s占约94%；Python缓存/复用阶段
  仅1.068x。同一E2fd末态400步启停窗口原implicit/Newton/AUTO 71.2219s，
  implicitfast/Newton/dense 25.0596s（2.8421x）；10μm位置等预定误差预算全部
  通过，最大tip/全末端误差6.36952e-9m。dt/原迭代容差/材料/碰撞等保持，
  数值方法变化非逐位等价。CG和PGS候选未过预算，保留失败结果而不采用。
  证据：`simulation_output/finite_tip_E2fe_performance_gate_v1/report.json`、
  `finite_tip_E2fe_numerical_screen_v1/20260906T080748719543Z/implicitfast_newton_dense/`。
- 同日E2ff绑定该配置，继承E2fd feed2 step6400完整状态，20mm/s、50ms启停、
  额外20mm推进1.05s+保持.5s共6200步。已执行800步/.2s可续接前缀，打印
  14.39steps/s（约55.6s积分记录），额外近端3.4975mm、tip净位移1.479529mm、
  累计路程7.626875mm、ΔZ+1.126424mm、最大跟踪误差2.524403μm、最大穿透
  .200367mm，无新增warning。存在反复摆动，未确认20mm/s物理合理。现有有限
  链无入口补料，完整20mm/100mm未执行，正式数据权限不变；下一次从801步继续。
  证据：`simulation_output/finite_tip_E2ff_20mm_continuation_v1/feed20/progress.json`、
  `simulation_output/finite_tip_E2ff_20mm_continuation_preflight_v2/report.json`。
- 同日E2ff800步已完成实际回读/渲染路径核验：重复前缀CLI不执行第801步，
  205保护文件hash、完整末态和模型数组不变，8积分API调用均0；5/5视频帧
  全部解码。曲线和固定局部末帧已agent查看，用户视觉待确认。报告：
  `simulation_output/finite_tip_E2ff_20mm_continuation_v1/prefix_verification/20260906T082036476831Z/report.json`；
  代表图与短视频：`renders/20260906T082125575765Z/`。冻结runner纯检查14/14，
  独立审查性能来源9/9、v1源码快照3/3通过；无新增积分。
- 同日E2ff独立纯JSON/NPZ前缀审计通过，800行/4chunk/5checkpoint和完整来源
  hash、E2fd末态精确继承、调度及运动/力/接触恒等式全通过；190合同来源和
  9性能来源闭包一致。证据：`simulation_output/finite_tip_E2ff_prefix_audit_v1/report.json`。
  审计未导入或调用MuJoCo，完整20mm及物理权限仍关闭。

- 2026-09-06：用户完成E2ff完整6200步，并明确先优化单条速度，后续再研究多进程；
  当前仍不能生成数据。额外近端19.999890mm，tip净位移14.841551mm、累计路程
  103.201061mm、末态ΔZ+4.316280mm、最大穿透.470580mm。最后.5s保持段净运动
  .292065mm但累计路程16.949273mm、Z方向变化88次；最后.1s斜率-.497750mm/s。
  振荡未解决，不视为平衡或物理验收；仍无入口补料。独立JSON/NPZ/视频审计
  6200行、31chunk、32checkpoint、5artifact及32/32视频帧全通过，未运行MuJoCo。
  报告：`simulation_output/finite_tip_E2ff_completed_audit_v1/report.json`；实际图视频：
  `simulation_output/finite_tip_E2ff_20mm_continuation_v1/renders/20260906T083705843710Z/`。
- 同日E2fg phase A在原E2ff起步/减速各400步上串行比较，复用位置速度缓存后
  25.948962/27.342878s降至22.698914/23.976959s（1.14318/1.14038倍）。逐步完整
  状态、几何和力诊断完全一致。容差1e-6无额外收益，1e-4起步超预算，均未采用。
  279保护源和模型数组保持，未新增完整推进或数据生成。结果位于
  `simulation_output/finite_tip_E2fg_single_speed_v1/20260906T084815788048Z/report.json`。

- 同日E2fg phase B在同状态/动作/0.25ms步长/原1e-8容差下，每步只执行一次
  动力学求解；将真实积分前force packet与步末geometry明确分开。起步/减速/
  保持三个400步窗口26.403673/28.100117/23.544519s降至14.333861/14.854579/
  13.714777s，合并1.81917x（耗时降低45.03%）。全部逐步完整状态、端点、速度、
  接触几何和对齐后的步初力/qacc packet精确相等；step4000 fresh MjData磁盘
  恢复后至4200也精确相等。286保护源及模型数组/选项未改。计时不含准备、
  落盘、渲染；仅短窗口，完整6200重放和物理验收尚未完成，数据权限不变。
  证据：`simulation_output/finite_tip_E2fg_single_solve_v2/20260906T090507305976Z/report.json`；
  中文对照图同目录`E2fg_单条速度优化_中文.png`，已root目视核验。

- 同日E2fg phase B独立纯JSON/NPZ审计通过，三窗口全部数组/步初packet、力和
  冲量预算、原400/4000/4200/6200 checkpoint、续跑后缀、286保护源和30输入
  artifact均确认。独立重算合并速度比78.0483091/42.9032170=1.819171488。
  未导入或调用MuJoCo。报告：
  `simulation_output/finite_tip_E2fg_phaseB_independent_audit_v1/report.json`。

- 同日新E2fg单条重放入口完成零步preflight及实际400步prefix，200/400保存点
  均打印27.47steps/s（含chunk/checkpoint落盘，不含模型准备/原轨迹回读和末尾
  source检查）。400个步初力packet、尖端和进给均与phaseB逐值相同；0/200/400
  完整checkpoint与原E2ff相同。重复真实prefix CLI时8积分API均0，297保护文件
  和模型数组不变，未运行第401步。当前新入口400/6200，剩余5800步完整重放
  待用户运行，旧20mm结果保留，物理/视觉/正式数据权限不变。报告：
  `simulation_output/finite_tip_E2fg_single_solve_replay_v1/prefix_audit.json`；零步证据：
  `zero_step_verification/20260906T092337782525Z/report.json`。入口纯AST/源码16/16通过。

- 2026-09-06：用户完成新E2fg步进器完整20mm重放6200步。root通过真实prepare/
  load_progress路径回读，6200行/31chunk/32完整checkpoint与原E2ff对应字段精确
  一致，8积分API调用0，模型数组及356保护文件hash不变。实际近端19.999890mm、
  tip净位移14.841551mm、累计路程103.201061mm、最大穿透.470580mm；保持段
  净运动.292065mm但累计16.949273mm，原振荡保持。新步初力不与旧步末力同义比较。
  同段checkpoint1000→6200的文件保存跨度约403.951188→205.050138s，约1.97001倍；
  属时间戳估算，含诊断/保存、不含准备/渲染，非严格全程计时。中文重合曲线与
  耗时图已root查看。报告和图片：
  `simulation_output/finite_tip_E2fg_single_solve_replay_v1/completed_verification/20260906T104944990898Z/`。
  当前模型下单条计算优化验证完成，后续转回物理振荡/入口材料问题；视觉/物理/
  正式数据权限不变，未新增推进、物理参数修改或多进程。

- 同日E2fg完整重放独立纯数据审计PASS：6200行、31chunk、32完整状态（每个4588
  值）、286来源hash、1200个phaseB步初force packet与继承前400产物全部一致；
  360输入文件未改。未导入MuJoCo或项目runtime，未积分。报告：
  `simulation_output/finite_tip_E2fg_completed_replay_audit_v1/report.json`。


- 2026-09-06：用户要求从完整E2fg末态再推进5cm，之后回到振荡问题。新E2fh
  runner/contract保持原模型、材料、接触、dt和single-solve；20mm/s、50ms启停，
  2.55s推进加0.5s保持共12200步，不补入口材料。起点完整状态等于E2fg6200，
  实际残余进给速度-4.3571903e-05m/s保留，近端参考连续。
  root零步preflight及实际400步prefix通过：参考1.5mm、实际1.497343mm，尖端净
  位移.838830mm、路程4.501636mm、最大穿透.251069mm、最大tracking误差.003331mm。
  两个200步chunk含计算与trace/checkpoint写入耗时6.828/7.547s，29.29/26.50steps/s。
  真实重复prefix CLI和保存态渲染在8积分API阻断下调用均0；初态/保存400完整状态
  精确，模型数组/options及372保护文件未改。全景/跟随局部短视频各3帧均完整解码，
  中文图已root目视检查。39/39纯测试及360源hash/3gate独立静态审计通过。
  中间未冻结runner变动被preflight checksum阻挡且未积分，已恢复ed2a冻结hash重验。
  当前400/12200、next401；剩余11800步待用户运行，完整5cm/物理/用户视觉验收未完成。
  未调振荡/未多进程/未训练。prefix证据：
  `simulation_output/finite_tip_E2fh_50mm_continuation_v1/prefix_verification/20260906T111947722494Z/report.json`；
  图视频：同run的`renders/20260906T111921565575Z/`；独审：
  `simulation_output/finite_tip_E2fh_independent_audit_v1/report.json`。


- 2026-09-06：用户E2fh额外50mm在6214步因tracking_within_limit=false停止，
  实际滞后.100094533mm超过.1mm；其它controller/solver/numeric门通过。
  committed6200（31chunks/32cps，实际额外30.407272mm）；6201–6213未提交行及
  rejected6214完整状态/行独立保存。失败时近端额外30.470048mm、tip净位移
  46.694285mm、path212.510609mm、穿透.584851mm；全段最大穿透.605606mm。
  独立纯数据核验360source/hash及所有记录闭合，唯一门为tracking。
  近端B43起的首批source45–56与extension墙片0305/0273/0274/0306负载增长；
  不得称为真实硬件插入力/堵转。固定feed轴与不补材料边界是待验证机制。
  root零积分回读和中文图/视频生成完成，8API均0、模型array/options和440源
  未变；视频通过6200的32/32帧均完整解码，未声称完整50mm。证据：
  `simulation_output/finite_tip_E2fh_tracking_failure_diagnosis_v1/20260906T121021719289Z/`。
  root随后原native/full-forward路径仅重复6201–6214共14步：全部步初force
  packet/control/tip/feed/endcontacts逐值相同，最终fullstate精确等于失败6214，
  同一步重现tracking失败。mj_step14/其它API0、431源和模型/options未变；
  native feed constraint=-1.414082914N、actuator=+1.401379361N。报告：
  `simulation_output/finite_tip_E2fh_failure14_replay_v1/report.json`（37/37纯测试）。
  排除该14步失败窗口的single缓存路径差异；未放宽门/改物理/继续50mm，
  failed_closed保持，剩余约19.53mm尚未完成，后续先查近端接触几何。


- 2026-09-06：用户同意并要求继续入口接触几何排查。提取0/4000/5000/6000/
  6200/6214六个保存状态，18个B43–60活动碰撞胶囊与全部5440墙胶囊做有限线段
  距离核验；55对已记录接触的解析距离最大差1.28e-17m。最终B43/B44最小墙间隙
  .991633/.121408mm，均未接触；接触从4000步B59–60/墙0337迁移到5000步
  B52–56/墙0305，再到6214步B45–52/墙0273/0274/0305/0306（17对）。
  source45–56占最后步初反向轴向逐接触力总和98.508922%；净轴向墙力-1.414083N。
  步末几何和步初力保持分离，6214只作几何图、不伪配当前力。
  root另做静态B43原轴平移扫掠：全部墙最小间隙保守推进+二分首次相切为
  +1.735788129mm/墙0274，计划尚余19.529952mm。此为假设静态几何上限，
  未运行后续动力学，未将根部直接撞墙认作已发生6214失败的唯一原因。
  保持根部直线行程则剩余路径与当前碰撞墙存在冲突；墙代理和可见内腔错位尚未证实。
  extractor431、root渲染439保护源及模型array/options未改，8积分API均0；
  三时刻固定v4朝向对比/量化图/静态扫掠图已root目视检查，待用户视觉确认。
  报告：`simulation_output/finite_tip_E2fh_proximal_geometry_v1/report.json`、
  `simulation_output/finite_tip_E2fh_proximal_load_ledger_v1/report.json`、
  `simulation_output/finite_tip_E2fh_proximal_contact_mechanism_v1/20260906T125034505524Z/report.json`、
  `simulation_output/finite_tip_E2fh_fixed_root_sweep_v1/20260906T125315914801Z/report.json`。
  adapter完成提取；audit已写账本但最终回报、design后续受额度限制，root本地补完。
  failed_closed仍6200/拒绝6214；未继续50mm、调门限/物理/振荡，未训练。


- 2026-09-06：用户批准E2fi固定入口+材料补入的短程对照。A只读E2fh5001–6200；
  B从原5000完整checkpoint新初始化，增加普通source8–42共35节/7.806499259e-5kg，
  anchor为旧轴与入口平面交点（距登记site0.274172微米），旧实际q48.330329mm
  重基为0.650662mm；q/reference/ctrl同移，qvel、warmstart、旧姿态/速度/参考应变、
  7个头部weld与旧材料/接触参数保留。四bank各22/22；独立初态24个既有contact，
  新材料wall/self/other各0。静态8API均0，模型运行态4opt/12gravcomp差异已明确复现。
  v1静态prime调用mj_step1被guard拦截，未积分；v2将prime移到授权runtime并检查
  完整状态不变。v2在250首次补入后，于251因pre/post tip相差2.78e-17m却做exact比较
  停止，唯一失败为start_tip_continuity，其它动态门全真；durable200/失败251全部保留。
  v3仅修正下一步tip参照为当前换模后端点，导入0/200CP与1–200行，201–251物理字段
  exact回放且251完整状态exact；两项tip派生路程明确重算，差1.24e-17/1.39e-17m。
  此后完成1200独特步/0.3s，birth250/638/1024、source8→7→6→5，三次各22/22。
  实际累计积分v2的251+v3的1000=1251步（含51回放），原动态门限/物理参数未改。
  B/A实际进给6.029245/5.940464mm；tip三维净位移4.569354/9.571533mm；最大跟踪误差
  .071923/.092871mm；最大穿透.662734/.576283mm；平均净轴向墙力-.043123/-.665197N。
  平均反向反作用减小，但初始化瞬态/最大穿透及tip位移变化不能隐去，也不能当成沿管
  进给距离、效率或单因素原因。B同时改变边界/材料库存/质量，入口到首材料起点仍是
  一节以内的空缺近似；并非真实滑动支承/送料硬件验证。500受保护源hash绑定，旧E2fh
  failed6214/durable6200和50mm未完成不变；未调门/振荡/训练/硬件/算法/清理/提交推送。
  完整报告`simulation_output/finite_tip_E2fi_fixed_inlet_release_v3/numeric_report.json`；
  固定v4朝向A/B图、量化曲线及0.25x视频位于该目录`renders/20260906T140326557428Z/`，
  7/7帧全解码、完整状态和临时显示修改恢复；入口坐标/身份图位于
  `simulation_output/finite_tip_E2fi_release_timing_v1/`，15输入前后hash一致。
  root已看初态/末态和曲线，用户视觉确认仍待定，formal/physical_validation仍false。
  本轮代理均high/forknone：adapter与audit用gpt-5.6-sol，design用gpt-6-astra；
  适配/初态碰撞、代码及数据回读、物理解释和渲染分别完成，root负责集成与实际短程运行。

  同次独立纯JSON/NPZ/XML读回`simulation_output/finite_tip_E2fi_smoke_readback_v2/report.json`
  15/15通过；B峰值反向净轴力1.354082N和最大tracking.071923mm在local1，最大穿透
  .662734mm在7；A三者最大值在global6200（1.283476N/.092871mm/.576283mm）。
  B最后200步mean净轴力-.056583N，max反向.109815N，mean/max绝对tracking
  .004175/.006073mm，maxpen.416219mm；全程峰值由初始化瞬态主导。
  251完整向量exact由冻结runner实际fail-closed门验证；v3没有单独保存251CP，
  因而纯读回只能确认该门的源码/记录与其余字段，不主张独立第二次完整向量比较。

- 2026-09-06：用户认为E2fi B更合理，但tip仍上翘；只记录入口/补料局部认可，
  tip及整体物理验收仍未通过。E2fj只读提取A/B每200步保存状态，新增固定v4朝向
  头颈三时刻局部图和量化图；524保护源前后hash一致，全部fullstate/model恢复，
  8积分API均0。B头轴仰角35.650372→34.029910°，head/247夹角.937012→.894515°。
  tip净Z+.972572mm由base248平移+1.331282、整头转动−.352228、头内残差−.006481mm
  组成（精确运动学分解，不是受力因果分解）。头部保持旧的陡倾姿态，新增高度主要
  随基座平移；末态head对登记墙环中心线参考夹角17.787428°，参考并非内腔真值。
  独立力账本7/7、2400A/B行：B source255有767步contact，净墙面Z力向上0步，
  全墙净Z力947/1200步向下；无“末端被墙直接向上顶”的记录证据，机械主因未定。
  实际头为248–255七weld，243–247为未weld颈部；头内现有非零畸变不能当完美刚体。
  当前末端墙间隙−.110972mm，tip问题未宣称修复。root已看局部图/曲线。
  报告`simulation_output/finite_tip_E2fj_tip_geometry_v1/20260906T144026211902Z/report.json`、
  `simulation_output/finite_tip_E2fj_tip_lift_force_v1/report.json`；物理审查位于
  `simulation_output/finite_tip_E2fj_tip_mechanism_v1/`。未改物理/参考形状/门限，未续50mm。
  本轮audit=gpt-5.6-sol、design=gpt-6-astra，均high/forknone；root实现零积分提取渲染。
  随后独立JSON/numpy几何回读15/15通过，524保护源+6产物hash无变化/不匹配，
  14行端点重算最大误差1.7347e-18m；无MuJoCo或项目模块导入/积分。
  报告`simulation_output/finite_tip_E2fj_geometry_readback_v1/report.json`。
  下一项建议B400同source7完整初态的200步feed-vs-hold对照，本轮未执行。

- 2026-09-06：用户批准E2fk同B400完整初态的200步/0.05s feed-vs-hold-reference对照。
  A只读已有E2fi B401–600，B保持row400参考.000752164608815m、参考速度0；原实际q
  .000748104972796m、qvel.020256081630027m/s、ctrl.000752080916122m均保留至首步。
  同source7/262材料胶囊、独立MjData；预检fullstate/geometry/setup/prime精确，8API0。
  B200步完成、无birth，implicit200/step1 201/其他6API0；4个50行chunk、5个完整CP保存。
  每步API与B身份门在commit前核验；物理/控制/连续性门均通过，未清速度/调参数/门限。
  A/B实际推进1.000019/.003847mm；tip净Z+.252499/−.106352mm，base248净Z
  +.274275/−.074481mm；末头轴34.854296/34.808703°，初始34.952746°。
  max tracking .004484/.003260mm，max penetration .361720/.405162mm，mean净轴墙力
  −.032651/−.005901N。A−B末tip差+.358851mm，其中基座平移差+.348756mm，仅运动学。
  B先升后降且振荡：最高+.523246mm@52、最低−.247521mm@115；A−B有82/200步为负，
  最后55步为正、最后50步平均+.280276mm。不能从50ms末态称稳态、单调下垂或解决上翘。
  独立static19/19、dynamic18/18，542合同源未变；PD力最大误差2.05e-15N，5CP完整状态
  与初态/行核对，A未重积分，source255仍无正向净墙面Z力。数字报告位于
  `simulation_output/finite_tip_E2fk_feed_hold_v1/numeric_report.json`，数值statusnumeric_complete。
  原进程随后因v1渲染角度定义门退出1：body248原点与capsule近端固定局部偏移约7.09e-11m，
  角度差3.02e-9°超过1e-9°显示门。原代码/合同/数字/失败证据均保留；新renderer v2统一
  body248/body247姿态定义，保留真实胶囊与原角度门，只读恢复564保护输入生成新图。
  四frame-arm局部偏移变化≤1.32e-17m，tip端点差0，8API0，所有状态/model/显示均恢复。
  root已看末态局部对照和六面板曲线，视觉待用户；图/报告：
  `simulation_output/finite_tip_E2fk_saved_render_v2/20260906T153653983767Z/`。
  E2fh仍failed6214/durable6200，50mm未完成；本轮未延长hold/feed、训练、硬件、算法、清理或提交推送。
  audit代理gpt-5.6-sol/high/forknone完成独立审查；design代理gpt-6-astra/high/forknone完成
  渲染/定义修正/物理解释；root完成冻结、实际200步和零积分重渲染。

- 2026-09-07：用户批准E2fl从E2fk B200完整状态继续保持0.45s=1800步，总hold0.5s。
  source7/262材料capsule、实际q/v、warmstart、ctrl/ref保持，无birth/调参/新feed参考。
  568输入冻结；零步恢复/四组注册几何兼容均通过、8API0；static23/23、8个损坏夹具拒绝。
  实际进程退出0，implicit1800/step1 1801/其余0，9chunks/10CP完整；dynamic17/17。
  全hold tip Z=-0.689984mm（新增段-0.583632），base248 Z=-0.563581mm；head34.952746
  →34.808703(.05s)→34.382936度(.5s)，head-neck末0.910194度。新窗口max tracking
  .000485mm、max penetration .089188mm。末50ms tip Z峰峰值.169145mm、3D差分速度
  RMS25.201878mm/s、垂直RMS19.773033mm/s、OLS漂移-1.977793mm/s；早期大振荡减小但
  后期仍有细振荡+下移，不能称平衡或上翘已修复。延长段232–255步首墙接触记录均0。
  四保存时刻最近注册主干参考段均58，头轴参考夹角16.857699→16.526196度；该折线参照
  不是唯一真实血管轴，全头/全链安全未由单点截面余量证明。readonly render进程退出0，
  8API0且数值/CP/trace/source hash、fullstate/model/display恢复；root已看局部sheet/
  末态全链/曲线，目测裁决仍待用户。当前产物：
  `simulation_output/finite_tip_E2fl_hold_extension_v1/`及其
  `renders/20260906T161738014137Z/`；独立审计在`finite_tip_E2fl_audit_v1/`。
  E2fi仍1200 complete，E2fh仍failed6214/durable6200；未再延时、训练、硬件、算法、清理或提交推送。
  子代理audit=gpt-5.6-sol/high/fork none、design=gpt-6-astra/high/fork none；root集成/运行/视图验证。


- 2026-09-07：用户批准E2fm利用现有E2fk/E2fl保持状态审计能量与约束传力。
  14完整CP（15原文件，重复边界逐位相同）与2000步密集记录均只读；689保护输入、
  262材料body含细分半段、98个截面独立Newton–Euler/RNE闭合通过，8积分API全0。
  主审计进程exit0；独立static11/11、dynamic16/16，12个可比CP原步首qacc/actuator
  echo误差0；原18产物/hash均核验，重复入口exit1拒绝覆盖。中文主图root和审计代理
  均已看；--plot-only另实测exit0/8API0，原numeric不变，新增
  `finite_tip_E2fm_saved_energy_transfer_v1/renders/20260906T183659554593Z/`。
  全导丝K541.826→22.448nJ、磁头K99.448→2.378nJ；导丝/磁头ΔUg=-628.922/
  -109.698nJ。原生K+U不含cable插件完整弹性与软约束储能，不能做总机械能闭合。
  全模型另含4个各1g独立free marker，collision/actuator/eq/parent均与导丝隔离；
  原始qvel仅vz从-63.765到-68.67m/s、qacc仅重力-9.81，四体ΔK约+1.299187J，
  故全系统KE增长不能解释为导丝注能。未删除或改动这些marker。
  14CP原生阻尼功率均负；248插件抬尖力矩约-2.365→-2.333μNm，始终朝压低方向。
  后期近头各截面distal净接触载荷0、传入载荷仍变，净0不单独证明每点无接触。
  密集17JSON绑定/前后hash不变；末0.4s的2.5Hz格上tip/base/feed在82.5Hz的幅值
  59.795/56.305/.0641μm，tip/base OLS Z趋势-1.6081/-1.3745mm/s；频率格不是
  模态或根因。实际feed净动4.001054μm，ΣFstart·dq=+50.005888nJ，另一左端近似
  ΣFstart·vstart·dt=-134.128289nJ，保留分歧、不称准确功或总能量。
  主产物在`finite_tip_E2fm_saved_energy_transfer_v1/`，密集分析在
  `finite_tip_E2fm_dense_motion_v1/`，独立审计在`finite_tip_E2fm_audit_v1/`。
  dense代理QA移动两份草稿至系统Temp超出分配目录，精确清理被自动审批拒绝，文件仍
  保留且停止重试；完整路径见其scope_note.md。旧源数据未改，未涉及算法、硬件或训练。

  同日E2fm补充只读hold400/.10s、hold1800/.45s的逐接触功率归属，706源冻结、
  8API0；static12/12、dynamic16/16通过。28条contact的点速/角速、局部与世界力、
  normal/slide/spin/roll功率及按材料/壁geom分组独立复算闭合，误差≤2.71e-20W；
  两CP qpos/qvel/qacc/qcontact与主NPZ逐元素相同，model/options/fullstate未改。
  .10s normal+129.055204μW/slide-14.650307→总+114.404898，主要217/221材料接触
  正在负gap下法向分离；.45s normal-49.473534/slide-6.202764→总-55.676298μW，
  主要211/208材料向壁面接近。几何/active/非零wrench数分别11/11/9、17/17/9，
  都是cable-wall、0self。两帧224–255无任何接触记录，载荷经近端传入磁头；不证明
  全区间无接触、周期净注能或82.5Hz唯一原因。已提议0.05s步长减半诊断，尚未授权/执行。
  新报告`finite_tip_E2fm_contact_power_v1/numeric_report.json`，物理解释在
  `finite_tip_E2fm_energy_design_v1/physical_result.md`；root完成最终中文接触图v3，
  旧v1/v2绘图QA草稿留在项目内且标记未接受。视觉待用户，所有正式用途门维持关闭。
  三代理：audit gpt-5.6-sol/high/forknone，design gpt-6-astra/high/forknone，
  dense_motion gpt-5.6-sol/high/forknone；本轮全部任务完成。root完成主审计与最终集成。


- 2026-09-07：用户批准E2fn从hold400/.10s同一完整状态进行步长减半诊断。
  A只读原dt=.00025s、hold401–600的200步；B独立复制模型，仅dt=.000125s，
  唯一400步/.05s运行exit0。初始完整向量与首力包相同；solref原值及有效截断一致，
  711输入与原模型不变，262材料capsule/7头weld、实际q/v/warmstart/ref保留。
  implicit400/step1 401/其余0；8trace、8逐步fullstate batch、9CP完整，400行24门
  全通过。独立static16/16、dynamic20/20（只读JSON/NPZ/hash，0新积分）。
  含共同起点的201点比较：tip Z峰峰185.852→130.444μm，去趋势RMS44.751→34.268μm；
  base248峰峰172.847→120.224μm，去趋势RMS42.567→32.216μm。tip B−A差
  RMS64.059/max130.010/end-33.433μm；前10ms差RMS9.573μm、后40ms71.449μm。
  两支峰相位逐渐分离、都仍振荡；仅证明单轨迹步长敏感性，不证明收敛、精确模态
  或真实对应。实际feed窗口净变A+.035675/B-.111649μm；最大穿透64.445→50.406μm。
  B能量/功率为400个真实步首，hold.10–.149875s，不能冒充.15s终态能量。
  全导丝KE首/末采样77.940/46.549nJ、磁头10.036/10.184nJ，起伏持续。
  4300接触功率独立复算闭合≤5.42e-20W；全部B步首材料224–255无接触。
  总sliding瞬时功率均值-4.794675μW，14/400样本为正，峰+.709938μW，主要来自
  上游材料136/135及177；不能由瞬时正功率声称周期净造能，A也缺逐步点速度对照。
  产物在`finite_tip_E2fn_half_timestep_v1/`、`finite_tip_E2fn_comparison_v1/`、
  `finite_tip_E2fn_design_v1/`和`finite_tip_E2fn_audit_v1/`。root查看中文比较图v2及
  能量图；v1重叠草稿留项目内、QA false，v2不改原数值/hash；用户视觉接受待确认。
  formal/policy/physical/reset/initialization各门仍关闭。未续时、训练、碰硬件或算法，
  未清理或提交推送；dt=.0000625s/800步第三分支只是建议，未执行。
  三复用代理audit=gpt-5.6-sol/high、design=gpt-6-astra/high、dense_motion=gpt-5.6-sol/high，
  均fork none。root完成runner、400步执行、复算、图像检查与文档集成。

- 2026-09-07：[reporting][decision] 用户要求先把当前审计落到组会文档，再进行下一步。
  已新增`docs/group-meeting-audit-summary-20260907.md`，汇总E2fk–E2fn的保持对照、
  延长保持、材料能量口径、截面传力、逐接触功率和步长敏感性，附两张代表图及原始
  证据链接。本文保留“单轨迹步长敏感、尚未收敛/确定唯一根因/真实验证”的边界。
  文档写入和链接核对完成时第三步长尚未开始。用户本次已授权随后从同一hold400
  完整状态用dt=.0000625s运行800步/.05s；旧两支只读，不从半步长终态续跑。
  组会文档独立事实核对12/12、本地链接11/11、代表图3/3通过；采用3处非阻塞措辞
  精化，明确分支起点状态、接触参数不变及力账本闭合，未改变数值或实验结论。

- 2026-09-07：[E2fo][diagnostic] 组会审计快照写入并冻结后，完成用户本轮授权的第三
  步长唯一800步/.05s运行，进程exit0。C从原hold400/.10s完整状态复制，只改
  dt=.0000625s；A/B旧轨迹只读，初态完整向量及位姿/首力包一致，solref有效截断
  三支相同。747保护输入hash不变；implicit800/step1 801/其余0，800行执行门通过，
  16trace/16逐步pre-post完整状态batch/17CP。独立static15/15、dynamic24/24通过，
  动态复核仅JSON/NPZ/SHA，未导入模型、forward或积分；C输出复核前后不变。
  共同起点加200末态的201点上，tip Z峰峰A/B/C为185.852/130.444/107.345μm，
  去趋势RMS44.751/34.268/21.648μm；base去趋势RMS42.567/32.216/20.230μm。
  tip Z的B−A/C−B差RMS64.059/37.441μm，比0.584483；三维RMS比0.640136。
  前10ms的tip Z比0.985865、三维比1.267622，base为1.005285/1.183707，第一窗
  三维反增；后40ms tip Z比0.582377。tip B/C末态Z差-1.622μm，三维仍49.346μm，
  全窗Z最大差82.458μm。原生tip波峰间隔中位12.250/10.375/8.500ms，不支持
  精确模态、收敛阶或全程一致收敛。root原始trace18项关键指标复算与报告一致。
  B/C共同400步首hold.10–.149875s，材料链平均KE37.607253→22.345435nJ，
  头部5.308936→3.121530nJ；总正slide样本14/400→29/400，峰+.709938→+.777942μW。
  C原生800步首另列57/800，7471接触功率独立复算最大差4.07e-20W；总slide峰C329
  的主要正项来自材料135/136与壁0916，单接触峰C72为材料136/壁0916 +1.104880μW。
  B400/C800采样步首材料224–255无接触记录；不外推连续时刻或唯一运动原因。
  正瞬时功率不证明周期净造能；A无逐步完整能量、B/C无.15s终态能量，插件与软约束
  储能仍缺，正式用途/真实验证/初始化/reset各门保持关闭。物理解读10/10只读检查、
  31输入hash一致；v2仅精化一句“正耗散”的语义，v1、数值、图及原读回均保留。
  新增`docs/group-meeting-audit-update-E2fo-20260907.md`记录第三步长实测；原
  `docs/group-meeting-audit-summary-20260907.md`保留实验前快照。更新数据交接及
  commands历史命令/重入边界。主要产物在`finite_tip_E2fo_quarter_timestep_v1/`、
  `finite_tip_E2fo_comparison_v1/`、`finite_tip_E2fo_design_v1/`、`finite_tip_E2fo_audit_v1/`。
  root已查看中文比较图v1/v3及能量图，布局通过；v3只改组会文案，v1/v2及原数值保留。
  用户最终视觉接受待确认。未增加第四步长/续时、训练、算法/硬件改动、清理或提交推送。
  候选后续为既有B/C正slide峰值步首状态的零积分接触求解语义审计，尚未执行。
  三复用代理均fork none：audit=gpt-5.6-sol/high、dense_motion=gpt-5.6-sol/high、
  design=gpt-6-astra/high；root负责组会先行门、C执行、独立复算、图像QA和文档集成。
  组会实测补充最终独立事实核对14/14通过，无修正项；`meeting_addendum_review.json`
  绑定最终文档SHA256=6e6877bf671669049b5c7eb9a50e6e1ff5b9d13b7fdb4f004262fecf96fb3151，
  文档与14份直接证据hash一致，包含静态15/15、动态24/24和747保护输入的核验结果。

- 2026-09-07：[E2fp][performance][handoff] 用户要求将当前长程feed交由用户续跑，
  并整理此前速率优化到组会文档。v2已通过STOP_REQUESTED正常提交38311步完整
  尾批并退出，status=interrupted、failure=null；99次birth、source=-94，实际额外
  feed191.556970mm、实际tip距左支出口186.563472mm，尚未到达。max penetration
  1.227000mm、max tracking .008885mm；implicit38311、step1 38412、prime101。
  966保护源、581保存包、99份birth XML及迁移门只读复核通过，0新积分；PID66640
  已退出且run.lock释放。末CP SHA=b7f277371002546746010146919e14a7d92ea24e71f0e05b6c1b5d21c4ef5bbb。
  暂停记录为`simulation_output/finite_tip_E2fp_user_handoff_v1/report.json`。
  新增`tools/resume_finite_tip_E2fp_left_outlet.ps1`，检查进程/锁/源码绑定/末CP，
  仅清除暂停标记后调用v2 --run --resume，从38312继续；独立日志、终态只读渲染，
  failed_closed不续积分。真实-CheckOnly退出0；独立PowerShell Parser0错误及源码
  审查通过。未实际重新启动用户长作业。
  新增`docs/group-meeting-speed-optimization-20260907.md`，记录E2fe短窗71.2219
  →25.0596s、2.8421x，以及E2fg三个400步窗合计78.048309→42.903217s、1.81917x；
  两轮基线/计时范围不同，不相乘称整体倍数。区分20mm/s物理参考与计算吞吐，注明
  单次求解步首力/末态几何语义、数值与渲染分开、精简记录、完整续跑、按需编译和
  取消MagicMock实参历史的对象保留修复。内存采样约19.8GB→v2多次4.9–6.5GB只作
  非配对工程观察，没有新的速度/内存倍数。当前仍有模型增长与接触求解成本。
  root查看了原E2fg中文速度图并在文档注明图的历史快照时点；旧组会审计文档保留，
  更新当前data handoff和commands。无训练、算法/硬件改动、清理或提交推送。
  本次两个复用代理均fork none：audit=/root/e2fc_recovery_audit，gpt-5.6-sol/high，
  分叉补审随用户交接方向中止后完成续跑脚本审查；dense=/root/e2fm_dense_motion，
  gpt-5.6-sol/high，完成速度源证据及会议稿只读复核。root负责正常暂停、交接快照、
  脚本、文档与只读验证。用户最终视觉确认及正式用途边界继续保留。

- 2026-09-12：[algorithm/VLA][LIBERO B4b] 用户运行的固定 task-finetuned PI0.5
  完成 Spatial 10任务×10episode，独立复算为96/100；任务0..9成功数
  `[10,10,10,10,9,8,10,10,10,9]`。4次失败均达到280步；记录评测耗时477.087539s。
  本机/远端CPU审查39项工件、实际初始化调度、checkpoint与视频检查通过，10项负向
  trace测试通过，100个视频全部可解码；未加载模型、仿真或执行优化步骤。
  保留观测结果但严格动作协议不通过：100个episode各存在至少一次环境输入动作越界，
  全局[-1.0331023,1.0188165]；当前源显示原生OSC有裁剪、夹爪使用符号处理，不能从
  episode极值认定超限维度/次数或解释失败原因。配置360未传到运行时，实际保存视频
  为256，不能据配置字段称真实渲染尺寸违规；policy双路张量未直接记录。
  四段失败视频取样与成功参考末帧已查看，`viewed_not_accepted`，用户接受待确认；
  保存视频不含最后动作后的终态。不等同官方发表协议、世界模型收益或导丝真实系统
  效果。证据：`simulation_output/pi05_libero_spatial_b4b_audit_local_v2/report.json`、
  `visual_review.json`；结果与命令写入算法轨文档，未改数据轨文档或自动重跑。

- 2026-09-12：[algorithm/VLA][LIBERO action boundary] 经用户批准，完成task0/
  seed1000/init0的20步动作边界诊断，保留原生280步horizon；耗时67.952s，未训练、
  未加噪评测、未修改模型/控制器或额外裁剪。12项CPU测试本地与4090通过，10项真实
  边界检查通过。机械臂6维均未超限，夹爪8/20步低于-1，最低-1.009158492；原生夹爪
  按符号控制且内部累积量饱和，轻微幅值超限未改变本窗口开方向。每步OSC1次、夹爪
  25次调用均记录；机械臂裁剪机制存在但本窗口未触发。两路原始/策略输入均256，
  模型内部224并含一个无效掩码空相机；实际state8。107份B4b原工件未改变。
  证据：`simulation_output/pi05_libero_action_boundary_v1/report.json`及原生调用日志。
  中文双视角/逐维图已查看，`viewed_not_accepted`；仅动作前取样，不是终态/成功率。
  本窗口不能回填历史100episode的越界维度或作失败归因；原96/100观测结果及严格
  原始动作协议限定仍保留。下一步需明确原生控制器与额外wrapper裁剪的协议区别，
  然后安排配对低对比度实验，未自动启动。

- 2026-09-12：[algorithm/VLA][LIBERO paired low contrast] 新建独立原生控制器对照
  协议与入口，保留旧B4b结果/协议，不训练、不修改PI0.5/控制器、不加动作wrapper裁剪。
  固定low_contrast第二档alpha0.45，双路图像变换；每组重跑clean，实际初始化payload、
  全部起始观测及首次推理RNG指纹必须一致。23项CPU测试在本机/4090均通过；真实
  task0双条件完整预检67.715s完成，两者均82步成功，仅一对，不作鲁棒性总体结论。
  107份旧B4b工件未变；主体10任务×2条件=20episode已准备，未由agent启动，交用户跑。
  主体必须与预检代码/协议/资源哈希一致，不复用旧96%或预检成绩、不据结果调强度。
  首帧双视角中文对照图和末次动作前主视角已查看，`viewed_not_accepted`；不是终态
  图像判定。证据：`simulation_output/pi05_libero_low_contrast_check_v1/report.json`，
  report SHA256=34043b8a6681732d4fbe788707a9d80d5508a04e12531be8c1805e70035a1047。
  未读取新真实数据，未改数据轨/SOFA轨文件，未开始world-model训练或宣称算法创新收益。

- 2026-09-12：[algorithm/VLA][LIBERO paired low contrast v2] 用户运行的新鲜环境
  v2 screen 完成20个episode/10组配对，固定alpha=0.45、每任务init0/seed1000；
  clean与low_contrast各10/10成功，差0个百分点，10组均双成功；耗时193.602s，
  optimizer steps=0。此前v1在task4初始化配对失败的记录保留，不混入本次成绩。
  本轮只读复算screen/check/reset哈希与资源证据链、20个fresh-env关闭记录及前置
  guard，核验2071条动作/观测/策略输入，80张首末PNG重建实际policy tensor哈希
  全匹配，确认噪声确实进入模型；393份历史工件及完整权重资源重哈希通过。
  本次机械臂6维越界计数0，夹爪757，按预先固定的原生控制器协议仅作诊断，未额外
  裁剪，也不回填旧B4b结果。13张代表图已查看，`viewed_not_accepted`；末帧为最后
  动作前，未用于重判成功。结果为单初始化有限筛查的天花板表现，不证明普遍鲁棒、
  无退化、真实血管氧化适配或world-model收益；保留固定基线，不据全成功事后加大
  噪声。证据：`simulation_output/pi05_libero_low_contrast_v2_screen_v1/report.json`
  （SHA256=8b59c322385b12e2c44dd68facd9ff0a8f182dca83c696bab41982ad7369b1b5）、
  `post_run_audit.json`和`visual_review.json`。未启动新实验/训练，未改数据或SOFA轨。

- 2026-09-14：[algorithm/VLA][BC/ACT clean Push-T] 恢复研究时发现用户此前的完整
  benchmark已完成，现核对固定100个seed、200条episode与57783步日志：BC0/100，
  ACT22/100；平均episode最大覆盖率0.202458/0.512645，均零动作越界、原协议通过。
  ACT相对BC提高22个百分点；BC基值为零，不报告相对成功率增幅。两者同训练数据/
  split/100000更新的单训练种子比较，不代表多种子稳定优势。官方预训练DP67/100
  单列外部参考，不能作三模型同数据/预算排名；与导丝/实机表现无直接对应。
  首个固定场景的记录动作重放：BC停在物体外，ACT推动但未完成；中文图已查看，
  viewed_not_accepted。首次ACT跨进程回放一致性检查失败，诊断重试600步全匹配；
  未修改动作/容差/环境，首次差异原因未明，不宣称跨进程逐位复现。没有重跑原基准、
  新训练或硬件动作。证据：`simulation_output/pusht_bc_act_benchmark_audit_v1_diagnose/report.json`
  及`first_seed_replay_zh.png`；算法轨交接已更新，real10权重保留待用户到现场。
