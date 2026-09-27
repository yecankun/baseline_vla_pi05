# Project Notes for Future Agents

## Big Picture

This repository is an inherited guidewire intervention project. The original
codebase focuses on real-camera perception, real robot control, and behavior
cloning models for a dual-arm setup. The current research direction is to move
from a small imitation-learning baseline toward a stronger VLA-style system with
tactile/contact information. Because real-robot data collection is expensive and
the user is not always physically in the lab, the current priority is building a
usable simulation pipeline before returning to the real system.

The present simulation track is intentionally staged:

1. Build a lightweight MuJoCo MVP with the real vessel mesh and two robot assets.
2. Use it to collect image/state/action imitation data.
3. Train a small dual-camera/state baseline to validate the data/control loop.
4. Improve simulator realism and control semantics before considering Isaac Sim
   or larger VLA models.

Do not jump to OpenVLA/large-model work yet. The current bottleneck is still
simulation fidelity, action semantics, and stable data generation.

Highest priority:

```text
simulation realism and real-system correspondence > BC rollout success
```

The small BC model is only a loop-closure/regression check. Do not treat a
simulation rollout as solved if it depends on a force, expert controller,
oracle state, hidden centerline correction, exact contact/wall feedback, or
tip-relative anchor that cannot exist in the real setup.

The current project route is Route 2: formal synthetic data should avoid
oracle tip-relative anchor/control unless a real sensing/control path is later
defined. Use `--formal-data` on collection scripts and `--formal-eval` on
rollout scripts to make these checks fail closed. Current oracle experts are
diagnostic only.

## Documentation Index

Start with these docs before making non-trivial changes:

- `docs/project-state.md`: current mainline, effective environment, recent
  experiment conclusions, and the agreed next step.
- `docs/decision-log.md`: high-impact decisions and why they were made,
  especially action schema, simulation route, and data strategy.
- `docs/commands.md`: copyable current commands for collection, training,
  rollout, diagnostics, and controlled comparisons.
- `docs/sim-vs-real-alignment.md`: latest real-data anchored comparison between
  `branchs` Elite/magnetic-arm trajectories and MuJoCo simulated robot/magnetic
  motion.
- `docs/open-questions.md`: unresolved research questions that should not be
  mistaken for simple implementation bugs.
- `docs/experiment-registry.md`: experiment naming, output directory
  conventions, and which results are formal references.
- `docs/handoff.md`: stage handoff, current blocker, and recommended next
  commands.
- `docs/data-and-action-schema.md`: current observation/action semantics,
  dataset layout, and senior-data interpretation.
- `docs/simulation-assumptions.md`: intentional simulator simplifications and
  current realism boundaries.
- `docs/vla-target-interface.md`: target observation/action contract for the
  future VLA system and the small-BC interface tests.
- `docs/control-layer-contract.md`: hybrid policy/controller split for Elite
  TCP-delta execution and Piper discrete intent execution.
- `docs/real-observable-interface-audit.md`: tiered boundary for direct real
  signals, controller state, estimator/tactile signals, and simulation-only
  privileged fields before formal training.
- `docs/real-data-collection-checklist.md`: minimum real pilot capture schema
  for sim-to-real shadow evaluation, including required images, Elite TCP pose,
  Piper state/action labels, synchronization, estimator placeholders, and JSONL
  record layout.
- `docs/visual-tip-contact-estimator-plan.md`: current no-new-hardware
  perception route for rule-based and learned visual guidewire tip/contact
  estimators.
- `docs/senior-thesis-interface-audit.md`: targeted audit of the local ignored
  senior thesis PDF, used as paper-backed evidence for real-system observation,
  action, estimator, and controller semantics.
- `docs/troubleshooting.md`: common failures and diagnostic flow.

There is also a Codex skill for this project:

- `project-2026-guidewire-mujoco`: workflow guardrails for MuJoCo guidewire
  data collection, training, rollout, diagnostics, and handoff work. The
  canonical project copy lives in
  `.codex/skills/project-2026-guidewire-mujoco`. A user-level installed copy may
  exist at `C:\Users\Silence\.codex\skills\project-2026-guidewire-mujoco` only
  so Codex can auto-discover it. Prefer editing the project copy first, then
  sync/install it if needed.

## Repository Map

- `readme.md`: original short project description in Chinese.
- `intervention/`: senior's trained models.
  - `bc_elirobot`: model for the magnetic guidance arm only.
  - `bc_piper`: model for the guidewire feeding arm only.
  - `dual_bc`: dual-arm behavior cloning model.
- `data/`: original collection-related assets/code.
- `utils/`: original utilities.
  - `utils/interface/opengl_interface.py`: old OpenGL STL/robot interface. It
    requires `numpy-stl`, PyOpenGL, and a system GLUT/freeglut runtime on
    Windows.
  - `utils/interface/model/0422.stl`: vessel mesh currently used by simulation.
- `branchs/`: senior-provided real trajectory/data files. These are useful for
  rough scene alignment, but should not be treated as guidewire centerline
  supervision.
- `robot_assets/`: cloned ROS robot repositories for the two arms.
- `simulation/`: all new simulation/data/training/evaluation code.
- `tools/`: visualization, calibration, diagnostic, and analysis scripts.
- `simulation_output/`: generated datasets, models, rollouts, videos, camera and
  scene configs. This folder can become large and should be cleaned periodically.

## Current Simulation State

The active environment is `simulation/mujoco_guided_wire_env.py`.

Important config/assets:

- Vessel mesh: `utils/interface/model/0422.stl`
- Route config: `simulation/routes/vessel_0422_wire_route_v1.json`
- Robot scene config: `simulation_output/robot_scene_mvp/scene_config.json`
- Camera config: `simulation_output/mujoco_camera_config.json`

The two robot roles are:

- Piper: located near the vessel entrance and responsible for feeding/retracting
  the guidewire. It should not follow the guidewire tip through the vessel.
- Elite: magnetic guidance arm. It positions the magnetic tool above/near the
  guidewire tip and helps guide the tip at bends/branches.

The external compatibility action schema is:

```text
piper_feed_elite_joint
```

This means:

- Piper action is one scalar normalized feed delta: `piper_feed`.
- Elite execution can still be represented as six joint targets:
  `elite_joints`.

Current VLA-facing learning semantics are narrower:

```text
Elite: TCP delta -> IK -> joints
Piper: discrete hold/feed/retract, with senior-compatible hold/feed as the
minimum real interface
```

Do not change the external schema casually, but do not mistake the compatibility
fields for the final VLA policy target. See `docs/vla-target-interface.md`.
The current control-layer decision is hybrid: policy predicts Elite TCP delta
and Piper intent; the controller owns IK, timing, limits, safety, and executed
command logs. See `docs/control-layer-contract.md`.

## Key Modeling Decisions

1. Senior's real trajectory data is likely Elite/magnetic-arm trajectory data,
   not the true guidewire path.
   - Use it only as a rough reference for scene alignment.
   - Do not use it as `elite_reference_path` in the simulator.
   - Do not force the guidewire to follow it.

2. The current MuJoCo physical guidance mode removed the artificial centerline
   force.
   - This is more physically honest.
   - The guidewire is now driven by magnetic influence, Piper feeding, gravity,
     wall interaction, and internal stiffness/damping.
   - Expect more contact and more sensitivity to control quality.

3. Some wall/contact metrics can be slightly negative.
   - `min_segment_distance_to_wall < 0` means part of the polyline is slightly
     inside or past the vessel boundary approximation.
   - Small negatives have appeared even in visually acceptable rollouts.
   - Track the magnitude and video, not only the sign.

4. Avoid visual leakage in training data.
   - Hide tool markers and path tubes for formal data collection.
   - Keep robot visuals enabled if rollout will use robot visuals; otherwise the
     model sees a different image distribution.

## Current Empirical Findings

The latest important diagnosis compared two trained models in the same current
environment:

- `simulation_output/baseline_mujoco_physical_feed_action_v1`
- `simulation_output/baseline_mujoco_physical_feed_action_weighted_v1`

The weighted model used `piper_feed_negative_loss_weight=4.0`. It learned Piper
rollback better, but made Elite joint outputs much less smooth.

Diagnostic results:

```text
unweighted:
left  target_jump_max=0.1138 target_jump_p95=0.0135 executed_joint_jump_max=0.0601
right target_jump_max=0.1061 target_jump_p95=0.0218 executed_joint_jump_max=0.0561

weighted:
left  target_jump_max=0.1014 target_jump_p95=0.0463 executed_joint_jump_max=0.0582
right target_jump_max=0.2067 target_jump_p95=0.0746 executed_joint_jump_max=0.1057
```

Conclusion:

- The new physical environment exposed the issue, but the main visible Elite
  jump problem is amplified by the weighted training strategy.
- Expert labels are much smoother than the closed-loop policy outputs.
- Future training should not blindly reuse `piper_feed_negative_loss_weight=4.0`
  without protecting Elite smoothness.

## Recommended Next Steps

The immediate handoff target is no longer basic data collection or BC rollout
tuning. The current dataset, baseline, rollout diagnostics, real-data alignment
table, and simulation-expert validity audit already exist. The next agent
should start from:

- `docs/project-state.md`
- `docs/handoff.md`
- `docs/simulation-expert-validity-audit.md`
- `docs/sim-vs-real-alignment.md`
- `docs/vla-target-interface.md`
- `docs/control-layer-contract.md`
- `docs/senior-thesis-interface-audit.md`
- `docs/commands.md`

Current priority:

1. Treat formal data generation as invalid if it depends on simulation-only
   privileged state. Current `--formal-data` intentionally rejects the oracle
   MuJoCo/tip-centric experts.
2. Treat `branchs/branch1` and `branchs/branch2` as real Elite/magnetic-arm
   motion anchors, not guidewire trajectories.
3. Use the real pose step scale, roughly 5 mm p95, to tune simulated visible
   Elite motion.
4. Do not simply increase `--elite-smoothness-weight`; the smooth4 experiment
   improved the right branch but worsened the left branch.
5. Try rollout-time Elite joint rate limiting first, while preserving the
   external `piper_feed_elite_joint` action schema.
6. If rate limiting is insufficient, implement action-delta or rate-limited
   target training while still outputting Piper feed plus six Elite joints.
7. Implement the hybrid control-layer contract: policy-level Piper intent must
   be explicit, while real/sim controller execution handles timing, cooldown,
   bounds, and logged executed commands.

Useful current references:

```text
simulation_output/mujoco_physical_feed_action_dataset_v2
simulation_output/baseline_mujoco_physical_feed_action_v2
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout
simulation_output/baseline_mujoco_physical_feed_action_v2_smooth4
simulation_output/baseline_mujoco_physical_feed_action_v2_smooth4_rollout
simulation_output/real_branch_data_inspection
simulation_output/sim_vs_real_alignment_smooth4
```

## Useful Commands

Rollout a trained checkpoint:

```powershell
.\.venv\Scripts\python.exe -m simulation.eval_mujoco_guided_wire_rollout `
  --checkpoint simulation_output\baseline_mujoco_physical_feed_action_v1\best_model.pt `
  --out simulation_output\some_rollout_name `
  --tasks left right `
  --guidance-mode physical `
  --start-fraction 0.58 `
  --max-steps 700 `
  --camera-size 224 `
  --render-width 960 `
  --render-height 720 `
  --policy-every 5 `
  --progress-log-every 100 `
  --piper-feed-phase-guard `
  --video `
  --video-camera overview `
  --render-every 2
```

Analyze Elite joint/pose jump diagnostics:

```powershell
.\.venv\Scripts\python.exe tools\analyze_mujoco_elite_rollout.py `
  simulation_output\some_rollout_name `
  --no-plots
```

Train the dual-camera/state baseline:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700
```

If testing Piper negative weighting, start conservatively:

```powershell
.\.venv\Scripts\python.exe -m simulation.train_dual_arm_baseline `
  --manifest simulation_output\mujoco_physical_feed_action_dataset_v2\manifest.json `
  --out simulation_output\baseline_mujoco_physical_feed_action_v2_piperw2 `
  --epochs 8 `
  --batch-size 32 `
  --image-size 224 `
  --max-steps 700 `
  --piper-feed-negative-loss-weight 2.0
```

## Things to Be Careful About

- Do not delete recent dual-arm/MuJoCo outputs without asking. Older 2D/early
  centerline outputs are usually less valuable, but confirm before cleanup.
- Long-running data collection and training are usually run manually by the
  user. Provide commands rather than starting huge jobs unless asked.
- On Windows, prefer `.\.venv\Scripts\python.exe` for project commands.
- For long-running work such as data collection, training, full rollout, or
  video-rendered evaluation, provide copyable commands for the user to run
  instead of starting the job directly, unless the user explicitly asks the
  agent to run it.
- This directory may not be a Git repository in the current workspace; do not
  rely on Git history being available.
- Use `rg`/PowerShell search to inspect files. Use `apply_patch` for edits.
- Do not change action semantics casually. The current sim-to-real preference is
  still Piper feed scalar plus Elite six-joint command.
- If a rollout succeeds numerically but the video looks wrong, inspect:
  - `elite_target_joint_step_linf`
  - `elite_executed_joint_step_linf`
  - `elite_tool_step`
  - `magnetic_step`
  - `tip_to_elite_tool`
  - `tip_to_magnetic`

## Longer-Term Direction

After the new MuJoCo dataset and baseline are stable:

1. Add explicit Elite smoothness regularization or action-delta modeling while
   preserving the six-joint output interface.
2. Improve physical realism: guidewire stiffness/contact, magnetic mapping,
   wall contact approximation, material/lighting/background domain randomization.
3. Validate whether the simulated cameras should match real side/top cameras
   more closely.
4. Compare against senior's real datasets only as a sim-to-real diagnostic, not
   as direct guidewire path supervision.
5. Move to Isaac Sim only after the MuJoCo control/data semantics are stable
   enough to justify the heavier simulator.
