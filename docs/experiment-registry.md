# Experiment Registry

Last updated: 2026-06-19

This document records experiment naming and which results are considered useful
references. Generated artifacts live under `simulation_output/`.

## Naming Rules

Recommended directory style:

```text
simulation_output/<pipeline>_<version>
simulation_output/<model_name>_rollout
simulation_output/<model_name>_rollout_<condition>
simulation_output/<dataset_name>
```

Examples:

```text
mujoco_physical_feed_action_dataset_v2
baseline_mujoco_physical_feed_action_v2
baseline_mujoco_physical_feed_action_v2_rollout
compare_unweighted_newenv_rollout
```

Use descriptive suffixes for ablations:

```text
_piperw2
_guard
_rate_limit
_video
_smooth_elite
```

## Formal Reference Results

These are currently meaningful references:

```text
simulation_output/mujoco_physical_feed_action_dataset_v2
simulation_output/baseline_mujoco_physical_feed_action_v2
simulation_output/baseline_mujoco_physical_feed_action_v2_smooth4
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate015_r1
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate010_r1
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate010_smooth025_r1
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1_video
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_jerk001_r1
simulation_output/baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_jerk001_r1_video
simulation_output/baseline_mujoco_physical_feed_action_v2_elitedelta
simulation_output/baseline_mujoco_physical_feed_action_v2_elitedelta_rollout
simulation_output/real_branch_data_inspection
simulation_output/real_control_bandwidth
simulation_output/sim_vs_real_alignment
simulation_output/sim_vs_real_alignment_smooth4
simulation_output/sim_vs_real_alignment_rate015_r1
simulation_output/sim_vs_real_alignment_rate010_r1
simulation_output/sim_vs_real_alignment_rate010_smooth025_r1
simulation_output/sim_vs_real_alignment_elitedelta
docs/_rollout_control_bandwidth_check
docs/_rollout_control_bandwidth_accel002_check
docs/_rollout_control_bandwidth_accel002_video_check
docs/_elite_visual_jitter_source_accel002_video_check
docs/_rollout_control_bandwidth_jerk001_check
docs/_elite_visual_jitter_source_jerk001_check
docs/_rollout_control_bandwidth_jerk001_video_check
docs/_elite_visual_jitter_source_jerk001_video_check
simulation_output/baseline_mujoco_physical_feed_action_v1
simulation_output/baseline_mujoco_physical_feed_action_weighted_v1
simulation_output/compare_unweighted_newenv_rollout
simulation_output/compare_weighted_newenv_rollout
```

Reason:

- `mujoco_physical_feed_action_dataset_v2` is the current physical-environment
  dataset: 80/80 successful episodes, 8232 samples, action type
  `piper_feed_elite_joint`, and no missing images in sparse file check.
- `baseline_mujoco_physical_feed_action_v2` is the first baseline trained on
  that dataset without high Piper negative weighting. Best observed validation
  loss was `0.000727` at epoch 5. Its rollout succeeded in progress but still
  had high Elite target jump p95 around `0.053-0.055`, while the expert labels
  in the dataset have Elite joint step p95 around `0.00561`.
- `real_branch_data_inspection` summarizes the real data in `branchs/`: 24 real
  paths, 4039 synchronized pose/image frames, binary Piper labels, and real pose
  position-step p95 around 5 mm.
- `real_control_bandwidth` is the current real Elite/magnetic-arm bandwidth
  anchor. It reports position-step p95 median/max `5.072 / 5.564 mm/frame`,
  acceleration p95 median `2.982 mm/frame^2`, jerk p95 median
  `2.835 mm/frame^3`, direction-reversal median `0.0000`, and zero-step
  fraction median `0.4131`.
- `sim_vs_real_alignment` compares real Elite/magnetic-arm pose with simulated
  `elirobot_pose` and `magnetic_pose`. It explicitly does not treat real pose as
  guidewire ground truth.
- `sim_vs_real_alignment_smooth4` is the latest alignment table after the
  smoothness-loss experiment. It shows visible rollout `elirobot_pose` motion is
  still jumpier than real pose scale, while `magnetic_pose` remains smoother.
- `baseline_mujoco_physical_feed_action_v2_smooth4` tested adjacent-sample Elite
  smoothness loss. It completed both branches, but closed-loop jitter did not
  consistently improve: left target p95 worsened to about `0.0667`, while right
  improved to about `0.0318`.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate015_r1` tested
  rollout-time Elite joint rate limiting at `0.015` rad/env-step. It preserved
  left/right success and capped executed joint jumps to about `0.015`, reducing
  visible Elite pose step max to about `10-11 mm`; raw policy target p95 stayed
  around `0.054`.
- `sim_vs_real_alignment_rate015_r1` shows the rate-limited rollout moved
  visible `elirobot_pose` step p95 median from the previous `~11.05 mm` toward
  `~7.81 mm`, closer to the real branch pose anchor `~5.07 mm`, while
  `magnetic_pose` remained smooth at about `~1.07 mm` p95.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate010_r1` tightened the
  rollout-time Elite joint rate limit to `0.010` rad/env-step. It preserved
  left/right success, capped executed joint jumps to about `0.010`, and reduced
  visible Elite pose step max to about `7-8 mm`; raw policy target p95 remained
  around `0.046-0.051`.
- `sim_vs_real_alignment_rate010_r1` is the current best real-aligned motion
  reference: visible `elirobot_pose` step p95 median is about `5.78 mm`, close
  to the real branch pose anchor `~5.07 mm`, while `magnetic_pose` remains
  smooth at about `~1.11 mm` p95.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate010_smooth025_r1` adds
  stronger rollout-time Elite low-pass smoothing (`elite_motion_smoothing=0.25`)
  on top of the `0.010` rad/env-step joint rate limit. It preserved left/right
  success and reduced visible Elite pose max steps to about `5.7-6.7 mm`, but
  increased target-to-executed tracking lag.
- `sim_vs_real_alignment_rate010_smooth025_r1` shows visible `elirobot_pose`
  step p95 median at about `4.37 mm`, slightly below the real branch pose anchor
  `~5.07 mm`, while `magnetic_pose` remains smooth at about `~1.11 mm` p95.
- `baseline_mujoco_physical_feed_action_v2_elitedelta` trained Elite outputs as
  deltas from current Elite joints while preserving the external
  `piper_feed_elite_joint` rollout schema. Its rollout completed both branches
  and reduced target jump p95, but it is not a good reference model because the
  visible Elite tool drifted far from the guidewire/magnetic point.
- `sim_vs_real_alignment_elitedelta` confirms the naive delta model regressed
  visible robot motion scale: `elirobot_pose` step p95 median returned to about
  `11.02 mm`, while `magnetic_pose` stayed smooth around `1.03 mm`.
- `docs/_rollout_control_bandwidth_check` is the first output from
  `tools/compare_rollout_control_bandwidth.py`. It confirms that
  `rate010_smooth025_r1` matches real step p95 reasonably well but still has
  high visible Elite jerk and direction-reversal behavior, explaining why video
  still looks jittery.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1` adds
  `elite_joint_accel_limit=0.002` on top of `elite_joint_rate_limit=0.010`.
  It preserved left/right success and reduced visible `elirobot_pose`
  acceleration/jerk to `1.647 / 1.997`, below the real anchor
  `2.982 / 2.835`, while keeping step p95 close to real at `4.777 mm`.
- `docs/_rollout_control_bandwidth_accel002_check` compares `rate010_r1`,
  `rate010_smooth025_r1`, and `rate010_accel002_r1`; it shows direction
  reversal dropping from roughly `0.108-0.118` to `0.0231` with acceleration
  limiting.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_r1_video`
  is the video-rendered acceleration-limit rollout. It preserved left/right
  success (`454/459` steps) and matched the real bandwidth anchor well:
  visible `elirobot_pose` step p95 `4.817 mm`, acceleration p95
  `1.702 mm/frame^2`, and jerk p95 `2.031 mm/frame^3`.
- `docs/_rollout_control_bandwidth_accel002_video_check` records that the video
  rollout is close to or below the real bandwidth anchor, while
  `docs/_elite_visual_jitter_source_accel002_video_check` records the remaining
  diagnosis: policy target jitter remains, executed joint jerk is still a
  concern, and logged tool pose is already smooth. The user still observed
  visual twitching, so this is not solved by tool-pose bandwidth matching alone.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_jerk001_r1`
  is the first confirmed jerk-limit rollout. Its metadata records
  `elite_joint_jerk_limit=0.001`, and it preserved left/right success
  (`466/481` steps). Compared with `accel002`, it reduced visible
  `elirobot_pose` jerk p95 from `1.997` to `0.758 mm/frame^3` and direction
  reversal from `0.0231` to `0.0063`, but increased visible step p95 from
  `4.777` to `6.220 mm` and increased median `tip_to_elite_tool` from about
  `33 mm` to about `38 mm`.
- `docs/_rollout_control_bandwidth_jerk001_check` and
  `docs/_elite_visual_jitter_source_jerk001_check` are the corresponding
  diagnostics. The result is promising enough for video review, but it is a
  tradeoff rather than a clear numeric win.
- `baseline_mujoco_physical_feed_action_v2_rollout_rate010_accel002_jerk001_r1_video`
  is the video-rendered jerk-limit rollout. It preserved left/right success
  (`457/471` steps) and reduced visible `elirobot_pose` jerk p95 to
  `0.798 mm/frame^3`, but visible step p95 increased to `6.107 mm` and the user
  reported the video is still poor, even worse than `accel002`.
- `docs/_rollout_control_bandwidth_jerk001_video_check` and
  `docs/_elite_visual_jitter_source_jerk001_video_check` record this negative
  result. Do not keep tightening execution-layer jerk limits as the main fix;
  inspect robot visual/link mapping or policy target jitter instead.
- Historical rollouts now show that older checkpoints had much lower policy
  target p95 values: around `0.005-0.007` for centerline/physical_guidance/old
  feed_action rollout, `0.013-0.022` for unweighted feed_action_v1 in the newer
  environment, and `0.046-0.086` for weighted/current variants.
- They were compared in the same current environment.
- They support the conclusion that high Piper negative weighting amplified Elite
  joint-output instability.

## Temporary or Historical Outputs

Older outputs from 2D, early centerline, or pre-dual-arm stages are useful for
history but should not be treated as current baselines.

Examples of less-current categories:

```text
mesh_3d*
dual_arm_dataset*
baseline_dual_arm*
mujoco_physical_guidance_stiffer*
```

Do not delete recent MuJoCo dual-arm outputs without user approval.

## Required Files in a Dataset

A usable dataset directory should contain:

```text
manifest.json
episode_*/meta.json
episode_*/states.jsonl
episode_*/actions.jsonl
episode_*/frames/side/*.png
episode_*/frames/top/*.png
```

The manifest should include:

- accepted episodes;
- rejected episodes if any;
- sample list;
- action mode;
- joint names;
- collection settings.

## Required Files in a Model Output

A usable trained model directory should contain:

```text
best_model.pt
train_log.json
```

Important fields in `train_log.json`:

- manifest path;
- action mode;
- state/action dimensions;
- piper and Elite joint names;
- train/validation loss;
- separate Piper and Elite loss.

## Required Files in a Rollout Output

A rollout output should contain one subfolder per task:

```text
left/meta.json
left/states.jsonl
left/actions.jsonl
left/rollout.mp4   # if --video was used
right/meta.json
right/states.jsonl
right/actions.jsonl
right/rollout.mp4
summary.json
```

After running diagnostics, it may also contain:

```text
elite_diagnostics_summary_all.json
left/elite_diagnostics_summary.json
right/elite_diagnostics_summary.json
```

## Key Metrics

For rollout success:

- `success`
- `steps`
- `distance_to_target`
- `path_progress`
- `max_contact_strength`
- `min_segment_distance_to_wall`

For Elite visual/control stability:

- `elite_target_joint_step_linf`
- `elite_executed_joint_step_linf`
- `elite_target_to_executed_joint_l2`
- `elite_tool_step`
- `magnetic_step`
- `tip_to_elite_tool`
- `tip_to_magnetic`

For dataset quality:

- accepted episode count;
- rejected episode count;
- sample count;
- distribution of `progress_delta`;
- distribution of wall/contact metrics;
- visual inspection of random episodes.
