"""Freeze matched terminal-score supervision, without model/training execution.

Both arms reuse DS0's architecture, original inputs/windows and MSE form. Only
the supervised scalar changes: negative terminal visual Dice vs terminal task
reward. This preparation does not provide or authorize a training runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import pusht_direct_action_scorer as ds
import pusht_recorded_reward_targets as reward_data

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "simulation_output/pusht_reward_geometry_audit_v1"
OUT = ROOT / "simulation_output/pusht_terminal_score_pair_pretrain_v1"
VARIANTS = ("visual_terminal", "reward_terminal")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(values):
    v = np.asarray(values, dtype=np.float64)
    return {"count": len(v), "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()), "unique": len(np.unique(v)),
            "zero": int((v == 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse overwrite: {args.output}")
    start = time.monotonic()
    audit = json.loads((args.audit / "report.json").read_text())
    if (audit["data_revision"] != reward_data.REVISION
            or not audit["native_shift1_reproduces_all_recorded_rewards"]
            or audit["reward_comparisons"]["native_pose_shift1"]["float32_exact_equal"] != 25650):
        raise ValueError("native-pose reward equivalence must first pass the full pinned-source audit")
    data = ds.base.ObjectData(device="cpu")
    rewards = reward_data.RecordedRewardData(data)
    with np.load(args.audit / "geometry_diagnostic_targets.npz", allow_pickle=False) as geometry:
        if not (np.array_equal(geometry["episodes"], data.episodes)
                and np.array_equal(geometry["frames"], data.frames)):
            raise ValueError("geometry targets must match the original row/episode ownership")
        native_reward = geometry["native_reward"].astype(np.float32)
    anchors = np.concatenate((data.train, data.val))
    labels = np.empty((len(anchors), 2), dtype=np.float32)
    # Reuse old numerical inputs; observed future grids are target-only.
    for lo in range(0, len(anchors), 512):
        ids = anchors[lo:lo + 512]
        endpoint = torch.as_tensor(ids + ds.HORIZON)
        labels[lo:lo + len(ids), 0] = (-ds.base.dice_cost(data.grids[endpoint], data.goal)).numpy()
        labels[lo:lo + len(ids), 1] = rewards.reward[ids + ds.HORIZON - 1].numpy()
    if not np.isfinite(labels).all() or not np.array_equal(labels[:, 1], native_reward[anchors + ds.HORIZON]):
        raise ValueError("incomplete or wrongly aligned terminal targets")
    if (labels[:, 0] > 0).any() or (labels[:, 0] < -1).any() or (labels[:, 1] < 0).any() or (labels[:, 1] > 1).any():
        raise ValueError("target sign/range changed")
    slices = {"train": slice(0, len(data.train)), "val": slice(len(data.train), None)}
    stats = {split: {variant: summarize(labels[rows, j]) for j, variant in enumerate(VARIANTS)}
             for split, rows in slices.items()}
    if any(stats["train"][v]["std"] <= 1e-8 for v in VARIANTS):
        raise ValueError("constant target")
    plan = {
        "schema": "pusht_terminal_score_pair_v1", "status": "supervision_prepared_not_trained",
        "variants": list(VARIANTS),
        "targets": {"visual_terminal": "-Dice(observed_RGB_grid[t+8],fixed_original_training_goal)",
                    "reward_terminal": "next.reward[t+7]=float32(clip(native_coverage_at_raw_pose[t+8]/0.95,0,1))"},
        "target_parametrization": "terminal_value_for_both_neither_is_an_improvement_delta",
        "network_class": "unchanged_pusht_direct_action_scorer.DirectActionScorer",
        "architecture": ds.PLAN["architecture"],
        "fixed_budget": {k: ds.PLAN[k] for k in ("seed", "steps", "batch_size", "lr", "weight_decay", "grad_clip",
                                                  "checkpoint_every", "log_every", "horizon")},
        "initialization": "reset_same_seed_per_arm_identical_initial_trainable_parameters_no_warmstart",
        "sampling": ds.PLAN["sampling"],
        "paired_sampling": "reset_same_numpy_sampler_seed_per_arm_identical_ordered_anchors_each_update",
        "loss": "unchanged_MSE_to_standardized_scalar",
        "target_normalization": {v: {"mean": stats["train"][v]["mean"], "std": stats["train"][v]["std"],
                                     "source": "original_21958_training_windows_population_stats"} for v in VARIANTS},
        "normalization_and_goal": "original_state_action_normalization_and_episode1_frame117_goal_buffer_for_both",
        "candidate_rule": ds.PLAN["candidate_rule"], "selection": ds.PLAN["selection"],
        "input_fields": ["current_RGB_grid", "observable_agent_XY", "candidate_native_absolute_XY_chunk"],
        "no_target_or_pose_in_policy_input": True,
        "checkpoint_selection": "each_arm_final10000_only_no_validation_tuning",
        "prediction_evaluation": "same2002_validation_windows_raw_score_MAE_RMSE_by_window_and_episode_macro",
        "imbalance": "report_zero_target_counts_and_episode_macro_not_only_pooled_window_error",
        "efficiency": "record_same4090_training_elapsed_time_and_peak_cuda_memory_per_arm_no_extra_benchmark",
        "generality": "one_training_seed_one_public_task_no_robustness_simreal_or_architecture_claim",
        "cross_target_error_caveat": "different_target_units_do_not_compare_raw_visual_MAE_to_reward_MAE_as_policy_gain",
        "candidate_evaluation": "separately_scoped_after_training_same_contexts_candidates_budget_for_both",
        "old57_contexts": "already_inspected_development_not_untouched_test_never_training_labels",
        "old_DS0": "historical_delta_target_reference_not_the_matched_visual_terminal_arm",
        "training_outputs": "/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_terminal_score_pair_v1/<variant>/",
        "training_runner_implemented": False, "training_authorized_by_preparation": False,
        "cache_identity": data.identity, "reward_geometry_audit": str(args.audit),
    }
    args.output.mkdir(parents=True)
    write(args.output / "comparison_contract.json", plan)
    np.savez_compressed(args.output / "paired_terminal_targets.npz", anchors=anchors,
                        is_train=np.arange(len(anchors)) < len(data.train), targets=labels,
                        episodes=data.episodes[anchors], frames=data.frames[anchors])
    report = {"schema": plan["schema"], "status": plan["status"], "statistics": stats,
              "all_original_windows_preserved": len(data.train) == 21958 and len(data.val) == 2002,
              "all_reward_targets_equal_static_native_geometry_float32": True,
              "geometry_current_values_available_at_all_episode_starts": True,
              "missing_initial_reward_changes_to_v1_sidecar": False,
              "validation_used_for_normalization_or_selection": False,
              "model_forward_calls": 0, "optimizer_steps": 0, "environment_steps": 0,
              "elapsed_seconds": time.monotonic() - start}
    write(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
