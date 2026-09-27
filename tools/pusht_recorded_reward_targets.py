"""Offline Push-T outcome targets; reward is never a scorer observation.

This is a separate sidecar, not a change to DS0 or its feature cache. The
published reward has a documented historical next-frame convention, but its
equivalence to the currently installed environment has not been established.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import pusht_object_dynamics as base
else:
    import pusht_object_dynamics as base

ROOT = Path(__file__).resolve().parents[1]
TARGET_ROOT = ROOT / "simulation_output/pusht_recorded_reward_targets_v1"
SCHEMA = "pusht_recorded_reward_targets_v1"
REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
CONTRACT = {
    "schema": SCHEMA,
    "target": "terminal_recorded_reward_8step",
    "horizon": base.HORIZON,
    "current_observation": "RGB_object_grid[t],observable_agentXY[t]",
    "candidate_input": "observed_absoluteXY_actions[t:t+8]",
    "target_mapping": "next.reward[t+7] -> recorded_reward_of_frame[t+8]",
    "direction": "higher_is_better",
    "target_units": "published_reward_unchanged_not_exact_current_environment_coverage",
    "historical_converter_formula": "clip(geometric_coverage/0.95,0,1)",
    "historical_time_mapping": "next.reward[i]=reward[min(i+1,last_frame)]",
    "snapshot_timing_evidence": "historical_official_converter_and_episode_tail_pattern",
    "exact_snapshot_generator_runtime_verified": False,
    "current_environment_coverage_equivalence_verified": False,
    "current_reward_mapping": "next.reward[t-1] only for frame[t]>0",
    "missing_current_reward": "NaN_and_false_mask_at_episode_start_never_zero_imputed",
    "delta_reward": "diagnostic_only_terminal_minus_current_when_current_available",
    "window_selection": "unchanged_original_complete_all9_visible_windows_no_padding",
    "target_normalization": "population_mean_std_of_original_training_windows_only",
    "policy_input_fields": ["current_grid", "agent_pos", "actions"],
    "forbidden_policy_inputs": ["reward", "done", "success", "future_grid", "episode_id", "frame_id"],
    "candidate_outcomes_used": False,
    "architecture_loss_candidates_changed": False,
    "training_authorized_by_preparation": False,
}
PROVENANCE = {
    "pinned_dataset": "https://huggingface.co/datasets/lerobot/pusht/tree/" + REVISION,
    "historical_official_converter": (
        "https://github.com/huggingface/lerobot/blob/8e7d697/"
        "lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py"
    ),
    "converter_lines": "74-85, 115-149, 176-185",
    "historical_geometry_implementation": (
        "https://github.com/huggingface/gym-pusht/blob/"
        "e0684ff988d223808c0a9dcfaba9dc4991791370/gym_pusht/envs/pusht.py"
    ),
    "limitation": (
        "The snapshot does not identify its exact raw-converter/dependency runtime or retain "
        "block pose. Historical code explains the field and timing, but is not an exact "
        "reproduction of these labels or a current-environment coverage calibration."
    ),
}


class RecordedRewardData:
    """Reuse ObjectData inputs/cohorts, return the scalar target separately.

    The terminal target preserves episode-start windows. A gain target would
    need the missing initial reward or a newly declared matched-cohort baseline.
    This class deliberately does not expose a delta-training mode.
    """

    def __init__(self, data: base.ObjectData, root: Path = TARGET_ROOT):
        self.data = data
        self.manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        m = self.manifest
        if m["contract"] != CONTRACT or m["cache_identity"] != data.identity:
            raise ValueError("reward sidecar must match the unchanged feature cache and contract")
        if m["status"] != "prepared_recorded_reward_only":
            raise ValueError("reward sidecar is not prepared")
        with np.load(Path(root) / "targets.npz", allow_pickle=False) as a:
            if not (np.array_equal(a["episodes"], data.episodes)
                    and np.array_equal(a["frames"], data.frames)
                    and np.array_equal(a["train"], data.train)
                    and np.array_equal(a["val"], data.val)):
                raise ValueError("row alignment or original train/validation cohort changed")
            reward = a["next_reward"]
            if reward.shape != data.episodes.shape or not np.isfinite(reward).all():
                raise ValueError("missing or nonfinite recorded reward; never impute")
            self.reward = torch.as_tensor(reward, dtype=torch.float32, device=data.grids.device)
        self.train, self.val = data.train, data.val
        self.allowed = np.zeros(len(data.episodes), dtype=bool)
        self.allowed[np.concatenate((self.train, self.val))] = True
        self.target_stats = m["target_stats"]

    def batch(self, anchors):
        """Return ((current_grid, agent_pos, action_chunk), terminal_target).

        Shapes are [B,24,24], [B,2], [B,1,8,2] and [B,1]. Only the first
        tuple may be passed to the unchanged DirectActionScorer.forward.
        """
        ids = torch.as_tensor(anchors, dtype=torch.long, device=self.data.grids.device)
        if ids.ndim != 1 or not len(ids):
            raise ValueError("nonempty one-dimensional original window anchors required")
        rows = ids.cpu().numpy()
        if (rows < 0).any() or (rows >= len(self.allowed)).any() or not self.allowed[rows].all():
            raise ValueError("only original eligible windows are supported; no padded targets")
        action_rows = ids[:, None] + torch.arange(base.HORIZON, device=ids.device)[None]
        inputs = (self.data.grids[ids], self.data.states[ids], self.data.actions[action_rows][:, None])
        target = self.reward[ids + base.HORIZON - 1, None]
        return inputs, target
