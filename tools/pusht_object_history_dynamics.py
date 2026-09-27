"""Two observable frames for the residual Push-T predictor, not a policy.

Uses the original cache/windows and shared encoder. Past/current observations
are explicit inputs; future grids remain loss targets and goals remain scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from . import pusht_object_dynamics as base
    from . import pusht_object_residual_dynamics as residual
else:
    import pusht_object_dynamics as base
    import pusht_object_residual_dynamics as residual

SCHEMA = "pusht_rgb_object_history_dynamics_v1"
CACHE_ROOT = base.CACHE_ROOT
TRAIN_ROOT = base.TRAIN_ROOT.parent / "pusht_object_history_dynamics_v1"
HISTORY = {
    "frames": "previous=t-1,current=t_in_same_episode_and_consecutive_recorded_frames",
    "previous_required_valid": "RGB_object_visible_and_previous_frame_exists",
    "missing": "copy_current_grid_and_XY_into_previous_slots_set_history_valid_false",
    "windows": "keep_all_original_train_and_val_anchors_no_history_filtering",
    "visual_motion": "shared_encode(current_grid)-shared_encode(previous_grid)",
    "agent_motion": "(current_XY-previous_XY)/existing_training_state_std",
    "motion_units": "one_recorded_frame_difference_not_physical_velocity",
    "projection_input": "128_visual_delta+2_normalized_XY_delta+1_history_valid",
    "projection": "bias_free_Linear131_to128_added_to_original_current_context",
    "statistics": "no_new_normalization_fit",
    "previous_action_or_goal_or_future_or_hidden_state_input": False,
}
PLAN = {**residual.PLAN, "schema": SCHEMA,
    "architecture": "residual128_GRU_plus_shared_encoder_two_frame_difference_projection",
    "initialization": "same_seed_residual_trunk_zero_decoder_default_history_projection",
    "temporal_mapping": "grid[t-1:t+1],agentXY[t-1:t+1],valid,action[t:t+8]->grid[t+1:t+9]",
    "history": HISTORY, "history_projection_parameters": 131 * 128,
    "base_model_schema": residual.SCHEMA,
    "comparison": "same_windows_and_budget_vs_residual;history_attribution_needs_capacity_matched_history_off_control",
}
MISSING_REASONS = ("available", "episode_start", "nonconsecutive_frame", "previous_object_invalid")
evaluate = base.evaluate
dice_cost = base.dice_cost


def history_indices(episodes, frames, visible):
    """No wraparound, future access, adjacent-episode history or new exclusions."""
    current = np.arange(len(episodes), dtype=np.int64)
    previous = np.maximum(current - 1, 0)
    start = (current == 0) | (episodes[previous] != episodes)
    gap = ~start & (frames[previous] + 1 != frames)
    invalid = ~start & ~gap & ~visible[previous]
    reasons = np.zeros(len(current), dtype=np.uint8)
    reasons[start], reasons[gap], reasons[invalid] = 1, 2, 3
    available = reasons == 0
    return np.where(available, previous, current), available, reasons


@dataclass(frozen=True)
class ObservableHistory:
    current_xy: torch.Tensor       # [B,2], native observed coordinates
    previous_grid: torch.Tensor    # [B,24,24], canonical copy if unavailable
    previous_xy: torch.Tensor      # [B,2], canonical copy if unavailable
    valid: torch.Tensor            # [B], Boolean; not a measured motion label


def make_history(current, current_xy, *, previous_grid=None, previous_xy=None, valid=None):
    """Construct a stateless history input; caller owns episode/time alignment.

    Omit all three previous fields for a reset/missing observation. Explicitly
    unavailable previous contents (including NaNs) are ignored, not multiplied
    by zero. Availability cannot certify a stale frame; check timing upstream.
    """
    b = len(current)
    if current.shape != (b, 24, 24) or current_xy.shape != (b, 2):
        raise ValueError("current grid/XY require [B,24,24]/[B,2]")
    if previous_grid is None and previous_xy is None and valid is None:
        valid = torch.zeros(b, dtype=torch.bool, device=current.device)
        previous_grid, previous_xy = current, current_xy
    elif previous_grid is None or previous_xy is None or valid is None:
        raise ValueError("provide all previous-grid/XY/valid fields or none")
    if previous_grid.shape != current.shape or previous_xy.shape != current_xy.shape or valid.shape != (b,) or valid.dtype != torch.bool:
        raise ValueError("history shape or Boolean availability mismatch")
    return ObservableHistory(current_xy,
        torch.where(valid[:, None, None], previous_grid, current),
        torch.where(valid[:, None], previous_xy, current_xy), valid)


class ObjectData(base.ObjectData):
    """Same cache/schema/anchors/targets; attach only an observable predecessor."""

    def __init__(self, root: Path = CACHE_ROOT, device: str = "cuda"):
        super().__init__(root, device)
        self.previous_indices, self.history_valid, self.history_reasons = history_indices(self.episodes, self.frames, self.valid)

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        current, xy, actions, targets = super().batch(indices)
        previous = torch.as_tensor(self.previous_indices[indices], device=self.grids.device)
        history = make_history(current, xy, previous_grid=self.grids[previous],
            previous_xy=self.states[previous], valid=torch.as_tensor(self.history_valid[indices], device=self.grids.device))
        return current, history, actions, targets


class ObjectGridDynamics(residual.ObjectGridDynamics):
    """Keep the residual GRU/output; add a small shared-feature history context."""

    def __init__(self, stats: dict):
        super().__init__(stats)
        self.history_embed = nn.Linear(131, base.PLAN["hidden_dim"], bias=False)

    def history_features(self, current, history, encoded_current=None):
        history = make_history(current, history.current_xy, previous_grid=history.previous_grid,
            previous_xy=history.previous_xy, valid=history.valid)
        encoded_current = self.encode(current.flatten(1)) if encoded_current is None else encoded_current
        visual_delta = encoded_current - self.encode(history.previous_grid.flatten(1))
        xy_delta = (history.current_xy - history.previous_xy) / self.state_std
        return torch.cat((visual_delta, xy_delta, history.valid[:, None].to(current.dtype)), dim=1)

    def forward(self, current, history: ObservableHistory, actions):
        b = len(current)
        if not isinstance(history, ObservableHistory):
            raise TypeError("explicit ObservableHistory required, not the old single-frame XY tensor")
        if actions.ndim != 4 or actions.shape[0] != b or actions.shape[2:] != (8, 2):
            raise ValueError("absolute native actions require [B,K,8,2]")
        encoded = self.encode(current.flatten(1))
        initial = encoded + self.state_embed((history.current_xy - self.state_mean) / self.state_std)
        initial = initial + self.history_embed(self.history_features(current, history, encoded))
        k = actions.shape[1]
        hidden = initial[:, None].expand(-1, k, -1).reshape(b * k, -1)
        normalized = ((actions - self.action_mean) / self.action_std).reshape(b * k, 8, 2)
        output = []
        for t in range(8):
            hidden = self.cell(self.action_embed(normalized[:, t]), hidden)
            delta = self.decode(hidden).reshape(b, k, 24, 24)
            output.append((current[:, None] + delta).clamp(0, 1))
        return torch.stack(output, dim=2)


def load_final(path: Path, *, device: str = "cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["schema"] != SCHEMA or payload["kind"] != "final" or payload["step"] != PLAN["steps"] or payload["plan"] != PLAN:
        raise ValueError("requires this fixed final history checkpoint, not a base/residual checkpoint")
    model = ObjectGridDynamics(payload["cache_identity"]["normalization"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)
