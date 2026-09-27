"""Small action-conditioned Push-T RGB-object predictor, separate from ACT.

The visible occupancy grid comes from pusht_object_goal, not simulator state.
Future images are supervision only. This module has no environment executor.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from . import pusht_object_goal as vision
    from .pusht_world_model_adapter import CandidateSet, HORIZON
else:
    import pusht_object_goal as vision
    from pusht_world_model_adapter import CandidateSet, HORIZON

SCHEMA = "pusht_rgb_object_dynamics_v1"
CACHE_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1")
TRAIN_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_object_dynamics_v1")
PLAN = {
    "schema": SCHEMA, "space_id": vision.SPACE_ID, "seed": 20260915,
    "steps": 10000, "batch_size": 64, "hidden_dim": 128, "horizon": HORIZON,
    "lr": 0.0003, "weight_decay": 0.0001, "grad_clip": 1.0,
    "checkpoint_every": 1000, "log_every": 100,
    "architecture": "flattened_spatial_grid_linear_LayerNorm_SiLU_state_projection_action_GRU_sigmoid_grid_decoder",
    "loss": "quadratic_soft_Dice_to_observed_future_grid_mean_all8_steps",
    "temporal_mapping": "grid[t],agentXY[t],action[t:t+8]->grid[t+1:t+9]",
    "window_validity": "complete_episode_window_and_all9_RGB_objects_valid_no_padding",
    "sampling": "uniform_eligible_training_windows_with_replacement",
    "normalization": "unchanged_BC_ACT_training_only_state_action_mean_std_grid_div16",
    "checkpoint_selection": "fixed_final10000_only_no_intermediate_validation",
    "goal_used_in_training_loss": False, "future_teacher_forcing": False,
    "source_render_rule": vision.PLAN, "online_invalid_current": "retain_ACT_no_object_forecast",
    "baseline": "same_window_current_object_grid_persistence",
    "action_ablation": "replace_actions_by_existing_training_mean_at_inference_not_separate_training",
    "heldout_metrics": ["dice_all8", "grid_mse_all8", "terminal_dice",
                        "terminal_goal_cost_mae", "terminal_centroid_xy_mae",
                        "terminal_area_relative_error"],
    "checkpoint_count_for_selection": 1, "training_seeds": 1,
    "candidate_ranking": "requires_separate_saved_development_check_after_training_not_implied_by_prediction_loss",
    "robustness_or_cross_task_evaluated": False, "policy_rollout_in_this_run": False,
}


def full_windows(episodes, frames, valid, split):
    """Keep original episode ownership; explicitly exclude invalid9-frame windows."""
    anchors = np.arange(len(episodes) - HORIZON, dtype=np.int64)
    indices = anchors[:, None] + np.arange(HORIZON + 1)[None]
    complete = (episodes[indices] == episodes[anchors, None]).all(1)
    complete &= (frames[indices] == frames[anchors, None] + np.arange(HORIZON + 1)).all(1)
    visible = valid[indices].all(1)
    selected, counts = {}, {}
    for name in ("train", "val"):
        owned = np.isin(episodes[anchors], split[name + "_episodes"])
        full = anchors[owned & complete]
        kept = anchors[owned & complete & visible]
        if not len(kept):
            raise ValueError(f"no visible complete windows in {name}")
        selected[name] = kept
        counts[name] = {"full_windows": len(full), "eligible_windows": len(kept),
                       "excluded_visibility_windows": len(full) - len(kept),
                       "episodes_with_eligible_windows": len(np.unique(episodes[kept]))}
    if np.intersect1d(selected["train"], selected["val"]).size:
        raise ValueError("overlapping train/validation windows")
    return selected, counts


def dice_cost(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Quadratic soft-Dice loss per spatial grid; target must have observed mass."""
    numerator = (prediction - target).square().sum((-2, -1))
    denominator = prediction.square().sum((-2, -1)) + target.square().sum((-2, -1))
    return numerator / denominator.clamp_min(1e-8)


class ObjectGridDynamics(nn.Module):
    """Causal128-hidden GRU; each output sees current grid/state and action prefix.

    Row-major flattening preserves image coordinates. The sigmoid decoder
    predicts a full24x24 visible occupancy grid, not RGB or a hidden rigid pose.
    No teacher-forced future, goal, source ID, coverage or acceptance metadata.
    """

    def __init__(self, stats: dict):
        super().__init__()
        self.space_id = vision.SPACE_ID
        hidden = PLAN["hidden_dim"]
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(name, torch.tensor(stats[name], dtype=torch.float32))
        self.encode = nn.Sequential(nn.Linear(24 * 24, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.state_embed = nn.Linear(2, hidden)
        self.action_embed = nn.Linear(2, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.decode = nn.Linear(hidden, 24 * 24)

    def forward(self, current, agent_pos, actions):
        b = len(current)
        if current.shape != (b, 24, 24) or agent_pos.shape != (b, 2):
            raise ValueError("current grid/state require [B,24,24]/[B,2]")
        if actions.ndim != 4 or actions.shape[0] != b or actions.shape[2:] != (HORIZON, 2):
            raise ValueError("absolute native actions require [B,K,8,2]")
        k = actions.shape[1]
        initial = self.encode(current.flatten(1)) + self.state_embed((agent_pos - self.state_mean) / self.state_std)
        hidden = initial[:, None].expand(-1, k, -1).reshape(b * k, -1)
        normalized = ((actions - self.action_mean) / self.action_std).reshape(b * k, HORIZON, 2)
        output = []
        for t in range(HORIZON):
            hidden = self.cell(self.action_embed(normalized[:, t]), hidden)
            output.append(self.decode(hidden).sigmoid().reshape(b, k, 24, 24))
        return torch.stack(output, dim=2)

    def loss(self, prediction, target):
        return dice_cost(prediction, target).mean()


@dataclass(frozen=True)
class ObjectCandidateAdvice:
    predicted: vision.ObjectGridFeatures  # NumPy [B,5,8,24,24], NaN if not predicted
    costs: np.ndarray  # [B,5], NaN for invalid observation, inf for rejected candidate
    selected_index: np.ndarray  # invalid observation retains reference0
    eligible_observation: np.ndarray
    reasons: tuple[str, ...]


@torch.inference_mode()
def predict_and_score(model: ObjectGridDynamics, current: vision.ObjectGridFeatures,
                      current_valid: np.ndarray, agent_pos: torch.Tensor,
                      candidates: CandidateSet, goal: vision.ObjectGridFeatures) -> ObjectCandidateAdvice:
    """Pure inference advice, never execution; unavailable vision preserves ACT.

    This has no uncertainty gate or policy-approval claim. A random model may
    be exercised by the pretraining check only, not used as a trained planner.
    """
    if model.training or current.space_id != model.space_id or goal.space_id != model.space_id:
        raise ValueError("requires eval mode and matching object-grid feature spaces")
    b = len(current.values)
    flags = np.asarray(current_valid)
    if flags.shape != (b,) or flags.dtype != bool:
        raise ValueError("one explicit Boolean visibility flag per observation is required")
    if current.values.shape != (b, 24, 24) or goal.values.shape != (b, 24, 24):
        raise ValueError("current and goal grids must both be [B,24,24]")
    if candidates.actions.shape != (b, 5, HORIZON, 2) or candidates.valid.shape != (b, 5):
        raise ValueError("requires the unchanged five native8-step candidates")
    bounded = (torch.isfinite(candidates.actions) & (candidates.actions >= 0) & (candidates.actions <= 512)).all((-2, -1))
    if not torch.equal(candidates.valid, bounded) or not bool(bounded[:, 0].all()):
        raise ValueError("candidate validity must match bounds and preserve reference; no clipping")
    for values in (current.values[flags], goal.values):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any() or (values.sum((-2, -1)) <= 0).any():
            raise ValueError("valid object grids must be finite nonempty occupancies")
    device = model.state_mean.device
    if agent_pos.shape != (b, 2) or not bool(torch.isfinite(agent_pos).all()):
        raise ValueError("requires observable finite native agentXY")
    predicted = np.full((b, 5, HORIZON, 24, 24), np.nan, dtype=np.float32)
    costs = np.full((b, 5), np.nan, dtype=np.float64)
    selected = np.zeros(b, dtype=np.int64)
    usable = candidates.valid & torch.as_tensor(flags, device=candidates.valid.device)[:, None]
    rows, columns = torch.where(usable)
    if len(rows):
        row_ids, col_ids = rows.cpu().numpy(), columns.cpu().numpy()
        grids = torch.from_numpy(current.values[row_ids]).to(device)
        states = agent_pos.to(device)[rows.to(device)]
        actions = candidates.actions[rows, columns].to(device)[:, None]
        values = model(grids, states, actions)[:, 0].cpu().numpy()
        predicted[row_ids, col_ids] = values
        scores = vision.score_candidate_grids(vision.ObjectGridFeatures(predicted[flags]),
                    vision.ObjectGridFeatures(goal.values[flags]), candidates.valid.cpu().numpy()[flags])
        costs[flags], selected[flags] = scores.costs, scores.selected_index
    reasons = tuple("object_forecast_score_no_calibrated_gate" if v else "invalid_current_object_retain_ACT" for v in flags)
    return ObjectCandidateAdvice(vision.ObjectGridFeatures(predicted), costs, selected, flags.copy(), reasons)


class ObjectData:
    def __init__(self, root: Path = CACHE_ROOT, device: str = "cuda"):
        self.root = Path(root)
        m = self.manifest = json.loads((self.root / "manifest.json").read_text())
        if m["schema"] != SCHEMA or m["status"] != "prepared" or m["plan"] != PLAN:
            raise ValueError("incompatible/incomplete object-grid cache")
        counts = np.load(self.root / "grid_counts.npy", allow_pickle=False)
        if counts.dtype != np.uint8 or counts.shape != (25650, 24, 24) or counts.max() > 16:
            raise ValueError("requires all25650 frame-aligned4x4 occupancy counts")
        self.grids = torch.from_numpy(counts.astype(np.float32) / 16).to(device)
        with np.load(self.root / "arrays.npz", allow_pickle=False) as a:
            self.states = torch.from_numpy(a["states"]).to(device)
            self.actions = torch.from_numpy(a["actions"]).to(device)
            self.episodes, self.frames, self.valid = a["episodes"], a["frames"], a["valid"]
        anchors, counted = full_windows(self.episodes, self.frames, self.valid, m["split"])
        if counted != m["window_counts"] or not np.array_equal(counts.sum((1, 2)) > 0, self.valid):
            raise ValueError("cache validity/window counts changed")
        self.train, self.val = anchors["train"], anchors["val"]
        self.stats = m["normalization"]
        self.goal = self.grids[m["goal"]["global_index"]].clone()
        self.identity = {k: m[k] for k in ("schema", "source_revision", "source_root", "split",
                         "normalization", "goal", "representation", "window_counts")}

    def batch(self, indices):
        indices = torch.as_tensor(indices, device=self.grids.device, dtype=torch.long)
        action_rows = indices[:, None] + torch.arange(HORIZON, device=indices.device)[None]
        return self.grids[indices], self.states[indices], self.actions[action_rows][:, None], self.grids[action_rows + 1][:, None]


def grid_centroid(grid):
    coordinates = (torch.arange(24, device=grid.device, dtype=grid.dtype) + .5) / 24
    mass = grid.sum((-2, -1)).clamp_min(1e-8)
    return torch.stack(((grid.sum(-2) * coordinates).sum(-1) / mass,
                        (grid.sum(-1) * coordinates).sum(-1) / mass), dim=-1)


@torch.inference_mode()
def evaluate(model: ObjectGridDynamics, data: ObjectData):
    """Final-only held-out prediction metrics on identical eligible windows."""
    model.eval()
    buckets = {k: [] for k in ("observed_actions", "mean_action_inference_ablation", "persistence")}
    for start in range(0, len(data.val), PLAN["batch_size"]):
        z, state, actions, target = data.batch(data.val[start:start + PLAN["batch_size"]])
        future = target[:, 0]
        target_cost = dice_cost(future[:, -1], data.goal)
        targets_area = future[:, -1].sum((-2, -1))
        predictions = {"observed_actions": model(z, state, actions)[:, 0],
            "mean_action_inference_ablation": model(z, state, model.action_mean.expand_as(actions))[:, 0],
            "persistence": z[:, None].expand_as(future)}
        for name, p in predictions.items():
            scores = dice_cost(p, future)
            metrics = torch.stack((scores.mean(1), (p - future).square().mean((1, 2, 3)), scores[:, -1],
                (dice_cost(p[:, -1], data.goal) - target_cost).abs(),
                (grid_centroid(p[:, -1]) - grid_centroid(future[:, -1])).abs().mean(1),
                (p[:, -1].sum((-2, -1)) - targets_area).abs() / targets_area), dim=1)
            buckets[name].append(metrics.cpu())
    episodes = data.episodes[data.val]
    names, result = PLAN["heldout_metrics"], {}
    for name, values in buckets.items():
        measured = torch.cat(values).numpy()
        per_episode = {str(int(ep)): dict(zip(names, measured[episodes == ep].mean(0).tolist())) for ep in np.unique(episodes)}
        result[name] = {"window_mean": dict(zip(names, measured.mean(0).tolist())), "per_episode": per_episode,
                       "episode_macro": {k: float(np.mean([r[k] for r in per_episode.values()])) for k in names}}
    return {"metrics": result, "eligible_windows": len(data.val), "eligible_episodes": len(np.unique(episodes)),
            "selection_condition": "all9_visible_frames_complete_episode_windows", "all_source_episodes": 20,
            "validation_for_checkpoint_selection": False, "candidate_ranking_evaluated": False,
            "policy_benefit_evaluated": False, "action_ablation_separately_trained": False}


def load_final(path: Path, *, device: str = "cuda"):
    """Load only this project's fixed final checkpoint, not old ResNet dynamics."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["schema"] != SCHEMA or payload["kind"] != "final" or payload["step"] != PLAN["steps"] or payload["plan"] != PLAN:
        raise ValueError("requires the fixed final object-grid checkpoint")
    model = ObjectGridDynamics(payload["cache_identity"]["normalization"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)
