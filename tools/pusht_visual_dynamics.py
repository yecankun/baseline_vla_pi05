"""Small native Push-T visual dynamics in the frozen ACT spatial feature space.

Not DINO-WM, not a guidewire policy, and not an environment/action executor.
The only inputs are current image features, observable agent XY and native XY
action targets. Future frames are supervision, never conditioning inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from .pusht_world_model_adapter import CandidateSet, HORIZON, VisualFeatures
else:
    from pusht_world_model_adapter import CandidateSet, HORIZON, VisualFeatures

SCHEMA = "pusht_act_spatial_dynamics_v1"
CACHE_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_act_wm_v1")
TRAIN_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_act_wm_v1")
PLAN = {"schema": SCHEMA, "seed": 20260915, "steps": 10000, "batch_size": 64,
        "hidden_dim": 128, "lr": 0.0003, "weight_decay": 0.0001, "grad_clip": 1.0,
        "checkpoint_every": 1000, "log_every": 100, "horizon": HORIZON,
        "loss": "mean_squared_train_normalized_visual_error_all_8_steps",
        "checkpoint_selection": "fixed_final_step_only", "frame_skip": 1,
        "goal_rule": "last_recorded_RGB_of_lowest_ID_training_episode",
        "goal_used_in_training_loss": False, "policy_rollout_in_this_run": False}


def temporal_anchors(episodes: np.ndarray, split: dict) -> tuple[np.ndarray, np.ndarray]:
    """Full windows only: actions[t:t+8] supervise images[t+1:t+9]."""
    anchor = np.arange(len(episodes) - HORIZON, dtype=np.int64)
    # Production source loader already verifies contiguous, complete episode blocks.
    anchor = anchor[episodes[anchor] == episodes[anchor + HORIZON]]
    train = anchor[np.isin(episodes[anchor], split["train_episodes"])]
    val = anchor[np.isin(episodes[anchor], split["val_episodes"])]
    if not len(train) or not len(val) or np.intersect1d(train, val).size:
        raise ValueError("empty or overlapping temporal split")
    return train, val


class SpatialDynamics(nn.Module):
    """Shared spatial3x3 mixing + action-driven per-patch GRU, hidden128.

    Each predicted frame sees only its action prefix. Predict a residual from
    the current feature grid; no teacher-forced future input, pixel decoder,
    learned success/contact head or policy fine-tuning.
    """

    def __init__(self, stats: dict, space_id: str, hidden_dim: int = 128):
        super().__init__()
        self.space_id, self.hidden_dim = space_id, hidden_dim
        for name in ("visual_mean", "visual_std", "state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(name, torch.tensor(stats[name], dtype=torch.float32))
        self.encode = nn.Linear(512, hidden_dim)
        self.state_embed = nn.Linear(2, hidden_dim)
        self.action_embed = nn.Linear(2, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, 9, hidden_dim))
        self.spatial_mix = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.decode = nn.Linear(hidden_dim, 512)
        nn.init.normal_(self.decode.weight, std=0.001)
        nn.init.zeros_(self.decode.bias)

    def forward(self, current: torch.Tensor, agent_pos: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        b = current.shape[0]
        if current.shape != (b, 9, 512) or agent_pos.shape != (b, 2):
            raise ValueError("current visual/agent_pos must be [B,9,512] / [B,2]")
        if actions.ndim != 4 or actions.shape[0] != b or actions.shape[2:] != (HORIZON, 2):
            raise ValueError("actions must be absolute native targets [B,K,8,2]")
        k = actions.shape[1]
        z = (current - self.visual_mean) / self.visual_std
        initial = self.encode(z) + self.position + self.state_embed(
            (agent_pos - self.state_mean) / self.state_std)[:, None]
        hidden = initial[:, None].expand(-1, k, -1, -1).reshape(b * k, 9, self.hidden_dim)
        normalized_actions = ((actions - self.action_mean) / self.action_std).reshape(b * k, HORIZON, 2)
        output = []
        for t in range(HORIZON):
            mixed = self.spatial_mix(hidden.transpose(1, 2).reshape(b * k, self.hidden_dim, 3, 3))
            mixed = mixed.flatten(2).transpose(1, 2) + self.action_embed(normalized_actions[:, t])[:, None]
            hidden = self.cell(mixed.reshape(-1, self.hidden_dim), hidden.reshape(-1, self.hidden_dim))
            hidden = hidden.reshape(b * k, 9, self.hidden_dim)
            predicted = z[:, None] + self.decode(hidden).reshape(b, k, 9, 512)
            output.append(predicted * self.visual_std + self.visual_mean)
        return torch.stack(output, dim=2)

    @torch.inference_mode()
    def predict(self, current: VisualFeatures, agent_pos: torch.Tensor, actions: torch.Tensor) -> VisualFeatures:
        if self.training or current.space_id != self.space_id:
            raise ValueError("prediction requires eval mode and the matching frozen visual space")
        if not bool(torch.isfinite(actions).all()) or bool(((actions < 0) | (actions > 512)).any()):
            raise ValueError("predict only valid native action candidates; no clipping")
        return VisualFeatures(self(current.values, agent_pos, actions), self.space_id)

    @torch.inference_mode()
    def predict_candidates(self, current: VisualFeatures, agent_pos: torch.Tensor, candidates: CandidateSet) -> VisualFeatures:
        """Predict valid candidates only; leave rejected chunks NaN for the scorer."""
        batch, candidate = torch.where(candidates.valid)
        if not len(batch) or not bool(candidates.valid[:, 0].all()):
            raise ValueError("candidate set must retain every batch's ACT reference")
        predicted = self.predict(VisualFeatures(current.values[batch], current.space_id),
                                 agent_pos[batch], candidates.actions[batch, candidate][:, None]).values[:, 0]
        output = predicted.new_full((*candidates.valid.shape, HORIZON, 9, 512), torch.nan)
        output[batch, candidate] = predicted
        return VisualFeatures(output, self.space_id)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((predicted - target) / self.visual_std).square().mean()


class FeatureData:
    """Load only a completed local feature pack; metadata stays out of forward()."""

    def __init__(self, root: Path = CACHE_ROOT, device: str = "cuda"):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        m = self.manifest
        if m["schema"] != SCHEMA or m["status"] != "prepared" or m["plan"] != PLAN:
            raise ValueError("incompatible or incomplete feature pack")
        self.stats, self.space_id = m["normalization"], m["feature_space_id"]
        self.features = torch.from_numpy(np.load(self.root / "features.npy", allow_pickle=False)).to(device)
        with np.load(self.root / "arrays.npz", allow_pickle=False) as arrays:
            self.states = torch.from_numpy(arrays["states"]).to(device)
            self.actions = torch.from_numpy(arrays["actions"]).to(device)
            self.episodes = arrays["episodes"]
            self.frames = arrays["frames"]
        self.train, self.val = temporal_anchors(self.episodes, m["split"])
        if list(self.features.shape) != m["feature_shape"] or self.features.shape != (len(self.episodes), 9, 512):
            raise ValueError("feature cache shape mismatch")
        if len(self.train) != m["train_windows"] or len(self.val) != m["val_windows"]:
            raise ValueError("temporal window counts changed")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("nonfinite feature cache")
        self.goal = self.features[m["goal"]["global_index"]].clone()
        self.identity = {key: m[key] for key in ("schema", "source_revision", "split_sha256", "feature_space_id", "goal", "normalization")}

    def batch(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.features.device)
        future = indices[:, None] + torch.arange(HORIZON, device=indices.device)[None]
        return (self.features[indices], self.states[indices], self.actions[future][:, None],
                self.features[future + 1][:, None])


@torch.inference_mode()
def evaluate(model: SpatialDynamics, data: FeatureData, batch_size: int = 64) -> dict:
    """Fixed held-out windows, final-only: predictor vs persistence/action ablation.

    Mean-action ablation is inference-only, not a separately trained baseline.
    Aggregate windows and episode macros explicitly; no rollout success claim.
    """
    model.eval()
    buckets = {name: [] for name in ("observed_actions", "mean_action_inference_ablation", "persistence")}
    for start in range(0, len(data.val), batch_size):
        z, state, actions, target = data.batch(data.val[start:start + batch_size])
        predictions = {"observed_actions": model(z, state, actions),
                       "mean_action_inference_ablation": model(z, state, model.action_mean.expand_as(actions)),
                       "persistence": z[:, None, None].expand_as(target)}
        target_goal_cost = (target[:, 0, -1] - data.goal).square().mean((-2, -1))
        for name, prediction in predictions.items():
            error = prediction[:, 0] - target[:, 0]
            measures = torch.stack((
                (error / model.visual_std).square().mean((1, 2, 3)),
                error[:, -1].square().mean((1, 2)),
                error.abs().mean((1, 2, 3)),
                ((prediction[:, 0, -1] - data.goal).square().mean((1, 2)) - target_goal_cost).abs(),
            ), dim=1)
            buckets[name].append(measures.cpu())
    names = ("normalized_feature_mse_all8", "terminal_raw_feature_mse", "raw_feature_mae_all8", "terminal_goal_cost_mae")
    result = {}
    episodes = data.episodes[data.val]
    for name, values in buckets.items():
        measured = torch.cat(values).numpy()
        per_episode = {str(int(ep)): dict(zip(names, measured[episodes == ep].mean(axis=0).tolist())) for ep in np.unique(episodes)}
        result[name] = {"window_mean": dict(zip(names, measured.mean(axis=0).tolist())), "per_episode": per_episode,
                        "episode_macro": {key: float(np.mean([row[key] for row in per_episode.values()])) for key in names}}
    return {"windows": len(data.val), "episodes": len(np.unique(episodes)), "metrics": result,
            "validation_used_for_checkpoint_selection": False, "policy_benefit_evaluated": False,
            "mean_action_ablation_is_separately_trained": False}


def load_final(path: Path, *, device: str = "cuda") -> SpatialDynamics:
    """Read our own completed training checkpoint, never an external pickle."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["plan"] != PLAN or payload["step"] != PLAN["steps"] or payload["kind"] != "final":
        raise ValueError("not the fixed final visual-dynamics checkpoint")
    model = SpatialDynamics(payload["cache_identity"]["normalization"], payload["cache_identity"]["feature_space_id"], PLAN["hidden_dim"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)
