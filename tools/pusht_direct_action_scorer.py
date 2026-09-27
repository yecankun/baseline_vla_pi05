"""DS0: directly predict the observed eight-step visual-goal improvement.

Separate from ACT and the object dynamics models. The fixed goal is a training
image, not a future observation. No environment, privileged state, or execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from . import pusht_object_dynamics as base
    from . import pusht_object_goal as vision
    from .pusht_world_model_adapter import CandidateSet, HORIZON
else:
    import pusht_object_dynamics as base
    import pusht_object_goal as vision
    from pusht_world_model_adapter import CandidateSet, HORIZON

SCHEMA = "pusht_direct_action_scorer_v1"
TRAIN_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_direct_action_scorer_v1")
TARGET_EPS = 1e-7
PLAN = {
    "schema": SCHEMA, "variant": "DS0", "space_id": vision.SPACE_ID,
    **{k: base.PLAN[k] for k in ("seed", "steps", "batch_size", "lr", "weight_decay",
                               "grad_clip", "checkpoint_every", "log_every")},
    "horizon": HORIZON,
    "architecture": "shared_grid_576_128_LayerNorm_SiLU_then_concat274_MLP128_64_1",
    "initialization": "fresh_random_no_pretrained_encoder",
    "target": "Dice(current,train_goal)-Dice(observed_grid_t_plus8,train_goal)",
    "target_normalization": "population_mean_std_of_all_eligible_training_windows_only",
    "loss": "MSE_of_standardized_observed_effect_only",
    "near_zero_effect_epsilon": TARGET_EPS,
    "sampling": base.PLAN["sampling"], "normalization": base.PLAN["normalization"],
    "candidate_rule": "unchanged_ACT_reference_plus4_ramps_offset8_horizon8_no_clipping",
    "selection": "argmax_first_index_on_exact_ties_reference_if_co_best",
    "checkpoint_selection": "fixed_final10000_only_no_intermediate_validation",
    "development_candidates_used_for_training": False,
    "heldout_source": "existing_reused_development_validation_not_fresh_test",
    "training_seeds": 1, "policy_rollout": False, "dynamics_auxiliary": False,
}


def observed_effect(current, future, goal):
    """Raw target [B,1] for actual actions; future is [B,1,8,24,24]."""
    if future.shape != (len(current), 1, HORIZON, 24, 24):
        raise ValueError("only complete observed single-action-chunk targets are supported")
    return (base.dice_cost(current, goal) - base.dice_cost(future[:, 0, -1], goal))[:, None]


def strata(values):
    return {"negative": values < -TARGET_EPS, "near_zero": np.abs(values) <= TARGET_EPS,
            "positive": values > TARGET_EPS}


@torch.no_grad()
def training_target_stats(data: base.ObjectData):
    goal = data.manifest["goal"]
    index = goal["global_index"]
    if (int(data.episodes[index]), int(data.frames[index]), index) != (1, 117, 278):
        raise ValueError("fixed training goal changed")
    if int(data.episodes[index]) not in data.manifest["split"]["train_episodes"]:
        raise ValueError("goal must belong to training")
    indices = torch.as_tensor(data.train, device=data.grids.device)
    values = (base.dice_cost(data.grids[indices], data.goal)
              - base.dice_cost(data.grids[indices + HORIZON], data.goal)).double()
    mean, std = float(values.mean()), float(values.std(unbiased=False))
    if not bool(torch.isfinite(values).all()) or std <= 1e-8:
        raise ValueError("nonfinite or effectively constant training target; no synthetic negatives")
    raw = values.cpu().numpy()
    return {"mean": mean, "std": std, "population_std": True, "samples": len(raw),
            "source": "eligible_training_windows_only", "min": float(raw.min()), "max": float(raw.max()),
            "strata_counts": {key: int(mask.sum()) for key, mask in strata(raw).items()}}


class DirectActionScorer(nn.Module):
    """Forward returns standardized effect [B,K]; predict_effect returns raw units.

    Goal and normalization are immutable checkpoint buffers. Candidate chunks
    are scored independently, with all eight ordered absolute-XY actions.
    """

    def __init__(self, stats, target_stats, goal):
        super().__init__()
        self.space_id = vision.SPACE_ID
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(name, torch.as_tensor(stats[name], dtype=torch.float32).clone())
        self.register_buffer("goal", torch.as_tensor(goal, dtype=torch.float32).clone())
        self.register_buffer("target_mean", torch.tensor(target_stats["mean"], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(target_stats["std"], dtype=torch.float32))
        self.encode = nn.Sequential(nn.Linear(576, 128), nn.LayerNorm(128), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(274, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, current, agent_pos, actions):
        b = len(current)
        if current.shape != (b, 24, 24) or agent_pos.shape != (b, 2):
            raise ValueError("current grid/state must be [B,24,24]/[B,2]")
        if actions.ndim != 4 or actions.shape[0] != b or actions.shape[2:] != (HORIZON, 2):
            raise ValueError("native absolute-XY action chunks must be [B,K,8,2]")
        k = actions.shape[1]
        context = torch.cat((self.encode(current.flatten(1)),
                             self.encode(self.goal.flatten()[None]).expand(b, -1),
                             (agent_pos - self.state_mean) / self.state_std), dim=-1)
        ordered = ((actions - self.action_mean) / self.action_std).flatten(2)
        return self.score(torch.cat((context[:, None].expand(-1, k, -1), ordered), dim=-1)).squeeze(-1)

    def predict_effect(self, current, agent_pos, actions):
        return self(current, agent_pos, actions) * self.target_std + self.target_mean

    def loss(self, standardized_prediction, raw_target):
        return (standardized_prediction - (raw_target - self.target_mean) / self.target_std).square().mean()


@dataclass(frozen=True)
class ScorerAdvice:
    scores: np.ndarray  # [B,5], higher better; invalid candidates -inf, unavailable rows NaN
    selected_index: np.ndarray
    selected_chunk: torch.Tensor  # [B,8,2], original native values, no execution
    eligible_observation: np.ndarray


def select_candidates(scores, candidates: CandidateSet, eligible):
    """Pure selection; all-equal scores retain candidate0, no safety claim."""
    b = len(scores)
    if scores.shape != (b, 5) or candidates.actions.shape != (b, 5, HORIZON, 2):
        raise ValueError("requires the existing five eight-step candidate interface")
    valid = (torch.isfinite(candidates.actions) & (candidates.actions >= 0)
             & (candidates.actions <= 512)).all((-2, -1))
    if not torch.equal(valid, candidates.valid) or not bool(valid[:, 0].all()):
        raise ValueError("candidate validity must match bounds and preserve ACT; never clip")
    eligible = np.asarray(eligible)
    if eligible.shape != (b,) or eligible.dtype != bool:
        raise ValueError("one explicit observation-valid flag per context is required")
    allowed = valid.cpu().numpy() & eligible[:, None]
    if not np.isfinite(scores[allowed]).all():
        raise ValueError("nonfinite eligible score")
    scores = np.array(scores, dtype=np.float64, copy=True)
    scores[~valid.cpu().numpy()] = -np.inf
    scores[~eligible] = np.nan
    selected = np.zeros(b, dtype=np.int64)
    selected[eligible] = scores[eligible].argmax(1)
    chosen = candidates.actions[torch.arange(b, device=valid.device),
                                torch.as_tensor(selected, device=valid.device)].clone()
    return ScorerAdvice(scores, selected, chosen, eligible.copy())


@torch.inference_mode()
def predict_and_score(model, current: vision.ObjectGridFeatures, current_valid, agent_pos, candidates):
    if model.training or current.space_id != model.space_id:
        raise ValueError("requires eval mode and the same RGB-object representation")
    flags = np.asarray(current_valid)
    b = len(current.values)
    if flags.shape != (b,) or flags.dtype != bool or current.values.shape != (b, 24, 24):
        raise ValueError("explicit current grid and validity are required")
    if not np.isfinite(current.values[flags]).all() or not bool(torch.isfinite(agent_pos).all()):
        raise ValueError("nonfinite observable input")
    scores = np.full((b, 5), np.nan, dtype=np.float64)
    rows, columns = torch.where(candidates.valid & torch.as_tensor(flags, device=candidates.valid.device)[:, None])
    if len(rows):
        ids = rows.cpu().numpy()
        z = torch.as_tensor(current.values[ids], device=model.goal.device)
        values = model.predict_effect(z, agent_pos[rows], candidates.actions[rows, columns][:, None])[:, 0]
        scores[ids, columns.cpu().numpy()] = values.cpu().numpy()
    return select_candidates(scores, candidates, flags)


def summarize_effects(prediction, target, episodes):
    """Window and episode-macro error, with near-zero imbalance explicit."""
    prediction, target, episodes = (np.asarray(x) for x in (prediction, target, episodes))
    if not (prediction.shape == target.shape == episodes.shape) or not len(target):
        raise ValueError("aligned nonempty prediction/target/episode rows required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite effect metrics")

    def metrics(mask):
        error = prediction[mask] - target[mask]
        return {"effect_mae": float(np.abs(error).mean()), "effect_rmse": float(np.sqrt(np.square(error).mean())),
                "effect_bias": float(error.mean())}

    result = {}
    for name, mask in {"all": np.ones(len(target), dtype=bool), **strata(target)}.items():
        per_ep = {str(int(ep)): metrics(mask & (episodes == ep)) for ep in np.unique(episodes[mask])}
        result[name] = {"windows": int(mask.sum()), "episodes": len(per_ep),
                        "window_mean": metrics(mask) if mask.any() else None, "per_episode": per_ep,
                        "episode_macro": {key: float(np.mean([v[key] for v in per_ep.values()]))
                                          for key in ("effect_mae", "effect_rmse", "effect_bias")} if per_ep else None}
    return result


@torch.inference_mode()
def evaluate(model, data):
    model.eval()
    predicted, actual = [], []
    for start in range(0, len(data.val), PLAN["batch_size"]):
        current, state, actions, future = data.batch(data.val[start:start + PLAN["batch_size"]])
        predicted.append(model.predict_effect(current, state, actions)[:, 0].cpu().numpy())
        actual.append(observed_effect(current, future, data.goal)[:, 0].cpu().numpy())
    p, y = np.concatenate(predicted), np.concatenate(actual)
    references = {"direct_scorer": p, "zero_effect_persistence": np.zeros_like(y),
                  "training_mean_effect": np.full_like(y, float(model.target_mean))}
    return {"metrics": {name: summarize_effects(values, y, data.episodes[data.val]) for name, values in references.items()},
            "eligible_windows": len(y), "eligible_episodes": len(np.unique(data.episodes[data.val])),
            "raw_effect_units": "quadratic_Dice_cost_reduction_at_t_plus8",
            "validation_for_checkpoint_selection": False, "fresh_test": False,
            "candidate_ranking_evaluated": False, "policy_benefit_evaluated": False}


def load_final(path, *, device="cuda"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if (payload["schema"] != SCHEMA or payload["plan"] != PLAN
            or payload["kind"] != "final" or payload["step"] != PLAN["steps"]):
        raise ValueError("requires DS0 fixed-final checkpoint")
    model = DirectActionScorer(payload["cache_identity"]["normalization"], payload["target_stats"],
                               payload["model"]["goal"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)
