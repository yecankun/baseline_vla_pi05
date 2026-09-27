"""ACT-reference-preserving goal-cost gate; no training or environment access.

The threshold is an IN-SAMPLE training residual scale, not a confidence bound
for counterfactual actions. Existing greedy scoring and checkpoints stay intact.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import pusht_bc_act_models as models
    from . import pusht_visual_dynamics as wm
    from .pusht_world_model_adapter import CandidateScores, CandidateSet, VisualFeatures, score_visual_goal
else:
    import pusht_bc_act_models as models
    import pusht_visual_dynamics as wm
    from pusht_world_model_adapter import CandidateScores, CandidateSet, VisualFeatures, score_visual_goal

PLAN = {
    "schema": "pusht_act_wm_train_residual_gate_v1",
    "score": "terminal_raw_visual_feature_MSE_uniform_patches",
    "residual": "abs(predicted_terminal_goal_cost_minus_recorded_terminal_goal_cost)",
    "residual_source": "all22000_training_windows_observed_actions_in_sample",
    "quantile": 0.95, "quantile_method": "higher", "multiplier": 2.0,
    "threshold_rule": "2_times_training_window_residual_q95",
    "acceptance": "nonreference_argmin_and_reference_minus_best_cost_strictly_above_threshold",
    "horizon": 8, "offset_xy": 8.0, "act_chunk_size": 16,
    "act_checkpoint": "fixed_final100000", "wm_checkpoint": "fixed_final10000",
    "goal": "train_episode1_frame117_index278", "new_optimizer_steps": 0,
    "independent_calibration": False, "counterfactual_coverage_guarantee": False,
    "validation_or_test_used_to_fit_threshold": False,
    "diagnostic_rule": "middle_full_window_of_each_training_episode_by_sorted_ID",
}


@dataclass(frozen=True)
class GatedScores:
    scores: CandidateScores
    greedy_index: torch.Tensor
    predicted_gain: torch.Tensor
    accepted: torch.Tensor


@torch.inference_mode()
def retain_reference(candidates: CandidateSet, scored: CandidateScores, threshold: float) -> GatedScores:
    """Apply one scalar margin to validated visual scores, preserving native chunks.

    No clipping, interpolation, oracle input, or replacement objective. Invalid
    predictions still fail in score_visual_goal; this gate is not an error fallback.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative cost scale")
    rows = torch.arange(scored.costs.shape[0], device=scored.costs.device)
    best = scored.costs.argmin(dim=1)
    gain = scored.costs[:, 0] - scored.costs[rows, best]
    accepted = (best != 0) & (gain > threshold)
    selected = torch.where(accepted, best, 0)
    result = CandidateScores(scored.costs, selected, candidates.actions[rows, selected].clone())
    return GatedScores(result, best, gain, accepted)


class ReferencePreservingSelector:
    """One-observation execute8 queue, using the same frozen ACT chunk16 prior."""

    def __init__(self, prior, world_model, goal: VisualFeatures, threshold: float):
        if prior.space_id != goal.space_id or prior.space_id != world_model.space_id:
            raise ValueError("ACT, dynamics and goal feature spaces differ")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid residual threshold")
        self.prior, self.world_model, self.goal, self.threshold = prior, world_model, goal, threshold
        self.reset()

    def reset(self):
        self.prior.policy.reset()
        self.queue, self.reference_queue = deque(), deque()
        self.action_calls, self.decisions = 0, 0
        self.last_decision = self.last_trace = None

    @torch.inference_mode()
    def select_action(self, observation):
        image, _state = models.validate_observation(observation)
        if image.shape[0] != 1:
            raise ValueError("the episode-scoped executor requires batch size one")
        before = len(self.queue)
        if before != (-self.action_calls) % PLAN["horizon"]:
            raise ValueError("execute8 queue/reset schedule drift")
        self.last_decision = None
        if not self.queue:
            request = self.prior.prepare(observation, offset_xy=PLAN["offset_xy"])
            candidates = request["candidates"]
            prediction = self.world_model.predict_candidates(request["current_visual"], request["agent_pos"], candidates)
            greedy = score_visual_goal(candidates, prediction, self.goal)
            gated = retain_reference(candidates, greedy, self.threshold)
            chosen, reference = gated.scores.selected_chunk, candidates.actions[:, 0]
            self.queue.extend(chosen[:, i].clone() for i in range(PLAN["horizon"]))
            self.reference_queue.extend(reference[:, i].clone() for i in range(PLAN["horizon"]))
            self.decisions += 1
            self.selected_index = int(gated.scores.selected_index.item())
            self.last_decision = {
                "decision": self.decisions, "selected_index": self.selected_index,
                "greedy_index": int(gated.greedy_index.item()), "accepted": bool(gated.accepted.item()),
                "predicted_gain": float(gated.predicted_gain.item()), "threshold": self.threshold,
                "reason": "margin_exceeds_training_residual_scale" if bool(gated.accepted.item()) else "retain_ACT",
                "valid": candidates.valid[0].tolist(),
                "costs": [float(x) if math.isfinite(x) else None for x in greedy.costs[0].tolist()],
                "reference_chunk": reference[0].tolist(), "selected_chunk": chosen[0].tolist(),
            }
        action, reference = self.queue.popleft(), self.reference_queue.popleft()
        if self.prior.policy.action_calls != 0 or len(self.prior.policy.policy._action_queue) != 0:
            raise ValueError("selector consumed the original ACT queue")
        self.last_trace = {"before": before, "after": len(self.queue), "chunk_generated": before == 0,
            "selected_index": self.selected_index, "reference_action": reference[0].tolist(),
            "max_abs_reference_offset": float((action - reference).abs().max())}
        self.action_calls += 1
        return action


def load_selector(prior, calibration_path: Path) -> ReferencePreservingSelector:
    """Load the completed gate with the unchanged goal cache/final dynamics only.

    Recalibration is required after changing the model, goal, horizon or score.
    Caller owns/reset the returned queue; this function never opens an environment.
    """
    artifact = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    manifest = json.loads((wm.CACHE_ROOT / "manifest.json").read_text())
    if artifact["status"] != "completed_training_residual_calibration" or artifact["plan"] != PLAN:
        raise ValueError("incompatible or incomplete gate artifact")
    keys = ("source_revision", "split_sha256", "feature_space_id", "goal")
    if artifact["source_identity"] != {key: manifest[key] for key in keys}:
        raise ValueError("gate source/goal provenance differs")
    if artifact["wm_checkpoint"] != str(wm.TRAIN_ROOT / "final.pt"):
        raise ValueError("gate is bound to the original final world model")
    threshold = artifact["threshold"]
    if threshold != PLAN["multiplier"] * artifact["residual_q95"]:
        raise ValueError("gate threshold differs from the fixed training-residual rule")
    model = wm.load_final(wm.TRAIN_ROOT / "final.pt", device=prior.policy.device)
    cached = np.load(wm.CACHE_ROOT / "features.npy", mmap_mode="r", allow_pickle=False)
    goal = VisualFeatures(torch.from_numpy(cached[manifest["goal"]["global_index"]].copy())[None].to(prior.policy.device),
                          manifest["feature_space_id"])
    return ReferencePreservingSelector(prior, model, goal, threshold)
