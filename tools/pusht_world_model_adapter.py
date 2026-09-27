"""Frozen ACT prior and visual-goal scoring for a future native Push-T world model.

This is NOT a trained dynamics model or a rollout policy. Actions retain native
absolute XY coordinates (0..512). No LIBERO/guidewire padding, oracle state,
automatic goal selection, optimizer, action clipping, or environment stepping.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

if __package__:
    from . import pusht_bc_act_models as models
else:
    import pusht_bc_act_models as models

SCHEMA = "pusht_act_visual_goal_interface_v1"
HORIZON = 8
LABELS = ("act_reference", "ramp_x_plus", "ramp_x_minus", "ramp_y_plus", "ramp_y_minus")


@dataclass(frozen=True)
class VisualFeatures:
    # Current/goal: [B,P,D]; predicted post-action frames: [B,K,8,P,D].
    values: torch.Tensor
    space_id: str


@dataclass(frozen=True)
class CandidateSet:
    actions: torch.Tensor  # [B,5,8,2], unnormalized absolute native targets
    valid: torch.Tensor  # [B,5]; reject a whole out-of-bounds chunk
    offset_xy: float
    labels: tuple[str, ...] = LABELS


@dataclass(frozen=True)
class CandidateScores:
    costs: torch.Tensor  # [B,K], invalid candidates have +inf
    selected_index: torch.Tensor  # [B], ties choose reference/lowest index
    selected_chunk: torch.Tensor  # [B,8,2], NOT automatically executed

    @property
    def first_action(self) -> torch.Tensor:
        return self.selected_chunk[:, 0]


def make_candidates(reference: torch.Tensor, *, offset_xy: float) -> CandidateSet:
    """Five deterministic candidates, including the exact ACT first-eight chunk.

    Alternative k at step h adds (h+1)/8 * direction[k] * offset_xy.
    offset_xy is a declared interface parameter, not a validated/tuned amplitude.
    """
    if reference.ndim != 3 or reference.shape[1:] != (HORIZON, 2) or reference.shape[0] < 1:
        raise ValueError("reference must be [B,8,2]")
    if not torch.is_floating_point(reference) or not bool(torch.isfinite(reference).all()):
        raise ValueError("reference must be finite floating native actions")
    if bool(((reference < 0) | (reference > 512)).any()):
        raise ValueError("ACT reference is out of bounds; no clipping or fallback")
    if not math.isfinite(offset_xy) or offset_xy <= 0:
        raise ValueError("offset_xy must be a positive finite native-coordinate offset")
    direction = reference.new_tensor(((1, 0), (-1, 0), (0, 1), (0, -1)))
    ramp = torch.arange(1, HORIZON + 1, device=reference.device, dtype=reference.dtype) / HORIZON
    actions = reference[:, None].repeat(1, len(LABELS), 1, 1)
    actions[:, 1:] += offset_xy * direction[None, :, None, :] * ramp[None, None, :, None]
    valid = (torch.isfinite(actions) & (actions >= 0) & (actions <= 512)).all(dim=(-2, -1))
    return CandidateSet(actions, valid, offset_xy)


class FrozenACTPrior:
    """Use an already-audited ACT, without consuming/resetting its action queue.

    The frozen ResNet spatial map is a provisional interface feature space,
    NOT DINOv2 and NOT evidence of goal quality or learned visual dynamics.
    """

    def __init__(self, policy, binding: dict):
        config = policy.policy.config
        if policy.name != "act" or config.chunk_size != 16 or config.n_action_steps != HORIZON:
            raise ValueError("requires the frozen native ACT chunk16/execute8 prior")
        if config.temporal_ensemble_coeff is not None:
            raise ValueError("this prior does not support temporal ensembling")
        if any(p.requires_grad for p in policy.policy.parameters()) or policy.policy.training:
            raise ValueError("load the frozen inference policy, not a trainable ACT")
        self.policy = policy
        self.space_id = (
            f"act/{binding['model_state_sha256']}/resnet18_feature_map/imagenet96_rowmajor_v1"
        )

    @torch.inference_mode()
    def candidates(self, observation: dict, *, offset_xy: float) -> CandidateSet:
        normalized = self.policy.pre(models.act_batch(observation))
        chunk = self.policy.policy.predict_action_chunk(normalized)
        if chunk.shape != (observation[models.IMAGE].shape[0], 16, 2):
            raise ValueError("ACT chunk shape changed")
        # Use precisely the same postprocessor and per-action shape as the baseline queue.
        reference = torch.stack([
            self.policy.post(chunk[:, i]).to(self.policy.device) for i in range(HORIZON)
        ], dim=1)
        return make_candidates(reference, offset_xy=offset_xy)

    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor) -> VisualFeatures:
        # Image-only path: no fake goal agent state or environment truth is supplied.
        normalized = models.imagenet_normalize(image)
        feature_map = self.policy.policy.model.backbone(normalized)["feature_map"]
        return VisualFeatures(feature_map.flatten(2).transpose(1, 2).contiguous(), self.space_id)

    @torch.inference_mode()
    def prepare(self, observation: dict, *, offset_xy: float) -> dict:
        """World-model input: current visual tokens, raw observable XY, candidates.

        A future predictor must normalize state/actions with its own train-only
        stats, and predict after EACH of the eight native control steps. Neither
        goal RGB nor scoring/role metadata is silently made a policy observation.
        """
        image, state = models.validate_observation(observation)
        return {"current_visual": self.encode_image(image), "agent_pos": state.clone(),
                "candidates": self.candidates(observation, offset_xy=offset_xy)}


@torch.inference_mode()
def score_visual_goal(
    candidates: CandidateSet,
    predicted: VisualFeatures,
    goal: VisualFeatures,
    *,
    goal_patch_weights: torch.Tensor | None = None,
) -> CandidateScores:
    """Terminal spatial MSE, optionally weighted by a GOAL-ONLY patch mask.

    The caller supplies a goal RGB encoded in exactly the prediction space; no
    validation/future-frame goal retrieval occurs here. Patch weights must be
    declared from that goal image, not object pose/contact/segmentation truth.
    No model is supplied by this module: fabricated predictions are test fixtures
    only, never a fallback for a missing trained predictor.
    """
    actions, valid = candidates.actions, candidates.valid
    future, target = predicted.values, goal.values
    if not predicted.space_id or predicted.space_id != goal.space_id:
        raise ValueError("prediction and goal feature spaces must match")
    if actions.ndim != 4 or actions.shape[1:] != (len(LABELS), HORIZON, 2):
        raise ValueError("candidate actions must be [B,5,8,2]")
    b, k, h, _ = actions.shape
    if valid.shape != (b, k) or valid.dtype != torch.bool or not bool(valid[:, 0].all()):
        raise ValueError("valid [B,5] mask must retain the ACT reference")
    in_bounds = (torch.isfinite(actions) & (actions >= 0) & (actions <= 512)).all(dim=(-2, -1))
    if not torch.equal(valid, in_bounds):
        raise ValueError("candidate validity mask must match native bounds")
    if target.ndim != 3 or target.shape[0] != b or min(target.shape[1:]) < 1:
        raise ValueError("goal features must be [B,P,D]")
    if future.shape != (b, k, h, *target.shape[1:]):
        raise ValueError("predictions must be [B,5,8,P,D], post-action native steps")
    if any(x.device != actions.device or x.dtype != actions.dtype for x in (future, target)):
        raise ValueError("actions, predictions and goal must share floating dtype/device")
    if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(future[valid]).all()):
        raise ValueError("goal and valid-candidate predictions must be finite")
    patch_mse = (future[:, :, -1] - target[:, None]).square().mean(dim=-1)
    if goal_patch_weights is None:
        costs = patch_mse.mean(dim=-1)
    else:
        w = goal_patch_weights
        if w.shape != target.shape[:2] or w.device != target.device or w.dtype != target.dtype:
            raise ValueError("goal_patch_weights must be floating [B,P] on the feature device")
        if not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or bool((w.sum(-1) <= 0).any()):
            raise ValueError("goal patch weights must be finite, nonnegative, and nonempty")
        costs = (patch_mse * w[:, None]).sum(-1) / w.sum(-1, keepdim=True)
    costs = costs.masked_fill(~valid, torch.inf)
    if not bool(torch.isfinite(costs[valid]).all()):
        raise ValueError("visual cost overflow on a valid candidate")
    selected = costs.argmin(dim=1)  # torch argmin returns the first index on ties.
    chunk = actions[torch.arange(b, device=actions.device), selected].clone()
    return CandidateScores(costs, selected, chunk)
