from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


MODEL_VERSION = "pi05_action_effect_world_model_v1"
TASK_TO_ID = {"left": 0, "right": 1}
GUIDANCE_EFFECT_TO_ID = {"ineffective_or_unknown": 0, "left": 1, "right": 2}
ACTIVE_ACTION_DIM = 9

FORBIDDEN_POLICY_KEYS = {
    "contact_flag",
    "contact_normal",
    "contact_strength",
    "diagnostic_targets",
    "distance_to_wall",
    "estimated_contact_flag",
    "estimated_contact_risk_flag",
    "estimated_contact_risk_probability",
    "heading",
    "lateral_offset",
    "path_progress",
    "piper_event_id",
    "route_progress",
    "sample_role",
    "segment_min_distance_to_wall",
    "tip_pos",
    "training_weight",
}


@dataclass(frozen=True)
class ActionEffectWorldModelConfig:
    visual_dim: int = 256
    state_dim: int = 32
    hidden_dim: int = 256
    context_len: int = 4
    horizon: int = 3
    action_dim: int = ACTIVE_ACTION_DIM
    task_count: int = 2
    guidance_class_count: int = 3
    branch_class_count: int = 2
    dropout: float = 0.1

    def validate(self) -> None:
        for name in (
            "visual_dim",
            "state_dim",
            "hidden_dim",
            "context_len",
            "horizon",
            "action_dim",
            "task_count",
            "guidance_class_count",
            "branch_class_count",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.action_dim != ACTIVE_ACTION_DIM:
            raise ValueError(
                f"action_dim must preserve six Elite deltas plus three Piper intents, got {self.action_dim}"
            )
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class CandidateScoreConfig:
    branch_weight: float = 1.0
    guidance_weight: float = 0.75
    uncertainty_weight: float = 0.2
    invalid_feed_weight: float = 0.25
    elite_motion_weight: float = 0.01


@dataclass(frozen=True)
class ActionEffectLossConfig:
    visual_weight: float = 1.0
    state_weight: float = 0.25
    guidance_weight: float = 0.5
    branch_weight: float = 0.5
    invalid_feed_weight: float = 0.25
    uncertainty_weight: float = 0.1
    degradation_consistency_weight: float = 0.25


@dataclass(frozen=True)
class OxidationCoverageConfig:
    contrast_min: float = 0.45
    contrast_max: float = 0.85
    haze_min: float = 0.05
    haze_max: float = 0.30
    color_gain_min: float = 0.80
    color_gain_max: float = 1.12
    low_frequency_grid: int = 4
    blur_probability: float = 0.5

    def validate(self) -> None:
        if not 0.0 < self.contrast_min <= self.contrast_max <= 1.0:
            raise ValueError("contrast range must be inside (0, 1]")
        if not 0.0 <= self.haze_min <= self.haze_max < 1.0:
            raise ValueError("haze range must be inside [0, 1)")
        if not 0.0 < self.color_gain_min <= self.color_gain_max:
            raise ValueError("color gain range must be positive")
        if self.low_frequency_grid < 2:
            raise ValueError("low_frequency_grid must be at least 2")
        if not 0.0 <= self.blur_probability <= 1.0:
            raise ValueError("blur_probability must be in [0, 1]")


def _nested_key_paths(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            paths.append((key_text, path))
            paths.extend(_nested_key_paths(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_nested_key_paths(child, f"{prefix}[{index}]"))
    return paths


def validate_policy_batch_no_privileged_fields(batch: dict[str, Any]) -> None:
    violations = sorted(
        path
        for key, path in _nested_key_paths(batch)
        if key in FORBIDDEN_POLICY_KEYS or path in FORBIDDEN_POLICY_KEYS
    )
    if violations:
        raise ValueError(f"privileged/offline-only fields entered policy batch: {violations[:8]}")


def encode_active_action(
    elite_tcp_delta_6d: torch.Tensor,
    piper_intent_id: torch.Tensor,
) -> torch.Tensor:
    if elite_tcp_delta_6d.shape[-1] != 6:
        raise ValueError(
            f"elite_tcp_delta_6d must end in six values, got {tuple(elite_tcp_delta_6d.shape)}"
        )
    ids = piper_intent_id.long()
    if tuple(ids.shape) != tuple(elite_tcp_delta_6d.shape[:-1]):
        raise ValueError(
            "piper_intent_id shape must match elite_tcp_delta_6d prefix: "
            f"{tuple(ids.shape)} != {tuple(elite_tcp_delta_6d.shape[:-1])}"
        )
    if torch.any((ids < 0) | (ids > 2)):
        raise ValueError("piper_intent_id must use retract=0, hold=1, feed=2")
    piper_one_hot = F.one_hot(ids, num_classes=3).to(dtype=elite_tcp_delta_6d.dtype)
    return torch.cat([elite_tcp_delta_6d, piper_one_hot], dim=-1)


def decode_active_action(action: torch.Tensor) -> dict[str, torch.Tensor]:
    if action.shape[-1] != ACTIVE_ACTION_DIM:
        raise ValueError(f"active action must be {ACTIVE_ACTION_DIM}D")
    return {
        "elite_tcp_delta_6d": action[..., :6],
        "piper_intent_id": action[..., 6:9].argmax(dim=-1),
    }


def action32_candidates_to_active(action_32: torch.Tensor) -> torch.Tensor:
    """Convert PI0.5 compatibility chunks to the explicit nine active dims."""
    if action_32.ndim not in {3, 4} or action_32.shape[-1] < ACTIVE_ACTION_DIM:
        raise ValueError("PI05 action candidates must be [B, H, D] or [B, K, H, D] with D>=9")
    elite = action_32[..., :6]
    piper_intent = action_32[..., 6:9].argmax(dim=-1)
    return encode_active_action(elite, piper_intent)


@torch.inference_mode()
def extract_pi05_multiview_visual_latent(
    policy: nn.Module,
    images: list[torch.Tensor] | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Pool frozen PI0.5 image tokens into [B, V, D] feature-pack latents.

    Images must already have passed the same PI0.5 preprocessing used by the
    policy. This helper intentionally bypasses language/action tokens and never
    reads simulator diagnostic truth.
    """
    if not images:
        raise ValueError("at least one preprocessed PI05 camera tensor is required")
    model = getattr(policy, "model", None)
    paligemma = getattr(model, "paligemma_with_expert", None)
    embed_image = getattr(paligemma, "embed_image", None)
    if not callable(embed_image):
        raise ValueError("policy does not expose PI05 paligemma_with_expert.embed_image")
    batch = int(images[0].shape[0])
    pooled: list[torch.Tensor] = []
    for view_index, image in enumerate(images):
        if image.ndim != 4 or image.shape[0] != batch or image.shape[1] != 3:
            raise ValueError(
                f"camera {view_index} must be a preprocessed [B, 3, H, W] tensor"
            )
        tokens = embed_image(image)
        if tokens.ndim != 3 or tokens.shape[0] != batch:
            raise ValueError("PI05 image embedding must be [B, patches, D]")
        pooled.append(tokens.float().mean(dim=1))
    return torch.stack(pooled, dim=1)


class OxidationCoverageAugmenter:
    """Coverage-only photometric degradation with no geometric warp.

    The transform is deliberately not presented as a fitted oxidation model.
    It preserves spatial alignment so the clean/degraded pair can share action
    and dynamics targets.
    """

    def __init__(self, config: OxidationCoverageConfig | None = None) -> None:
        self.config = config or OxidationCoverageConfig()
        self.config.validate()

    @staticmethod
    def _uniform(generator: torch.Generator, low: float, high: float) -> float:
        return float(low + (high - low) * torch.rand((), generator=generator).item())

    def __call__(
        self,
        images: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if images.ndim < 4 or images.shape[-3] != 3:
            raise ValueError("images must have shape [..., 3, H, W]")
        if not torch.is_floating_point(images):
            raise ValueError("images must be floating point in [0, 1]")
        if images.numel() and (float(images.min()) < 0.0 or float(images.max()) > 1.0):
            raise ValueError("images must be in [0, 1]")

        original_shape = images.shape
        flat = images.reshape(-1, *images.shape[-3:])
        count, _, height, width = flat.shape
        contrast = self._uniform(
            generator, self.config.contrast_min, self.config.contrast_max
        )
        haze = self._uniform(generator, self.config.haze_min, self.config.haze_max)
        color_gain = torch.tensor(
            [
                self._uniform(
                    generator,
                    self.config.color_gain_min,
                    self.config.color_gain_max,
                )
                for _ in range(3)
            ],
            dtype=flat.dtype,
            device=flat.device,
        ).view(1, 3, 1, 1)

        channel_mean = flat.mean(dim=(-2, -1), keepdim=True)
        degraded = channel_mean + contrast * (flat - channel_mean)
        degraded = degraded * color_gain

        low_frequency = torch.rand(
            count,
            1,
            self.config.low_frequency_grid,
            self.config.low_frequency_grid,
            generator=generator,
            dtype=torch.float32,
        ).to(device=flat.device, dtype=flat.dtype)
        low_frequency = F.interpolate(
            low_frequency,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).clamp_(0.0, 1.0)
        haze_map = haze * (0.35 + 0.65 * low_frequency)
        haze_color = torch.tensor(
            [0.58, 0.52, 0.38], dtype=flat.dtype, device=flat.device
        ).view(1, 3, 1, 1)
        degraded = degraded * (1.0 - haze_map) + haze_color * haze_map

        blur_draw = float(torch.rand((), generator=generator).item())
        blur_applied = blur_draw < self.config.blur_probability
        if blur_applied:
            degraded = F.avg_pool2d(degraded, kernel_size=3, stride=1, padding=1)

        degraded = degraded.clamp(0.0, 1.0).reshape(original_shape)
        metadata = {
            "schema": "project2026_oxidation_coverage_transform_v1",
            "claim": "coverage_only_not_fitted_to_real_oxidation_distribution",
            "geometric_warp": False,
            "contrast": contrast,
            "haze": haze,
            "color_gain": color_gain.flatten().detach().cpu().tolist(),
            "blur_applied": blur_applied,
        }
        return degraded, metadata


class ActionEffectWorldModel(nn.Module):
    """Short-horizon, action-conditioned latent dynamics model.

    PI0.5 remains the action prior. This module consumes pooled visual latents,
    observable state history, task, and explicit candidate action chunks. It
    does not consume raw simulator truth or redefine the shared action output.
    """

    def __init__(self, config: ActionEffectWorldModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, hidden),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(2 * config.state_dim),
            nn.Linear(2 * config.state_dim, hidden),
            nn.GELU(),
        )
        self.task_embedding = nn.Embedding(config.task_count, hidden)
        self.context_fusion = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.context_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(config.action_dim),
            nn.Linear(config.action_dim, hidden),
            nn.GELU(),
        )
        self.action_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.next_visual_head = nn.Linear(hidden, config.visual_dim)
        self.state_delta_head = nn.Linear(hidden, config.state_dim)
        self.guidance_head = nn.Linear(hidden, config.guidance_class_count)
        self.branch_head = nn.Linear(hidden, config.branch_class_count)
        self.invalid_feed_head = nn.Linear(hidden, 1)
        self.effect_log_variance_head = nn.Linear(hidden, 1)

    def forward(
        self,
        *,
        history_visual_latent: torch.Tensor,
        history_state: torch.Tensor,
        task_id: torch.Tensor,
        candidate_actions: torch.Tensor,
        history_visual_valid: torch.Tensor | None = None,
        history_state_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if history_visual_latent.ndim != 4:
            raise ValueError("history_visual_latent must be [B, T, V, D]")
        batch, context, views, visual_dim = history_visual_latent.shape
        if context != cfg.context_len or visual_dim != cfg.visual_dim:
            raise ValueError(
                "history visual shape does not match config: "
                f"{tuple(history_visual_latent.shape)}"
            )
        if tuple(history_state.shape) != (batch, context, cfg.state_dim):
            raise ValueError("history_state must be [B, context_len, state_dim]")
        if history_state_valid is None:
            history_state_valid = torch.ones_like(history_state, dtype=torch.bool)
        elif tuple(history_state_valid.shape) != (batch, context, cfg.state_dim):
            raise ValueError("history_state_valid must be [B, context_len, state_dim]")
        if tuple(task_id.shape) != (batch,):
            raise ValueError("task_id must be [B]")
        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must be [B, K, H, 9]")
        if (
            candidate_actions.shape[0] != batch
            or candidate_actions.shape[2] != cfg.horizon
            or candidate_actions.shape[3] != cfg.action_dim
        ):
            raise ValueError("candidate action shape does not match config")

        visual_encoded = self.visual_encoder(history_visual_latent)
        if history_visual_valid is None:
            visual_context = visual_encoded.mean(dim=2)
        else:
            if tuple(history_visual_valid.shape) != (batch, context, views):
                raise ValueError("history_visual_valid must be [B, T, V]")
            weights = history_visual_valid.to(visual_encoded.dtype).unsqueeze(-1)
            denom = weights.sum(dim=2).clamp_min(1.0)
            visual_context = (visual_encoded * weights).sum(dim=2) / denom
        state_input = torch.cat(
            [
                history_state.masked_fill(~history_state_valid.bool(), 0.0),
                history_state_valid.to(history_state.dtype),
            ],
            dim=-1,
        )
        state_context = self.state_encoder(state_input)
        task_context = self.task_embedding(task_id.long()).unsqueeze(1).expand(-1, context, -1)
        fused = self.context_fusion(
            torch.cat([visual_context, state_context, task_context], dim=-1)
        )
        _, context_hidden = self.context_gru(fused)

        candidate_count = candidate_actions.shape[1]
        action_input = self.action_encoder(candidate_actions).reshape(
            batch * candidate_count, cfg.horizon, cfg.hidden_dim
        )
        initial = (
            context_hidden.transpose(0, 1)
            .unsqueeze(1)
            .expand(-1, candidate_count, -1, -1)
            .reshape(batch * candidate_count, 1, cfg.hidden_dim)
            .transpose(0, 1)
        )
        action_hidden, _ = self.action_gru(action_input, initial.contiguous())
        action_hidden = action_hidden.reshape(
            batch, candidate_count, cfg.horizon, cfg.hidden_dim
        )

        return {
            "context_latent": context_hidden.squeeze(0),
            "pred_next_visual_latent": self.next_visual_head(action_hidden),
            "pred_state_delta": self.state_delta_head(action_hidden),
            "guidance_logits": self.guidance_head(action_hidden),
            "branch_logits": self.branch_head(action_hidden),
            "invalid_feed_logits": self.invalid_feed_head(action_hidden).squeeze(-1),
            "effect_log_variance": self.effect_log_variance_head(action_hidden)
            .squeeze(-1)
            .clamp(-8.0, 8.0),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "version": MODEL_VERSION,
            "config": asdict(self.config),
            "policy_input": [
                "PI05 visual latent history",
                "observable state_32 history",
                "state_32 validity history",
                "task id",
                "candidate elite_tcp_delta_6d + piper_intent_id chunks",
            ],
            "policy_output_unchanged": "elite_tcp_delta_6d + piper_intent_id",
            "numeric_input_contract": "train-only normalized state_32 and Elite action dims; Piper one-hot unchanged",
            "missing_state_contract": "normalized zero plus explicit state_32 validity history",
            "uses_exact_tip_contact_wall_route_as_input": False,
            "world_model_role": "short-horizon action-effect prediction and candidate reranking",
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_value = mask.to(dtype=values.dtype)
    while mask_value.ndim < values.ndim:
        mask_value = mask_value.unsqueeze(-1)
    expanded = mask_value.expand_as(values)
    denom = expanded.sum().clamp_min(1.0)
    return (values * expanded).sum() / denom


def action_effect_world_model_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: ActionEffectLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or ActionEffectLossConfig()
    visual_target = targets["next_visual_latent"]
    state_target = targets["state_delta"]
    if outputs["pred_next_visual_latent"].shape[1] != 1:
        raise ValueError("training loss expects one executed action candidate")
    pred_visual = outputs["pred_next_visual_latent"][:, 0]
    pred_state = outputs["pred_state_delta"][:, 0]
    pred_logvar = outputs["effect_log_variance"][:, 0]
    visual_mask = targets.get("visual_target_valid")
    state_mask = targets.get("state_target_valid")

    visual_element = F.smooth_l1_loss(pred_visual, visual_target, reduction="none")
    state_element = F.smooth_l1_loss(pred_state, state_target, reduction="none")
    visual_loss = _masked_mean(visual_element, visual_mask)
    state_loss = _masked_mean(state_element, state_mask)

    visual_residual = (pred_visual - visual_target).square().mean(dim=-1)
    uncertainty_element = 0.5 * (
        torch.exp(-pred_logvar) * visual_residual.detach() + pred_logvar
    )
    uncertainty_loss = _masked_mean(uncertainty_element, visual_mask)

    guidance_labels = targets.get("guidance_effect_id")
    if guidance_labels is None or not (guidance_labels >= 0).any():
        guidance_loss = pred_visual.sum() * 0.0
    else:
        guidance_loss = F.cross_entropy(
            outputs["guidance_logits"][:, 0].reshape(-1, 3),
            guidance_labels.reshape(-1).long(),
            ignore_index=-1,
        )

    branch_labels = targets.get("branch_outcome_id")
    if branch_labels is None or not (branch_labels >= 0).any():
        branch_loss = pred_visual.sum() * 0.0
    else:
        branch_loss = F.cross_entropy(
            outputs["branch_logits"][:, 0].reshape(-1, 2),
            branch_labels.reshape(-1).long(),
            ignore_index=-1,
        )

    invalid_labels = targets.get("invalid_feed_flag")
    if invalid_labels is None:
        invalid_feed_loss = pred_visual.sum() * 0.0
    else:
        valid = invalid_labels >= 0
        if valid.any():
            invalid_feed_loss = F.binary_cross_entropy_with_logits(
                outputs["invalid_feed_logits"][:, 0][valid],
                invalid_labels[valid].to(dtype=pred_visual.dtype),
            )
        else:
            invalid_feed_loss = pred_visual.sum() * 0.0

    components = {
        "visual": visual_loss,
        "state": state_loss,
        "guidance": guidance_loss,
        "branch": branch_loss,
        "invalid_feed": invalid_feed_loss,
        "uncertainty": uncertainty_loss,
    }
    total = (
        cfg.visual_weight * visual_loss
        + cfg.state_weight * state_loss
        + cfg.guidance_weight * guidance_loss
        + cfg.branch_weight * branch_loss
        + cfg.invalid_feed_weight * invalid_feed_loss
        + cfg.uncertainty_weight * uncertainty_loss
    )
    return total, components


def paired_degradation_consistency_loss(
    clean_outputs: dict[str, torch.Tensor],
    degraded_outputs: dict[str, torch.Tensor],
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    latent_element = F.smooth_l1_loss(
        degraded_outputs["pred_next_visual_latent"],
        clean_outputs["pred_next_visual_latent"].detach(),
        reduction="none",
    )
    state_element = F.smooth_l1_loss(
        degraded_outputs["pred_state_delta"],
        clean_outputs["pred_state_delta"].detach(),
        reduction="none",
    )
    guidance_element = F.kl_div(
        F.log_softmax(degraded_outputs["guidance_logits"], dim=-1),
        F.softmax(clean_outputs["guidance_logits"].detach(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    branch_element = F.kl_div(
        F.log_softmax(degraded_outputs["branch_logits"], dim=-1),
        F.softmax(clean_outputs["branch_logits"].detach(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    latent = _masked_mean(latent_element, sample_mask)
    state = _masked_mean(state_element, sample_mask)
    guidance = _masked_mean(guidance_element, sample_mask)
    branch = _masked_mean(branch_element, sample_mask)
    return latent + 0.25 * state + 0.25 * guidance + 0.25 * branch


def score_action_candidates(
    outputs: dict[str, torch.Tensor],
    *,
    task_id: torch.Tensor,
    candidate_actions: torch.Tensor,
    candidate_policy_eligible: torch.Tensor,
    config: CandidateScoreConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or CandidateScoreConfig()
    batch, candidates, horizon, _ = candidate_actions.shape
    if tuple(candidate_policy_eligible.shape) != (batch, candidates):
        raise ValueError("candidate_policy_eligible must be [B, K]")
    if tuple(task_id.shape) != (batch,):
        raise ValueError("task_id must be [B]")

    branch_probability = outputs["branch_logits"].softmax(dim=-1)[:, :, -1]
    branch_index = task_id.long().view(batch, 1, 1).expand(-1, candidates, 1)
    target_branch = branch_probability.gather(-1, branch_index).squeeze(-1)

    guidance_probability = outputs["guidance_logits"].softmax(dim=-1)
    guidance_index = (task_id.long() + 1).view(batch, 1, 1, 1).expand(
        -1, candidates, horizon, 1
    )
    target_guidance = guidance_probability.gather(-1, guidance_index).squeeze(-1).mean(dim=-1)
    uncertainty = torch.exp(0.5 * outputs["effect_log_variance"]).mean(dim=-1)
    invalid_feed = torch.sigmoid(outputs["invalid_feed_logits"]).mean(dim=-1)
    elite_motion = candidate_actions[..., :3].norm(dim=-1).sum(dim=-1)

    raw_score = (
        cfg.branch_weight * target_branch
        + cfg.guidance_weight * target_guidance
        - cfg.uncertainty_weight * uncertainty
        - cfg.invalid_feed_weight * invalid_feed
        - cfg.elite_motion_weight * elite_motion
    )
    score = raw_score.masked_fill(~candidate_policy_eligible.bool(), float("-inf"))
    if torch.any(~candidate_policy_eligible.bool().any(dim=1)):
        raise ValueError("each batch item must have at least one policy-eligible candidate")
    return {
        "score": score,
        "selected_index": score.argmax(dim=1),
        "target_branch_probability": target_branch,
        "target_guidance_probability": target_guidance,
        "uncertainty": uncertainty,
        "invalid_feed_probability": invalid_feed,
        "elite_motion": elite_motion,
    }
