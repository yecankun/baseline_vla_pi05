"""Experimental shared temporal prefix for PI05 and one-step latent dynamics.

The pretrained policy stays frozen. The SAME trainable prefix tensor is used
by its flow-matching action expert and by the future predictor. This is not a
detached feature scorer. Future observations are loss targets only.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import types

import torch
from torch import nn
import torch.nn.functional as F


VERSION = "pi05_joint_temporal_prefix_world_v1"


@dataclass(frozen=True)
class JointTemporalConfig:
    visual_dim: int = 2048
    prefix_dim: int = 2048
    state_dim: int = 32
    views: int = 2
    history: int = 3
    hidden_dim: int = 128
    prefix_tokens: int = 2
    token_scale: float = 0.1


def visual_target(features: torch.Tensor) -> torch.Tensor:
    """Fixed per-view feature normalization, with no fitted test statistics."""
    return F.layer_norm(features.float(), (features.shape[-1],))


class SharedTemporalPrefix(nn.Module):
    def __init__(self, config: JointTemporalConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.visual_proj = nn.Linear(config.visual_dim, h)
        self.state_proj = nn.Linear(config.state_dim, h)
        self.age_proj = nn.Linear(1, h, bias=False)
        self.view_embedding = nn.Parameter(torch.randn(config.views, h) * 0.02)
        self.slot_embedding = nn.Parameter(torch.randn(config.history, h) * 0.02)
        self.queries = nn.Parameter(torch.randn(config.prefix_tokens, h) * 0.02)
        self.attention = nn.MultiheadAttention(h, 4, dropout=0.0, batch_first=True)
        self.to_prefix = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, config.prefix_dim))

    def forward(self, visual_history, state_history, history_valid, relative_times):
        c = self.config
        if visual_history.ndim != 4 or tuple(visual_history.shape[1:]) != (c.history, c.views, c.visual_dim):
            raise ValueError("visual_history must be [B, history, views, visual_dim]")
        b = visual_history.shape[0]
        if tuple(state_history.shape) != (b, c.history, c.state_dim):
            raise ValueError("state_history shape differs from the fixed layout")
        if tuple(history_valid.shape) != (b, c.history) or not history_valid[:, -1].all():
            raise ValueError("current observation is required; mask only missing past slots")
        if tuple(relative_times.shape) != (b, c.history) or (relative_times > 0).any():
            raise ValueError("history timestamps must be relative to current time and nonpositive")
        # Mask before the projections as well as in attention: missing entries
        # cannot contaminate a token even when their placeholder values change.
        valid = history_valid.bool()
        visual = torch.where(valid[..., None, None], visual_history, 0.0)
        state = torch.where(valid[..., None], state_history, 0.0)
        ages = torch.where(valid, relative_times, 0.0)
        sequence = self.visual_proj(visual_target(visual))
        sequence = sequence + self.state_proj(state.float()).unsqueeze(2)
        sequence = sequence + self.age_proj(ages.float().unsqueeze(-1)).unsqueeze(2)
        sequence = sequence + self.view_embedding[None, None]
        sequence = sequence + self.slot_embedding[None, :, None]
        sequence = sequence.flatten(1, 2)
        padding = (~valid[..., None].expand(-1, -1, c.views)).flatten(1, 2)
        query = self.queries.unsqueeze(0).expand(b, -1, -1)
        attended, _ = self.attention(query, sequence, sequence, key_padding_mask=padding, need_weights=False)
        prefix = self.to_prefix(attended + query)
        return F.layer_norm(prefix, (c.prefix_dim,)) * c.token_scale


class JointTemporalWorld(nn.Module):
    def __init__(self, config: JointTemporalConfig | None = None):
        super().__init__()
        self.config = config or JointTemporalConfig()
        c = self.config
        self.temporal = SharedTemporalPrefix(c)
        self.world = nn.Sequential(
            nn.LayerNorm(c.prefix_tokens * c.prefix_dim + 9),
            nn.Linear(c.prefix_tokens * c.prefix_dim + 9, c.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim * 2, c.views * c.visual_dim),
        )
        # Small, nonzero residual initialization permits a future-loss gradient
        # into the shared prefix on the very first backward pass.
        nn.init.normal_(self.world[-1].weight, std=0.001)
        nn.init.zeros_(self.world[-1].bias)

    def encode_context(self, visual_history, state_history, history_valid, relative_times):
        return self.temporal(visual_history, state_history, history_valid, relative_times)

    def predict_future(self, shared_prefix, current_visual, action_9):
        """action_9: normalized Elite 6D + canonical, unnormalized intent one-hot.

        Recorded measured Elite delta / requested Piper intent during training;
        proposed actions during inference. Neither is proof of distal execution.
        """
        if action_9.shape != (shared_prefix.shape[0], 9):
            raise ValueError("world-model action must be [B,9]")
        residual = self.world(torch.cat([shared_prefix.float().flatten(1), action_9.float()], dim=-1))
        residual = residual.reshape(-1, self.config.views, self.config.visual_dim)
        return visual_target(current_visual).detach() + residual

    @staticmethod
    def future_loss(prediction, future_visual):
        return F.mse_loss(prediction.float(), visual_target(future_visual).detach())

    def metadata(self):
        return {"version": VERSION, "config": asdict(self.config),
                "output_interface": "elite_tcp_delta_6d + piper_intent_id",
                "future_target": "next recorded observation, not a fixed physical-time horizon",
                "target_encoder": "same frozen PI05 vision weights, detached targets",
                "trainable": "shared temporal prefix and latent dynamics only",
                "piper_head": "existing frozen state_only head",
                "lora_enabled": False, "response_supervision_enabled": False,
                "reranking_enabled": False, "contact_input_enabled": False}


class TemporalPrefixInjection:
    """Opt-in prefix extension; absent context is exactly the original policy.

    Install after loading the old checkpoint and its existing state adapter.
    No LeRobot source or old state_dict key is modified. Keep the context alive
    through backward if gradient checkpointing is enabled in a later trainer.
    """
    def __init__(self, policy):
        self.model = policy.model
        self.original = self.model.embed_prefix
        self.active_prefix = None
        self.injected_calls = 0
        self.last_extra_tokens = 0
        owner = self

        def embed_with_temporal_prefix(model, *args, **kwargs):
            embeddings, pad, att = owner.original(*args, **kwargs)
            prefix = owner.active_prefix
            if prefix is None:
                return embeddings, pad, att
            if prefix.shape[0] != embeddings.shape[0] or prefix.shape[-1] != embeddings.shape[-1]:
                raise ValueError("temporal prefix does not match the PI05 prefix batch/width")
            # Do NOT detach here: action loss must reach the temporal adapter.
            extra = prefix.to(device=embeddings.device, dtype=embeddings.dtype)
            shape = extra.shape[:2]
            owner.injected_calls += 1
            owner.last_extra_tokens = int(shape[1])
            return (torch.cat([embeddings, extra], dim=1),
                    torch.cat([pad, torch.ones(shape, dtype=pad.dtype, device=pad.device)], dim=1),
                    torch.cat([att, torch.zeros(shape, dtype=att.dtype, device=att.device)], dim=1))

        self.model.embed_prefix = types.MethodType(embed_with_temporal_prefix, self.model)

    @contextmanager
    def condition(self, shared_prefix):
        previous = self.active_prefix
        self.active_prefix = shared_prefix
        try:
            yield
        finally:
            self.active_prefix = previous

    def close(self):
        self.model.embed_prefix = self.original
        self.active_prefix = None
