from __future__ import annotations

import types
from typing import Any

import torch
from torch import nn


STATE_CONDITIONING_VERSION = "state_prefix_token_v1"


def _state_conditioning_metadata(model: nn.Module, state_dim: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "version": STATE_CONDITIONING_VERSION,
        "state_dim": int(state_dim),
        "token_position": "after_image_and_language_prefix",
        "projected_dim": int(model.state_proj[-1].out_features),
        "uses_exact_contact_or_wall_truth": False,
        "loss_scope": "Elite TCP translation delta dims 0:3 only",
    }


def enable_state_conditioning(policy: nn.Module, state_dim: int) -> dict[str, Any]:
    """Attach a state prefix token after pretrained weights are loaded.

    The adapter is deliberately added after PI05 checkpoint loading so the
    pinned base remains fully covered. Runtime wrappers pass state through the
    existing PI05 prefix and sampling paths without changing LeRobot sources.
    """
    model = policy.model
    if getattr(model, "_project2026_state_conditioned", False):
        return dict(model._project2026_state_conditioning)

    prefix_dim = int(model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
    model.state_proj = nn.Sequential(
        nn.LayerNorm(int(state_dim)),
        nn.Linear(int(state_dim), prefix_dim),
    ).to(next(model.parameters()).device)

    original_embed_prefix = model.embed_prefix
    original_forward = model.forward
    original_sample_actions = model.sample_actions
    original_predict_action_chunk = policy.predict_action_chunk

    def embed_prefix_with_state(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = original_embed_prefix(images, img_masks, tokens, masks)
        state = getattr(self, "_project2026_state_context", None)
        if state is None:
            return embs, pad_masks, att_masks
        state_token = self.state_proj(state.float()).to(dtype=embs.dtype).unsqueeze(1)
        state_pad = torch.ones(
            state_token.shape[0], 1, dtype=pad_masks.dtype, device=pad_masks.device
        )
        state_att = torch.zeros(
            state_token.shape[0], 1, dtype=att_masks.dtype, device=att_masks.device
        )
        return (
            torch.cat([embs, state_token], dim=1),
            torch.cat([pad_masks, state_pad], dim=1),
            torch.cat([att_masks, state_att], dim=1),
        )

    def forward_with_state(self, *args, state=None, **kwargs):
        previous = getattr(self, "_project2026_state_context", None)
        self._project2026_state_context = state
        try:
            return original_forward(*args, **kwargs)
        finally:
            self._project2026_state_context = previous

    def sample_actions_with_state(self, *args, state=None, **kwargs):
        previous = getattr(self, "_project2026_state_context", None)
        # Native PI05 calls sample_actions without a state kwarg. Preserve the
        # context installed by predict_action_chunk instead of clearing it.
        self._project2026_state_context = previous if state is None else state
        try:
            return original_sample_actions(*args, **kwargs)
        finally:
            self._project2026_state_context = previous

    def predict_action_chunk_with_state(self, batch, *args, **kwargs):
        state = batch.get("observation.state") if isinstance(batch, dict) else None
        previous = getattr(model, "_project2026_state_context", None)
        model._project2026_state_context = state
        try:
            return original_predict_action_chunk(batch, *args, **kwargs)
        finally:
            model._project2026_state_context = previous

    model.embed_prefix = types.MethodType(embed_prefix_with_state, model)
    model.forward = types.MethodType(forward_with_state, model)
    model.sample_actions = types.MethodType(sample_actions_with_state, model)
    policy.predict_action_chunk = types.MethodType(predict_action_chunk_with_state, policy)
    model._project2026_state_conditioned = True
    model._project2026_state_conditioning = _state_conditioning_metadata(model, state_dim)
    policy._project2026_state_conditioned = True
    policy._project2026_state_conditioning = dict(model._project2026_state_conditioning)
    return dict(model._project2026_state_conditioning)


def state_conditioning_enabled(policy: nn.Module) -> bool:
    return bool(getattr(policy, "_project2026_state_conditioned", False))
