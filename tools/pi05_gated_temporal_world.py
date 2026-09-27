"""Identity-initialized temporal residual on PI05's existing state token.

The shared history representation / optional future predictor are unchanged.
Unlike appended tokens, a zero residual preserves prefix length and masks.
"""
from __future__ import annotations

from contextlib import contextmanager
import types

import torch
from torch import nn

from pi05_joint_temporal_world import JointTemporalWorld


VERSION = "pi05_gated_state_temporal_world_v1"


class GatedTemporalWorld(JointTemporalWorld):
    def __init__(self, config=None):
        super().__init__(config)
        self.policy_gate = nn.Parameter(torch.zeros(()))

    def policy_residual(self, shared_prefix):
        # Do not branch on gate==0: the gate needs a gradient at initialization.
        return self.policy_gate.tanh() * shared_prefix.mean(dim=1)

    def policy_parameters(self):
        return [self.policy_gate, *self.temporal.parameters()]

    def metadata(self):
        result = super().metadata()
        result.update(version=VERSION, policy_connection="add gated history residual to existing final state token",
                      gate="signed tanh(scalar), initialized at zero; not a probability",
                      extra_policy_tokens=0, world_trainable=any(p.requires_grad for p in self.world.parameters()),
                      trainable="temporal module + scalar gate; future predictor only if explicitly enabled")
        return result


class StateTokenResidualInjection:
    """Install after strict base load and the existing state-conditioning hook."""
    def __init__(self, policy):
        self.model = policy.model
        if not getattr(self.model, "_project2026_state_conditioned", False):
            raise ValueError("gated residual requires the checkpoint's existing state token")
        self.original = self.model.embed_prefix
        self.active_residual = None
        self.injected_calls = 0
        self.last_info = None
        owner = self

        def embed_with_residual(model, *args, **kwargs):
            embeddings, pad, att = owner.original(*args, **kwargs)
            residual = owner.active_residual
            if residual is None:
                return embeddings, pad, att
            if getattr(model, "_project2026_state_context", None) is None:
                raise ValueError("no active state; refusing to modify a language/image token")
            if tuple(residual.shape) != (embeddings.shape[0], embeddings.shape[-1]):
                raise ValueError("temporal residual must match [batch, prefix width]")
            state = embeddings[:, -1:]
            delta = residual.to(device=embeddings.device, dtype=embeddings.dtype).unsqueeze(1)
            # Non-inplace addition, no detach, no extra attention positions.
            updated = torch.cat([embeddings[:, :-1], state + delta], dim=1)
            owner.injected_calls += 1
            owner.last_info = {"prefix_tokens_before": int(embeddings.shape[1]),
                               "prefix_tokens_after": int(updated.shape[1]),
                               "embedding_dtype": str(embeddings.dtype),
                               "masks_reused_unchanged": True}
            return updated, pad, att

        self.model.embed_prefix = types.MethodType(embed_with_residual, self.model)

    @contextmanager
    def condition(self, residual):
        previous = self.active_residual
        self.active_residual = residual
        try:
            yield
        finally:
            self.active_residual = previous

    def close(self):
        self.model.embed_prefix = self.original
        self.active_residual = None


class PrecastStateTokenResidualInjection:
    """Opt-in residual before the existing state projection's first BF16 cast.

    The original state-conditioning hook still owns token concatenation, dtype
    conversion and masks. Existing training/deployment keeps the post-cast
    injection unless this class is explicitly installed after checkpoint load.
    """
    def __init__(self, policy):
        self.model = policy.model
        if not getattr(self.model, "_project2026_state_conditioned", False):
            raise ValueError("pre-cast residual requires the existing state projection")
        self.active_residual = None
        self.injected_calls = 0
        self.last_info = None

        def add_before_cast(module, inputs, projected):
            residual = self.active_residual
            if residual is None:
                return projected
            if getattr(self.model, "_project2026_state_context", None) is None:
                raise ValueError("no active state for pre-cast residual")
            if projected.dtype != torch.float32:
                raise ValueError("pre-cast experiment requires native FP32 state projection output")
            if residual.shape != projected.shape:
                raise ValueError("temporal residual must match [batch, state-token width]")
            # Do not special-case zero or detach: zero-gate identity must come
            # from the arithmetic, without severing the gate's gradient path.
            updated = projected + residual.to(device=projected.device, dtype=projected.dtype)
            self.injected_calls += 1
            self.last_info = {"projection_dtype": str(projected.dtype),
                              "addition_dtype": str(updated.dtype),
                              "extra_policy_tokens": 0,
                              "connection": "state_proj output, before existing prefix cast"}
            return updated

        self.handle = self.model.state_proj.register_forward_hook(add_before_cast)

    @contextmanager
    def condition(self, residual):
        previous = self.active_residual
        self.active_residual = residual
        try:
            yield
        finally:
            self.active_residual = previous

    def close(self):
        self.handle.remove()
        self.active_residual = None
