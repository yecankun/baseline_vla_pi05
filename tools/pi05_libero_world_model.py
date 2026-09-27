"""Independent native LIBERO action/task-conditioned forward model.

No PI0.5 import/load, action selection, loss, optimizer or data acquisition.
This module accepts already extracted visual latents and train-normalized
native state/action coordinates. Closed-registry task lookup is not semantic
language grounding. An untrained forward pass establishes wiring only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ in {None, ""}:
    from pi05_libero_world_model_adapter import IMAGE_KEYS, validate_task_registry
else:
    from .pi05_libero_world_model_adapter import IMAGE_KEYS, validate_task_registry


MODEL_SCHEMA = "pi05_libero_native_action_task_world_model_v1"
INPUT_KEYS = frozenset({
    "history_visual_latent", "history_visual_valid", "history_state",
    "history_state_valid", "task_instruction", "candidate_actions",
})
OUTPUT_KEYS = frozenset({"pred_future_visual_latent", "pred_state_delta"})
FLOAT_KEYS = ("history_visual_latent", "history_state", "candidate_actions")
MASK_KEYS = ("history_visual_valid", "history_state_valid")


@dataclass(frozen=True)
class LiberoWorldModelConfig:
    visual_dim: int = 2048
    hidden_dim: int = 128
    context_len: int = 4
    horizon: int = 3
    state_dim: int = 8
    action_dim: int = 7
    views: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("visual_dim", "hidden_dim", "context_len", "horizon", "state_dim", "action_dim", "views"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, not a coerced dimension")
        if (self.state_dim, self.action_dim, self.views) != (8, 7, 2):
            raise ValueError("native LIBERO requires state_dim=8, action_dim=7, views=2; guidewire9D is incompatible")
        if (type(self.dropout) not in {int, float} or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite in [0,1)")


def _tensor(value: Any, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense torch tensor")
    if value.dtype != dtype or tuple(value.shape) != shape or value.device != device:
        raise ValueError(f"{name} requires {dtype} shape={shape} on {device}; no dtype/layout/device inference")


class LiberoWorldModel(nn.Module):
    """Ordered two-view context GRU with independent causal candidate GRUs."""

    def __init__(self, config: LiberoWorldModelConfig, task_registry: list[dict]):
        super().__init__()
        if not isinstance(config, LiberoWorldModelConfig):
            raise ValueError("LiberoWorldModelConfig is required")
        config.__post_init__()
        self.config = config
        registry = validate_task_registry(task_registry)
        # Scalars/strings are immutable. No mutable caller dictionary survives.
        self._task_registry = tuple((task, int(registry[task]["source_task_index"]), registry[task]["task_instruction"])
                                    for task in sorted(registry))
        self._task_encoding = MappingProxyType({row[2]: index for index, row in enumerate(self._task_registry)})
        h = config.hidden_dim
        self.view_projections = nn.ModuleList([nn.Linear(config.visual_dim, h) for _ in range(config.views)])
        self.history_projection = nn.Linear(config.views * h + config.views + 2 * config.state_dim, h)
        self.context_gru = nn.GRU(h, h, batch_first=True)
        self.task_embedding = nn.Embedding(len(self._task_registry), h)
        self.context_task_projection = nn.Linear(2 * h, h)
        self.action_projection = nn.Linear(config.action_dim, h)
        self.action_gru = nn.GRU(h, h, batch_first=True)
        self.dropout = nn.Dropout(float(config.dropout))
        self.visual_residual_head = nn.Linear(h, config.views * config.visual_dim)
        self.state_residual_head = nn.Linear(h, config.state_dim)

    def metadata(self) -> dict:
        encoding = [{"task_id": task, "source_task_index": source, "task_instruction": instruction,
                     "embedding_index": index} for index, (task, source, instruction) in enumerate(self._task_registry)]
        payload = json.dumps(encoding, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return {
            "schema": MODEL_SCHEMA, "config": asdict(self.config),
            "task_registry_encoding": encoding,
            "task_registry_encoding_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "task_conditioning": "exact native instruction closed-registry trainable lookup; indices sorted by task_id",
            "semantic_or_open_language_grounding": False,
            "input_contract": {
                "history_visual_latent": "float32 [B,T,2,D]; frozen PI0.5 features supplied by caller",
                "history_visual_valid": "bool [B,T,2]",
                "history_state": "float32 [B,T,8]; already train-normalized",
                "history_state_valid": "bool [B,T,8]",
                "task_instruction": "list[str] of length B; exact known native text only",
                "candidate_actions": "finite float32 [B,K,H,7]; already train-normalized; no [-1,1] clipping",
            },
            "output_contract": {
                "pred_future_visual_latent": "float32 [B,K,H,2,D]; current masked latent plus learned residual",
                "pred_state_delta": "float32 [B,K,H,8]; normalized current-relative coordinate residual, not SO3 rotation",
            },
            "view_order": list(IMAGE_KEYS),
            "masking": "torch.where before every input projection; valid values finite; masks also enter context",
            "candidate_causality": "unidirectional action GRU per candidate; step h depends only on candidate actions through h",
            "cross_candidate_mixing": False, "second_normalization": False,
            "pi05_loading_in_module": False, "loss_or_optimizer_in_module": False,
            "action_selection_in_module": False, "training_started_by_module": False,
            "forward_verification_is_model_quality_evidence": False,
        }

    def forward(
        self, *, history_visual_latent: torch.Tensor, history_visual_valid: torch.Tensor,
        history_state: torch.Tensor, history_state_valid: torch.Tensor,
        task_instruction: list[str], candidate_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if not isinstance(history_visual_latent, torch.Tensor) or history_visual_latent.ndim != 4:
            raise ValueError("history_visual_latent requires [B,T,2,D]")
        b = history_visual_latent.shape[0]
        if b <= 0:
            raise ValueError("empty batches are unsupported")
        parameter = next(self.parameters())
        if parameter.dtype != torch.float32:
            raise ValueError("native forward model requires float32 parameters")
        device = parameter.device
        _tensor(history_visual_latent, "history_visual_latent", (b, cfg.context_len, cfg.views, cfg.visual_dim), torch.float32, device)
        _tensor(history_visual_valid, "history_visual_valid", (b, cfg.context_len, cfg.views), torch.bool, device)
        _tensor(history_state, "history_state", (b, cfg.context_len, cfg.state_dim), torch.float32, device)
        _tensor(history_state_valid, "history_state_valid", (b, cfg.context_len, cfg.state_dim), torch.bool, device)
        if not isinstance(candidate_actions, torch.Tensor) or candidate_actions.ndim != 4 or candidate_actions.shape[1] <= 0:
            raise ValueError("candidate_actions requires nonempty [B,K,H,7]")
        k = candidate_actions.shape[1]
        _tensor(candidate_actions, "candidate_actions", (b, k, cfg.horizon, cfg.action_dim), torch.float32, device)
        if (type(task_instruction) is not list or len(task_instruction) != b
                or any(type(text) is not str or text not in self._task_encoding for text in task_instruction)):
            raise ValueError("task_instruction requires one exact known native instruction per batch row")
        if not bool(torch.isfinite(history_visual_latent[history_visual_valid]).all()):
            raise ValueError("valid visual latents must be finite")
        if not bool(torch.isfinite(history_state[history_state_valid]).all()):
            raise ValueError("valid state coordinates must be finite")
        if not bool(torch.isfinite(candidate_actions).all()):
            raise ValueError("all candidate actions must be finite; actions cannot be imputed")

        # Masking by multiplication would preserve invalid NaN/Inf. Clean before
        # any projection and use the same cleaned current latent in the skip.
        visual = torch.where(history_visual_valid[..., None], history_visual_latent, 0.0)
        state = torch.where(history_state_valid, history_state, 0.0)
        views = [torch.tanh(layer(visual[:, :, index])) for index, layer in enumerate(self.view_projections)]
        history = torch.cat([*views, history_visual_valid.to(torch.float32), state, history_state_valid.to(torch.float32)], dim=-1)
        history = self.dropout(torch.tanh(self.history_projection(history)))
        _, context = self.context_gru(history)
        task_ids = torch.tensor([self._task_encoding[text] for text in task_instruction], dtype=torch.long, device=device)
        initial = torch.tanh(self.context_task_projection(torch.cat([context[-1], self.task_embedding(task_ids)], dim=-1)))
        initial = initial[:, None, :].expand(b, k, cfg.hidden_dim).reshape(1, b * k, cfg.hidden_dim).contiguous()
        actions = candidate_actions.reshape(b * k, cfg.horizon, cfg.action_dim)
        actions = self.dropout(torch.tanh(self.action_projection(actions)))
        future, _ = self.action_gru(actions, initial)
        future = self.dropout(future).reshape(b, k, cfg.horizon, cfg.hidden_dim)
        residual = self.visual_residual_head(future).reshape(b, k, cfg.horizon, cfg.views, cfg.visual_dim)
        outputs = {
            "pred_future_visual_latent": visual[:, -1, None, None] + residual,
            "pred_state_delta": self.state_residual_head(future),
        }
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in outputs.values()):
            raise ValueError("native world-model output must remain finite float32")
        return outputs


def collate_window_inputs(inputs_list: list[dict], device: str | torch.device = "cpu") -> dict:
    """Copy only existing native window inputs into owned tensor batches.

    No targets, metadata, identifiers or diagnostic fields are accepted. Masks
    are not coerced from numeric arrays. Invalid coordinates may be nonfinite;
    the forward model validates observed values and cleans masked ones.
    """
    if type(inputs_list) is not list or not inputs_list:
        raise ValueError("inputs_list must be a nonempty list of native window input dictionaries")
    for item in inputs_list:
        if type(item) is not dict or set(item) != INPUT_KEYS:
            raise ValueError("only the six native input keys are accepted; targets/metadata/oracle fields forbidden")
        if type(item["task_instruction"]) is not str or not item["task_instruction"].strip():
            raise ValueError("native task instruction must be nonempty text")
        for key in FLOAT_KEYS:
            value = item[key]
            if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
                raise ValueError(f"{key} must be a floating NumPy array from native window inputs")
        for key in MASK_KEYS:
            value = item[key]
            if not isinstance(value, np.ndarray) or value.dtype != np.bool_:
                raise ValueError(f"{key} must be a boolean NumPy mask; no coercion")
    shapes = {key: inputs_list[0][key].shape for key in (*FLOAT_KEYS, *MASK_KEYS)}
    if any(item[key].shape != shape for item in inputs_list for key, shape in shapes.items()):
        raise ValueError("all native window input shapes must match across the batch")
    visual, state, action = (shapes[key] for key in FLOAT_KEYS)
    if (len(visual) != 3 or visual[0] <= 0 or visual[1] != 2 or visual[2] <= 0
            or state != (visual[0], 8) or len(action) != 3 or min(action[:2]) <= 0 or action[2] != 7
            or shapes["history_visual_valid"] != visual[:2] or shapes["history_state_valid"] != state):
        raise ValueError("native window input shapes require [T,2,D], [T,8], [K,H,7] with matching masks")
    batch = {"task_instruction": [item["task_instruction"] for item in inputs_list]}
    for key in (*FLOAT_KEYS, *MASK_KEYS):
        dtype = torch.bool if key in MASK_KEYS else torch.float32
        # torch.tensor copies storage even on CPU; it never aliases caller data.
        batch[key] = torch.tensor(np.stack([item[key] for item in inputs_list]), dtype=dtype, device=device)
    if not bool(torch.isfinite(batch["history_visual_latent"][batch["history_visual_valid"]]).all()):
        raise ValueError("valid collated visual values must be finite float32")
    if not bool(torch.isfinite(batch["history_state"][batch["history_state_valid"]]).all()):
        raise ValueError("valid collated state values must be finite float32")
    if not bool(torch.isfinite(batch["candidate_actions"]).all()):
        raise ValueError("collated actions must be finite float32 without imputation")
    return batch
