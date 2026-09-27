"""Identity-initialized object-grid residual variant; original model stays intact.

Same observable inputs, 128-hidden GRU, parameter count and Dice loss. Only
the output rule and output-layer initialization differ. This is not a policy.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

if __package__:
    from . import pusht_object_dynamics as base
else:
    import pusht_object_dynamics as base

SCHEMA = "pusht_rgb_object_residual_dynamics_v1"
CACHE_ROOT = base.CACHE_ROOT  # Reuse the original cache and its original schema.
TRAIN_ROOT = base.TRAIN_ROOT.parent / "pusht_object_residual_dynamics_v1"
PLAN = {**base.PLAN,
    "schema": SCHEMA,
    "architecture": "unchanged128_GRU_current_grid_identity_plus_signed_residual_clamp01",
    "output_rule": "clamp(current_grid+decode(hidden_t),0,1)_each_horizon_relative_to_current",
    "initialization": "same_seed_and_trunk_as_base_zero_decoder_weight_and_bias",
    "initial_forecast": "exact_current_grid_persistence_not_learned_stationarity",
    "first_backward": "decoder_can_receive_gradient_trunk_zero_until_decoder_is_nonzero",
    "base_model_schema": base.SCHEMA,
    "cache_schema": base.SCHEMA,
    "base_checkpoint_compatible": False,
    "warm_start_from_trained_model": False,
    "comparison": "existing_fixed_final10000_base_and_persistence_same_cache_split_budget",
}

# Keep data, scoring, invalid-observation handling and final metrics unchanged.
ObjectData = base.ObjectData
evaluate = base.evaluate
predict_and_score = base.predict_and_score
dice_cost = base.dice_cost


class ObjectGridDynamics(base.ObjectGridDynamics):
    """Predict signed occupancy changes from the CURRENT grid, not prior output.

Zero decoder gives exact identity at initialization. The clamp keeps occupancy
in [0,1]; it does not enforce rigid shape, area conservation or contact physics.
Future grids and goal/coverage metadata never enter this forward method.
"""

    def __init__(self, stats: dict):
        super().__init__(stats)
        nn.init.zeros_(self.decode.weight)
        nn.init.zeros_(self.decode.bias)

    def forward(self, current, agent_pos, actions):
        b = len(current)
        if current.shape != (b, 24, 24) or agent_pos.shape != (b, 2):
            raise ValueError("current grid/state require [B,24,24]/[B,2]")
        if actions.ndim != 4 or actions.shape[0] != b or actions.shape[2:] != (8, 2):
            raise ValueError("absolute native actions require [B,K,8,2]")
        k = actions.shape[1]
        initial = self.encode(current.flatten(1)) + self.state_embed((agent_pos - self.state_mean) / self.state_std)
        hidden = initial[:, None].expand(-1, k, -1).reshape(b * k, -1)
        normalized = ((actions - self.action_mean) / self.action_std).reshape(b * k, 8, 2)
        output = []
        for t in range(8):
            hidden = self.cell(self.action_embed(normalized[:, t]), hidden)
            residual = self.decode(hidden).reshape(b, k, 24, 24)
            output.append((current[:, None] + residual).clamp(0, 1))
        return torch.stack(output, dim=2)


def load_final(path: Path, *, device: str = "cuda"):
    """Versioned loading is mandatory: matching state_dict shapes are insufficient."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["schema"] != SCHEMA or payload["kind"] != "final" or payload["step"] != PLAN["steps"] or payload["plan"] != PLAN:
        raise ValueError("requires the fixed final residual checkpoint, not the base model")
    model = ObjectGridDynamics(payload["cache_identity"]["normalization"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(False)
