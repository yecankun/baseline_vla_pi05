"""Parameter-matched, explicit adjacent-change representation for gated PI05.

Only observable history is consumed. The existing optional world predictor and
canonical action interface are unchanged; auxiliary training remains opt-in.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from pi05_gated_temporal_world import GatedTemporalWorld
from pi05_joint_temporal_world import SharedTemporalPrefix, visual_target


VERSION = "pi05_gated_adjacent_change_v1"


class MotionTemporalPrefix(SharedTemporalPrefix):
    """Reuse all original parameter shapes, but encode two changes + current.

    Differences are taken AFTER per-observation visual normalization. Do not
    layer-normalize each difference or divide by a possibly short time interval:
    that would amplify weak/noisy motion and discard its magnitude.
    """
    def motion_inputs(self, visual_history, state_history, history_valid, relative_times):
        c = self.config
        b = visual_history.shape[0]
        if c.history != 3 or tuple(visual_history.shape) != (b, 3, c.views, c.visual_dim):
            raise ValueError("motion input requires [B,3,views,visual_dim]")
        if tuple(state_history.shape) != (b, 3, c.state_dim):
            raise ValueError("motion input requires checkpoint-normalized [B,3,state_dim]")
        if tuple(history_valid.shape) != (b, 3) or not history_valid[:, -1].all():
            raise ValueError("current observation must be valid")
        if tuple(relative_times.shape) != (b, 3) or (relative_times > 0).any():
            raise ValueError("timestamps must be relative to current and nonpositive")
        valid = history_valid.bool()
        pair_valid = valid[:, :-1] & valid[:, 1:]
        visual = visual_target(torch.where(valid[..., None, None], visual_history, 0.0))
        states = torch.where(valid[..., None], state_history.float(), 0.0)
        ages = torch.where(valid, relative_times.float(), 0.0)
        durations = ages[:, 1:] - ages[:, :-1]
        if (durations[pair_valid] < 0).any():
            raise ValueError("valid adjacent observations must be chronological")
        visual_delta = torch.where(pair_valid[..., None, None], visual[:, 1:] - visual[:, :-1], 0.0)
        state_delta = torch.where(pair_valid[..., None], states[:, 1:] - states[:, :-1], 0.0)
        intervals = torch.where(pair_valid, durations, 0.0)
        return {
            "visual": torch.cat([visual_delta, visual[:, -1:]], dim=1),
            "states": torch.cat([state_delta, states[:, -1:]], dim=1),
            "times": torch.cat([intervals, torch.zeros_like(intervals[:, :1])], dim=1),
            "valid": torch.cat([pair_valid, valid[:, -1:]], dim=1),
        }

    def forward(self, visual_history, state_history, history_valid, relative_times):
        inputs = self.motion_inputs(visual_history, state_history, history_valid, relative_times)
        c, b = self.config, visual_history.shape[0]
        sequence = self.visual_proj(inputs["visual"])
        sequence = sequence + self.state_proj(inputs["states"]).unsqueeze(2)
        sequence = sequence + self.age_proj(inputs["times"].unsqueeze(-1)).unsqueeze(2)
        sequence = sequence + self.view_embedding[None, None]
        sequence = sequence + self.slot_embedding[None, :, None]
        sequence = sequence.flatten(1, 2)
        padding = (~inputs["valid"][..., None].expand(-1, -1, c.views)).flatten(1, 2)
        query = self.queries.unsqueeze(0).expand(b, -1, -1)
        attended, _ = self.attention(query, sequence, sequence, key_padding_mask=padding, need_weights=False)
        prefix = self.to_prefix(attended + query)
        return F.layer_norm(prefix, (c.prefix_dim,)) * c.token_scale


class MotionGatedTemporalWorld(GatedTemporalWorld):
    def __init__(self, config=None):
        super().__init__(config)
        self.temporal = MotionTemporalPrefix(self.config)

    def metadata(self):
        result = super().metadata()
        result.update(
            version=VERSION,
            temporal_representation="[z(t-1)-z(t-2), z(t)-z(t-1), z(t)] plus matching normalized-state changes",
            temporal_times="[t(t-1)-t(t-2), t(t)-t(t-1), 0] in seconds; not velocity",
            visual_difference_normalization="normalize each observation before difference; no delta renormalization",
            missing_history="mask a change unless both endpoint observations are valid",
            parameter_shapes="identical to original GatedTemporalWorld; metadata required to distinguish semantics",
        )
        return result
