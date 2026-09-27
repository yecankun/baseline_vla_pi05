"""Matched global-summary / regional-moment response readouts, not a policy.

Both arms have identical trainable parameters. Human geometry is never an input.
The frozen localizer is loaded from the matching source episode fold only.
"""
from __future__ import annotations

import torch
from torch import nn

from real10_spatial_response import spatial_grid

ARMS = ("global_broadcast", "regional_4x4")
REGION_GRID = 4
REGIONS = REGION_GRID**2
TOKEN_CHANNELS = 11
INPUT_DIM = 3 * 2 * REGIONS * TOKEN_CHANNELS + 2
STD_FLOOR = .001


def select_representation(tokens, arm):
    if arm == "regional_4x4":
        return tokens
    if arm == "global_broadcast":
        return (tokens.sum(-2, keepdim=True) / REGIONS).expand_as(tokens)
    raise ValueError(f"unknown paired arm: {arm}")


def temporal_features(tokens, dt_s, task_right, arm):
    state = select_representation(tokens, arm).flatten(2)
    n, t, _ = state.shape
    anchor = state[:, :1].expand(n, t - 1, -1)
    context = torch.stack((task_right[:, None].expand_as(dt_s), dt_s / 1.5), -1)
    return torch.cat((anchor, state[:, 1:] - anchor,
                      state[:, 1:] - state[:, :-1], context), -1)


class RegionResponse(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(64, 8, 1), nn.GELU())
        self.location = nn.Conv2d(8, 1, 1)
        self.response = nn.Sequential(nn.Linear(INPUT_DIM, 8), nn.GELU(), nn.Linear(8, 1))
        self.register_buffer("feature_mean", torch.zeros(INPUT_DIM))
        self.register_buffer("feature_scale", torch.ones(INPUT_DIM))
        self.project.requires_grad_(False)
        self.location.requires_grad_(False)

    @torch.no_grad()
    def load_localizer(self, state):
        self.project.load_state_dict({k.removeprefix("project."): v for k, v in state.items()
                                      if k.startswith("project.")})
        self.location.load_state_dict({k.removeprefix("location."): v for k, v in state.items()
                                       if k.startswith("location.")})

    def region_tokens(self, maps):
        n, t, v, c, h, w = maps.shape
        if (v, c, h, w) != (2, 64, 56, 56):
            raise ValueError("expected original N,T,2,64,56,56 frozen maps")
        z = self.project(maps.reshape(n*t*v, c, h, w))
        p = self.location(z).flatten(2).softmax(-1).reshape(-1, 1, h, w)
        xy = spatial_grid(maps.device, maps.dtype).permute(2, 0, 1).unsqueeze(0)
        weighted = torch.cat((p*z, p*xy, p), 1)
        # NCHW -> channel, grid-y, within-y, grid-x, within-x. Sum within each cell.
        moments = weighted.reshape(-1, TOKEN_CHANNELS, REGION_GRID, h//REGION_GRID,
                                   REGION_GRID, w//REGION_GRID).sum((3, 5))
        return moments.permute(0, 2, 3, 1).reshape(n, t, v, REGIONS, TOKEN_CHANNELS)

    @torch.no_grad()
    def fit_normalizer(self, train_features, train_valid):
        known = train_features[train_valid]
        if not known.numel() or not torch.isfinite(known).all():
            raise ValueError("finite training transitions required for normalization")
        self.feature_mean.copy_(known.mean(0))
        self.feature_scale.copy_(known.std(0, unbiased=False).clamp_min(STD_FLOOR))

    def readout(self, features, transition_valid):
        local = self.response((features-self.feature_mean)/self.feature_scale).squeeze(-1)
        count = transition_valid.sum(1)
        if torch.any(count == 0):
            raise ValueError("every window needs an observed transition")
        window = local.masked_fill(~transition_valid, 0).sum(1)/count
        return {"window_logit": window, "local_logits": local}

    def forward(self, maps, dt_s, task_right, transition_valid, arm):
        tokens = self.region_tokens(maps)
        features = temporal_features(tokens, dt_s, task_right, arm)
        return {**self.readout(features, transition_valid), "region_tokens": tokens}
