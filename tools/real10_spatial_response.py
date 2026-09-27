"""Spatial response prototype, separate from PI05 and action-effect prediction.

forward() never receives human points, visibility, vessel landmarks or labels.
The optional point loss is evaluated only on known tip locations in TRAIN folds.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


ARMS = {"spatial_window": 0.0, "spatial_window_tip_aux": 0.1}
GRID_SIZE = 56
IMAGE_SIZE = 224
TIP_SIGMA_CELLS = 1.0


def spatial_grid(device, dtype):
    # conv7/s2/p3 -> maxpool3/s2/p1 -> layer1: stride 4, first center 0.5 px.
    coordinate = (torch.arange(GRID_SIZE, device=device, dtype=dtype) * 4 + .5) / IMAGE_SIZE
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


class SpatialResponse(nn.Module):
    """Shared two-view location/appearance readout, with fixed mean time pooling."""

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(64, 8, 1), nn.GELU())
        self.location = nn.Conv2d(8, 1, 1)
        self.response = nn.Sequential(nn.Linear(62, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, maps, dt_s, task_right, transition_valid):
        n, t, v, c, h, w = maps.shape
        if (v, c, h, w) != (2, 64, GRID_SIZE, GRID_SIZE):
            raise ValueError("expected N,T,2,64,56,56 frozen layer1 maps")
        z = self.project(maps.reshape(n*t*v, c, h, w))
        heatmap_logits = self.location(z).reshape(n, t, v, h, w)
        attention = heatmap_logits.flatten(-2).softmax(-1)
        values = z.reshape(n, t, v, 8, h*w)
        appearance = (values * attention.unsqueeze(-2)).sum(-1)
        grid = spatial_grid(maps.device, maps.dtype).reshape(h*w, 2)
        xy = attention @ grid
        state = torch.cat((appearance, xy), dim=-1).flatten(2)  # two views * (8+2)
        anchor = state[:, :1].expand(-1, t-1, -1)
        context = torch.stack((task_right[:, None].expand_as(dt_s), dt_s / 1.5), dim=-1)
        local = self.response(torch.cat((anchor, state[:, 1:]-anchor,
                                        state[:, 1:]-state[:, :-1], context), dim=-1)).squeeze(-1)
        count = transition_valid.sum(1)
        if torch.any(count == 0):
            raise ValueError("each response window needs an observed transition")
        score = local.masked_fill(~transition_valid, 0).sum(1) / count
        return {"window_logit": score, "local_logits": local,
                "heatmap_logits": heatmap_logits, "location_uv": xy}


def point_distribution(xy_uv, valid):
    """Gaussian coordinate encoding, not a measured annotation uncertainty.

    Invalid locations receive no target mass and MUST be masked out of loss.
    This does not label an unannotated frame as empty/background.
    """
    safe = xy_uv.masked_fill(~valid[..., None], 0)
    grid = spatial_grid(xy_uv.device, xy_uv.dtype)
    distance = (grid - safe[..., None, None, :]) * GRID_SIZE
    logits = -distance.square().sum(-1) / (2 * TIP_SIGMA_CELLS**2)
    distribution = logits.flatten(-2).softmax(-1).reshape_as(logits)
    return distribution.masked_fill(~valid[..., None, None], 0)


def response_losses(output, response_y, xy_uv, tip_valid, train_window, pos_weight, arm):
    """Same structure, inputs and response objective; only auxiliary weight differs."""
    if arm not in ARMS or not torch.any(train_window):
        raise ValueError("known arm and at least one training window required")
    response = F.binary_cross_entropy_with_logits(
        output["window_logit"][train_window], response_y[train_window], pos_weight=pos_weight)
    active = tip_valid & train_window[:, None, None]
    distribution = point_distribution(xy_uv, active)
    logp = output["heatmap_logits"].flatten(-2).log_softmax(-1)
    per_point = -(distribution.flatten(-2) * logp).sum(-1) / math.log(GRID_SIZE**2)
    count = active.sum((1, 2))
    # Equal weight per labeled training WINDOW, not per repeated point/frame.
    if torch.any(count > 0):
        per_window = per_point.sum((1, 2)) / count.clamp_min(1)
        localization = per_window[count > 0].mean()
    else:
        localization = output["heatmap_logits"].sum() * 0
    return {"total": response + ARMS[arm] * localization,
            "response": response, "localization": localization,
            "active_tip_mask": active}
