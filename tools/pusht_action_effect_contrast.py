"""Same-start action-effect supervision, separate from policy execution.

Both arms reuse DirectActionScorer unchanged. This is a controlled loss
ablation, not a new world-model architecture or an established contribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .pusht_direct_action_scorer import DirectActionScorer
else:
    from pusht_direct_action_scorer import DirectActionScorer

SCHEMA = "pusht_same_start_effect_pair_v1"
ARMS = ("pointwise", "pointwise_plus_effect_difference")
INPUT_FIELDS = ("current", "agent_xy", "actions")


def effect_difference_loss(prediction, target, *, pair_weight):
    """Inputs [B,5] share one training-only target standardization.

    All ten unordered pairs (including true ties) receive equal weight within
    each context. No cross-context pair, invented negative, or pair sampling.
    Half squared difference error = K/(K-1) times within-context error variance;
    this reweights existing label information rather than creating new labels.
    """
    if prediction.ndim != 2 or prediction.shape[1] != 5 or target.shape != prediction.shape:
        raise ValueError("requires aligned [context,5] predictions and actually executed targets")
    if pair_weight not in (0.0, 1.0):
        raise ValueError("fixed ablation uses pair weight 0 or 1, not a tuning sweep")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("nonfinite prediction or target")
    error = prediction - target
    pointwise = error.square().mean()
    i, j = torch.triu_indices(5, 5, offset=1, device=prediction.device)
    pairwise = (error[:, i] - error[:, j]).square().mean() / 2
    return {"loss": pointwise + pair_weight * pairwise,
            "pointwise": pointwise, "effect_difference": pairwise}


def training_target_stats(targets):
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("complete five-candidate training groups required")
    mean, std = float(values.mean()), float(values.std())
    if std <= 1e-8:
        raise ValueError("training targets constant; report data limitation rather than fake labels")
    return {"mean": mean, "std": std, "source": "new_training_contexts_only"}


class PairPack:
    """Split metadata and future labels cannot be passed to the scorer forward.

    Schema-probe data are deliberately unusable as training data. Splits must
    be assigned by source/reset seed before deriving contexts and candidates.
    """

    def __init__(self, root, *, for_training=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("incompatible pair pack")
        if for_training and (self.manifest["data_role"] != "train"
                             or not self.manifest["training_allowed"]):
            raise ValueError("development schema probe is not an independent training pack")
        with np.load(self.root / "observations.npz", allow_pickle=False) as a:
            if set(a.files) != set(INPUT_FIELDS):
                raise ValueError("only current grid, observable agent XY and candidate actions are inputs")
            self.observations = {name: a[name].copy() for name in INPUT_FIELDS}
        with np.load(self.root / "targets.npz", allow_pickle=False) as a:
            if a.files != ["terminal_coverage"]:
                raise ValueError("future coverage belongs only in the target sidecar")
            self.targets = a["terminal_coverage"].copy()
        self.contexts = [json.loads(x) for x in (self.root / "contexts.jsonl").read_text(encoding="utf-8").splitlines()]
        n = len(self.contexts)
        shapes = {"current": (n, 24, 24), "agent_xy": (n, 2), "actions": (n, 5, 8, 2)}
        if n == 0 or any(self.observations[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("observation rows/shapes do not align with context metadata")
        if self.targets.shape != (n, 5) or not np.isfinite(self.targets).all():
            raise ValueError("all five actual outcomes must be present; no padding")
        if not all(np.isfinite(v).all() for v in self.observations.values()):
            raise ValueError("nonfinite input")
        for name, low, high in (("current", 0, 1), ("agent_xy", 0, 512), ("actions", 0, 512)):
            if ((self.observations[name] < low) | (self.observations[name] > high)).any():
                raise ValueError(f"{name} outside unchanged native range")
        if ((self.targets < 0) | (self.targets > 1)).any():
            raise ValueError("invalid measured coverage")
        keys = {(row["source_seed"], row["anchor_step"]) for row in self.contexts}
        if len(keys) != n or self.manifest["contexts"] != n:
            raise ValueError("duplicate or misaligned context metadata")

    def inputs(self, rows, *, device="cpu"):
        return tuple(torch.as_tensor(self.observations[name][rows], dtype=torch.float32, device=device)
                     for name in INPUT_FIELDS)


def require_disjoint_sources(*packs):
    used = set()
    for pack in packs:
        ids = {r["source_seed"] for r in pack.contexts}
        if used & ids:
            raise ValueError("source/reset seed crosses pair-pack split boundary")
        used.update(ids)
