"""Native LIBERO K1 supervised objectives and offline sufficient-stat metrics.

No model, PI0.5, optimizer, backward call, action ranking or training entrypoint.
Targets are separate from model inputs. Metrics are numerical diagnostics and
cannot establish source independence, model quality or real-system readiness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


TARGET_KEYS = frozenset({"future_visual_latent", "future_visual_valid", "state_delta", "state_target_valid"})
PREDICTION_KEYS = frozenset({"pred_future_visual_latent", "pred_state_delta"})
TARGET_FLOAT_KEYS = ("future_visual_latent", "state_delta")
TARGET_MASK_KEYS = ("future_visual_valid", "state_target_valid")
STATE_COORDINATES = (
    "eef_position_x", "eef_position_y", "eef_position_z",
    "eef_axis_angle_x", "eef_axis_angle_y", "eef_axis_angle_z",
    "gripper_qpos_0", "gripper_qpos_1",
)


@dataclass(frozen=True)
class NativeWorldModelLossConfig:
    visual_weight: float = 1.0
    state_weight: float = 0.25
    beta: float = 1.0

    def __post_init__(self):
        for name in ("visual_weight", "state_weight", "beta"):
            value = getattr(self, name)
            if type(value) not in {float, int} or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, not bool/coerced text")
        if self.visual_weight < 0 or self.state_weight < 0 or not (self.visual_weight > 0 or self.state_weight > 0):
            raise ValueError("loss weights must be nonnegative and cannot both be zero")
        if self.beta <= 0:
            raise ValueError("SmoothL1 beta must be positive")


def objective_contract() -> dict:
    return {
        "schema": "pi05_libero_native_world_model_objectives_v1",
        "candidate_scope": "exactly K=1 executed demonstration action candidate; no ranking or candidate averaging",
        "target_keys": sorted(TARGET_KEYS), "prediction_keys": sorted(PREDICTION_KEYS),
        "formula": "visual_weight * valid_scalar_mean(SmoothL1_visual) + state_weight * valid_scalar_mean(SmoothL1_normalized_state)",
        "default_weights": {"visual": 1.0, "normalized_state": 0.25}, "default_smooth_l1_beta": 1.0,
        "visual_denominator": "number of valid horizon/view positions multiplied by D",
        "state_denominator": "number of valid horizon/state-coordinate positions",
        "unsupported_loss_branch": "zero; fail if every positively weighted branch has zero support",
        "unsupported_metrics": "count 0, mae/rmse null; never report unsupported error as zero",
        "masking": "valid prediction and target values finite; invalid prediction and target cleared with where before subtraction",
        "loss_precision": "float32 elementwise SmoothL1; float64 sums/count division; branch means and weighted scalar float32",
        "loss_audit_sums": "detached float64 sums of the float32 per-element SmoothL1 values, not batch-mean averages",
        "metric_precision": "detach float32 values then promote to float64 BEFORE subtraction; row-ordered float64 accumulation",
        "metric_vs_loss_precision": "different elementwise arithmetic paths; not a bitwise-equivalence promise",
        "state_target": "current-relative normalized coordinate residual, not per-step accumulated delta or SO3 relative rotation",
        "native_state_error": "(predicted normalized residual - target normalized residual) * train_state_std; never add mean",
        "native_state_aggregate": "forbidden across mixed coordinate units; report each coordinate and position/axis_angle_coordinates/gripper groups",
        "state_std_provenance": "caller must supply already verified train-only statistics; class validates values, not dataset provenance",
        "constant_coordinate_std_one": "normalizer fallback, not physical calibration evidence",
        "state_coordinates": list(STATE_COORDINATES),
        "training_started_by_module": False, "optimizer_or_backward_in_module": False,
        "policy_actions_or_ranking_in_module": False, "additional_objectives": [],
    }


def _dense_tensor(value: Any, *, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
    if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
            or value.dtype != dtype or value.device != device or tuple(value.shape) != shape):
        raise ValueError(f"{name} requires dense {dtype} shape={shape} on {device}; no implicit broadcasting")


def _validate(predictions: dict, targets: dict):
    if type(predictions) is not dict or set(predictions) != PREDICTION_KEYS:
        raise ValueError("predictions require exactly the two native prediction keys")
    if type(targets) is not dict or set(targets) != TARGET_KEYS:
        raise ValueError("targets require exactly four native target keys; metadata/oracle fields forbidden")
    visual_pred = predictions["pred_future_visual_latent"]
    if not isinstance(visual_pred, torch.Tensor) or visual_pred.ndim != 5:
        raise ValueError("visual predictions require [B,1,H,2,D]")
    b, k, h, views, d = visual_pred.shape
    if min(b, h, d) <= 0 or k != 1 or views != 2:
        raise ValueError("native supervised objective requires nonempty B/H/D, two views and exactly K=1")
    device = visual_pred.device
    _dense_tensor(visual_pred, name="pred_future_visual_latent", shape=(b, 1, h, 2, d), dtype=torch.float32, device=device)
    _dense_tensor(predictions["pred_state_delta"], name="pred_state_delta", shape=(b, 1, h, 8), dtype=torch.float32, device=device)
    for key, shape, dtype in (
        ("future_visual_latent", (b, h, 2, d), torch.float32),
        ("future_visual_valid", (b, h, 2), torch.bool),
        ("state_delta", (b, h, 8), torch.float32),
        ("state_target_valid", (b, h, 8), torch.bool),
    ):
        _dense_tensor(targets[key], name=key, shape=shape, dtype=dtype, device=device)
        if targets[key].requires_grad:
            raise ValueError("targets must be detached; target gradients are forbidden")
    vp, sp = visual_pred[:, 0], predictions["pred_state_delta"][:, 0]
    vt, st = targets["future_visual_latent"], targets["state_delta"]
    vm, sm = targets["future_visual_valid"], targets["state_target_valid"]
    for name, value, mask in (("visual prediction", vp, vm), ("visual target", vt, vm),
                              ("state prediction", sp, sm), ("state target", st, sm)):
        if not bool(torch.isfinite(value[mask]).all()):
            raise ValueError(f"valid {name} values must be finite")
    # Never subtract invalid NaN/Inf or multiply them by zero after arithmetic.
    return (torch.where(vm[..., None], vp, 0.0), torch.where(vm[..., None], vt, 0.0),
            torch.where(sm, sp, 0.0), torch.where(sm, st, 0.0), vm, sm, (b, h, d))


def native_world_model_loss(predictions: dict, targets: dict, config: NativeWorldModelLossConfig | None = None):
    config = NativeWorldModelLossConfig() if config is None else config
    if not isinstance(config, NativeWorldModelLossConfig):
        raise ValueError("NativeWorldModelLossConfig required")
    config.__post_init__()
    vp, vt, sp, st, vm, sm, (_, _, d) = _validate(predictions, targets)
    visual_count = int(vm.sum().item()) * d
    state_count = int(sm.sum().item())
    if not ((config.visual_weight > 0 and visual_count > 0) or (config.state_weight > 0 and state_count > 0)):
        raise ValueError("no observed support in any positively weighted loss branch")
    visual_elements = F.smooth_l1_loss(vp, vt, reduction="none", beta=float(config.beta))
    state_elements = F.smooth_l1_loss(sp, st, reduction="none", beta=float(config.beta))
    if not bool(torch.isfinite(visual_elements).all()) or not bool(torch.isfinite(state_elements).all()):
        raise ValueError("float32 SmoothL1 elementwise overflow or nonfinite output")
    visual_sum = visual_elements.sum(dtype=torch.float64)
    state_sum = state_elements.sum(dtype=torch.float64)
    visual_loss = (visual_sum / max(visual_count, 1)).to(torch.float32)
    state_loss = (state_sum / max(state_count, 1)).to(torch.float32)
    loss = config.visual_weight * visual_loss + config.state_weight * state_loss
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise ValueError("weighted float32 loss overflow or nonfinite output")
    return loss, {"visual_loss": visual_loss, "state_loss": state_loss,
                  "visual_sum": visual_sum.detach(), "state_sum": state_sum.detach(),
                  "visual_count": visual_count, "state_count": state_count}


def collate_window_targets(targets_list: list[dict], device: str | torch.device = "cpu") -> dict:
    if type(targets_list) is not list or not targets_list:
        raise ValueError("targets_list requires a nonempty list of native target dictionaries")
    for item in targets_list:
        if type(item) is not dict or set(item) != TARGET_KEYS:
            raise ValueError("only four native target keys accepted; inputs/metadata/oracle fields forbidden")
        for key in TARGET_FLOAT_KEYS:
            if not isinstance(item[key], np.ndarray) or item[key].dtype.kind != "f":
                raise ValueError(f"{key} requires a floating NumPy array")
        for key in TARGET_MASK_KEYS:
            if not isinstance(item[key], np.ndarray) or item[key].dtype != np.bool_:
                raise ValueError(f"{key} requires a boolean NumPy mask, never numeric coercion")
    shapes = {key: targets_list[0][key].shape for key in TARGET_KEYS}
    if any(item[key].shape != shape for item in targets_list for key, shape in shapes.items()):
        raise ValueError("target shapes must agree across batch rows")
    visual = shapes["future_visual_latent"]
    if (len(visual) != 3 or min(visual) <= 0 or visual[1] != 2
            or shapes["future_visual_valid"] != visual[:2] or shapes["state_delta"] != (visual[0], 8)
            or shapes["state_target_valid"] != (visual[0], 8)):
        raise ValueError("native targets require [H,2,D], [H,2], [H,8], [H,8] without candidate broadcasting")
    batch = {key: torch.tensor(np.stack([item[key] for item in targets_list]),
                              dtype=torch.bool if key in TARGET_MASK_KEYS else torch.float32, device=device)
             for key in TARGET_KEYS}
    if not bool(torch.isfinite(batch["future_visual_latent"][batch["future_visual_valid"]]).all()):
        raise ValueError("valid collated visual targets must be finite float32")
    if not bool(torch.isfinite(batch["state_delta"][batch["state_target_valid"]]).all()):
        raise ValueError("valid collated state targets must be finite float32")
    return batch


def _stats(absolute_sum, squared_sum, count) -> dict:
    absolute_sum, squared_sum, count = float(np.sum(absolute_sum, dtype=np.float64)), float(np.sum(squared_sum, dtype=np.float64)), int(np.sum(count, dtype=np.int64))
    if not math.isfinite(absolute_sum) or not math.isfinite(squared_sum):
        raise ValueError("metric summary float64 sum overflow")
    return {"count": count, "absolute_error_sum": absolute_sum, "squared_error_sum": squared_sum,
            "mae": absolute_sum / count if count else None,
            "rmse": math.sqrt(squared_sum / count) if count else None}


class NativeWorldModelMetrics:
    """Detached row-ordered float64 sums/counts; never means of batch means."""

    def __init__(self, state_std):
        if type(state_std) is list:
            if any(type(value) not in {int, float} or isinstance(value, bool) for value in state_std):
                raise ValueError("state_std list requires eight finite positive numeric values")
        elif not isinstance(state_std, np.ndarray) or state_std.dtype.kind not in "fi":
            raise ValueError("state_std must be an eight-dimensional numeric list or NumPy array")
        std = np.array(state_std, dtype=np.float64, copy=True)
        if std.shape != (8,) or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("state_std requires exactly eight finite positive train-only standard deviations")
        self._state_std = tuple(map(float, std))
        self._horizon = self._visual_dim = None
        self._accumulators = None
        self._window_count = self._update_count = 0

    @torch.no_grad()
    def update(self, predictions: dict, targets: dict) -> None:
        vp, vt, sp, st, vm, sm, (b, h, d) = _validate(predictions, targets)
        if self._horizon is not None and (h, d) != (self._horizon, self._visual_dim):
            raise ValueError("metric horizon/visual width cannot change across updates")
        visual_error = (vp.detach().to(torch.float64) - vt.detach().to(torch.float64)).cpu().numpy()
        state_error = (sp.detach().to(torch.float64) - st.detach().to(torch.float64)).cpu().numpy()
        visual_valid = vm.detach().cpu().numpy().astype(np.int64)
        state_valid = sm.detach().cpu().numpy().astype(np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            native_error = state_error * np.asarray(self._state_std)
            batches = {
                "visual_abs": np.abs(visual_error).sum(axis=-1, dtype=np.float64),
                "visual_sq": np.square(visual_error).sum(axis=-1, dtype=np.float64),
                "visual_count": visual_valid * d,
                "state_abs": np.abs(state_error), "state_sq": np.square(state_error), "state_count": state_valid,
                "native_abs": np.abs(native_error), "native_sq": np.square(native_error),
            }
        if any(not np.isfinite(value).all() for value in batches.values()):
            raise ValueError("float64 metric error, squared error or native scale overflow")
        if self._accumulators is None:
            accumulators = {key: np.zeros(value.shape[1:], dtype=value.dtype) for key, value in batches.items()}
        else:
            accumulators = {key: value.copy() for key, value in self._accumulators.items()}
        # Maintain identical addition order for the same window sequence even
        # when callers partition it into uneven or differently masked batches.
        with np.errstate(over="ignore", invalid="ignore"):
            for row in range(b):
                for key, value in batches.items():
                    accumulators[key] += value[row]
        if any(not np.isfinite(value).all() for value in accumulators.values()):
            raise ValueError("float64 accumulated metric sum overflow")
        self._accumulators = accumulators
        self._horizon, self._visual_dim = h, d
        self._window_count += b
        self._update_count += 1

    def summary(self) -> dict:
        base = {
            "schema": "pi05_libero_native_world_model_metrics_v1", "update_count": self._update_count,
            "window_count": self._window_count, "horizon": self._horizon, "visual_dim": self._visual_dim,
            "state_std": list(self._state_std), "state_coordinates": list(STATE_COORDINATES),
            "metric_precision": "detached float64 errors and row-ordered sufficient-stat accumulation",
            "state_native_semantics": "current-relative coordinate error scaled only by train std; axis-angle coordinates are not SO3 rotation error",
            "state_std_source_independently_verified_by_metric": False,
        }
        if self._accumulators is None:
            a = {key: np.zeros((0, 2 if key.startswith("visual") else 8), dtype=np.int64 if key.endswith("count") else np.float64)
                 for key in ("visual_abs", "visual_sq", "visual_count", "state_abs", "state_sq", "state_count", "native_abs", "native_sq")}
            h = 0
        else:
            a, h = self._accumulators, self._horizon

        def stat(prefix, selector=...):
            count_key = "visual_count" if prefix == "visual" else "state_count"
            return _stats(a[f"{prefix}_abs"][selector], a[f"{prefix}_sq"][selector], a[count_key][selector])

        base["visual"] = {
            "aggregate": stat("visual"), "per_horizon": [stat("visual", step) for step in range(h)],
            "per_view": [stat("visual", (slice(None), view)) for view in range(2)],
            "per_horizon_view": [[stat("visual", (step, view)) for view in range(2)] for step in range(h)],
        }
        base["state_normalized"] = {
            "aggregate": stat("state"), "per_horizon": [stat("state", step) for step in range(h)],
            "per_coordinate": [stat("state", (slice(None), coordinate)) for coordinate in range(8)],
            "per_horizon_coordinate": [[stat("state", (step, coordinate)) for coordinate in range(8)] for step in range(h)],
        }
        base["state_native"] = {
            "per_coordinate": [stat("native", (slice(None), coordinate)) for coordinate in range(8)],
            "per_horizon_coordinate": [[stat("native", (step, coordinate)) for coordinate in range(8)] for step in range(h)],
            "groups": {name: stat("native", (slice(None), coordinates)) for name, coordinates in (
                ("position", slice(0, 3)), ("axis_angle_coordinates", slice(3, 6)), ("gripper", slice(6, 8)))},
        }
        return base
