"""Parameter-free native LIBERO persistence reference, not a learned policy.

Accepts the existing native six-field input contract but predicts only from
the latest visual observation. Future targets remain a separate scoring input.
No model construction, PI0.5 loading, action selection or optimization.
"""
from __future__ import annotations

import torch

if __package__ in {None, ""}:
    from pi05_libero_world_model import INPUT_KEYS, _tensor
    from pi05_libero_world_model_objectives import _validate as _validate_targets
else:
    from .pi05_libero_world_model import INPUT_KEYS, _tensor
    from .pi05_libero_world_model_objectives import _validate as _validate_targets


def persistence_contract() -> dict:
    return {
        "schema": "pi05_libero_native_world_model_persistence_v1",
        "input_keys": sorted(INPUT_KEYS),
        "candidate_scope": "exactly K=1 executed demonstration candidate; no action selection",
        "visual_prediction": "repeat latest history visual latent across H; invalid latest views replaced with zero",
        "state_prediction": "zero current-relative normalized coordinate residual in all eight coordinates",
        "output_contract": {
            "pred_future_visual_latent": "owned float32 [B,1,H,2,D]",
            "pred_state_delta": "owned float32 [B,1,H,8]",
        },
        "prediction_value_dependencies": ["latest history_visual_latent", "latest history_visual_valid"],
        "unused_prediction_values": ["earlier history", "history_state", "candidate_actions", "task_instruction"],
        "input_validation": "strict six keys; native state8/action7/views2; float32/bool; valid history values and every action finite",
        "task_validation": "nonempty text per batch row; no task registry, lookup or semantic encoding",
        "normalization": "already train-normalized input state/actions; no clipping or second normalization",
        "visual_scoring_support": "original future_visual_valid AND latest history_visual_valid, broadcast across H",
        "state_scoring_support": "original state_target_valid AND latest history_state_valid, broadcast across H",
        "target_validation_order": "validate all originally declared valid target values before intersecting masks",
        "invalid_target_values": "clear outside common support using where; return owned detached copies",
        "older_valid_observation_fallback": False,
        "empty_common_support": "mask helper returns zero values/false masks; downstream loss support checks remain unchanged",
        "comparison_requirement": "learned/untrained model must be scored on exactly the same windows and common support",
        "state_residual_is_so3_rotation": False,
        "parameters": 0,
        "model_construction_or_pi05_loading": False,
        "training_or_optimizer_or_backward": False,
        "future_targets_used_for_prediction": False,
        "gradient_tracking": False,
    }


def _validate_inputs(inputs: dict) -> tuple[int, int, int, torch.device]:
    if type(inputs) is not dict or set(inputs) != INPUT_KEYS:
        raise ValueError("persistence requires exactly six native input keys; targets/metadata/oracle forbidden")
    visual = inputs["history_visual_latent"]
    if not isinstance(visual, torch.Tensor) or visual.ndim != 4:
        raise ValueError("history_visual_latent requires [B,T,2,D]")
    b, t, views, d = visual.shape
    if min(b, t, d) <= 0 or views != 2:
        raise ValueError("history requires nonempty B/T/D and exactly two ordered views")
    actions = inputs["candidate_actions"]
    if not isinstance(actions, torch.Tensor) or actions.ndim != 4:
        raise ValueError("candidate_actions requires [B,1,H,7]")
    action_b, k, h, action_dim = actions.shape
    if action_b != b or k != 1 or h <= 0 or action_dim != 7:
        raise ValueError("persistence requires native action7, matching batch, positive H and exactly K=1")
    device = visual.device
    for key, shape, dtype in (
        ("history_visual_latent", (b, t, 2, d), torch.float32),
        ("history_visual_valid", (b, t, 2), torch.bool),
        ("history_state", (b, t, 8), torch.float32),
        ("history_state_valid", (b, t, 8), torch.bool),
        ("candidate_actions", (b, 1, h, 7), torch.float32),
    ):
        _tensor(inputs[key], key, shape, dtype, device)
    tasks = inputs["task_instruction"]
    if type(tasks) is not list or len(tasks) != b or any(type(task) is not str or not task.strip() for task in tasks):
        raise ValueError("task_instruction requires one nonempty string per batch row; no inferred task mapping")
    for value_key, mask_key in (
        ("history_visual_latent", "history_visual_valid"),
        ("history_state", "history_state_valid"),
    ):
        if not bool(torch.isfinite(inputs[value_key][inputs[mask_key]]).all()):
            raise ValueError(f"valid {value_key} values must be finite, including earlier history")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("every candidate action value must be finite")
    return b, h, d, device


@torch.no_grad()
def persistence_predictions(inputs: dict) -> dict[str, torch.Tensor]:
    """Repeat current visual features and predict zero normalized state change."""
    b, h, d, device = _validate_inputs(inputs)
    anchor = torch.where(inputs["history_visual_valid"][:, -1, :, None],
                         inputs["history_visual_latent"][:, -1], 0.0)
    # Clone the expanded view so outputs neither alias input nor share horizon
    # storage. Clearing first prevents invalid NaN/Inf from surviving the anchor.
    return {
        "pred_future_visual_latent": anchor[:, None, None].expand(b, 1, h, 2, d).clone(),
        "pred_state_delta": torch.zeros((b, 1, h, 8), dtype=torch.float32, device=device),
    }


@torch.no_grad()
def persistence_scoring_targets(inputs: dict, targets: dict) -> dict[str, torch.Tensor]:
    """Copy targets onto latest-observation common support; never impute history."""
    predictions = persistence_predictions(inputs)
    # Validate BEFORE intersecting masks: a latest invalid observation must not
    # hide nonfinite future values that the source declared valid. This helper
    # performs no loss reduction, so empty common support is still representable.
    _, visual, _, state, visual_valid, state_valid, _ = _validate_targets(predictions, targets)
    common_visual = visual_valid & inputs["history_visual_valid"][:, -1, None, :]
    common_state = state_valid & inputs["history_state_valid"][:, -1, None, :]
    return {
        "future_visual_latent": torch.where(common_visual[..., None], visual, 0.0).detach().clone(),
        "future_visual_valid": common_visual.detach().clone(),
        "state_delta": torch.where(common_state, state, 0.0).detach().clone(),
        "state_target_valid": common_state.detach().clone(),
    }
