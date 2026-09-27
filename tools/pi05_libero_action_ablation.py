"""Owned six-input transforms for a prospective native LIBERO action ablation.

No model, loss, target, optimizer, checkpoint or dataset is modified here. The
zero arm removes per-window action information in both future train and eval
inputs. Zero in train-normalized coordinates denotes the train action mean;
it is NOT physical zero motion, a hold command, or a counterfactual outcome.
"""
from __future__ import annotations

from typing import Any

import torch

if __package__ in {None, ""}:
    from pi05_libero_world_model import FLOAT_KEYS, INPUT_KEYS, MASK_KEYS
else:
    from .pi05_libero_world_model import FLOAT_KEYS, INPUT_KEYS, MASK_KEYS


ACTION_ABLATION_MODES = ("observed_action", "normalized_zero_action")


def action_ablation_contract() -> dict[str, Any]:
    """Return a fresh, serializable description; this does not authorize a run."""
    return {
        "schema": "pi05_libero_native_action_ablation_inputs_v1",
        "modes": list(ACTION_ABLATION_MODES),
        "input_shapes": {
            "history_visual_latent": "float32 [B,4,2,2048]",
            "history_visual_valid": "bool [B,4,2]",
            "history_state": "float32 [B,4,8]",
            "history_state_valid": "bool [B,4,8]",
            "task_instruction": "list[str] length B; exact source text preserved",
            "candidate_actions": "finite float32 [B,1,3,7]; train-normalized",
        },
        "device": "cpu",
        "ordinary_detached_owned_dense_tensors": True,
        "observed_action": "copy the observed train-normalized actions unchanged",
        "normalized_zero_action": (
            "replace every train-normalized action coordinate by zero in both "
            "train and evaluation; a constant train action mean with no "
            "per-window action information, not physical zero motion or hold"
        ),
        "normalization": "caller must use the same train-only normalization in both arms; no second normalization",
        "normalization_provenance_verified_by_transform": False,
        "task_registry_membership": "validated by the unchanged native model; this transform preserves exact text",
        "masked_history": "preserve masks and invalid values; require finite values only at valid coordinates",
        "invalid_actions": "reject nonfinite actions before ablation, even in the zero arm",
        "targets_or_metadata_accepted": False,
        "arm_marker_passed_to_model": False,
        "zero_arm_gradient_note": (
            "action_projection.weight may have an all-zero gradient because "
            "its input is constant zero; the graph remains connected. This "
            "module does not change the model, loss or gradient guards."
        ),
        "training_started_by_module": False,
        "causal_or_counterfactual_claim_allowed": False,
        "quality_or_generalization_claim_from_schema_smoke": False,
    }


def action_ablation_inputs(inputs: dict, mode: str) -> dict:
    """Validate and copy exactly six native inputs, then apply the chosen arm.

    Call on ordinary, non-gradient CPU batches before a model inference/grad
    context. Both arms return independent storage for all five tensors and an
    owned task list, with identical shapes, dtypes and masks. Targets, metadata,
    arm flags and labels are neither accepted nor returned. The caller applies
    the same arm at train and evaluation time and retains all targets unchanged.

    The zero arm is an information ablation, not a replacement action label or
    a simulated intervention: normalized zero represents the train action mean.
    """
    if type(mode) is not str or mode not in ACTION_ABLATION_MODES:
        raise ValueError(f"mode must be one of {ACTION_ABLATION_MODES}")
    if type(inputs) is not dict or set(inputs) != INPUT_KEYS:
        raise ValueError("only the six native input keys are accepted; targets/metadata/arm markers forbidden")
    if torch.is_inference_mode_enabled():
        raise ValueError("construct ablation inputs outside inference_mode to retain ordinary tensors")

    visual = inputs["history_visual_latent"]
    if not isinstance(visual, torch.Tensor) or visual.ndim != 4 or visual.shape[0] <= 0:
        raise ValueError("history_visual_latent requires a nonempty [B,4,2,2048] tensor")
    batch = visual.shape[0]
    shapes = {
        "history_visual_latent": (batch, 4, 2, 2048),
        "history_visual_valid": (batch, 4, 2),
        "history_state": (batch, 4, 8),
        "history_state_valid": (batch, 4, 8),
        "candidate_actions": (batch, 1, 3, 7),
    }
    for key in (*FLOAT_KEYS, *MASK_KEYS):
        value = inputs[key]
        dtype = torch.bool if key in MASK_KEYS else torch.float32
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"{key} must be a dense strided torch tensor")
        if value.dtype != dtype or value.device.type != "cpu" or tuple(value.shape) != shapes[key]:
            raise ValueError(f"{key} requires {dtype} {shapes[key]} on cpu; no coercion")
        if value.requires_grad or torch.is_inference(value):
            raise ValueError(f"{key} must be an ordinary non-gradient tensor, not an inference tensor")
    tasks = inputs["task_instruction"]
    if (type(tasks) is not list or len(tasks) != batch
            or any(type(text) is not str or not text.strip() for text in tasks)):
        raise ValueError("task_instruction must be a nonempty text list with one string per batch row")

    # Check actions BEFORE zeroing; the information ablation must not conceal
    # corrupt source values. Invalid history values stay masked by the model.
    if not bool(torch.isfinite(inputs["candidate_actions"]).all()):
        raise ValueError("all original candidate actions must be finite before ablation")
    if not bool(torch.isfinite(visual[inputs["history_visual_valid"]]).all()):
        raise ValueError("valid visual coordinates must be finite")
    if not bool(torch.isfinite(inputs["history_state"][inputs["history_state_valid"]]).all()):
        raise ValueError("valid state coordinates must be finite")

    result = {key: inputs[key].detach().clone() for key in (*FLOAT_KEYS, *MASK_KEYS)}
    result["task_instruction"] = list(tasks)
    if mode == "normalized_zero_action":
        result["candidate_actions"].zero_()
    return result
