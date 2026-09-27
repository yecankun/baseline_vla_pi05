"""Thin native training/checkpoint guards for the separately gated action study.

No dataset, feature extraction, run loop or optimizer is created at import. The
old native optimizer/loss/update/checkpoint implementations remain unchanged.
Inputs must already be transformed for their arm by action_ablation_inputs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import torch

from pi05_libero_action_ablation import ACTION_ABLATION_MODES, action_ablation_inputs
from pi05_libero_action_study_sampling import canonical_sha256
from pi05_libero_world_model_objectives import TARGET_KEYS, native_world_model_loss
import pi05_libero_world_model_training as native
from smoke_pi05_libero_world_model import parameter_sha


build_optimizer = native.build_optimizer
OPTIMIZER_CONFIG = native.OPTIMIZER_CONFIG
FINAL_STEP = 200
SEEDS = (20260912, 20260913, 20260914)
BINDING_SCHEMA = "pi05_libero_action_study_final_checkpoint_binding_v1"
ENVELOPE_SCHEMA = "pi05_libero_action_study_final_checkpoint_envelope_v1"
BINDING_HASH_KEYS = frozenset({"study_plan_sha256", "training_plan_sha256", "training_authorization_sha256",
                               "sampling_plan_sha256", "initial_parameter_sha256", "final_parameter_sha256"})
BINDING_KEYS = BINDING_HASH_KEYS | {"schema", "seed", "arm", "step", "checkpoint_kind"}
AUTHORIZATION_FIELD_SEMANTICS = (
    "legacy_metadata.authorization_sha256 is the canonical full arm/seed/provenance binding SHA256; "
    "it is not the training authorization file SHA256, which is separately explicit in binding and envelope"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _arm(arm):
    _require(type(arm) is str and arm in ACTION_ABLATION_MODES, "unknown action-study arm")


def _owned_batch(inputs, targets, arm):
    _arm(arm)
    # Reuse the six-input validation/copy boundary without applying a second
    # ablation. Supplying observed values while naming the zero arm is an error.
    owned_inputs = action_ablation_inputs(inputs, "observed_action")
    _require(arm != "normalized_zero_action" or bool((owned_inputs["candidate_actions"] == 0).all()),
             "normalized-zero arm requires already transformed exact-zero candidate actions")
    _require(type(targets) is dict and set(targets) == TARGET_KEYS, "exact separate native targets required")
    for key, value in targets.items():
        _require(isinstance(value, torch.Tensor) and value.layout == torch.strided and value.device.type == "cpu"
                 and not value.requires_grad and not torch.is_inference(value),
                 f"targets.{key} must be ordinary detached dense CPU tensors")
    owned_targets = {key: value.detach().clone() for key, value in targets.items()}
    return owned_inputs, owned_targets


def _batch_signature(batch):
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = {"dtype": str(value.dtype), "shape": list(value.shape), "version": value._version,
                           "bytes_sha256": hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        else:
            result[key] = deepcopy(value)
    return canonical_sha256(result)


def _gradient_details(params, arm):
    rows, total = [], 0.0
    for name, parameter in params:
        grad = parameter.grad
        _require(grad is not None and grad.layout == torch.strided and bool(torch.isfinite(grad).all()),
                 f"missing or nonfinite gradient: {name}")
        squared = float(grad.detach().double().square().sum())
        total += squared
        rows.append({"name": name, "shape": list(grad.shape), "l2_norm": math.sqrt(squared),
                     "finite": True, "exact_zero": bool((grad == 0).all())})
    norm = math.sqrt(total)
    _require(math.isfinite(norm) and norm > 0, "global gradient norm must be finite and positive")
    action = [row for row in rows if row["name"] == "action_projection.weight"]
    _require(len(action) == 1, "unchanged native action_projection.weight required")
    _require(arm != "normalized_zero_action" or action[0]["exact_zero"],
             "normalized-zero action_projection.weight gradient must be exactly zero")
    return {"parameters_with_grad": len(rows), "global_grad_norm": norm,
            "per_parameter_gradients": rows, "action_projection_weight_exact_zero_grad": action[0]["exact_zero"],
            "individual_zero_gradients_allowed": True}


def _zero_hook(model, arm):
    if arm != "normalized_zero_action":
        return None
    def check(gradient):
        _require(bool(torch.isfinite(gradient).all()) and bool((gradient == 0).all()),
                 "normalized-zero action_projection.weight gradient must be exactly zero before optimizer step")
        return gradient
    return model.action_projection.weight.register_hook(check)


def gradient_probe(model, inputs: dict, targets: dict, arm: str) -> dict:
    """One native backward check with no optimizer construction or update.

    Model must already be trainable/in train mode. Parameter bytes and version
    counters, modes and caller input/target bytes remain unchanged. All parameter
    gradients are cleared to None on successful or failing exits.
    """
    hook = None
    try:
        params = native._model_parameters(model, trainable=True)
        before_sha = parameter_sha(model)
        before_versions = [p._version for _, p in params]
        before_modes = [m.training for m in model.modules()]
        _require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), "gradient probe requires enabled ordinary autograd")
        _require(all(before_modes), "gradient probe requires every native module in train mode")
        owned_inputs, owned_targets = _owned_batch(inputs, targets, arm)
        input_sha, target_sha = _batch_signature(inputs), _batch_signature(targets)
        model.zero_grad(set_to_none=True)
        hook = _zero_hook(model, arm)
        predictions = model(**owned_inputs)
        loss, details = native_world_model_loss(predictions, owned_targets)
        _require(loss.ndim == 0 and loss.requires_grad and loss.grad_fn is not None and bool(torch.isfinite(loss)),
                 "finite scalar differentiable native loss required")
        loss.backward()
        gradients = _gradient_details(params, arm)
        _require(parameter_sha(model) == before_sha and [p._version for _, p in params] == before_versions,
                 "gradient-only probe changed native parameter bytes or versions")
        _require([m.training for m in model.modules()] == before_modes, "gradient-only probe changed module mode")
        _require(_batch_signature(inputs) == input_sha and _batch_signature(targets) == target_sha,
                 "gradient-only probe changed caller input/target bytes or versions")
        return {"schema": "pi05_libero_action_study_gradient_probe_v1", "arm": arm,
                "loss": float(loss.detach()), "visual_loss": float(details["visual_loss"].detach()),
                "state_loss": float(details["state_loss"].detach()), **gradients,
                "parameter_sha256_before": before_sha, "parameter_sha256_after": parameter_sha(model),
                "parameter_hash_algorithm": "unchanged smoke_pi05_libero_world_model.parameter_sha",
                "parameter_versions_unchanged": True, "input_target_bytes_versions_unchanged": True,
                "input_signature_sha256": input_sha, "target_signature_sha256": target_sha,
                "gradients_cleared_on_exit": True, "optimizer_constructed": False,
                "backward_calls": 1, "optimizer_steps": 0, "training_started": False}
    finally:
        if hook is not None:
            hook.remove()
        if isinstance(model, torch.nn.Module):
            model.zero_grad(set_to_none=True)


def train_step(model, optimizer, inputs: dict, targets: dict, *, arm: str,
               expected_step: int, clip_norm: float = 1.0) -> dict:
    """Arm validation plus unchanged native update, at most final200.

    For the zero arm, an exact-zero gradient hook fails before optimizer.step.
    The old primitive owns fresh/counted AdamW state, all-gradient finite checks,
    positive global norm, fixed clipping and actual parameter-change checks.
    """
    params = native._model_parameters(model, trainable=True)
    owned_inputs, owned_targets = _owned_batch(inputs, targets, arm)
    input_sha, target_sha = _batch_signature(inputs), _batch_signature(targets)
    hook = _zero_hook(model, arm)
    try:
        result = native.train_step(model, optimizer, owned_inputs, owned_targets,
                                   clip_norm=clip_norm, expected_step=expected_step)
        gradients = _gradient_details(params, arm)
        _require(_batch_signature(inputs) == input_sha and _batch_signature(targets) == target_sha,
                 "native step changed caller input/target bytes or versions")
        return {**result, "arm": arm,
                "action_projection_weight_exact_zero_grad": gradients["action_projection_weight_exact_zero_grad"],
                "per_parameter_gradients_after_clip": gradients["per_parameter_gradients"],
                "individual_zero_gradients_allowed": True, "input_target_bytes_versions_unchanged": True}
    finally:
        if hook is not None:
            hook.remove()


def make_checkpoint_binding(*, seed: int, arm: str, study_plan_sha256: str,
                            training_plan_sha256: str, training_authorization_sha256: str,
                            sampling_plan_sha256: str, initial_parameter_sha256: str,
                            final_parameter_sha256: str) -> dict:
    result = {"schema": BINDING_SCHEMA, "seed": seed, "arm": arm,
              "study_plan_sha256": study_plan_sha256, "training_plan_sha256": training_plan_sha256,
              "training_authorization_sha256": training_authorization_sha256,
              "sampling_plan_sha256": sampling_plan_sha256, "initial_parameter_sha256": initial_parameter_sha256,
              "final_parameter_sha256": final_parameter_sha256, "step": FINAL_STEP,
              "checkpoint_kind": "final_diagnostic_no_resume"}
    _binding(result)
    return result


def _binding(binding):
    _require(type(binding) is dict and set(binding) == BINDING_KEYS, "exact action-study checkpoint binding keys required")
    _require(binding["schema"] == BINDING_SCHEMA, "wrong action-study checkpoint binding schema")
    _require(type(binding["seed"]) is int and binding["seed"] in SEEDS, "checkpoint seed is not one of the frozen study seeds")
    _arm(binding["arm"])
    _require(type(binding["step"]) is int and binding["step"] == FINAL_STEP
             and binding["checkpoint_kind"] == "final_diagnostic_no_resume", "only final200 diagnostic checkpoint binding allowed")
    for key in BINDING_HASH_KEYS:
        native._digest(binding[key])
    _require(binding["initial_parameter_sha256"] != binding["final_parameter_sha256"], "initial/final native parameter hashes must differ")


def _bound_metadata(metadata, binding):
    _binding(binding)
    _require(type(metadata) is dict and set(metadata) == native.METADATA_KEYS, "exact native checkpoint metadata keys required")
    _require(metadata["authorization_sha256"] == binding["training_authorization_sha256"],
             "caller metadata must explicitly contain the actual training authorization file SHA")
    _require(metadata["protocol_sha256"] == binding["study_plan_sha256"],
             "checkpoint protocol SHA must equal the explicitly bound study-plan SHA")
    for key in ("sampling_plan_sha256", "initial_parameter_sha256", "final_parameter_sha256", "step", "checkpoint_kind"):
        _require(type(metadata[key]) is type(binding[key]) and metadata[key] == binding[key], f"checkpoint metadata/binding differs: {key}")
    result = deepcopy(metadata)
    result["authorization_sha256"] = canonical_sha256(binding)
    return result


def save_final_checkpoint(path, model, optimizer, metadata: dict, binding: dict) -> dict:
    """Return an explicit envelope over unchanged atomic-no-overwrite payload.

    Caller metadata names the REAL authorization SHA. Only the copied legacy
    metadata substitutes the full binding digest, making seed/arm inseparable
    from the old restricted checkpoint format. Original metadata is untouched.
    Caller writes the envelope to its own fresh, hash-bound experiment report.
    """
    bound = _bound_metadata(metadata, binding)
    digest = native.save_final_checkpoint(path, model, optimizer, bound)
    return {"schema": ENVELOPE_SCHEMA, "checkpoint_sha256": digest,
            "binding": deepcopy(binding), "binding_sha256": canonical_sha256(binding),
            "training_authorization_sha256": binding["training_authorization_sha256"],
            "legacy_metadata": bound, "authorization_field_semantics": AUTHORIZATION_FIELD_SEMANTICS,
            "parameter_hash_algorithm": "unchanged smoke_pi05_libero_world_model.parameter_sha",
            "checkpoint_is_final_only_not_resumable": True}


def load_final_checkpoint(path, model, envelope: dict, expected_binding: dict, *, expected_metadata: dict) -> dict:
    """Reject wrong arm/seed/provenance before unchanged weights_only loading."""
    _binding(expected_binding)
    expected_keys = {"schema", "checkpoint_sha256", "binding", "binding_sha256", "training_authorization_sha256",
                     "legacy_metadata", "authorization_field_semantics", "parameter_hash_algorithm",
                     "checkpoint_is_final_only_not_resumable"}
    _require(type(envelope) is dict and set(envelope) == expected_keys, "exact action-study checkpoint envelope keys required")
    _require(envelope["schema"] == ENVELOPE_SCHEMA
             and canonical_sha256(envelope["binding"]) == canonical_sha256(expected_binding)
             and envelope["binding_sha256"] == canonical_sha256(expected_binding), "checkpoint arm/seed/provenance binding mismatch")
    _require(envelope["training_authorization_sha256"] == expected_binding["training_authorization_sha256"]
             and envelope["authorization_field_semantics"] == AUTHORIZATION_FIELD_SEMANTICS
             and envelope["parameter_hash_algorithm"] == "unchanged smoke_pi05_libero_world_model.parameter_sha"
             and envelope["checkpoint_is_final_only_not_resumable"] is True, "checkpoint authorization/parameter-hash/final-only semantics differ")
    metadata = envelope["legacy_metadata"]
    _require(type(metadata) is dict and set(metadata) == native.METADATA_KEYS, "exact bound native checkpoint metadata required")
    _require(metadata["authorization_sha256"] == envelope["binding_sha256"], "legacy authorization field must bind exact arm/seed envelope")
    expected_bound = _bound_metadata(expected_metadata, expected_binding)
    _require(canonical_sha256(expected_bound) == canonical_sha256(metadata),
             "checkpoint legacy metadata differs from caller-verified complete provenance")
    return native.load_final_checkpoint(path, model, expected_bound, envelope["checkpoint_sha256"])
