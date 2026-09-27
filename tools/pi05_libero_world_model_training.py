"""Guarded CPU training primitives for the separately authorized 200-step pilot.

This module does not acquire data or start a run. Final checkpoints are audit
artifacts, not resumable training sessions. Existing native model/loss are used
unchanged; PI0.5, guidewire models and rollout execution are absent.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch

from pi05_libero_world_model import INPUT_KEYS, LiberoWorldModel
from pi05_libero_world_model_objectives import native_world_model_loss
from smoke_pi05_libero_world_model import parameter_sha


OPTIMIZER_CONFIG = {
    "name": "AdamW", "lr": 0.001, "weight_decay": 0.0,
    "betas": [0.9, 0.999], "eps": 1e-8, "amsgrad": False,
    "foreach": False, "fused": False, "maximize": False,
}
CHECKPOINT_SCHEMA = "pi05_libero_native_final_diagnostic_checkpoint_v1"
FINAL_STEP = 200
HASH_KEYS = frozenset({
    "protocol_sha256", "authorization_sha256", "source_report_sha256",
    "feature_report_sha256", "manifest_sha256", "split_sha256",
    "normalization_sha256", "sampling_plan_sha256",
    "initial_parameter_sha256", "final_parameter_sha256",
})
METADATA_KEYS = HASH_KEYS | {"model_config", "task_registry", "step", "checkpoint_kind"}


def _json(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("finite JSON-compatible metadata/config required") from error


def _digest(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA256 must contain exactly 64 lowercase hex characters")
    return value


def _model_parameters(model, *, trainable: bool) -> list[tuple[str, torch.nn.Parameter]]:
    if not isinstance(model, LiberoWorldModel):
        raise ValueError("existing native LiberoWorldModel required")
    params = list(model.named_parameters())
    if not params:
        raise ValueError("native model must have parameters")
    for name, value in params:
        if (value.device.type != "cpu" or value.dtype != torch.float32 or value.layout != torch.strided
                or not bool(torch.isfinite(value).all()) or (trainable and not value.requires_grad)):
            raise ValueError(f"{name}: finite dense CPU float32 parameters, all trainable during updates, required")
    return params


def _optimizer_group(group: dict) -> None:
    expected = {key: value for key, value in OPTIMIZER_CONFIG.items() if key != "name"}
    expected["betas"] = tuple(expected["betas"])
    expected.update(capturable=False, differentiable=False)
    required = set(expected) | {"params"}
    # PyTorch versions expose this AdamW invariant either explicitly or through
    # the class implementation. Never allow it to become False when present.
    if set(group) not in (required, required | {"decoupled_weight_decay"}):
        raise ValueError("optimizer parameter-group fields differ from fixed AdamW contract")
    for key, value in expected.items():
        actual = group[key]
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"optimizer setting differs from fixed protocol: {key}")
    if "decoupled_weight_decay" in group and group["decoupled_weight_decay"] is not True:
        raise ValueError("AdamW decoupled weight decay must remain enabled")


def _state_entry(state: dict, parameter: torch.Tensor, expected_step: int) -> None:
    if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("AdamW state requires step/exp_avg/exp_avg_sq for every parameter")
    step = state["step"]
    if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
            or step.device.type != "cpu" or step.dtype != torch.float32
            or not bool(torch.isfinite(step)) or step.item() != expected_step):
        raise ValueError(f"every AdamW counter must equal {expected_step}")
    for key in ("exp_avg", "exp_avg_sq"):
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.device.type != "cpu"
                or value.layout != torch.strided or value.shape != parameter.shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"finite CPU float32 correctly shaped AdamW {key} required")
    if bool((state["exp_avg_sq"] < 0).any()):
        raise ValueError("AdamW squared moment cannot be negative")


def _optimizer(optimizer, params, expected_step: int) -> None:
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("one fixed AdamW parameter group required")
    group = optimizer.param_groups[0]
    _optimizer_group(group)
    expected = [value for _, value in params]
    if len(group["params"]) != len(expected) or any(actual is not wanted for actual, wanted in zip(group["params"], expected)):
        raise ValueError("optimizer must own every native parameter exactly once in model order")
    if expected_step == 0:
        if optimizer.state:
            raise ValueError("first update requires empty fresh AdamW state")
    else:
        if set(optimizer.state) != set(expected):
            raise ValueError("AdamW state missing or adding parameters")
        for parameter in expected:
            _state_entry(optimizer.state[parameter], parameter, expected_step)


def build_optimizer(model, configdict: dict) -> torch.optim.AdamW:
    if type(configdict) is not dict or _json(configdict) != _json(OPTIMIZER_CONFIG):
        raise ValueError("optimizer config must exactly match all nine frozen protocol fields")
    params = _model_parameters(model, trainable=True)
    optimizer = torch.optim.AdamW(
        [value for _, value in params], lr=0.001, weight_decay=0.0,
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False, foreach=False,
        fused=False, maximize=False, capturable=False, differentiable=False,
    )
    _optimizer(optimizer, params, 0)
    return optimizer


def _parameter_summaries(params) -> list[tuple[float, float]]:
    with torch.no_grad():
        return [(float(value.double().sum()), float(value.double().square().sum())) for _, value in params]


def _gradient_norm(params) -> float:
    total = 0.0
    for name, value in params:
        grad = value.grad
        if grad is None or grad.layout != torch.strided or not bool(torch.isfinite(grad).all()):
            raise ValueError(f"missing or nonfinite gradient: {name}")
        total += float(grad.detach().double().square().sum())
    norm = math.sqrt(total)
    if not math.isfinite(norm):
        raise ValueError("global gradient norm is not finite")
    return norm


def train_step(model, optimizer, inputs: dict, targets: dict, *, clip_norm: float = 1.0,
               expected_step: int) -> dict:
    if type(expected_step) is not int or not 1 <= expected_step <= FINAL_STEP:
        raise ValueError("expected_step must be a 1-based integer within the fixed 200-update budget")
    if type(clip_norm) not in {int, float} or clip_norm != 1.0:
        raise ValueError("fixed global L2 gradient clipping norm is exactly 1.0")
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise ValueError("training update requires enabled gradients, never inference mode")
    if not all(module.training for module in model.modules()):
        raise ValueError("every native model module must be in train mode")
    params = _model_parameters(model, trainable=True)
    _optimizer(optimizer, params, expected_step - 1)
    if type(inputs) is not dict or set(inputs) != INPUT_KEYS:
        raise ValueError("exactly six native inputs required; targets/metadata cannot enter model")
    if type(targets) is not dict:
        raise ValueError("native targets must be a separate dictionary")
    for owner, batch in (("inputs", inputs), ("targets", targets)):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and (value.requires_grad or torch.is_inference(value)):
                raise ValueError(f"{owner}.{key}: ordinary detached tensors required; only model parameters may receive gradients")
    before = _parameter_summaries(params)
    optimizer.zero_grad(set_to_none=True)
    predictions = model(**inputs)
    loss, details = native_world_model_loss(predictions, targets)
    if loss.ndim != 0 or not loss.requires_grad or loss.grad_fn is None or not bool(torch.isfinite(loss)):
        raise ValueError("finite scalar differentiable native loss required")
    loss.backward()
    pre_norm = _gradient_norm(params)
    if pre_norm <= 0:
        raise ValueError("zero global gradient: no effective learning update")
    torch.nn.utils.clip_grad_norm_([value for _, value in params], max_norm=1.0, norm_type=2.0,
                                  error_if_nonfinite=True, foreach=False)
    post_norm = _gradient_norm(params)
    if post_norm <= 0 or post_norm > 1.0 + 1e-6:
        raise ValueError("clipped gradient must remain positive and within the fixed global norm")
    optimizer.step()
    _model_parameters(model, trainable=True)
    _optimizer(optimizer, params, expected_step)
    changed = before != _parameter_summaries(params)
    if not changed:
        raise ValueError("parameter value summaries unchanged after the actual optimizer update")
    return {
        "step": expected_step, "loss": float(loss.detach()),
        "visual_loss": float(details["visual_loss"].detach()), "state_loss": float(details["state_loss"].detach()),
        "visual_sum": float(details["visual_sum"]), "state_sum": float(details["state_sum"]),
        "visual_count": details["visual_count"], "state_count": details["state_count"],
        "global_grad_norm_before_clip": pre_norm, "global_grad_norm_after_clip": post_norm,
        "parameters_with_grad": len(params), "parameter_changed": changed,
        "parameter_change_check": "per-parameter float64 sum and squared sum; runner separately verifies final full SHA256",
    }


def _metadata(metadata: dict, model) -> None:
    if type(metadata) is not dict or set(metadata) != METADATA_KEYS:
        raise ValueError("final checkpoint requires exact provenance metadata keys")
    for key in HASH_KEYS:
        _digest(metadata[key])
    if metadata["initial_parameter_sha256"] == metadata["final_parameter_sha256"]:
        raise ValueError("final checkpoint must differ from initial parameters")
    if type(metadata["step"]) is not int or metadata["step"] != FINAL_STEP:
        raise ValueError("only final update200 checkpoint is supported")
    if metadata["checkpoint_kind"] != "final_diagnostic_no_resume":
        raise ValueError("checkpoint is final diagnostic only, never resumable")
    if _json(metadata["model_config"]) != _json(asdict(model.config)):
        raise ValueError("checkpoint model_config mismatch")
    registry = [{key: entry[key] for key in ("task_id", "source_task_index", "task_instruction")}
                for entry in model.metadata()["task_registry_encoding"]]
    given = metadata["task_registry"]
    if (type(given) is not list or any(type(entry) is not dict for entry in given)
            or _json(sorted(given, key=lambda entry: entry.get("task_id", -1))) != _json(registry)):
        raise ValueError("checkpoint task registry must match model exactly; original list order is preserved")
    _json(metadata)


def _optimizer_payload(state: dict, model_state: dict) -> None:
    if type(state) is not dict or set(state) != {"state", "param_groups"}:
        raise ValueError("exact optimizer state payload required")
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != 1:
        raise ValueError("checkpoint requires one optimizer group")
    _optimizer_group(groups[0])
    indices = list(range(len(model_state)))
    if groups[0]["params"] != indices or type(state["state"]) is not dict or set(state["state"]) != set(indices):
        raise ValueError("checkpoint AdamW indices must cover every native parameter exactly once")
    for index, value in enumerate(model_state.values()):
        _state_entry(state["state"][index], value, FINAL_STEP)


def save_final_checkpoint(path, model, optimizer, metadata: dict) -> str:
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(f"checkpoint already exists: {path}")
    params = _model_parameters(model, trainable=False)
    _optimizer(optimizer, params, FINAL_STEP)
    _metadata(metadata, model)
    if parameter_sha(model) != metadata["final_parameter_sha256"]:
        raise ValueError("final parameter SHA256 differs from metadata")
    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {"schema": CHECKPOINT_SCHEMA, "metadata": deepcopy(metadata),
               "model_state_dict": model_state, "optimizer_state_dict": deepcopy(optimizer.state_dict())}
    _optimizer_payload(payload["optimizer_state_dict"], model_state)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-filesystem hard-link publication is atomic and rejects existing
        # destinations on both Windows and Linux; rename/replace could overwrite.
        os.link(temporary, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_final_checkpoint(path, model, expected_metadata: dict, expected_sha256: str) -> dict:
    expected_sha256 = _digest(expected_sha256)
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("checkpoint file SHA256 mismatch before deserialization")
    _model_parameters(model, trainable=False)
    _metadata(expected_metadata, model)
    # Read the same already-hashed bytes, avoiding a hash/load path race. Never
    # deserialize unknown Python objects or load a checkpoint with weights_only=False.
    import io
    payload = torch.load(io.BytesIO(content), weights_only=True, map_location="cpu")
    if type(payload) is not dict or set(payload) != {"schema", "metadata", "model_state_dict", "optimizer_state_dict"}:
        raise ValueError("unexpected final checkpoint payload fields")
    if payload["schema"] != CHECKPOINT_SCHEMA or _json(payload["metadata"]) != _json(expected_metadata):
        raise ValueError("checkpoint schema/provenance metadata mismatch")
    state = payload["model_state_dict"]
    expected_state = model.state_dict()
    if type(state) is not dict or list(state) != list(expected_state):
        raise ValueError("checkpoint native parameter keys/order mismatch")
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type != "cpu"
                or value.dtype != torch.float32 or value.shape != expected_state[key].shape
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"invalid native checkpoint parameter: {key}")
    _optimizer_payload(payload["optimizer_state_dict"], state)
    model.load_state_dict(state, strict=True)
    if parameter_sha(model) != expected_metadata["final_parameter_sha256"]:
        raise ValueError("reloaded final parameter SHA256 differs from metadata")
    return payload
