"""Strict final-only checkpoints for the fixed projection-scale comparison.

This entrypoint never trains, resumes, fits statistics, or adapts an old model
schema. Caller-supplied SHA256 provenance is required at load. The new envelope
distinguishes the normalized reference from the projection-scaled variant even
though their trainable parameter names and capacities are identical.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import torch

import run_pi05_libero_projection_scale_control as gate
import pi05_libero_projection_scale_control as control
import pi05_libero_world_model_training as native
from pi05_libero_visual_normalization import NormalizedLiberoWorldModel, validate_statistics
from pi05_libero_world_model import LiberoWorldModelConfig
from pi05_libero_world_model_adapter import sha256_file


SCHEMA = "libero_projection_scale_control_final_checkpoint_v1"
FINAL_STEP = 200
BINDING_KEYS = frozenset({
    "seed", "arm", "variant", "step", "plan_sha256", "smoke_report_sha256",
    "normalization_sha256", "normalization_applied", "projection_scale_control_applied",
    "draw_sha256", "initial_named_parameter_sha256", "final_named_parameter_sha256", "final_state_sha256",
})
PAYLOAD_KEYS = frozenset({
    "schema", "binding", "config", "registry", "model_state", "normalization",
    "model_metadata", "projection_scale_control", "optimizer_state_saved", "resumable",
})
base = gate.base
named_parameter_sha = gate.previous.named_parameter_sha


def _same_json(actual, expected, message):
    # Canonical JSON distinguishes bool/int, unlike Python dictionary equality.
    base.require(native._json(actual) == native._json(expected), message)


def _binding(binding):
    base.require(type(binding) is dict and set(binding) == BINDING_KEYS, "exact checkpoint binding required")
    base.require(type(binding["seed"]) is int and binding["seed"] in base.SEEDS
                 and type(binding["arm"]) is str and binding["arm"] in base.ARMS
                 and type(binding["variant"]) is str and binding["variant"] in gate.VARIANTS
                 and type(binding["step"]) is int and binding["step"] == FINAL_STEP,
                 "checkpoint seed/arm/variant/final200 differs")
    for key in BINDING_KEYS:
        if key.endswith("sha256"):
            native._digest(binding[key])
    scaled = binding["variant"] == gate.VARIANTS[1]
    base.require(binding["normalization_applied"] is True
                 and binding["projection_scale_control_applied"] is scaled,
                 "checkpoint normalization/scale variant flags differ")
    base.require(binding["initial_named_parameter_sha256"] != binding["final_named_parameter_sha256"],
                 "final checkpoint must differ from initial named parameters")
    return scaled


def _statistics(visual_stats, config, binding):
    validate_statistics(visual_stats, config.views, config.visual_dim)
    serialized = (json.dumps(visual_stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    base.require(hashlib.sha256(serialized).hexdigest() == binding["normalization_sha256"],
                 "checkpoint normalization content digest differs")


def _model(model, scaled):
    expected_class = control.ProjectionScaledLiberoWorldModel if scaled else NormalizedLiberoWorldModel
    expected_layer = control.ScaledVisualLinear if scaled else torch.nn.Linear
    base.require(type(model) is expected_class
                 and len(model.view_projections) == 2
                 and all(type(layer) is expected_layer for layer in model.view_projections),
                 "checkpoint variant/model/layer type differs")
    native._model_parameters(model, trainable=False)
    base.require(not any(module.training for module in model.modules())
                 and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                 "checkpoint requires eval/frozen parameters and cleared gradients")
    if scaled:
        control._check_version(getattr(model, control.VERSION_BUFFER, None), device=torch.device("cpu"))
    else:
        base.require(control.VERSION_BUFFER not in model.state_dict(), "reference cannot contain scale marker")


def _state_into(fresh, state, binding):
    expected = fresh.state_dict()
    base.require(type(state) is dict and set(state) == set(expected), "checkpoint state keys differ")
    for name, value in state.items():
        base.require(isinstance(value, torch.Tensor) and value.layout == torch.strided
                     and value.device.type == "cpu" and not value.requires_grad
                     and value.dtype == expected[name].dtype and value.shape == expected[name].shape
                     and bool(torch.isfinite(value).all()), "checkpoint tensor contract differs: " + name)
    # Enforces constructor-bound means/scales and the new version marker.
    fresh.load_state_dict(state, strict=True, assign=False)
    base.require(base.parameter_hash(fresh) == binding["final_state_sha256"]
                 and named_parameter_sha(fresh) == binding["final_named_parameter_sha256"],
                 "checkpoint final state/named-parameter digest differs")


def _fresh(config_dict, registry, visual_stats, binding):
    base.require(type(config_dict) is dict and set(config_dict) == set(asdict(LiberoWorldModelConfig())),
                 "checkpoint exact native configuration required")
    config = LiberoWorldModelConfig(**config_dict)
    # These alter computation without necessarily changing parameter shapes or
    # initialization SHA. They cannot be inferred from the parameter digest.
    base.require(config.context_len == 4 and config.horizon == 3
                 and type(config.dropout) is float and config.dropout == 0.0,
                 "checkpoint fixed history4/horizon3/dropout0 configuration differs")
    _statistics(visual_stats, config, binding)
    fresh = gate.build_model(binding["seed"], registry, binding["variant"], visual_stats, config)
    base.require(named_parameter_sha(fresh) == binding["initial_named_parameter_sha256"],
                 "checkpoint seed/config/registry initial named-parameter digest differs")
    return fresh


def load_checkpoint(path, *, expected_sha256, expected_binding, registry, visual_stats):
    """Load only the explicitly bound final200 diagnostic; return model/evidence.

    No implicit old-format conversion, optimizer state, nonstrict load, resume,
    or caller-controlled marker insertion is supported. File bytes, envelope,
    config, registry, formula, train-only normalization and all tensors are
    validated before a usable model is returned. The returned model is CPU eval
    with gradients disabled and independent parameter/buffer storage.
    """
    native._digest(expected_sha256)
    scaled = _binding(expected_binding)
    data = Path(path).read_bytes()
    base.require(hashlib.sha256(data).hexdigest() == expected_sha256, "checkpoint file SHA256 differs")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    base.require(type(payload) is dict and set(payload) == PAYLOAD_KEYS and payload["schema"] == SCHEMA,
                 "checkpoint envelope schema differs; old checkpoints are incompatible")
    _binding(payload["binding"])
    _same_json(payload["binding"], expected_binding, "checkpoint provenance binding differs")
    _same_json(payload["registry"], registry, "checkpoint registry differs")
    _same_json(payload["normalization"], visual_stats, "checkpoint normalization payload differs")
    base.require(payload["optimizer_state_saved"] is False and payload["resumable"] is False,
                 "checkpoint is final diagnostic only, never resumable")
    fresh = _fresh(payload["config"], registry, visual_stats, expected_binding)
    metadata = fresh.metadata()
    _same_json(payload["model_metadata"], metadata, "checkpoint model/schema/formula metadata differs")
    _same_json(payload["projection_scale_control"], metadata.get("projection_scale_control") if scaled else None,
               "checkpoint scale formula/version/compute metadata differs")
    _state_into(fresh, payload["model_state"], expected_binding)
    _model(fresh, scaled)
    base.require(sha256_file(path) == expected_sha256, "checkpoint bytes changed during load")
    return fresh, dict(path=Path(path).name, sha256=expected_sha256, schema=SCHEMA,
        binding=deepcopy(expected_binding), projection_scale_control=deepcopy(payload["projection_scale_control"]),
        optimizer_state_saved=False, resumable=False)


def save_reload(path, model, optimizer, *, registry, visual_stats, binding):
    """Validate final200 AdamW state, save exclusively, and strict-reload.

    Optimizer moments/counters are checked in memory but are never serialized.
    Save does not modify the trained model or its optimizer. Existing or failed
    output files are preserved; this helper provides no retry/overwrite path.
    """
    scaled = _binding(binding)
    _model(model, scaled)
    native._optimizer(optimizer, native._model_parameters(model, trainable=False), FINAL_STEP)
    fresh = _fresh(asdict(model.config), registry, visual_stats, binding)
    _same_json(model.metadata(), fresh.metadata(), "checkpoint model metadata differs from bound initialization")
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    _state_into(fresh, state, binding)
    base.require(not Path(path).exists(), "checkpoint overwrite forbidden")
    metadata = model.metadata()
    payload = dict(schema=SCHEMA, binding=deepcopy(binding), config=asdict(model.config),
        registry=deepcopy(registry), model_state=state, normalization=deepcopy(visual_stats),
        model_metadata=deepcopy(metadata),
        projection_scale_control=deepcopy(metadata.get("projection_scale_control")) if scaled else None,
        optimizer_state_saved=False, resumable=False)
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
    digest = sha256_file(path)
    return load_checkpoint(path, expected_sha256=digest, expected_binding=binding,
                           registry=registry, visual_stats=visual_stats)
