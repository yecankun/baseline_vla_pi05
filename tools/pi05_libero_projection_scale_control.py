"""Fixed visual-projection output scale control; interface/numerics only.

The normalized native model's forward, tanh, loss and raw visual skip are reused
unchanged. Only each visual Linear's output is transformed. This per-vector
operator neither centers features nor uses batch statistics or learned scales.
Its RMS bound does NOT prevent individual coordinates from saturating tanh.
Old native/normalization checkpoints do not encode this new computation: a
separate schema and persistent version marker are required, not retrofitting an
old checkpoint and relabelling the result as its original prediction.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

if __package__ in {None, ""}:
    from pi05_libero_visual_normalization import NormalizedLiberoWorldModel
else:
    from .pi05_libero_visual_normalization import NormalizedLiberoWorldModel


SCHEMA = "pi05_libero_projection_scale_control_world_model_v1"
FORMULA = "z / sqrt(1 + mean(z**2, dim=-1, keepdim=True))"
VERSION = 1
VERSION_BUFFER = "projection_scale_control_version"
COMPUTE_DTYPE = "float64_intermediate_float32_output"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def smooth_projection_scale(z: torch.Tensor) -> torch.Tensor:
    """Apply the fixed formula over the final hidden coordinate axis.

    Keeps device, shape, dtype and autograd connectivity; never writes ``z``.
    Float64 square/mean/sqrt/division supports large finite float32 inputs
    without square overflow. Nonfinite intermediates still fail before division,
    so an infinite denominator cannot silently produce a false zero vector.
    No epsilon, clipping, learned gain, centering or alternative computation is
    silently substituted. The output is cast back to float32; its rounding
    applies to the mathematical RMS<=1 statement, and individual coordinates
    can exceed1. Large-input gradients may underflow on their float32 return
    path: no representable gradient lower bound is claimed.
    """
    _require(isinstance(z, torch.Tensor) and z.layout == torch.strided and z.dtype == torch.float32
             and z.ndim >= 1 and z.numel() > 0 and z.shape[-1] > 0,
             "projection scale requires a nonempty dense float32 tensor with a last hidden axis")
    _require(bool(torch.isfinite(z).all()), "projection scale input must be finite")
    work = z.to(torch.float64)
    squared = work.square()
    _require(bool(torch.isfinite(squared).all()), "projection square overflow; scaling refused")
    mean_squared = squared.mean(dim=-1, keepdim=True)
    _require(bool(torch.isfinite(mean_squared).all()), "projection mean-square reduction overflow; scaling refused")
    denominator = torch.sqrt(1.0 + mean_squared)
    _require(bool(torch.isfinite(denominator).all()) and bool((denominator >= 1).all()),
             "projection scale denominator must remain finite and at least1")
    result = (work / denominator).to(torch.float32)
    _require(bool(torch.isfinite(result).all()), "projection scale output must remain finite")
    return result


class ScaledVisualLinear(nn.Linear):
    """Independent copy of a native Linear, with the fixed post-Linear operator.

    Calling only Module.__init__ avoids Linear.reset_parameters and consumes no
    RNG. Parameters remain ``weight`` then ``bias`` with identical initial bytes,
    shapes, devices and requires_grad flags; no original layer/storage survives.
    Inheriting Linear retains its public in_features/out_features interface.
    """

    def __init__(self, linear: nn.Linear):
        _require(type(linear) is nn.Linear, "wrap exactly one original torch.nn.Linear, not an already scaled/custom layer")
        _require(type(linear.in_features) is int and type(linear.out_features) is int
                 and linear.in_features > 0 and linear.out_features > 0, "positive native Linear dimensions required")
        for name, shape in (("weight", (linear.out_features, linear.in_features)), ("bias", (linear.out_features,))):
            value = getattr(linear, name)
            if name == "bias" and value is None:
                continue
            _require(isinstance(value, nn.Parameter) and value.layout == torch.strided and value.dtype == torch.float32
                     and tuple(value.shape) == shape and not torch.is_inference(value) and bool(torch.isfinite(value).all()),
                     "original Linear parameters must be ordinary finite float32 tensors with native shape")
        _require(not torch.is_inference_mode_enabled(), "construct independent trainable Linear parameters outside inference_mode")
        nn.Module.__init__(self)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=linear.weight.requires_grad)
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=linear.bias.requires_grad)
        self.training = linear.training

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return smooth_projection_scale(F.linear(input, self.weight, self.bias))


def _check_version(value, *, device=None):
    _require(isinstance(value, torch.Tensor) and value.layout == torch.strided
             and value.dtype == torch.int64 and value.ndim == 0 and not value.requires_grad
             and (device is None or value.device == device) and bool(value == VERSION),
             "projection scale checkpoint/version marker missing, modified or incompatible")


class ProjectionScaledLiberoWorldModel(NormalizedLiberoWorldModel):
    """Original normalized forward with only its two visual Linears wrapped.

    No trainable parameter or RNG draw is added. The sole additional state is an
    int64 compatibility marker, not a learned parameter. Checkpoints must bind
    this new model schema/formula as well as the inherited normalization source.
    """

    def __init__(self, config, task_registry, statistics):
        super().__init__(config, task_registry, statistics)
        self.view_projections = nn.ModuleList([ScaledVisualLinear(layer) for layer in self.view_projections])
        self.register_buffer(VERSION_BUFFER, torch.tensor(VERSION, dtype=torch.int64,
                             device=next(self.parameters()).device), persistent=True)

    def metadata(self):
        result = super().metadata()
        result["schema"] = SCHEMA
        result["projection_scale_control"] = dict(formula=FORMULA, version=VERSION, version_buffer=VERSION_BUFFER,
            compute_dtype=COMPUTE_DTYPE,
            fixed_additive_constant=1.0, normalization_axis="last_hidden_dimension", trainable_scale=False,
            centering=False, batch_statistics=False, applied_layers=["view_projections.0", "view_projections.1"],
            placement="after visual Linear and before unchanged native tanh", raw_visual_skip_unchanged=True,
            state_actions_targets_loss_unchanged=True, extra_trainable_parameters=0,
            rms_bound_implies_coordinate_nonsaturation=False, requires_new_checkpoint_schema=True,
            old_native_or_normalization_only_checkpoint_compatible=False,
            representable_gradient_lower_bound_claimed=False,
            training_or_quality_claim_from_numerical_smoke=False)
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        _require(strict is True and assign is False and isinstance(state_dict, Mapping),
                 "projection scale checkpoints require strict owned loading")
        _check_version(state_dict.get(VERSION_BUFFER))
        # Retains constructor-bound visual_mean/visual_scale verification and
        # strict native parameter names/shapes from the unchanged parent loader.
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def forward(self, *, history_visual_latent, history_visual_valid, history_state, history_state_valid,
                task_instruction, candidate_actions):
        _check_version(getattr(self, VERSION_BUFFER, None), device=next(self.parameters()).device)
        return super().forward(history_visual_latent=history_visual_latent, history_visual_valid=history_visual_valid,
            history_state=history_state, history_state_valid=history_state_valid,
            task_instruction=task_instruction, candidate_actions=candidate_actions)
