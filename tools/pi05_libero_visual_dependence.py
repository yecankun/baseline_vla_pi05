"""Read-only visual-input interventions and actual native residual diagnostics.

No targets, feature extraction, fitting, optimization, or action selection are
accepted here. The caller pins checkpoints and proves that the fixed anchor is
the cached first frame of train episode1312 (the minimum train episode ID).
The helper never estimates an anchor or loads a feature/dataset/checkpoint.

Reported component drifts are separate magnitudes, not additive causal shares:
the learned residual and current-frame skip can cancel, with float32 rounding.
"""
from __future__ import annotations

import math
from typing import Any

import torch

if __package__ in {None, ""}:
    from pi05_libero_action_ablation import action_ablation_inputs
    from pi05_libero_world_model import FLOAT_KEYS, INPUT_KEYS, MASK_KEYS, OUTPUT_KEYS, LiberoWorldModel
else:
    from .pi05_libero_action_ablation import action_ablation_inputs
    from .pi05_libero_world_model import FLOAT_KEYS, INPUT_KEYS, MASK_KEYS, OUTPUT_KEYS, LiberoWorldModel


CONDITIONS = ("clean", "repeat_current", "fixed_train_anchor")
DRIFT_KEYS = ("state_output_delta", "visual_output_delta", "visual_residual_delta",
              "anchor_skip_delta", "history_input_delta")


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Byte-exact, including signed zero; no allclose/tolerance."""
    return (isinstance(left, torch.Tensor) and left.layout == torch.strided
            and left.dtype == right.dtype and left.device == right.device and left.shape == right.shape
            and torch.equal(left.detach().contiguous().view(torch.uint8),
                            right.detach().contiguous().view(torch.uint8)))


def _same_inputs(actual: dict, expected: dict, *, skip_visual: bool = False) -> bool:
    if type(actual) is not dict or set(actual) != INPUT_KEYS:
        return False
    for key in (*FLOAT_KEYS, *MASK_KEYS):
        if skip_visual and key == "history_visual_latent":
            continue
        if not _same_tensor(actual[key], expected[key]):
            return False
    return type(actual["task_instruction"]) is list and actual["task_instruction"] == expected["task_instruction"]


def _input_copy(inputs: dict) -> dict:
    copied = action_ablation_inputs(inputs, "observed_action")
    _require(all(bool(copied[key].all()) for key in MASK_KEYS),
             "visual-dependence diagnostics require all history masks true; masks must not be changed")
    return copied


def _float_tensor(value: Any, shape: tuple[int, ...], name: str, *, ordinary: bool = True) -> None:
    _require(isinstance(value, torch.Tensor) and value.layout == torch.strided,
             f"{name} must be a dense tensor")
    _require(value.dtype == torch.float32 and value.device.type == "cpu" and tuple(value.shape) == shape,
             f"{name} requires float32 {shape} on cpu; no coercion")
    _require(not value.requires_grad and (not ordinary or not torch.is_inference(value)),
             f"{name} must be detached" + (" and ordinary" if ordinary else ""))
    _require(bool(torch.isfinite(value).all()), f"{name} must be finite")


def transform_inputs(raw: dict, arm: str, condition: str, anchor: torch.Tensor) -> dict:
    """Return owned six-input tensors outside inference_mode, changing only vision.

    ``anchor`` is a caller-proven native train-cache vector, never a fitted mean
    or zero placeholder. Provenance cannot be inferred from tensor values and
    must be checked by the runner, even though shape/finite/nonzero checks pass.
    Both masks stay all true. The old action transform exclusively owns action
    arm semantics, including rejecting corrupt observed actions before zeroing.
    """
    _require(type(condition) is str and condition in CONDITIONS, f"condition must be one of {CONDITIONS}")
    _float_tensor(anchor, (2, 2048), "fixed train anchor")
    _require(bool(torch.count_nonzero(anchor)), "fixed train anchor cannot be a zero placeholder")
    original = _input_copy(raw)
    result = action_ablation_inputs(raw, arm)
    visual = result["history_visual_latent"]
    if condition == "repeat_current":
        result["history_visual_latent"] = visual[:, -1:].expand_as(visual).clone()
    elif condition == "fixed_train_anchor":
        result["history_visual_latent"] = anchor[None, None].expand_as(visual).clone()
    _require(_same_inputs(raw, original), "visual transform mutated caller inputs")
    return result


def _model_state(model: LiberoWorldModel) -> tuple:
    _require(isinstance(model, LiberoWorldModel), "the existing native LiberoWorldModel is required")
    cfg = model.config
    _require((cfg.visual_dim, cfg.context_len, cfg.horizon, cfg.state_dim, cfg.action_dim, cfg.views)
             == (2048, 4, 3, 8, 7, 2), "fixed native study dimensions are required")
    _require(all(not module.training for module in model.modules()), "every model module must already be in eval mode")
    values = []
    for name, parameter in model.named_parameters():
        _require(parameter.device.type == "cpu" and parameter.dtype == torch.float32
                 and parameter.layout == torch.strided and not torch.is_inference(parameter),
                 "model parameters must be ordinary float32 CPU tensors")
        _require(not parameter.requires_grad and parameter.grad is None,
                 "model must already be frozen and gradient-free")
        _require(bool(torch.isfinite(parameter).all()), "model parameters must be finite")
        values.append((name, id(parameter), parameter.data_ptr(), parameter._version))
    _require(bool(values), "model must contain native parameters")
    return tuple(values), tuple((name, id(module)) for name, module in model.named_modules())


def _hook_state(model: LiberoWorldModel) -> tuple:
    """Preserve existing hook order, identity, and kwargs/always-called flags."""
    attributes = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
                  "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_backward_hooks", "_backward_pre_hooks")
    return tuple((name, tuple((attribute, tuple((key, id(value)) for key, value in
                                              getattr(module, attribute).items())) for attribute in attributes))
                 for name, module in model.named_modules())


def capture_prediction(model: LiberoWorldModel, inputs: dict) -> dict:
    """Run native forward once under inference and capture the actual head output.

    The temporary hook is read-only and is removed on every exit. Neither a
    subtraction-derived residual nor a second forward pass is used. Call this
    outside inference_mode with already transformed, ordinary CPU inputs.
    Full checkpoint content hashes belong to the runner's before/after pins;
    each call checks parameter identity/storage/version and frozen eval state.
    """
    snapshot = _input_copy(inputs)
    before = _model_state(model)
    hooks_before = _hook_state(model)
    batch = snapshot["history_visual_latent"].shape[0]
    visual_shape = (batch, 1, 3, 2, 2048)
    captured = []

    def hook(_module, _args, output):
        _require(not captured, "visual_residual_head must run exactly once")
        _float_tensor(output, (batch, 1, 3, 4096), "actual visual residual head", ordinary=False)
        captured.append(output.detach().clone())
        return None

    handle = model.visual_residual_head.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            predictions = model(**inputs)
        _require(len(captured) == 1, "visual_residual_head must run exactly once")
        _require(type(predictions) is dict and set(predictions) == OUTPUT_KEYS, "native prediction keys changed")
        _float_tensor(predictions["pred_future_visual_latent"], visual_shape, "predicted visual", ordinary=False)
        _float_tensor(predictions["pred_state_delta"], (batch, 1, 3, 8), "predicted state", ordinary=False)
        # Cloning OUTSIDE inference produces independent ordinary return values.
        residual = captured[0].reshape(visual_shape).clone()
        anchor = snapshot["history_visual_latent"][:, -1, None, None].expand(visual_shape).clone()
        _require(_same_tensor(predictions["pred_future_visual_latent"], anchor + residual),
                 "native predicted visual must exactly equal latest latent plus actual captured residual")
        result = {"predictions": {key: value.detach().clone() for key, value in predictions.items()},
                  "residual": residual, "anchor": anchor}
    finally:
        handle.remove()
        _require(_hook_state(model) == hooks_before, "native forward changed the original hook registries")
        _require(_same_inputs(inputs, snapshot), "native forward mutated its input batch")
        _require(_model_state(model) == before, "native forward changed model identity/storage/parameter versions")
    return result


def _validate_capture(capture: dict, inputs: dict) -> None:
    _require(type(capture) is dict and set(capture) == {"predictions", "residual", "anchor"},
             "capture requires exactly predictions/residual/anchor")
    batch = inputs["history_visual_latent"].shape[0]
    visual_shape = (batch, 1, 3, 2, 2048)
    predictions = capture["predictions"]
    _require(type(predictions) is dict and set(predictions) == OUTPUT_KEYS, "capture prediction keys changed")
    _float_tensor(predictions["pred_state_delta"], (batch, 1, 3, 8), "capture state")
    for name, value in (("capture visual", predictions["pred_future_visual_latent"]),
                        ("capture residual", capture["residual"]), ("capture anchor", capture["anchor"])):
        _float_tensor(value, visual_shape, name)
    expected_anchor = inputs["history_visual_latent"][:, -1, None, None].expand(visual_shape)
    _require(_same_tensor(capture["anchor"], expected_anchor), "capture anchor differs from its own latest input")
    _require(_same_tensor(predictions["pred_future_visual_latent"], capture["anchor"] + capture["residual"]),
             "capture does not preserve exact native skip reconstruction")


class DriftTotals:
    """Scalar-coordinate drift sums, not independent sample counts or causal shares.

    Row-ordered float64 reductions keep accumulation independent of how the
    same windows are batched. The caller owns per-episode and macro reporting.
    ``exact_changed_count`` counts nonzero scalar differences, not windows.
    Failed updates are atomic and leave previous totals unchanged.
    """

    def __init__(self) -> None:
        self.windows = 0
        self.updates = 0
        self._totals = {key: {"count": 0, "sum_abs": 0.0, "sum_sq": 0.0,
                              "max_abs": 0.0, "exact_changed_count": 0} for key in DRIFT_KEYS}

    def update(self, clean_capture: dict, changed_capture: dict, clean_inputs: dict, changed_inputs: dict) -> None:
        clean = _input_copy(clean_inputs)
        changed = _input_copy(changed_inputs)
        _require(_same_inputs(changed, clean, skip_visual=True),
                 "visual drift comparison changed state/actions/masks/task or batch size")
        _validate_capture(clean_capture, clean)
        _validate_capture(changed_capture, changed)
        pairs = {
            "state_output_delta": (clean_capture["predictions"]["pred_state_delta"], changed_capture["predictions"]["pred_state_delta"]),
            "visual_output_delta": (clean_capture["predictions"]["pred_future_visual_latent"], changed_capture["predictions"]["pred_future_visual_latent"]),
            "visual_residual_delta": (clean_capture["residual"], changed_capture["residual"]),
            "anchor_skip_delta": (clean_capture["anchor"], changed_capture["anchor"]),
            "history_input_delta": (clean["history_visual_latent"], changed["history_visual_latent"]),
        }
        # Promote BEFORE subtraction; float32 output subtraction could conceal
        # coordinate differences or overflow, despite finite original tensors.
        increments = {}
        for key, (left, right) in pairs.items():
            delta = right.to(torch.float64) - left.to(torch.float64)
            increments[key] = [{"count": row.numel(), "sum_abs": float(row.abs().sum()),
                                "sum_sq": float(row.square().sum()), "max_abs": float(row.abs().max()),
                                "exact_changed_count": int(torch.count_nonzero(row))} for row in delta]
        for key, rows in increments.items():
            total = self._totals[key]
            for row in rows:
                for field in ("count", "sum_abs", "sum_sq", "exact_changed_count"):
                    total[field] += row[field]
                total["max_abs"] = max(total["max_abs"], row["max_abs"])
        self.windows += clean["history_visual_latent"].shape[0]
        self.updates += 1

    def summary(self) -> dict:
        result = {"windows": self.windows, "updates": self.updates}
        for key, values in self._totals.items():
            count = values["count"]
            result[key] = {**values, "mae": values["sum_abs"] / count if count else None,
                           "rmse": math.sqrt(values["sum_sq"] / count) if count else None}
        return result
