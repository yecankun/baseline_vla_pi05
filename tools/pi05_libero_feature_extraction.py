"""Strict native LIBERO image boundary for frozen PI0.5 feature extraction.

This module does not load checkpoints or datasets, select actions, or train a
model. The caller supplies an already loaded, evaluation-mode PI0.5 policy.
State, task text, actions, and offline metadata belong to the dataset adapter,
not to this image-only boundary. Real pinned-checkpoint execution is a separate
integration check; a fake-policy unit test does not establish that gate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from typing import Any

import torch
from torch import nn

if __package__ in {None, ""}:
    from pi05_action_effect_world_model import extract_pi05_multiview_visual_latent
else:
    from .pi05_action_effect_world_model import extract_pi05_multiview_visual_latent


NATIVE_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
EMPTY_IMAGE_KEY = "observation.images.empty_camera_0"
FEATURE_INTERFACE_VERSION = "pi05_libero_native_visual_features_v1"


def _validate_image(
    image: Any,
    *,
    name: str,
    batch_size: int | None,
    low: float,
    high: float,
) -> int:
    if not isinstance(image, torch.Tensor) or not torch.is_floating_point(image):
        raise ValueError(f"{name} must be a floating point torch tensor")
    if image.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor")
    if image.ndim != 4 or image.shape[1] != 3 or any(size <= 0 for size in image.shape):
        raise ValueError(f"{name} must be nonempty BCHW with exactly three channels")
    if batch_size is not None and image.shape[0] != batch_size:
        raise ValueError(f"{name} has a different batch size")
    if not bool(torch.isfinite(image).all()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((image < low).any()) or bool((image > high).any()):
        raise ValueError(f"{name} must be in [{low:g}, {high:g}]")
    return int(image.shape[0])


@torch.inference_mode()
def extract_native_visual_features(
    policy: nn.Module, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Return finite pooled image latents ``[B, 2, D]`` in native view order.

    Input images must be floating point BCHW in [0, 1]. Layout/range inference,
    integer scaling, view substitution, and unknown cameras are deliberately
    unsupported. One declared, entirely masked empty camera is permitted but
    never embedded. The policy's own ``_preprocess_images`` applies
    checkpoint resize/pad and [-1, 1] conversion before its vision embedder.

    The supplied policy and every child must already be in eval mode. This
    function disables gradients without changing modes, weights, or parameter
    ``requires_grad`` flags. It passes cloned inputs to preprocessing so the
    caller's image tensors cannot be changed by in-place preprocessing.
    """
    if not isinstance(policy, nn.Module):
        raise ValueError("policy must be a torch.nn.Module")
    if any(module.training for module in policy.modules()):
        raise ValueError("policy and every child module must already be in eval mode")
    if not isinstance(batch, Mapping):
        raise ValueError("batch must be an image-only mapping")
    if set(batch) != set(NATIVE_IMAGE_KEYS):
        raise ValueError(
            "image-only batch requires exactly observation.images.image and "
            "observation.images.image2; state/task/offline/privileged fields are forbidden"
        )
    config = getattr(policy, "config", None)
    image_features = getattr(config, "image_features", None)
    if not isinstance(image_features, Mapping):
        raise ValueError("policy.config.image_features must be an ordered mapping")
    configured_keys = tuple(image_features)
    if configured_keys not in (NATIVE_IMAGE_KEYS, (*NATIVE_IMAGE_KEYS, EMPTY_IMAGE_KEY)):
        raise ValueError("policy.config.image_features must contain native keys in order and at most declared empty_camera_0")
    has_empty_camera = len(configured_keys) == 3
    if getattr(config, "empty_cameras", None) != int(has_empty_camera):
        raise ValueError("policy.config.empty_cameras must agree with the declared optional empty view")
    preprocess = getattr(policy, "_preprocess_images", None)
    if not callable(preprocess):
        raise ValueError("policy does not expose PI05 _preprocess_images")

    batch_size: int | None = None
    device: torch.device | None = None
    prepared_batch: dict[str, torch.Tensor] = {}
    for key in NATIVE_IMAGE_KEYS:
        value = batch[key]
        batch_size = _validate_image(value, name=key, batch_size=batch_size, low=0.0, high=1.0)
        if device is not None and value.device != device:
            raise ValueError("both native images must be on the same device")
        device = value.device
        prepared_batch[key] = value.detach().clone()

    result = preprocess(prepared_batch)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("PI05 preprocessing must return (images, image_masks)")
    images, masks = result
    if not isinstance(images, (list, tuple)) or len(images) != len(configured_keys):
        raise ValueError("PI05 preprocessing image count must match the declared view layout")
    if not isinstance(masks, (list, tuple)) or len(masks) != len(configured_keys):
        raise ValueError("PI05 preprocessing image-mask count must match the declared view layout")
    for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
        _validate_image(image, name=f"preprocessed view {index}", batch_size=batch_size, low=-1.0, high=1.0)
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != (batch_size,)
        ):
            raise ValueError(f"preprocessed view {index} requires a boolean [B] mask")
        if index < 2 and not bool(mask.all()):
            raise ValueError(f"preprocessed real view {index} requires an all-valid mask")
        if index == 2 and (bool(mask.any()) or not bool((image == -1).all())):
            raise ValueError("declared empty camera requires an all-false mask and exact -1 image")

    latents = extract_pi05_multiview_visual_latent(policy, images[:2])
    if (
        not isinstance(latents, torch.Tensor)
        or not torch.is_floating_point(latents)
        or latents.ndim != 3
        or tuple(latents.shape[:2]) != (batch_size, 2)
        or latents.shape[2] <= 0
        or not bool(torch.isfinite(latents).all())
    ):
        raise ValueError("PI05 pooled visual features must be finite floating [B, 2, D] with D > 0")
    return latents.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Print the interface only; no policy/data load")
    args = parser.parse_args()
    if not args.describe:
        parser.print_help()
        return
    print(json.dumps({
        "version": FEATURE_INTERFACE_VERSION,
        "input_keys_in_view_order": list(NATIVE_IMAGE_KEYS),
        "input_image_layout": "BCHW, float, 3 channels, finite [0,1]",
        "preprocessing": "caller-loaded policy._preprocess_images before frozen image embedding",
        "output": "finite [B,2,D], patch-mean pooled, no language/action tokens",
        "policy_requirement": "all modules eval; two valid native views plus at most declared, masked empty_camera_0",
        "checkpoint_loaded": False,
        "training_started": False,
        "real_checkpoint_integration": "pending",
    }, indent=2))


if __name__ == "__main__":
    main()
