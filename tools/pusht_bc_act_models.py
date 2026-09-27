"""Compact BC and full official ACT interfaces for native Push-T.

The compact BC is a new reference, NOT an exact reproduction of the senior
student's architecture. These models use only RGB and observable agent state.
Synthetic smoke statistics must never be reused for training or scored runs.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


IMAGE = "observation.image"
STATE = "observation.state"
OBSERVATION_KEYS = (IMAGE, STATE)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCHEMA = "pusht_bc_act_architecture_interfaces_v1"
ACT_ARCHITECTURE = {
    "n_obs_steps": 1,
    "chunk_size": 16,
    "n_action_steps": 8,
    "vision_backbone": "resnet18",
    "pretrained_backbone_weights": None,
    "replace_final_stride_with_dilation": False,
    "pre_norm": False,
    "dim_model": 512,
    "n_heads": 8,
    "dim_feedforward": 3200,
    "feedforward_activation": "relu",
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "use_vae": True,
    "latent_dim": 32,
    "n_vae_encoder_layers": 4,
    "dropout": 0.1,
    "kl_weight": 10.0,
    "temporal_ensemble_coeff": None,
}


def _finite_float(value: Any, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating torch tensor")
    if tuple(value.shape) != shape or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor with shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def validate_observation(observation: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject extras, including reward, coverage, success, task, and simulator truth."""
    if not isinstance(observation, Mapping) or set(observation) != set(OBSERVATION_KEYS):
        raise ValueError(f"observation must contain exactly {OBSERVATION_KEYS}; extra fields are forbidden")
    image = observation[IMAGE]
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] <= 0:
        raise ValueError("image must be nonempty [B,3,96,96]")
    image = _finite_float(image, (image.shape[0], 3, 96, 96), IMAGE)
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("raw image must be floating in [0,1]; no layout/range inference")
    state = _finite_float(observation[STATE], (image.shape[0], 2), STATE)
    if state.device != image.device or state.dtype != image.dtype:
        raise ValueError("image/state must have the same device and floating dtype")
    return image, state


def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    """Shared fixed visual normalization; state/action normalization is external."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] <= 0:
        raise ValueError("image must be nonempty [B,3,96,96]")
    _finite_float(image, (image.shape[0], 3, 96, 96), IMAGE)
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("ImageNet normalization expects raw [0,1] images")
    mean = image.new_tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    return (image - mean) / std


class CompactPushTBC(nn.Module):
    """96px RGB + externally normalized state -> normalized native 2D action.

    Three 5x5 stride-2 padded convolutions produce [B,64,12,12]. The linear
    action output has no sigmoid or clipping. Fit state/action mean/std using
    the future accepted training split only, outside this model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(130, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, observation: Mapping[str, Any]) -> torch.Tensor:
        image, state = validate_observation(observation)
        first_parameter = next(self.parameters())
        if image.device != first_parameter.device or image.dtype != first_parameter.dtype:
            raise ValueError("observation dtype/device must match BC parameters")
        visual = self.visual(imagenet_normalize(image))
        prediction = self.head(torch.cat((visual, state), dim=-1))
        return _finite_float(prediction, (image.shape[0], 2), "normalized action")

    def loss(self, observation: Mapping[str, Any], normalized_action: torch.Tensor) -> torch.Tensor:
        predicted = self(observation)
        target = _finite_float(normalized_action, tuple(predicted.shape), "normalized action target")
        if target.device != predicted.device or target.dtype != predicted.dtype:
            raise ValueError("action target dtype/device must match the prediction")
        return F.mse_loss(predicted, target)

    @torch.inference_mode()
    def select_action(self, observation: Mapping[str, Any]) -> torch.Tensor:
        if self.training:
            raise ValueError("BC select_action requires eval mode")
        return self(observation)

    def reset(self) -> None:
        """Stateless BC has no observation or action queue to clear."""

    @staticmethod
    def describe() -> dict[str, Any]:
        return {
            "name": "compact_visual_state_bc_reference",
            "exact_senior_architecture_reproduction": False,
            "convolution_channels": [32, 64, 64], "kernel": 5, "stride": 2, "padding": 2,
            "flatten_dim": 9216, "visual_projection_dim": 128, "mlp_hidden_dim": 256,
            "state_dim": 2, "action_dim": 2, "loss": "MSE in externally normalized action space",
            "action_activation": "linear; no sigmoid/clipping", "queue": None,
            "image_normalization": "fixed ImageNet mean/std",
            "state_action_normalization": "external train-only mean/std; never test/validation fit",
        }


def make_act_config(device: str = "cpu") -> Any:
    """Lazy official full-size LeRobot ACT config; no model or download yet."""
    if importlib.metadata.version("lerobot") != "0.4.4":
        raise RuntimeError("ACT interface is pinned to LeRobot 0.4.4")
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    config = ACTConfig(
        **ACT_ARCHITECTURE,
        device=device,
        push_to_hub=False,
        use_amp=False,
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        normalization_mapping={key: NormalizationMode.MEAN_STD for key in ("VISUAL", "STATE", "ACTION")},
    )
    for name, expected in ACT_ARCHITECTURE.items():
        if getattr(config, name) != expected:
            raise ValueError(f"official ACT configuration drifted: {name}")
    return config


def make_act_policy_and_processors(dataset_stats: Mapping[str, Any], device: str = "cpu") -> tuple[Any, Any, Any]:
    """Construct the official architecture with explicitly supplied statistics.

    The caller owns train-only statistics/provenance. This factory never loads
    a policy or backbone checkpoint and deliberately has no stats fallback.
    """
    config = make_act_config(device)
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    policy = ACTPolicy(config).to(device)
    pre, post = make_pre_post_processors(config, dataset_stats=dict(dataset_stats))
    return policy, pre, post


def act_batch(
    observation: Mapping[str, Any], *, action: torch.Tensor | None = None,
    action_is_pad: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Strict raw observation routing; targets are explicit separate arguments."""
    image, _ = validate_observation(observation)
    result = dict(observation)
    if (action is None) != (action_is_pad is None):
        raise ValueError("ACT action and action_is_pad must be supplied together")
    if action is not None:
        action = _finite_float(action, (image.shape[0], 16, 2), "ACT native action target")
        if action.device != image.device or action.dtype != image.dtype:
            raise ValueError("ACT target dtype/device must match its observations")
        if not isinstance(action_is_pad, torch.Tensor) or action_is_pad.dtype != torch.bool or tuple(action_is_pad.shape) != (image.shape[0], 16):
            raise ValueError("ACT action_is_pad must be boolean [B,16]")
        if action_is_pad.device != image.device or bool(action_is_pad.all(dim=1).any()):
            raise ValueError("ACT mask must share the input device and retain a valid target per sample")
        result.update({"action": action, "action_is_pad": action_is_pad})
    return result


def synthetic_only_stats() -> dict[str, dict[str, torch.Tensor]]:
    """Fabricated smoke values; not dataset statistics or evaluation bounds."""
    return {
        IMAGE: {"mean": torch.tensor(IMAGENET_MEAN).reshape(3, 1, 1), "std": torch.tensor(IMAGENET_STD).reshape(3, 1, 1)},
        STATE: {"mean": torch.full((2,), 256.0), "std": torch.full((2,), 128.0)},
        "action": {"mean": torch.full((2,), 256.0), "std": torch.full((2,), 128.0)},
    }


@contextmanager
def _seeded_torch(seed: int, device: str):
    cuda = torch.device(device).type == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable; no silent CPU fallback")
    devices = list(range(torch.cuda.device_count())) if cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        yield


def _parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> dict[str, Any]:
    _finite_float(tensor, shape, "smoke output")
    return {"shape": list(tensor.shape), "finite": True, "min": float(tensor.min()), "max": float(tensor.max())}


def smoke(model_name: str, *, device: str = "cpu", seed: int = 20260913) -> dict[str, Any]:
    if model_name not in {"bc", "act"}:
        raise ValueError("model must be bc or act")
    started = time.monotonic()
    with _seeded_torch(seed, device):
        images = torch.rand(2, 3, 96, 96, device=device)
        native_state = torch.tensor([[128.0, 256.0], [384.0, 64.0]], device=device)
        native_action = torch.rand(2, 16, 2, device=device) * 512
        mask = torch.zeros(2, 16, dtype=torch.bool, device=device)
        if model_name == "bc":
            policy = CompactPushTBC().to(device)
            observation = {IMAGE: images, STATE: (native_state - 256) / 128}
            before = _parameter_hash(policy)
            with torch.no_grad():
                loss = policy.loss(observation, (native_action[:, 0] - 256) / 128)
            policy.eval()
            policy.reset()
            normalized = policy.select_action(observation)
            policy.reset()
            replay = policy.select_action(observation)
            native = normalized * 128 + 256
            configuration = policy.describe()
            visual_normalization_max_error = 0.0
            loss_components = {"mse": float(loss)}
        else:
            policy, pre, post = make_act_policy_and_processors(synthetic_only_stats(), device)
            observation = {IMAGE: images, STATE: native_state}
            before = _parameter_hash(policy)
            training_batch = pre(act_batch(observation, action=native_action, action_is_pad=mask))
            reference_normalized_image = imagenet_normalize(images)
            visual_normalization_max_error = float((training_batch[IMAGE] - reference_normalized_image).abs().max())
            if visual_normalization_max_error > 1e-6:
                raise ValueError("ACT preprocessing differs from shared ImageNet normalization")
            policy.train()  # Exercise the official VAE loss path; no backward or optimizer.
            with torch.no_grad():
                loss, official_losses = policy(training_batch)
            loss_components = {name: float(value) for name, value in official_losses.items()}
            policy.eval()
            policy.reset()
            pre.reset()
            post.reset()
            with torch.inference_mode():
                normalized = policy.select_action(pre(act_batch(observation)))
                native = post(normalized)
                policy.reset()
                pre.reset()
                post.reset()
                replay = policy.select_action(pre(act_batch(observation)))
            configuration = {**ACT_ARCHITECTURE, "implementation": "official LeRobot 0.4.4 ACT", "loss": "official L1 + KL (unchanged)", "input_features": {IMAGE: [3, 96, 96], STATE: [2]}, "output_features": {"action": [2]}}
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise ValueError("synthetic loss must be finite scalar")
        if not torch.equal(normalized, replay):
            raise ValueError("reset did not reproduce deterministic eval output")
        after = _parameter_hash(policy)
        if before != after or any(parameter.grad is not None for parameter in policy.parameters()):
            raise ValueError("synthetic smoke changed parameters or populated gradients")
        return {
            "status": "passed", "model": model_name, "config": configuration,
            "device": device, "seed": seed, "batch_size": 2,
            "parameter_count": sum(parameter.numel() for parameter in policy.parameters()),
            "loss": float(loss), "loss_components": loss_components,
            "normalized_action": _finite_shape(normalized, (2, 2)),
            "synthetic_unnormalized_action": _finite_shape(native, (2, 2)),
            "reset_replay_exact": True, "parameters_unchanged": True,
            "parameter_sha256": after, "visual_normalization_max_error": visual_normalization_max_error,
            "policy_observation_allowlist": list(OBSERVATION_KEYS),
            "stats_source": "explicitly_synthetic_only_not_training_or_evaluation_stats",
            "pretrained_weights_loaded": False, "backbone_weights_downloaded": False,
            "train_mode_loss_forward": model_name == "act",
            "backward_calls": 0, "optimizer_steps": 0, "environment_steps": 0,
            "dataset_loaded": False, "benchmark_score_claim_allowed": False,
            "runtime_seconds": time.monotonic() - started,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--models", choices=("bc", "act", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    report: dict[str, Any] = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": [], "training_started": False, "backward_calls": 0,
        "optimizer_steps": 0, "environment_steps": 0, "dataset_loaded": False,
        "noisy_input_used": False, "benchmark_score_claim_allowed": False,
        "torch_runtime": torch.__version__,
        "package_versions": {},
    }
    for name in ("lerobot", "torch", "torchvision", "numpy", "safetensors"):
        try:
            report["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][name] = None
    try:
        for model_name in (("bc", "act") if args.models == "both" else (args.models,)):
            report["models"].append(smoke(model_name, device=args.device, seed=args.seed))
        report["status"] = "passed"
    except Exception as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"status={report['status']} optimizer_steps=0 environment_steps=0 report={args.out}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
