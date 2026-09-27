from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

from vla_benchmark_contract import BENCHMARK_PROFILES, profile_fingerprint


POLICIES = {
    "act": (
        "lerobot.policies.act.configuration_act",
        "ACTConfig",
        "lerobot.policies.act.modeling_act",
        "ACTPolicy",
    ),
    "diffusion": (
        "lerobot.policies.diffusion.configuration_diffusion",
        "DiffusionConfig",
        "lerobot.policies.diffusion.modeling_diffusion",
        "DiffusionPolicy",
    ),
    "smolvla": (
        "lerobot.policies.smolvla.configuration_smolvla",
        "SmolVLAConfig",
        "lerobot.policies.smolvla.modeling_smolvla",
        "SmolVLAPolicy",
    ),
    "pi05": (
        "lerobot.policies.pi05.configuration_pi05",
        "PI05Config",
        "lerobot.policies.pi05.modeling_pi05",
        "PI05Policy",
    ),
}


def make_features(profile: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.configs.types import FeatureType, PolicyFeature

    input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=profile.image_shape)
        for key in profile.image_keys
    }
    input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(profile.state_dim,))
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(profile.action_dim,)),
    }
    return input_features, output_features


def config_kwargs(policy_name: str, profile: Any, *, device: str = "cpu") -> dict[str, Any]:
    input_features, output_features = make_features(profile)
    common = {
        "input_features": input_features,
        "output_features": output_features,
        "device": device,
    }
    if policy_name == "act":
        return {
            **common,
            "chunk_size": 4,
            "n_action_steps": 2,
            "pretrained_backbone_weights": None,
            "dim_model": 64,
            "n_heads": 4,
            "dim_feedforward": 128,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "use_vae": False,
        }
    if policy_name == "diffusion":
        return {
            **common,
            "n_obs_steps": 2,
            "horizon": 8,
            "n_action_steps": 4,
            "drop_n_last_frames": 3,
            "pretrained_backbone_weights": None,
            "down_dims": (64, 128),
            "diffusion_step_embed_dim": 32,
            "num_train_timesteps": 4,
            "num_inference_steps": 2,
            "compile_model": False,
        }
    if policy_name == "smolvla":
        return {
            **common,
            "chunk_size": 4,
            "n_action_steps": 2,
            "max_state_dim": max(32, profile.state_dim),
            "max_action_dim": max(32, profile.action_dim),
            # Keep each benchmark's raw image contract, then use SmolVLA's
            # native visual-tower preprocessing size inside the policy.
            "resize_imgs_with_padding": (512, 512),
            "load_vlm_weights": False,
            "compile_model": False,
        }
    if policy_name == "pi05":
        return {
            **common,
            "chunk_size": 4,
            "n_action_steps": 2,
            "max_state_dim": max(32, profile.state_dim),
            "max_action_dim": max(32, profile.action_dim),
            "image_resolution": profile.image_shape[1:],
            "freeze_vision_encoder": True,
            "train_expert_only": True,
            "gradient_checkpointing": False,
            "compile_model": False,
        }
    raise KeyError(policy_name)


def import_policy(policy_name: str) -> tuple[type, type]:
    config_module, config_class_name, model_module, policy_class_name = POLICIES[policy_name]
    config_class = getattr(importlib.import_module(config_module), config_class_name)
    policy_class = getattr(importlib.import_module(model_module), policy_class_name)
    return config_class, policy_class


def lightweight_forward_backward(
    policy_name: str,
    *,
    config_class: type,
    policy_class: type,
    profile: Any,
    device: str,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(20260911)
    kwargs = config_kwargs(policy_name, profile, device=device)
    if policy_name == "smolvla":
        kwargs.update(
            {
                "chunk_size": 2,
                "n_action_steps": 1,
                "num_vlm_layers": 1,
                "num_expert_layers": 1,
                "expert_width_multiplier": 0.25,
            }
        )
    config = config_class(**kwargs)
    policy = policy_class(config).to(device)
    policy.train()
    batch_size = 1
    batch: dict[str, Any] = {}
    for key in profile.image_keys:
        if policy_name == "diffusion":
            batch[key] = torch.rand(
                batch_size,
                config.n_obs_steps,
                *profile.image_shape,
                dtype=torch.float32,
                device=device,
            )
        else:
            batch[key] = torch.rand(
                batch_size,
                *profile.image_shape,
                dtype=torch.float32,
                device=device,
            )
    if policy_name == "diffusion":
        batch["observation.state"] = torch.zeros(
            batch_size,
            config.n_obs_steps,
            profile.state_dim,
            dtype=torch.float32,
            device=device,
        )
        batch["action"] = torch.zeros(
            batch_size,
            config.horizon,
            profile.action_dim,
            dtype=torch.float32,
            device=device,
        )
        batch["action_is_pad"] = torch.zeros(
            batch_size,
            config.horizon,
            dtype=torch.bool,
            device=device,
        )
    else:
        batch["observation.state"] = torch.zeros(
            batch_size,
            profile.state_dim,
            dtype=torch.float32,
            device=device,
        )
        batch["action"] = torch.zeros(
            batch_size,
            config.chunk_size,
            profile.action_dim,
            dtype=torch.float32,
            device=device,
        )
        batch["action_is_pad"] = torch.zeros(
            batch_size,
            config.chunk_size,
            dtype=torch.bool,
            device=device,
        )
        if policy_name == "smolvla":
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

            batch[OBS_LANGUAGE_TOKENS] = torch.tensor([[0, 1]], dtype=torch.long, device=device)
            batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(1, 2, dtype=torch.bool, device=device)

    loss, _ = policy(batch)
    if not torch.isfinite(loss):
        raise ValueError(f"non-finite synthetic loss: {loss}")
    loss.backward()
    grad_tensors = sum(parameter.grad is not None for parameter in policy.parameters())
    if grad_tensors <= 0:
        raise ValueError("synthetic backward produced no gradients")
    result = {
        "status": "passed",
        "profile": profile.name,
        "device": device,
        "finite_loss": float(loss.detach().cpu().item()),
        "parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
        "gradient_tensor_count": int(grad_tensors),
        "synthetic_backward_executed": True,
        "optimizer_steps": 0,
    }
    del policy, batch, loss
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def probe_policy(policy_name: str, *, lightweight_forward: bool, device: str) -> dict[str, Any]:
    try:
        config_class, policy_class = import_policy(policy_name)
    except Exception as exc:
        return {"status": "failed", "import_ok": False, "error": repr(exc)}

    profile_results: dict[str, Any] = {}
    for profile_name, profile in BENCHMARK_PROFILES.items():
        try:
            config = config_class(**config_kwargs(policy_name, profile))
            config.validate_features()
            profile_results[profile_name] = {
                "status": "passed",
                "input_image_keys": list(config.image_features),
                "state_dim": profile.state_dim,
                "action_dim": int(config.action_feature.shape[0]),
                "task_conditioning": profile.task_conditioning,
                "profile_fingerprint": profile_fingerprint(profile),
            }
        except Exception as exc:
            profile_results[profile_name] = {"status": "failed", "error": repr(exc)}

    passed = all(value["status"] == "passed" for value in profile_results.values())
    execution_results: dict[str, Any] = {}
    if lightweight_forward and policy_name in {"act", "diffusion", "smolvla"}:
        profile_names = ("project_d0", "pusht") if policy_name in {"act", "diffusion"} else ("project_d0",)
        for profile_name in profile_names:
            try:
                execution_results[profile_name] = lightweight_forward_backward(
                    policy_name,
                    config_class=config_class,
                    policy_class=policy_class,
                    profile=BENCHMARK_PROFILES[profile_name],
                    device=device,
                )
            except Exception as exc:
                execution_results[profile_name] = {"status": "failed", "error": repr(exc)}

    return {
        "status": "passed" if passed else "failed",
        "import_ok": True,
        "config_class": f"{config_class.__module__}.{config_class.__name__}",
        "policy_class": f"{policy_class.__module__}.{policy_class.__name__}",
        "policy_constructor": str(inspect.signature(policy_class)),
        "config_profiles": profile_results,
        "model_constructed": bool(execution_results),
        "forward_executed": bool(execution_results),
        "execution_profiles": execution_results,
        "optimizer_steps": 0,
    }


def dependency_status(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    return {"available": spec is not None, "origin": None if spec is None else spec.origin}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-training import/config probe for the Project 2026 baseline and benchmark matrix."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--lightweight-forward",
        action="store_true",
        help=(
            "Construct ACT, Diffusion, and a reduced random SmolVLA configuration, then run synthetic "
            "loss/backward without optimizer steps or pretrained policy weights."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy_reports = {
        name: probe_policy(name, lightweight_forward=args.lightweight_forward, device=args.device)
        for name in POLICIES
    }
    execution_failures = [
        f"{policy_name}/{profile_name}"
        for policy_name, policy_report in policy_reports.items()
        for profile_name, execution_report in policy_report.get("execution_profiles", {}).items()
        if execution_report["status"] != "passed"
    ]
    report = {
        "status": "passed" if all(item["status"] == "passed" for item in policy_reports.values()) else "failed",
        "scope": "architecture_import_and_feature_config_only",
        "training_started": False,
        "optimizer_steps": 0,
        "lightweight_forward_requested": bool(args.lightweight_forward),
        "lightweight_execution_failures": execution_failures,
        "lerobot_version": importlib.metadata.version("lerobot"),
        "policies": policy_reports,
        "benchmark_runtime_dependencies": {
            "pusht": dependency_status("gym_pusht"),
            "libero": dependency_status("libero"),
        },
        "boundaries": {
            "openvla_oft_included": False,
            "project_output_interface": "elite_tcp_delta_6d + piper_intent_id",
            "external_benchmarks_keep_native_action_spaces": True,
            "model_selection_on_corrupted_evaluation": False,
            "privileged_policy_inputs_allowed": False,
        },
        "interpretation": {
            "passed_means": "policy classes import and configs accept every declared feature/action profile",
            "does_not_mean": "weights, forward pass, environment runtime, training quality, or deployment are validated",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    ready = sum(item["status"] == "passed" for item in policy_reports.values())
    forward_passed = sum(
        execution_report["status"] == "passed"
        for policy_report in policy_reports.values()
        for execution_report in policy_report.get("execution_profiles", {}).values()
    )
    print(
        f"status={report['status']} policies={ready}/{len(policy_reports)} "
        f"lightweight_forwards={forward_passed} failures={len(execution_failures)} "
        f"pusht_env={report['benchmark_runtime_dependencies']['pusht']['available']} "
        f"libero_env={report['benchmark_runtime_dependencies']['libero']['available']} "
        "optimizer_steps=0"
    )
    if report["status"] != "passed" or execution_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
