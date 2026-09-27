#!/usr/bin/env python
"""Zero-training pinned PI0.5 + Push-T integration smoke.

The full pinned PI0.5 checkpoint is loaded fail-closed from the existing local
cache. Chunk length and flow steps are reduced only to bound runtime. Push-T
bounds provide smoke-only QUANTILES statistics, and a fixed task instruction is
used because the native environment has no language task. This is interface
evidence, not a policy score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pi05_pretrained_loader import (
    DEFAULT_PI05_REPO,
    DEFAULT_PI05_REVISION,
    PI05PretrainedLoadRejected,
    load_verified_pi05_weights,
)
from probe_lerobot_baseline_architectures import config_kwargs, import_policy
from smoke_lerobot_public_envs import (
    _array_summary,
    _json_default,
    _make_environment,
    _package_versions,
    _tree_summary,
)
from vla_benchmark_contract import BENCHMARK_PROFILES, profile_fingerprint


SCHEMA_VERSION = "pi05_pusht_policy_integration_smoke_v1"
PUSHT_TASK_INSTRUCTION = "Push the T-shaped block to the target."
EXPECTED_MODEL_SIZE_BYTES = 14_467_165_872
OPENPI_PATCH_SOURCE_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
EXPECTED_TRANSFORMERS_VERSION = "4.53.2"
EXPECTED_TOKENIZERS_VERSION = "0.21.4"
EXPECTED_TRANSFORMERS_PATCH_SHA256 = {
    "models/gemma/configuration_gemma.py": "26fd0d0f52730ab32b6f337eb73ae9a26cb71ae0915bc49eced1d0e2ed8d0ec6",
    "models/gemma/modeling_gemma.py": "17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde",
    "models/paligemma/modeling_paligemma.py": "c945d330b829264f927f3ee5339f977238631772de07a28e855e265bb85fd53c",
    "models/siglip/check.py": "466e8eb7887ac5ccc98118a3169298d6ab65a284d834bae74751e3854dbc4245",
    "models/siglip/modeling_siglip.py": "ef2e99500f263fdd78db9d4d9a255e29d844eb92f53c25ad640f8056700ef84b",
}


def _smoke_normalization_stats() -> dict[str, dict[str, np.ndarray]]:
    """Return explicit native-bound smoke values, never dataset quantiles."""
    state_low = np.asarray([0.0, 0.0], dtype=np.float32)
    state_high = np.asarray([512.0, 512.0], dtype=np.float32)
    # PI0.5's output is in pretrained normalized coordinates. A central
    # quarter-span maps normalized +/-1 to 192..320 and leaves headroom for a
    # short two-step flow sample without clipping the decoded action.
    action_low = np.asarray([192.0, 192.0], dtype=np.float32)
    action_high = np.asarray([320.0, 320.0], dtype=np.float32)
    return {
        "observation.state": {"q01": state_low, "q99": state_high},
        "action": {"q01": action_low, "q99": action_high},
    }


def _processor_steps(pipeline: Any) -> list[str]:
    return [f"{type(step).__module__}.{type(step).__name__}" for step in pipeline.steps]


def _tensor_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return _array_summary(value)


def _render_first(environment: Any) -> np.ndarray:
    rendered = environment.call("render")
    frame = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3 or not np.isfinite(frame).all():
        raise ValueError(f"invalid RGB render: {frame.shape}")
    return frame.astype(np.uint8)


def _make_rollout_sheet(snapshots: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    title_font = ImageFont.truetype(str(font_path), 22)
    label_font = ImageFont.truetype(str(font_path), 18)
    panel_size = 280
    label_height = 38
    title_height = 62
    columns = 2
    rows = (len(snapshots) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (panel_size * columns, title_height + (panel_size + label_height) * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 14),
        "PI0.5 + Push-T 固定权重接入（非性能结果）",
        fill="black",
        font=title_font,
    )
    for index, snapshot in enumerate(snapshots):
        row, column = divmod(index, columns)
        left = column * panel_size
        top = title_height + row * (panel_size + label_height)
        action = snapshot.get("action")
        label = (
            "重置帧"
            if action is None
            else f"步骤 {snapshot['step']}｜action=({action[0]:.1f}, {action[1]:.1f})"
        )
        draw.text((left + 8, top + 7), label, fill="black", font=label_font)
        image = Image.fromarray(snapshot["frame"][..., :3]).resize(
            (panel_size, panel_size), Image.Resampling.BILINEAR
        )
        canvas.paste(image, (left, top + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "shape": [canvas.height, canvas.width, 3],
        "font": str(font_path),
        "visual_status": "not_viewed",
    }


def _policy_batch(
    *,
    raw_observation: dict[str, Any],
    environment: Any,
    env_preprocessor: Any,
    policy_preprocessor: Any,
) -> dict[str, Any]:
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    observation = preprocess_observation(raw_observation)
    observation = add_envs_task(environment, observation)
    if observation.get("task") != [""]:
        raise ValueError(f"unexpected native Push-T task contract: {observation.get('task')!r}")
    observation["task"] = [PUSHT_TASK_INSTRUCTION]
    observation = env_preprocessor(observation)
    return policy_preprocessor(observation)


def _prompt_state_bin_count(prompt: str) -> int:
    match = re.search(r", State: ([0-9 ]+);\nAction: $", prompt)
    if match is None:
        return 0
    return len(match.group(1).split())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _transformers_patch_evidence() -> dict[str, Any]:
    import inspect
    from importlib import metadata

    import transformers
    from transformers.models.gemma.configuration_gemma import GemmaConfig
    from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

    root = Path(transformers.__file__).resolve().parent
    files = {}
    for relative_path, expected_hash in EXPECTED_TRANSFORMERS_PATCH_SHA256.items():
        path = root / relative_path
        actual_hash = _sha256_file(path) if path.is_file() else None
        files[relative_path] = {
            "path": str(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "matches": actual_hash == expected_hash,
        }

    config = GemmaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        use_adarms=True,
        adarms_cond_dim=8,
    )
    norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=config.adarms_cond_dim)
    forward_parameters = list(inspect.signature(norm.forward).parameters)
    norm_state_keys = sorted(norm.state_dict())
    transformers_version = metadata.version("transformers")
    tokenizers_version = metadata.version("tokenizers")
    checks = {
        "transformers_version_pinned": transformers_version == EXPECTED_TRANSFORMERS_VERSION,
        "tokenizers_version_pinned": tokenizers_version == EXPECTED_TOKENIZERS_VERSION,
        "all_patch_hashes_match": all(item["matches"] for item in files.values()),
        "adarms_forward_accepts_cond": forward_parameters == ["x", "cond"],
        "adarms_dense_parameters_present": norm_state_keys == ["dense.bias", "dense.weight"],
        "adarms_cond_dim_matches": int(norm.cond_dim) == 8,
    }
    return {
        "source": "Physical-Intelligence/openpi transformers_replace",
        "source_commit": OPENPI_PATCH_SOURCE_COMMIT,
        "transformers_version": transformers_version,
        "tokenizers_version": tokenizers_version,
        "files": files,
        "adarms_forward_parameters": forward_parameters,
        "adarms_state_keys": norm_state_keys,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    started = time.monotonic()
    environment = None
    try:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("PI0.5 smoke must run with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
        if args.pretrained_revision != DEFAULT_PI05_REVISION:
            raise ValueError(
                f"revision must remain pinned to {DEFAULT_PI05_REVISION}, got {args.pretrained_revision}"
            )

        transformers_patch = _transformers_patch_evidence()
        if transformers_patch["status"] != "passed":
            raise RuntimeError(
                f"OpenPI Transformers patch audit failed: {transformers_patch['checks']}"
            )

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        env_config, environment = _make_environment("pusht", args)
        profile = BENCHMARK_PROFILES["pusht"]
        config_class, policy_class = import_policy("pi05")
        kwargs = config_kwargs("pi05", profile, device=args.device)
        kwargs.update(
            {
                "chunk_size": 2,
                "n_action_steps": 2,
                "num_inference_steps": 2,
                "image_resolution": (224, 224),
                "dtype": "bfloat16",
                "freeze_vision_encoder": True,
                "train_expert_only": True,
                "gradient_checkpointing": False,
                "compile_model": False,
            }
        )
        policy_config = config_class(**kwargs)
        stats = _smoke_normalization_stats()
        policy_preprocessor, policy_postprocessor = make_pre_post_processors(
            policy_config,
            dataset_stats=stats,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_config,
            policy_config,
        )

        policy = policy_class(policy_config).to(args.device)
        load_report = load_verified_pi05_weights(
            policy,
            args.pretrained_name_or_path,
            revision=args.pretrained_revision,
            cache_dir=args.pretrained_cache_dir,
            local_files_only=True,
            min_loaded_parameter_fraction=1.0,
            max_missing_keys=0,
            max_unexpected_keys=0,
            compute_file_sha256=False,
        )
        policy.eval()
        policy.reset()
        # Decouple action sampling from random tensors consumed during model
        # construction; this makes repeated integration smokes deterministic.
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        raw_observation, reset_info = environment.reset(seed=args.seed)
        snapshots: list[dict[str, Any]] = [
            {"step": 0, "frame": _render_first(environment), "action": None}
        ]
        first_batch = None
        first_prompt = None
        first_language_valid_token_count = None
        first_normalized_action = None
        first_native_action = None
        first_queue_state = None
        per_step: list[dict[str, Any]] = []
        actions: list[np.ndarray] = []
        all_actions_finite = True
        all_actions_in_bounds = True
        all_rewards_finite = True
        all_steps_returned = True

        for step_index in range(args.episode_length):
            batch = _policy_batch(
                raw_observation=raw_observation,
                environment=environment,
                env_preprocessor=env_preprocessor,
                policy_preprocessor=policy_preprocessor,
            )
            with torch.inference_mode():
                normalized_action = policy.select_action(batch)
            native_action = policy_postprocessor(normalized_action)
            action_transition = env_postprocessor({ACTION: native_action})
            native_action = action_transition[ACTION]
            action_numpy = native_action.detach().cpu().numpy()
            if action_numpy.shape != (1, profile.action_dim):
                raise ValueError(f"unexpected action shape: {action_numpy.shape}")
            finite = bool(np.isfinite(action_numpy).all())
            inside = bool(environment.single_action_space.contains(action_numpy[0]))
            all_actions_finite = all_actions_finite and finite
            all_actions_in_bounds = all_actions_in_bounds and inside
            if not finite or not inside:
                raise ValueError(
                    f"PI0.5 smoke action violates native Push-T bounds at step {step_index + 1}: "
                    f"{action_numpy.tolist()}"
                )
            if first_batch is None:
                first_batch = _tree_summary(batch)
                first_prompt = str(batch["task"][0])
                first_language_valid_token_count = int(
                    batch[OBS_LANGUAGE_ATTENTION_MASK].sum().detach().cpu().item()
                )
                first_normalized_action = _tensor_summary(normalized_action)
                first_native_action = _tensor_summary(native_action)
                first_queue_state = {
                    "remaining_actions_after_first_pop": len(policy._action_queue),
                }

            raw_observation, reward, terminated, truncated, _ = environment.step(action_numpy)
            reward_finite = bool(np.isfinite(np.asarray(reward)).all())
            all_rewards_finite = all_rewards_finite and reward_finite
            all_steps_returned = all_steps_returned and raw_observation is not None
            action_vector = action_numpy[0].astype(float)
            actions.append(action_vector)
            per_step.append(
                {
                    "step": step_index + 1,
                    "action": action_vector.tolist(),
                    "action_finite": finite,
                    "action_inside_native_bounds": inside,
                    "reward_finite": reward_finite,
                    "terminated": bool(np.any(np.asarray(terminated))),
                    "truncated": bool(np.any(np.asarray(truncated))),
                }
            )
            if step_index + 1 in {1, 10, args.episode_length}:
                snapshots.append(
                    {
                        "step": step_index + 1,
                        "frame": _render_first(environment),
                        "action": action_vector.tolist(),
                    }
                )
            if bool(np.any(np.asarray(terminated)) or np.any(np.asarray(truncated))):
                break

        if any(
            value is None
            for value in (
                first_batch,
                first_prompt,
                first_language_valid_token_count,
                first_normalized_action,
                first_native_action,
                first_queue_state,
            )
        ):
            raise AssertionError("no policy batch was executed")

        action_array = np.stack(actions, axis=0)
        token_summary = first_batch[OBS_LANGUAGE_TOKENS]
        mask_summary = first_batch[OBS_LANGUAGE_ATTENTION_MASK]
        resolved = load_report["resolved"]
        checks = {
            "offline_mode_enabled": os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "openpi_transformers_patch_verified": transformers_patch["status"] == "passed",
            "pinned_revision_resolved": resolved["resolved_revision"] == DEFAULT_PI05_REVISION,
            "checkpoint_size_matches": resolved["model_size_bytes"] == EXPECTED_MODEL_SIZE_BYTES,
            "pretrained_load_status_loaded": load_report["status"] == "loaded",
            "pretrained_unique_parameter_coverage_one": load_report[
                "loaded_parameter_fraction"
            ]
            == 1.0,
            "pretrained_no_effective_missing_keys": load_report["missing_key_count"] == 0,
            "pretrained_no_unexpected_keys": load_report["unexpected_key_count"] == 0,
            "pretrained_no_shape_mismatches": load_report["mismatched_shape_count"] == 0,
            "pretrained_tied_alias_verified": load_report["tied_weight_alias_count"] == 1
            and all(
                item["source_matches_checkpoint"]
                and item["alias_matches_checkpoint"]
                and item["shared_storage_after_load"]
                for item in load_report["tied_weight_alias_verification"]
            ),
            "one_environment_batch_processed": {
                "observation.image",
                "observation.state",
                "task",
                OBS_LANGUAGE_TOKENS,
                OBS_LANGUAGE_ATTENTION_MASK,
            }.issubset(first_batch),
            "first_batch_image_shape_matches": first_batch["observation.image"]["shape"]
            == [1, 3, 96, 96],
            "first_batch_state_shape_matches": first_batch["observation.state"]["shape"] == [1, 2],
            "prompt_contains_constant_task": first_prompt.startswith(
                f"Task: {PUSHT_TASK_INSTRUCTION}, State: "
            ),
            "prompt_contains_32_state_bins": _prompt_state_bin_count(first_prompt) == 32,
            "language_tokens_fixed_length_200": token_summary["shape"] == [1, 200],
            "language_mask_matches_tokens": mask_summary["shape"] == token_summary["shape"],
            "language_mask_has_valid_tokens": first_language_valid_token_count > 0,
            "policy_select_action_returned_finite_2d": bool(
                first_normalized_action.get("all_finite")
                and first_normalized_action["shape"] == [1, 2]
            ),
            "policy_postprocessor_returned_finite_2d": bool(
                first_native_action.get("all_finite") and first_native_action["shape"] == [1, 2]
            ),
            "action_queue_retains_one_after_first_pop": first_queue_state[
                "remaining_actions_after_first_pop"
            ]
            == 1,
            "all_actions_finite": all_actions_finite,
            "all_actions_inside_native_bounds": all_actions_in_bounds,
            "all_rewards_finite": all_rewards_finite,
            "all_environment_steps_returned": all_steps_returned,
            "episode_has_transitions": bool(per_step),
            "no_optimizer_constructed_or_stepped": True,
        }
        artifact = _make_rollout_sheet(snapshots, args.artifact)
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "policy": {
                "name": "pi05",
                "class": f"{policy_class.__module__}.{policy_class.__name__}",
                "parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
                "pretrained_policy_weights_loaded": True,
                "pretrained_backbone_weights_loaded": True,
                "config": {
                    "chunk_size": policy_config.chunk_size,
                    "n_action_steps": policy_config.n_action_steps,
                    "num_inference_steps": policy_config.num_inference_steps,
                    "paligemma_variant": policy_config.paligemma_variant,
                    "action_expert_variant": policy_config.action_expert_variant,
                    "dtype": policy_config.dtype,
                    "max_state_dim": policy_config.max_state_dim,
                    "max_action_dim": policy_config.max_action_dim,
                    "tokenizer_max_length": policy_config.tokenizer_max_length,
                    "image_resolution": list(policy_config.image_resolution),
                },
            },
            "pretrained_load": load_report,
            "transformers_patch": transformers_patch,
            "profile": {**profile.to_dict(), "fingerprint": profile_fingerprint(profile)},
            "processor_contract": {
                "order": [
                    "preprocess_observation",
                    "add_envs_task",
                    "inject_profile_constant_task",
                    "env_preprocessor",
                    "policy_preprocessor",
                    "policy.select_action",
                    "policy_postprocessor",
                    "env_postprocessor",
                    "environment.step",
                ],
                "env_preprocessor_steps": _processor_steps(env_preprocessor),
                "policy_preprocessor_steps": _processor_steps(policy_preprocessor),
                "policy_postprocessor_steps": _processor_steps(policy_postprocessor),
                "env_postprocessor_steps": _processor_steps(env_postprocessor),
                "normalization_stats_source": (
                    "push_t_native_bounds_smoke_only_not_dataset_quantile_estimates"
                ),
                "normalization_stats": {
                    key: {name: value.tolist() for name, value in item.items()}
                    for key, item in stats.items()
                },
                "task_source": (
                    "benchmark_contract_constant_instruction_replacing_official_empty_push_t_task"
                ),
                "constant_task_instruction": PUSHT_TASK_INSTRUCTION,
                "first_prompt": first_prompt,
                "first_prompt_state_bin_count": _prompt_state_bin_count(first_prompt),
            },
            "queue_contract_after_first_action": first_queue_state,
            "reset_seed": args.seed,
            "reset_info_keys": sorted(str(key) for key in reset_info),
            "first_policy_batch": first_batch,
            "first_language_valid_token_count": first_language_valid_token_count,
            "first_normalized_action": first_normalized_action,
            "first_native_action": first_native_action,
            "transitions_executed": len(per_step),
            "action_summary": {
                "shape": list(action_array.shape),
                "min": float(action_array.min()),
                "max": float(action_array.max()),
                "mean": action_array.mean(axis=0).astype(float).tolist(),
                "sha256": hashlib.sha256(np.ascontiguousarray(action_array).tobytes()).hexdigest(),
            },
            "per_step": per_step,
            "checks": checks,
            "visual_artifact": artifact,
        }
    except PI05PretrainedLoadRejected as exc:
        return {
            "status": "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pretrained_load": exc.report,
            "traceback": traceback.format_exc().splitlines()[-50:],
        }
    except Exception as exc:
        return {
            "status": "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines()[-50:],
        }
    finally:
        if environment is not None:
            environment.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument("--pretrained-name-or-path", default=DEFAULT_PI05_REPO)
    parser.add_argument("--pretrained-revision", default=DEFAULT_PI05_REVISION)
    parser.add_argument(
        "--pretrained-cache-dir",
        type=Path,
        default=Path("/home/zsw/.cache/huggingface/hub"),
    )
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument(
        "--libero-assets-dir", type=Path, default=Path("simulation_output/libero_assets_v1")
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    result = _run(args)
    pretrained_loaded = result.get("pretrained_load", {}).get("status") == "loaded"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "scope": "pinned_pi05_pusht_one_batch_one_episode_integration_only",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "python": sys.version,
        "package_versions": _package_versions(),
        "result": result,
        "boundaries": {
            "public_dataset_loaded": False,
            "project_dataset_loaded": False,
            "checkpoint_loaded": pretrained_loaded,
            "pretrained_weights_loaded": pretrained_loaded,
            "pretrained_revision_pinned": DEFAULT_PI05_REVISION,
            "network_access_allowed": False,
            "training_started": False,
            "backward_executed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "benchmark_score_claim_allowed": False,
            "guidewire_capability_claim_allowed": False,
            "success_or_reward_interpretation_allowed": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    transitions = result.get("transitions_executed", 0)
    loaded_fraction = result.get("pretrained_load", {}).get("loaded_parameter_fraction")
    print(
        f"status={report['status']} policy=pi05 env=pusht transitions={transitions} "
        f"loaded_fraction={loaded_fraction} offline=true optimizer_steps=0 report={args.out}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
