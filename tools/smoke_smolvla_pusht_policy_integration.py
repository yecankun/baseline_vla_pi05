#!/usr/bin/env python
"""Zero-training reduced SmolVLA + Push-T integration smoke.

The policy is randomly initialized and intentionally reduced. Only the small
cached SmolVLM config/tokenizer/processor files are used; Hugging Face network
access is disabled before any SmolVLA import. Push-T bounds provide smoke-only
MEAN_STD statistics. This is interface evidence, not a policy score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from probe_lerobot_baseline_architectures import config_kwargs, import_policy
from smoke_lerobot_public_envs import (
    _array_summary,
    _json_default,
    _make_environment,
    _package_versions,
    _tree_summary,
)
from vla_benchmark_contract import BENCHMARK_PROFILES, profile_fingerprint


SCHEMA_VERSION = "smolvla_pusht_policy_integration_smoke_v1"
MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PUSHT_TASK_INSTRUCTION = "Push the T-shaped block to the target."
REQUIRED_CACHED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _smoke_normalization_stats() -> dict[str, dict[str, np.ndarray]]:
    """Return explicit bounds-derived stats, not dataset estimates."""
    midpoint = np.asarray([256.0, 256.0], dtype=np.float32)
    state_scale = np.asarray([256.0, 256.0], dtype=np.float32)
    # One eighth of the native 512-pixel span keeps an untrained flow sample
    # useful for a bounds-valid interface smoke without clipping its output.
    action_scale = np.asarray([64.0, 64.0], dtype=np.float32)
    return {
        "observation.state": {"mean": midpoint, "std": state_scale},
        "action": {"mean": midpoint, "std": action_scale},
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
        "SmolVLA + Push-T 随机权重接入（非性能结果）",
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


def _cached_processor_evidence() -> dict[str, Any]:
    hub_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    model_cache = hub_root / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    refs_main = model_cache / "refs/main"
    revision = refs_main.read_text(encoding="utf-8").strip() if refs_main.is_file() else None
    snapshot = model_cache / "snapshots" / revision if revision else None
    files: dict[str, Any] = {}
    for name in REQUIRED_CACHED_FILES:
        path = snapshot / name if snapshot is not None else None
        files[name] = {
            "present": bool(path is not None and path.is_file()),
            "size_bytes": int(path.stat().st_size) if path is not None and path.is_file() else None,
        }
    weight_patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    weight_files = (
        sorted(
            str(path.relative_to(model_cache))
            for pattern in weight_patterns
            for path in model_cache.rglob(pattern)
            if path.is_file()
        )
        if model_cache.is_dir()
        else []
    )
    return {
        "model_id": MODEL_ID,
        "cache_path": str(model_cache),
        "revision": revision,
        "required_files": files,
        "all_required_processor_files_present": all(item["present"] for item in files.values()),
        "cached_weight_files": weight_files,
        "cached_policy_or_backbone_weight_file_count": len(weight_files),
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
            raise RuntimeError("SmolVLA smoke must run with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")

        cache_evidence_before = _cached_processor_evidence()
        if not cache_evidence_before["all_required_processor_files_present"]:
            raise FileNotFoundError("cached SmolVLM config/tokenizer/processor files are incomplete")
        if cache_evidence_before["cached_policy_or_backbone_weight_file_count"] != 0:
            raise RuntimeError("unexpected cached model weights found; smoke must not depend on them")

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        env_config, environment = _make_environment("pusht", args)
        profile = BENCHMARK_PROFILES["pusht"]
        config_class, policy_class = import_policy("smolvla")
        kwargs = config_kwargs("smolvla", profile, device=args.device)
        kwargs.update(
            {
                "chunk_size": 2,
                "n_action_steps": 2,
                "num_steps": 2,
                "num_vlm_layers": 1,
                "num_expert_layers": 1,
                "expert_width_multiplier": 0.25,
                "load_vlm_weights": False,
                "compile_model": False,
            }
        )
        policy_config = config_class(**kwargs)
        policy = policy_class(policy_config).to(args.device)
        policy.eval()
        policy.reset()

        stats = _smoke_normalization_stats()
        policy_preprocessor, policy_postprocessor = make_pre_post_processors(
            policy_config,
            dataset_stats=stats,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_config,
            policy_config,
        )

        raw_observation, reset_info = environment.reset(seed=args.seed)
        snapshots: list[dict[str, Any]] = [
            {"step": 0, "frame": _render_first(environment), "action": None}
        ]
        first_batch = None
        first_normalized_action = None
        first_native_action = None
        first_queue_state = None
        first_language_valid_token_count = None
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
                    f"SmolVLA smoke action violates native Push-T bounds at step {step_index + 1}: "
                    f"{action_numpy.tolist()}"
                )
            if first_batch is None:
                first_batch = _tree_summary(batch)
                first_normalized_action = _tensor_summary(normalized_action)
                first_native_action = _tensor_summary(native_action)
                first_queue_state = {
                    "remaining_actions_after_first_pop": len(policy._queues[ACTION]),
                }
                first_language_valid_token_count = int(
                    batch[OBS_LANGUAGE_ATTENTION_MASK].sum().detach().cpu().item()
                )

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

        if (
            first_batch is None
            or first_normalized_action is None
            or first_native_action is None
            or first_queue_state is None
            or first_language_valid_token_count is None
        ):
            raise AssertionError("no policy batch was executed")
        action_array = np.stack(actions, axis=0)
        required_batch_keys = {
            "observation.image",
            "observation.state",
            "task",
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        }
        token_summary = first_batch[OBS_LANGUAGE_TOKENS]
        mask_summary = first_batch[OBS_LANGUAGE_ATTENTION_MASK]
        checks = {
            "offline_mode_enabled": os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "cached_processor_files_complete": cache_evidence_before[
                "all_required_processor_files_present"
            ],
            "no_cached_weights_used_or_required": cache_evidence_before[
                "cached_policy_or_backbone_weight_file_count"
            ]
            == 0,
            "one_environment_batch_processed": required_batch_keys.issubset(first_batch),
            "first_batch_image_shape_matches": first_batch["observation.image"]["shape"]
            == [1, 3, 96, 96],
            "first_batch_state_shape_matches": first_batch["observation.state"]["shape"] == [1, 2],
            "push_t_constant_task_preserved": first_batch["task"].get("value")
            == [f"{PUSHT_TASK_INSTRUCTION}\n"],
            "language_tokens_nonempty": token_summary["shape"][0] == 1
            and token_summary["shape"][1] >= 1,
            "language_mask_matches_tokens": mask_summary["shape"] == token_summary["shape"],
            "language_mask_has_valid_token": first_language_valid_token_count >= 1,
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
        cache_evidence_after = _cached_processor_evidence()
        checks["cache_weight_file_count_unchanged"] = (
            cache_evidence_after["cached_policy_or_backbone_weight_file_count"]
            == cache_evidence_before["cached_policy_or_backbone_weight_file_count"]
        )
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "policy": {
                "name": "smolvla",
                "class": f"{policy_class.__module__}.{policy_class.__name__}",
                "random_initialization_seed": args.seed,
                "pretrained_policy_weights_loaded": False,
                "pretrained_backbone_weights_loaded": False,
                "parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
                "config": {
                    "chunk_size": policy_config.chunk_size,
                    "n_action_steps": policy_config.n_action_steps,
                    "num_steps": policy_config.num_steps,
                    "num_vlm_layers": policy_config.num_vlm_layers,
                    "num_expert_layers": policy_config.num_expert_layers,
                    "expert_width_multiplier": policy_config.expert_width_multiplier,
                    "max_state_dim": policy_config.max_state_dim,
                    "max_action_dim": policy_config.max_action_dim,
                    "tokenizer_max_length": policy_config.tokenizer_max_length,
                    "resize_imgs_with_padding": list(policy_config.resize_imgs_with_padding),
                    "vlm_model_name": policy_config.vlm_model_name,
                    "load_vlm_weights": policy_config.load_vlm_weights,
                },
            },
            "profile": {**profile.to_dict(), "fingerprint": profile_fingerprint(profile)},
            "processor_contract": {
                "order": [
                    "preprocess_observation",
                    "add_envs_task",
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
                "normalization_stats_source": "push_t_native_bounds_smoke_only_not_dataset_statistics",
                "normalization_stats": {
                    key: {name: value.tolist() for name, value in item.items()}
                    for key, item in stats.items()
                },
                "task_source": (
                    "benchmark_contract_constant_instruction_replacing_official_empty_push_t_task"
                ),
                "constant_task_instruction": PUSHT_TASK_INSTRUCTION,
                "task_after_smolvla_newline_processor": first_batch["task"],
            },
            "offline_cache_evidence_before": cache_evidence_before,
            "offline_cache_evidence_after": cache_evidence_after,
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
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "scope": "reduced_smolvla_pusht_random_weight_one_batch_one_episode_integration_only",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "python": sys.version,
        "package_versions": _package_versions(),
        "result": result,
        "boundaries": {
            "public_dataset_loaded": False,
            "project_dataset_loaded": False,
            "checkpoint_loaded": False,
            "pretrained_weights_loaded": False,
            "cached_small_processor_resources_only": True,
            "policy_randomly_initialized": True,
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
    print(
        f"status={report['status']} policy=smolvla env=pusht transitions={transitions} "
        f"random_weights=true offline=true optimizer_steps=0 report={args.out}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
