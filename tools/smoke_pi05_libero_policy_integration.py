#!/usr/bin/env python
"""Zero-training pinned PI0.5 + LIBERO-Spatial task-0 integration smoke.

The full pinned PI0.5 checkpoint is loaded fail-closed from the existing local
cache. Chunk length and flow steps are reduced only to bound runtime. The native
LIBERO task text is preserved. Explicit semantic and action-bound values provide
smoke-only QUANTILES statistics. This is interface evidence, not a policy score.
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

from pi05_pretrained_loader import (
    DEFAULT_PI05_REPO,
    DEFAULT_PI05_REVISION,
    PI05PretrainedLoadRejected,
    load_verified_pi05_weights,
)
from probe_lerobot_baseline_architectures import config_kwargs, import_policy
from smoke_lerobot_public_envs import (
    _json_default,
    _make_environment,
    _package_versions,
    _prepare_libero_config,
    _tree_summary,
)
from smoke_pi05_pusht_policy_integration import (
    EXPECTED_MODEL_SIZE_BYTES,
    _processor_steps,
    _prompt_state_bin_count,
    _tensor_summary,
    _transformers_patch_evidence,
)
from smoke_smolvla_libero_policy_integration import _policy_batch, _raw_camera_frames
from vla_benchmark_contract import BENCHMARK_PROFILES, profile_fingerprint


SCHEMA_VERSION = "pi05_libero_policy_integration_smoke_v1"


def _smoke_normalization_stats() -> dict[str, dict[str, np.ndarray]]:
    """Return explicit semantic/bound smoke values, never dataset quantiles."""
    state_scale = np.asarray(
        [1.5, 1.5, 1.5, np.pi, np.pi, np.pi, 0.04, 0.04], dtype=np.float32
    )
    action_scale = np.full(7, 0.2, dtype=np.float32)
    return {
        "observation.state": {"q01": -state_scale, "q99": state_scale},
        "action": {"q01": -action_scale, "q99": action_scale},
    }


def _make_rollout_sheet(snapshots: list[dict[str, Any]], output_path: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    title_font = ImageFont.truetype(str(font_path), 21)
    label_font = ImageFont.truetype(str(font_path), 17)
    panel_size = 260
    label_height = 36
    title_height = 62
    rows = len(snapshots)
    canvas = Image.new(
        "RGB",
        (panel_size * 2, title_height + (panel_size + label_height) * rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 14),
        "PI0.5 + LIBERO task 0 固定权重接入（非性能结果）",
        fill="black",
        font=title_font,
    )
    for row, snapshot in enumerate(snapshots):
        top = title_height + row * (panel_size + label_height)
        action = snapshot.get("action")
        if action is None:
            left_label = "重置帧｜主视角"
        else:
            norm = float(np.linalg.norm(np.asarray(action, dtype=np.float64)))
            left_label = f"步骤 {snapshot['step']}｜主视角｜动作范数 {norm:.2f}"
        draw.text((7, top + 6), left_label, fill="black", font=label_font)
        draw.text((panel_size + 7, top + 6), "腕部视角", fill="black", font=label_font)
        for column, frame in enumerate(snapshot["frames"]):
            image = Image.fromarray(frame[..., :3]).resize(
                (panel_size, panel_size), Image.Resampling.BILINEAR
            )
            canvas.paste(image, (column * panel_size, top + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "shape": [canvas.height, canvas.width, 3],
        "font": str(font_path),
        "visual_status": "not_viewed",
        "columns": ["agentview_image", "robot0_eye_in_hand_image"],
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

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

        libero_config_evidence = _prepare_libero_config(args.libero_config_dir, args.libero_assets_dir)
        env_config, environment = _make_environment("libero", args)
        profile = BENCHMARK_PROFILES["libero_spatial"]
        native_task = environment.get_attr("task_description")[0]
        if not isinstance(native_task, str) or not native_task.strip():
            raise ValueError(f"invalid native LIBERO task text: {native_task!r}")

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
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, policy_config)

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
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        raw_observation, reset_info = environment.reset(seed=args.seed)
        snapshots: list[dict[str, Any]] = [
            {"step": 0, "frames": _raw_camera_frames(raw_observation), "action": None}
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
                native_task=native_task,
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
                    f"PI0.5 smoke action violates native LIBERO bounds at step {step_index + 1}: "
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
                first_queue_state = {"remaining_actions_after_first_pop": len(policy._action_queue)}

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
                        "frames": _raw_camera_frames(raw_observation),
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
        required_batch_keys = {
            *profile.image_keys,
            "observation.state",
            "task",
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
        }
        checks = {
            "offline_mode_enabled": os.environ.get("HF_HUB_OFFLINE") == "1"
            and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "openpi_transformers_patch_verified": transformers_patch["status"] == "passed",
            "pinned_revision_resolved": resolved["resolved_revision"] == DEFAULT_PI05_REVISION,
            "checkpoint_size_matches": resolved["model_size_bytes"] == EXPECTED_MODEL_SIZE_BYTES,
            "pretrained_load_status_loaded": load_report["status"] == "loaded",
            "pretrained_unique_parameter_coverage_one": load_report["loaded_parameter_fraction"]
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
            "one_environment_batch_processed": required_batch_keys.issubset(first_batch),
            "env_batch_agentview_is_native_256": first_batch[profile.image_keys[0]]["shape"]
            == [1, 3, 256, 256],
            "env_batch_wristview_is_native_256": first_batch[profile.image_keys[1]]["shape"]
            == [1, 3, 256, 256],
            "first_batch_state_shape_matches": first_batch["observation.state"]["shape"]
            == [1, 8],
            "prompt_contains_native_task": first_prompt.startswith(f"Task: {native_task}, State: "),
            "prompt_contains_32_state_bins": _prompt_state_bin_count(first_prompt) == 32,
            "language_tokens_fixed_length_200": token_summary["shape"] == [1, 200],
            "language_mask_matches_tokens": mask_summary["shape"] == token_summary["shape"],
            "language_mask_has_valid_tokens": first_language_valid_token_count > 0,
            "policy_select_action_returned_finite_7d": bool(
                first_normalized_action.get("all_finite")
                and first_normalized_action["shape"] == [1, 7]
            ),
            "policy_postprocessor_returned_finite_7d": bool(
                first_native_action.get("all_finite") and first_native_action["shape"] == [1, 7]
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
            "libero_runtime_config": libero_config_evidence,
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
                "normalization_stats_source": (
                    "libero_state_semantic_units_and_native_action_bounds_smoke_only_"
                    "not_dataset_quantile_estimates"
                ),
                "normalization_stats": {
                    key: {name: value.tolist() for name, value in item.items()}
                    for key, item in stats.items()
                },
                "state_layout": [
                    "eef_position_x",
                    "eef_position_y",
                    "eef_position_z",
                    "eef_axis_angle_x",
                    "eef_axis_angle_y",
                    "eef_axis_angle_z",
                    "gripper_qpos_0",
                    "gripper_qpos_1",
                ],
                "task_source": "native_libero_task_description_preserved_without_replacement",
                "native_task_instruction": native_task,
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
        "--libero-config-dir",
        type=Path,
        default=Path("simulation_output/libero_runtime_config_v1"),
    )
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
        "scope": "pinned_pi05_libero_task0_one_batch_short_rollout_integration_only",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "python": sys.version,
        "package_versions": _package_versions(),
        "arguments": {
            "libero_task_id": args.libero_task_id,
            "libero_config_dir": str(args.libero_config_dir),
            "libero_assets_dir": str(args.libero_assets_dir),
            "episode_length": args.episode_length,
            "seed": args.seed,
            "device": args.device,
            "pretrained_name_or_path": args.pretrained_name_or_path,
            "pretrained_revision": args.pretrained_revision,
        },
        "result": result,
        "boundaries": {
            "public_dataset_loaded": False,
            "project_dataset_loaded": False,
            "checkpoint_loaded": pretrained_loaded,
            "pretrained_weights_loaded": pretrained_loaded,
            "pretrained_revision_pinned": DEFAULT_PI05_REVISION,
            "native_language_task_required_and_preserved": True,
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
        f"status={report['status']} policy=pi05 env=libero_spatial task=0 "
        f"transitions={transitions} loaded_fraction={loaded_fraction} offline=true "
        f"optimizer_steps=0 report={args.out}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
