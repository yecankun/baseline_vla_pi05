#!/usr/bin/env python
"""Run paired clean/noisy fixed-action episodes on Push-T and LIBERO.

This is an environment and observation-pipeline smoke, not policy evaluation.
No dataset, policy, checkpoint, training loop, or optimizer is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from smoke_lerobot_public_envs import (
    _as_numpy,
    _compare_trees,
    _expected_action,
    _flatten_leaves,
    _json_default,
    _make_environment,
    _package_versions,
    _prepare_libero_config,
    _processed_observation,
    _safe_number,
)
from vla_benchmark_contract import BENCHMARK_PROFILES, VisualCorruption
from vla_visual_noise_wrapper import (
    RAW_ENV_IMAGE_PATHS,
    EvaluationVisualObservationWrapper,
    get_mapping_path,
)


SCHEMA_VERSION = "lerobot_public_noise_pair_smoke_v1"


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(_as_numpy(value)).tobytes()).hexdigest()


def _equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
        return left == right
    if isinstance(left, (list, tuple)) and all(isinstance(item, str) for item in left):
        return list(left) == list(right)
    try:
        return bool(np.array_equal(_as_numpy(left), _as_numpy(right)))
    except Exception:
        return left == right


def _non_image_exact(clean: Mapping[str, Any], noisy: Mapping[str, Any], image_paths: tuple[str, ...]) -> bool:
    clean_leaves = _flatten_leaves(clean)
    noisy_leaves = _flatten_leaves(noisy)
    non_image_paths = set(clean_leaves) - set(image_paths)
    return set(clean_leaves) == set(noisy_leaves) and all(
        _equal_value(clean_leaves[path], noisy_leaves[path]) for path in non_image_paths
    )


def _processed_checks(kind: str, clean: Mapping[str, Any], noisy: Mapping[str, Any]) -> dict[str, Any]:
    profile_name = "pusht" if kind == "pusht" else "libero_spatial"
    image_paths = BENCHMARK_PROFILES[profile_name].image_keys
    image_checks: dict[str, Any] = {}
    for path in image_paths:
        clean_image = _as_numpy(get_mapping_path(clean, path))
        noisy_image = _as_numpy(get_mapping_path(noisy, path))
        image_checks[path] = {
            "shape_preserved": clean_image.shape == noisy_image.shape,
            "dtype_preserved": clean_image.dtype == noisy_image.dtype,
            "changed": bool(not np.array_equal(clean_image, noisy_image)),
            "clean_sha256": _array_sha256(clean_image),
            "corrupted_sha256": _array_sha256(noisy_image),
            "mean_absolute_change": float(
                np.abs(clean_image.astype(np.float32) - noisy_image.astype(np.float32)).mean()
            ),
        }
    return {
        "state_exact": _equal_value(clean["observation.state"], noisy["observation.state"]),
        "task_exact": _equal_value(clean["task"], noisy["task"]),
        "images": image_checks,
    }


def _to_hwc_uint8(value: Any) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected an image leaf, got {image.shape}")
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
        image = np.rint(np.clip(image * scale, 0.0, 255.0))
    return image.astype(np.uint8)


def _save_pair_sheet(
    *,
    kind: str,
    clean: Mapping[str, Any],
    noisy: Mapping[str, Any],
    image_paths: tuple[str, ...],
    corruption: VisualCorruption,
    output_path: Path,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font = ImageFont.truetype(str(font_path), 22)
    small_font = ImageFont.truetype(str(font_path), 17)
    pairs = [
        (_to_hwc_uint8(get_mapping_path(clean, path)), _to_hwc_uint8(get_mapping_path(noisy, path)))
        for path in image_paths
    ]
    max_width = max(left.shape[1] for left, _ in pairs)
    corruption_labels = {
        "gaussian_sensor_noise": "高斯传感器噪声",
        "low_contrast": "低对比度",
        "blur": "模糊",
        "occlusion": "遮挡",
    }
    title = (
        f"{kind.upper()}：干净/噪声成对观测"
        f"（{corruption_labels[corruption.name]}，等级 {corruption.severity}）"
    )
    row_labels = [f"{path}｜左：干净观测　右：噪声观测" for path in image_paths]
    title_width = font.getbbox(title)[2] - font.getbbox(title)[0]
    label_width = max(small_font.getbbox(label)[2] - small_font.getbbox(label)[0] for label in row_labels)
    canvas_width = max(max_width * 2, title_width + 24, label_width + 16)
    title_height = 54
    label_height = 34
    row_heights = [left.shape[0] + label_height for left, _ in pairs]
    canvas = Image.new("RGB", (canvas_width, title_height + sum(row_heights)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black", font=font)
    top = title_height
    image_left = (canvas_width - max_width * 2) // 2
    for label, (left, right), row_height in zip(row_labels, pairs, row_heights):
        draw.text((8, top + 5), label, fill="black", font=small_font)
        image_top = top + label_height
        canvas.paste(Image.fromarray(left[..., :3]), (image_left, image_top))
        canvas.paste(Image.fromarray(right[..., :3]), (image_left + max_width, image_top))
        top += row_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "shape": [canvas.height, canvas.width, 3],
        "font": str(font_path),
        "visual_status": "not_viewed",
    }


def _run_pair(kind: str, args: argparse.Namespace, corruption: VisualCorruption) -> dict[str, Any]:
    started = time.monotonic()
    clean_env = None
    noisy_env = None
    try:
        _, clean_env = _make_environment(kind, args)
        _, noisy_base_env = _make_environment(kind, args)
        image_paths = RAW_ENV_IMAGE_PATHS[kind]
        noisy_env = EvaluationVisualObservationWrapper(
            noisy_base_env,
            image_paths=image_paths,
            corruption=corruption,
            seed=args.noise_seed,
            episode_key=f"{kind}/task-{args.libero_task_id}/seed-{args.seed}",
            verify_determinism=True,
        )
        clean_observation, clean_info = clean_env.reset(seed=args.seed)
        noisy_observation, noisy_info = noisy_env.reset(seed=args.seed)
        source_comparison = _compare_trees(clean_observation, noisy_env.last_clean_observation)
        reset_non_image_exact = _non_image_exact(clean_observation, noisy_observation, image_paths)
        clean_processed = _processed_observation(kind, clean_env, clean_observation)
        noisy_processed = _processed_observation(kind, noisy_env, noisy_observation)
        reset_processed = _processed_checks(kind, clean_processed, noisy_processed)
        initial_clean = clean_observation
        initial_noisy = noisy_observation

        action, expected_low, expected_high = _expected_action(kind)
        action_batch = np.expand_dims(action, axis=0)
        action_space = clean_env.single_action_space
        action_contract_preserved = bool(
            action_space.contains(action)
            and noisy_env.single_action_space.contains(action)
            and np.allclose(action_space.low, expected_low)
            and np.allclose(action_space.high, expected_high)
        )

        transition_records: list[dict[str, Any]] = []
        all_source_exact = source_comparison["all_exact"]
        all_non_image_exact = reset_non_image_exact
        all_rewards_exact = True
        all_done_exact = True
        all_processed_state_exact = reset_processed["state_exact"]
        all_processed_task_exact = reset_processed["task_exact"]
        all_processed_images_changed = all(
            item["changed"] for item in reset_processed["images"].values()
        )
        for step_index in range(args.episode_length):
            clean_next, clean_reward, clean_terminated, clean_truncated, _ = clean_env.step(action_batch)
            noisy_next, noisy_reward, noisy_terminated, noisy_truncated, _ = noisy_env.step(action_batch)
            source_exact = _compare_trees(clean_next, noisy_env.last_clean_observation)["all_exact"]
            non_image_exact = _non_image_exact(clean_next, noisy_next, image_paths)
            reward_exact = bool(np.array_equal(_as_numpy(clean_reward), _as_numpy(noisy_reward)))
            done_exact = bool(
                np.array_equal(_as_numpy(clean_terminated), _as_numpy(noisy_terminated))
                and np.array_equal(_as_numpy(clean_truncated), _as_numpy(noisy_truncated))
            )
            clean_processed = _processed_observation(kind, clean_env, clean_next)
            noisy_processed = _processed_observation(kind, noisy_env, noisy_next)
            processed = _processed_checks(kind, clean_processed, noisy_processed)
            images_changed = all(item["changed"] for item in processed["images"].values())
            transition_records.append(
                {
                    "step": step_index + 1,
                    "source_observation_exact_before_corruption": source_exact,
                    "non_image_observation_exact": non_image_exact,
                    "reward_exact": reward_exact,
                    "termination_exact": done_exact,
                    "processed_state_exact": processed["state_exact"],
                    "processed_task_exact": processed["task_exact"],
                    "processed_images_changed": images_changed,
                }
            )
            all_source_exact = all_source_exact and source_exact
            all_non_image_exact = all_non_image_exact and non_image_exact
            all_rewards_exact = all_rewards_exact and reward_exact
            all_done_exact = all_done_exact and done_exact
            all_processed_state_exact = all_processed_state_exact and processed["state_exact"]
            all_processed_task_exact = all_processed_task_exact and processed["task_exact"]
            all_processed_images_changed = all_processed_images_changed and images_changed
            if bool(np.any(_as_numpy(clean_terminated)) or np.any(_as_numpy(clean_truncated))):
                break

        wrapper_records_valid = all(
            view["changed"] and view["deterministic_duplicate"]
            for frame in noisy_env.frame_records
            for view in frame["views"]
        )
        artifact_path = args.artifact_dir / (
            f"{kind}_{corruption.name}_s{corruption.severity}_reset_pair.png"
        )
        artifact = _save_pair_sheet(
            kind=kind,
            clean=initial_clean,
            noisy=initial_noisy,
            image_paths=image_paths,
            corruption=corruption,
            output_path=artifact_path,
        )
        checks = {
            "independent_source_observations_exact_before_corruption": all_source_exact,
            "all_non_image_observation_leaves_exact": all_non_image_exact,
            "all_rewards_exact": all_rewards_exact,
            "all_termination_flags_exact": all_done_exact,
            "processed_state_exact": all_processed_state_exact,
            "processed_task_exact": all_processed_task_exact,
            "processed_images_changed_every_frame": all_processed_images_changed,
            "raw_corruption_changed_and_deterministic": wrapper_records_valid,
            "native_action_contract_preserved": action_contract_preserved,
            "episode_has_at_least_one_transition": bool(transition_records),
        }
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "task_id": 0 if kind == "pusht" else args.libero_task_id,
            "task_description": getattr(clean_env.envs[0], "task_description", ""),
            "reset_seed": args.seed,
            "noise_seed": args.noise_seed,
            "corruption": {"name": corruption.name, "severity": corruption.severity},
            "raw_image_paths": list(image_paths),
            "fixed_native_action": action.astype(float).tolist(),
            "transitions_executed": len(transition_records),
            "reset_source_comparison": source_comparison,
            "reset_clean_info_keys": sorted(str(key) for key in clean_info),
            "reset_noisy_info_keys": sorted(str(key) for key in noisy_info),
            "reset_processed_checks": reset_processed,
            "checks": checks,
            "per_transition_checks": transition_records,
            "corruption_records": noisy_env.frame_records,
            "visual_artifact": artifact,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines()[-40:],
        }
    finally:
        if clean_env is not None:
            clean_env.close()
        if noisy_env is not None:
            noisy_env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--envs", choices=("all", "pusht", "libero"), default="all")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--noise-seed", type=int, default=20260911)
    parser.add_argument("--corruption", choices=("gaussian_sensor_noise", "low_contrast", "blur", "occlusion"), default="gaussian_sensor_noise")
    parser.add_argument("--severity", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--episode-length", type=int, default=20)
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument("--libero-config-dir", type=Path, default=Path("simulation_output/libero_runtime_config_v1"))
    parser.add_argument("--libero-assets-dir", type=Path, default=Path("simulation_output/libero_assets_v1"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    selected = ("pusht", "libero") if args.envs == "all" else (args.envs,)
    libero_config = None
    setup_failure = None
    if "libero" in selected:
        try:
            libero_config = _prepare_libero_config(args.libero_config_dir, args.libero_assets_dir)
        except Exception as exc:
            setup_failure = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    corruption = VisualCorruption(name=args.corruption, severity=args.severity)
    started = time.monotonic()
    environments = {
        kind: setup_failure if kind == "libero" and setup_failure else _run_pair(kind, args, corruption)
        for kind in selected
    }
    status = "passed" if all(item["status"] == "passed" for item in environments.values()) else "failed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "paired_clean_noisy_fixed_action_environment_episode_smoke",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "python": sys.version,
        "package_versions": _package_versions(),
        "arguments": {
            "envs": args.envs,
            "seed": args.seed,
            "noise_seed": args.noise_seed,
            "corruption": args.corruption,
            "severity": args.severity,
            "episode_length": args.episode_length,
            "libero_task_id": args.libero_task_id,
            "libero_assets_dir": str(args.libero_assets_dir),
        },
        "libero_runtime_config": libero_config,
        "environments": environments,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "boundaries": {
            "wrapper_position": "after_raw_environment_observation_before_policy_preprocessing",
            "evaluation_only": True,
            "fixed_actions_independent_of_observation": True,
            "dataset_loaded": False,
            "policy_or_checkpoint_loaded": False,
            "training_started": False,
            "optimizer_steps": 0,
            "public_actions_keep_native_semantics": True,
            "benchmark_score_claim_allowed": False,
            "guidewire_capability_claim_allowed": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = " ".join(f"{kind}={item['status']}" for kind, item in environments.items())
    print(f"status={status} {summary} optimizer_steps=0 report={args.out}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
