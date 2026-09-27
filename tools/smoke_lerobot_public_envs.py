#!/usr/bin/env python
"""Bounded, zero-training runtime smoke for LeRobot Push-T and LIBERO.

This tool intentionally stops after reset, preprocessing, render, and one
native environment step. It does not load datasets, policies, or checkpoints.
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "lerobot_public_env_runtime_smoke_v1"
PACKAGE_NAMES = (
    "lerobot",
    "gym-pusht",
    "pymunk",
    "hf-libero",
    "egl-probe",
    "hf-egl-probe",
    "mujoco",
    "robosuite",
    "transformers",
    "tokenizers",
    "opencv-python",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _prepare_libero_config(config_dir: Path, assets_dir: Path) -> dict[str, Any]:
    """Create a non-interactive config pointing only at packaged runtime assets."""
    import yaml

    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Cannot locate the installed hf-libero package")
    package_root = Path(spec.origin).resolve().parent
    benchmark_root = package_root / "libero"
    resolved_assets = assets_dir.resolve()
    required_asset_subdirs = (
        "articulated_objects",
        "scenes",
        "stable_hope_objects",
        "stable_scanned_objects",
        "textures",
        "turbosquid_objects",
    )
    missing_asset_subdirs = [
        name for name in required_asset_subdirs if not (resolved_assets / name).is_dir()
    ]
    if missing_asset_subdirs:
        raise FileNotFoundError(
            f"LIBERO assets are incomplete at {resolved_assets}; missing subdirectories: "
            f"{missing_asset_subdirs}. Transfer the complete lerobot/libero-assets directory; "
            "the smoke will not auto-download it."
        )

    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": package_root / "datasets",
        "assets": resolved_assets,
    }
    required = ("benchmark_root", "bddl_files", "init_states")
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing packaged LIBERO runtime paths: {missing}")

    resolved_config_dir = config_dir.resolve()
    resolved_config_dir.mkdir(parents=True, exist_ok=True)
    config_file = resolved_config_dir / "config.yaml"
    serialized_paths = {name: str(path.resolve()) for name, path in paths.items()}
    config_file.write_text(yaml.safe_dump(serialized_paths, sort_keys=True), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(resolved_config_dir)
    return {
        "config_file": str(config_file),
        "paths": {
            name: {"path": value, "exists": Path(value).exists()}
            for name, value in serialized_paths.items()
        },
        "demonstration_dataset_required_for_smoke": False,
        "assets_auto_download_allowed": False,
    }


def _as_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_number(value: Any) -> int | float | None:
    import numpy as np

    number = value.item() if hasattr(value, "item") else value
    if isinstance(number, (bool, np.bool_)):
        return int(bool(number))
    if isinstance(number, (int, np.integer)):
        return int(number)
    number = float(number)
    return number if np.isfinite(number) else None


def _json_default(value: Any) -> Any:
    """Serialize diagnostic metadata without weakening numeric JSON checks."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return repr(value)


def _array_summary(value: Any) -> dict[str, Any]:
    import numpy as np

    array = _as_numpy(value)
    contiguous = np.ascontiguousarray(array)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        summary["all_finite"] = bool(finite.all())
        if finite.any():
            summary["min"] = _safe_number(array[finite].min())
            summary["max"] = _safe_number(array[finite].max())
    return summary


def _flatten_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        leaves: dict[str, Any] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_leaves(value[key], path))
        return leaves
    return {prefix or "<root>": value}


def _tree_summary(value: Any) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for path, leaf in _flatten_leaves(value).items():
        if isinstance(leaf, (str, bytes)):
            summaries[path] = {"type": type(leaf).__name__, "value": str(leaf)}
        elif isinstance(leaf, (list, tuple)) and all(isinstance(item, str) for item in leaf):
            summaries[path] = {"type": type(leaf).__name__, "value": list(leaf)}
        else:
            try:
                summaries[path] = _array_summary(leaf)
            except Exception:
                summaries[path] = {"type": type(leaf).__name__, "repr": repr(leaf)}
    return summaries


def _tree_key_paths(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return [prefix or "<root>"]
    paths: list[str] = []
    for key in sorted(value, key=str):
        path = f"{prefix}.{key}" if prefix else str(key)
        paths.append(path)
        if isinstance(value[key], Mapping):
            paths.extend(_tree_key_paths(value[key], path))
    return paths


def _compare_trees(first: Any, second: Any) -> dict[str, Any]:
    import numpy as np

    first_leaves = _flatten_leaves(first)
    second_leaves = _flatten_leaves(second)
    paths_match = set(first_leaves) == set(second_leaves)
    per_path: dict[str, Any] = {}
    all_exact = paths_match
    state_exact = paths_match
    image_exact = paths_match
    for path in sorted(set(first_leaves) | set(second_leaves)):
        if path not in first_leaves or path not in second_leaves:
            per_path[path] = {"present_in_both": False, "exact": False}
            all_exact = False
            if "pixel" in path or "image" in path:
                image_exact = False
            else:
                state_exact = False
            continue
        left = _as_numpy(first_leaves[path])
        right = _as_numpy(second_leaves[path])
        shape_match = left.shape == right.shape
        exact = bool(shape_match and np.array_equal(left, right))
        entry: dict[str, Any] = {
            "present_in_both": True,
            "shape_match": bool(shape_match),
            "exact": exact,
        }
        if shape_match and left.size and np.issubdtype(left.dtype, np.number):
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            entry["max_abs_delta"] = _safe_number(delta.max())
        per_path[path] = entry
        all_exact = all_exact and exact
        if "pixel" in path or "image" in path:
            image_exact = image_exact and exact
        else:
            state_exact = state_exact and exact
    return {
        "paths_match": paths_match,
        "all_exact": all_exact,
        "state_exact": state_exact,
        "image_exact": image_exact,
        "per_path": per_path,
    }


def _space_summary(space: Any) -> dict[str, Any]:
    if hasattr(space, "spaces"):
        return {
            "type": type(space).__name__,
            "spaces": {str(key): _space_summary(value) for key, value in space.spaces.items()},
        }
    summary: dict[str, Any] = {
        "type": type(space).__name__,
        "shape": list(getattr(space, "shape", ()) or ()),
        "dtype": str(getattr(space, "dtype", None)),
    }
    if hasattr(space, "low") and hasattr(space, "high"):
        summary["low"] = _array_summary(space.low)
        summary["high"] = _array_summary(space.high)
    return summary


def _make_environment(kind: str, args: argparse.Namespace):
    from lerobot.envs.configs import LiberoEnv, PushtEnv
    from lerobot.envs.factory import make_env

    if kind == "pusht":
        config = PushtEnv(
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            episode_length=args.episode_length,
        )
        environments = make_env(config, n_envs=1, use_async_envs=False)
        return config, environments["pusht"][0]

    # hf-libero's get_assets_path() does not consult config.yaml. Bind its
    # process-local cache explicitly so a verified external asset directory is
    # used and the runtime cannot silently download a second copy.
    from libero import libero as libero_runtime

    libero_runtime._assets_path_cache = str(args.libero_assets_dir.resolve())
    config = LiberoEnv(
        task="libero_spatial",
        task_ids=[args.libero_task_id],
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        episode_length=args.episode_length,
        control_mode="relative",
    )
    environments = make_env(config, n_envs=1, use_async_envs=False)
    return config, environments["libero_spatial"][args.libero_task_id]


def _processed_observation(kind: str, environment: Any, observation: Any):
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    processed = preprocess_observation(observation)
    processed = add_envs_task(environment, processed)
    if kind == "libero":
        from lerobot.processor.env_processor import LiberoProcessorStep

        processed = LiberoProcessorStep().observation(processed)
    return processed


def _expected_action(kind: str):
    import numpy as np

    if kind == "pusht":
        return np.asarray([256.0, 256.0], dtype=np.float32), 0.0, 512.0
    return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), -1.0, 1.0


def _required_paths(kind: str) -> tuple[set[str], set[str]]:
    if kind == "pusht":
        return (
            {"agent_pos", "pixels"},
            {"observation.image", "observation.state", "task"},
        )
    return (
        {
            "pixels.image",
            "pixels.image2",
            "robot_state.eef.pos",
            "robot_state.eef.quat",
            "robot_state.gripper.qpos",
            "robot_state.joints.pos",
        },
        {
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "task",
        },
    )


def _runtime_once(kind: str, args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    import numpy as np

    config, environment = _make_environment(kind, args)
    try:
        observation, reset_info = environment.reset(seed=args.seed)
        processed = _processed_observation(kind, environment, observation)
        rendered = environment.call("render")
        rendered_first = rendered[0] if isinstance(rendered, (tuple, list)) else rendered

        action, expected_low, expected_high = _expected_action(kind)
        single_action_space = environment.single_action_space
        action_valid = bool(single_action_space.contains(action))
        next_observation, reward, terminated, truncated, step_info = environment.step(
            np.expand_dims(action, axis=0)
        )

        raw_leaf_paths = set(_flatten_leaves(observation))
        processed_leaf_paths = set(_flatten_leaves(processed))
        required_raw, required_processed = _required_paths(kind)
        low = _as_numpy(single_action_space.low)
        high = _as_numpy(single_action_space.high)
        reward_array = _as_numpy(reward)
        action_bounds_match = bool(
            np.allclose(low, expected_low)
            and np.allclose(high, expected_high)
            and list(single_action_space.shape) == list(action.shape)
        )
        processed_state = processed.get("observation.state")
        processed_state_dim = int(_as_numpy(processed_state).shape[-1]) if processed_state is not None else None
        expected_state_dim = 2 if kind == "pusht" else 8

        checks = {
            "reset_returned_observation": observation is not None,
            "raw_required_paths_present": required_raw.issubset(raw_leaf_paths),
            "processed_required_paths_present": required_processed.issubset(processed_leaf_paths),
            "processed_state_dim_matches": processed_state_dim == expected_state_dim,
            "native_action_shape_and_bounds_match": action_bounds_match,
            "chosen_action_inside_native_bounds": action_valid,
            "render_returned_finite_rgb": bool(
                _as_numpy(rendered_first).ndim == 3
                and _as_numpy(rendered_first).shape[-1] == 3
                and np.isfinite(_as_numpy(rendered_first)).all()
            ),
            "one_step_returned_finite_reward": bool(reward_array.size and np.isfinite(reward_array).all()),
            "one_step_observation_returned": next_observation is not None,
        }

        inner = environment.envs[0]
        libero_paths = None
        effective_libero_assets_path = None
        if kind == "libero":
            from libero.libero import get_assets_path, get_libero_path

            libero_paths = {}
            for path_name in ("bddl_files", "init_states", "assets"):
                resolved = Path(get_libero_path(path_name)).resolve()
                libero_paths[path_name] = {"path": str(resolved), "exists": resolved.exists()}
            effective_libero_assets_path = str(Path(get_assets_path()).resolve())
            checks["libero_assets_path_matches_explicit"] = (
                Path(effective_libero_assets_path) == args.libero_assets_dir.resolve()
            )
        runtime = {
            "config": {
                "type": config.type,
                "task": config.task,
                "task_ids": getattr(config, "task_ids", None),
                "obs_type": config.obs_type,
                "render_mode": config.render_mode,
                "episode_length": config.episode_length,
                "control_mode": getattr(config, "control_mode", None),
            },
            "environment_class": f"{type(inner).__module__}.{type(inner).__name__}",
            "metadata": dict(getattr(inner, "metadata", {})),
            "task_name": getattr(inner, "task", config.task),
            "task_description": getattr(inner, "task_description", ""),
            "task_id": getattr(inner, "task_id", 0),
            "camera_names": getattr(inner, "camera_name", None),
            "libero_paths": libero_paths,
            "effective_libero_assets_path": effective_libero_assets_path,
            "libero_init_state_count": (
                int(len(inner._init_states))
                if kind == "libero" and getattr(inner, "_init_states", None) is not None
                else None
            ),
            "reset_seed": args.seed,
            "reset_info_keys": _tree_key_paths(reset_info),
            "raw_observation": _tree_summary(observation),
            "processed_observation": _tree_summary(processed),
            "observation_space": _space_summary(environment.single_observation_space),
            "native_action_space": _space_summary(single_action_space),
            "chosen_action": _array_summary(action),
            "rendered_frame": _array_summary(rendered_first),
            "one_step": {
                "reward": _array_summary(reward),
                "terminated": _array_summary(terminated),
                "truncated": _array_summary(truncated),
                "info_keys": _tree_key_paths(step_info),
                "next_observation": _tree_summary(next_observation),
            },
            "checks": checks,
        }
        return runtime, observation
    finally:
        environment.close()


def _run_environment(kind: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    try:
        first_runtime, first_observation = _runtime_once(kind, args)
        _, second_observation = _runtime_once(kind, args)
        determinism = _compare_trees(first_observation, second_observation)
        first_runtime["deterministic_reset_replay"] = determinism
        first_runtime["checks"].update(
            {
                "deterministic_reset_all_raw_leaves_exact": determinism["all_exact"],
                "deterministic_reset_state_exact": determinism["state_exact"],
                "deterministic_reset_images_exact": determinism["image_exact"],
            }
        )
        all_checks_passed = all(first_runtime["checks"].values())
        first_runtime["status"] = "passed" if all_checks_passed else "failed"
        first_runtime["runtime_seconds"] = round(time.monotonic() - started, 3)
        return first_runtime
    except Exception as exc:
        return {
            "status": "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines()[-30:],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--envs", choices=("all", "pusht", "libero"), default="all")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument(
        "--libero-config-dir",
        type=Path,
        default=Path("simulation_output/libero_runtime_config_v1"),
    )
    parser.add_argument(
        "--libero-assets-dir",
        type=Path,
        default=Path("simulation_output/libero_assets_v1"),
    )
    parser.add_argument("--episode-length", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    selected = ("pusht", "libero") if args.envs == "all" else (args.envs,)
    libero_config = None
    libero_setup_failure = None
    if "libero" in selected:
        try:
            libero_config = _prepare_libero_config(args.libero_config_dir, args.libero_assets_dir)
        except Exception as exc:
            libero_setup_failure = {
                "status": "failed",
                "runtime_seconds": 0.0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc().splitlines()[-30:],
            }
    started = time.monotonic()
    environments = {
        kind: libero_setup_failure if kind == "libero" and libero_setup_failure else _run_environment(kind, args)
        for kind in selected
    }
    status = "passed" if all(item["status"] == "passed" for item in environments.values()) else "failed"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "public_environment_reset_preprocess_render_one_step_only",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "python": sys.version,
        "package_versions": _package_versions(),
        "environment_variables": {
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "arguments": {
            "envs": args.envs,
            "seed": args.seed,
            "libero_task_id": args.libero_task_id,
            "libero_config_dir": str(args.libero_config_dir),
            "libero_assets_dir": str(args.libero_assets_dir),
            "episode_length": args.episode_length,
        },
        "libero_runtime_config": libero_config,
        "environments": environments,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "boundaries": {
            "dataset_loaded": False,
            "policy_or_checkpoint_loaded": False,
            "training_started": False,
            "optimizer_steps": 0,
            "public_actions_keep_native_semantics": True,
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
