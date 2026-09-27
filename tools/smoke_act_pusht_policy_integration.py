#!/usr/bin/env python
"""Zero-training ACT + Push-T one-batch/one-episode integration smoke.

The policy is deliberately small and randomly initialized. Environment bounds
provide smoke-only normalization statistics so the official ACT pre/post
processors can be exercised without downloading a dataset or checkpoint. The
result is interface evidence, never a policy-quality benchmark.
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


SCHEMA_VERSION = "act_pusht_policy_integration_smoke_v1"


def _smoke_normalization_stats() -> dict[str, dict[str, np.ndarray]]:
    """Return explicit bounds-derived stats, not dataset estimates."""
    midpoint = np.asarray([256.0, 256.0], dtype=np.float32)
    state_scale = np.asarray([256.0, 256.0], dtype=np.float32)
    action_scale = np.asarray([128.0, 128.0], dtype=np.float32)
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
    title_font = ImageFont.truetype(str(font_path), 24)
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
        (14, 12),
        "ACT + Push-T 随机权重接入 Smoke（非性能结果）",
        fill="black",
        font=title_font,
    )
    for index, snapshot in enumerate(snapshots):
        row, column = divmod(index, columns)
        left = column * panel_size
        top = title_height + row * (panel_size + label_height)
        action = snapshot.get("action")
        if action is None:
            label = "重置帧"
        else:
            label = f"步骤 {snapshot['step']}｜action=({action[0]:.1f}, {action[1]:.1f})"
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
    observation = env_preprocessor(observation)
    return policy_preprocessor(observation)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION

    started = time.monotonic()
    env_config = None
    environment = None
    try:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        env_config, environment = _make_environment("pusht", args)
        profile = BENCHMARK_PROFILES["pusht"]
        config_class, policy_class = import_policy("act")
        policy_config = config_class(**config_kwargs("act", profile, device=args.device))
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
        snapshots: list[dict[str, Any]] = [{"step": 0, "frame": _render_first(environment), "action": None}]
        first_batch = None
        first_normalized_action = None
        first_native_action = None
        per_step: list[dict[str, Any]] = []
        actions: list[np.ndarray] = []
        all_actions_finite = True
        all_actions_in_bounds = True
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
                    f"ACT smoke action violates native Push-T bounds at step {step_index + 1}: "
                    f"{action_numpy.tolist()}"
                )
            if first_batch is None:
                first_batch = _tree_summary(batch)
                first_normalized_action = _tensor_summary(normalized_action)
                first_native_action = _tensor_summary(native_action)

            raw_observation, reward, terminated, truncated, _ = environment.step(action_numpy)
            all_steps_returned = all_steps_returned and raw_observation is not None
            action_vector = action_numpy[0].astype(float)
            actions.append(action_vector)
            per_step.append(
                {
                    "step": step_index + 1,
                    "action": action_vector.tolist(),
                    "action_finite": finite,
                    "action_inside_native_bounds": inside,
                    "reward_finite": bool(np.isfinite(np.asarray(reward)).all()),
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

        if first_batch is None or first_normalized_action is None or first_native_action is None:
            raise AssertionError("no policy batch was executed")
        action_array = np.stack(actions, axis=0)
        required_batch_keys = {"observation.image", "observation.state", "task"}
        checks = {
            "one_environment_batch_processed": required_batch_keys.issubset(first_batch),
            "first_batch_image_shape_matches": first_batch["observation.image"]["shape"] == [1, 3, 96, 96],
            "first_batch_state_shape_matches": first_batch["observation.state"]["shape"] == [1, 2],
            "policy_select_action_returned_finite_2d": bool(
                first_normalized_action.get("all_finite")
                and first_normalized_action["shape"] == [1, 2]
            ),
            "policy_postprocessor_returned_finite_2d": bool(
                first_native_action.get("all_finite") and first_native_action["shape"] == [1, 2]
            ),
            "all_actions_finite": all_actions_finite,
            "all_actions_inside_native_bounds": all_actions_in_bounds,
            "all_environment_steps_returned": all_steps_returned,
            "episode_has_transitions": bool(per_step),
            "no_optimizer_constructed_or_stepped": True,
        }
        artifact = _make_rollout_sheet(snapshots, args.artifact)
        return {
            "status": "passed" if all(checks.values()) else "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "policy": {
                "name": "act",
                "class": f"{policy_class.__module__}.{policy_class.__name__}",
                "random_initialization_seed": args.seed,
                "pretrained_policy_weights_loaded": False,
                "pretrained_backbone_weights_loaded": False,
                "parameter_count": int(sum(parameter.numel() for parameter in policy.parameters())),
                "config": {
                    "chunk_size": policy_config.chunk_size,
                    "n_action_steps": policy_config.n_action_steps,
                    "dim_model": policy_config.dim_model,
                    "n_heads": policy_config.n_heads,
                    "n_encoder_layers": policy_config.n_encoder_layers,
                    "n_decoder_layers": policy_config.n_decoder_layers,
                    "use_vae": policy_config.use_vae,
                },
            },
            "profile": {
                **profile.to_dict(),
                "fingerprint": profile_fingerprint(profile),
            },
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
            },
            "reset_seed": args.seed,
            "reset_info_keys": sorted(str(key) for key in reset_info),
            "first_policy_batch": first_batch,
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
    # Kept for compatibility with the shared environment factory signature.
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument("--libero-assets-dir", type=Path, default=Path("simulation_output/libero_assets_v1"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    result = _run(args)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": result["status"],
        "scope": "act_pusht_random_weight_one_batch_one_episode_integration_only",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "python": sys.version,
        "package_versions": _package_versions(),
        "result": result,
        "boundaries": {
            "public_dataset_loaded": False,
            "project_dataset_loaded": False,
            "checkpoint_loaded": False,
            "pretrained_weights_loaded": False,
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
        f"status={report['status']} policy=act env=pusht transitions={transitions} "
        f"random_weights=true optimizer_steps=0 report={args.out}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

