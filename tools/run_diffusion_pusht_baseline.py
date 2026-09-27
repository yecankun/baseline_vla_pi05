#!/usr/bin/env python
"""Frozen clean Push-T evaluation for the separately verified Diffusion checkpoint.

Preflight imports/checks schema only. Smoke and benchmark require --execute,
never train, never retry, and refuse an existing output directory.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "schema": "diffusion_pusht_clean_baseline_v1",
    "environment": "gym_pusht/PushT-v0",
    "clean_only": True,
    "smoke_seeds": [20260913],
    "benchmark_seeds": list(range(100000, 100100)),
    "smoke_max_steps": 20,
    "benchmark_max_steps": 300,
    "env_episode_length": 300,
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "noise_scheduler_type": "DDPM",
    "num_train_timesteps": 100,
    "num_inference_steps": 100,
    "config_num_inference_steps": None,
    "native_action_dim": 2,
    "native_action_bounds": [0.0, 512.0],
    "raw_image_shape": [96, 96, 3],
    "visualization_shape": [384, 384, 3],
    "device": "cuda",
    "amp": False,
    "tf32": False,
    "deterministic_algorithms": True,
    "episode_rng": "reset python/numpy/torch/cuda to episode seed",
    "success_source": "environment info.is_success, never reward",
    "coverage_source": "environment info.coverage, never reward",
    "autoreset": "step synchronous vector.envs[0] directly; stop on first done",
    "clip_native_action": False,
    "retry": "none; explicit fresh attempt only",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_plan(stage: str) -> list[int]:
    if stage not in {"smoke", "benchmark"}:
        raise ValueError("stage must be smoke or benchmark")
    return list(PROTOCOL[f"{stage}_seeds"])


def output_directory(stage: str, attempt: str | None = None, root: Path = ROOT) -> Path:
    seed_plan(stage)
    if attempt is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", attempt):
        raise ValueError("attempt must be 1..48 letters/digits/underscore/hyphen, no path components")
    suffix = f"_{attempt}" if attempt else ""
    return root / "simulation_output" / f"diffusion_pusht_{stage}_v1{suffix}"


def _scalar(value: Any, name: str) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(f"{name} must contain exactly one environment value")
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (tuple, list, Mapping)):
        raise ValueError(f"{name} must be scalar")
    return value


def _boolean(value: Any, name: str) -> bool:
    value = _scalar(value, name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def info_metrics(info: Mapping[str, Any], *, terminal: bool) -> dict[str, Any]:
    """Read actual environment information, including explicitly valid final_info."""
    if not isinstance(info, Mapping):
        raise ValueError("environment info must be a mapping")
    selected = info
    source = "info"
    if terminal and "final_info" in info:
        valid = _boolean(info.get("_final_info", True), "_final_info")
        if valid:
            final = info["final_info"]
            if hasattr(final, "tolist"):
                final = final.tolist()
            if isinstance(final, (list, tuple)) and len(final) == 1:
                final = final[0]
            if not isinstance(final, Mapping):
                raise ValueError("valid final_info must contain one mapping")
            selected, source = final, "final_info"
    if "is_success" not in selected or "coverage" not in selected:
        raise ValueError("actual info.is_success and info.coverage are required; reward is not a substitute")
    success = _boolean(selected["is_success"], "is_success")
    coverage = _scalar(selected["coverage"], "coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise ValueError("coverage must be numeric")
    coverage = float(coverage)
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be finite in [0,1]")
    return {"is_success": success, "coverage": coverage, "source": source}


def wilson95(successes: int, count: int) -> list[float] | None:
    if not isinstance(successes, int) or not isinstance(count, int) or not 0 <= successes <= count:
        raise ValueError("invalid binomial counts")
    if count == 0:
        return None
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def aggregate(episodes: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    planned = seed_plan(stage)
    observed = [episode["seed"] for episode in episodes]
    prefix_matches = observed == planned[:len(observed)] and len(observed) <= len(planned)
    if not prefix_matches:
        raise ValueError("episode records do not match the frozen seed-plan prefix")
    count = len(episodes)
    successes = sum(bool(item["is_success"]) for item in episodes)
    complete = observed == planned
    outside_steps = sum(item["native_action_out_of_bounds_steps"] for item in episodes)
    outside_elements = sum(item["native_action_out_of_bounds_elements"] for item in episodes)
    return {
        "episodes_completed": count, "episodes_planned": len(planned),
        "seed_schedule_complete": complete, "successes": successes,
        "observed_success_rate": successes / count if count else None,
        "success_wilson95": wilson95(successes, count),
        "mean_episode_max_coverage": sum(item["max_coverage"] for item in episodes) / count if count else None,
        "native_action_out_of_bounds_steps": outside_steps,
        "native_action_out_of_bounds_elements": outside_elements,
        "strict_protocol_pass": complete and outside_steps == 0,
        "benchmark_score_claim_allowed": stage == "benchmark" and complete and outside_steps == 0,
    }


def verify_migration_pin(path: Path, expected: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("migration report SHA256 must be exactly 64 hexadecimal characters")
    actual = _sha(path)
    if actual != expected.lower():
        raise ValueError("migration report does not match the explicitly pinned SHA256")
    return actual


def verify_smoke_report(path: Path, binding: dict[str, Any], *, current_sources: dict[str, Any] | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("stage") != "smoke"
        or report.get("status") != "completed"
        or report.get("strict_protocol_pass") is not True
        or report.get("protocol_sha256") != canonical_hash(PROTOCOL)
        or report.get("checkpoint_binding") != binding
    ):
        raise ValueError("benchmark requires a completed strict-pass smoke under exactly the same protocol/binding")
    summary = aggregate(report.get("episodes", []), "smoke")
    if not summary["strict_protocol_pass"]:
        raise ValueError("smoke episode trace summary is incomplete or out of bounds")
    if current_sources is not None:
        actual = {name: item["sha256"] for name, item in current_sources.items()}
        expected = {name: item["sha256"] for name, item in report.get("source_files", {}).items()}
        if actual != expected:
            raise ValueError("source file hashes differ from the successful smoke")
    return {"path": str(path.resolve()), "sha256": _sha(path), "environment_reference": {name: report.get(name) for name in ("environment_gym_kwargs", "environment_step_source_sha256", "package_versions")}}


def verify_environment_parity(reference: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("environment_gym_kwargs", "environment_step_source_sha256", "package_versions"):
        if reference.get(key) is None or reference.get(key) != current.get(key):
            raise ValueError(f"environment/runtime differs from successful smoke: {key}")


def validate_env_kwargs(kwargs: dict[str, Any]) -> None:
    expected = {"obs_type": "pixels_agent_pos", "render_mode": "rgb_array", "visualization_width": 384,
                "visualization_height": 384, "max_episode_steps": 300}
    if kwargs != expected:
        raise ValueError("native Push-T gym kwargs differ from the frozen protocol")


def validate_native_action(action: Any, low: Any, high: Any) -> tuple[Any, int]:
    import numpy as np
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    array = np.asarray(action)
    if array.shape != (1, 2) or array.dtype.kind != "f" or not np.isfinite(array).all():
        raise ValueError("native action must be finite floating [1,2]")
    low, high = np.asarray(low), np.asarray(high)
    if low.shape != (2,) or high.shape != (2,) or not np.array_equal(low, [0, 0]) or not np.array_equal(high, [512, 512]):
        raise ValueError("native Push-T action-space bounds changed")
    outside = int(((array[0] < low) | (array[0] > high)).sum())
    return array, outside  # Deliberately no clipping or dtype/value conversion.


def _write(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _imports() -> dict[str, Any]:
    import numpy as np
    import torch
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.configs import PushtEnv
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    if __package__ in {None, ""}:
        from diffusion_pusht_checkpoint import MIGRATED, load_verified
        from smoke_lerobot_public_envs import _make_environment, _package_versions
    else:
        from .diffusion_pusht_checkpoint import MIGRATED, load_verified
        from .smoke_lerobot_public_envs import _make_environment, _package_versions
    return locals()


def _seed(seed: int, runtime: dict[str, Any]) -> None:
    random.seed(seed)
    runtime["np"].random.seed(seed)
    runtime["torch"].manual_seed(seed)
    runtime["torch"].cuda.manual_seed_all(seed)


def _render(env: Any, np: Any) -> Any:
    frame = np.asarray(env.render())
    if list(frame.shape) != PROTOCOL["visualization_shape"] or frame.dtype != np.uint8:
        raise ValueError("native Push-T render must be uint8 RGB 384x384")
    return frame.copy()


def _sheet(snapshots: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFont
    fonts = [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("C:/Windows/Fonts/msyh.ttc")]
    font_path = next((font for font in fonts if font.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("Chinese font missing; no unlabelled or missing-glyph substitute")
    title_font = ImageFont.truetype(str(font_path), 21)
    label_font = ImageFont.truetype(str(font_path), 18)
    width, panel, title, label = 768, 384, 74, 32
    canvas = Image.new("RGB", (width, title + math.ceil(len(snapshots) / 2) * (panel + label)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), "Diffusion Policy / Push-T 已验证预训练权重", font=title_font, fill="black")
    draw.text((12, 39), "20 步接入检查｜非完整基准成绩｜无训练", font=label_font, fill="black")
    for index, item in enumerate(snapshots):
        left, top = (index % 2) * panel, title + (index // 2) * (panel + label)
        draw.text((left + 8, top + 4), f"步骤 {item['step']}", font=label_font, fill="black")
        canvas.paste(Image.fromarray(item["frame"]), (left, top + label))
    canvas.save(path)
    return {"path": str(path), "sha256": _sha(path), "font": str(font_path), "visual_status": "not_viewed"}


def _validate_policy(policy: Any) -> None:
    for name in ("n_obs_steps", "horizon", "n_action_steps", "noise_scheduler_type", "num_train_timesteps"):
        if getattr(policy.config, name, None) != PROTOCOL[name]:
            raise ValueError(f"checkpoint config differs from frozen {name}={PROTOCOL[name]}")
    if policy.config.num_inference_steps is not None or policy.diffusion.num_inference_steps != 100:
        raise ValueError("preserve source config num_inference_steps=None with effective DDPM 100")
    if any(module.training for module in policy.modules()):
        raise ValueError("verified policy must already be in eval mode")
    parameters = list(policy.parameters())
    if not parameters or any(str(param.dtype) != "torch.float32" or param.device.type != "cuda" or param.requires_grad for param in parameters):
        raise ValueError("all verified policy parameters must be frozen float32 on CUDA")


def run(stage: str, output: Path, runtime: dict[str, Any], migration_sha: str | None = None, smoke_report: Path | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = {"runner": {"path": str(Path(__file__).resolve()), "sha256": _sha(Path(__file__))}}
    report: dict[str, Any] = {
        "schema": PROTOCOL["schema"], "stage": stage, "protocol": PROTOCOL,
        "protocol_sha256": canonical_hash(PROTOCOL), "source_files": sources,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "status": "running", "checkpoint_binding": None, "episodes": [],
        "training_started": False, "optimizer_steps": 0, "clean_only": True,
        "guidewire_or_real_system_claim_allowed": False,
    }
    _write(output / "started.json", report)
    _write(output / "status.json", {"status": "running", "pid": os.getpid(), "episodes_completed": 0})
    vector = None
    try:
        for name in ("load_verified", "_make_environment", "preprocess_observation", "add_envs_task", "make_env_pre_post_processors"):
            path = Path(inspect.getfile(runtime[name]))
            sources[name] = {"path": str(path), "sha256": _sha(path)}
        if stage == "benchmark" and migration_sha is None:
            raise ValueError("benchmark requires --migration-report-sha256")
        if migration_sha is not None:
            report["explicit_migration_report_sha256"] = verify_migration_pin(runtime["MIGRATED"] / "migration_report.json", migration_sha)
        np, torch = runtime["np"], runtime["torch"]
        if not torch.cuda.is_available():
            raise RuntimeError("frozen protocol requires CUDA; CPU fallback is forbidden")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        policy, pre, post, binding = runtime["load_verified"](device="cuda")
        report["checkpoint_binding"] = binding
        if migration_sha is not None and binding["migration_report_sha256"] != migration_sha.lower():
            raise ValueError("loaded binding differs from the explicitly pinned migration report")
        if stage == "benchmark":
            report["smoke_prerequisite"] = verify_smoke_report(smoke_report or output_directory("smoke") / "report.json", binding, current_sources=sources)
        _validate_policy(policy)
        env_config, vector = runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
        if len(vector.envs) != 1:
            raise ValueError("exactly one synchronous native environment is required")
        env = vector.envs[0]
        report["environment_gym_kwargs"] = env_config.gym_kwargs
        validate_env_kwargs(report["environment_gym_kwargs"])
        report["environment_step_source_sha256"] = canonical_hash(inspect.getsource(type(env.unwrapped).step))
        report["package_versions"] = runtime["_package_versions"]()
        if stage == "benchmark":
            verify_environment_parity(report["smoke_prerequisite"]["environment_reference"], report)
        env_pre, env_post = runtime["make_env_pre_post_processors"](env_config, policy.config)
        report["processor_order"] = ["preprocess_observation", "add_envs_task", "env_preprocessor", "policy_preprocessor", "policy.select_action", "policy_postprocessor", "env_postprocessor", "single_environment.step"]
        report["processor_steps"] = {name: [type(step).__name__ for step in pipeline.steps] for name, pipeline in (("env_pre", env_pre), ("pre", pre), ("post", post), ("env_post", env_post))}
        _write(output / "started.json", report)
        snapshots: list[dict[str, Any]] = []
        for episode_index, seed in enumerate(seed_plan(stage)):
            episode_start = time.monotonic()
            _seed(seed, runtime)
            policy.reset()
            for processor in (pre, post, env_pre, env_post):
                processor.reset()
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            raw, _reset_info = env.reset(seed=seed)
            reset_pixels = np.asarray(raw.get("pixels"))
            if list(reset_pixels.shape) != PROTOCOL["raw_image_shape"] or reset_pixels.dtype != np.uint8:
                raise ValueError("actual native Push-T observation must be uint8 96x96x3")
            if np.asarray(raw.get("agent_pos")).shape != (2,):
                raise ValueError("actual native Push-T agent_pos must be 2D")
            reset_hash = hashlib.sha256(reset_pixels.tobytes() + np.asarray(raw["agent_pos"]).tobytes()).hexdigest()
            _append(output / "per_step.jsonl", {"kind": "reset", "episode": episode_index, "seed": seed, "observation_sha256": reset_hash})
            if stage == "smoke":
                snapshots.append({"step": 0, "frame": _render(env, np)})
            coverages: list[float] = []
            success, outside_steps, outside_elements = False, 0, 0
            terminated = truncated = False
            max_steps = PROTOCOL[f"{stage}_max_steps"]
            for step_index in range(1, max_steps + 1):
                batch = runtime["preprocess_observation"](raw)
                batch = runtime["add_envs_task"](vector, batch)
                batch = env_pre(batch)
                batch = pre(batch)
                if tuple(batch["observation.image"].shape) != (1, 3, 96, 96) or tuple(batch["observation.state"].shape) != (1, 2):
                    raise ValueError("processed policy input shape mismatch")
                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                    predicted = policy.select_action(batch)
                    if tuple(predicted.shape) != (1, 2) or not bool(torch.isfinite(predicted).all()):
                        raise ValueError("predicted normalized action must be finite [1,2]")
                    if len(policy.diffusion.noise_scheduler.timesteps) != 100:
                        raise ValueError("actual DDPM inference schedule is not 100 steps")
                    native = env_post({"action": post(predicted)})["action"]
                action, n_outside = validate_native_action(native, env.action_space.low, env.action_space.high)
                outside_steps += int(n_outside > 0)
                outside_elements += n_outside
                _append(output / "per_step.jsonl", {"kind": "action", "episode": episode_index, "seed": seed, "step": step_index, "native_action": action[0].tolist(), "native_action_out_of_bounds_elements": n_outside})
                raw, reward, term, trunc, info = env.step(action[0])
                terminated, truncated = _boolean(term, "terminated"), _boolean(trunc, "truncated")
                metrics = info_metrics(info, terminal=terminated or truncated)
                numeric_reward = float(_scalar(reward, "reward"))
                if not math.isfinite(numeric_reward):
                    raise ValueError("nonfinite environment reward")
                coverages.append(metrics["coverage"])
                success = success or metrics["is_success"]
                row = {"kind": "step", "episode": episode_index, "seed": seed, "step": step_index, "native_action": action[0].tolist(), "native_action_out_of_bounds_elements": n_outside, "reward": numeric_reward, "terminated": terminated, "truncated": truncated, **metrics}
                _append(output / "per_step.jsonl", row)
                if stage == "smoke" and (step_index in {1, 10, max_steps} or terminated or truncated):
                    snapshots.append({"step": step_index, "frame": _render(env, np)})
                if terminated or truncated:
                    break
            episode = {"episode": episode_index, "seed": seed, "steps": len(coverages), "is_success": success, "max_coverage": max(coverages), "terminated": terminated, "truncated": truncated, "stopped_at_stage_cap": not (terminated or truncated), "native_action_out_of_bounds_steps": outside_steps, "native_action_out_of_bounds_elements": outside_elements, "seconds": time.monotonic() - episode_start, "reset_observation_sha256": reset_hash}
            report["episodes"].append(episode)
            _append(output / "per_episode.jsonl", episode)
            _write(output / "status.json", {"status": "running", "pid": os.getpid(), "episodes_completed": len(report["episodes"]), "seed": seed})
            print(f"stage={stage} completed={len(report['episodes'])}/{len(seed_plan(stage))} seed={seed} steps={episode['steps']} success={episode['is_success']} seconds={episode['seconds']:.3f}", flush=True)
        report.update(aggregate(report["episodes"], stage))
        if stage == "smoke":
            report["visual_artifact"] = _sheet(snapshots, output / "smoke_sheet_zh.png")
            report["benchmark_runtime_estimate_seconds_full_horizon_from_smoke"] = sum(item["seconds"] for item in report["episodes"]) / sum(item["steps"] for item in report["episodes"]) * 300 * 100
            report["runtime_estimate_caveat"] = "20-step throughput extrapolation only; early success and warmup change full-run runtime"
        report["status"] = "completed"
    except BaseException as exc:
        report.update({"status": "failed", "strict_protocol_pass": False, "benchmark_score_claim_allowed": False, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "partial_trace_preserved": True})
    finally:
        if vector is not None:
            try:
                vector.close()
            except Exception as exc:
                report.update({"status": "failed", "strict_protocol_pass": False, "benchmark_score_claim_allowed": False, "close_error": str(exc)})
        report["runtime_seconds"] = time.monotonic() - started
        _write(output / "report.json", report)
        _write(output / "status.json", {"status": report["status"], "pid": os.getpid(), "exit_code": 0 if report["status"] == "completed" else 1, "episodes_completed": len(report["episodes"]), "strict_protocol_pass": report.get("strict_protocol_pass", False)})
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "benchmark"), default="preflight")
    parser.add_argument("--execute", action="store_true", help="Explicitly execute the selected frozen stage")
    parser.add_argument("--attempt", help="Explicit fresh output suffix after an inspected failure; never auto-retry")
    parser.add_argument("--migration-report-sha256", help="Pin the previously audited migration report (required for benchmark)")
    parser.add_argument("--smoke-report", type=Path, help="Successful matching smoke report; default fixed smoke v1 report")
    args = parser.parse_args(argv)
    if args.stage != "preflight" and not args.execute:
        parser.error("smoke/benchmark require --execute")
    if args.stage == "preflight" and (args.execute or args.attempt):
        parser.error("preflight neither executes a policy nor creates an attempt")
    if args.stage == "benchmark" and args.migration_report_sha256 is None:
        parser.error("benchmark requires --migration-report-sha256")
    if args.migration_report_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", args.migration_report_sha256):
        parser.error("migration SHA256 must contain exactly 64 hexadecimal characters")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    runtime = _imports()
    if args.stage == "preflight":
        validate_env_kwargs(runtime["PushtEnv"](obs_type="pixels_agent_pos", render_mode="rgb_array", episode_length=300).gym_kwargs)
        print(json.dumps({"status": "passed", "stage": "preflight", "protocol": PROTOCOL, "protocol_sha256": canonical_hash(PROTOCOL), "package_versions": runtime["_package_versions"](), "checkpoint_loaded": False, "environment_constructed": False, "optimizer_steps": 0}, indent=2))
        return 0
    output = output_directory(args.stage, args.attempt)
    report = run(args.stage, output, runtime, args.migration_report_sha256, args.smoke_report)
    print(json.dumps({"status": report["status"], "strict_protocol_pass": report.get("strict_protocol_pass", False), "episodes_completed": len(report["episodes"]), "report": str(output / "report.json")}, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
