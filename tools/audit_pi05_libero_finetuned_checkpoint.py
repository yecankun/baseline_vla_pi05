from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pi05_pretrained_loader import PI05PretrainedLoadRejected, load_verified_pi05_weights


SCHEMA_VERSION = "pi05_libero_finetuned_checkpoint_audit_v1"
DEFAULT_CHECKPOINT = Path(
    "/home/zsw/models/project_2026/pi05_libero_finetuned_8e174154"
)
DEFAULT_PROTOCOL = Path("docs/libero-spatial-score-protocol-v1.json")
EXPECTED_PHYSICAL_PARENT = Path(
    "/media/zsw/SSD1T/project_2026_weights_v1/models"
)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_map(config: Any, name: str) -> dict[str, dict[str, Any]]:
    features = getattr(config, name)
    return {
        key: {"type": value.type.value, "shape": list(value.shape)}
        for key, value in features.items()
    }


def _processor_steps(pipeline: Any) -> list[str]:
    return [type(step).__name__ for step in getattr(pipeline, "steps", [])]


def _processor_state(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "tensors": {
                key: list(handle.get_slice(key).get_shape()) for key in handle.keys()
            },
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    started = time.monotonic()
    protocol = _load_json(args.protocol)
    route = protocol["checkpoint_routes"]["pi05_primary"]
    checkpoint = args.checkpoint.absolute()
    resolved = checkpoint.resolve()
    model_path = checkpoint / route["model_file"]
    required_files = {
        "config": checkpoint / "config.json",
        "model": model_path,
        "preprocessor": checkpoint / "policy_preprocessor.json",
        "postprocessor": checkpoint / "policy_postprocessor.json",
        "normalizer_state": checkpoint
        / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "unnormalizer_state": checkpoint
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    missing_files = [name for name, path in required_files.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"missing checkpoint files: {missing_files}")
    if not resolved.is_relative_to(EXPECTED_PHYSICAL_PARENT):
        raise ValueError(
            f"checkpoint must resolve below {EXPECTED_PHYSICAL_PARENT}, got {resolved}"
        )
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("audit must run with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")

    model_sha256 = _sha256(model_path)
    raw_config = _load_json(required_files["config"])
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = args.device
    config.compile_model = False
    config.gradient_checkpointing = False
    config.n_action_steps = 1
    config.num_inference_steps = 2

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
    )
    policy = PI05Policy(config).to(args.device)
    load_report = load_verified_pi05_weights(
        policy,
        checkpoint,
        revision=None,
        local_files_only=True,
        min_loaded_parameter_fraction=1.0,
        max_missing_keys=0,
        max_unexpected_keys=0,
        compute_file_sha256=False,
    )

    input_features = _feature_map(config, "input_features")
    output_features = _feature_map(config, "output_features")
    smoke_info = _load_json(args.official_smoke_info)
    smoke_video_paths = smoke_info["overall"].get("video_paths", [])
    smoke_videos_exist = all((args.project_root / path).is_file() for path in smoke_video_paths)
    checks = {
        "checkpoint_logical_path_exists": checkpoint.is_dir(),
        "checkpoint_resolves_to_ssd": resolved.is_relative_to(EXPECTED_PHYSICAL_PARENT),
        "protocol_repo_matches": route["repo_id"] == "lerobot/pi05_libero_finetuned",
        "protocol_revision_is_pinned": route["revision"]
        == "8e174154ef5f6c60a8da12ae99c303d8963138c1",
        "model_size_matches_protocol": model_path.stat().st_size == route["model_size_bytes"],
        "model_sha256_matches_protocol": model_sha256 == route["model_sha256"],
        "raw_config_type_pi05": raw_config.get("type") == "pi05",
        "two_native_camera_features_match": {
            key: input_features[key]["shape"]
            for key in ("observation.images.image", "observation.images.image2")
        }
        == {
            "observation.images.image": [3, 256, 256],
            "observation.images.image2": [3, 256, 256],
        },
        "state_feature_is_8d": input_features["observation.state"]["shape"] == [8],
        "action_feature_is_7d": output_features == {
            "action": {"type": "ACTION", "shape": [7]}
        },
        "one_declared_empty_camera": input_features[
            "observation.images.empty_camera_0"
        ]["shape"]
        == [3, 224, 224]
        and config.empty_cameras == 1,
        "normalization_mapping_matches_protocol": {
            key: value.value for key, value in config.normalization_mapping.items()
        }
        == route["normalization"],
        "preprocessor_loaded_from_checkpoint": bool(_processor_steps(preprocessor)),
        "postprocessor_loaded_from_checkpoint": bool(_processor_steps(postprocessor)),
        "verified_load_status_loaded": load_report["status"] == "loaded",
        "unique_parameter_coverage_one": load_report["loaded_parameter_fraction"] == 1.0,
        "no_effective_missing_keys": load_report["missing_key_count"] == 0,
        "no_unexpected_keys": load_report["unexpected_key_count"] == 0,
        "no_shape_mismatches": load_report["mismatched_shape_count"] == 0,
        "one_verified_tied_alias": load_report["tied_weight_alias_count"] == 1,
        "tied_alias_source_and_target_match": all(
            item["source_matches_checkpoint"]
            and item["alias_matches_checkpoint"]
            and item["shared_storage_after_load"]
            for item in load_report["tied_weight_alias_verification"]
        ),
        "official_smoke_is_one_episode": smoke_info["overall"]["n_episodes"] == 1,
        "official_smoke_is_task_zero": len(smoke_info["per_task"]) == 1
        and smoke_info["per_task"][0]["task_id"] == 0,
        "official_smoke_video_exists": smoke_videos_exist,
        "no_optimizer_or_training": True,
        "no_score_claim": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "runtime_seconds": round(time.monotonic() - started, 3),
        "scope": "checkpoint_loader_and_schema_preflight_only",
        "checkpoint": {
            "logical_path": str(checkpoint),
            "resolved_path": str(resolved),
            "model_size_bytes": model_path.stat().st_size,
            "model_sha256": model_sha256,
            "repo_id": route["repo_id"],
            "revision": route["revision"],
        },
        "config": {
            "input_features": input_features,
            "output_features": output_features,
            "normalization_mapping": {
                key: value.value for key, value in config.normalization_mapping.items()
            },
            "chunk_size": config.chunk_size,
            "audit_n_action_steps": config.n_action_steps,
            "audit_num_inference_steps": config.num_inference_steps,
            "compile_model": config.compile_model,
            "gradient_checkpointing": config.gradient_checkpointing,
        },
        "processors": {
            "preprocessor_steps": _processor_steps(preprocessor),
            "postprocessor_steps": _processor_steps(postprocessor),
            "normalizer_state": _processor_state(required_files["normalizer_state"]),
            "unnormalizer_state": _processor_state(required_files["unnormalizer_state"]),
        },
        "verified_load": load_report,
        "official_one_step_smoke": {
            "info_path": str(args.official_smoke_info),
            "episodes": smoke_info["overall"]["n_episodes"],
            "task_id": smoke_info["per_task"][0]["task_id"],
            "environment_steps": 1,
            "video_paths": smoke_video_paths,
            "videos_exist": smoke_videos_exist,
            "success_metric_reportable": False,
        },
        "checks": checks,
        "boundaries": {
            "environment_constructed_by_audit": False,
            "policy_inference_by_audit": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "training_started": False,
            "gate_b4a_executed": False,
            "benchmark_score_claim_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed local PI0.5 LIBERO checkpoint and tied-alias audit."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--official-smoke-info",
        type=Path,
        default=Path("simulation_output/pi05_libero_finetuned_loader_smoke_v1/eval_info.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        report = _run(args)
    except PI05PretrainedLoadRejected as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "verified_load": exc.report,
            "traceback": traceback.format_exc().splitlines()[-80:],
        }
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines()[-80:],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checks = report.get("checks", {})
    false_checks = [name for name, passed in checks.items() if not passed]
    print(
        f"status={report['status']} checks={len(checks)} false_checks={len(false_checks)} "
        f"optimizer_steps={report.get('boundaries', {}).get('optimizer_steps')} "
        f"gate_b4a_executed={report.get('boundaries', {}).get('gate_b4a_executed')} "
        f"out={args.out}",
        flush=True,
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
