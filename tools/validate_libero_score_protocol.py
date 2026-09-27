#!/usr/bin/env python
"""Fail-closed audit for the frozen LIBERO score protocol.

This tool reads Hub metadata and small JSON configuration files only. It never
downloads model weights, videos, parquet shards, or demonstrations. Optional
runtime checks inspect the already installed LIBERO task definition and init
state file without constructing a policy or running an episode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "libero_spatial_score_protocol_audit_v1"
MAX_METADATA_RESPONSE_BYTES = 2 * 1024 * 1024
REQUIRED_ASSET_SUBDIRS = (
    "articulated_objects",
    "scenes",
    "stable_hope_objects",
    "stable_scanned_objects",
    "textures",
    "turbosquid_objects",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _http_json(url: str, network_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "project-2026-protocol-audit/1"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        payload = response.read(MAX_METADATA_RESPONSE_BYTES + 1)
        if len(payload) > MAX_METADATA_RESPONSE_BYTES:
            raise ValueError(f"metadata response exceeded {MAX_METADATA_RESPONSE_BYTES} bytes: {url}")
        network_ledger.append(
            {
                "url": url,
                "bytes_read": len(payload),
                "content_type": response.headers.get("Content-Type"),
            }
        )
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object from {url}")
    return value


def _feature_shape(config: dict[str, Any], key: str) -> list[int] | None:
    feature = config.get("input_features", {}).get(key)
    return feature.get("shape") if isinstance(feature, dict) else None


def _output_shape(config: dict[str, Any], key: str) -> list[int] | None:
    feature = config.get("output_features", {}).get(key)
    return feature.get("shape") if isinstance(feature, dict) else None


def _model_file(meta: dict[str, Any]) -> dict[str, Any] | None:
    for item in meta.get("siblings", []):
        if item.get("rfilename") == "model.safetensors":
            return item
    return None


def _repo_size(meta: dict[str, Any]) -> int:
    sizes = [item.get("size") for item in meta.get("siblings", [])]
    if any(size is None for size in sizes):
        raise ValueError(f"Hub metadata did not expose every file size for {meta.get('id')}")
    return int(sum(int(size) for size in sizes))


def _static_checks(manifest: dict[str, Any]) -> dict[str, bool]:
    environment = manifest["environment_contract"]
    observation = manifest["observation_contract"]
    action = manifest["action_contract"]
    stages = manifest["evaluation_stages"]
    task0 = stages["b4a_task0_preflight_score"]
    full = stages["b4b_full_spatial_score"]
    decision = manifest["decision"]
    large = manifest["large_file_transfer"]
    robust = manifest["paired_visual_robustness"]
    routes = manifest["checkpoint_routes"]
    return {
        "manifest_schema_matches": manifest.get("schema_version")
        == "libero_spatial_score_protocol_v1",
        "manifest_is_frozen_pre_acquisition": manifest.get("status") == "frozen_pre_acquisition",
        "public_benchmark_scope_only": manifest.get("scope") == "public_libero_benchmark_only",
        "downloads_not_authorized": not decision["checkpoint_download_authorized"]
        and not decision["dataset_download_authorized"],
        "training_and_evaluation_not_authorized": not decision["training_authorized"]
        and not decision["evaluation_authorized"],
        "suite_and_task_ids_fixed": environment["suite"] == "libero_spatial"
        and environment["task_ids"] == list(range(10))
        and environment["task_count"] == 10,
        "relative_control_and_horizon_fixed": environment["control_mode"] == "relative"
        and environment["init_states"] is True
        and environment["max_episode_steps"] == 280,
        "two_native_camera_features_fixed": observation["image_features"]
        == {
            "observation.images.image": [3, 256, 256],
            "observation.images.image2": [3, 256, 256],
        },
        "state_and_task_contract_fixed": observation["state_feature"]["dimension"] == 8
        and len(observation["state_feature"]["layout"]) == 8
        and observation["task_feature"] == "native_libero_task_instruction",
        "diagnostic_oracles_forbidden": observation["diagnostic_oracle_fields_allowed"] is False,
        "native_action_contract_fixed": action["dimension"] == 7
        and action["native_low"] == -1.0
        and action["native_high"] == 1.0
        and action["control_mode"] == "relative"
        and action["clipping_allowed"] is False
        and action["guidewire_action_reinterpretation_allowed"] is False,
        "task0_preflight_has_ten_fixed_episodes": task0["task_ids"] == [0]
        and task0["episodes_per_task"] == 10
        and task0["total_episodes"] == 10
        and task0["init_state_indices"] == list(range(10))
        and task0["episode_seeds"] == list(range(1000, 1010)),
        "task0_preflight_not_published_aggregate": task0["published_aggregate_comparison_allowed"]
        is False,
        "full_suite_has_one_hundred_fixed_episodes": full["task_ids"] == list(range(10))
        and full["episodes_per_task"] == 10
        and full["total_episodes"] == 100
        and full["init_state_indices_per_task"] == list(range(10))
        and full["episode_seeds_per_task"] == list(range(1000, 1010)),
        "sequential_eval_fixed": task0["batch_size"] == full["batch_size"] == 1
        and task0["max_parallel_tasks"] == full["max_parallel_tasks"] == 1,
        "clean_success_metric_fixed": manifest["metric_contract"]["primary"]
        == "episode_success_rate"
        and manifest["metric_contract"]["success_source"] == "libero_environment_is_success"
        and manifest["metric_contract"]["report_per_task"] is True
        and manifest["metric_contract"]["report_suite_macro_average"] is True
        and manifest["metric_contract"]["reward_as_primary_metric_allowed"] is False,
        "pi05_primary_route_selected": routes["pi05_primary"]["status"]
        == "selected_pending_user_transfer_and_gate_b4a",
        "smolvla_direct_route_rejected": routes["smolvla_existing_finetuned"]["status"]
        == "rejected_for_direct_score"
        and len(routes["smolvla_existing_finetuned"]["rejection_reasons"]) >= 4,
        "smolvla_training_route_not_authorized": routes["smolvla_secondary_training"][
            "training_authorized"
        ]
        is False,
        "paired_noise_is_evaluation_only": robust["enabled_only_after_clean_score"] is True
        and robust["noisy_checkpoint_selection_allowed"] is False
        and robust["noisy_training_allowed"] is False
        and robust["corruptions"]
        == ["gaussian_sensor_noise", "low_contrast", "blur", "occlusion"]
        and robust["severities"] == [1, 2, 3],
        "large_files_are_user_transfer": large["threshold_bytes"] == 104857600
        and large["transfer_owner"] == "user"
        and set(large["resources_requiring_user_transfer"])
        == {"lerobot/pi05_libero_finetuned", "lerobot/smolvla_base", "lerobot/libero"},
    }


def _hub_checks(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    network_ledger: list[dict[str, Any]] = []
    routes = manifest["checkpoint_routes"]
    dataset_route = manifest["demonstration_dataset_route"]

    specs = {
        "pi05_primary": ("models", routes["pi05_primary"]["repo_id"], routes["pi05_primary"]),
        "smolvla_existing": (
            "models",
            routes["smolvla_existing_finetuned"]["repo_id"],
            routes["smolvla_existing_finetuned"],
        ),
        "smolvla_base": (
            "models",
            routes["smolvla_secondary_training"]["base_repo_id"],
            {
                "revision": routes["smolvla_secondary_training"]["base_revision"],
                "repository_size_bytes": routes["smolvla_secondary_training"][
                    "base_repository_size_bytes"
                ],
                "model_size_bytes": routes["smolvla_secondary_training"]["base_model_size_bytes"],
                "model_sha256": routes["smolvla_secondary_training"]["base_model_sha256"],
            },
        ),
        "libero_dataset": ("datasets", dataset_route["repo_id"], dataset_route),
    }

    repo_evidence: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for name, (kind, repo_id, expected) in specs.items():
        meta = _http_json(f"https://huggingface.co/api/{kind}/{repo_id}?blobs=true", network_ledger)
        actual_size = _repo_size(meta)
        evidence = {
            "repo_id": repo_id,
            "revision": meta.get("sha"),
            "last_modified": meta.get("lastModified"),
            "gated": meta.get("gated"),
            "private": meta.get("private"),
            "file_count": len(meta.get("siblings", [])),
            "repository_size_bytes": actual_size,
        }
        checks[f"{name}_revision_matches"] = meta.get("sha") == expected["revision"]
        checks[f"{name}_repository_size_matches"] = actual_size == expected["repository_size_bytes"]
        checks[f"{name}_is_public_and_ungated"] = meta.get("private") is False and not meta.get("gated")
        if kind == "models":
            model_file = _model_file(meta)
            lfs = model_file.get("lfs", {}) if model_file else {}
            evidence["model_file"] = {
                "size_bytes": model_file.get("size") if model_file else None,
                "lfs_sha256": lfs.get("sha256"),
            }
            checks[f"{name}_model_size_matches"] = bool(
                model_file and model_file.get("size") == expected["model_size_bytes"]
            )
            checks[f"{name}_model_lfs_sha256_matches"] = lfs.get("sha256") == expected[
                "model_sha256"
            ]
        repo_evidence[name] = evidence

    pi = routes["pi05_primary"]
    pi_config = _http_json(
        f"https://huggingface.co/{pi['repo_id']}/resolve/{pi['revision']}/config.json",
        network_ledger,
    )
    checks.update(
        {
            "pi05_config_type_matches": pi_config.get("type") == "pi05",
            "pi05_config_camera_shapes_match": _feature_shape(
                pi_config, "observation.images.image"
            )
            == [3, 256, 256]
            and _feature_shape(pi_config, "observation.images.image2") == [3, 256, 256],
            "pi05_config_state_and_action_match": _feature_shape(
                pi_config, "observation.state"
            )
            == [8]
            and _output_shape(pi_config, "action") == [7],
            "pi05_config_normalization_matches": pi_config.get("normalization_mapping")
            == {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
            "pi05_config_visual_and_empty_camera_match": pi_config.get("image_resolution")
            == [224, 224]
            and pi_config.get("empty_cameras") == 1,
            "pi05_eval_override_is_ten_actions": pi["n_action_steps_override"] == 10,
        }
    )

    smol = routes["smolvla_existing_finetuned"]
    smol_config = _http_json(
        f"https://huggingface.co/{smol['repo_id']}/resolve/{smol['revision']}/config.json",
        network_ledger,
    )
    smol_processor = _http_json(
        f"https://huggingface.co/{smol['repo_id']}/resolve/{smol['revision']}/policy_preprocessor.json",
        network_ledger,
    )
    rename_map = smol_processor.get("steps", [{}])[0].get("config", {}).get("rename_map", {})
    smol_image_keys = sorted(
        key for key, value in smol_config.get("input_features", {}).items() if value.get("type") == "VISUAL"
    )
    checks.update(
        {
            "smolvla_detects_state_dimension_mismatch": _feature_shape(
                smol_config, "observation.state"
            )
            == [6]
            and smol["declared_state_dimension"] == 6,
            "smolvla_detects_three_camera_mismatch": smol_image_keys
            == sorted(smol["declared_image_keys"])
            and len(smol_image_keys) == 3,
            "smolvla_processor_only_renames_two_cameras": rename_map
            == {
                "observation.images.image": "observation.images.camera1",
                "observation.images.image2": "observation.images.camera2",
            }
            and "observation.images.camera3" not in rename_map.values(),
        }
    )

    dataset = dataset_route
    info = _http_json(
        f"https://huggingface.co/datasets/{dataset['repo_id']}/resolve/{dataset['revision']}/meta/info.json",
        network_ledger,
    )
    stats = _http_json(
        f"https://huggingface.co/datasets/{dataset['repo_id']}/resolve/{dataset['revision']}/meta/stats.json",
        network_ledger,
    )
    features = info.get("features", {})
    checks.update(
        {
            "dataset_counts_match": info.get("total_episodes") == dataset["episodes"]
            and info.get("total_frames") == dataset["frames"]
            and info.get("total_tasks") == dataset["tasks"],
            "dataset_camera_contract_matches": features.get("observation.images.image", {}).get(
                "shape"
            )
            == [256, 256, 3]
            and features.get("observation.images.image2", {}).get("shape") == [256, 256, 3],
            "dataset_state_action_contract_matches": features.get("observation.state", {}).get(
                "shape"
            )
            == [8]
            and features.get("action", {}).get("shape") == [7],
            "dataset_mean_std_stats_present": len(stats.get("observation.state", {}).get("mean", []))
            == 8
            and len(stats.get("observation.state", {}).get("std", [])) == 8
            and len(stats.get("action", {}).get("mean", [])) == 7
            and len(stats.get("action", {}).get("std", [])) == 7,
        }
    )
    return (
        {
            "repositories": repo_evidence,
            "pi05_config": pi_config,
            "smolvla_config": smol_config,
            "smolvla_rename_map": rename_map,
            "dataset_info": {
                "codebase_version": info.get("codebase_version"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "total_tasks": info.get("total_tasks"),
                "fps": info.get("fps"),
            },
            "network_ledger": network_ledger,
            "total_metadata_bytes_read": sum(item["bytes_read"] for item in network_ledger),
        },
        checks,
    )


def _runtime_checks(
    manifest: dict[str, Any], *, config_dir: Path, assets_dir: Path
) -> tuple[dict[str, Any], dict[str, bool]]:
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())
    from libero import libero as libero_runtime
    from libero.libero import benchmark, get_libero_path
    import torch
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS

    libero_runtime._assets_path_cache = str(assets_dir.resolve())
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = torch.load(init_path, weights_only=False)  # nosec B614 - trusted installed package
    expected = manifest["environment_contract"]
    expected_task = expected["task_0"]
    missing_assets = [name for name in REQUIRED_ASSET_SUBDIRS if not (assets_dir / name).is_dir()]
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("lerobot", "hf-libero", "transformers", "tokenizers")
    }
    free_bytes = shutil.disk_usage(assets_dir).free
    evidence = {
        "package_versions": package_versions,
        "task_count": len(suite.tasks),
        "task_0_name": task.name,
        "task_0_instruction": task.language,
        "bddl_path": str(bddl_path.resolve()),
        "bddl_size_bytes": bddl_path.stat().st_size,
        "bddl_sha256": _sha256_file(bddl_path),
        "init_state_path": str(init_path.resolve()),
        "init_state_size_bytes": init_path.stat().st_size,
        "init_state_sha256": _sha256_file(init_path),
        "init_state_count": len(init_states),
        "task_suite_max_steps": TASK_SUITE_MAX_STEPS["libero_spatial"],
        "asset_directory": str(assets_dir.resolve()),
        "missing_required_asset_subdirs": missing_assets,
        "filesystem_free_bytes": free_bytes,
    }
    checks = {
        "runtime_versions_match": package_versions
        == {
            "lerobot": manifest["software_contract"]["lerobot"],
            "hf-libero": manifest["software_contract"]["hf-libero"],
            "transformers": manifest["software_contract"]["transformers"],
            "tokenizers": manifest["software_contract"]["tokenizers"],
        },
        "runtime_suite_has_ten_tasks": len(suite.tasks) == expected["task_count"] == 10,
        "runtime_task0_identity_matches": task.name == expected_task["task_name"]
        and task.language == expected_task["task_instruction"],
        "runtime_bddl_hash_matches": evidence["bddl_sha256"] == expected_task["bddl_sha256"],
        "runtime_init_state_hash_and_count_match": evidence["init_state_sha256"]
        == expected_task["init_state_file_sha256"]
        and len(init_states) == expected_task["available_init_state_count"],
        "runtime_horizon_matches": TASK_SUITE_MAX_STEPS["libero_spatial"]
        == expected["max_episode_steps"],
        "runtime_asset_tree_complete": not missing_assets,
        "runtime_asset_path_matches": assets_dir.resolve() == Path(expected["asset_directory"]).resolve(),
    }
    return evidence, checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-hub", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    started = time.monotonic()
    report: dict[str, Any]
    try:
        manifest = _read_json(args.manifest)
        checks = _static_checks(manifest)
        hub_evidence = None
        runtime_evidence = None
        if args.check_hub:
            hub_evidence, hub_checks = _hub_checks(manifest)
            checks.update(hub_checks)
        if args.check_runtime:
            runtime_evidence, runtime_checks = _runtime_checks(
                manifest,
                config_dir=args.libero_config_dir,
                assets_dir=args.libero_assets_dir,
            )
            checks.update(runtime_checks)
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if all(checks.values()) else "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": _sha256_file(args.manifest),
                "schema_version": manifest.get("schema_version"),
                "status": manifest.get("status"),
            },
            "verification_scope": {
                "static": True,
                "hub_metadata": args.check_hub,
                "installed_runtime": args.check_runtime,
            },
            "checks": checks,
            "hub_evidence": hub_evidence,
            "runtime_evidence": runtime_evidence,
            "decision": manifest["decision"],
            "boundaries": {
                "large_blob_downloaded": False,
                "checkpoint_loaded": False,
                "dataset_loaded": False,
                "environment_constructed": False,
                "episode_executed": False,
                "training_started": False,
                "optimizer_steps": 0,
                "score_claim_allowed": False,
            },
        }
    except Exception as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "runtime_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines()[-50:],
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    false_checks = [name for name, passed in report.get("checks", {}).items() if not passed]
    print(
        f"status={report['status']} checks={len(report.get('checks', {}))} "
        f"false_checks={len(false_checks)} hub={args.check_hub} runtime={args.check_runtime} "
        f"large_blob_downloaded=false optimizer_steps=0 out={args.out}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
