from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pi05_libero_b4a_result_audit_v1"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    if not root.is_dir():
        raise FileNotFoundError(f"asset root not found: {root}")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        file_hash = _sha256(path)
        row = {
            "path": relative.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_hash,
        }
        rows.append(row)
        digest.update(
            f"{row['path']}\t{row['size_bytes']}\t{row['sha256']}\n".encode("utf-8")
        )
    return rows, digest.hexdigest()


def _require_substrings(path: Path, required: dict[str, str]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "required_semantics": {
            name: snippet in text for name, snippet in required.items()
        },
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    protocol = _load_json(args.protocol)
    stage = protocol["evaluation_stages"]["b4a_task0_preflight_score"]
    route = protocol["checkpoint_routes"]["pi05_primary"]
    eval_info = _load_json(args.eval_info)
    checkpoint_audit = _load_json(args.checkpoint_audit)

    per_task = eval_info["per_task"]
    task_metrics = per_task[0]["metrics"] if len(per_task) == 1 else {}
    successes = task_metrics.get("successes", [])
    sum_rewards = task_metrics.get("sum_rewards", [])
    max_rewards = task_metrics.get("max_rewards", [])
    video_paths = task_metrics.get("video_paths", [])
    videos = [args.project_root / path for path in video_paths]
    video_records = [
        {
            "episode_index": index,
            "path": str(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256(path) if path.is_file() else None,
        }
        for index, path in enumerate(videos)
    ]

    expected_asset_rows, expected_asset_manifest = _payload_manifest(
        args.expected_assets
    )
    effective_asset_rows, effective_asset_manifest = _payload_manifest(
        args.effective_assets
    )

    import lerobot

    package_root = Path(lerobot.__file__).resolve().parent
    eval_source = _require_substrings(
        package_root / "scripts" / "lerobot_eval.py",
        {
            "seed_start_uses_config_seed": "start_seed=cfg.seed",
            "seed_batch_start_increments_by_env_count": (
                "start_seed + (batch_ix * env.num_envs)"
            ),
            "seed_batch_end_increments_by_env_count": (
                "start_seed + ((batch_ix + 1) * env.num_envs)"
            ),
        },
    )
    libero_source = _require_substrings(
        package_root / "envs" / "libero.py",
        {
            "initial_state_starts_at_episode_index": (
                "self.init_state_id = self.episode_index"
            ),
            "initial_state_advances_by_env_count": (
                "self.init_state_id += self._reset_stride"
            ),
            "reset_stride_equals_env_count": "self._reset_stride = n_envs",
        },
    )

    seeds = list(range(args.seed, args.seed + args.episodes))
    init_state_indices = list(range(args.episodes))
    episode_records = [
        {
            "episode_index": index,
            "seed": seeds[index],
            "init_state_index": init_state_indices[index],
            "success": successes[index],
            "sum_reward": sum_rewards[index],
            "max_reward": max_rewards[index],
            "video": video_records[index],
        }
        for index in range(min(len(successes), len(video_records)))
    ]

    overall = eval_info["overall"]
    checks = {
        "protocol_scope_is_public_libero_only": protocol["scope"]
        == "public_libero_benchmark_only",
        "stage_is_task0_preflight": stage["task_ids"] == [0]
        and stage["purpose"] == "score_pipeline_preflight_not_official_suite_average",
        "executed_task_is_only_task0": len(per_task) == 1
        and per_task[0]["task_group"] == "libero_spatial"
        and per_task[0]["task_id"] == 0,
        "episode_count_matches_protocol": args.episodes
        == stage["total_episodes"]
        == overall["n_episodes"]
        == len(successes),
        "seed_schedule_matches_protocol": seeds == stage["episode_seeds"],
        "init_state_schedule_matches_protocol": init_state_indices
        == stage["init_state_indices"],
        "batch_size_matches_protocol": args.batch_size == stage["batch_size"],
        "max_parallel_tasks_matches_protocol": args.max_parallel_tasks
        == stage["max_parallel_tasks"],
        "episode_length_matches_protocol": args.episode_length
        == stage["max_episode_steps"],
        "n_action_steps_matches_checkpoint_route": args.n_action_steps
        == route["n_action_steps_override"],
        "all_ten_episodes_succeeded": len(successes) == 10
        and all(value is True for value in successes),
        "success_rate_is_100_percent": overall["pc_success"] == 100.0,
        "reward_vectors_are_complete": len(sum_rewards) == args.episodes
        and len(max_rewards) == args.episodes,
        "ten_nonempty_videos_exist": len(video_records) == args.episodes
        and all(
            item["size_bytes"] is not None and item["size_bytes"] > 0
            for item in video_records
        ),
        "video_hashes_are_unique": len(
            {item["sha256"] for item in video_records}
        )
        == args.episodes,
        "checkpoint_audit_passed": checkpoint_audit["status"] == "passed",
        "checkpoint_hash_matches_protocol": checkpoint_audit["checkpoint"][
            "model_sha256"
        ]
        == route["model_sha256"],
        "checkpoint_unique_parameter_coverage_one": checkpoint_audit[
            "verified_load"
        ]["loaded_parameter_fraction"]
        == 1.0,
        "checkpoint_has_no_effective_load_failures": checkpoint_audit[
            "verified_load"
        ]["missing_key_count"]
        == 0
        and checkpoint_audit["verified_load"]["unexpected_key_count"] == 0
        and checkpoint_audit["verified_load"]["mismatched_shape_count"] == 0,
        "lerobot_version_matches_protocol": importlib.metadata.version("lerobot")
        == protocol["software_contract"]["lerobot"],
        "seed_source_semantics_verified": all(
            eval_source["required_semantics"].values()
        ),
        "init_state_source_semantics_verified": all(
            libero_source["required_semantics"].values()
        ),
        "asset_payload_file_counts_match": len(expected_asset_rows)
        == len(effective_asset_rows)
        and len(expected_asset_rows) > 0,
        "asset_payload_manifests_match": expected_asset_rows == effective_asset_rows
        and expected_asset_manifest == effective_asset_manifest,
        "visual_sheet_exists": args.visual_sheet.is_file()
        and args.visual_sheet.stat().st_size > 0,
        "full_suite_not_claimed": stage["published_aggregate_comparison_allowed"]
        is False,
        "no_noise_or_training_in_command": args.noise_executed is False
        and args.training_executed is False
        and args.optimizer_steps == 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "runtime_seconds": round(time.monotonic() - started, 3),
        "scope": "libero_spatial_task0_gate_b4a_clean_score_preflight_only",
        "executed_configuration": {
            "checkpoint": str(args.checkpoint),
            "task_group": "libero_spatial",
            "task_ids": [0],
            "episodes": args.episodes,
            "seed": args.seed,
            "derived_episode_seeds": seeds,
            "derived_init_state_indices": init_state_indices,
            "batch_size": args.batch_size,
            "max_parallel_tasks": args.max_parallel_tasks,
            "episode_length": args.episode_length,
            "control_mode": "relative",
            "init_states": True,
            "n_action_steps": args.n_action_steps,
        },
        "metrics": {
            "success_count": sum(bool(value) for value in successes),
            "episode_count": overall["n_episodes"],
            "task0_success_rate_percent": overall["pc_success"],
            "avg_sum_reward": overall["avg_sum_reward"],
            "avg_max_reward": overall["avg_max_reward"],
            "eval_seconds": overall["eval_s"],
            "eval_seconds_per_episode": overall["eval_ep_s"],
            "official_ten_task_suite_macro_average_available": False,
        },
        "episodes": episode_records,
        "checkpoint_evidence": {
            "audit_path": str(args.checkpoint_audit),
            "audit_sha256": _sha256(args.checkpoint_audit),
            "model_sha256": checkpoint_audit["checkpoint"]["model_sha256"],
            "unique_parameter_coverage": checkpoint_audit["verified_load"][
                "loaded_parameter_fraction"
            ],
        },
        "runtime_source_evidence": {
            "lerobot_version": importlib.metadata.version("lerobot"),
            "eval": eval_source,
            "libero_environment": libero_source,
        },
        "asset_evidence": {
            "configured_path": str(args.expected_assets),
            "effective_upstream_path": str(args.effective_assets),
            "effective_path_matches_configured_path": args.expected_assets.resolve()
            == args.effective_assets.resolve(),
            "metadata_cache_excluded": True,
            "payload_file_count": len(expected_asset_rows),
            "payload_manifest_sha256": expected_asset_manifest,
            "payloads_identical": expected_asset_rows == effective_asset_rows,
        },
        "visual_evidence": {
            "sheet": str(args.visual_sheet),
            "sheet_sha256": _sha256(args.visual_sheet),
            "agent_viewed": True,
            "visual_status": "viewed_not_accepted",
        },
        "eval_info": {
            "path": str(args.eval_info),
            "sha256": _sha256(args.eval_info),
        },
        "checks": checks,
        "boundaries": {
            "gate_b4a_executed": True,
            "clean_evaluation": True,
            "task0_preflight_score_reportable": True,
            "full_ten_task_suite_executed": False,
            "published_suite_macro_average_claim_allowed": False,
            "noise_evaluation_executed": args.noise_executed,
            "training_executed": args.training_executed,
            "optimizer_steps": args.optimizer_steps,
            "guidewire_or_real_system_claim_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of the clean PI0.5 LIBERO B4a task-0 result."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/libero-spatial-score-protocol-v1.json"),
    )
    parser.add_argument(
        "--eval-info",
        type=Path,
        default=Path("simulation_output/pi05_libero_task0_b4a_v1/eval_info.json"),
    )
    parser.add_argument(
        "--checkpoint-audit",
        type=Path,
        default=Path(
            "simulation_output/pi05_libero_finetuned_checkpoint_audit_v1.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/zsw/models/project_2026/pi05_libero_finetuned_8e174154"),
    )
    parser.add_argument(
        "--expected-assets",
        type=Path,
        default=Path("simulation_output/libero_assets_v1"),
    )
    parser.add_argument(
        "--effective-assets",
        type=Path,
        default=Path("/home/zsw/.cache/libero/assets"),
    )
    parser.add_argument(
        "--visual-sheet",
        type=Path,
        default=Path(
            "simulation_output/pi05_libero_task0_b4a_v1/visual_audit/"
            "b4a_final_frames_sheet_zh.png"
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-parallel-tasks", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=280)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--optimizer-steps", type=int, default=0)
    parser.add_argument("--noise-executed", action="store_true")
    parser.add_argument("--training-executed", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = _run(args)
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
        f"status={report['status']} checks={len(checks)} "
        f"false_checks={len(false_checks)} "
        f"task0_success_rate={report.get('metrics', {}).get('task0_success_rate_percent')} "
        f"full_suite={report.get('boundaries', {}).get('full_ten_task_suite_executed')} "
        f"out={args.out}",
        flush=True,
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
