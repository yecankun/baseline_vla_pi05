"""Frozen native-LIBERO paired diagnostic contract; no simulator or model import.

This is a new protocol, not an amendment of the historical B4b score protocol.
Raw finite action overshoot is reported, while unmodified native controller
saturation is allowed. Additional action-wrapper clipping is never allowed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROTOCOL_ID = "libero_spatial_low_contrast_paired_v1"
CONDITIONS = ("clean", "low_contrast")
HISTORICAL_PROTOCOL_SHA256 = "b72b854d04b05930c2c1b35f384938a875e144002963f08771a2e6fc649dbdd1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAIR_HASHES = (
    "init_state_sha256",
    "initial_observation_sha256",
    "first_inference_rng_sha256",
)

# Exact critical values prevent an edited JSON file silently broadening this
# diagnostic or falling back to a historical clean score.
_CRITICAL = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "evidence_scope": "public_benchmark_paired_diagnostic_not_architecture_gain",
    "benchmark": {
        "suite": "libero_spatial", "native_state_dim": 8, "native_action_dim": 7,
        "native_horizon": 280, "control_mode": "relative", "init_states": True,
        "num_steps_wait": 10, "batch_size": 1, "use_async_envs": False,
        "max_parallel_tasks": 1, "raw_image_paths": ["pixels.image", "pixels.image2"],
        "raw_image_shape_hwc": [256, 256, 3], "raw_image_dtype": "uint8",
        "policy_image_shape_chw": [3, 256, 256], "internal_image_hw": [224, 224],
    },
    "schedule": {
        "conditions": list(CONDITIONS), "seed": 1000, "init_state_index": 0,
        "check_task_ids": [0], "screen_task_ids": list(range(10)),
        "check_episode_count": 2, "screen_episode_count": 20,
        "full_episodes": True, "reuse_check_episodes_for_screen": False,
        "reuse_historical_clean_scores": False,
    },
    "policy": {
        "chunk_size": 50, "n_action_steps": 10, "num_inference_steps": 10,
        "dtype": "bfloat16", "compile_model": True, "compile_mode": "max-autotune",
        "empty_cameras": 1, "n_obs_steps": 1, "optimizer_steps": 0,
        "checkpoint_or_architecture_changes": False,
    },
    "corruption": {
        "name": "low_contrast", "severity": 2, "alpha": 0.45, "seed": 20260911,
        "image_paths": ["pixels.image", "pixels.image2"],
        "insertion_boundary": "raw_observation_before_policy_preprocessing",
        "formula": "per_channel_spatial_mean + alpha * (pixel - per_channel_spatial_mean)",
        "quantization": "float32_unit_interval_then_numpy_rint_uint8",
        "independent_rng": True, "geometry_state_task_reward_termination_unchanged": True,
        "evaluation_only": True,
    },
    "controller": {
        "raw_action_reference_bounds": [-1.0, 1.0],
        "finite_native_7d_required": True, "raw_overshoot_is_strict_failure": False,
        "raw_overshoot_reporting": "per_episode_per_dimension_count_and_extrema",
        "additional_wrapper_clipping_allowed": False,
        "native_osc_clip_and_scale_retained": True,
        "native_gripper_sign_and_internal_clip_retained": True,
        "native_methods_must_remain_unchanged": True,
    },
    "pairing": {
        "policy_queue_reset_each_arm": True, "global_rng_seed_before_each_rollout": 1000,
        "actual_init_payload_hash_required": True,
        "initial_observation_hash_scope": "complete_raw_post_settle_observation_before_corruption",
        "first_inference_rng_hash_scope": "python_numpy_torch_cpu_and_all_cuda_rng_states",
        "equal_initial_observation_required": True, "equal_first_inference_rng_required": True,
        "action_sequences_may_diverge": True,
    },
    "execution": {
        "check_expected_minutes_max": 5, "screen_expected_minutes_range": [5, 10],
        "screen_user_run_by_default": True, "separate_fresh_stage_output_directories": True,
        "overwrite_or_resume_partial_results": False, "source_hash_drift_is_failure": True,
    },
    "reporting": {
        "success_source": "native_environment_success",
        "delta_pp": "100 * (low_contrast_success_rate - clean_success_rate)",
        "relative_drop": "(clean_success_rate - low_contrast_success_rate) / clean_success_rate",
        "relative_drop_when_clean_zero": None,
        "paired_outcomes": ["both_success", "clean_only", "low_contrast_only", "both_failure"],
        "per_task_required": True, "statistical_significance_claim": False,
    },
    "claims": {
        "training": False, "model_selection": False, "architecture_gain": False,
        "robustness_certainty": False, "real_system_validated": False,
        "guidewire_score": False, "historical_b4b_reclassification": False,
    },
}


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    """Type-aware equality: JSON true must not pass as integer 1."""
    if type(actual) is not type(expected):
        raise ValueError(f"protocol type mismatch: {path}")
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"protocol keys mismatch: {path}")
        for key, value in expected.items():
            _require_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"protocol list length mismatch: {path}")
        for index, value in enumerate(expected):
            _require_equal(actual[index], value, f"{path}[{index}]")
    elif actual != expected:
        raise ValueError(f"protocol value mismatch: {path}")


def _validate_hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid SHA256: {field}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load strict fixed semantics and complete resource pins (not execute them)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != set(_CRITICAL) | {"provenance"}:
        raise ValueError("invalid protocol top-level keys")
    for key, expected in _CRITICAL.items():
        _require_equal(data[key], expected, key)
    provenance = data["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    expected_keys = {"historical_protocol", "source_files", "checkpoint", "runtime_sources", "versions", "boundary_evidence"}
    if set(provenance) != expected_keys:
        raise ValueError("invalid provenance keys")
    _require_equal(provenance["historical_protocol"], {
        "path": "docs/libero-spatial-score-protocol-v1.json",
        "sha256": HISTORICAL_PROTOCOL_SHA256,
    }, "provenance.historical_protocol")
    expected_sources = {"tools/run_pi05_libero_b4b.py", "tools/vla_benchmark_contract.py", "tools/vla_visual_noise_wrapper.py"}
    if not isinstance(provenance["source_files"], dict) or set(provenance["source_files"]) != expected_sources:
        raise ValueError("source_files must pin the old runner and both corruption modules")
    checkpoint = provenance["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"logical_path", "resolved_path", "sha256_by_file"}:
        raise ValueError("invalid checkpoint pins")
    _require_equal(checkpoint["logical_path"], "/home/zsw/models/project_2026/pi05_libero_finetuned_8e174154", "checkpoint.logical_path")
    _require_equal(checkpoint["resolved_path"], "/media/zsw/SSD1T/project_2026_weights_v1/models/pi05_libero_finetuned_8e174154", "checkpoint.resolved_path")
    expected_checkpoint_files = {
        "model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    if not isinstance(checkpoint["sha256_by_file"], dict) or set(checkpoint["sha256_by_file"]) != expected_checkpoint_files:
        raise ValueError("all six checkpoint files must be pinned")
    runtime = provenance["runtime_sources"]
    prefix = "/home/zsw/miniconda3/envs/project2026-pi/lib/python3.10/site-packages/"
    expected_runtime = {prefix + item for item in (
        "lerobot/envs/libero.py", "lerobot/policies/pi05/modeling_pi05.py", "lerobot/scripts/lerobot_eval.py",
        "robosuite/controllers/base_controller.py", "robosuite/controllers/osc.py",
        "robosuite/models/grippers/panda_gripper.py", "robosuite/robots/single_arm.py",
    )}
    if not isinstance(runtime, dict) or set(runtime) != expected_runtime:
        raise ValueError("all seven runtime/controller sources must be pinned")
    for group_name, group in (("source_files", provenance["source_files"]), ("checkpoint", checkpoint["sha256_by_file"]), ("runtime_sources", runtime)):
        for filename, digest in group.items():
            _validate_hash(digest, f"{group_name}.{filename}")
    _require_equal(provenance["versions"], {
        "lerobot": "0.4.4", "hf-libero": "0.1.4", "transformers": "4.53.2", "tokenizers": "0.21.4",
    }, "provenance.versions")
    boundary = provenance["boundary_evidence"]
    if not isinstance(boundary, dict) or set(boundary) != {"manifest_path", "manifest_sha256", "effective_policy_config_path", "effective_policy_config_sha256"}:
        raise ValueError("invalid boundary evidence pins")
    for name in ("manifest", "effective_policy_config"):
        suffix = "run_manifest.json" if name == "manifest" else "effective_policy_config.json"
        _require_equal(boundary[f"{name}_path"], f"simulation_output/pi05_libero_action_boundary_v1/{suffix}", f"boundary.{name}_path")
        _validate_hash(boundary[f"{name}_sha256"], f"boundary.{name}_sha256")
    return data


def schedule(stage: str) -> list[dict[str, int]]:
    """Return pair identities only; callers run clean and low_contrast per pair."""
    if stage not in {"check", "screen"}:
        raise ValueError("stage must be check or screen")
    return [{"task_id": task, "seed": 1000, "init_state_index": 0} for task in (range(1) if stage == "check" else range(10))]


def summarize_pairs(episodes: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    """Fail closed on incomplete, duplicated or nonpaired recorded executions.

    The hashes must be captured at actual runtime boundaries, not inferred from
    the schedule. This CPU function verifies their agreement, not their origin.
    """
    selected = schedule(stage)
    expected = {(row["task_id"], row["seed"], row["init_state_index"]) for row in selected}
    if len(episodes) != 2 * len(expected):
        raise ValueError("episode count does not match complete paired schedule")
    grouped: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = {}
    for row in episodes:
        if not isinstance(row, Mapping):
            raise ValueError("episode must be a mapping")
        if any(type(row.get(field)) is not int for field in ("task_id", "seed", "init_state_index", "steps")):
            raise ValueError("task, seed, init index and steps must be integers")
        key = (row["task_id"], row["seed"], row["init_state_index"])
        if key not in expected:
            raise ValueError(f"episode outside fixed schedule: {key}")
        condition = row.get("condition")
        if condition not in CONDITIONS:
            raise ValueError("unknown episode condition")
        if type(row.get("success")) is not bool:
            raise ValueError("success must be a native boolean")
        if not 1 <= row["steps"] <= 280:
            raise ValueError("steps outside native episode horizon")
        for field in _PAIR_HASHES:
            _validate_hash(row.get(field), field)
        group = grouped.setdefault(key, {})
        if condition in group:
            raise ValueError(f"duplicate episode condition: {key}/{condition}")
        group[condition] = row
    if set(grouped) != expected:
        raise ValueError("missing scheduled pair")
    outcomes = {"both_success": 0, "clean_only": 0, "low_contrast_only": 0, "both_failure": 0}
    per_task = []
    clean_count = low_count = 0
    for identity in selected:
        key = (identity["task_id"], identity["seed"], identity["init_state_index"])
        group = grouped[key]
        if set(group) != set(CONDITIONS):
            raise ValueError(f"incomplete conditions: {key}")
        clean, low = group["clean"], group["low_contrast"]
        for field in _PAIR_HASHES:
            if clean[field] != low[field]:
                raise ValueError(f"pair mismatch: {key}/{field}")
        clean_success, low_success = clean["success"], low["success"]
        clean_count += int(clean_success)
        low_count += int(low_success)
        outcome = ("both_success" if low_success else "clean_only") if clean_success else ("low_contrast_only" if low_success else "both_failure")
        outcomes[outcome] += 1
        per_task.append({**identity, "clean_success": clean_success, "low_contrast_success": low_success,
                         "clean_steps": clean["steps"], "low_contrast_steps": low["steps"],
                         "outcome": outcome, "delta_pp": 100 * (int(low_success) - int(clean_success)),
                         **{field: clean[field] for field in _PAIR_HASHES}})
    count = len(selected)
    return {
        "protocol_id": PROTOCOL_ID, "stage": stage, "paired_schedule_verified": True,
        "pair_count": count, "episode_count": 2 * count,
        "clean_success_count": clean_count, "low_contrast_success_count": low_count,
        "clean_success_rate": clean_count / count, "low_contrast_success_rate": low_count / count,
        "delta_pp": 100 * (low_count - clean_count) / count,
        "relative_drop": (clean_count - low_count) / clean_count if clean_count else None,
        "paired_outcomes": outcomes, "per_task": per_task,
        "statistical_significance_claim": False, "evidence_scope": _CRITICAL["evidence_scope"],
    }
