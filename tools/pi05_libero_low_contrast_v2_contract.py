"""Strict lifecycle-only successor of the frozen low-contrast v1 contract.

No model, controller, noise or metric change is authorized. The parent files
remain byte-pinned. Runtime hash provenance is recorded by the runner; these
CPU validators establish completeness/equality, not the origin of its hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pi05_libero_low_contrast_contract as v1


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "libero_spatial_low_contrast_paired_v2"
_HASH = re.compile(r"^[0-9a-f]{64}$")
_CONDITIONS = ("clean", "low_contrast")
_IDENTITY_FIELDS = ("task_id", "seed", "init_state_index")
_POLICY_HASHES = ("init_state_sha256", "initial_observation_sha256", "first_inference_rng_sha256")
_RESET_HASHES = ("init_state_sha256", "initial_observation_sha256", "post_reset_rng_sha256")

EXPECTED = {
    "schema_version": 2,
    "protocol_id": PROTOCOL_ID,
    "evidence_scope": "public_benchmark_paired_diagnostic_lifecycle_repair_not_architecture_gain",
    "parent_protocol": {
        "path": "docs/libero-spatial-low-contrast-protocol-v1.json",
        "sha256": "11dbb573db0bab86e883863ec38f0721f695bb345ad037b845c2bbd37d06bf72",
        "inheritance": "all_parent_semantics_unchanged_except_explicit_lifecycle_and_stages_below",
    },
    "implementation_sources": {
        "tools/run_pi05_libero_low_contrast.py": "6a8dd7a1e891d4ef350967a93b9feebf26bcc3b8f408e38ba9188dc2a9156586",
        "tools/pi05_libero_low_contrast_contract.py": "5027b653fc0145ad641d10ff87f7a323f10ed94c45d3584475868a49f773e734",
    },
    "lifecycle": {
        "native_environment_per_condition": "new_original_single_task_synchronous_environment",
        "reuse_environment_between_conditions_or_tasks": False,
        "seed_python_numpy_torch_before_environment_construction": 1000,
        "seed_python_numpy_torch_before_official_rollout": 1000,
        "native_init_state_index": 0,
        "native_num_steps_wait": 10,
        "native_horizon": 280,
        "policy_episode_implementation": "unchanged_pinned_v1_episode",
        "close_environment_after_each_condition": True,
        "additional_state_forcing_allowed": False,
        "pair_hash_equality_may_be_relaxed": False,
        "reset_rng_hash_scope": "python_numpy_torch_cpu_and_all_cuda_rng_states_after_native_reset",
        "reset_observation_hash_scope": "complete_raw_post_settle_observation_before_corruption",
    },
    "stages": {
        "conditions": ["clean", "low_contrast"],
        "seed": 1000,
        "init_state_index": 0,
        "reset-check": {"task_ids": list(range(10)), "row_count": 20, "policy_loaded": False,
                        "policy_inference": False, "optimizer_steps": 0},
        "check": {"task_ids": [0, 4], "episode_count": 4, "full_episodes": True,
                  "purpose": "task0_and_task4_lifecycle_regression"},
        "screen": {"task_ids": list(range(10)), "episode_count": 20, "full_episodes": True,
                   "user_run_by_default": True},
        "separate_fresh_stage_output_directories": True,
        "reuse_earlier_stage_or_historical_episodes": False,
        "overwrite_or_resume_partial_results": False,
    },
    "claims": {
        "training": False, "model_selection": False, "architecture_gain": False,
        "robustness_certainty": False, "real_system_validated": False,
        "historical_b4b_reclassification": False,
        "parent_v1_results_reclassified_as_v2": False,
        "reset_checks_are_policy_performance_evidence": False,
    },
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate protocol key: {key}")
        result[key] = value
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Return the strict delta, checking frozen parent JSON/code on this host.

    Paths in the delta are repository-relative, not relative to a caller's
    temporary copy. Runtime dependency/checkpoint pins stay in the parent.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    v1._require_equal(data, EXPECTED, "v2")
    parent = data["parent_protocol"]
    if _sha256(ROOT / parent["path"]) != parent["sha256"]:
        raise ValueError("frozen parent protocol SHA256 differs")
    for source, digest in data["implementation_sources"].items():
        if _sha256(ROOT / source) != digest:
            raise ValueError(f"frozen parent implementation SHA256 differs: {source}")
    v1.load_protocol(ROOT / parent["path"])
    return data


def schedule(stage: str) -> list[dict[str, int]]:
    if stage not in {"reset-check", "check", "screen"}:
        raise ValueError("stage must be reset-check, check or screen")
    return [{"task_id": task, "seed": 1000, "init_state_index": 0}
            for task in EXPECTED["stages"][stage]["task_ids"]]


def _validate_pairs(rows, stage, hash_fields, *, policy_rows):
    selected = schedule(stage)
    expected = {tuple(row[key] for key in _IDENTITY_FIELDS) for row in selected}
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 2 * len(selected):
        raise ValueError("row count does not match complete paired schedule")
    grouped = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("row must be a mapping")
        if any(type(row.get(key)) is not int for key in _IDENTITY_FIELDS):
            raise ValueError("task, seed and init index must be integers")
        identity = tuple(row[key] for key in _IDENTITY_FIELDS)
        if identity not in expected:
            raise ValueError(f"row outside fixed schedule: {identity}")
        condition = row.get("condition")
        if type(condition) is not str or condition not in _CONDITIONS:
            raise ValueError("unknown row condition")
        for field in hash_fields:
            digest = row.get(field)
            if type(digest) is not str or _HASH.fullmatch(digest) is None:
                raise ValueError(f"invalid SHA256: {field}")
        if policy_rows:
            if type(row.get("success")) is not bool:
                raise ValueError("success must be a native boolean")
            if type(row.get("steps")) is not int or not 1 <= row["steps"] <= 280:
                raise ValueError("steps must be an integer inside native horizon")
        else:
            for field in ("success", "steps", "first_inference_rng_sha256"):
                if field in row:
                    raise ValueError(f"reset-only row must not report policy metric: {field}")
            for field in ("policy_loaded", "policy_inference"):
                if field in row and (type(row[field]) is not bool or row[field]):
                    raise ValueError(f"reset-only row contradicts no-policy scope: {field}")
            if "optimizer_steps" in row and (type(row["optimizer_steps"]) is not int or row["optimizer_steps"] != 0):
                raise ValueError("reset-only row contradicts zero optimizer steps")
            if "policy_steps" in row and (type(row["policy_steps"]) is not int or row["policy_steps"] != 0):
                raise ValueError("reset-only row contradicts zero policy steps")
        group = grouped.setdefault(identity, {})
        if condition in group:
            raise ValueError(f"duplicate condition: {identity}/{condition}")
        group[condition] = row
    if set(grouped) != expected:
        raise ValueError("missing scheduled pair")
    for identity, group in grouped.items():
        if set(group) != set(_CONDITIONS):
            raise ValueError(f"missing condition: {identity}")
        for field in hash_fields:
            if group["clean"][field] != group["low_contrast"][field]:
                raise ValueError(f"pair mismatch: {identity}/{field}")
    return selected, grouped


def summarize_pairs(episodes: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    """Same v1 metrics and strict row semantics, with v2 stage identities."""
    if stage not in {"check", "screen"}:
        raise ValueError("policy summary stage must be check or screen")
    selected, grouped = _validate_pairs(episodes, stage, _POLICY_HASHES, policy_rows=True)
    outcomes = {"both_success": 0, "clean_only": 0, "low_contrast_only": 0, "both_failure": 0}
    per_task = []
    clean_count = low_count = 0
    for identity in selected:
        key = tuple(identity[field] for field in _IDENTITY_FIELDS)
        clean, low = grouped[key]["clean"], grouped[key]["low_contrast"]
        clean_success, low_success = clean["success"], low["success"]
        clean_count += int(clean_success)
        low_count += int(low_success)
        outcome = ("both_success" if low_success else "clean_only") if clean_success else ("low_contrast_only" if low_success else "both_failure")
        outcomes[outcome] += 1
        per_task.append({**identity, "clean_success": clean_success, "low_contrast_success": low_success,
                         "clean_steps": clean["steps"], "low_contrast_steps": low["steps"],
                         "outcome": outcome, "delta_pp": 100 * (int(low_success) - int(clean_success)),
                         **{field: clean[field] for field in _POLICY_HASHES}})
    count = len(selected)
    return {"protocol_id": PROTOCOL_ID, "stage": stage, "paired_schedule_verified": True,
            "pair_count": count, "episode_count": 2 * count,
            "clean_success_count": clean_count, "low_contrast_success_count": low_count,
            "clean_success_rate": clean_count / count, "low_contrast_success_rate": low_count / count,
            "delta_pp": 100 * (low_count - clean_count) / count,
            "relative_drop": (clean_count - low_count) / clean_count if clean_count else None,
            "paired_outcomes": outcomes, "per_task": per_task,
            "statistical_significance_claim": False, "evidence_scope": EXPECTED["evidence_scope"]}


def validate_reset_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate twenty real reset records without creating policy score fields."""
    selected, grouped = _validate_pairs(rows, "reset-check", _RESET_HASHES, policy_rows=False)
    return {"protocol_id": PROTOCOL_ID, "stage": "reset-check", "paired_schedule_verified": True,
            "pair_count": len(selected), "row_count": len(rows), "policy_loaded": False,
            "policy_inference": False, "optimizer_steps": 0,
            "per_task": [{**identity, **{field: grouped[tuple(identity[key] for key in _IDENTITY_FIELDS)]["clean"][field]
                                        for field in _RESET_HASHES}} for identity in selected],
            "evidence_scope": "native_reset_pair_identity_only_not_policy_performance"}
