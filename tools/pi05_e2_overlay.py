from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from probe_pi05_lerobot_adapter import PI05ProbeDataset, collate_probe_batch


PACK_SCHEMA = "project2026_e2_candidate_pack_v1"
POLICY_SCHEMA = "project2026_e2_policy_sample_v1"
VALIDATION_SCHEMA = "project2026_e2_candidate_pack_validation_v1"
ALLOWED_ROLES = {
    "feed_request",
    "recovery_response",
    "clear_response",
    "observe_effect",
    "execution_hold",
}

CONTROLLER_EVENT_STATUS_VALUES = ("idle", "delayed", "executing", "completed")
CONTROLLER_EVENT_STATE_LAYOUT = {
    "piper_busy": (16, 17),
    "piper_cooldown": (17, 18),
    "piper_request_accepted": (18, 19),
    "piper_executed_feed": (19, 20),
    "piper_event_age_steps_saturating": (20, 21),
    "piper_event_status_one_hot": (21, 25),
    "piper_event_state_valid": (25, 26),
    "piper_event_age_steps_valid": (26, 27),
}
CONTROLLER_EVENT_STATE_START = 16
CONTROLLER_EVENT_STATE_END = 27
CONTROLLER_POLICY_FIELDS = (
    "piper_busy",
    "piper_cooldown",
    "piper_request_accepted",
    "piper_event_status",
    "piper_event_age_steps",
    "piper_executed_feed",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _resolve_project_path(project_root: Path, value: str) -> Path:
    root = project_root.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"overlay path escapes project root: {value}") from exc
    return path


def _image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    vector = [float(item) for item in value]
    if not np.isfinite(np.asarray(vector, dtype=np.float64)).all():
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _encode_controller_event_state(controller_state: Any, label: str) -> torch.Tensor:
    """Encode observable controller state into the reserved state_32 tail.

    A wholly absent/empty controller state maps to zeros for compatibility with
    the existing base dataset. Once controller event state is supplied, the
    non-age fields are required so false values are not confused with omission.
    """
    encoded = torch.zeros(
        CONTROLLER_EVENT_STATE_END - CONTROLLER_EVENT_STATE_START,
        dtype=torch.float32,
    )
    if controller_state is None or controller_state == {}:
        return encoded
    if not isinstance(controller_state, dict):
        raise ValueError(f"{label} must be an object when present")

    required = (
        "piper_busy",
        "piper_cooldown",
        "piper_request_accepted",
        "piper_event_status",
        "piper_executed_feed",
    )
    missing = [key for key in required if key not in controller_state]
    if missing:
        raise ValueError(f"{label} is partially present; missing {missing}")

    for offset, key in enumerate(
        ("piper_busy", "piper_cooldown", "piper_request_accepted")
    ):
        value = controller_state[key]
        if not isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be boolean")
        encoded[offset] = float(value)

    executed_feed = float(controller_state["piper_executed_feed"])
    if not np.isfinite(executed_feed) or not 0.0 <= executed_feed <= 1.0:
        raise ValueError(f"{label}.piper_executed_feed must be finite in [0, 1]")
    encoded[3] = executed_feed

    age = controller_state.get("piper_event_age_steps")
    if age is not None:
        age_value = float(age)
        if not np.isfinite(age_value) or age_value < 0.0:
            raise ValueError(f"{label}.piper_event_age_steps must be null or non-negative")
        encoded[4] = age_value / (1.0 + age_value)
        encoded[10] = 1.0

    status = str(controller_state["piper_event_status"])
    if status not in CONTROLLER_EVENT_STATUS_VALUES:
        raise ValueError(
            f"{label}.piper_event_status={status!r} is not one of "
            f"{CONTROLLER_EVENT_STATUS_VALUES}"
        )
    encoded[5 + CONTROLLER_EVENT_STATUS_VALUES.index(status)] = 1.0
    encoded[9] = 1.0
    return encoded


def _nested_key_paths(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            paths.append((key_text, path))
            paths.extend(_nested_key_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_nested_key_paths(child, f"{prefix}[{index}]"))
    return paths


class E2OverlayDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        weights: list[dict[str, Any]],
        project_root: Path,
        base_state_mean: torch.Tensor,
        image_size: int,
    ) -> None:
        if base_state_mean.numel() != 32:
            raise ValueError(f"base_state_mean must be 32D, got {base_state_mean.numel()}")
        self.rows = rows
        self.weight_rows = weights
        self.project_root = project_root.resolve()
        self.base_state_mean = base_state_mean.detach().float().cpu().reshape(32)
        self.image_size = int(image_size)
        self.training_weights = [float(row["training_weight"]) for row in weights]
        self.sample_roles = [str(row["sample_role"]) for row in weights]
        self.sample_ids = [str(row["sample_id"]) for row in rows]
        episodes = sorted({str(row["episode_instance_id"]) for row in rows})
        self.episode_indices = {episode: 100000 + idx for idx, episode in enumerate(episodes)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        state_row = row["state"]
        action_row = row["action"]
        task = str(row["task"])
        state = self.base_state_mean.clone()
        state[:6] = torch.tensor(
            _finite_vector(state_row["elite_tcp_pose_6d"], 6, "state.elite_tcp_pose_6d"),
            dtype=torch.float32,
        )
        state[14:16] = torch.tensor(
            [1.0, 0.0] if task == "left" else [0.0, 1.0], dtype=torch.float32
        )
        state[CONTROLLER_EVENT_STATE_START:CONTROLLER_EVENT_STATE_END] = (
            _encode_controller_event_state(
                state_row.get("controller_state"),
                "state.controller_state",
            )
        )
        elite_delta = torch.tensor(
            _finite_vector(action_row["elite_tcp_delta_6d"], 6, "action.elite_tcp_delta_6d"),
            dtype=torch.float32,
        )
        piper_id = int(action_row["piper_intent_id"])
        if piper_id not in {1, 2}:
            raise ValueError(f"overlay supports hold/feed only, got piper_intent_id={piper_id}")
        action = torch.zeros(32, dtype=torch.float32)
        action[:6] = elite_delta
        action[6 + piper_id] = 1.0
        images = row["images"]
        side_path = _resolve_project_path(self.project_root, str(images["side"]))
        top_path = _resolve_project_path(self.project_root, str(images["top"]))
        return {
            "observation.images.side": _image_tensor(side_path, self.image_size),
            "observation.images.top": _image_tensor(top_path, self.image_size),
            "observation.state": state,
            "action": action.unsqueeze(0),
            "task": str(row["instruction"]),
            "elite_tcp_delta_6d": elite_delta,
            "piper_intent_id": torch.tensor(piper_id, dtype=torch.long),
            "episode_index": torch.tensor(
                self.episode_indices[str(row["episode_instance_id"])], dtype=torch.long
            ),
            "frame_index": torch.tensor(int(row["step"]), dtype=torch.long),
        }


def load_e2_overlay(
    pack: Path,
    *,
    project_root: Path,
    base_state_mean: torch.Tensor,
    image_size: int,
    allow_diagnostic_overlay: bool,
) -> tuple[E2OverlayDataset, dict[str, Any]]:
    if not allow_diagnostic_overlay:
        raise ValueError("E2 pack is diagnostic-only; pass --allow-diagnostic-overlay explicitly")
    pack_dir = pack if pack.is_dir() else pack.parent
    manifest = _read_json(pack_dir / "manifest.json")
    validation = _read_json(pack_dir / "validation.json")
    family_split = _read_json(pack_dir / "family_split.json")
    if manifest.get("schema") != PACK_SCHEMA:
        raise ValueError(f"unexpected E2 pack schema: {manifest.get('schema')!r}")
    readiness = manifest.get("readiness", {})
    if readiness.get("diagnostic_overlay_allowed") is not True:
        raise ValueError("manifest does not allow diagnostic overlay")
    if readiness.get("policy_training_ready") is not False or readiness.get("formal_data_allowed") is not False:
        raise ValueError("E2 overlay readiness flags changed; re-audit before use")
    if validation.get("schema") != VALIDATION_SCHEMA or validation.get("passed") is not True:
        raise ValueError("E2 validation report did not pass")
    if any(check.get("passed") is not True for check in validation.get("checks", [])):
        raise ValueError("E2 validation contains a failed check")
    if family_split.get("standalone_validation_available") is not False:
        raise ValueError("E2 overlay unexpectedly declares standalone validation")
    if family_split.get("validation_episodes"):
        raise ValueError("E2 overlay validation episodes must remain empty")
    if set(family_split.get("episode_to_partition", {}).values()) != {"overlay_train"}:
        raise ValueError("all E2 episodes must remain overlay_train")

    rows = _read_jsonl(pack_dir / str(manifest["outputs"]["policy_samples"]))
    weight_rows = _read_jsonl(pack_dir / str(manifest["outputs"]["training_weights"]))
    expected = int(manifest.get("counts", {}).get("records", -1))
    if len(rows) != expected or len(weight_rows) != expected:
        raise ValueError(f"E2 row count mismatch: policy={len(rows)} weights={len(weight_rows)} expected={expected}")
    row_ids = [str(row.get("sample_id")) for row in rows]
    weight_ids = [str(row.get("sample_id")) for row in weight_rows]
    if row_ids != weight_ids or len(set(row_ids)) != expected:
        raise ValueError("E2 policy/weight rows are not one-to-one and ordered")
    contract = manifest.get("policy_observation_contract", {})
    field_provenance = contract.get("field_provenance", {})
    for field in CONTROLLER_POLICY_FIELDS:
        field_path = f"state.controller_state.{field}"
        metadata = field_provenance.get(field_path, {})
        if metadata.get("policy_input_allowed") is not True:
            raise ValueError(f"E2 manifest does not allow policy input field {field_path}")
    forbidden = {str(value) for value in contract.get("forbidden_policy_fields", [])}
    if not forbidden:
        raise ValueError("E2 manifest must declare forbidden_policy_fields")
    diagnostic_truth_name = str(contract.get("diagnostic_truth_file", ""))
    if diagnostic_truth_name != str(manifest["outputs"]["diagnostic_targets"]):
        raise ValueError("E2 diagnostic target declaration mismatch")
    for row, weight_row in zip(rows, weight_rows, strict=True):
        if row.get("schema") != POLICY_SCHEMA:
            raise ValueError(f"unexpected policy row schema for {row.get('sample_id')}")
        sample_id = str(row.get("sample_id"))
        task = row.get("task")
        if task not in {"left", "right"}:
            raise ValueError(f"unexpected task for {sample_id}: {task!r}")
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ValueError(f"missing task instruction for {sample_id}")
        images = row.get("images")
        if not isinstance(images, dict) or not all(isinstance(images.get(key), str) for key in ("side", "top")):
            raise ValueError(f"missing side/top image path for {sample_id}")
        state_row = row.get("state")
        action_row = row.get("action")
        if not isinstance(state_row, dict) or not isinstance(action_row, dict):
            raise ValueError(f"missing state/action object for {sample_id}")
        _finite_vector(state_row.get("elite_tcp_pose_6d"), 6, f"{sample_id}.state.elite_tcp_pose_6d")
        _encode_controller_event_state(
            state_row.get("controller_state"),
            f"{sample_id}.state.controller_state",
        )
        _finite_vector(action_row.get("elite_tcp_delta_6d"), 6, f"{sample_id}.action.elite_tcp_delta_6d")
        if int(action_row.get("piper_intent_id", -1)) not in {1, 2}:
            raise ValueError(f"E2 supports hold/feed only for {sample_id}")
        if row.get("provenance", {}).get("diagnostic_truth_in_policy_state") is not False:
            raise ValueError(f"diagnostic truth marker rejected for {sample_id}")
        nested_keys = _nested_key_paths(row)
        violations = sorted(path for key, path in nested_keys if key in forbidden or path in forbidden)
        if violations:
            raise ValueError(f"forbidden policy field for {sample_id}: {violations[:5]}")
        provenance = row.get("provenance", {})
        aligned_values = {
            "episode_instance_id": row.get("episode_instance_id"),
            "source_trajectory_id": provenance.get("source_trajectory_id"),
            "scenario_family_id": provenance.get("scenario_family_id"),
        }
        for key, expected_value in aligned_values.items():
            if weight_row.get(key) != expected_value:
                raise ValueError(
                    f"E2 weight sidecar {key} mismatch for {sample_id}: "
                    f"{weight_row.get(key)!r} != {expected_value!r}"
                )
        role = str(weight_row.get("sample_role"))
        if role not in ALLOWED_ROLES:
            raise ValueError(f"unexpected offline role: {role!r}")
        if weight_row.get("policy_input_allowed") is not False:
            raise ValueError(f"weight sidecar must be policy-disabled for {row.get('sample_id')}")
        value = float(weight_row.get("training_weight"))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid training weight for {sample_id}: {value}")

    row_episodes = {str(row["episode_instance_id"]) for row in rows}
    split_episodes = set(family_split.get("episode_to_partition", {}))
    if row_episodes != split_episodes:
        raise ValueError("E2 policy rows and family split do not declare the same episodes")

    dataset = E2OverlayDataset(
        rows=rows,
        weights=weight_rows,
        project_root=project_root,
        base_state_mean=base_state_mean,
        image_size=image_size,
    )
    role_counts = Counter(dataset.sample_roles)
    role_weight_sums: dict[str, float] = defaultdict(float)
    for role, weight in zip(dataset.sample_roles, dataset.training_weights, strict=True):
        role_weight_sums[role] += weight
    report = {
        "enabled": True,
        "schema": PACK_SCHEMA,
        "pack": str(pack),
        "records": len(dataset),
        "episodes": len(dataset.episode_indices),
        "effective_source_trajectories": int(manifest["counts"]["effective_source_trajectories"]),
        "scenario_families": int(manifest["counts"]["scenario_families"]),
        "weight_sum": float(sum(dataset.training_weights)),
        "weight_min": float(min(dataset.training_weights)),
        "weight_max": float(max(dataset.training_weights)),
        "role_counts": dict(sorted(role_counts.items())),
        "role_weight_sums": dict(sorted(role_weight_sums.items())),
        "policy_training_ready": False,
        "formal_data_allowed": False,
        "validation_use": "forbidden; base manifest validation only",
        "model_selection_use": "forbidden",
        "policy_observation_fields_used": [
            "images.side",
            "images.top",
            "instruction",
            "state.elite_tcp_pose_6d",
            *[f"state.controller_state.{field}" for field in CONTROLLER_POLICY_FIELDS],
            "task",
        ],
        "state_32_mapping": {
            "elite_tcp_pose_6d": [0, 6],
            "base_mean_imputation_for_existing_piper_tactile_slots": [6, 14],
            "task_one_hot": [14, 16],
            **{
                key: list(value)
                for key, value in CONTROLLER_EVENT_STATE_LAYOUT.items()
            },
            "reserved_neutral_base_mean_imputation": [27, 32],
            "event_status_values": list(CONTROLLER_EVENT_STATUS_VALUES),
            "missing_controller_event_state": "all zeros in dims 16:27",
            "event_age_encoding": "age_steps / (1 + age_steps); null -> 0 with validity=0",
        },
        "offline_only_fields": ["training_weight", "sample_role"],
        "diagnostic_targets_opened": False,
    }
    return dataset, report


def make_weighted_training_loader(
    base_dataset: Any,
    train_indices: list[int],
    *,
    overlay_dataset: E2OverlayDataset | None,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_steps: int,
) -> tuple[DataLoader, dict[str, Any]]:
    base_probe = PI05ProbeDataset(base_dataset, train_indices)
    if overlay_dataset is None:
        combined: Dataset = base_probe
        weights = torch.ones(len(base_probe), dtype=torch.double)
        overlay_records = 0
        overlay_weight_sum = 0.0
    else:
        combined = ConcatDataset([base_probe, overlay_dataset])
        weights = torch.tensor(
            [1.0] * len(base_probe) + overlay_dataset.training_weights,
            dtype=torch.double,
        )
        overlay_records = len(overlay_dataset)
        overlay_weight_sum = float(sum(overlay_dataset.training_weights))
    generator = torch.Generator().manual_seed(int(seed))
    draws = max(int(max_steps), 1) * max(int(batch_size), 1)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=draws,
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_probe_batch,
        drop_last=False,
    )
    base_weight_sum = float(len(base_probe))
    total_weight = base_weight_sum + overlay_weight_sum
    report = {
        "mode": "weighted_with_replacement",
        "seed": int(seed),
        "optimizer_steps": max(int(max_steps), 1),
        "draws": draws,
        "base_records": len(base_probe),
        "overlay_records": overlay_records,
        "base_weight_sum": base_weight_sum,
        "overlay_weight_sum": overlay_weight_sum,
        "expected_overlay_draw_fraction": (
            overlay_weight_sum / total_weight if total_weight > 0.0 else 0.0
        ),
        "sampler_weight_sha256": hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
    }
    return loader, report
