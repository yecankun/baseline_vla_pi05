from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from probe_pi05_lerobot_adapter import PI05ProbeDataset, collate_probe_batch


PACK_SCHEMA = "project2026_contact_risk_diagnostic_pack_v1"
SAMPLE_SCHEMA = "project2026_contact_risk_sample_v1"
RISK_SEMANTICS = "early_near_wall_or_contact_warning; not a strict contact label"
FORBIDDEN_POLICY_KEYS = {
    "contact_flag",
    "contact_strength",
    "distance_to_wall",
    "segment_min_distance_to_wall",
    "sim_contact_flag",
    "sim_contact_strength",
    "sim_distance_to_wall",
    "sim_segment_min_distance_to_wall",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(nested_keys(item))
    return keys


def numeric_vector(value: Any, length: int, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite shape ({length},), got {array.shape}")
    return array


def resolve_image(project_root: Path, pack_dir: Path, reference: Any) -> Path:
    path = Path(str(reference).replace("\\", "/"))
    if path.is_absolute() and path.exists():
        return path
    for candidate in (project_root / path, pack_dir / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found: {reference!r}")


def load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def state_32(sample: dict[str, Any], *, risk_on: bool) -> torch.Tensor:
    state = sample["state"]
    out = np.zeros(32, dtype=np.float32)
    out[0:6] = numeric_vector(state.get("elite_tcp_pose_6d"), 6, label="elite_tcp_pose_6d")
    insertion = state.get("piper_insertion_length")
    out[6:8] = [float(state["piper_step"]), 0.0 if insertion is None else float(insertion)]
    if risk_on:
        out[8:11] = [
            float(state["estimated_contact_risk_flag"]),
            float(state["estimated_contact_risk_probability"]),
            float(state["contact_estimator_confidence"]),
        ]
        out[11:14] = 1.0
    task = str(sample["task"])
    if task not in {"left", "right"}:
        raise ValueError(f"unsupported task: {task!r}")
    out[14:16] = [1.0, 0.0] if task == "left" else [0.0, 1.0]
    return torch.from_numpy(out)


def action_32(sample: dict[str, Any]) -> torch.Tensor:
    action = sample["action"]
    piper = int(action["piper_intent_id"])
    if piper not in {0, 1, 2}:
        raise ValueError(f"invalid piper_intent_id: {piper}")
    out = np.zeros(32, dtype=np.float32)
    out[0:6] = numeric_vector(action.get("elite_tcp_delta_6d"), 6, label="elite_tcp_delta_6d")
    out[6 + piper] = 1.0
    return torch.from_numpy(out)


class ContactRiskAdapterDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        project_root: Path,
        pack_dir: Path,
        image_size: int,
        risk_on: bool,
        episode_indices: dict[str, int],
    ) -> None:
        self.rows = rows
        self.project_root = project_root
        self.pack_dir = pack_dir
        self.image_size = image_size
        self.risk_on = risk_on
        self.episode_indices = episode_indices

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.rows[index]
        images = sample["images"]
        state = state_32(sample, risk_on=self.risk_on)
        action = action_32(sample)
        return {
            "observation.images.side": load_image(
                resolve_image(self.project_root, self.pack_dir, images["side"]), self.image_size
            ),
            "observation.images.top": load_image(
                resolve_image(self.project_root, self.pack_dir, images["top"]), self.image_size
            ),
            "observation.state": state,
            "action": action,
            "task": str(sample["instruction"]),
            "elite_tcp_delta_6d": torch.as_tensor(sample["action"]["elite_tcp_delta_6d"], dtype=torch.float32),
            "piper_intent_id": int(sample["action"]["piper_intent_id"]),
            "episode_index": self.episode_indices[str(sample["episode"])],
            "frame_index": int(sample["step"]),
        }


def validate_pack(manifest: dict[str, Any], samples: list[dict[str, Any]], targets: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema") != PACK_SCHEMA:
        errors.append(f"unexpected manifest schema: {manifest.get('schema')!r}")
    if manifest.get("policy_input_allowed") is not False:
        errors.append("manifest policy_input_allowed must be false")
    if manifest.get("risk_semantics") != RISK_SEMANTICS:
        errors.append("manifest risk semantics mismatch")
    exact = manifest.get("exact_contact", {})
    if exact.get("in_policy_samples") is not False or exact.get("policy_input_allowed") is not False:
        errors.append("exact-contact isolation gate is not fail-closed")
    expected_samples = int(manifest.get("counts", {}).get("samples", -1))
    expected_targets = int(manifest.get("counts", {}).get("diagnostic_targets", -1))
    if len(samples) != expected_samples:
        errors.append(f"sample count {len(samples)} != manifest {expected_samples}")
    if len(targets) != expected_targets:
        errors.append(f"target count {len(targets)} != manifest {expected_targets}")

    target_ids = {str(row.get("sample_id")) for row in targets}
    sample_ids: set[str] = set()
    risk_counts: Counter[int] = Counter()
    piper_counts: Counter[int] = Counter()
    task_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    missing_in_targets = 0
    forbidden_hits: Counter[str] = Counter()
    policy_allowed_records = 0
    rotation_nonzero = 0
    risk_by_task: dict[str, Counter[int]] = defaultdict(Counter)
    risk_by_piper: dict[int, Counter[int]] = defaultdict(Counter)

    for row in samples:
        line = int(row.get("_line_no", -1))
        if row.get("schema") != SAMPLE_SCHEMA:
            errors.append(f"line {line}: unexpected sample schema")
        sample_id = str(row.get("sample_id"))
        if sample_id in sample_ids:
            errors.append(f"line {line}: duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        if sample_id not in target_ids:
            missing_in_targets += 1
        keys = nested_keys(row)
        for key in sorted(keys & FORBIDDEN_POLICY_KEYS):
            forbidden_hits[key] += 1
        state = row.get("state", {})
        provenance = row.get("provenance", {})
        if state.get("contact_risk_policy_input_allowed") is not False:
            policy_allowed_records += 1
        if provenance.get("policy_input_allowed") is not False:
            policy_allowed_records += 1
        if provenance.get("exact_contact_in_policy_state") is not False:
            errors.append(f"line {line}: exact contact is not isolated")
        risk = int(state.get("estimated_contact_risk_flag", -1))
        probability = float(state.get("estimated_contact_risk_probability", np.nan))
        confidence = float(state.get("contact_estimator_confidence", np.nan))
        if risk not in {0, 1} or not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            errors.append(f"line {line}: invalid risk flag/probability")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            errors.append(f"line {line}: invalid contact estimator confidence")
        action = row.get("action", {})
        elite = numeric_vector(action.get("elite_tcp_delta_6d"), 6, label=f"line {line} elite action")
        piper = int(action.get("piper_intent_id", -1))
        if piper not in {0, 1, 2}:
            errors.append(f"line {line}: invalid piper intent {piper}")
        if np.any(np.abs(elite[3:6]) > 1e-8):
            rotation_nonzero += 1
        task = str(row.get("task"))
        risk_counts[risk] += 1
        piper_counts[piper] += 1
        task_counts[task] += 1
        episode_counts[str(row.get("episode"))] += 1
        risk_by_task[task][risk] += 1
        risk_by_piper[piper][risk] += 1

    if forbidden_hits:
        errors.append(f"forbidden exact diagnostic keys in policy samples: {dict(forbidden_hits)}")
    if missing_in_targets:
        errors.append(f"{missing_in_targets} policy samples have no diagnostic target match")
    if len(target_ids) != len(targets):
        errors.append("diagnostic target sample_id values are not unique")
    if policy_allowed_records:
        errors.append(f"{policy_allowed_records} policy allowance gates are not false")

    return {
        "ok": not errors,
        "errors": errors[:50],
        "error_count": len(errors),
        "samples": len(samples),
        "diagnostic_targets": len(targets),
        "episodes": len(episode_counts),
        "tasks": dict(task_counts),
        "risk_flag": {str(key): int(value) for key, value in sorted(risk_counts.items())},
        "piper_intent_id": {str(key): int(value) for key, value in sorted(piper_counts.items())},
        "risk_by_task": {
            key: {str(flag): int(count) for flag, count in sorted(value.items())}
            for key, value in sorted(risk_by_task.items())
        },
        "risk_by_piper_intent_id": {
            str(key): {str(flag): int(count) for flag, count in sorted(value.items())}
            for key, value in sorted(risk_by_piper.items())
        },
        "elite_rotation_nonzero_records": rotation_nonzero,
        "forbidden_policy_key_hits": dict(forbidden_hits),
        "all_sample_ids_match_separate_targets": missing_in_targets == 0 and len(target_ids) == len(targets),
        "policy_input_allowed": False,
    }


def select_adapter_rows(samples: list[dict[str, Any]], max_records: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        key = (
            str(row["task"]),
            int(row["state"]["estimated_contact_risk_flag"]),
            int(row["action"]["piper_intent_id"]),
        )
        groups[key].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < max_records:
        added = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                added = True
                if len(selected) >= max_records:
                    break
        if not added:
            break
    return selected


def batch_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, list):
            result[key] = {"type": "list", "length": len(value)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed schema and existing-adapter smoke for the contact-risk diagnostic pack."
    )
    parser.add_argument("pack", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-adapter-records", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--check-all-images", action="store_true")
    parser.add_argument(
        "--allow-policy-disabled-risk-diagnostic",
        action="store_true",
        help="Required to construct the risk-on adapter view; never promotes the field for training.",
    )
    args = parser.parse_args()

    pack_dir = args.pack if args.pack.is_dir() else args.pack.parent
    manifest = read_json(pack_dir / "manifest.json")
    samples = read_jsonl(pack_dir / str(manifest.get("outputs", {}).get("samples", "samples.jsonl")))
    targets = read_jsonl(
        pack_dir / str(manifest.get("outputs", {}).get("diagnostic_targets", "diagnostic_targets.jsonl"))
    )
    schema_report = validate_pack(manifest, samples, targets)
    if not schema_report["ok"]:
        raise ValueError(f"contact-risk pack validation failed: {schema_report['errors'][:5]}")
    if not args.allow_policy_disabled_risk_diagnostic:
        raise ValueError(
            "risk-on view is policy-disabled; pass --allow-policy-disabled-risk-diagnostic for the explicit smoke only"
        )

    project_root = args.project_root.resolve()
    checked_images = 0
    if args.check_all_images:
        for row in samples:
            for reference in row["images"].values():
                resolve_image(project_root, pack_dir, reference)
                checked_images += 1

    selected = select_adapter_rows(samples, max(int(args.max_adapter_records), 1))
    episode_indices = {episode: idx for idx, episode in enumerate(sorted({str(row["episode"]) for row in samples}))}
    common = {
        "rows": selected,
        "project_root": project_root,
        "pack_dir": pack_dir,
        "image_size": int(args.image_size),
        "episode_indices": episode_indices,
    }
    off_dataset = ContactRiskAdapterDataset(risk_on=False, **common)
    on_dataset = ContactRiskAdapterDataset(risk_on=True, **common)
    indices = list(range(len(selected)))
    off_batch = collate_probe_batch([PI05ProbeDataset(off_dataset, indices)[idx] for idx in indices])
    on_batch = collate_probe_batch([PI05ProbeDataset(on_dataset, indices)[idx] for idx in indices])

    equal_keys = [
        "observation.images.side",
        "observation.images.top",
        "action",
        "elite_tcp_delta_6d",
        "piper_intent_id",
        "episode_index",
        "frame_index",
    ]
    equality = {key: bool(torch.equal(off_batch[key], on_batch[key])) for key in equal_keys}
    state_delta = on_batch["observation.state"] - off_batch["observation.state"]
    changed_dims = torch.nonzero(torch.any(state_delta != 0, dim=0), as_tuple=False).flatten().tolist()
    if not all(equality.values()):
        raise ValueError(f"risk views changed non-state adapter fields: {equality}")
    if any(dim not in range(8, 14) for dim in changed_dims):
        raise ValueError(f"risk view changed state dims outside 8:14: {changed_dims}")

    report = {
        "scope": "schema and existing-adapter smoke only; no model change, training, tactile claim, or real validation",
        "pack": str(pack_dir),
        "manifest_policy_input_allowed": manifest.get("policy_input_allowed"),
        "risk_on_authorization": "explicit diagnostic CLI flag only; field remains policy-disabled",
        "risk_semantics": manifest.get("risk_semantics"),
        "schema": schema_report,
        "images_checked_this_run": checked_images,
        "adapter_smoke": {
            "records": len(selected),
            "selected_sample_ids": [str(row["sample_id"]) for row in selected],
            "risk_off_state_layout": {
                "elite_tcp_pose_6d": [0, 6],
                "piper_state": [6, 8],
                "risk_context_zero": [8, 11],
                "risk_valid_zero": [11, 14],
                "task_one_hot": [14, 16],
                "padding": [16, 32],
            },
            "risk_on_state_layout": {
                "elite_tcp_pose_6d": [0, 6],
                "piper_state": [6, 8],
                "diagnostic_risk_flag_probability_confidence": [8, 11],
                "diagnostic_risk_valid": [11, 14],
                "task_one_hot": [14, 16],
                "padding": [16, 32],
            },
            "batch_shapes": batch_shapes(off_batch),
            "risk_off_on_non_state_equal": equality,
            "risk_off_on_changed_state_dims": changed_dims,
            "risk_on_state_delta_abs_max_by_dim": state_delta.abs().max(dim=0).values.tolist(),
            "model_structure_changed": False,
            "risk_used_as_training_label": False,
        },
        "limits": [
            "estimated_contact_risk is an early warning, not strict contact",
            "risk field remains policy_input_allowed=false",
            "diagnostic targets are ID-checked but never merged into adapter observations",
            "current real pilots remain calibration/OOD data, not training evidence",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"schema_ok={schema_report['ok']} samples={schema_report['samples']} episodes={schema_report['episodes']} "
        f"adapter_records={len(selected)} changed_state_dims={changed_dims} images_checked={checked_images}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
