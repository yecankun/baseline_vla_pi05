from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi05_e2_overlay import (
    CONTROLLER_EVENT_STATE_END,
    CONTROLLER_EVENT_STATE_LAYOUT,
    CONTROLLER_EVENT_STATE_START,
    CONTROLLER_EVENT_STATUS_VALUES,
    _encode_controller_event_state,
    _nested_key_paths,
    _resolve_project_path,
    load_e2_overlay,
    make_weighted_training_loader,
)


class FakeBaseDataset:
    def __init__(self, image_size: int) -> None:
        image = torch.zeros(3, image_size, image_size, dtype=torch.float32)
        state = torch.zeros(32, dtype=torch.float32)
        state[14] = 1.0
        action = torch.zeros(32, dtype=torch.float32)
        action[7] = 1.0
        self.row = {
            "observation.images.side": image,
            "observation.images.top": image,
            "observation.state": state,
            "action": action,
            "task": "left",
            "elite_tcp_delta_6d": torch.zeros(6, dtype=torch.float32),
            "piper_intent_id": 1,
            "episode_index": 0,
            "frame_index": 0,
        }

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index not in (0, 1):
            raise IndexError(index)
        return self.row


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the algorithm-side E2 weighted-overlay adapter.")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--base-openpi-arrays",
        type=Path,
        default=Path(
            "simulation_output/openpi_compat_pack_frontup_step024_temporal_stride5_v1/"
            "openpi_arrays.npz"
        ),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rejected_without_opt_in = False
    try:
        load_e2_overlay(
            args.pack,
            project_root=args.project_root,
            base_state_mean=torch.zeros(32),
            image_size=args.image_size,
            allow_diagnostic_overlay=False,
        )
    except ValueError as exc:
        rejected_without_opt_in = "diagnostic-only" in str(exc)
    if not rejected_without_opt_in:
        raise AssertionError("diagnostic overlay did not fail closed without explicit opt-in")

    dataset, overlay_report = load_e2_overlay(
        args.pack,
        project_root=args.project_root,
        base_state_mean=torch.arange(32, dtype=torch.float32),
        image_size=args.image_size,
        allow_diagnostic_overlay=True,
    )
    if len(dataset) != 836:
        raise AssertionError(f"unexpected E2 record count: {len(dataset)}")
    if overlay_report["diagnostic_targets_opened"] is not False:
        raise AssertionError("diagnostic target separation marker changed")

    encoded_event_states = torch.stack(
        [
            _encode_controller_event_state(
                row["state"].get("controller_state"),
                f"{row['sample_id']}.state.controller_state",
            )
            for row in dataset.rows
        ]
    )
    event_dim_stats: dict[str, dict[str, Any]] = {}
    for name, (start, end) in CONTROLLER_EVENT_STATE_LAYOUT.items():
        local_start = start - CONTROLLER_EVENT_STATE_START
        local_end = end - CONTROLLER_EVENT_STATE_START
        values = encoded_event_states[:, local_start:local_end]
        event_dim_stats[name] = {
            "state_32_slice": [start, end],
            "min": values.amin(dim=0).tolist(),
            "max": values.amax(dim=0).tolist(),
            "nonconstant": [
                bool(value)
                for value in (values.amax(dim=0) > values.amin(dim=0)).tolist()
            ],
        }
    expected_nonconstant = {
        "piper_busy",
        "piper_executed_feed",
        "piper_event_age_steps_saturating",
        "piper_event_status_one_hot",
        "piper_event_age_steps_valid",
    }
    for name in expected_nonconstant:
        if not any(event_dim_stats[name]["nonconstant"]):
            raise AssertionError(f"E2 event field did not enter a nonconstant state dimension: {name}")
    current_pack_constant_fields = {
        name
        for name in ("piper_cooldown", "piper_request_accepted")
        if not any(event_dim_stats[name]["nonconstant"])
    }
    if current_pack_constant_fields != {"piper_cooldown", "piper_request_accepted"}:
        raise AssertionError(
            f"unexpected E2 constant controller fields: {sorted(current_pack_constant_fields)}"
        )

    synthetic = _encode_controller_event_state(
        {
            "piper_busy": False,
            "piper_cooldown": True,
            "piper_request_accepted": True,
            "piper_event_status": "completed",
            "piper_event_age_steps": 0,
            "piper_executed_feed": 0.0,
        },
        "synthetic.controller_state",
    )
    if synthetic[1:3].tolist() != [1.0, 1.0]:
        raise AssertionError("cooldown/request_accepted adapter toggles are not wired")
    synthetic_with_event_id = _encode_controller_event_state(
        {
            "piper_busy": False,
            "piper_cooldown": True,
            "piper_request_accepted": True,
            "piper_event_id": "must-not-enter-state",
            "piper_event_status": "completed",
            "piper_event_age_steps": 0,
            "piper_executed_feed": 0.0,
        },
        "synthetic.controller_state",
    )
    if not torch.equal(synthetic, synthetic_with_event_id):
        raise AssertionError("raw piper_event_id changed the encoded policy state")

    missing_event_state = _encode_controller_event_state({}, "base.controller_state")
    if torch.count_nonzero(missing_event_state).item() != 0:
        raise AssertionError("missing controller event state must map to neutral zeros")
    base_arrays_path = (
        args.base_openpi_arrays
        if args.base_openpi_arrays.is_absolute()
        else args.project_root / args.base_openpi_arrays
    )
    with np.load(base_arrays_path) as base_arrays:
        base_state = np.asarray(base_arrays["state_32"], dtype=np.float32)
    if base_state.ndim != 2 or base_state.shape[1] != 32:
        raise AssertionError(f"unexpected base state_32 shape: {base_state.shape}")
    base_event_tail = base_state[:, CONTROLLER_EVENT_STATE_START:CONTROLLER_EVENT_STATE_END]
    if np.count_nonzero(base_event_tail) != 0:
        raise AssertionError("base records do not have neutral-zero controller event tail")

    missing_images: list[str] = []
    for row in dataset.rows:
        for camera in ("side", "top"):
            path = _resolve_project_path(args.project_root, str(row["images"][camera]))
            if not path.is_file():
                missing_images.append(str(path))
    if missing_images:
        raise AssertionError(f"missing E2 images: {missing_images[:5]}")

    nested = _nested_key_paths({"state": {"estimated": {"contact_flag": 1}}})
    if ("contact_flag", "state.estimated.contact_flag") not in nested:
        raise AssertionError("recursive forbidden-field traversal failed")

    action_mapping: dict[str, int] = {}
    for intent_id, expected_dim in ((1, 7), (2, 8)):
        index = next(i for i, row in enumerate(dataset.rows) if int(row["action"]["piper_intent_id"]) == intent_id)
        sample = dataset[index]
        if sample["action"].shape != (1, 32):
            raise AssertionError(f"unexpected action shape for intent {intent_id}: {sample['action'].shape}")
        nonzero = torch.nonzero(sample["action"][0, 6:9], as_tuple=False).flatten().tolist()
        if nonzero != [intent_id]:
            raise AssertionError(f"bad Piper one-hot for intent {intent_id}: {nonzero}")
        if float(sample["action"][0, expected_dim]) != 1.0:
            raise AssertionError(f"intent {intent_id} did not map to action dim {expected_dim}")
        if "sample_role" in sample or "training_weight" in sample:
            raise AssertionError("offline-only E2 fields leaked into a model sample")
        action_mapping[str(intent_id)] = expected_dim

    loader, sampler_report = make_weighted_training_loader(
        FakeBaseDataset(args.image_size),
        [0, 1],
        overlay_dataset=dataset,
        batch_size=2,
        num_workers=0,
        seed=123,
        max_steps=3,
    )
    if sampler_report["optimizer_steps"] != 3 or sampler_report["draws"] != 6 or len(loader) != 3:
        raise AssertionError(f"sampler step/draw contract failed: {sampler_report}")
    first_batch = next(iter(loader))
    forbidden_batch_keys = {"sample_role", "training_weight", "diagnostic_targets"}
    if forbidden_batch_keys & set(first_batch):
        raise AssertionError(f"offline-only fields leaked into batch: {forbidden_batch_keys & set(first_batch)}")

    report = {
        "status": "passed",
        "records": len(dataset),
        "checked_images": 2 * len(dataset),
        "rejected_without_opt_in": rejected_without_opt_in,
        "diagnostic_targets_opened": False,
        "action_mapping": action_mapping,
        "state_32_event_layout": {
            key: list(value) for key, value in CONTROLLER_EVENT_STATE_LAYOUT.items()
        },
        "event_status_values": list(CONTROLLER_EVENT_STATUS_VALUES),
        "event_dimension_stats": event_dim_stats,
        "current_pack_constant_fields": sorted(current_pack_constant_fields),
        "synthetic_constant_field_toggle_check": "passed",
        "raw_event_id_ignored_check": "passed",
        "missing_event_state_fill": {
            "strategy": "zeros",
            "slice": [CONTROLLER_EVENT_STATE_START, CONTROLLER_EVENT_STATE_END],
            "base_records_checked": int(base_state.shape[0]),
            "base_nonzero_values": int(np.count_nonzero(base_event_tail)),
        },
        "model_batch_keys": sorted(first_batch),
        "sampler": sampler_report,
        "overlay": overlay_report,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"status=passed records={len(dataset)} checked_images={2 * len(dataset)} "
        f"hold_dim={action_mapping['1']} feed_dim={action_mapping['2']} "
        f"overlay_fraction={sampler_report['expected_overlay_draw_fraction']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
