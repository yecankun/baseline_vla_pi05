"""Adapt the ten August real captures for a train-only PI05 prototype.

No collector changes, data acceptance, optimizer updates, or hardware calls.
The state encoder is also the preprocessing contract for a later live client.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np

from export_openpi_compat_to_lerobot import read_json, read_jsonl, task_text, write_json
from prepare_openpi_compat_pack import compute_stats, make_action_32, write_jsonl


ADAPTER_VERSION = "real10_pi05_observable_history_v1"
EPISODES = tuple(
    f"real_pilot_20260817_{task}_s_bend_replacement_tip_fixedudp_{number:03d}"
    for task, numbers in (("left", range(2, 7)), ("right", range(1, 6)))
    for number in numbers
)
RAW_TO_INTENT = {-1: 0, 0: 1, 1: 2}
STATE_LAYOUT = {
    "elite_tcp_pose_6d_xyz_mm_rpy_rad": [0, 6],
    "previous_piper_step_after_command_event_count_not_mm": [6, 7],
    "insertion_length_unavailable_zero": [7, 8],
    "tactile_context_unavailable_zero": [8, 14],
    "task_left_right_one_hot": [14, 16],
    "previous_piper_busy": [16, 17],
    "full_e2_event_fields_and_validity_unavailable_zero": [17, 27],
    "previous_piper_history_valid": [27, 28],
    "padding_zero": [28, 32],
}


def encode_real_state(elite_tcp_pose_6d, task: str, previous_controller=None):
    """Use current measured pose and the PREVIOUS record's controller snapshot.

    Current recorded controller fields may already describe the target command.
    At episode reset pass None; live inference must maintain the same one-step
    history, not inject the current label/command result. Missing history is
    zero plus state[27]=0. Never infer executed feed from async_started.
    """
    pose = np.asarray(elite_tcp_pose_6d, dtype=np.float32)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Elite pose must be six finite values [xyz_mm, rpy_rad]")
    if task not in ("left", "right"):
        raise ValueError(f"unsupported task: {task}")
    state = np.zeros(32, dtype=np.float32)
    mask = np.zeros(32, dtype=np.bool_)
    state[:6] = pose
    state[14 + (task == "right")] = 1
    mask[:6] = True
    mask[14:16] = True
    mask[27] = True
    if previous_controller is not None:
        count = previous_controller.get("piper_step_after_command")
        busy = previous_controller.get("piper_busy")
        if count is not None and busy is not None:
            if not np.isfinite(count) or count < 0 or not isinstance(busy, bool):
                raise ValueError("history requires a finite non-negative event count and bool busy")
            state[6] = count
            state[16] = float(busy)
            state[27] = 1
            mask[[6, 16]] = True
    return state, mask


def resize_real_bgr(image, image_size: int = 224):
    """Same square INTER_AREA preprocessing as the existing LeRobot exporter."""
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def quantiles(values):
    return dict(zip(("min", "median", "p95", "max"),
                    np.quantile(values, (0, .5, .95, 1)).astype(float).tolist()))


def prepare_pack(source_root: Path, out: Path, *, episodes=EPISODES, image_size=224):
    started = time.monotonic()
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    out.mkdir(parents=True, exist_ok=False)
    index, states, state_masks, deltas, intents = [], [], [], [], []
    episode_reports, source_manifests = [], {}
    record_dt, pose_dt, pose_image_lag = [], [], []
    source_shapes, decoded, image_bytes, source_rgb_bytes = Counter(), 0, 0, 0
    for episode in episodes:
        source = source_root / episode
        manifest = read_json(source / "manifest.json")
        rows = read_jsonl(source / "records.jsonl")
        if len(rows) < 2:
            raise ValueError(f"{episode}: need at least two observations")
        if not manifest.get("zero_orientation_delta"):
            raise ValueError(f"{episode}: expected the fixed-orientation capture contract")
        if manifest["camera_setup"].get("top_duplicated_from_side"):
            raise ValueError(f"{episode}: duplicate-camera data are outside this real10 adapter")
        source_manifests[episode] = manifest  # provenance only, never fed to policy
        first_index = len(index)
        previous_controller = None
        episode_intents = Counter()
        for position, row in enumerate(rows):
            task, step = row["task"], int(row["step"])
            if task != manifest["task"] or step != position:
                raise ValueError(f"{episode}: unexpected task/step at row {position}")
            controller = row["state"]["controller_state"]
            if controller.get("elite_pose_stale", False):
                raise ValueError(f"{episode}/{step}: stale Elite pose")
            state, mask = encode_real_state(row["state"]["elite_tcp_pose_6d"], task, previous_controller)
            timestamps = row["timestamps"]
            if max(timestamps["side_image"], timestamps["top_image"], timestamps["elite_pose"]) > timestamps["piper_action"]:
                raise ValueError(f"{episode}/{step}: observation later than recorded decision")
            if position and rows[position - 1]["timestamps"]["piper_action"] > row["timestamp"]:
                raise ValueError(f"{episode}/{step}: previous controller result is not causal history")
            pose_image_lag.append(timestamps["elite_pose"] - row["timestamp"])
            paths = {}
            # Keep terminal images as endpoint context, but not terminal action supervision.
            for view in ("side", "top"):
                image_path = source / row[f"{view}_image"]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"unreadable image: {image_path}")
                source_shapes[str(list(image.shape))] += 1
                source_rgb_bytes += image_path.stat().st_size
                relative = Path(episode) / view / f"{step:06d}.png"
                destination = out / "images" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), resize_real_bgr(image, image_size)):
                    raise OSError(f"cannot write {destination}")
                image_bytes += destination.stat().st_size
                decoded += 1
                paths[view] = relative.as_posix()
            if position == len(rows) - 1:
                if row["reference_action"]["piper_step_command"] != 0:
                    raise ValueError(f"{episode}: terminal feed would need separate supervision")
                continue
            following = rows[position + 1]
            dt = float(following["timestamp"] - row["timestamp"])
            elite_dt = float(following["timestamps"]["elite_pose"] - timestamps["elite_pose"])
            if dt <= 0 or elite_dt <= 0:
                raise ValueError(f"{episode}/{step}: non-monotonic observation time")
            delta = np.asarray(row["reference_action"]["elite_tcp_delta_6d"], dtype=np.float32)
            expected = np.asarray(following["state"]["elite_tcp_pose_6d"], dtype=np.float32)[:3] - state[:3]
            if delta.shape != (6,) or not np.isfinite(delta).all() or not np.allclose(delta[:3], expected, atol=1e-5, rtol=0):
                raise ValueError(f"{episode}/{step}: stored delta differs from next measured pose")
            if np.any(delta[3:] != 0):
                raise ValueError(f"{episode}/{step}: rotation is outside this prototype")
            raw_command = row["reference_action"]["piper_step_command"]
            intent = RAW_TO_INTENT[raw_command]
            if raw_command != 0 and row["reference_action"].get("piper_burst_count") != 1:
                raise ValueError(f"{episode}/{step}: non-single feed event")
            if intent == 0:
                raise ValueError(f"{episode}/{step}: this capture batch is hold/feed only")
            index.append({
                "sample_id": f"{episode}/{step:06d}", "split": "train", "task": task,
                "language_instruction": task_text(row), "pack_index": len(index),
                "observation.images.side": paths["side"], "observation.images.top": paths["top"],
                "source": {"episode": episode, "step": step, "next_step": int(following["step"])},
                "timing": {"timestamp": row["timestamp"],
                           "side_image": timestamps["side_image"], "top_image": timestamps["top_image"],
                           "elite_pose": timestamps["elite_pose"], "piper_action": timestamps["piper_action"],
                           "observation_dt_s": dt, "elite_action_dt_s": elite_dt,
                           "history_source_step": step - 1 if previous_controller is not None else None},
            })
            states.append(state)
            state_masks.append(mask)
            deltas.append(delta)
            intents.append(intent)
            episode_intents[str(intent)] += 1
            record_dt.append(dt)
            pose_dt.append(elite_dt)
            previous_controller = controller
        episode_reports.append({
            "episode": episode, "task": manifest["task"], "split": "train",
            "source_records": len(rows), "training_transitions": len(index) - first_index,
            "pack_index_range": [first_index, len(index)],
            "piper_intent_id": dict(episode_intents),
            "terminal_source_step": int(rows[-1]["step"]),
        })
        print(f"{episode}: {len(rows)} observations -> {len(index) - first_index} transitions", flush=True)

    state_array, elite_array = np.stack(states), np.stack(deltas)
    intent_array = np.asarray(intents, dtype=np.int64)
    target_arrays = {"elite_tcp_delta_6d": elite_array,
                     "piper_intent_one_hot": np.eye(3, dtype=np.float32)[intent_array]}
    action_array, action_mask, action_layout = make_action_32(target_arrays)
    np.savez_compressed(out / "openpi_arrays.npz", state_32=state_array,
                        state_mask_32=np.stack(state_masks), action_32=action_array,
                        action_mask_32=action_mask, elite_tcp_delta_6d=elite_array,
                        piper_intent_id=intent_array)
    write_jsonl(out / "index.jsonl", index)
    write_json(out / "source_episode_manifests.json", source_manifests)
    report = {
        "schema": "project2026_real10_adaptation_report_v1", "adapter_version": ADAPTER_VERSION,
        "episodes": episode_reports, "source_records": sum(x["source_records"] for x in episode_reports),
        "training_transitions": len(index), "terminal_rows_not_supervised": len(episodes),
        "piper_intent_id": dict(Counter(str(x) for x in intents)),
        "images_decoded_and_resized": decoded, "source_image_shapes": dict(source_shapes),
        "image_bytes": image_bytes, "source_rgb_bytes": source_rgb_bytes,
        "observation_dt_s": quantiles(record_dt), "elite_action_dt_s": quantiles(pose_dt),
        "pose_minus_image_timestamp_s": quantiles(pose_image_lag),
        "nonconstant_state_dimensions": np.flatnonzero(np.ptp(state_array, axis=0) > 0).tolist(),
        "history_valid_rows": int(state_array[:, 27].sum()),
        "history_busy_rows": int(state_array[:, 16].sum()),
        "elite_translation_abs_max_mm": np.abs(elite_array[:, :3]).max(axis=0).tolist(),
        "elite_rotation_all_zero": bool(np.all(elite_array[:, 3:] == 0)),
        "all_stored_translation_labels_match_next_measured_pose": True,
        "checks_passed": True, "elapsed_seconds": time.monotonic() - started,
        "visual_status": "not_viewed", "training_executed": False, "hardware_executed": False,
    }
    manifest = {
        "schema": "project_2026_openpi_compat_pack_v0", "adapter_version": ADAPTER_VERSION,
        "source_root": source_root.as_posix(), "source_episodes": list(episodes),
        "counts": {"samples": len(index), "episodes": len(episodes),
                   "source_records": report["source_records"], "splits": {"train": len(index), "val": 0},
                   "piper_intent_id": report["piper_intent_id"]},
        "training_use": {"scope": "user_authorized_real10_train_only_prototype",
                         "split_strategy": "all_10_episodes_train_no_validation",
                         "heldout_metric_claim_allowed": False,
                         "policy_training_ready": False, "formal_data_allowed": False,
                         "real_system_validated": False,
                         "terminal_policy": "keep endpoint images; omit artificial zero action without next observation"},
        "output": {"arrays_npz": "openpi_arrays.npz", "index_jsonl": "index.jsonl",
                   "image_root": "images", "image_paths_are_relative_to_image_root": True},
        "openpi_compat": {"state_dim": 32, "action_dim": 32, "state_layout": STATE_LAYOUT,
                          "action_layout": action_layout,
                          "preferred_target": "elite_tcp_delta_6d + piper_intent_id",
                          "raw_to_canonical_piper_id": RAW_TO_INTENT,
                          "history": "previous record only; reset/missing -> zeros with dim27=0",
                          "missing_fields": "zero, not neutral imputation from an old simulation checkpoint",
                          "excluded_inputs": ["current_command_and_label", "elite_requested_pose_and_joints",
                                              "elite_path_index_and_status", "user_event", "event_id",
                                              "depth", "estimated_fields", "diagnostic_targets"]},
        "input_preprocessing": {"image_size": image_size, "resize": "square_cv2_INTER_AREA",
                                "stored_format": "lossless_png", "policy_channels": "RGB",
                                "crop_or_enhancement": False},
        "timing": {"physical_sampling": "irregular_original_timestamps_preserved_in_index",
                   "lerobot_index_fps": 5, "lerobot_fps_is_physical": False,
                   "action": "unaltered next-measured-pose delta, mm/rad; not velocity",
                   "initial_policy_action_horizon": 1,
                   "fixed_time_action_chunks_allowed": False,
                   "observation_dt_s": report["observation_dt_s"],
                   "elite_action_dt_s": report["elite_action_dt_s"]},
        "stats": {"state_32": compute_stats(state_array), "action_32": compute_stats(action_array),
                  "elite_tcp_delta_6d": compute_stats(elite_array)},
    }
    write_json(out / "report.json", report)
    write_json(out / "manifest.json", manifest)
    return report


def make_archive(pack: Path):
    destination = pack.with_suffix(".zip")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(pack.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(pack.parent).as_posix())
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--archive", action="store_true", help="Create a user-upload ZIP alongside the pack")
    args = parser.parse_args()
    report = prepare_pack(args.source_root, args.out, image_size=args.image_size)
    print(json.dumps({key: report[key] for key in (
        "source_records", "training_transitions", "piper_intent_id", "images_decoded_and_resized",
        "image_bytes", "elapsed_seconds")}, indent=2))
    if args.archive:
        archive = make_archive(args.out)
        print(f"User transfer required: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
