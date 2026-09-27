"""Offline complete-episode pair from a hash-pinned public cache, never training.

Only the fixed diagnostic source episodes in the plan are read. Exact and
quantized transition-window duplicates fail closed before the feature stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from audit_libero_public_source import validate_episode_rows
from pi05_libero_world_model_adapter import DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, sha256_file
from probe_pi05_libero_public_features import (
    ORIENTATION, local_file, read_json, validate_source, write_json,
)

PLAN_SHA256 = "bad36b6a93769d4210792c60a42892a325d30694d8a087316deeafab7de48a53"
BASE_REPORT_SHA256 = "0d431859588a2f14758fbf1c27b17e1ec6114f161c14f5cc246ae2e55317c4e5"


def validate_plan(plan):
    expected = {
        "schema": "pi05_libero_feature_pair_plan_v2", "source_repo_id": DEMO_REPO,
        "source_revision": DEMO_REVISION, "prior_source_report_sha256": BASE_REPORT_SHA256,
        "selected_task_id": 9, "source_task_index": 39, "fps": 10.0,
        "context_len": 4, "horizon": 3, "orientation": ORIENTATION, "max_records": 400,
        "episodes": [{"episode_index": 1400, "record_count": 140, "partition": "train"},
                     {"episode_index": 1402, "record_count": 173, "partition": "validation"}],
        "duplicate_guard": {"window_length": 7, "state_quantization": 0.0001,
                            "action_quantization": 0.001,
                            "reject_cross_split_matching_state_action_window": True,
                            "reject_cross_split_matching_two_view_pixel_window": True},
        "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
        "policy_generalization_evaluation_allowed": False, "training_ready": False,
        "training_authorized": False, "full_suite_coverage": False,
    }
    for key, value in expected.items():
        if key not in plan or type(plan[key]) is not type(value) or plan[key] != value:
            raise ValueError(f"fixed diagnostic plan changed: {key}")
    return plan


def hash_array(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def transition_signatures(state, action, window=7, quantized=False):
    if state.shape != (len(state), 8) or action.shape != (len(state), 7) or len(state) < window:
        raise ValueError("complete native state/action arrays required")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("finite trajectories required for duplicate audit")
    state, action = state.astype("<f4"), action.astype("<f4")
    if quantized:
        state, action = np.rint(state.astype(np.float64)/1e-4), np.rint(action.astype(np.float64)/1e-3)
        if np.max(np.abs(state)) >= 2**62 or np.max(np.abs(action)) >= 2**62:
            raise ValueError("quantized trajectory overflows safe integer range")
        state, action = state.astype("<i8"), action.astype("<i8")
    # Seven states and SIX connecting actions: the last state's unrelated action
    # must never hide a duplicated transition window or duplicated terminal suffix.
    return [hashlib.sha256(state[i:i+window].tobytes() + action[i:i+window-1].tobytes()).hexdigest()
            for i in range(len(state)-window+1)]


def pixel_signatures(frame_hashes, window=7):
    return [hashlib.sha256("".join(frame_hashes[i:i+window]).encode()).hexdigest()
            for i in range(len(frame_hashes)-window+1)]


def matching_runs(left, right):
    previous = np.zeros(len(right)+1, dtype=np.int64)
    longest = 0
    for a in left:
        current = np.zeros_like(previous)
        for j, b in enumerate(right):
            if a == b:
                current[j+1] = previous[j]+1
                longest = max(longest, int(current[j+1]))
        previous = current
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    suffix = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        suffix += 1
    return {"longest_exact_contiguous_match": longest, "common_prefix": prefix, "common_suffix": suffix}


def duplicate_audit(first, second):
    s1, a1, h1 = first
    s2, a2, h2 = second
    if len(h1) != len(s1) or len(h2) != len(s2):
        raise ValueError("pixel hashes must cover the complete state rows")
    matches = {}
    for quantized in (False, True):
        v1 = transition_signatures(s1, a1, quantized=quantized)
        v2 = transition_signatures(s2, a2, quantized=quantized)
        matches["quantized_transition_windows" if quantized else "exact_transition_windows"] = len(set(v1) & set(v2))
    matches["two_view_pixel_windows"] = len(set(pixel_signatures(h1)) & set(pixel_signatures(h2)))
    nearest_xyz = np.linalg.norm(s1[:, None, :3]-s2[None, :, :3], axis=-1)
    audit = {
        "status": "passed" if not any(matches.values()) else "rejected",
        "window_length": 7, "connecting_action_count": 6,
        "state_quantization": 1e-4, "action_quantization": 1e-3,
        "cross_split_matching_windows": matches,
        "state_runs": matching_runs([hash_array(s) for s in s1], [hash_array(s) for s in s2]),
        "action_runs": matching_runs([hash_array(a) for a in a1[:-1]], [hash_array(a) for a in a2[:-1]]),
        "two_view_pixel_runs": matching_runs(h1, h2),
        "initial_xyz_distance": float(np.linalg.norm(s1[0,:3]-s2[0,:3])),
        "median_nearest_xyz_distance_first_to_second": float(np.median(nearest_xyz.min(axis=1))),
        "family_independence_verified": False,
        "meaning": "only these exact/quantized duplicate tests; no verified latent family independence or effective sample count",
    }
    return audit


def decode_episode(video_paths, metadata, count, fps, out):
    import av
    from PIL import Image
    array = np.lib.format.open_memmap(out / "images.npy", mode="w+", dtype=np.uint8,
                                     shape=(count, 2, 256, 256, 3))
    evidence = []
    try:
        for vi, key in enumerate(IMAGE_KEYS):
            start = float(metadata[f"videos/{key}/from_timestamp"])
            stop = float(metadata[f"videos/{key}/to_timestamp"])
            if not np.isclose(stop-start, count/fps, atol=2e-5, rtol=0):
                raise ValueError("video offset interval differs from complete row count")
            found = {}
            with av.open(str(video_paths[vi])) as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height, float(stream.average_rate)) != (256, 256, fps):
                    raise ValueError("video dimensions/fps drift")
                for frame in container.decode(stream):
                    timestamp = float(frame.pts * stream.time_base)
                    if timestamp < start-.001:
                        continue
                    if timestamp > start+(count-1)/fps+.001:
                        break
                    index = round((timestamp-start)*fps)
                    if not 0 <= index < count or index in found or abs(timestamp-(start+index/fps)) > .001:
                        raise ValueError("video timestamp duplicate/gap/off-grid")
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                        raise ValueError("native RGB image shape/dtype changed")
                    array[index, vi] = rgb
                    found[index] = {"frame_index": index, "video_timestamp": timestamp,
                                    "decoded_rgb_sha256": hash_array(rgb)}
                    if index in {0, count//2, count-1}:
                        Image.fromarray(rgb).save(out / f"view{vi}_frame{index:04d}.png")
            if set(found) != set(range(count)):
                raise ValueError("not all declared episode video frames were decoded")
            evidence.append({"key": key, "from_timestamp": start, "to_timestamp": stop,
                             "frames": [found[i] for i in range(count)]})
        array.flush()
    finally:
        array._mmap.close()
    write_json(out / "decoded_frames.json", evidence)
    return [evidence[0]["frames"][i]["decoded_rgb_sha256"] +
            evidence[1]["frames"][i]["decoded_rgb_sha256"] for i in range(count)]


def run(args):
    import pandas as pd
    started = time.monotonic()
    if sha256_file(args.plan) != PLAN_SHA256:
        raise ValueError("frozen diagnostic plan hash changed")
    plan = validate_plan(read_json(args.plan))
    prior = validate_source(args.base_source_root, BASE_REPORT_SHA256)
    root = args.base_source_root.resolve()
    paths = {f["path"]: local_file(root, f["local_path"]) for f in prior["report"]["files"]}
    info = read_json(paths["meta/info.json"])
    metadata_table = pd.read_parquet(paths["meta/episodes/chunk-000/file-000.parquet"])
    if info["fps"] != plan["fps"]:
        raise ValueError("source fps differs from fixed plan")
    selected_task = [r for r in prior["report"]["task_registry"] if r["task_id"] == plan["selected_task_id"]]
    if len(selected_task) != 1 or selected_task[0]["source_task_index"] != plan["source_task_index"]:
        raise ValueError("prior task mapping differs from pair plan")
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "status.json", {"status": "running", "optimizer_steps": 0})
    try:
        (args.out / "plan.json").write_bytes(args.plan.read_bytes())
        episodes, sequences = [], []
        for choice in plan["episodes"]:
            eid = choice["episode_index"]
            md = metadata_table[metadata_table.episode_index == eid]
            if len(md) != 1:
                raise ValueError("source episode metadata missing/ambiguous")
            metadata = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in md.iloc[0].to_dict().items()}
            if metadata["length"] != choice["record_count"]:
                raise ValueError("complete episode count differs from predeclared plan")
            dpath = info["data_path"].format(chunk_index=metadata["data/chunk_index"], file_index=metadata["data/file_index"])
            if dpath not in paths:
                raise ValueError("source pair needs uncached payload; do not auto-download")
            data = pd.read_parquet(paths[dpath])
            data = data[data.episode_index == eid].sort_values("frame_index")
            # Convert each float32 value through tolist without lossy decimal rounding.
            records = []
            for _, row in data.iterrows():
                records.append({"episode_index": int(row.episode_index), "frame_index": int(row.frame_index),
                                "index": int(row["index"]), "timestamp": float(row.timestamp),
                                "task_index": int(row.task_index), "task": selected_task[0]["task_instruction"],
                                "observation.state": row["observation.state"].tolist(), "action": row.action.tolist()})
            adapted = validate_episode_rows(records, metadata, selected_task, info["fps"])
            state = np.asarray([r["observation.state"] for r in records], dtype=np.float32)
            action = np.asarray([r["action"] for r in records], dtype=np.float32)
            if not np.array_equal(state, np.stack(data["observation.state"])) or not np.array_equal(action, np.stack(data.action)):
                raise ValueError("state/action changed through record adaptation")
            child = args.out / f"episode_{eid}"
            child.mkdir(exist_ok=False)
            write_json(child / "records.json", records)
            video_keys = [info["video_path"].format(video_key=key, chunk_index=metadata[f"videos/{key}/chunk_index"],
                          file_index=metadata[f"videos/{key}/file_index"]) for key in IMAGE_KEYS]
            if any(k not in paths for k in video_keys):
                raise ValueError("source pair requires unverified video payload")
            pixel_hashes = decode_episode([paths[k] for k in video_keys], metadata, len(records), info["fps"], child)
            episodes.append({"episode_index": eid, "task_id": adapted[0]["task_id"],
                             "source_task_index": plan["source_task_index"], "record_count": len(records),
                             "complete_episode": True, "metadata": metadata,
                             "images_path": f"episode_{eid}/images.npy", "images_sha256": sha256_file(child / "images.npy"),
                             "records_path": f"episode_{eid}/records.json", "records_sha256": sha256_file(child / "records.json"),
                             "source_trajectory_id": f"{DEMO_REPO}@{DEMO_REVISION}/episode{eid}",
                             "leakage_group_id": f"source_episode_{eid}",
                             "float32_parquet_roundtrip_equal": True,
                             "action_change_rows_at_1e_3": int(np.sum(np.any(np.abs(np.diff(action, axis=0)) > 1e-3, axis=1)))})
            sequences.append((state, action, pixel_hashes))
            print(f"EPISODE_SOURCE_COMPLETE episode={eid} rows={len(records)}", flush=True)
        if sum(e["record_count"] for e in episodes) > plan["max_records"]:
            raise ValueError("whole-job record cap exceeded")
        # Shared MP4 storage is permitted only with non-overlapping episode intervals.
        for key in IMAGE_KEYS:
            a, b = [e["metadata"] for e in episodes]
            if (a[f"videos/{key}/chunk_index"], a[f"videos/{key}/file_index"]) == (b[f"videos/{key}/chunk_index"], b[f"videos/{key}/file_index"]):
                if min(a[f"videos/{key}/to_timestamp"], b[f"videos/{key}/to_timestamp"]) - max(a[f"videos/{key}/from_timestamp"], b[f"videos/{key}/from_timestamp"]) > 1e-6:
                    raise ValueError("cross-split source video intervals overlap")
        duplicates = duplicate_audit(*sequences)
        write_json(args.out / "duplicate_suffix_audit.json", duplicates)
        if duplicates["status"] != "passed":
            raise ValueError("cross-split duplicate sequence detected; retain pair, no truncation/replacement")
        # A second cache readback detects any mutation during decoding; no old file is written.
        validate_source(root, BASE_REPORT_SHA256)
        report = {
            "schema": "pi05_libero_source_pair_v1", "status": "passed",
            "source": {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION},
            "fps": info["fps"], "orientation": ORIENTATION, "task_registry": selected_task,
            "episodes": episodes, "row_count": sum(e["record_count"] for e in episodes),
            "split": {"train_episode_indices": [1400], "validation_episode_indices": [1402]},
            "plan_path": "plan.json", "plan_sha256": PLAN_SHA256,
            "prior_source_report_sha256": BASE_REPORT_SHA256, "prior_source_root": str(root),
            "prior_source_files_rehashed": prior["verified_files"], "prior_source_unchanged_after_decode": True,
            "duplicate_suffix_audit": duplicates,
            "source_trajectory_identity_scope": "public_native_episode_index_not_verified_original_independent_family",
            "leakage_group_scope": "source_episode_only",
            "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
            "benchmark_unseen_checkpoint_holdout": False, "full_suite_coverage": False,
            "training_ready": False, "optimizer_steps": 0, "policy_loaded": False, "network_downloads": 0,
            "timing_scope": "stored record/video cadence only; upstream hidden resets and physical timing unverified",
            "code_sha256": sha256_file(Path(__file__)),
            "runtime_seconds": round(time.monotonic()-started, 3), "visual_status": "not_viewed",
        }
        report["output_sha256"] = {p.relative_to(args.out).as_posix(): sha256_file(p)
                                   for p in args.out.rglob("*") if p.is_file() and p.name != "status.json"}
        write_json(args.out / "report.json", report)
        write_json(args.out / "status.json", {"status": "completed", "report_sha256": sha256_file(args.out / "report.json")})
        print(json.dumps({"status": "passed", "rows": report["row_count"], "split": report["split"],
                          "runtime_seconds": report["runtime_seconds"], "training_ready": False}), flush=True)
    except Exception:
        write_json(args.out / "status.json", {"status": "failed", "traceback": traceback.format_exc(), "optimizer_steps": 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("docs/libero-feature-pair-plan-v2.json"))
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
