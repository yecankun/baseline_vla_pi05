"""Bounded pinned public-demo audit and seven-row input probe; no model/training.

Downloads only metadata and payload files needed for one declared complete
episode. It does not create a training dataset, split or feature pack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from pi05_libero_world_model_adapter import (
    DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, adapt_native_record, sha256_file,
    validate_task_registry,
)

MAX_FILE_BYTES = 100_000_000
MAX_TOTAL_BYTES = 90_000_000
META_PATHS = ("README.md", "meta/info.json", "meta/tasks.parquet",
              "meta/episodes/chunk-000/file-000.parquet")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def verified_payload(data: bytes, entry: dict) -> str:
    if not 0 < entry["size"] <= MAX_FILE_BYTES or len(data) != entry["size"]:
        raise ValueError("payload size mismatch or user-transfer boundary exceeded")
    sha = hashlib.sha256(data).hexdigest()
    if entry["lfs_sha256"]:
        if sha != entry["lfs_sha256"]:
            raise ValueError("LFS SHA256 mismatch")
    elif hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() != entry["blob_id"]:
        raise ValueError("Git blob identity mismatch")
    return sha


def task_mapping(source_rows, native_rows):
    if len(source_rows) != 40 or len({r["source_task_index"] for r in source_rows}) != 40:
        raise ValueError("expected 40 unique pinned source task IDs")
    if {r["source_task_index"] for r in source_rows} != set(range(40)):
        raise ValueError("unexpected source task index range")
    mapping = []
    for native in native_rows:
        matches = [r for r in source_rows if r["task_instruction"] == native["task_instruction"]]
        if len(matches) != 1:
            raise ValueError("native task text has no unique exact source match")
        mapping.append({"task_id": native["task_id"], **matches[0]})
    if set(validate_task_registry(mapping)) != set(range(10)):
        raise ValueError("all ten native Spatial tasks required")
    return mapping


def validate_episode_rows(records, episode, registry, fps):
    count = int(episode["length"])
    if len(records) != count or count < 7:
        raise ValueError("selected complete episode must have all rows, at least seven")
    adapted = []
    for i, row in enumerate(records):
        if (type(row["frame_index"]) is not int or row["frame_index"] != i
                or row["episode_index"] != int(episode["episode_index"])
                or row["index"] != int(episode["dataset_from_index"]) + i):
            raise ValueError("episode/index/frame gap, duplicate or reset")
        if not np.isclose(row["timestamp"], i / fps, atol=2e-5, rtol=0):
            raise ValueError("record timestamp disagrees with metadata fps")
        # Never recover declared-present public state by imputing it away.
        if "observation.state" not in row or np.shape(row["observation.state"]) != (8,):
            raise ValueError("declared 8D public state missing or inconsistent")
        selected = {k: row[k] for k in ("episode_index", "frame_index", "task_index", "task",
                                       "observation.state", "action")}
        adapted.append(adapt_native_record(selected, registry))
    if len({r["task_id"] for r in adapted}) != 1:
        raise ValueError("task changes inside selected episode")
    return adapted


def decode_views(video_paths, episode, fps, count, out):
    import av
    from PIL import Image
    probe = np.empty((7, 2, 256, 256, 3), dtype=np.uint8)
    evidence = []
    # Whole selected episode is timestamp/decodability checked, not just seven frames.
    for vi, key in enumerate(IMAGE_KEYS):
        start = float(episode[f"videos/{key}/from_timestamp"])
        stop = float(episode[f"videos/{key}/to_timestamp"])
        if not np.isclose(stop - start, count / fps, atol=2e-5, rtol=0):
            raise ValueError("video time interval disagrees with complete episode")
        timestamps = start + np.arange(count) / fps
        found = {}
        with av.open(str(video_paths[vi])) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (256, 256) or float(stream.average_rate) != fps:
                raise ValueError("native video geometry/fps differs from source metadata")
            # Sequential decode avoids wrong-episode results from approximate keyframe seeking.
            for frame in container.decode(stream):
                timestamp = float(frame.pts * stream.time_base)
                if timestamp < start - 0.001:
                    continue
                if timestamp > timestamps[-1] + 0.001:
                    break
                idx = round((timestamp - start) * fps)
                if idx < 0 or idx >= count or abs(timestamp - timestamps[idx]) > 0.001 or idx in found:
                    raise ValueError("duplicate/off-grid video timestamp")
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
                    raise ValueError("decoded video is not native RGB uint8")
                found[idx] = {"frame_index": idx, "video_timestamp": timestamp,
                              "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
                if idx < 7:
                    probe[idx, vi] = rgb
                if idx in {0, count // 2, count - 1}:
                    Image.fromarray(rgb).save(out / f"source_view{vi}_frame{idx:04d}.png")
        if set(found) != set(range(count)):
            raise ValueError("video does not contain every declared episode frame")
        evidence.append({"key": key, "from_timestamp": start, "to_timestamp": stop,
                         "frame_count": count, "frames": list(found.values())})
    np.save(out / "probe_images.npy", probe, allow_pickle=False)
    write_json(out / "decoded_frames.json", evidence)
    return evidence


def run(args):
    import inspect
    import pandas as pd
    import requests
    from huggingface_hub import HfApi

    started = time.monotonic()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"status": "running", "pid": os.getpid()})
    try:
        api = HfApi(endpoint="https://huggingface.co")
        hub = api.dataset_info(DEMO_REPO, revision=DEMO_REVISION, files_metadata=True, timeout=30)
        if hub.sha != DEMO_REVISION:
            raise ValueError("Hub revision did not resolve exactly")
        inventory = {s.rfilename: {"path": s.rfilename, "size": s.size, "blob_id": s.blob_id,
                                 "lfs_sha256": s.lfs.sha256 if s.lfs else None}
                     for s in hub.siblings}
        write_json(out / "hub_inventory.json", inventory)
        downloaded, files = 0, []

        def fetch(path):
            nonlocal downloaded
            if path not in inventory or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("unknown or unsafe source file path")
            item = inventory[path]
            if not 0 < item["size"] <= MAX_FILE_BYTES or downloaded + item["size"] > MAX_TOTAL_BYTES:
                raise ValueError("download budget exceeded; user transfer required, never split to bypass")
            target = out / "source" / path
            if target.exists():
                return target
            print(f"DOWNLOAD {path} bytes={item['size']}", flush=True)
            url = f"https://huggingface.co/datasets/{DEMO_REPO}/resolve/{DEMO_REVISION}/{path}"
            chunks, size = [], 0
            with requests.get(url, timeout=(15, 45), stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > item["size"]:
                        raise ValueError("server exceeded pinned file size")
                    chunks.append(chunk)
            data = b"".join(chunks)
            sha = verified_payload(data, item)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            downloaded += size
            files.append({**item, "sha256": sha, "local_path": str(target.relative_to(out)), "url": url})
            return target

        paths = {p: fetch(p) for p in META_PATHS}
        info = json.loads(paths["meta/info.json"].read_text())
        if (info["codebase_version"] != "v3.0" or info["total_episodes"] != 1693
                or info["total_frames"] != 273465 or info["total_tasks"] != 40 or info["fps"] != 10.0):
            raise ValueError("pinned metadata totals/format drift")
        for key, shape, dtype in [("observation.state", [8], "float32"), ("action", [7], "float32")]:
            if info["features"][key]["shape"] != shape or info["features"][key]["dtype"] != dtype:
                raise ValueError("native state/action metadata drift")
        for key in IMAGE_KEYS:
            if info["features"][key]["shape"] != [256, 256, 3] or info["features"][key]["dtype"] != "video":
                raise ValueError("native camera metadata drift")
        tasks = pd.read_parquet(paths["meta/tasks.parquet"])
        source_tasks = [{"source_task_index": int(row.task_index), "task_instruction": str(text)}
                        for text, row in tasks.iterrows()]
        os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config.resolve())
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        registry = task_mapping(source_tasks, [{"task_id": i, "task_instruction": suite.get_task(i).language}
                                               for i in range(10)])
        eps = pd.read_parquet(paths["meta/episodes/chunk-000/file-000.parquet"])
        if (len(eps) != info["total_episodes"] or eps.episode_index.tolist() != list(range(len(eps)))
                or (eps.length <= 0).any() or eps.length.sum() != info["total_frames"]
                or eps.dataset_from_index.tolist() != np.r_[0, np.cumsum(eps.length)[:-1]].tolist()
                or eps.dataset_to_index.tolist() != np.cumsum(eps.length).tolist()):
            raise ValueError("episode metadata coverage is inconsistent")
        episode = {k: (int(v) if float(v).is_integer() else float(v))
                   for k, v in eps.iloc[args.episode_index].to_dict().items()}
        if episode["episode_index"] != args.episode_index:
            raise ValueError("requested episode not found")
        data_path = info["data_path"].format(chunk_index=episode["data/chunk_index"],
                                              file_index=episode["data/file_index"])
        data = pd.read_parquet(fetch(data_path))
        rows = data[data.episode_index == args.episode_index].sort_values("frame_index")
        records = json.loads(rows.to_json(orient="records", double_precision=15))
        task_by_id = {r["source_task_index"]: r["task_instruction"] for r in source_tasks}
        for row in records:
            row["task"] = task_by_id[row["task_index"]]
        adapted = validate_episode_rows(records, episode, registry, info["fps"])
        write_json(out / "selected_episode_records.json", records)
        video_paths = [fetch(info["video_path"].format(video_key=key,
                        chunk_index=episode[f"videos/{key}/chunk_index"],
                        file_index=episode[f"videos/{key}/file_index"])) for key in IMAGE_KEYS]
        decoded = decode_views(video_paths, episode, info["fps"], len(records), out)
        probe_records = [{k: row[k] for k in ("episode_index", "frame_index", "task_index", "task",
                                             "observation.state", "action")} for row in records[:7]]
        write_json(out / "probe_records.json", probe_records)
        selected = {"episode_index": args.episode_index, "task_id": adapted[0]["task_id"],
                    "source_task_index": records[0]["task_index"], "record_count": len(records),
                    "complete_episode": True, "metadata": episode,
                    "completeness_scope": "all declared rows and both videos decoded; upstream hidden-reset/physical-time provenance not certified"}
        report = {"schema": "pi05_libero_public_source_audit_v1", "status": "passed",
                  "source": {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": hub.sha},
                  "runtime_seconds": round(time.monotonic() - started, 3), "files": files,
                  "download_bytes": downloaded, "whole_repository_bytes": sum(s.size for s in hub.siblings),
                  "metadata": {"episodes": len(eps), "frames": info["total_frames"], "tasks": len(tasks), "fps": info["fps"]},
                  "task_registry": registry, "selected_episode": selected,
                  "native_benchmark_source_sha256": sha256_file(Path(inspect.getfile(benchmark))),
                  "probe": {"frame_indices": list(range(7)), "row_count": 7,
                            "images_path": "probe_images.npy", "images_sha256": sha256_file(out / "probe_images.npy"),
                            "records_path": "probe_records.json", "records_sha256": sha256_file(out / "probe_records.json"),
                            "orientation": "stored_dataset_rgb_no_extra_flip"},
                  "orientation_basis": "published LeRobot dataset convention; consume stored demo RGB without environment raw-camera flip",
                  "orientation_reference": "https://huggingface.co/docs/lerobot/env_processor",
                  "timing_scope": "stored 10Hz record/video consistency, not independently verified physical control cadence",
                  "decoded_view_frames": [d["frame_count"] for d in decoded],
                  "training_ready": False, "feature_pack_complete": False, "policy_loaded": False,
                  "optimizer_steps": 0, "simulation_steps": 0, "visual_status": "not_viewed",
                  "upstream_converter_provenance_verified": False,
                  "evaluation_outputs_used": False,
                  "code_sha256": sha256_file(Path(__file__))}
        report["output_sha256"] = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file() and p.name != "status.json"}
        write_json(out / "report.json", report)
        write_json(out / "status.json", {"status": "completed", "report_sha256": sha256_file(out / "report.json")})
        print(json.dumps({"status": "passed", "episode": selected, "download_bytes": downloaded,
                          "mapping_source_ids": [r["source_task_index"] for r in registry],
                          "runtime_seconds": report["runtime_seconds"]}), flush=True)
    except Exception:
        write_json(out / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=1400,
                        help="Fixed diagnostic choice, never selected by reward/success")
    parser.add_argument("--libero-config", type=Path, default=Path("simulation_output/libero_runtime_config_v1"))
    args = parser.parse_args()
    if not 0 <= args.episode_index < 1693:
        parser.error("episode index out of pinned metadata range")
    run(args)
