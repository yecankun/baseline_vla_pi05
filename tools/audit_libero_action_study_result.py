"""Read back the fixed completed source audit; never decode, extract or train.

Source files are read-only. Optional exclusive outputs are separate siblings.
The small review bundle contains metadata and existing representative PNGs only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from libero_action_study_source_checks import audit_sequences
from prepare_libero_action_study_sources import (
    ROOT, IMAGE_KEYS, dump, json_lines, prepare, read, require, sha256_file,
    validate_episode_rows, verified,
)

REPORT_SHA = "20d34377c636469e2d06a61cc7519db6fb0ee6c33379d9631b058870d101ef5c"
DUPLICATE_SHA = "0dafcedc14abc892145207b0ec17e1ba47aed7f244f8f41b039647d4040da6b4"


def audit():
    import pyarrow.parquet as pq
    from PIL import Image

    started = time.monotonic()
    context = prepare(ROOT)  # Offline pinned source/index/code rehash only.
    root = Path(context["identity"]["output"])
    report = read(verified(root, "report.json", REPORT_SHA))
    duplicate = read(verified(root, "duplicate_audit.json", DUPLICATE_SHA))
    require(report["identity"] == context["identity"] == read(root / "identity.json"), "identity drift")
    status = read(root / "status.json")
    require(status["status"] == status["phase"] == "completed", "not completed")
    require(status["report_sha256"] == REPORT_SHA, "status report hash drift")
    require(not (root / "active_invocation.lock").exists(), "active invocation lock remains")
    require(not list(root.rglob("*.part")), "unexpected partial files; review separately")
    for name, digest in report["output_sha256"].items():
        verified(root, name, digest)
    require(report["status"] == "passed_declared_source_checks", "source gate did not pass")
    require(report["duplicate_audit"] == duplicate, "embedded duplicate audit differs")
    require(report["selected_split"] == context["selection"]["split"], "selection changed")

    downloads = read(root / "download_report.json")
    expected = {e["path"]: e for e in context["missing"]}
    require(len(downloads["files"]) == len(expected) == 12, "download count")
    require({e["path"] for e in downloads["files"]} == set(expected), "download allowlist differs")
    for entry in downloads["files"]:
        require(entry == {**expected[entry["path"]], "reused": False}, "download inventory differs")
        verified(root / "source", entry["path"], entry["lfs_sha256"], entry["size"])
    attempts = json_lines(root / "download_attempts.jsonl")
    reservations = json_lines(root / "budget_reservations.jsonl")
    require(len(attempts) == len(reservations) == 12, "unexpected retries; review separately")
    require(Counter(a["path"] for a in attempts) == Counter(expected.keys()), "attempt paths differ")
    require(Counter(r["path"] for r in reservations) == Counter(expected.keys()), "reservation paths differ")
    for attempt in attempts:
        require(attempt["attempt"] == 1 and attempt["error_type"] is None, "failed/retried attempt")
        require(attempt["received_bytes"] == expected[attempt["path"]]["size"], "received bytes differ")
    for reservation in reservations:
        require(reservation["attempt"] == 1 and reservation["reserved_bytes"] == expected[reservation["path"]]["size"], "reservation differs")
    for record in (report, status, downloads):
        require(record["cumulative_response_payload_bytes"] == record["conservative_reserved_bytes"] == 230354490, "accounting total differs")

    choices = context["selection"]["episodes"] + [
        {"episode_index": 1400, "partition": "old_development", "record_count": 140},
        {"episode_index": 1402, "partition": "old_development", "record_count": 173},
    ]
    require(len(report["episodes"]) == 14, "episode count")
    info = read(context["source_paths"]["meta/info.json"])
    registry = [r for r in context["prior"]["task_registry"] if r["task_id"] == 9]
    sequences, summaries = {}, []
    png_count = 0
    for episode, choice in zip(report["episodes"], choices):
        eid, count = choice["episode_index"], choice["record_count"]
        require(all(episode[k] == choice[k] for k in ("episode_index", "partition", "record_count")), "episode choice differs")
        require(episode["complete_episode"] is True, "incomplete episode")
        rows = read(root / episode["records_path"])
        md = episode["metadata"]
        validate_episode_rows(rows, md, registry, info["fps"])
        data_name = info["data_path"].format(chunk_index=int(md["data/chunk_index"]), file_index=int(md["data/file_index"]))
        raw = pq.read_table(context["source_paths"][data_name], columns=context["auth"]["projected_source_columns"], filters=[("episode_index", "=", eid)]).to_pylist()
        raw.sort(key=lambda r: r["frame_index"])
        for row in raw:
            row["task"] = registry[0]["task_instruction"]
        require(raw == rows and len(rows) == count, "saved rows differ from pinned raw source")
        state = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
        action = np.asarray([r["action"] for r in rows], dtype=np.float32)
        frames = read(root / episode["decoded_frames_path"])
        require(len(frames) == 2, "two views required")
        array = np.load(root / episode["images_path"], mmap_mode="r", allow_pickle=False)
        require(array.shape == (count, 2, 256, 256, 3) and array.dtype == np.uint8, "RGB array schema drift")
        max_error = 0.0
        for view, key in enumerate(IMAGE_KEYS):
            evidence = frames[view]
            start, stop = md[f"videos/{key}/from_timestamp"], md[f"videos/{key}/to_timestamp"]
            require(evidence["key"] == key and evidence["from_timestamp"] == start and evidence["to_timestamp"] == stop, "video interval mismatch")
            require(len(evidence["frames"]) == count, "decoded frame count")
            for i, frame in enumerate(evidence["frames"]):
                error = abs(frame["video_timestamp"] - (start + i / 10))
                max_error = max(max_error, error)
                require(frame["frame_index"] == i and error <= .001 and frame["video_timestamp"] < stop - .001, "PTS sequence or half-open boundary mismatch")
                require(hashlib.sha256(array[i, view].tobytes()).hexdigest() == frame["decoded_rgb_sha256"], "RGB hash mismatch")
                if i in {0, count // 2, count - 1}:
                    png = (root / episode["images_path"]).parent / f"view{view}_frame{i:04d}.png"
                    with Image.open(png) as image:
                        require(image.mode == "RGB" and np.array_equal(np.asarray(image), array[i, view]), "representative PNG differs")
                    png_count += 1
        hashes = [frames[0]["frames"][i]["decoded_rgb_sha256"] + frames[1]["frames"][i]["decoded_rgb_sha256"] for i in range(count)]
        sequences[eid] = state, action, hashes
        summaries.append({"episode_index": eid, "partition": episode["partition"], "rows": count, "rgb_frames": count * 2, "max_video_grid_error_seconds": max_error})
        del array
    recalculated = audit_sequences(report["episodes"], sequences, known_family_links=context["auth"]["known_family_links"])
    require(recalculated == duplicate, "recomputed all-pair evidence differs")
    require(duplicate["pair_count"] == 91 and duplicate["blocking_pair_count"] == 0, "pair gate mismatch")
    for record in (report, status):
        require(record["features_extracted"] == record["optimizer_steps"] == 0, "unexpected features or training")
    for field in ("training_ready", "formal_data_allowed", "family_independence_verified", "normalization_fitted", "policy_loaded", "new_ids_replaced"):
        require(report[field] is False, f"scope flag changed: {field}")
    require(sha256_file(root / "report.json") == REPORT_SHA, "report changed during readback")
    return {
        "schema": "libero_action_study_completed_result_readback_v1",
        "status": "passed_source_hash_rows_rgb_and_pair_readback_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": REPORT_SHA, "duplicate_audit_sha256": DUPLICATE_SHA,
        "audit_tool_sha256": sha256_file(Path(__file__)),
        "verified_output_files": len(report["output_sha256"]),
        "verified_input_files": len(context["identity"]["input_sha256"]),
        "verified_code_files": len(context["identity"]["code_sha256"]),
        "new_videos_verified": 12, "old_cached_videos_verified": 2,
        "successful_attempts": 12, "failed_attempts": 0, "retry_attempts": 0,
        "received_and_reserved_bytes_each": 230354490,
        "episodes": summaries, "complete_rows": 1971, "verified_rgb_frames": 3942,
        "verified_representative_pngs": png_count, "pair_count": 91,
        "blocking_pair_count": duplicate["blocking_pair_count"],
        "same_partition_groups": duplicate["same_partition_groups"],
        "pair_relationship_counts": dict(Counter(p["relationship"] for p in duplicate["pairs"])),
        "source_runtime_seconds": report["runtime_seconds"],
        "source_final_status_runtime_seconds": status["runtime_seconds"],
        "runtime_seconds": time.monotonic() - started,
        "visual_status": "not_viewed", "source_modified": False,
        "new_video_decode": False, "features_extracted": 0, "optimizer_steps": 0,
        "training_ready": False, "formal_data_allowed": False,
        "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--review-bundle", type=Path)
    args = parser.parse_args()
    root = ROOT / "simulation_output/libero_action_study_sources_v1"
    for path in (args.out, args.review_bundle):
        if path:
            require(path.resolve().parent == root.parent and not path.exists(), "output must be a fresh sibling of immutable source directory")
    result = audit()
    if args.review_bundle:
        members = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".png"} and "source" not in p.relative_to(root).parts)
        require(sum(p.stat().st_size for p in members) < 90_000_000, "review bundle exceeds small transfer scope")
        with args.review_bundle.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path in members:
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        result["review_bundle"] = {"path": str(args.review_bundle), "size": args.review_bundle.stat().st_size, "sha256": sha256_file(args.review_bundle), "member_count": len(members), "excludes": ["MP4", "NPY", "Parquet"]}
    if args.out:
        dump(args.out, result, exclusive=True)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
