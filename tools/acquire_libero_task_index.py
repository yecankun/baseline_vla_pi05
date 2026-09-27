"""Bounded public task-index acquisition, never video, action/state analysis or training."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
import traceback

from plan_pi05_libero_action_ablation import COLS, META, PLAN_SHA, integral, read, select_split, sha, task_inventory

AUTH_SHA = "94fd464a73f338fe84e719ce7016b47f4865d9a11c64fdfab6d1b5e85d3bd016"
PLANNER_SHA = "51457b29735ee5aef46649b367ab9fc3572a283967a3507c3ecf4a229887ac2f"
VIEWS = ("observation.images.image", "observation.images.image2")
META_COLUMNS = ["episode_index", "length", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"] + [
    f"videos/{view}/{field}" for view in VIEWS for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]


def dump(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_path(root, name):
    if type(name) is not str or "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("unsafe relative artifact path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes root")
    return path


def checked(root, name, digest):
    path = safe_path(root, name)
    if sha(path) != digest:
        raise ValueError(f"pinned file hash mismatch: {name}")
    return path


def validate_entry(entry, data_only=True):
    pattern = r"data/chunk-\d{3}/file-\d{3}\.parquet" if data_only else r"(?:data/chunk-\d{3}/file-\d{3}\.parquet|videos/observation\.images\.image2?/chunk-\d{3}/file-\d{3}\.mp4)"
    if (type(entry) is not dict or re.fullmatch(pattern, entry.get("path", "")) is None
            or type(entry.get("size")) is not int or entry["size"] <= 0
            or (data_only and entry["size"] > 100_000_000)
            or type(entry.get("lfs_sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}", entry["lfs_sha256"]) is None):
        raise ValueError("safe path, positive bounded integer size and LFS SHA256 required")


def budget_entries(names, inventory, max_bytes=50_000_000):
    if len(names) != len(set(names)):
        raise ValueError("duplicate acquisition paths")
    entries = []
    for name in sorted(names):
        if name not in inventory or inventory[name].get("path") != name:
            raise ValueError("missing or mismatched inventory path")
        entry = inventory[name]
        validate_entry(entry)
        entries.append(entry)
    total = sum(e["size"] for e in entries)
    if total > max_bytes:
        raise ValueError("total acquisition byte budget exceeded")
    return entries, total


def prepare(args):
    if args.authorization_sha256 != AUTH_SHA or sha(args.authorization) != AUTH_SHA:
        raise ValueError("hard-pinned current acquisition authorization required")
    auth = read(args.authorization)
    if auth["execution_authorized"] is not True or sha(args.plan) != PLAN_SHA or auth["study_plan_sha256"] != PLAN_SHA:
        raise ValueError("authorization/study plan mismatch")
    if sha(Path(__file__).with_name("plan_pi05_libero_action_ablation.py")) != PLANNER_SHA:
        raise ValueError("original selector implementation drift")
    root = args.source_root.resolve()
    prior = read(checked(root, "report.json", auth["source_audit_report_sha256"]))
    inventory = read(checked(root, "hub_inventory.json", auth["hub_inventory_sha256"]))
    if sha(args.cached_report) != auth["cached_projection_report_sha256"]:
        raise ValueError("cached metadata report mismatch")
    cached = read(args.cached_report)
    if prior["source"] != cached["source"] or prior["source"]["revision"] != auth["source_revision"]:
        raise ValueError("pinned source identity mismatch")
    files = {e["path"]: e for e in prior["files"]}
    fixed = {"report.json": auth["source_audit_report_sha256"], "hub_inventory.json": auth["hub_inventory_sha256"]}
    for name in ("meta/info.json", META, "data/chunk-000/file-330.parquet"):
        entry = files[name]
        checked(root, entry["local_path"], entry["sha256"])
        if inventory[name]["size"] != entry["size"] or inventory[name]["lfs_sha256"] not in (None, entry["sha256"]):
            raise ValueError("cache audit and immutable inventory differ")
        fixed[entry["local_path"]] = entry["sha256"]
    entries, total = budget_entries(cached["missing_metadata_projection_shards"], inventory)
    if (len(entries) != auth["new_file_count"] or total != auth["new_unique_file_bytes"]
            or max(e["size"] for e in entries) != auth["maximum_actual_file_bytes"]):
        raise ValueError("acquisition shape/byte budget differs from user-approved scope")
    return auth, read(args.plan), prior, inventory, fixed, entries, total


def validate_metadata(episodes, total_episodes, total_frames):
    rows = sorted(episodes, key=lambda e: integral(e["episode_index"]))
    offset = 0
    for i, row in enumerate(rows):
        count = integral(row["length"])
        if (integral(row["episode_index"]) != i or count <= 0
                or integral(row["dataset_from_index"]) != offset
                or integral(row["dataset_to_index"]) != offset + count):
            raise ValueError("global episode/index coverage gap or overlap")
        offset += count
        for view in VIEWS:
            start, stop = row[f"videos/{view}/from_timestamp"], row[f"videos/{view}/to_timestamp"]
            if (not math.isfinite(start) or not math.isfinite(stop) or start < 0
                    or abs(stop - start - count / 10.0) > 2e-5):
                raise ValueError("video interval metadata differs from complete episode length")
    if len(rows) != total_episodes or offset != total_frames:
        raise ValueError("declared global episode/frame totals differ")


def validate_shard_rows(name, rows, by_id, info):
    for row in rows:
        if set(row) != set(COLS):
            raise ValueError("only five projected index columns allowed")
        eid = integral(row["episode_index"])
        if eid not in by_id:
            raise ValueError("projected episode absent from metadata")
        md = by_id[eid]
        expected = info["data_path"].format(chunk_index=integral(md["data/chunk_index"]),
                                             file_index=integral(md["data/file_index"]))
        if expected != name:
            raise ValueError("projected episode does not belong to metadata-declared shard")


class Downloads:
    def __init__(self, out, auth, consumed=0, reserved=0):
        self.out, self.auth = out.resolve(), auth
        self.lock, self.bytes, self.stop = threading.Lock(), consumed, threading.Event()
        self.reserved = reserved

    def reserve(self, entry, attempt):
        # Durable conservative accounting before the request; interrupted or
        # failed attempts never release their full pinned-size reservation.
        with self.lock:
            if self.reserved + entry["size"] > self.auth["max_cumulative_payload_response_bytes"]:
                self.stop.set()
                raise ValueError("cumulative request reservation budget exceeded")
            with (self.out / "budget_reservations.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"path": entry["path"], "attempt": attempt,
                                      "reserved_bytes": entry["size"]}) + "\n")
                log.flush()
                os.fsync(log.fileno())
            self.reserved += entry["size"]

    def fetch(self, entry):
        validate_entry(entry)
        target = safe_path(self.out / "source", entry["path"])
        if target.exists():
            if target.stat().st_size != entry["size"] or sha(target) != entry["lfs_sha256"]:
                raise ValueError("corrupt existing cache; refuse silent replacement")
            return {"path": entry["path"], "size": entry["size"], "sha256": entry["lfs_sha256"], "reused": True}
        import requests
        url = f"https://huggingface.co/datasets/{self.auth['source_repo_id']}/resolve/{self.auth['source_revision']}/{entry['path']}"
        for attempt in range(1, self.auth["max_attempts_per_file_per_invocation"] + 1):
            if self.stop.is_set():
                raise RuntimeError("acquisition cancelled after another file failed")
            self.reserve(entry, attempt)
            received, error, part = 0, None, None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=target.name+".", suffix=".part", delete=False) as stream:
                    part = Path(stream.name)
                    with requests.get(url, stream=True, timeout=(10, 30)) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(65536):
                            received += len(chunk)
                            with self.lock:
                                self.bytes += len(chunk)
                                if self.bytes > self.auth["max_cumulative_payload_response_bytes"]:
                                    self.stop.set()
                                    raise ValueError("cumulative response payload budget exceeded")
                            if received > entry["size"]:
                                raise ValueError("server response exceeded pinned file size")
                            stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                if received != entry["size"] or sha(part) != entry["lfs_sha256"]:
                    raise ValueError("download size/SHA mismatch; keep part, do not publish")
                os.link(part, target)  # atomic, refuses any existing target
                part.unlink()
                return {"path": entry["path"], "size": received, "sha256": entry["lfs_sha256"], "reused": False}
            except requests.RequestException as exc:
                error = type(exc).__name__
                if attempt == self.auth["max_attempts_per_file_per_invocation"]:
                    raise RuntimeError(f"bounded transport attempts exhausted: {entry['path']}") from exc
                time.sleep(0.5)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                with self.lock:
                    with (self.out / "download_attempts.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"path": entry["path"], "attempt": attempt, "received_bytes": received,
                            "error_type": error, "part_path": str(part.relative_to(self.out)) if part else None,
                            "cumulative_response_bytes": self.bytes}) + "\n")


def selected_payloads(selection, episodes, inventory, info):
    by_id = {integral(e["episode_index"]): e for e in episodes}
    records, unique, unique_data = [], {}, {}
    for choice in selection["episodes"]:
        md = by_id[choice["episode_index"]]
        data = info["data_path"].format(chunk_index=integral(md["data/chunk_index"]), file_index=integral(md["data/file_index"]))
        validate_entry(inventory[data])
        unique_data[data] = inventory[data]
        videos = []
        for view in VIEWS:
            name = info["video_path"].format(video_key=view, chunk_index=integral(md[f"videos/{view}/chunk_index"]),
                                              file_index=integral(md[f"videos/{view}/file_index"]))
            validate_entry(inventory[name], data_only=False)
            unique[name] = inventory[name]
            videos.append({"path": name, "view": view, "from_timestamp": md[f"videos/{view}/from_timestamp"],
                           "to_timestamp": md[f"videos/{view}/to_timestamp"]})
        records.append({**choice, "data_path": data, "videos": videos,
            "source_trajectory_id": f"lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4/episode{choice['episode_index']}"})
    for i, left in enumerate(records):
        for right in records[i+1:]:
            if left["partition"] != right["partition"]:
                for a, b in zip(left["videos"], right["videos"]):
                    if a["path"] == b["path"] and min(a["to_timestamp"], b["to_timestamp"]) - max(a["from_timestamp"], b["from_timestamp"]) > 1e-6:
                        raise ValueError("cross-split video interval metadata overlaps")
    return {"episodes": records, "unique_video_files": list(unique.values()),
            "unique_video_bytes": sum(v["size"] for v in unique.values()),
            "unique_data_paths": sorted({r["data_path"] for r in records}),
            "unique_data_files": list(unique_data.values()),
            "unique_data_bytes": sum(v["size"] for v in unique_data.values()),
            "data_files_already_hash_verified_by_index_stage": True,
            "additional_data_download_bytes_required": 0,
            "video_files_requiring_user_transfer": [v["path"] for v in unique.values() if v["size"] > 100_000_000],
            "video_download_authorized": False, "images_decoded": 0, "source_duplicate_audit_passed": False,
            "source_acceptance_pending": True, "family_independence_verified": False}


def run(args):
    auth, plan, prior, inventory, fixed, entries, total = prepare(args)
    if args.out.name != auth["output_directory_name"]:
        raise ValueError("fixed index-acquisition output identity required")
    identity = {"authorization_sha256": AUTH_SHA, "plan_sha256": PLAN_SHA,
        "code_sha256": sha(__file__), "selector_sha256": PLANNER_SHA, "bound_source_files": fixed,
        "cached_projection_sha256": auth["cached_projection_report_sha256"],
        "new_files": entries, "new_unique_file_bytes": total}
    if args.stage == "budget":
        print(json.dumps({"authorization_sha256": AUTH_SHA, "plan_sha256": PLAN_SHA,
            "code_sha256": identity["code_sha256"], "new_file_count": len(entries),
            "new_unique_file_bytes": total, "maximum_file_bytes": max(e["size"] for e in entries),
            "acquisition_started": False, "workers": auth["workers"], "interpreted_columns": COLS}), flush=True)
        return
    if args.execute_index_acquisition is not True:
        raise ValueError("explicit --execute-index-acquisition required")
    if args.resume:
        if read(args.out / "identity.json") != identity or read(args.out / "status.json")["status"] != "failed":
            raise ValueError("only same-identity failed acquisition can explicitly resume")
        if any((args.out / name).exists() for name in ("task_inventory.json", "selection.json", "selected_payload_inventory.json", "report.json")):
            raise ValueError("selection/result already materialized; no acquisition resume")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        dump(args.out / "identity.json", identity, exclusive=True)
    consumed = sum(reads["received_bytes"] for reads in (json.loads(line) for line in
        (args.out / "download_attempts.jsonl").read_text().splitlines())) if (args.out / "download_attempts.jsonl").exists() else 0
    reserved = sum(json.loads(line)["reserved_bytes"] for line in
        (args.out / "budget_reservations.jsonl").read_text().splitlines()) if (args.out / "budget_reservations.jsonl").exists() else 0
    if consumed > reserved or reserved > auth["max_cumulative_payload_response_bytes"]:
        raise ValueError("download accounting is inconsistent; no resume")
    downloader = Downloads(args.out, auth, consumed, reserved)
    start = time.monotonic()
    selection = None
    dump(args.out / "status.json", {"status": "running", "pid": os.getpid(), "optimizer_steps": 0})
    try:
        outcomes = []
        with ThreadPoolExecutor(max_workers=auth["workers"]) as pool:
            futures = {pool.submit(downloader.fetch, entry): entry["path"] for entry in entries}
            for future in as_completed(futures):
                try:
                    outcomes.append(future.result())
                except BaseException:
                    downloader.stop.set()
                    for pending in futures:
                        pending.cancel()
                    raise
                if len(outcomes) % 25 == 0 or len(outcomes) == len(entries):
                    status = {"status": "running", "pid": os.getpid(), "verified_files": len(outcomes),
                        "required_files": len(entries), "cumulative_response_bytes": downloader.bytes,
                        "runtime_seconds": time.monotonic()-start, "optimizer_steps": 0}
                    dump(args.out / "status.json", status)
                    print("TASK_INDEX_ACQUIRE", len(outcomes), "/", len(entries), flush=True)
        import pyarrow.parquet as pq
        files = {e["path"]: e for e in prior["files"]}
        info = read(args.source_root / files["meta/info.json"]["local_path"])
        if info["fps"] != 10 or info["total_episodes"] != 1693 or info["total_frames"] != 273465:
            raise ValueError("pinned source dimensions/fps mismatch")
        episodes = pq.read_table(args.source_root / files[META]["local_path"], columns=META_COLUMNS).to_pylist()
        validate_metadata(episodes, 1693, 273465)
        expected_paths = {info["data_path"].format(chunk_index=integral(e["data/chunk_index"]), file_index=integral(e["data/file_index"])) for e in episodes}
        if expected_paths != {e["path"] for e in entries} | {"data/chunk-000/file-330.parquet"}:
            raise ValueError("acquired shard set does not cover entire pinned metadata")
        by_id, rows = {integral(e["episode_index"]): e for e in episodes}, []
        for name in sorted(expected_paths):
            root = args.source_root / "source" if name == "data/chunk-000/file-330.parquet" else args.out / "source"
            entry = inventory[name]
            path = checked(root, name, entry["lfs_sha256"])
            shard_rows = pq.read_table(path, columns=COLS).to_pylist()
            validate_shard_rows(name, shard_rows, by_id, info)
            rows.extend(shard_rows)
        task_rows, missing = task_inventory(episodes, rows, 10.0)
        if missing or len(rows) != 273465:
            raise ValueError("incomplete full task inventory; no selection")
        selection = select_split(task_rows, [], plan)
        if selection["status"] != "metadata_split_selected_payload_audit_pending":
            raise ValueError("frozen prospective selection cannot be completed")
        payloads = selected_payloads(selection, episodes, inventory, info)
        for name, digest in fixed.items():
            checked(args.source_root, name, digest)
        if sha(__file__) != identity["code_sha256"] or sha(args.plan) != PLAN_SHA or sha(args.authorization) != AUTH_SHA:
            raise ValueError("implementation or protocol changed during acquisition")
        dump(args.out / "task_inventory.json", task_rows, exclusive=True)
        dump(args.out / "selection.json", selection, exclusive=True)
        dump(args.out / "selected_payload_inventory.json", payloads, exclusive=True)
        report = {"schema": "libero_action_study_full_task_index_v1", "status": "metadata_selection_complete_source_audit_pending",
            **identity, "source": prior["source"], "interpreted_columns": COLS,
            "network_semantics": "full Parquet payloads transferred; action/state columns not decoded/interpreted",
            "metadata_episode_count": len(episodes), "projected_frames": len(rows), "task_known_episodes": len(task_rows),
            "missing_task_count": 0, "source_task39_total": sum(e["source_task_index"] == 39 for e in task_rows),
            "selection": selection, "payload_inventory": payloads,
            "downloaded_unique_files_this_invocation": sum(not o["reused"] for o in outcomes),
            "reused_verified_files_this_invocation": sum(o["reused"] for o in outcomes),
            "cumulative_response_payload_bytes": downloader.bytes, "files": sorted(outcomes, key=lambda o:o["path"]),
            "conservative_request_reserved_bytes": downloader.reserved,
            "runtime_seconds": time.monotonic()-start, "actions_states_interpreted": False,
            "video_downloads": 0, "images_decoded": 0, "features_extracted": 0, "optimizer_steps": 0,
            "training_ready": False, "formal_data_allowed": False, "family_independence_verified": False,
            "checkpoint_training_overlap_unknown": True,
            "output_sha256": {name: sha(args.out / name) for name in ("identity.json", "task_inventory.json", "selection.json", "selected_payload_inventory.json", "download_attempts.jsonl", "budget_reservations.jsonl")}}
        dump(args.out / "report.json", report, exclusive=True)
        dump(args.out / "status.json", {"status": "completed", "report_sha256": sha(args.out / "report.json"), "optimizer_steps": 0})
        print("TASK_INDEX_COMPLETE", json.dumps(selection["split"]), sha(args.out / "report.json"), flush=True)
    except BaseException as error:
        dump(args.out / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
            "cumulative_response_payload_bytes": downloader.bytes, "runtime_seconds": time.monotonic()-start,
            "conservative_request_reserved_bytes": downloader.reserved, "metadata_selection_if_computed": selection,
            "optimizer_steps": 0, "resume_requires_explicit_flag": True})
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "cached-report", "plan", "authorization", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--stage", choices=("budget", "acquire"), required=True)
    parser.add_argument("--execute-index-acquisition", action="store_true")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
