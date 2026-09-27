"""Fixed public source acquisition and integrity audit; never features or training.

Full job is user-run. --stage preflight is offline and safe to repeat without
--preflight-out. --resume applies only to failed download-phase invocations.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from acquire_libero_task_index import Downloads, safe_path
from audit_libero_public_source import validate_episode_rows
from pi05_libero_world_model_adapter import IMAGE_KEYS, sha256_file
from prepare_libero_source_pair import decode_episode
from plan_pi05_libero_action_ablation import COLS, META, integral, read, select_split

ROOT = Path(__file__).resolve().parents[1]
AUTH_SHA = "792ace5e0df13f826154ff3a66330331fff3a532619daf6df7e64e695a54c0d6"
DEPENDENCIES = {
    "acquire_libero_task_index.py": "dbe9fd1a9de1054e7924baec3cc79ac488a01dfc6e6ae6aae97801650cfd85ac",
    "prepare_libero_source_pair.py": "5e7169f3039677752955931851be458e1fb41f60b54cb2d48947b0fd60b09d89",
    "audit_libero_public_source.py": "2b944cf131a2338fd5e9c7b31d3360372ed4a8be40399a745f6c6d896e5a740c",
    "pi05_libero_world_model_adapter.py": "924cfe56e8b1d6653b86073d08988f09491d6c02b40439965315d492de37fab0",
    "probe_pi05_libero_public_features.py": "fefef839ce8ae7d7826cb1d52a2c9827a96cd7e2035c4df69339601461812f75",
    "plan_pi05_libero_action_ablation.py": "51457b29735ee5aef46649b367ab9fc3572a283967a3507c3ecf4a229887ac2f",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verified(root, name, digest, size=None):
    path = safe_path(root, name)
    require(path.is_file() and sha256_file(path) == digest, f"pinned hash mismatch: {name}")
    require(size is None or path.stat().st_size == size, f"pinned size mismatch: {name}")
    return path


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def video_entry(entry):
    require(type(entry) is dict and type(entry.get("path")) is str
            and re.fullmatch(r"videos/observation\.images\.image2?/chunk-\d{3}/file-\d{3}\.mp4", entry["path"]),
            "only immutable native video shard paths allowed")
    require(type(entry.get("size")) is int and 0 < entry["size"] <= 100_000_000,
            "video exceeds single-file boundary or has invalid size")
    require(type(entry.get("lfs_sha256")) is str and re.fullmatch(r"[0-9a-f]{64}", entry["lfs_sha256"]), "video SHA required")


class VideoDownloads(Downloads):
    """Separate video allowlist; reuse only the tested durable reservation helper."""

    def __init__(self, out, auth, allowed, consumed=0, reserved=0):
        super().__init__(out, auth, consumed, reserved)
        self.allowed = {e["path"]: e for e in allowed}
        require(len(self.allowed) == len(allowed), "duplicate video acquisition path")
        for entry in allowed:
            video_entry(entry)

    def fetch(self, entry):
        video_entry(entry)
        require(entry == self.allowed.get(entry["path"]), "video is outside frozen acquisition inventory")
        target = safe_path(self.out / "source", entry["path"])
        if target.exists():
            verified(self.out / "source", entry["path"], entry["lfs_sha256"], entry["size"])
            return {**entry, "reused": True}
        import requests
        url = f"https://huggingface.co/datasets/{self.auth['source_repo_id']}/resolve/{self.auth['source_revision']}/{entry['path']}"
        for attempt in range(1, self.auth["max_attempts_per_file_per_invocation"] + 1):
            if self.stop.is_set():
                raise RuntimeError("cancelled after another video failed")
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
                                require(self.bytes <= self.auth["max_cumulative_payload_response_bytes"], "response byte budget exceeded")
                            require(received <= entry["size"], "response exceeded pinned video size")
                            stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                require(received == entry["size"] and sha256_file(part) == entry["lfs_sha256"], "video size/SHA mismatch; preserve part")
                os.link(part, target)  # refuses existing destination, including races
                part.unlink()
                return {**entry, "reused": False}
            except requests.RequestException as exc:
                error = type(exc).__name__
                if attempt == self.auth["max_attempts_per_file_per_invocation"]:
                    raise RuntimeError(f"bounded video transport attempts exhausted: {entry['path']}") from exc
                time.sleep(0.5)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                with self.lock:
                    with (self.out / "download_attempts.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"path": entry["path"], "attempt": attempt, "received_bytes": received,
                            "error_type": error, "part_path": part.relative_to(self.out).as_posix() if part else None}) + "\n")
                        log.flush()
                        os.fsync(log.fileno())


def prepare(repo):
    auth_path = repo / "docs/libero-action-study-source-authorization-v1.json"
    require(sha256_file(auth_path) == AUTH_SHA, "source-scope authorization mismatch")
    auth = read(auth_path)
    plan_path = verified(repo, "docs/libero-action-ablation-study-plan-v1.json", auth["study_plan_sha256"])
    plan = read(plan_path)
    base = repo / "simulation_output/libero_public_source_audit_v1"
    index_root = repo / "simulation_output/libero_action_study_task_index_v1"
    index = read(verified(index_root, "report.json", auth["index_report_sha256"]))
    for name, digest in index["output_sha256"].items():
        verified(index_root, name, digest)
    recovery = read(verified(repo, "simulation_output/libero_action_study_task_index_v1_recovery_audit.json", auth["recovery_audit_sha256"]))
    require(recovery["status"] == "passed_metadata_and_recovery_accounting_only", "task-index recovery audit required")
    prior = read(verified(base, "report.json", auth["base_source_report_sha256"]))
    hub = read(verified(base, "hub_inventory.json", auth["hub_inventory_sha256"]))
    selection = read(verified(index_root, "selection.json", auth["selection_sha256"]))
    require(selection == select_split(read(index_root / "task_inventory.json"), [], plan), "frozen episode selection drift")
    require(index["source"] == prior["source"] == {"kind": "public_demonstrations", "repo_id": auth["source_repo_id"],
                                                  "revision": auth["source_revision"]}, "source identity mismatch")
    old_files = {e["path"]: e for e in prior["files"]}
    source_paths, input_hashes = {}, {}

    def bind(root, name, digest, size=None):
        path = verified(root, name, digest, size)
        input_hashes[str(path)] = digest
        return path

    for path in (auth_path, plan_path, base / "report.json", base / "hub_inventory.json",
                 index_root / "report.json", index_root / "selection.json",
                 repo / "simulation_output/libero_action_study_task_index_v1_recovery_audit.json"):
        input_hashes[str(path.resolve())] = sha256_file(path)
    for name in ("meta/info.json", META, "meta/tasks.parquet", "data/chunk-000/file-330.parquet"):
        entry = old_files[name]
        source_paths[name] = bind(base, entry["local_path"], entry["sha256"], entry["size"])
    payloads = read(index_root / "selected_payload_inventory.json")
    for entry in payloads["unique_data_files"]:
        require(entry == hub[entry["path"]], "data inventory drift")
        source_paths[entry["path"]] = bind(index_root / "source", entry["path"], entry["lfs_sha256"], entry["size"])
    cached, missing = [], []
    for entry in payloads["unique_video_files"]:
        video_entry(entry)
        require(entry == hub[entry["path"]], "video inventory drift")
        if entry["path"] in old_files:
            old = old_files[entry["path"]]
            source_paths[entry["path"]] = bind(base, old["local_path"], entry["lfs_sha256"], entry["size"])
            cached.append(entry)
        else:
            missing.append(entry)
    require(len(missing) == 12 and sum(e["size"] for e in missing) == 230354490, "new video budget drift")
    require(len(cached) == 2 and sum(e["size"] for e in cached) == 43064421, "cached video budget drift")
    require(len(selection["episodes"]) == 12 and sum(e["record_count"] for e in selection["episodes"]) == 1658, "selected row budget drift")
    code_hashes = {}
    for name, digest in DEPENDENCIES.items():
        code_hashes[name] = sha256_file(verified(repo / "tools", name, digest))
    for name in (Path(__file__).name, "libero_action_study_source_checks.py"):
        code_hashes[name] = sha256_file(repo / "tools" / name)
    return {"auth": auth, "plan": plan, "prior": prior, "index": index, "selection": selection,
            "hub": hub, "source_paths": source_paths, "missing": missing, "cached": cached,
            "identity": {"authorization_sha256": AUTH_SHA, "selection_sha256": auth["selection_sha256"],
                "input_sha256": input_hashes, "code_sha256": code_hashes, "new_videos": missing,
                "output": str((repo / "simulation_output" / auth["output_directory_name"]).resolve())}}


def preflight(context):
    import av
    import pyarrow.parquet as pq
    from PIL import Image
    from libero_action_study_source_checks import audit_sequences
    auth = context["auth"]
    out = Path(context["identity"]["output"])
    require(shutil.disk_usage(out.parent).free >= 2_000_000_000, "at least 2GB free required")
    info = read(context["source_paths"]["meta/info.json"])
    require(info["fps"] == 10 and info["total_frames"] == 273465, "source cadence/totals drift")
    for name, path in context["source_paths"].items():
        if name.startswith("data/"):
            require(set(auth["projected_source_columns"]) <= set(pq.read_schema(path).names), "native source schema missing columns")
    return {"schema": "libero_action_study_sources_preflight_v1", "status": "passed_preflight_not_full_source_audit",
            "identity": context["identity"], "selected_split": context["selection"]["split"],
            "selected_new_rows": 1658, "old_development_rows": 313,
            "required_new_video_files": len(context["missing"]), "required_new_video_bytes": 230354490,
            "already_cached_video_files": len(context["cached"]), "already_cached_video_bytes": 43064421,
            "packages": {n: importlib.metadata.version(n) for n in ("av", "pyarrow", "numpy", "pillow", "requests")},
            "network_downloads": 0, "source_action_state_rows_interpreted": 0, "images_decoded": 0,
            "features_extracted": 0, "optimizer_steps": 0, "training_ready": False,
            "full_job_user_run": True, "output_already_exists": out.exists()}


def check_resume(out, identity):
    require(read(out / "identity.json") == identity, "resume identity/code/input drift")
    status = read(out / "status.json")
    require(status["status"] == "failed" and status["phase"] == "download", "only failed download phase may resume")
    require(not any((out / name).exists() for name in ("episodes", "report.json", "download_report.json")),
            "decode/finalization output exists; preserve for review")
    attempts = json_lines(out / "download_attempts.jsonl")
    reservations = json_lines(out / "budget_reservations.jsonl")
    consumed = sum(e["received_bytes"] for e in attempts)
    reserved = sum(e["reserved_bytes"] for e in reservations)
    require(consumed == status["cumulative_response_payload_bytes"] <= reserved <= 500_000_000
            and reserved == status["conservative_reserved_bytes"],
            "incomplete or inconsistent download accounting; no resume")
    attempt_number = len(list(out.glob("prior_failure_*.json"))) + 1
    dump(out / f"prior_failure_{attempt_number:03d}.json", status, exclusive=True)
    return consumed, reserved


def decode_selected(context, out, source_paths):
    import pyarrow.parquet as pq
    from libero_action_study_source_checks import audit_sequences
    auth, selection = context["auth"], context["selection"]
    info = read(source_paths["meta/info.json"])
    metadata = {integral(r["episode_index"]): r for r in pq.read_table(source_paths[META]).to_pylist()}
    registry = [r for r in context["prior"]["task_registry"] if r["task_id"] == 9]
    require(len(registry) == 1 and registry[0]["source_task_index"] == 39, "task mapping drift")
    choices = selection["episodes"] + [{"episode_index": 1400, "partition": "old_development", "record_count": 140},
                                        {"episode_index": 1402, "partition": "old_development", "record_count": 173}]
    episodes, sequences = [], {}
    children = out / "episodes"
    children.mkdir(exist_ok=False)
    for choice in choices:
        eid = choice["episode_index"]
        md = metadata[eid]
        require(integral(md["length"]) == choice["record_count"], "complete episode row count mismatch")
        data_name = info["data_path"].format(chunk_index=integral(md["data/chunk_index"]), file_index=integral(md["data/file_index"]))
        require(data_name in source_paths, "unverified Parquet source")
        rows = pq.read_table(source_paths[data_name], columns=auth["projected_source_columns"],
                             filters=[("episode_index", "=", eid)]).to_pylist()
        rows.sort(key=lambda r: integral(r["frame_index"]))
        for row in rows:
            row["task"] = registry[0]["task_instruction"]
        adapted = validate_episode_rows(rows, md, registry, info["fps"])
        state = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
        action = np.asarray([r["action"] for r in rows], dtype=np.float32)
        require(np.array_equal(state, np.stack([r["state"] for r in adapted]))
                and np.array_equal(action, np.stack([r["action"] for r in adapted])), "native adaptation changed values")
        child = children / f"episode_{eid}"
        child.mkdir(exist_ok=False)
        dump(child / "records.json", rows, exclusive=True)
        videos = [info["video_path"].format(video_key=k, chunk_index=integral(md[f"videos/{k}/chunk_index"]),
                   file_index=integral(md[f"videos/{k}/file_index"])) for k in IMAGE_KEYS]
        require(all(k in source_paths for k in videos), "unverified video source")
        hashes = decode_episode([source_paths[k] for k in videos], md, len(rows), info["fps"], child)
        sequences[eid] = (state, action, hashes)
        episodes.append({"episode_index": eid, "partition": choice["partition"], "metadata": md,
            "record_count": len(rows), "complete_episode": True,
            "source_trajectory_id": f"{auth['source_repo_id']}@{auth['source_revision']}/episode{eid}",
            "images_path": (child / "images.npy").relative_to(out).as_posix(),
            "images_sha256": sha256_file(child / "images.npy"),
            "records_path": (child / "records.json").relative_to(out).as_posix(),
            "records_sha256": sha256_file(child / "records.json"),
            "decoded_frames_path": (child / "decoded_frames.json").relative_to(out).as_posix(),
            "action_change_rows_at_1e_3": int(np.sum(np.any(np.abs(np.diff(action, axis=0)) > 1e-3, axis=1)))})
        print(f"SOURCE_EPISODE_COMPLETE episode={eid} partition={choice['partition']} rows={len(rows)}", flush=True)
    require(sum(e["record_count"] for e in episodes) == 1971, "all-row total mismatch")
    duplicates = audit_sequences(episodes, sequences, known_family_links=auth["known_family_links"])
    dump(out / "duplicate_audit.json", duplicates, exclusive=True)
    return episodes, duplicates


def execute(context, preflight_result, resume=False):
    out = Path(context["identity"]["output"])
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    lock = out / "active_invocation.lock"
    # Exclusive acquisition prevents two explicit resume commands from racing.
    # A hard-kill leaves the lock for operator review; never auto-remove it.
    dump(lock, {"pid": os.getpid(), "code_sha256": sha256_file(Path(__file__))}, exclusive=True)
    try:
        return _execute_locked(context, preflight_result, resume)
    finally:
        lock.unlink()


def _execute_locked(context, preflight_result, resume=False):
    auth, identity = context["auth"], context["identity"]
    out = Path(identity["output"])
    if resume:
        consumed, reserved = check_resume(out, identity)
    else:
        dump(out / "identity.json", identity, exclusive=True)
        dump(out / "preflight.json", preflight_result, exclusive=True)
        consumed, reserved = 0, 0
    downloader = VideoDownloads(out, auth, context["missing"], consumed, reserved)
    started, phase = time.monotonic(), "download"

    def status(kind, **extra):
        dump(out / "status.json", {"status": kind, "phase": phase, "pid": os.getpid(),
             "timestamp_utc": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.monotonic()-started,
             "cumulative_response_payload_bytes": downloader.bytes, "conservative_reserved_bytes": downloader.reserved,
             "features_extracted": 0, "optimizer_steps": 0, **extra})

    status("running")
    try:
        outcomes = []
        with ThreadPoolExecutor(max_workers=auth["workers"]) as pool:
            futures = [pool.submit(downloader.fetch, entry) for entry in context["missing"]]
            for future in as_completed(futures):
                try:
                    outcomes.append(future.result())
                except BaseException:
                    downloader.stop.set()
                    for pending in futures:
                        pending.cancel()
                    raise
                status("running", verified_videos=len(outcomes), required_videos=12)
                print(f"SOURCE_VIDEO_VERIFIED {len(outcomes)}/12", flush=True)
        source_paths = dict(context["source_paths"])
        for entry in context["missing"]:
            source_paths[entry["path"]] = verified(out / "source", entry["path"], entry["lfs_sha256"], entry["size"])
        dump(out / "download_report.json", {"files": sorted(outcomes, key=lambda e:e["path"]),
            "unique_required_bytes": 230354490, "cumulative_response_payload_bytes": downloader.bytes,
            "conservative_reserved_bytes": downloader.reserved}, exclusive=True)
        phase = "source_audit"
        status("running")
        episodes, duplicates = decode_selected(context, out, source_paths)
        for name, digest in identity["input_sha256"].items():
            require(sha256_file(Path(name)) == digest, "source changed during full audit")
        for name, digest in identity["code_sha256"].items():
            require(sha256_file(ROOT / "tools" / name) == digest, "code changed during full audit")
        for entry in context["missing"]:
            verified(out / "source", entry["path"], entry["lfs_sha256"], entry["size"])
        require(duplicates["status"] in {"passed_declared_duplicate_checks_only", "rejected_source_isolation"},
                "unknown duplicate audit decision")
        passed = duplicates["status"] == "passed_declared_duplicate_checks_only"
        require(passed == (duplicates["blocking_pair_count"] == 0), "inconsistent source isolation decision")
        report = {"schema": "libero_action_study_sources_v1", "status": "passed_declared_source_checks" if passed else "rejected_source_overlap",
            "identity": identity, "source": context["index"]["source"], "episodes": episodes,
            "selected_split": context["selection"]["split"], "selected_new_rows": 1658,
            "old_development_rows": 313, "decoded_selected_view_frames": 3942,
            "orientation": auth["orientation"], "duplicate_audit": duplicates,
            "source_integrity_and_declared_duplicate_checks_passed": passed,
            "new_ids_replaced": False, "feature_pack_complete": False, "features_extracted": 0,
            "normalization_fitted": False, "optimizer_steps": 0, "policy_loaded": False,
            "training_ready": False, "formal_data_allowed": False, "family_independence_verified": False,
            "checkpoint_training_overlap_unknown": True, "visual_status": "not_viewed",
            "timing_scope": "stored record/video cadence; not upstream reset or physical timing certification",
            "runtime_seconds": time.monotonic()-started, "cumulative_response_payload_bytes": downloader.bytes,
            "conservative_reserved_bytes": downloader.reserved,
            "output_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob("*")
                              if p.is_file() and p.name not in {"status.json", "active_invocation.lock"}
                              and "source" not in p.relative_to(out).parts}}
        dump(out / "report.json", report, exclusive=True)
        phase = "completed"
        status("completed" if passed else "completed_rejected", report_sha256=sha256_file(out / "report.json"))
        print(json.dumps({"status": report["status"], "report_sha256": sha256_file(out / "report.json"),
                          "runtime_seconds": report["runtime_seconds"], "training_ready": False}), flush=True)
        return 0 if passed else 3
    except BaseException as exc:
        status("failed", error_type=type(exc).__name__, error=str(exc),
               resume_allowed_only_if_download_phase=phase == "download")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run"), required=True)
    parser.add_argument("--execute-authorized-source-audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    args = parser.parse_args()
    if args.stage == "run":
        require(args.execute_authorized_source_audit, "explicit source-audit execution flag required")
    else:
        require(not args.resume and not args.execute_authorized_source_audit, "preflight never executes or resumes")
    context = prepare(ROOT)
    checked = preflight(context)
    if args.stage == "preflight":
        if args.preflight_out:
            dump(args.preflight_out, checked, exclusive=True)
        print(json.dumps(checked), flush=True)
        return 0
    return execute(context, checked, args.resume)


if __name__ == "__main__":
    sys.exit(main())
