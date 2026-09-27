"""Read-only JSON/hash audit of the completed task-index acquisition and recovery.

No network, Parquet decoding, image decoding, model or optimization. The result
is a companion audit; the original report, including its byte-counter omission,
is never rewritten. Full source bytes must be present on the execution host.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from plan_pi05_libero_action_ablation import PLAN_SHA, read, select_split, sha

REPORT_SHA = "ff4be0d04a5a79282b59307a646932f7756ceb0412e9b4e7c4392cabaa3aa781"
EXECUTED_SHA = "b2a5b055341c0c219302c2158f227e13139e18290122121a203b062049f38505"
AUTH_SHA = "94fd464a73f338fe84e719ce7016b47f4865d9a11c64fdfab6d1b5e85d3bd016"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified(root, name, digest, size=None):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "path escapes audit root")
    require(path.is_file() and sha(path) == digest, f"hash mismatch: {name}")
    require(size is None or path.stat().st_size == size, f"size mismatch: {name}")
    return path


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def independent_ids(entries, plan):
    source = plan["source"]
    candidates = []
    for row in entries:
        if (row["source_task_index"] == source["source_task_index"]
                and row["episode_index"] not in source["excluded_episode_indices"]
                and row["record_count"] >= source["minimum_complete_rows"]):
            eid = row["episode_index"]
            key = f"{source['revision']}/{source['native_task_id']}/{eid}/native-action-study-v1"
            candidates.append((hashlib.sha256(key.encode("ascii")).hexdigest(), eid))
    return [eid for _, eid in sorted(candidates)[:12]]


def audit(root, source_root, plan_path):
    report = read(verified(root, "report.json", REPORT_SHA))
    require(read(root / "status.json") == {"status": "completed", "report_sha256": REPORT_SHA,
                                          "optimizer_steps": 0}, "completed status mismatch")
    for name, digest in report["output_sha256"].items():
        verified(root, name, digest)
    verified(root, "executed_acquire_libero_task_index.py", EXECUTED_SHA)
    require(report["code_sha256"] == EXECUTED_SHA and sha(plan_path) == PLAN_SHA, "execution/plan drift")
    auth_path = plan_path.with_name("libero-task-index-acquisition-authorization-v1.json")
    require(sha(auth_path) == AUTH_SHA and report["authorization_sha256"] == AUTH_SHA, "authorization drift")
    auth, plan, identity = read(auth_path), read(plan_path), read(root / "identity.json")
    require(all(report[k] == v for k, v in identity.items()), "report identity differs")
    for name, digest in report["bound_source_files"].items():
        verified(source_root, name, digest)
    prior = read(source_root / "report.json")
    inventory = read(source_root / "hub_inventory.json")
    entries = read(root / "task_inventory.json")
    require(len(entries) == 1693 and {e["episode_index"] for e in entries} == set(range(1693)), "episode coverage")
    require(sum(e["record_count"] for e in entries) == 273465, "row count")
    selection = read(root / "selection.json")
    require(selection == report["selection"] == select_split(entries, [], plan), "frozen selector mismatch")
    require([e["episode_index"] for e in selection["episodes"]] == independent_ids(entries, plan), "independent ranking mismatch")
    require(sum(e["record_count"] for e in selection["episodes"]) <= 12000, "whole episode budget")
    files = {e["path"]: e for e in identity["new_files"]}
    require(len(files) == 376 and sum(e["size"] for e in files.values()) == 20208168, "acquisition inventory shape")
    for name, entry in files.items():
        require(entry == inventory[name], "immutable source inventory mismatch")
        verified(root / "source", name, entry["lfs_sha256"], entry["size"])
    outcomes = {e["path"]: e for e in report["files"]}
    require(len(report["files"]) == len(outcomes) == 376 and set(outcomes) == set(files), "outcomes missing/duplicate")
    for name, outcome in outcomes.items():
        require(outcome["sha256"] == files[name]["lfs_sha256"] and outcome["size"] == files[name]["size"], "outcome mismatch")
    first = read(verified(root, "first_attempt_failure_status.json",
        "6b142a19ef65d4b28e6a1c091e6c45b5b75295d5a5d6cb9f527e2680520ae0b1"))
    verified(root, "first_attempt_budget_reservations.jsonl", "6a32d096a95a00339dadd747ca9b8b82b115c46372f2d8236719921b8e0b49b2")
    verified(root, "first_attempt_download_attempts.jsonl", hashlib.sha256(b"").hexdigest(), 0)
    old_reservations = lines(root / "first_attempt_budget_reservations.jsonl")
    attempts, reservations = lines(root / "download_attempts.jsonl"), lines(root / "budget_reservations.jsonl")
    first_paths = {e["path"] for e in old_reservations}
    require(len(first_paths) == len(old_reservations) == 4 and first["metadata_selection_if_computed"] is None, "first failure shape")
    first_bytes = sum(files[name]["size"] for name in first_paths)
    require(first_bytes == first["cumulative_response_payload_bytes"] == first["conservative_request_reserved_bytes"] == 211143, "initial byte reconstruction")
    expected_reservations = old_reservations + [{"path": e["path"], "attempt": e["attempt"],
                                               "reserved_bytes": files[e["path"]]["size"]} for e in attempts]
    canonical = lambda rows: Counter(json.dumps(r, sort_keys=True) for r in rows)
    require(canonical(reservations) == canonical(expected_reservations), "reservation accounting mismatch")
    successes = [e for e in attempts if e["error_type"] is None]
    errors = [e for e in attempts if e["error_type"] is not None]
    require(len(successes) == 372 and {e["path"] for e in successes} == set(files) - first_paths, "success/reuse coverage")
    require(len(errors) == 2 and all(e["received_bytes"] == 0 and e["error_type"] == "SSLError" for e in errors), "unexpected failure accounting")
    for e in successes:
        require(e["received_bytes"] == files[e["path"]]["size"], "successful response byte mismatch")
    for name in files:
        require(outcomes[name]["reused"] is (name in first_paths), "reuse marker mismatch")
    require(report["downloaded_unique_files_this_invocation"] == 372 and report["reused_verified_files_this_invocation"] == 4, "report file counts")
    response_bytes = sum(e["received_bytes"] for e in attempts)
    reserved_bytes = sum(e["reserved_bytes"] for e in reservations)
    require(response_bytes == report["cumulative_response_payload_bytes"] == 19997025, "resume byte field mismatch")
    require(reserved_bytes == report["conservative_request_reserved_bytes"] <= auth["max_cumulative_payload_response_bytes"], "reserved budget mismatch")
    expected_parts = {(root / e["part_path"]).resolve() for e in errors}
    actual_parts = {p.resolve() for p in (root / "source").rglob("*.part")}
    require(actual_parts == expected_parts and all(p.stat().st_size == 0 for p in actual_parts), "unexpected partial files")
    payloads = read(root / "selected_payload_inventory.json")
    require(payloads == report["payload_inventory"], "payload inventory differs")
    cached_files = {e["path"]: e for e in prior["files"]}
    video_files = payloads["unique_video_files"]
    require(len({e["path"] for e in video_files}) == len(video_files), "video duplicates")
    cached_video, missing_video = [], []
    for entry in video_files:
        require(entry == inventory[entry["path"]], "video inventory provenance mismatch")
        cached = cached_files.get(entry["path"])
        if cached:
            verified(source_root, cached["local_path"], entry["lfs_sha256"], entry["size"])
            cached_video.append(entry)
        else:
            missing_video.append(entry)
    require(sum(e["size"] for e in video_files) == payloads["unique_video_bytes"], "video byte total")
    require(all(report[k] == 0 for k in ("video_downloads", "images_decoded", "features_extracted", "optimizer_steps")), "scope counter")
    require(all(report[k] is False for k in ("actions_states_interpreted", "training_ready", "formal_data_allowed", "family_independence_verified")), "readiness promotion")
    return {"schema": "libero_task_index_recovery_audit_v1", "status": "passed_metadata_and_recovery_accounting_only",
        "code_sha256": sha(__file__), "original_report_sha256": REPORT_SHA, "executed_code_sha256": EXECUTED_SHA,
        "metadata_episodes": 1693, "projected_row_count_from_inventory": 273465, "selection": selection["split"],
        "first_verified_unlogged_payload_bytes": first_bytes, "logged_resume_response_payload_bytes": response_bytes,
        "reconstructed_total_response_payload_bytes": first_bytes + response_bytes, "conservative_reserved_bytes": reserved_bytes,
        "original_report_counter_omits_first_invocation": True, "original_report_rewritten": False,
        "verified_new_parquet_files": len(files), "tracked_empty_failed_parts": len(actual_parts),
        "unique_video_files": len(video_files), "unique_video_bytes": sum(e["size"] for e in video_files),
        "verified_cached_video_files": len(cached_video), "verified_cached_video_bytes": sum(e["size"] for e in cached_video),
        "missing_video_files": len(missing_video), "missing_video_bytes": sum(e["size"] for e in missing_video),
        "parquet_columns_decoded_by_this_audit": 0, "images_decoded": 0, "optimizer_steps": 0,
        "source_quality_audit_passed": False, "training_ready": False, "family_independence_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-root", "plan", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "new companion audit output required")
    result = audit(args.root, args.source_root, args.plan)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)
