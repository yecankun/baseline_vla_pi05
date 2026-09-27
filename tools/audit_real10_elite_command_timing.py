"""Read-only Real10 Elite command/timing audit; no hardware, images or training.

Pair keyboard submissions with later logged absolute targets. Keep every event,
including busy skips and terminal requests. Never create a policy delta from a
future measured pose, or label a macro move_joint target as a PI05 step action.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import math
from pathlib import Path
import statistics
import time

from prepare_real10_event_windows import (
    finite, frame_id, read_json, read_jsonl, timing, write_json, write_jsonl,
)


SCHEMA = "real10_elite_command_timing_audit_v1"
VIEWS = ("side", "top")
HORIZON_S = 1.5  # Same nominal duration as the existing response windows.


def controller(row):
    return row["state"]["controller_state"]


def lower_bound(row):
    ts = timing(row)
    # Pose timestamp is recorded BEFORE the RPC; this is not exact input-ready
    # time. Loop order, not this scalar alone, puts the full observation before
    # the keyboard request. piper_action bounds submission from above.
    return max(ts[k] for k in ("side_image", "top_image", "elite_pose"))


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(.95 * len(values)) - 1], "max": values[-1]}


def distance(a, b):
    return math.dist(a[:3], b[:3])


def source_excerpt(path, names):
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    excerpts = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            excerpts.append({"name": node.name, "line_start": node.lineno,
                             "line_end": node.end_lineno,
                             "source": "\n".join(lines[node.lineno - 1:node.end_lineno])})
    return {"path": path.as_posix(), "historical_capture_version_verified": False,
            "scope": "current installed source read as text; never imported", "functions": excerpts}


def check_csv(path, rows):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        logs = list(reader)
    if len(logs) != len(rows):
        raise ValueError(f"CSV/record count mismatch: {path}")
    delays = []
    for log, row in zip(logs, rows):
        c = controller(row)
        event = c.get("user_event") or {}
        comparisons = (
            (int(log["step"]), row["step"]),
            (log["elite_submit_status"], event.get("elite_submit_status") or ""),
            (log["elite_path_status"], c["elite_path_status"]),
            (int(log["elite_path_index"]), c["elite_path_index"]),
            (read_json_value(log["elite_requested_tcp_pose_6d"]), c["elite_requested_tcp_pose_6d"]),
            (read_json_value(log["elite_requested_joints"]), c["elite_requested_joints"]),
            (log["piper_requested"], row["reference_action"]["piper_command_label"]),
        )
        if any(a != b for a, b in comparisons):
            raise ValueError(f"CSV/JSON command mismatch: {path}/{row['step']}")
        delays.append(float(log["timestamp"]) - timing(row)["piper_action"])
    return {"rows_matched": len(rows), "fields": fields,
            "csv_timestamp_minus_post_submit_s": stats(delays),
            "timestamp_semantics": "loop logging after image writes; not send/start/completion"}


def read_json_value(value):
    import json
    return json.loads(value)


def audit_episode(episode, args, reviewed):
    root = args.source_root / episode
    rows = read_jsonl(root / "records.jsonl")
    manifest = read_json(root / "manifest.json")
    config = manifest["elite_path"]
    if (config["mode"] != "keyboard" or config["execute"] is not True
            or manifest["piper_feed_on_elite_path_step"] is not False):
        raise ValueError(f"unsupported collection mode: {episode}")
    if [r["step"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"nonsequential source rows: {episode}")
    bounds = [lower_bound(r) for r in rows]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"nonmonotonic host observation bounds: {episode}")
    if any(lower_bound(r) > timing(r)["piper_action"] for r in rows):
        raise ValueError(f"inverted request bracket: {episode}")
    csv_report = check_csv(args.csv_root / f"{episode}_controller_commands.csv", rows)

    # Current workspace path is a corroborating reference, not an archived
    # capture-time path. The logged target itself is the command value source.
    path_file = args.path_root / Path(config["path_file"]).name
    path_xyz = [list(map(float, s.replace(",", " ").split()))[:3]
                for s in path_file.read_text().splitlines() if s.strip() and not s.startswith("#")]
    stamps, initial = {}, []
    for row in rows:
        c = controller(row)
        t = c.get("elite_path_command_timestamp")
        if not finite(t):
            raise ValueError(f"missing command timestamp: {episode}/{row['step']}")
        target, joints = c["elite_requested_tcp_pose_6d"], c["elite_requested_joints"]
        if not (len(target) == 6 and all(finite(v) for v in target) and all(finite(v) for v in joints)):
            raise ValueError("invalid recorded command values")
        payload = {"timestamp_after_move_joint_call_s": t, "path_index": c["elite_path_index"],
                   "requested_tcp_pose_6d": target, "requested_joints_sdk_native": joints}
        if t in stamps:
            if stamps[t]["payload"] != payload:
                raise ValueError("same command timestamp has inconsistent target/index/joints")
        else:
            stamps[t] = {"payload": payload, "first_snapshot_step": row["step"],
                         "first_snapshot_status": c["elite_path_status"],
                         "snapshot_available_by_s": timing(row)["piper_action"]}
    for t in list(stamps):
        if t < bounds[0]:
            initial.append(stamps.pop(t))
    if len(initial) != 1 or initial[0]["payload"]["path_index"] != config["start_index"]:
        raise ValueError("unexpected pre-recording approach snapshot")

    events = []
    for row in rows:
        user = controller(row).get("user_event") or {}
        if user.get("elite_command") is not None:
            if user["elite_command"] != "next":
                raise ValueError("this source audit requires an explicit review of non-next events")
            events.append({"event_id": frame_id(episode, row["step"]), "source_episode": episode,
                           "task": row["task"], "step": row["step"],
                           "submit_status": user["elite_submit_status"],
                           "request_host_bracket_s": [lower_bound(row), timing(row)["piper_action"]]})
    submitted = [e for e in events if e["submit_status"] == "next_submitted"]
    pairs, used_stamps = [], []
    for i, event in enumerate(submitted):
        step = event["step"]
        anchor, c = rows[step], controller(rows[step])
        lo, hi = event["request_host_bracket_s"]
        next_lo = submitted[i + 1]["request_host_bracket_s"][0] if i + 1 < len(submitted) else math.inf
        matches = [stamp for t, stamp in stamps.items() if lo < t < next_lo]
        if len(matches) != 1:
            event["pairing_status"] = ("terminal_done_no_new_move" if not matches and c["elite_path_done"]
                                       else "unresolved_command_pairing")
            event["matched_command_count"] = len(matches)
            continue
        stamp = matches[0]
        p = stamp["payload"]
        ret = p["timestamp_after_move_joint_call_s"]
        used_stamps.append(ret)
        event["pairing_status"] = "unique_later_logged_target"
        event["timestamp_after_move_joint_call_s"] = ret
        future = [r for r in rows[step + 1:]
                  if lower_bound(r) <= lo + HORIZON_S
                  and min(timing(r)[f"{v}_image"] for v in VIEWS) > hi]
        endpoints = {v: timing(future[-1])[f"{v}_image"] if future else None for v in VIEWS}
        end = max(endpoints.values()) if future else hi
        later, overlap = [], []
        for row in rows[step + 1:]:
            start, stop = lower_bound(row), timing(row)["piper_action"]
            if start >= end:
                break
            user = controller(row).get("user_event") or {}
            elite = user.get("elite_submit_status") == "next_submitted"
            piper = row["reference_action"]["piper_step_command"] != 0
            if elite or piper:
                item = {"frame_id": frame_id(episode, row["step"]), "elite_submitted": elite,
                        "piper_requested_raw": row["reference_action"]["piper_step_command"],
                        "request_bracket_s": [start, stop]}
                (later if stop <= end else overlap).append(item)
        # Numeric position proximity is ONLY an observational timing check.
        # It is not measured completion, orientation convergence or guidewire motion.
        near = [r for r in rows[step + 1:]
                if timing(r)["piper_action"] < ret and not controller(r)["elite_pose_stale"]
                and distance(r["state"]["elite_tcp_pose_6d"], p["requested_tcp_pose_6d"]) <= 1.0]
        previous = rows[step - 1] if step else None
        history_valid = previous is not None and timing(previous)["piper_action"] <= min(
            timing(anchor)[f"{v}_image"] for v in VIEWS)
        old_target = c["elite_requested_tcp_pose_6d"]
        reviewed_item = reviewed.get(event["event_id"])
        exact_label_window = bool(reviewed_item and reviewed_item["future_frame_ids"] == [
            frame_id(episode, r["step"]) for r in future])
        pair = {
            **event,
            "anchor_observation": {
                "frame_id": event["event_id"], "images": {v: {
                    "path_relative_to_episode": anchor[f"{v}_image"],
                    "host_timestamp_s": timing(anchor)[f"{v}_image"]} for v in VIEWS},
                "measured_tcp_pose_6d": anchor["state"]["elite_tcp_pose_6d"],
                "pose_query_start_host_timestamp_s": timing(anchor)["elite_pose"],
                "pose_stale": c["elite_pose_stale"],
                "before_request_evidence": "collector loop reads images and pose before keyboard submission",
                "prior_controller_history_valid": history_valid,
                "prior_piper_busy": controller(previous)["piper_busy"] if history_valid else None,
            },
            "logged_request_audit_only": {
                "kind": "absolute_tcp_target_to_IK_to_move_joint",
                "target_tcp_pose_6d_xyz_mm_rpy_rad": p["requested_tcp_pose_6d"],
                "target_joints_sdk_native_unit_unverified": p["requested_joints_sdk_native"],
                "speed_parameter": config["speed"],
                "speed_semantics": "current installed SDK documents joint speed percent; capture SDK not frozen",
                "piper_intent_id": {0: 1, 1: 2, -1: 0}[anchor["reference_action"]["piper_step_command"]],
                "piper_burst_count": anchor["reference_action"]["piper_burst_count"],
                "target_source": "later command snapshot, not later measured pose",
                "target_logged_only_after_call_returns": True,
                "path_index_audit_only": p["path_index"],
                "current_path_xyz_matches": p["requested_tcp_pose_6d"][:3] == path_xyz[p["path_index"]],
                "index_advances_once_from_anchor_snapshot": p["path_index"] == c["elite_path_index"] + 1,
                "anchor_snapshot_contains_old_target": old_target != p["requested_tcp_pose_6d"],
                "target_offset_from_anchor_xyz_mm_audit_only": [
                    p["requested_tcp_pose_6d"][j] - anchor["state"]["elite_tcp_pose_6d"][j] for j in range(3)],
                "canonical_step_delta_conversion_supported": False,
            },
            "timing_audit": {
                "call_return_stamp_s": ret,
                "return_minus_submit_bracket_s": [ret - hi, ret - lo],
                "first_snapshot_frame_id": frame_id(episode, stamp["first_snapshot_step"]),
                "snapshot_available_by_s": stamp["snapshot_available_by_s"],
                "first_snapshot_status": stamp["first_snapshot_status"],
                "return_after_nominal_response_horizon": ret > lo + HORIZON_S,
                "within_1mm_sample_available_before_return_frame": frame_id(episode, near[0]["step"]) if near else None,
                "exact_send_or_motion_onset_s": None, "robot_completion_verified": False,
                "regular_call_return_value_recorded": False,
            },
            "future_response_audit_only": {
                "nominal_horizon_s": HORIZON_S, "reference_time_s": lo,
                "reference_uncertainty_bracket_s": [lo, hi],
                "frame_ids": [frame_id(episode, r["step"]) for r in future],
                "image_endpoint_host_s": endpoints,
                "episode_covers_nominal_horizon": min(timing(rows[-1])[f"{v}_image"] for v in VIEWS) >= lo + HORIZON_S,
                "later_requests_before_last_image": later, "request_brackets_overlap_last_image": overlap,
                "isolated_causal_effect_supported": False,
                "same_existing_human_label_window": exact_label_window,
                "existing_ui_index": reviewed_item["ui_index"] if exact_label_window else None,
                "human_label_copied_or_generated": False,
            },
        }
        pairs.append(pair)
    for event in events:
        if event["submit_status"].startswith("busy_skip"):
            event["pairing_status"] = "thread_busy_skip_not_new_robot_command"
    if len(used_stamps) != len(set(used_stamps)) or set(used_stamps) != set(stamps):
        raise ValueError("a fresh command timestamp was reused or could not be paired")
    return events, pairs, {
        "source_episode": episode, "task": manifest["task"], "records": len(rows),
        "submission_status_counts": dict(Counter(e["submit_status"] for e in events)),
        "pairing_status_counts": dict(Counter(e["pairing_status"] for e in events)),
        "paired_commands": len(pairs), "pre_recording_approach_snapshots_excluded": len(initial),
        "command_csv_check": csv_report, "elite_path_config": config,
        "capture_time_path_snapshot_present": False,
        "current_reference_path": path_file.as_posix(), "current_path_points": len(path_xyz),
        "recorded_elite_thread_busy_field_rows": sum("elite_path_command_busy" in controller(r) for r in rows),
        "recorded_pose_stale_rows": sum(controller(r)["elite_pose_stale"] for r in rows),
        "side_frame_interval_s": stats([timing(b)["side_image"] - timing(a)["side_image"] for a, b in zip(rows, rows[1:])]),
    }


def run(args):
    started = time.monotonic()
    if args.out.exists():
        raise FileExistsError(f"preserve existing output; use a fresh --out: {args.out}")
    manifest = read_json(args.pack / "manifest.json")
    episodes = manifest["source_episodes"]
    if len(episodes) != 10 or len(set(episodes)) != 10:
        raise ValueError("requires the original ten-episode allowlist")
    reviewed = {r["window_id"]: r for r in read_jsonl(args.annotations)}
    events, pairs, per_episode = [], [], []
    for episode in episodes:
        e, p, report = audit_episode(episode, args, reviewed)
        events.extend(e)
        pairs.extend(p)
        per_episode.append(report)
    counts = {
        "episodes": len(episodes), "records": sum(e["records"] for e in per_episode),
        "elite_keyboard_events": len(events), "paired_absolute_commands": len(pairs),
        "submission_status": dict(Counter(e["submit_status"] for e in events)),
        "pairing_status": dict(Counter(e["pairing_status"] for e in events)),
        "paired_by_task": dict(Counter(p["task"] for p in pairs)),
    }
    for key in ("current_path_xyz_matches", "index_advances_once_from_anchor_snapshot", "anchor_snapshot_contains_old_target"):
        counts[key] = sum(p["logged_request_audit_only"][key] for p in pairs)
    counts.update({
        "return_stamp_after_1p5s_horizon": sum(p["timing_audit"]["return_after_nominal_response_horizon"] for p in pairs),
        "within_1mm_observation_available_before_return": sum(bool(p["timing_audit"]["within_1mm_sample_available_before_return_frame"]) for p in pairs),
        "future_frames_present": sum(bool(p["future_response_audit_only"]["frame_ids"]) for p in pairs),
        "episode_covers_nominal_horizon": sum(p["future_response_audit_only"]["episode_covers_nominal_horizon"] for p in pairs),
        "later_requests_before_last_image": sum(bool(p["future_response_audit_only"]["later_requests_before_last_image"]) for p in pairs),
        "request_brackets_overlap_last_image": sum(bool(p["future_response_audit_only"]["request_brackets_overlap_last_image"]) for p in pairs),
        "existing_exact_human_label_windows": sum(p["future_response_audit_only"]["same_existing_human_label_window"] for p in pairs),
        "pre_request_prior_history_valid": sum(p["anchor_observation"]["prior_controller_history_valid"] for p in pairs),
        "pre_request_prior_piper_busy": sum(p["anchor_observation"]["prior_piper_busy"] is True for p in pairs),
        "paired_piper_intent_counts": dict(Counter(str(p["logged_request_audit_only"]["piper_intent_id"]) for p in pairs)),
        "canonical_complete_action9_samples_created": 0,
    })
    flags = {"policy_input_allowed": False, "policy_training_ready": False,
             "formal_data_allowed": False, "deployable": False,
             "training_executed": False, "hardware_executed": False}
    report = {
        "schema": SCHEMA, "source_root": args.source_root.as_posix(),
        "source_episodes": episodes, "counts": counts, "per_episode": per_episode,
        "timing_s": {
            "submission_bracket_width": stats([e["request_host_bracket_s"][1] - e["request_host_bracket_s"][0] for e in events]),
            "return_after_submission_upper_bound": stats([p["timing_audit"]["return_minus_submit_bracket_s"][0] for p in pairs]),
            "return_after_submission_lower_bound": stats([p["timing_audit"]["return_minus_submit_bracket_s"][1] for p in pairs]),
        },
        "evidence_boundaries": [
            "current collector/SDK source is not a version-frozen capture-time environment",
            "SDK move_joint defaults to blocking; wait_stop exits on non-PLAY, including error or pause",
            "collector ignores the normal move_joint return value; move_joint_sent is not verified success",
            "camera host timestamps and pose-query-start timestamps are not hardware-synchronized arrival times",
            "CSV log timestamps follow image writes; no extra send or completion record was discovered",
            "all rows retained in audit; command-event alignment is observational, not causal isolation",
            "all ten episodes were already used by PI05; these are not independent policy validation data",
            "no command targets or path IDs were joined into any existing model input",
        ],
        **flags,
    }
    evidence = {
        "collector": source_excerpt(args.collector, {
            "command_index", "submit_keyboard_command", "command_next", "finalize_record", "read_nonblocking"}),
        "sdk": source_excerpt(args.sdk_move_source, {"move_joint", "__wait_stop"}),
    }
    args.out.mkdir(parents=True)
    write_jsonl(args.out / "events.jsonl", events)
    write_jsonl(args.out / "command_pairs.jsonl", pairs)
    write_json(args.out / "source_evidence.json", evidence)
    report["runtime_s"] = time.monotonic() - started
    write_json(args.out / "report.json", report)
    print({"out": args.out.as_posix(), "counts": counts, "runtime_s": report["runtime_s"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("simulation_output/real10_event_source_metadata_v1"))
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--annotations", type=Path, default=Path("simulation_output/real10_pre_action_label_retest_v1/annotation_snapshot.jsonl"))
    parser.add_argument("--csv-root", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, default=Path("path"))
    parser.add_argument("--collector", type=Path, default=Path("data/collect/collect_real_shadow_pilot.py"))
    parser.add_argument("--sdk-move-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
