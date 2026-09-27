"""Build diagnostic, causal event windows from the existing real10 allowlist.

Read-only source adapter; standard library only. No policy, tracker, training,
hardware, image resizing, new split, or automatic contact labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time


SCHEMA = "real10_observable_event_windows_v1"
VIEWS = ("side", "top")
RAW_TO_INTENT = {-1: 0, 0: 1, 1: 2}
HISTORY_FIELDS = (
    "piper_busy", "piper_step_after_command", "piper_async_status",
    "piper_async_command", "piper_async_start_timestamp",
    "piper_async_done_timestamp", "piper_async_error",
    "piper_cooldown", "piper_request_accepted", "piper_executed_feed",
)
VISIBILITY = ("clear", "ambiguous", "unobservable")
MOTION = ("advance", "stationary", "retract", "uncertain")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timing(row):
    result = {key: row.get("timestamps", {}).get(key)
              for key in ("side_image", "top_image", "elite_pose", "piper_action")}
    if not all(finite(v) for v in result.values()):
        raise ValueError(f"step {row.get('step')}: missing/nonfinite required recorded timestamps")
    return result


def frame_id(episode, step):
    return f"{episode}/{step:06d}"


def image_ref(episode, path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("source image paths must be relative to their episode")
    return (Path(episode) / path).as_posix()


def observation(episode, row, previous):
    ts = timing(row)
    pose = row["state"]["elite_tcp_pose_6d"]
    if len(pose) != 6 or not all(finite(v) for v in pose):
        raise ValueError(f"{episode}/{row['step']}: invalid measured Elite pose")
    # Like the existing real10 encoder, use the PREVIOUS record only. Its
    # post-submit timestamp bounds snapshot availability, not physical motion.
    history_ok = previous is not None and timing(previous)["piper_action"] <= min(ts["side_image"], ts["top_image"])
    controller = previous["state"].get("controller_state", {}) if history_ok else {}
    values = {key: controller.get(key) for key in HISTORY_FIELDS}
    return {
        "frame_id": frame_id(episode, row["step"]), "source_episode": episode,
        "step": row["step"], "task": row["task"],
        "images": {view: {"path": image_ref(episode, row[f"{view}_image"]),
                          "recorded_timestamp_s": ts[f"{view}_image"]} for view in VIEWS},
        "observation_available_at_s": max(ts[key] for key in ("side_image", "top_image", "elite_pose")),
        "elite": {"tcp_pose_6d_xyz_mm_rpy_rad": pose, "recorded_timestamp_s": ts["elite_pose"],
                  "valid": not row["state"].get("controller_state", {}).get("elite_pose_stale", False)},
        "controller_history": {
            "source_step": previous["step"] if history_ok else None,
            "available_by_s": timing(previous)["piper_action"] if history_ok else None,
            "valid": history_ok, "values": values,
            "missing_fields": [key for key in HISTORY_FIELDS if key not in controller],
            "recorded_null_fields": [key for key in HISTORY_FIELDS if key in controller and controller[key] is None],
            "missing_reason": None if history_ok else ("episode_start" if previous is None else "previous_snapshot_not_available_before_images"),
        },
    }


def execution_record(episode, row):
    controller = row["state"].get("controller_state", {})
    command = row["reference_action"]["piper_step_command"]
    result = controller.get("piper_executed_command")
    accepted = None
    if command != 0 and isinstance(result, str):
        if "async_started" in result:
            accepted = True
        elif result.startswith("busy_skip"):
            accepted = False
    return {
        "frame_id": frame_id(episode, row["step"]), "step": row["step"],
        "post_submit_recorded_at_s": timing(row)["piper_action"],
        "requested_piper_intent_id": RAW_TO_INTENT[command],
        "requested_burst_count": row["reference_action"].get("piper_burst_count"),
        "submit_result_raw": result, "async_submit_accepted": accepted,
        "snapshot": {key: controller.get(key) for key in HISTORY_FIELDS},
        "feeder_response_raw": controller.get("feeder_last_response"),
        "physical_feed_executed": None,
        "physical_feed_evidence": "not_measured_by_controller_log",
    }


def summarize(values):
    values = sorted(values)
    if not values:
        return None
    return {"min": values[0], "median": values[len(values) // 2],
            "p95": values[min(len(values) - 1, int(.95 * len(values)))], "max": values[-1]}


def annotation_template(window):
    return {"window_id": window["window_id"], "source_episode": window["source_episode"],
            "split_group": window["split_group"], "source_split": window["source_split"],
            "status": "unreviewed", "reviewer": None,
            "anchor_visibility": {view: None for view in VIEWS},
            "future_motion_response": {view: None for view in VIEWS}, "notes": ""}


def build(source_root, pack_manifest, out, *, history_s=1.0, future_s=1.5, hold_windows=3):
    started = time.monotonic()
    base = read_json(pack_manifest)
    if base.get("adapter_version") != "real10_pi05_observable_history_v1":
        raise ValueError("this adapter requires the existing real10 allowlist manifest")
    if base["training_use"]["split_strategy"] != "all_10_episodes_train_no_validation":
        raise ValueError("source split changed; define an explicit migration before using this adapter")
    if not (history_s > 0 and future_s > 0 and hold_windows >= 0):
        raise ValueError("history/future must be positive; hold-windows must be nonnegative")
    out.mkdir(parents=True, exist_ok=False)
    observations, transitions, executions, windows, targets, reports = [], [], [], [], [], []
    dt_values, skew_ms, lag_ms = [], [], []
    missing, recorded_null = Counter(), Counter()
    for episode in base["source_episodes"]:
        if Path(episode).name != episode:
            raise ValueError("episode names must be single directory components")
        rows = read_jsonl(source_root / episode / "records.jsonl")
        manifest = read_json(source_root / episode / "manifest.json")
        if manifest["camera_setup"].get("top_duplicated_from_side"):
            raise ValueError(f"{episode}: not a dual-view source")
        if len(rows) < 2 or any(row["step"] != i or row["task"] != manifest["task"] for i, row in enumerate(rows)):
            raise ValueError(f"{episode}: unexpected step order/task")
        obs = [observation(episode, row, rows[i - 1] if i else None) for i, row in enumerate(rows)]
        logs = [execution_record(episode, row) for row in rows]
        for i, row in enumerate(rows):
            ts = timing(row)
            skew_ms.append(abs(ts["side_image"] - ts["top_image"]) * 1000)
            lag_ms.append((ts["elite_pose"] - min(ts["side_image"], ts["top_image"])) * 1000)
            missing.update(obs[i]["controller_history"]["missing_fields"])
            recorded_null.update(obs[i]["controller_history"]["recorded_null_fields"])
            if i == len(rows) - 1:
                continue  # No fabricated terminal action or future endpoint.
            nxt = rows[i + 1]
            nts = timing(nxt)
            durations = {key: nts[key] - ts[key] for key in ("side_image", "top_image", "elite_pose")}
            if min(durations.values()) <= 0 or obs[i + 1]["observation_available_at_s"] <= obs[i]["observation_available_at_s"]:
                raise ValueError(f"{episode}/{i}: nonmonotonic observation time")
            dt_values.append(durations["side_image"])
            transitions.append({
                "frame_id": obs[i]["frame_id"], "next_frame_id": obs[i + 1]["frame_id"],
                "source_episode": episode, "source_split": "train", "split_group": episode,
                "dt_s": durations,
                "stored_action_target": {
                    "elite_tcp_delta_6d": row["reference_action"]["elite_tcp_delta_6d"],
                    "piper_intent_id": logs[i]["requested_piper_intent_id"],
                },
                "elite_target_semantics": "stored_next_measured_pose_delta_not_known_command_at_anchor",
                "measured_elite_translation_delta_mm": [nxt["state"]["elite_tcp_pose_6d"][j] - row["state"]["elite_tcp_pose_6d"][j] for j in range(3)],
            })
        anchors = {i: "piper_request" for i, log in enumerate(logs[:-1]) if log["requested_piper_intent_id"] != 1}
        hold = [i for i, log in enumerate(logs[:-1]) if log["requested_piper_intent_id"] == 1]
        for j in range(hold_windows):
            if hold:
                anchors.setdefault(hold[min(len(hold) - 1, int((j + 1) * len(hold) / (hold_windows + 1)))], "time_coverage_hold_request")
        first_window = len(windows)
        times = [o["observation_available_at_s"] for o in obs]
        for anchor, reason in sorted(anchors.items()):
            t = times[anchor]
            history = [i for i in range(anchor + 1) if times[i] >= t - history_s]
            future = [i for i in range(anchor + 1, len(rows)) if times[i] <= t + future_s]
            window = {
                "window_id": frame_id(episode, anchor), "source_episode": episode,
                "split_group": episode, "source_split": "train", "task": manifest["task"],
                "anchor_frame_id": obs[anchor]["frame_id"], "anchor_timestamp_s": t,
                "history_frame_ids": [obs[i]["frame_id"] for i in history],
                "review_selection_reason": reason, "history_boundary_truncated": times[0] > t - history_s,
                "target_ref": frame_id(episode, anchor),
            }
            current = logs[anchor]
            start = current["snapshot"]["piper_async_start_timestamp"]
            # Match by recorded async start, not by event count/last-response alone.
            matched = [log for log in logs[anchor:]
                       if current["async_submit_accepted"] is True and start is not None
                       and log["snapshot"]["piper_async_start_timestamp"] == start
                       and finite(log["snapshot"]["piper_async_done_timestamp"])
                       and log["snapshot"]["piper_async_done_timestamp"] >= start]
            completion = matched[0] if matched else None
            targets.append({
                "window_id": window["window_id"], "future_frame_ids": [obs[i]["frame_id"] for i in future],
                "future_boundary_truncated": times[-1] < t + future_s,
                "actual_future_span_s": times[future[-1]] - t if future else 0,
                "anchor_execution_log_ref": current["frame_id"],
                "matched_executor_completion": None if completion is None else {
                    "observed_in_frame_id": completion["frame_id"],
                    "available_by_s": completion["post_submit_recorded_at_s"],
                    "done_timestamp_s": completion["snapshot"]["piper_async_done_timestamp"],
                    "error": completion["snapshot"]["piper_async_error"],
                    "physical_displacement_confirmed": False,
                },
                "subsequent_request_frame_ids": [logs[i]["frame_id"] for i in future if logs[i]["requested_piper_intent_id"] != 1],
                "warning": "observed multi-action response, not an isolated causal effect; future and completion are targets/audit only",
            })
            windows.append(window)
        observations.extend(obs)
        executions.extend(logs)
        reports.append({"episode": episode, "task": manifest["task"], "source_records": len(rows),
                        "transitions": len(rows) - 1, "review_windows": len(windows) - first_window,
                        "piper_requests": sum(log["requested_piper_intent_id"] != 1 for log in logs[:-1]),
                        "camera_serials": {v: manifest["camera_setup"].get(f"{v}_realsense_serial") for v in VIEWS}})
    if len(observations) != base["counts"]["source_records"] or len(transitions) != base["counts"]["samples"]:
        raise ValueError("source counts differ from the frozen real10 pack")
    for name, values in (("observations", observations), ("factual_transitions", transitions),
                         ("execution_log", executions), ("windows", windows), ("window_targets", targets),
                         ("annotation_template", [annotation_template(w) for w in windows])):
        write_jsonl(out / f"{name}.jsonl", values)
    contract = {
        "schema": SCHEMA, "source_pack_manifest": str(pack_manifest), "source_root_at_build": str(source_root),
        "source_episodes": base["source_episodes"], "history_s": history_s, "future_s": future_s,
        "hold_windows_per_episode": hold_windows, "source_split": "all_train_no_validation_unchanged",
        "split_group": "source_episode; this is not a new independence claim",
        "sampling": "all requested nonhold actions plus deterministic time-coverage hold windows; not a representative test set",
        "observation_file": "observations.jsonl", "history_file": "windows.jsonl",
        "target_files": ["factual_transitions.jsonl", "window_targets.jsonl", "execution_log.jsonl", "annotation_template.jsonl", "annotations.jsonl"],
        "timing": "recorded acquisition/host timestamps; retain irregular dt and camera skew, no hardware sync claim",
        "controller_timing": "previous-record snapshot only if available before both current images; current post-submit log is isolated",
        "missing": "absent fields use null plus missing_fields; recorded nulls are listed separately in recorded_null_fields, never coerced to false/zero or visual negatives",
        "recorded_null_semantics": "null async_error means no recorded error; null async start/done means no such timestamp logged in this snapshot; neither establishes physical motion",
        "physical_feed": "async accepted, counter increments, completion and UDP reply are not measured wire displacement",
        "elite_action": "stored pose delta includes next observation; not a known-at-anchor control command",
        "labels": {"anchor_visibility": list(VISIBILITY), "future_motion_response": list(MOTION),
                   "unreviewed": None, "source": "human_visual_review_only_not_contact_truth"},
        "policy_input_allowed": False, "policy_training_ready": False, "formal_data_allowed": False,
        "training_executed": False, "hardware_executed": False,
        "excluded": ["diagnostic_targets", "exact_contact", "wall_distance_truth", "tip_route_truth", "event_id", "depth", "collector_estimator_placeholders"],
    }
    report = {"schema": SCHEMA, "episodes": reports, "source_records": len(observations),
              "factual_transitions": len(transitions), "review_windows": len(windows),
              "selection_reasons": dict(Counter(w["review_selection_reason"] for w in windows)),
              "controller_missing_rows": dict(missing),
              "controller_recorded_null_rows": dict(recorded_null),
              "matched_executor_completions": sum(t["matched_executor_completion"] is not None for t in targets),
              "windows_with_subsequent_requests": sum(bool(t["subsequent_request_frame_ids"]) for t in targets),
              "windows_without_future": sum(not t["future_frame_ids"] for t in targets),
              "side_image_dt_s": summarize(dt_values), "side_top_skew_ms": summarize(skew_ms),
              "pose_minus_first_image_ms": summarize(lag_ms),
              "source_images_copied_or_decoded": 0, "visual_status": "not_viewed",
              "annotations_completed": 0, "training_executed": False, "hardware_executed": False,
              "elapsed_seconds": time.monotonic() - started}
    write_json(out / "manifest.json", contract)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--pack-manifest", type=Path, default=Path("simulation_output/real10_pi05_compat_v1/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--history-s", type=float, default=1.0)
    parser.add_argument("--future-s", type=float, default=1.5)
    parser.add_argument("--hold-windows", type=int, default=3)
    args = parser.parse_args()
    report = build(args.source_root, args.pack_manifest, args.out, history_s=args.history_s,
                   future_s=args.future_s, hold_windows=args.hold_windows)
    print(json.dumps({k: report[k] for k in ("source_records", "factual_transitions", "review_windows", "selection_reasons", "matched_executor_completions", "elapsed_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
