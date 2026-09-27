"""Export pre-action queries and separate human response targets, without training.

The recorded Elite BC target uses the NEXT measured pose. It is not a candidate
command: this adapter leaves that field unknown and rejects full-action packing.
Piper requests support only observational future-response supervision, not
isolated effects or counterfactual candidate ranking. No model/hardware imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time

from export_openpi_compat_to_lerobot import task_text
from prepare_real10_event_windows import (
    finite, frame_id, observation, read_json, read_jsonl, timing,
    write_json, write_jsonl,
)
from prepare_real_pi05_pack import ADAPTER_VERSION, STATE_LAYOUT, encode_real_state


SCHEMA = "real10_pre_action_response_interface_v1"
VIEWS = ("side", "top")
LABELS = ("stationary", "advance")
RAW_TO_INTENT = {-1: 0, 0: 1, 1: 2}


def image_ref(image_pack, fid, view):
    episode, step = fid.split("/")
    if Path(episode).name != episode or not step.isdigit() or view not in VIEWS:
        raise ValueError("invalid frame reference")
    return (image_pack / "images" / episode / view / f"{int(step):06d}.png").as_posix()


def history_observation(episode, rows, step):
    return observation(episode, rows[step], rows[step - 1] if step else None)


def pre_action_input(window, rows, image_pack, nominal_horizon_s):
    """Whitelist projection: no targets, annotations or future logs accepted.

    Paths are image-loader references only, never categorical model features.
    Relative image/pose/history times preserve recorded skew, not hardware sync.
    """
    episode = window["source_episode"]
    anchor_step = int(window["anchor_frame_id"].split("/")[1])
    cutoff = window["anchor_timestamp_s"]
    history = []
    for fid in window["history_frame_ids"]:
        if fid.split("/")[0] != episode:
            raise ValueError("history crosses an episode")
        step = int(fid.split("/")[1])
        if step > anchor_step:
            raise ValueError("future frame in pre-action history")
        obs = history_observation(episode, rows, step)
        if obs["observation_available_at_s"] > cutoff:
            raise ValueError("history not available at cutoff")
        previous = obs["controller_history"]
        state, valid = encode_real_state(
            obs["elite"]["tcp_pose_6d_xyz_mm_rpy_rad"], obs["task"],
            previous["values"] if previous["valid"] else None,
        )
        if not obs["elite"]["valid"]:
            valid[:6] = False
        history.append({
            "images": {v: {"path": image_ref(image_pack, fid, v),
                            "relative_time_s": obs["images"][v]["recorded_timestamp_s"] - cutoff}
                       for v in VIEWS},
            "state_32": state.tolist(), "state_valid_32": valid.tolist(),
            "elite_pose_relative_time_s": obs["elite"]["recorded_timestamp_s"] - cutoff,
            "controller_snapshot_relative_time_s": (
                previous["available_by_s"] - cutoff if previous["valid"] else None),
        })
    if not history or window["history_frame_ids"][-1] != window["anchor_frame_id"]:
        raise ValueError("history must end at the anchor")
    ref = rows[anchor_step]["reference_action"]
    intent = RAW_TO_INTENT[ref["piper_step_command"]]
    burst = ref["piper_burst_count"]
    if not isinstance(burst, int) or burst < 1:
        raise ValueError("invalid requested burst count")
    return {
        "task_instruction": task_text({"task": window["task"]}),
        "nominal_horizon_s": nominal_horizon_s,
        "history": history,
        "candidate_action": {
            "elite_tcp_delta_6d": None,
            "piper_intent_id": intent,
            "valid": {"elite_tcp_delta_6d": [False] * 6, "piper_intent_id": True},
        },
        "request_context": {"piper_burst_count": burst},
    }


def complete_action9(candidate):
    """Only an explicitly complete command may enter the existing 9D world API.

    Unknown Elite is NOT zero/hold. This helper never imputes or calls the model.
    A future adapter must establish command provenance before setting validity.
    """
    delta = candidate["elite_tcp_delta_6d"]
    valid = candidate["valid"]
    if (valid["elite_tcp_delta_6d"] != [True] * 6
            or valid["piper_intent_id"] is not True or delta is None):
        raise ValueError("incomplete candidate action; do not impute Elite with future pose or zero")
    intent = candidate["piper_intent_id"]
    if (len(delta) != 6 or not all(finite(v) for v in delta)
            or type(intent) is not int or intent not in (0, 1, 2)):
        raise ValueError("invalid complete candidate action")
    return [float(v) for v in delta] + [float(i == intent) for i in range(3)]


def temporal_audit(window, selected, target, rows, observations, executions):
    """Post-cutoff logs are audit data; never joined into model_input."""
    fid, episode = window["anchor_frame_id"], window["source_episode"]
    anchor = observations[fid]
    cutoff = window["anchor_timestamp_s"]
    step = anchor["step"]
    current = rows[step]["state"]["controller_state"]
    last = observations[target["future_frame_ids"][-1]]
    endpoint = {v: last["images"][v]["recorded_timestamp_s"] for v in VIEWS}
    end = max(endpoint.values())
    piper_definite, piper_overlap, elite_submissions, elite_overlap = [], [], [], []
    for row in rows[step + 1:]:
        ts = timing(row)
        start = max(ts[v] for v in ("side_image", "top_image", "elite_pose"))
        if start >= end:
            break  # Requests in this record occur after its observations.
        rowid = frame_id(episode, row["step"])
        ref = row["reference_action"]
        entry = {"frame_id": rowid, "request_time_bracket_s": [start, ts["piper_action"]]}
        if ref["piper_step_command"] != 0:
            item = {**entry, "piper_intent_id": RAW_TO_INTENT[ref["piper_step_command"]]}
            (piper_definite if ts["piper_action"] <= end else piper_overlap).append(item)
        event = row["state"]["controller_state"].get("user_event") or {}
        if event.get("elite_command") is not None:
            item = {**entry, "elite_command": event["elite_command"],
                    "submit_status": event.get("elite_submit_status")}
            (elite_submissions if ts["piper_action"] <= end else elite_overlap).append(item)

    # Sent timestamps may be visible only in a later snapshot. They are recorded
    # after move_joint returns, not exact onset/completion or a new anchor action.
    sent = {}
    for row in rows:
        controller = row["state"]["controller_state"]
        timestamp = controller.get("elite_path_command_timestamp")
        if finite(timestamp) and cutoff < timestamp <= end and timestamp not in sent:
            sent[timestamp] = {
                "recorded_after_send_at_s": timestamp,
                "first_snapshot_frame_id": frame_id(episode, row["step"]),
                "requested_tcp_pose_6d": controller.get("elite_requested_tcp_pose_6d"),
            }
    timestamp = current.get("elite_path_command_timestamp")
    log = executions[fid]
    return {
        "window_id": selected["window_id"], "ui_index": selected["ui_index"],
        "source_episode": episode, "input_cutoff_s": cutoff,
        "anchor_post_submit_recorded_at_s": log["post_submit_recorded_at_s"],
        "anchor_execution": log,
        "anchor_elite_user_event": (current.get("user_event") or {}).get("elite_command"),
        "anchor_elite_requested_pose_present": current.get("elite_requested_tcp_pose_6d") is not None,
        "anchor_elite_sent_timestamp_in_decision_bracket": bool(
            finite(timestamp) and cutoff < timestamp <= log["post_submit_recorded_at_s"]),
        "elite_candidate_missing_reason": "BC delta uses next measured pose; historical/pending setpoint is not a new anchor delta command",
        "response_image_endpoint_s": endpoint,
        "response_image_span_from_cutoff_s": {v: endpoint[v] - cutoff for v in VIEWS},
        "later_piper_requests_before_last_image": piper_definite,
        "piper_request_brackets_overlap_last_image": piper_overlap,
        "later_elite_submissions_before_last_image": elite_submissions,
        "elite_submission_brackets_overlap_last_image": elite_overlap,
        "elite_sent_timestamps_inside_response": list(sent.values()),
        "source_window_targets": target,
        "selection_reason_audit_only": selected["selection_reason_audit_only"],
        "isolated_causal_effect_supported": False,
    }


def build(args):
    started = time.monotonic()
    if args.out.exists():
        raise FileExistsError(f"preserve existing output; choose a fresh --out: {args.out}")
    manifest = read_json(args.pack / "manifest.json")
    frozen = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    folds = read_json(args.reference / "folds.json")
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    windows = {r["window_id"]: r for r in read_jsonl(args.pack / "windows.jsonl")}
    targets = {r["window_id"]: r for r in read_jsonl(args.pack / "window_targets.jsonl")}
    executions = {r["frame_id"]: r for r in read_jsonl(args.pack / "execution_log.jsonl")}
    latest = {r["window_id"]: r for r in read_jsonl(args.pack / "annotations_joint_v2.jsonl")}
    episodes = sorted({r["source_episode"] for r in frozen})
    if not set(episodes) <= set(manifest["source_episodes"]):
        raise ValueError("episode outside source allowlist")
    source = {e: read_jsonl(args.source_root / e / "records.jsonl") for e in episodes}
    for episode, rows in source.items():
        if [r["step"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"nonsequential source records: {episode}")
    if any(manifest[k] is not False for k in (
            "policy_input_allowed", "policy_training_ready", "formal_data_allowed")):
        raise ValueError("this interface is only for the diagnostic event pack")
    heldout = []
    for fold in folds:
        train, test = set(fold["train_indices"]), set(fold["test_indices"])
        if train & test or train | test != set(range(len(frozen))):
            raise ValueError("invalid frozen fold indices")
        expected = {i for i, r in enumerate(frozen) if r["source_episode"] == fold["heldout_episode"]}
        if test != expected:
            raise ValueError("frozen fold is not whole-episode")
        heldout.extend(test)
    if sorted(heldout) != list(range(len(frozen))):
        raise ValueError("each sample must be held out exactly once in source folds")

    queries, response_targets, audits = [], [], []
    image_paths, unknown_action_rejections = set(), 0
    mutation_checks, history_count = 0, 0
    for index, selected in enumerate(frozen):
        wid, episode = selected["window_id"], selected["source_episode"]
        window, target, rows = windows[wid], targets[wid], source[episode]
        if latest[wid] != selected["human_annotation"]:
            raise ValueError(f"annotation changed since frozen snapshot: {wid}")
        if (window["split_group"] != episode or window["source_split"] != "train"
                or selected["future_frame_ids"] != target["future_frame_ids"]):
            raise ValueError("source split or annotated horizon changed")
        annotation = selected["human_annotation"]
        if (annotation["annotation_source"] != "human_visual_review"
                or annotation["status"] != "reviewed" or annotation["joint_motion_response"] not in LABELS):
            raise ValueError("requires an explicit complete binary human label")
        for fid in window["history_frame_ids"] + target["future_frame_ids"]:
            if fid.split("/")[0] != episode:
                raise ValueError("response crosses episode")
            obs = history_observation(episode, rows, int(fid.split("/")[1]))
            if obs != observations[fid]:
                raise ValueError(f"event/source observation mismatch: {fid}")
        anchor = observations[window["anchor_frame_id"]]
        if window["anchor_timestamp_s"] != anchor["observation_available_at_s"]:
            raise ValueError("anchor cutoff mismatch")
        model_input = pre_action_input(window, rows, args.image_pack, manifest["future_s"])
        action = model_input["candidate_action"]
        log = executions[window["anchor_frame_id"]]
        if (action["piper_intent_id"] != log["requested_piper_intent_id"]
                or model_input["request_context"]["piper_burst_count"] != log["requested_burst_count"]
                or log["post_submit_recorded_at_s"] < window["anchor_timestamp_s"]):
            raise ValueError("Piper request/timing mismatch")
        queries.append({
            "sample_index": index, "window_id": wid,
            "provenance": {"source_episode": episode, "original_source_split": "train",
                           "input_cutoff_s": window["anchor_timestamp_s"],
                           "history_frame_ids": window["history_frame_ids"]},
            "model_input": model_input,
        })
        history_count += len(model_input["history"])
        for item in model_input["history"]:
            image_paths.update(item["images"][v]["path"] for v in VIEWS)

        future = []
        for fid in target["future_frame_ids"]:
            obs = observations[fid]
            if not window["anchor_timestamp_s"] < obs["observation_available_at_s"] <= window["anchor_timestamp_s"] + manifest["future_s"]:
                raise ValueError("response outside original annotated horizon")
            item = {"frame_id": fid, "images": {
                v: {"path": image_ref(args.image_pack, fid, v),
                    "relative_time_s": obs["images"][v]["recorded_timestamp_s"] - window["anchor_timestamp_s"]}
                for v in VIEWS},
                "measured_elite_pose_6d": obs["elite"]["tcp_pose_6d_xyz_mm_rpy_rad"],
                "pose_relative_time_s": obs["elite"]["recorded_timestamp_s"] - window["anchor_timestamp_s"]}
            if min(item["images"][v]["relative_time_s"] for v in VIEWS) <= 0:
                raise ValueError("response image is not after input cutoff")
            future.append(item)
            image_paths.update(item["images"][v]["path"] for v in VIEWS)
        response_targets.append({
            "sample_index": index, "window_id": wid,
            "joint_motion_response": annotation["joint_motion_response"],
            "label_index": LABELS.index(annotation["joint_motion_response"]),
            "motion_evidence": annotation["motion_evidence"],
            "anchor_visibility_target_only": annotation["anchor_visibility"],
            "future_observations": future,
            "label_scope": "unchanged full annotated interval; not single-action/contact truth",
        })
        audits.append(temporal_audit(window, selected, target, rows, observations, executions))
        try:
            complete_action9(action)
        except ValueError:
            unknown_action_rejections += 1
        else:
            raise ValueError("unknown Elite action unexpectedly accepted")

        # One integrated isolation check per episode, not a new training/test suite.
        if episode not in {r["provenance"]["source_episode"] for r in queries[:-1]}:
            changed = deepcopy(rows)
            step = anchor["step"]
            changed[step]["reference_action"]["elite_tcp_delta_6d"] = [999.0] * 6
            c = changed[step]["state"]["controller_state"]
            c["piper_busy"] = not c.get("piper_busy", False)
            c["piper_async_status"] = "FAKE_POST_SUBMIT_RESULT"
            c["elite_requested_tcp_pose_6d"] = [999.0] * 6
            for row in changed[step + 1:]:
                row["state"]["elite_tcp_pose_6d"] = [999.0] * 6
                row["reference_action"]["piper_step_command"] = -1
                row["state"]["controller_state"] = {}
            if pre_action_input(window, changed, args.image_pack, manifest["future_s"]) != model_input:
                raise ValueError("future/post-submit leakage into query")
            changed[step]["reference_action"]["piper_step_command"] = 0 if action["piper_intent_id"] == 2 else 1
            alternative = pre_action_input(window, changed, args.image_pack, manifest["future_s"])
            if alternative["history"] != model_input["history"] or alternative["candidate_action"] == action:
                raise ValueError("request input is not independent of observation history")
            mutation_checks += 1

    missing_images = [p for p in sorted(image_paths) if not Path(p).is_file()]
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} missing image references: {missing_images[:2]}")
    empty_state, empty_valid = encode_real_state([0.0] * 6, "left", None)
    if empty_state[27] != 0 or empty_valid[6] or empty_valid[16]:
        raise ValueError("existing missing-controller encoding changed")
    for intent in range(3):
        full = {"elite_tcp_delta_6d": [0.0] * 6, "piper_intent_id": intent,
                "valid": {"elite_tcp_delta_6d": [True] * 6, "piper_intent_id": True}}
        if complete_action9(full) != [0.0] * 6 + [float(i == intent) for i in range(3)]:
            raise ValueError("canonical action encoding changed")

    flags = {"policy_input_allowed": False, "policy_training_ready": False,
             "formal_data_allowed": False, "deployable": False,
             "training_executed": False, "hardware_executed": False}
    count = lambda key: sum(bool(r[key]) for r in audits)
    labels = Counter(t["joint_motion_response"] for t in response_targets)
    report = {
        "schema": SCHEMA, "status": "completed", **flags,
        "samples": len(queries), "episodes": len(episodes), "class_counts": dict(labels),
        "piper_request_counts": dict(Counter(str(q["model_input"]["candidate_action"]["piper_intent_id"]) for q in queries)),
        "request_by_response": {str(intent): dict(Counter(t["joint_motion_response"] for q, t in zip(queries, response_targets)
            if q["model_input"]["candidate_action"]["piper_intent_id"] == intent)) for intent in (1, 2)},
        "image_files_present": len(image_paths), "history_observations": history_count,
        "anchor_elite_requested_pose_present": count("anchor_elite_requested_pose_present"),
        "anchor_elite_user_events": count("anchor_elite_user_event"),
        "anchor_elite_sent_timestamp_in_decision_bracket": count("anchor_elite_sent_timestamp_in_decision_bracket"),
        "complete_joint_candidate_actions": 0,
        "unknown_elite_action_rejections": unknown_action_rejections,
        "windows_with_later_piper_requests": count("later_piper_requests_before_last_image"),
        "windows_with_piper_boundary_overlap": count("piper_request_brackets_overlap_last_image"),
        "windows_with_later_elite_submissions": count("later_elite_submissions_before_last_image"),
        "windows_with_elite_submission_boundary_overlap": count("elite_submission_brackets_overlap_last_image"),
        "windows_with_elite_sent_timestamp_in_response": count("elite_sent_timestamps_inside_response"),
        "windows_with_any_logged_later_action_indicator": sum(any(r[k] for k in (
            "later_piper_requests_before_last_image", "piper_request_brackets_overlap_last_image",
            "later_elite_submissions_before_last_image", "elite_submission_brackets_overlap_last_image",
            "elite_sent_timestamps_inside_response")) for r in audits),
        "response_image_span_from_cutoff_s": {v: {
            "min": min(r["response_image_span_from_cutoff_s"][v] for r in audits),
            "max": max(r["response_image_span_from_cutoff_s"][v] for r in audits)} for v in VIEWS},
        "checks": {"latest_human_snapshots_unchanged": len(frozen),
                   "whole_episode_folds_preserved": len(folds),
                   "future_and_post_submit_mutation_invariant_episodes": mutation_checks,
                   "candidate_request_changes_without_history_change_episodes": mutation_checks,
                   "existing_missing_history_encoding": "passed",
                   "complete_action_canonical_encoding": "passed"},
        "interpretation": "input/target separation ready for diagnostic request-conditioned prediction; no isolated causal/full Elite-action/counterfactual supervision",
        "elapsed_s": time.monotonic() - started,
    }
    protocol = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), **flags,
        "pack": args.pack.as_posix(), "reference": args.reference.as_posix(),
        "source_root": args.source_root.as_posix(), "image_pack": args.image_pack.as_posix(),
        "history_s": manifest["history_s"], "nominal_prediction_horizon_s": manifest["future_s"],
        "state_encoder": ADAPTER_VERSION, "state_layout_unchanged": STATE_LAYOUT,
        "model_input": "queries.jsonl/model_input only; image paths load pixels, never encode paths or episode IDs",
        "target_only": "response_targets.jsonl, annotation_snapshot.jsonl",
        "audit_only": "timing_audit.jsonl, folds.json, provenance, sample/window IDs",
        "candidate": "Piper request + burst only; unknown Elite is null and six false masks, never zero-imputed",
        "incomplete_action_boundary": "complete_action9 refuses incomplete records; existing world-model forward has no action mask and is not invoked",
        "target_horizon": "original full interval unchanged; actual per-view endpoints are target/audit only; no censoring or relabeling",
        "elite_sent_timestamp": "recorded after move_joint send, not physical onset/completion; previous/pending commands may continue",
        "split": "unchanged ten diagnostic leave-one-episode-out folds; reused development data, all ten already seen by PI05",
        "claims_excluded": ["counterfactual effects", "causal action attribution", "contact/tactile truth", "new policy success", "clean PI05 heldout evaluation"],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    for name, data in (("queries", queries), ("response_targets", response_targets),
                       ("timing_audit", audits), ("annotation_snapshot", frozen)):
        write_jsonl(args.out / f"{name}.jsonl", data)
    write_json(args.out / "folds.json", folds)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "report.json", report)
    print(f"{SCHEMA}: {len(queries)} queries, {len(episodes)} episodes, "
          f"{len(image_paths)} image references; full-action rejects={unknown_action_rejections}; "
          f"{report['elapsed_s']:.2f}s; no training/hardware", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_response_baseline_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("simulation_output/real10_event_source_metadata_v1"))
    parser.add_argument("--image-pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_action_effect_interface_v1"))
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
