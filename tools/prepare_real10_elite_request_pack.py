"""Build the 175-event, diagnostic Elite macro-request response pack.

Reuse the existing state encoder and images. Keep observations, requested
absolute targets, future supervision and audit metadata in different files.
No training, hardware, image decoding, relabeling or canonical delta conversion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time

from audit_real10_elite_command_timing import HORIZON_S, lower_bound
from prepare_real10_action_effect_interface import image_ref, pre_action_input
from prepare_real10_event_windows import (
    finite, frame_id, read_json, read_jsonl, timing, write_json, write_jsonl,
)
from prepare_real_pi05_pack import ADAPTER_VERSION, STATE_LAYOUT


SCHEMA = "real10_elite_macro_request_response_v1"
REQUEST_KIND = "absolute_tcp_target_to_IK_to_move_joint"
HISTORY_S = 1.0
VIEWS = ("side", "top")
OBS_KEYS = {"task_instruction", "nominal_horizon_s", "history"}
REQUEST_KEYS = {"motion_mode", "elite_target_tcp_pose_6d_xyz_mm_rpy_rad",
                "elite_target_valid_6d", "elite_path_speed_sdk_parameter",
                "piper_intent_id", "piper_burst_count"}


def model_observation(pair, rows, image_pack):
    """Only the pre-request prefix is inspected; use the existing state layout."""
    episode, step = pair["source_episode"], pair["step"]
    lo = lower_bound(rows[step])
    history_ids = [frame_id(episode, row["step"]) for row in rows[:step + 1]
                   if lower_bound(row) >= lo - HISTORY_S]
    window = {"source_episode": episode, "task": rows[step]["task"],
              "anchor_frame_id": frame_id(episode, step),
              "anchor_timestamp_s": lo, "history_frame_ids": history_ids}
    old = pre_action_input(window, rows, image_pack, HORIZON_S)
    obs = {key: old[key] for key in OBS_KEYS}
    # Clarify the timestamp semantics without changing the shared state encoder.
    for item in obs["history"]:
        item["elite_pose_query_start_relative_time_s"] = item.pop("elite_pose_relative_time_s")
    return obs, history_ids


def model_request(pair):
    """Offline reconstruction of a requested target, not a measured outcome.

    The later snapshot's target was an argument to IK/move_joint. Its timestamp,
    success/status, path index and joint values with unverified units are NOT
    conditioning fields. This is not the policy's 6D step-delta action API.
    """
    raw = pair["logged_request_audit_only"]
    target = raw["target_tcp_pose_6d_xyz_mm_rpy_rad"]
    if (raw["kind"] != REQUEST_KIND or len(target) != 6
            or not all(finite(x) for x in target)
            or raw["canonical_step_delta_conversion_supported"] is not False):
        raise ValueError("unsupported macro command or invalid target")
    return {"motion_mode": REQUEST_KIND,
            "elite_target_tcp_pose_6d_xyz_mm_rpy_rad": list(target),
            "elite_target_valid_6d": [True] * 6,
            "elite_path_speed_sdk_parameter": raw["speed_parameter"],
            "piper_intent_id": raw["piper_intent_id"],
            "piper_burst_count": raw["piper_burst_count"]}


def response_target(pair, rows, image_pack):
    episode, step = pair["source_episode"], pair["step"]
    lo, hi = pair["request_host_bracket_s"]
    anchor = rows[step]["state"]["elite_tcp_pose_6d"]
    audited = pair["future_response_audit_only"]["frame_ids"]
    expected = [frame_id(episode, r["step"]) for r in rows[step + 1:]
                if lower_bound(r) <= lo + HORIZON_S
                and min(timing(r)[f"{v}_image"] for v in VIEWS) > hi]
    if not expected or expected != audited:
        raise ValueError("audited response interval changed; do not silently rewindow")
    future = []
    for fid in audited:
        row = rows[int(fid.rsplit("/", 1)[1])]
        ts = timing(row)
        pose = row["state"]["elite_tcp_pose_6d"]
        valid = not row["state"]["controller_state"]["elite_pose_stale"]
        future.append({
            "images": {v: {"path": image_ref(image_pack, fid, v),
                            "relative_time_s": ts[f"{v}_image"] - lo} for v in VIEWS},
            "measured_elite_tcp_pose_6d_xyz_mm_rpy_rad": pose,
            "measured_pose_valid_6d": [valid] * 6,
            "pose_query_start_relative_time_s": ts["elite_pose"] - lo,
            "measured_elite_translation_from_anchor_mm": [pose[j] - anchor[j] for j in range(3)],
        })
    return {"future_observations": future,
            "joint_motion_response": None, "joint_motion_response_valid": False,
            "label_scope": "future observed images/Elite motion; NOT wire advance/contact/action success"}


def join_model_inputs(observations, requests):
    if len(observations) != len(requests):
        raise ValueError("observation/request count mismatch")
    result, seen = [], set()
    for index, (obs, req) in enumerate(zip(observations, requests)):
        if (obs["sample_index"] != index or req["sample_index"] != index
                or obs["sample_id"] != req["sample_id"] or obs["sample_id"] in seen):
            raise ValueError("observation/request order or identity mismatch")
        if set(obs["model_observation"]) != OBS_KEYS or set(req["model_request"]) != REQUEST_KEYS:
            raise ValueError("unexpected input fields; do not join audit or response fields")
        seen.add(obs["sample_id"])
        # IDs are only join keys. The returned model dictionaries contain no IDs.
        result.append({"observation": obs["model_observation"], "request": req["model_request"]})
    return result


def load_model_inputs(pack):
    """Load diagnostic macro queries ONLY; never open future targets/audit/folds.

    Image paths resolve from the project root and load pixels, not ID features.
    No model is invoked and no conversion to the policy action API is offered.
    """
    pack = Path(pack)
    manifest = read_json(pack / "manifest.json")
    if manifest["schema"] != SCHEMA or manifest["canonical_action9_compatible"] is not False:
        raise ValueError("not a diagnostic Elite macro-request pack")
    return join_model_inputs(read_jsonl(pack / "observations.jsonl"),
                             read_jsonl(pack / "requests.jsonl"))


def check_pair_source(pair, rows):
    step = pair["step"]
    anchor = rows[step]
    c = anchor["state"]["controller_state"]
    first_step = int(pair["timing_audit"]["first_snapshot_frame_id"].rsplit("/", 1)[1])
    snapshot = rows[first_step]["state"]["controller_state"]
    request = pair["logged_request_audit_only"]
    if (pair["event_id"] != frame_id(pair["source_episode"], step)
            or pair["pairing_status"] != "unique_later_logged_target"
            or c["user_event"]["elite_submit_status"] != "next_submitted"
            or pair["request_host_bracket_s"] != [lower_bound(anchor), timing(anchor)["piper_action"]]
            or pair["anchor_observation"]["measured_tcp_pose_6d"] != anchor["state"]["elite_tcp_pose_6d"]
            or snapshot["elite_requested_tcp_pose_6d"] != request["target_tcp_pose_6d_xyz_mm_rpy_rad"]
            or snapshot["elite_path_command_timestamp"] != pair["timestamp_after_move_joint_call_s"]
            or request["piper_intent_id"] != {0: 1, 1: 2, -1: 0}[anchor["reference_action"]["piper_step_command"]]
            or request["piper_burst_count"] != anchor["reference_action"]["piper_burst_count"]):
        raise ValueError(f"source/audited command mismatch: {pair['event_id']}")


def isolation_check(pair, rows, image_pack, expected_observation, expected_request):
    """One integrated leakage check per episode; no new smoke-test suite."""
    changed = deepcopy(rows)
    step = pair["step"]
    changed[step]["reference_action"]["elite_tcp_delta_6d"] = [999.0] * 6
    current = changed[step]["state"]["controller_state"]
    current["piper_busy"] = not current["piper_busy"]
    current["elite_requested_tcp_pose_6d"] = [999.0] * 6
    current["elite_path_status"] = "FAKE_POST_SUBMIT_RESULT"
    for row in changed[step + 1:]:
        row["state"]["elite_tcp_pose_6d"] = [999.0] * 6
        row["reference_action"]["elite_tcp_delta_6d"] = [999.0] * 6
        row["side_image"] = "FAKE_FUTURE_IMAGE"
    if model_observation(pair, changed, image_pack)[0] != expected_observation:
        raise ValueError("future or current post-submit data affected observation input")
    altered = deepcopy(pair)
    altered["timing_audit"] = {"FAKE_FUTURE": 999}
    altered["future_response_audit_only"] = {"FAKE_TARGET": 999}
    raw = altered["logged_request_audit_only"]
    raw["path_index_audit_only"] = 999
    raw["target_joints_sdk_native_unit_unverified"] = [999.0] * 6
    raw["target_offset_from_anchor_xyz_mm_audit_only"] = [999.0] * 3
    if model_request(altered) != expected_request:
        raise ValueError("audit metadata leaked into macro request")
    raw["target_tcp_pose_6d_xyz_mm_rpy_rad"][0] += 1
    if (model_request(altered) == expected_request
            or model_observation(altered, rows, image_pack)[0] != expected_observation):
        raise ValueError("requested target is not independent of observation")


def build(args):
    started = time.monotonic()
    if args.out.exists():
        raise FileExistsError(f"preserve existing output; choose a fresh --out: {args.out}")
    audit_report = read_json(args.audit / "report.json")
    pairs = read_jsonl(args.audit / "command_pairs.jsonl")
    episodes = audit_report["source_episodes"]
    image_manifest = read_json(args.image_pack / "manifest.json")
    if (audit_report["schema"] != "real10_elite_command_timing_audit_v1"
            or len(pairs) != 175 or len(episodes) != 10
            or image_manifest["adapter_version"] != ADAPTER_VERSION
            or episodes != image_manifest["source_episodes"]
            or image_manifest["training_use"]["split_strategy"] != "all_10_episodes_train_no_validation"):
        raise ValueError("unexpected source audit, allowlist or original split")
    rows_by_episode = {e: read_jsonl(args.source_root / e / "records.jsonl") for e in episodes}
    observations, requests, targets, audit_rows, groups = [], [], [], [], []
    checked_episodes, image_paths = set(), set()
    for index, pair in enumerate(pairs):
        episode, step = pair["source_episode"], pair["step"]
        rows = rows_by_episode[episode]
        check_pair_source(pair, rows)
        obs, history_ids = model_observation(pair, rows, args.image_pack)
        req = model_request(pair)
        target = response_target(pair, rows, args.image_pack)
        identity = {"sample_index": index, "sample_id": pair["event_id"]}
        observations.append({**identity, "model_observation": obs})
        requests.append({**identity, "model_request": req})
        targets.append({**identity, **target})
        groups.append({**identity, "source_episode": episode, "task": pair["task"],
                       "original_source_split": "train", "split_group": episode})
        audit_rows.append({**identity, "source_episode": episode,
                           "history_frame_ids": history_ids,
                           "history_boundary_truncated": lower_bound(rows[0]) > lower_bound(rows[step]) - HISTORY_S,
                           "source_command_pair": pair})
        for item in obs["history"]:
            if len(item["state_32"]) != 32 or len(item["state_valid_32"]) != 32:
                raise ValueError("existing state layout changed")
            if any(item["images"][v]["relative_time_s"] > 0 for v in VIEWS):
                raise ValueError("future image in observation history")
        anchor_state = obs["history"][-1]
        if not pair["anchor_observation"]["prior_controller_history_valid"]:
            if (any(anchor_state["state_32"][j] != 0 for j in (6, 16, 27))
                    or any(anchor_state["state_valid_32"][j] for j in (6, 16))):
                raise ValueError("missing-history fill/mask changed")
        for item in obs["history"] + target["future_observations"]:
            image_paths.update(item["images"][v]["path"] for v in VIEWS)
        if episode not in checked_episodes:
            isolation_check(pair, rows, args.image_pack, obs, req)
            checked_episodes.add(episode)

    missing = [p for p in image_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced images missing; first: {missing[0]}")
    # Inherit ONLY the whole-episode heldout order. The 45-window indices cannot
    # be reused for a different 175-event pack. This is still development LOEO.
    reference_folds = read_json(args.reference_folds)
    heldout = [f["heldout_episode"] for f in reference_folds]
    if len(heldout) != len(episodes) or set(heldout) != set(episodes):
        raise ValueError("reference episode fold set changed")
    folds = [{"heldout_episode": e,
              "train_indices": [g["sample_index"] for g in groups if g["source_episode"] != e],
              "test_indices": [g["sample_index"] for g in groups if g["source_episode"] == e]}
             for e in heldout]
    if sorted(i for fold in folds for i in fold["test_indices"]) != list(range(len(pairs))):
        raise ValueError("incomplete or duplicate episode-heldout coverage")
    counts = {
        "samples": len(pairs), "episodes": len(episodes),
        "by_task": dict(Counter(g["task"] for g in groups)),
        "history_observations": sum(len(o["model_observation"]["history"]) for o in observations),
        "future_observations": sum(len(t["future_observations"]) for t in targets),
        "future_frame_count_distribution": dict(Counter(str(len(t["future_observations"])) for t in targets)),
        "unique_image_references": len(image_paths), "missing_image_references": 0,
        "anchor_missing_controller_history": sum(not p["anchor_observation"]["prior_controller_history_valid"] for p in pairs),
        "history_boundary_truncated": sum(a["history_boundary_truncated"] for a in audit_rows),
        "later_request_windows_retained": sum(bool(p["future_response_audit_only"]["later_requests_before_last_image"]) for p in pairs),
        "joint_motion_labels_copied_or_generated": 0,
        "canonical_delta_actions_created": 0,
        "diagnostic_episode_folds": len(folds),
        "piper_intent_counts": dict(Counter(str(r["model_request"]["piper_intent_id"]) for r in requests)),
    }
    manifest = {
        "schema": SCHEMA, "purpose": "offline_observational_macro_request_response_supervision",
        "source_audit": args.audit.as_posix(), "source_root": args.source_root.as_posix(),
        "source_episodes": episodes, "source_split": "all_train_no_validation_unchanged",
        "image_pack": args.image_pack.as_posix(), "image_path_base": "repository_root",
        "observation_file": "observations.jsonl", "request_file": "requests.jsonl",
        "response_target_file": "response_targets.jsonl", "audit_file": "timing_audit.jsonl",
        "group_file": "sample_groups.jsonl", "diagnostic_fold_file": "folds.json",
        "history_s": HISTORY_S, "nominal_horizon_s": HORIZON_S,
        "state_encoder": ADAPTER_VERSION, "state_layout": STATE_LAYOUT,
        "normalization": "raw values; no fit; any future normalization must use training folds only",
        "request_semantics": {"motion_mode": REQUEST_KIND,
                              "pose_units": "xyz_mm_rpy_rad",
                              "speed": "unchanged SDK parameter; installed SDK says joint-speed percent, NOT mm/s",
                              "target_provenance": "request argument logged after blocking call, not future measured pose",
                              "canonical_action9_compatible": False},
        "timing": "host timestamps and pose-query-start reference; loop order proves pre-request, no exact onset/completion",
        "sampling": "all 175 audited commands retained, no outcome/visibility filtering, no new labels",
        "fold_scope": "same ten heldout episode names/order, indices rebuilt for 175 events; development only, not independent PI05 test",
        "reference_fold_file": args.reference_folds.as_posix(),
        "canonical_action9_compatible": False,
        "policy_input_allowed": False, "policy_training_ready": False,
        "formal_data_allowed": False, "real_system_validated": False,
        "training_executed": False, "hardware_executed": False, "counts": counts,
    }
    args.out.mkdir(parents=True)
    for name, values in (("observations", observations), ("requests", requests),
                         ("response_targets", targets), ("timing_audit", audit_rows),
                         ("sample_groups", groups)):
        write_jsonl(args.out / f"{name}.jsonl", values)
    write_json(args.out / "folds.json", folds)
    write_json(args.out / "manifest.json", manifest)
    inputs = load_model_inputs(args.out)
    if inputs != join_model_inputs(observations, requests):
        raise ValueError("saved model input roundtrip mismatch")
    validation = {
        "schema": SCHEMA, "status": "passed", "counts": counts,
        "source_pair_checks": len(pairs), "input_roundtrip_samples": len(inputs),
        "future_and_audit_isolation_checks": len(checked_episodes),
        "missing_history_fill_and_mask_checked": True,
        "episode_fold_coverage_checked": True,
        "image_check_scope": "file existence only; no decode, new render or visual acceptance",
        "all_commands_retained": True, "images_decoded": 0, "models_loaded": 0,
        "training_executed": False, "hardware_executed": False,
        "runtime_s": time.monotonic() - started,
    }
    write_json(args.out / "validation.json", validation)
    print({"out": args.out.as_posix(), **validation})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("simulation_output/real10_elite_command_timing_audit_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("simulation_output/real10_event_source_metadata_v1"))
    parser.add_argument("--image-pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--reference-folds", type=Path, default=Path("simulation_output/real10_pre_action_label_retest_v1/folds.json"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_elite_request_response_pack_v1"))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
