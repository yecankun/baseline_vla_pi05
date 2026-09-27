"""Read-only, hash-bound fixed12 LIBERO source adapter; no encoder or training.

The two old development episodes are rechecked as isolation evidence, never
returned as feature inputs. Source groups are descriptive leakage groups, not
independent families. No metadata is added to the native observation boundary.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    from libero_action_study_source_checks import audit_sequences
    from pi05_libero_world_model_adapter import DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, adapt_native_record, sha256_file
    from probe_pi05_libero_public_features import ORIENTATION, ROW_KEYS, checked_hash, checked_int, local_file, read_json
else:
    from .libero_action_study_source_checks import audit_sequences
    from .pi05_libero_world_model_adapter import DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, adapt_native_record, sha256_file
    from .probe_pi05_libero_public_features import ORIENTATION, ROW_KEYS, checked_hash, checked_int, local_file, read_json


SOURCE_SCHEMA = "libero_action_study_sources_v1"
TASK_REGISTRY = [{"task_id": 9, "source_task_index": 39,
                  "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]
FIXED_EPISODES = tuple(
    {"episode_index": eid, "partition": partition, "record_count": count}
    for partition, entries in (
        ("train", ((1633, 193), (1674, 119), (1419, 135), (1312, 145),
                   (1518, 133), (1520, 134), (1531, 142), (1690, 133))),
        ("validation", ((1530, 126), (1476, 125), (1458, 130), (1566, 143))),
    ) for eid, count in entries
)
OLD_EPISODES = ({"episode_index": 1400, "partition": "old_development", "record_count": 140},
                {"episode_index": 1402, "partition": "old_development", "record_count": 173})
SOURCE_ROW_KEYS = ROW_KEYS | {"timestamp", "index"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, name):
    path = local_file(root, name)
    _require(path.relative_to(root).as_posix() == name, "source path must be canonical without aliases")
    return path


def close_source(source: dict) -> None:
    """Same structure/cleanup contract as the unchanged two-episode builder."""
    for episode in source.get("episodes", []):
        images = episode.get("images")
        if isinstance(images, np.memmap) and not images._mmap.closed:
            images._mmap.close()


def source_unchanged(source: dict) -> bool:
    """Rehash every verified source output; raise on mutation, never repair."""
    for name, digest in source["verified_files"].items():
        _require(sha256_file(_path(source["root"], name)) == digest,
                 f"source changed after validation: {name}")
    return True


def _records(root, entry, outputs):
    eid, count = entry["episode_index"], entry["record_count"]
    for stem, filename in (("images", "images.npy"), ("records", "records.json"),
                           ("decoded_frames", "decoded_frames.json")):
        name = entry.get(f"{stem}_path")
        _require(name == f"episodes/episode_{eid}/{filename}" and name in outputs,
                 f"episode {stem} path must match complete hash-bound inventory")
        if stem != "decoded_frames":
            _require(outputs[name] == checked_hash(entry.get(f"{stem}_sha256")), "declaration/output hash mismatch")
    rows = read_json(_path(root, entry["records_path"]))
    _require(isinstance(rows, list) and len(rows) == count, "complete episode rows truncated or extended")
    md = entry.get("metadata", {})
    _require(checked_int(md.get("episode_index"), "metadata episode_index") == eid
             and checked_int(md.get("length"), "metadata length") == count, "complete episode metadata count/ID differs")
    start = checked_int(md.get("dataset_from_index"), "metadata global start")
    _require(checked_int(md.get("dataset_to_index"), "metadata global end") == start + count,
             "complete episode metadata interval differs")
    adapted = []
    for frame, record in enumerate(rows):
        _require(isinstance(record, dict) and set(record) == SOURCE_ROW_KEYS,
                 "source record unknown/missing fields; only native state/action/task are allowed")
        for key, width in (("observation.state", 8), ("action", 7)):
            values = record[key]
            _require(isinstance(values, list) and len(values) == width
                     and all(type(v) in {int, float} and np.isfinite(v) for v in values),
                     "raw native vectors require finite numbers without bool/string/imputation")
        row = adapt_native_record({key: record[key] for key in ROW_KEYS}, TASK_REGISTRY)
        _require(checked_int(record["episode_index"], "row episode_index") == eid
                 and checked_int(record["frame_index"], "row frame_index") == frame
                 and checked_int(record["task_index"], "row task_index") == 39
                 and checked_int(record["index"], "row global index") == start + frame
                 and row["task_id"] == 9, "source row order/ID gap/reset/task mismatch")
        stamp = record["timestamp"]
        _require(type(stamp) in {int, float} and np.isfinite(stamp)
                 and np.isclose(stamp, frame / 10, atol=2e-5, rtol=0), "consecutive native 10Hz timestamps required")
        _require(row["state_valid"].all()
                 and np.array_equal(row["state"], np.asarray(record["observation.state"], dtype=np.float32))
                 and np.array_equal(row["action"], np.asarray(record["action"], dtype=np.float32)),
                 "native state/action changed or imputed by adapter")
        adapted.append(row)
    frames = read_json(_path(root, entry["decoded_frames_path"]))
    _require(isinstance(frames, list) and len(frames) == 2, "exactly two native decoded views required")
    for key, evidence in zip(IMAGE_KEYS, frames, strict=True):
        _require(isinstance(evidence, dict) and set(evidence) == {"key", "from_timestamp", "to_timestamp", "frames"},
                 "unknown/missing decoded-view evidence fields")
        start, stop = md.get(f"videos/{key}/from_timestamp"), md.get(f"videos/{key}/to_timestamp")
        _require(all(type(v) in {int, float} and np.isfinite(v) for v in (start, stop))
                 and start >= 0 and np.isclose(stop - start, count / 10, atol=2e-5, rtol=0),
                 "complete native video interval required")
        _require(evidence["key"] == key and evidence["from_timestamp"] == start and evidence["to_timestamp"] == stop,
                 "decoded native view order/orientation/interval differs")
        _require(isinstance(evidence["frames"], list) and len(evidence["frames"]) == count, "decoded frame count differs")
        for frame, record in enumerate(evidence["frames"]):
            _require(isinstance(record, dict) and set(record) == {"frame_index", "video_timestamp", "decoded_rgb_sha256"},
                     "unknown/missing decoded-frame evidence fields")
            stamp = record["video_timestamp"]
            _require(checked_int(record["frame_index"], "decoded frame_index") == frame
                     and type(stamp) in {int, float} and np.isfinite(stamp)
                     and abs(stamp - (start + frame / 10)) <= .001 and stamp < stop - .001,
                     "decoded PTS order or half-open native interval differs")
            checked_hash(record["decoded_rgb_sha256"])
    state = np.asarray([row["state"] for row in adapted], dtype=np.float32)
    action = np.asarray([row["action"] for row in adapted], dtype=np.float32)
    hashes = [frames[0]["frames"][i]["decoded_rgb_sha256"] + frames[1]["frames"][i]["decoded_rgb_sha256"] for i in range(count)]
    return rows, adapted, frames, (state, action, hashes)


def validate_source(root: Path, plan: dict) -> dict:
    """Validate pinned outputs/complete rows/RGB/isolation before any encoder.

    The caller must pin its entire feature plan. This boundary additionally
    enforces fixed selection semantics; synthetic tests replace digests and
    small fixture counts explicitly and are not public-source evidence.
    """
    root = Path(root).resolve()
    _require(isinstance(plan, dict), "feature plan must be an object")
    expected = list(FIXED_EPISODES)
    _require(plan.get("episodes") == expected, "feature plan fixed episode IDs/counts/order differ")
    for episode in plan["episodes"]:
        checked_int(episode["episode_index"], "plan episode_index")
        checked_int(episode["record_count"], "plan record_count", 7)
    split = {f"{partition}_episode_indices": [e["episode_index"] for e in expected if e["partition"] == partition]
             for partition in ("train", "validation")}
    _require(plan.get("selected_split") == split, "feature plan fixed split differs")
    report_hash = checked_hash(plan.get("source_report_sha256"))
    _require(sha256_file(_path(root, "report.json")) == report_hash, "source report differs from externally pinned SHA256")
    report = read_json(root / "report.json")
    _require(report.get("schema") == SOURCE_SCHEMA and report.get("status") == "passed_declared_source_checks"
             and report.get("source_integrity_and_declared_duplicate_checks_passed") is True, "passed source integrity audit required")
    _require(report.get("source") == {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION},
             "only pinned native public demonstration identity allowed")
    _require(report.get("orientation") == ORIENTATION and report.get("selected_split") == split, "source orientation/fixed split differs")
    new_count = sum(e["record_count"] for e in expected)
    old_count = sum(e["record_count"] for e in OLD_EPISODES)
    _require(report.get("selected_new_rows") == new_count and report.get("old_development_rows") == old_count
             and report.get("decoded_selected_view_frames") == 2 * (new_count + old_count), "source complete row accounting differs")
    for key in ("new_ids_replaced", "feature_pack_complete", "normalization_fitted", "policy_loaded", "training_ready",
                "formal_data_allowed", "family_independence_verified"):
        _require(report.get(key) is False, f"diagnostic scope flag changed: {key}")
    _require(report.get("checkpoint_training_overlap_unknown") is True
             and type(report.get("features_extracted")) is int and report["features_extracted"] == 0
             and type(report.get("optimizer_steps")) is int and report["optimizer_steps"] == 0,
             "source feature/optimizer/overlap boundaries changed")
    identity = report.get("identity", {})
    _require(identity.get("selection_sha256") == checked_hash(plan.get("selection_sha256")), "source selection identity differs")
    inputs = identity.get("input_sha256", {})
    for filename, digest in (("docs/libero-action-ablation-study-plan-v1.json", plan.get("study_plan_sha256")),
                             ("simulation_output/libero_action_study_task_index_v1/selection.json", plan.get("selection_sha256"))):
        matches = [value for name, value in inputs.items() if name.endswith("/" + filename)]
        _require(matches == [checked_hash(digest)], "source study plan/selection provenance missing or ambiguous")
    outputs = report.get("output_sha256")
    _require(isinstance(outputs, dict) and outputs and "identity.json" in outputs, "complete source output inventory required")
    verified = {"report.json": report_hash}
    for name, digest in outputs.items():
        _require(name not in {"report.json", "status.json", "active_invocation.lock"}, "mutable/self-referential source inventory")
        _require(sha256_file(_path(root, name)) == checked_hash(digest), f"source output hash mismatch: {name}")
        verified[name] = digest
    _require(read_json(_path(root, "identity.json")) == identity, "source identity sidecar differs")
    duplicate_hash = checked_hash(plan.get("duplicate_audit_sha256"))
    _require(outputs.get("duplicate_audit.json") == duplicate_hash, "source duplicate audit SHA differs")
    duplicate = read_json(_path(root, "duplicate_audit.json"))
    _require(report.get("duplicate_audit") == duplicate, "standalone/embedded duplicate audit differs")
    _require(duplicate.get("status") == "passed_declared_duplicate_checks_only"
             and duplicate.get("blocking_pair_count") == 0 and duplicate.get("known_family_links") == [],
             "source isolation gate failed or new family claims were added")
    declarations = report.get("episodes")
    all_expected = expected + list(OLD_EPISODES)
    _require(isinstance(declarations, list) and len(declarations) == len(all_expected), "exact fixed12 plus old audit declarations required")
    source = {"root": root, "report": report, "report_sha256": report_hash, "plan": deepcopy(plan),
              "verified_files": verified, "episodes": [], "task_registry": deepcopy(TASK_REGISTRY),
              "selected_split": deepcopy(split), "source_groups": deepcopy(duplicate.get("same_partition_groups"))}
    sequences = {}
    try:
        for entry, choice in zip(declarations, all_expected, strict=True):
            _require(isinstance(entry, dict) and all(entry.get(key) == value for key, value in choice.items()),
                     "source declarations fixed episode IDs/counts/order differ")
            eid = checked_int(entry["episode_index"], "episode_index")
            count = checked_int(entry["record_count"], "record_count", 7)
            _require(entry.get("complete_episode") is True
                     and entry.get("source_trajectory_id") == f"{DEMO_REPO}@{DEMO_REVISION}/episode{eid}",
                     "complete native source trajectory identity required")
            rows, adapted, frames, sequences[eid] = _records(root, entry, outputs)
            if choice["partition"] == "old_development":
                continue  # Audit evidence only: no old image memmaps or feature inputs.
            images = np.load(_path(root, entry["images_path"]), mmap_mode="r", allow_pickle=False)
            item = {"declaration": deepcopy(entry), "records": rows, "adapted": adapted, "images": images}
            source["episodes"].append(item)
            _require(isinstance(images, np.memmap) and not images.flags.writeable
                     and images.dtype == np.uint8 and images.shape == (count, 2, 256, 256, 3),
                     "complete source RGB requires readonly uint8 memmap [N,2,256,256,3]")
            for frame in range(count):
                for view in range(2):
                    _require(hashlib.sha256(images[frame, view].tobytes()).hexdigest()
                             == frames[view]["frames"][frame]["decoded_rgb_sha256"], "source decoded RGB hash differs")
        recalculated = audit_sequences(declarations, sequences, known_family_links=())
        _require(recalculated == duplicate, "recomputed all-pair/signature/group evidence differs")
        groups = {eid: group for group in duplicate["same_partition_groups"] for eid in group["episode_indices"]}
        _require(set(groups) == {e["episode_index"] for e in expected}, "source group membership differs")
        for item in source["episodes"]:
            declaration = item["declaration"]
            group = groups[declaration["episode_index"]]
            _require(group["partition"] == declaration["partition"], "leakage group crosses fixed split")
            declaration.update(task_id=9, source_task_index=39, leakage_group_id=group["group_id"])
        source_unchanged(source)
        return source
    except BaseException:
        close_source(source)
        raise
