"""Pure-array, all-pair source isolation audit for the frozen LIBERO study.

No file/network/video/model access. The caller owns hash-pinned source identity,
fixed selected IDs, complete-row decoding and upstream family provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import json
import math
import re

import numpy as np

from prepare_libero_source_pair import duplicate_audit


VIEWS = ("observation.images.image", "observation.images.image2")
PARTITIONS = {"train", "validation", "old_development"}
FPS = 10.0


def _integer(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        raise ValueError(f"{label} must be a finite nonnegative integer")
    return int(value)


def _intervals(metadata, count):
    result = {}
    for view in VIEWS:
        prefix = f"videos/{view}/"
        chunk = _integer(metadata.get(prefix + "chunk_index"), prefix + "chunk_index")
        shard = _integer(metadata.get(prefix + "file_index"), prefix + "file_index")
        start, stop = metadata.get(prefix + "from_timestamp"), metadata.get(prefix + "to_timestamp")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in (start, stop)):
            raise ValueError("video interval timestamps must be finite numbers")
        if start < 0 or stop <= start or not math.isclose(stop-start, count/FPS, abs_tol=2e-5, rel_tol=0):
            raise ValueError("video interval must cover the complete 10 Hz episode")
        result[view] = {"chunk_index": chunk, "file_index": shard,
                        "from_timestamp": float(start), "to_timestamp": float(stop)}
    return result


class _Groups:
    def __init__(self, ids):
        self.parent = {eid: eid for eid in ids}

    def find(self, eid):
        while self.parent[eid] != eid:
            eid = self.parent[eid]
        return eid

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        self.parent[max(a, b)] = min(a, b)

    def members(self):
        groups = {}
        for eid in sorted(self.parent):
            groups.setdefault(self.find(eid), []).append(eid)
        return list(groups.values())


def audit_sequences(episodes, sequences, known_family_links=()):
    """Return JSON-safe duplicate/isolation evidence for every supplied pair.

    ``episodes`` contains dictionaries with episode_index, partition (train,
    validation or old_development), metadata and source_trajectory_id.
    ``sequences[eid]`` is (N x 8 float32 state, N x 7 float32 action,
    list of N lowercase 128-hex strings concatenating two RGB frame SHA256s).
    ``known_family_links`` contains two-ID sequences whose externally verified
    family/derivative relationship is supplied by the caller. Their transitive
    closure is honored; absent links mean unknown, never independent families.

    This helper audits all supplied IDs, not the frozen 12-ID selection/count.
    The caller must enforce that separate provenance gate. Old development
    pairs are descriptive; any new-vs-old or train-vs-validation overlap blocks.
    Matching new episodes within one partition are grouped, never dropped.
    """
    if not isinstance(episodes, (list, tuple)) or len(episodes) < 2:
        raise ValueError("at least two episode records are required")
    if not isinstance(sequences, Mapping):
        raise ValueError("sequences must be an episode-ID mapping")
    records, intervals = {}, {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("episode must be an object")
        eid = _integer(episode.get("episode_index"), "episode_index")
        if eid in records:
            raise ValueError("duplicate episode_index")
        partition = episode.get("partition")
        if not isinstance(partition, str) or partition not in PARTITIONS:
            raise ValueError("unknown episode partition")
        identity = episode.get("source_trajectory_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("source_trajectory_id must be a nonempty string")
        metadata = episode.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("episode metadata must be an object")
        if "episode_index" in metadata and _integer(metadata["episode_index"], "metadata episode_index") != eid:
            raise ValueError("metadata episode_index differs")
        count = _integer(metadata.get("length"), "metadata length")
        if count < 7:
            raise ValueError("complete episode must contain at least seven rows")
        if "record_count" in episode and _integer(episode["record_count"], "record_count") != count:
            raise ValueError("record_count differs from metadata length")
        if eid not in sequences:
            raise ValueError("sequence missing for episode")
        sequence = sequences[eid]
        if not isinstance(sequence, (list, tuple)) or len(sequence) != 3:
            raise ValueError("state/action/pixel hash sequence triple required")
        state, action, frame_hashes = sequence
        for label, array, width in (("state", state, 8), ("action", action, 7)):
            if (not isinstance(array, np.ndarray) or array.dtype != np.float32
                    or array.shape != (count, width) or not np.isfinite(array).all()):
                raise ValueError(f"{label} must be a finite complete raw float32 array")
        if (not isinstance(frame_hashes, list) or len(frame_hashes) != count
                or any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{128}", h) is None for h in frame_hashes)):
            raise ValueError("complete two-view frame hashes must be lowercase 128-hex strings")
        records[eid] = episode
        intervals[eid] = _intervals(metadata, count)
    if any(type(eid) is not int for eid in sequences) or set(sequences) != set(records):
        raise ValueError("sequence keys must exactly match integer episode IDs")

    families = _Groups(records)
    links, seen_links = [], set()
    if not isinstance(known_family_links, (list, tuple)):
        raise ValueError("known_family_links must be a sequence of two-ID links")
    for link in known_family_links:
        if not isinstance(link, (tuple, list)) or len(link) != 2:
            raise ValueError("known family link must contain two episode IDs")
        first, second = [_integer(eid, "family episode_index") for eid in link]
        if first not in records or second not in records or first == second:
            raise ValueError("known family link references an absent or identical episode")
        pair = tuple(sorted((first, second)))
        if pair in seen_links:
            raise ValueError("duplicate known family link")
        seen_links.add(pair)
        links.append(list(pair))
        families.union(first, second)

    new_ids = [eid for eid, e in records.items() if e["partition"] != "old_development"]
    groups = _Groups(new_ids)
    pairs, blocking_pairs, grouping_pairs = [], [], []
    for first, second in combinations(sorted(records), 2):
        a, b = records[first], records[second]
        pa, pb = a["partition"], b["partition"]
        old_pair = pa == pb == "old_development"
        isolation_required = not old_pair and pa != pb
        duplicate = duplicate_audit(sequences[first], sequences[second])
        matches = duplicate["cross_split_matching_windows"]
        reasons = [key for key, count in matches.items() if count]
        overlapping_intervals = []
        for view in VIEWS:
            ia, ib = intervals[first][view], intervals[second][view]
            same_shard = (ia["chunk_index"], ia["file_index"]) == (ib["chunk_index"], ib["file_index"])
            overlap = min(ia["to_timestamp"], ib["to_timestamp"]) - max(ia["from_timestamp"], ib["from_timestamp"])
            if same_shard and overlap > 1e-6:
                overlapping_intervals.append({"view": view, "first": ia, "second": ib,
                                              "overlap_seconds": float(overlap)})
        if overlapping_intervals:
            reasons.append("overlapping_source_video_intervals")
        if a["source_trajectory_id"] == b["source_trajectory_id"]:
            reasons.append("same_source_trajectory_id")
        if families.find(first) == families.find(second):
            reasons.append("known_family_or_derivative_link")
        blocked = isolation_required and bool(reasons)
        grouped = not old_pair and not isolation_required and bool(reasons)
        if blocked:
            blocking_pairs.append({"episode_indices": [first, second], "reasons": reasons})
        if grouped:
            groups.union(first, second)
            grouping_pairs.append({"episode_indices": [first, second], "reasons": reasons})
        pairs.append({
            "episode_indices": [first, second], "partitions": [pa, pb],
            "relationship": "old_development_only" if old_pair else
                "new_vs_old_development" if "old_development" in (pa, pb) else
                "new_cross_partition" if isolation_required else "new_same_partition",
            "isolation_required": isolation_required,
            "sequence_duplicate_audit": duplicate,
            "overlapping_video_intervals": overlapping_intervals,
            "match_reasons": reasons, "blocking_reasons": reasons if blocked else [],
            "decision": "blocked_leakage" if blocked else "same_partition_grouped" if grouped
                else "old_development_descriptive_only" if old_pair else "no_detected_match",
        })
    partition_groups = []
    for members in groups.members():
        partition_groups.append({"group_id": f"source_group_{min(members)}",
                                 "partition": records[members[0]]["partition"],
                                 "episode_indices": members})
    report = {
        "schema": "libero_action_study_source_sequence_audit_v1",
        "status": "rejected_source_isolation" if blocking_pairs else "passed_declared_duplicate_checks_only",
        "episode_count": len(records), "new_episode_count": len(new_ids),
        "old_development_episode_count": len(records)-len(new_ids),
        "pair_count": len(pairs), "pairs": pairs,
        "blocking_pairs": blocking_pairs, "blocking_pair_count": len(blocking_pairs),
        "same_partition_grouping_pairs": grouping_pairs,
        "same_partition_groups": partition_groups,
        "new_duplicate_or_linked_groups_with_multiple_episodes": sum(len(g["episode_indices"]) > 1 for g in partition_groups),
        "new_group_count": len(partition_groups),
        "new_group_counts_by_partition": {p: sum(g["partition"] == p for g in partition_groups)
                                           for p in ("train", "validation")},
        "known_family_links": sorted(links),
        "known_family_link_components": [g for g in families.members() if len(g) > 1],
        "family_provenance": "externally_supplied_links_not_exhaustive" if links else "unknown",
        "family_independence_verified": False,
        "group_scope": "detected sequence duplicates, interval/source identity overlaps and supplied family links only; not verified independent family or effective sample count",
        "legacy_pair_status_scope": "nested duplicate status and cross_split_matching_windows retain the original helper field names even for same-partition or old-only pairs",
        "window_length": 7, "connecting_action_count": 6,
        "terminal_action_has_no_successor_and_is_excluded_from_windows": True,
        "selected_ids_changed": False, "normalization_fitted": False,
        "training_ready": False, "formal_data_allowed": False,
        "checkpoint_training_overlap_unknown": True, "features_extracted": False,
        "policy_loaded": False, "optimizer_steps": 0,
    }
    # Keep output strict JSON; never serialize NaN/Inf as apparently valid evidence.
    json.dumps(report, allow_nan=False)
    return report
