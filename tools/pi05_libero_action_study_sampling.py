"""Frozen shared draw lists for the fixed LIBERO action study, no training.

This pure CPU helper receives already hash-verified train-window metadata. The
caller binds the complete eight-episode/1086-window cache and audited groups;
small subsets of those train IDs are also supported for synthetic tests. It
does not read features, construct policy inputs, create models or optimize.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

import torch


SCHEMA = "pi05_libero_action_study_shared_draw_manifest_v1"
SEEDS = (20260912, 20260913, 20260914)
TRAIN_EPISODE_IDS = frozenset((1633, 1674, 1419, 1312, 1518, 1520, 1531, 1690))
WINDOW_KEYS = frozenset(("episode_index", "history", "actions", "targets"))
ARMS = ("observed_action", "normalized_zero_action")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, *, low=0, high=2**63 - 1):
    _require(type(value) is int and low <= value <= high,
             f"{name} must be an integer in [{low}, {high}], not bool/coerced numeric")
    return value


def canonical_sha256(value) -> str:
    """SHA256 of strict sorted-key, compact ASCII JSON, not a file byte hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _validated_inputs(train_windows, groups):
    _require(type(train_windows) is list and train_windows, "nonempty train_windows list required")
    episode_windows, row_owners, seen_windows, canonical_windows = {}, {}, set(), []
    for window_index, window in enumerate(train_windows):
        _require(type(window) is dict and set(window) == WINDOW_KEYS,
                 "window fields must be exactly episode_index/history/actions/targets; no extra metadata")
        eid = _integer(window["episode_index"], "window episode_index")
        _require(eid in TRAIN_EPISODE_IDS, "window episode is not a frozen training episode; validation/old/unknown excluded")
        row = {"episode_index": eid}
        for key, width in (("history", 4), ("actions", 3), ("targets", 3)):
            values = window[key]
            _require(type(values) is list and len(values) == width,
                     f"{key} must be exactly {width} native global row indices")
            values = [_integer(v, f"{key} index") for v in values]
            _require(all(b == a + 1 for a, b in zip(values, values[1:])),
                     "window rows must be consecutive in native 4/3 layout")
            row[key] = values
        _require(row["actions"][0] == row["history"][-1]
                 and row["targets"] == [v + 1 for v in row["actions"]],
                 "window action/successor alignment differs from native 4/3 contract")
        signature = (eid, *row["history"], *row["actions"], *row["targets"])
        _require(signature not in seen_windows, "duplicate train window cannot inflate sampling support")
        seen_windows.add(signature)
        for global_row in set(row["history"] + row["actions"] + row["targets"]):
            _require(row_owners.setdefault(global_row, eid) == eid, "global row is shared across episode identities")
        episode_windows.setdefault(eid, []).append(window_index)
        canonical_windows.append(row)
    _require(type(groups) is dict and groups, "nonempty training-group mapping required")
    canonical_groups, memberships = {}, {}
    for group_id, episode_ids in groups.items():
        _require(type(group_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", group_id),
                 "group_id must be a nonempty bounded ASCII metadata identifier")
        _require(type(episode_ids) is list and episode_ids, "each training group must contain a nonempty episode list")
        checked = [_integer(eid, "group episode_index") for eid in episode_ids]
        _require(len(set(checked)) == len(checked), "duplicate episode inside training group")
        for eid in checked:
            _require(eid in TRAIN_EPISODE_IDS and eid in episode_windows,
                     "training group contains validation/old/unknown/windowless episode")
            _require(eid not in memberships, "episode belongs to more than one training group")
            memberships[eid] = group_id
        canonical_groups[group_id] = sorted(checked)
    _require(set(memberships) == set(episode_windows), "training groups must cover every window episode exactly once")
    canonical_groups = {key: canonical_groups[key] for key in sorted(canonical_groups)}
    return canonical_windows, canonical_groups, episode_windows


def _rng_sha256(generator):
    return hashlib.sha256(bytes(generator.get_state().tolist())).hexdigest()


def build_draw_manifest(train_windows: list[dict], groups: dict[str, list[int]], seed: int,
                        batches: int = 200, batch_size: int = 16) -> dict:
    """Materialize uniform group -> episode -> window draws with replacement.

    ``window_index`` is the position in the original ``train_windows`` list,
    never a global row index or an index after sorting/truncation. Group names
    and episode IDs are sorted; windows within each episode retain original
    list order. Each draw consumes exactly three scalar CPU ``torch.randint``
    calls, including levels containing a single choice. A private generator
    initialized directly from the prescribed seed never touches global RNG.

    The returned one list is shared by both arms. No arm-specific sampling or
    group-conditioned model input is offered. Making this prospective schedule
    is not authorization to execute any optimizer step.
    """
    _integer(seed, "seed")
    _require(seed in SEEDS, "seed must be one of the three frozen study seeds")
    _integer(batches, "batches", low=1, high=200)
    _integer(batch_size, "batch_size", low=1, high=16)
    windows, canonical_groups, episode_windows = _validated_inputs(train_windows, groups)
    group_ids = list(canonical_groups)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    initial_rng_sha = _rng_sha256(generator)
    draws = []
    def choice(size):
        return int(torch.randint(size, (1,), generator=generator, dtype=torch.int64, device="cpu").item())
    for batch_index in range(batches):
        for draw_in_batch in range(batch_size):
            group_id = group_ids[choice(len(group_ids))]
            members = canonical_groups[group_id]
            eid = members[choice(len(members))]
            window_indices = episode_windows[eid]
            window_index = window_indices[choice(len(window_indices))]
            draws.append({"window_index": window_index, "episode_index": eid, "group_id": group_id,
                          "batch_index": batch_index, "draw_in_batch": draw_in_batch})
    group_counts = Counter(d["group_id"] for d in draws)
    episode_counts = Counter(d["episode_index"] for d in draws)
    window_counts = Counter(d["window_index"] for d in draws)
    support = [{"group_id": group_id, "episode_index": eid, "window_count": len(episode_windows[eid]),
                "episode_probability": 1 / (len(group_ids) * len(members)),
                "per_window_probability": 1 / (len(group_ids) * len(members) * len(episode_windows[eid]))}
               for group_id, members in canonical_groups.items() for eid in members]
    result = {
        "schema": SCHEMA, "seed": seed, "batches": batches, "batch_size": batch_size,
        "draw_count": len(draws), "train_window_count": len(windows),
        "train_episode_indices": sorted(episode_windows), "groups": canonical_groups,
        "sampling_policy": "uniform_group_then_uniform_episode_then_uniform_complete_window",
        "sampling_with_replacement": True,
        "canonical_order": "lexicographic_group_id; ascending_episode_id; original_train_window_list_position",
        "train_windows_sha256": canonical_sha256(windows), "groups_sha256": canonical_sha256(canonical_groups),
        "input_sha256": canonical_sha256({"train_windows": windows, "groups": canonical_groups}),
        "hash_serialization": "sha256_compact_sorted_key_ascii_json_not_source_file_bytes",
        "rng": {"library": "torch", "torch_version": str(torch.__version__), "device": "cpu",
                "generator": "private_generator_manual_seed", "seed": seed,
                "calls_per_draw": 3, "primitive": "torch.randint(high,(1,),dtype=int64,device=cpu)",
                "initial_state_sha256": initial_rng_sha, "final_state_sha256": _rng_sha256(generator)},
        "draws": draws, "draws_sha256": canonical_sha256(draws),
        "counts": {"by_group": {gid: group_counts[gid] for gid in group_ids},
                   "by_episode": {str(eid): episode_counts[eid] for eid in sorted(episode_windows)},
                   "by_window_index": [window_counts[i] for i in range(len(windows))]},
        "sampling_probabilities": support, "shared_arms": list(ARMS),
        "group_metadata_policy_input": False, "normalization_fitted": False,
        "optimizer_steps": 0, "training_started": False,
        "evidence_scope": "prospective_shared_draw_schedule_only_not_execution_authorization",
    }
    canonical_sha256(result)  # Strict JSON guard; no tensors or NaN/Inf leak.
    return result


def verify_draw_manifest(manifest: dict, train_windows: list[dict], groups: dict[str, list[int]]) -> bool:
    """Rebuild and compare the entire schedule/summary, rejecting any drift.

    Runtime provenance is compared too: use the pinned torch environment for a
    full replay. Recorded draw bytes remain the authoritative shared schedule;
    silently regenerating under a different runtime is not supported.
    """
    _require(type(manifest) is dict, "draw manifest must be an object")
    expected = build_draw_manifest(train_windows, groups, manifest.get("seed"),
                                   manifest.get("batches"), manifest.get("batch_size"))
    _require(canonical_sha256(manifest) == canonical_sha256(expected),
             "draw manifest replay differs: inputs/draws/accounting/runtime/scope changed")
    return True
