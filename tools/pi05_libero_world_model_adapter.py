"""Native LIBERO feature/window contract. No LeRobot, model or training import.

This is deliberately separate from the guidewire state_32/Elite/Piper path.
Pack rows are observations BEFORE their same-row action. The final row's
action has no recorded successor and is excluded from windows/action stats.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "pi05_libero_native_feature_pack_v1"
SPLIT_SCHEMA = "pi05_libero_native_episode_split_v1"
DEMO_REPO = "lerobot/libero"
DEMO_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
STATE_LAYOUT = (
    "eef_position_x", "eef_position_y", "eef_position_z",
    "eef_axis_angle_x", "eef_axis_angle_y", "eef_axis_angle_z",
    "gripper_qpos_0", "gripper_qpos_1",
)
LAYOUT = {
    "state": list(STATE_LAYOUT), "state_dim": 8, "action_dim": 7,
    "image_keys": list(IMAGE_KEYS), "action_mode": "native_relative",
    "transition": "observation_t_action_t_to_observation_t_plus_1",
    "sampling": "consecutive_native_frames",
    "state_target": "current_relative_coordinate_residual_not_SO3_rotation",
}
ARRAY_NAMES = {
    "episode_index", "frame_index", "task_id", "state", "state_valid",
    "action", "visual_latent", "visual_valid",
}


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _keys(value: dict, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}: expected exactly {sorted(expected)}")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, not a coerced ID")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def validate_task_registry(rows: list[dict]) -> dict[int, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("explicit native instruction to Spatial task mapping required")
    result, source_ids, instructions = {}, set(), set()
    for row in rows:
        _keys(row, {"task_id", "source_task_index", "task_instruction"}, "task registry")
        task = _integer(row["task_id"], "Spatial task_id")
        source = _integer(row["source_task_index"], "source_task_index")
        instruction = _text(row["task_instruction"], "task_instruction")
        if task >= 10 or source >= 40 or task in result or source in source_ids or instruction in instructions:
            raise ValueError("ambiguous/out-of-range Spatial task mapping")
        result[task] = dict(row)
        source_ids.add(source)
        instructions.add(instruction)
    return result


def adapt_native_record(record: dict, task_registry: list[dict]) -> dict[str, Any]:
    """Adapt a selected LeRobot record, not a raw simulator dict.

    Caller explicitly selects these fields; reward/success/object state and
    arbitrary metadata are rejected. Images go through the separate extractor.
    A missing state requires an explicit all-false mask. No action imputation.
    """
    required = {"episode_index", "frame_index", "task_index", "task", "action"}
    optional = {"observation.state", "observation.state_valid"}
    if not required <= set(record) or set(record) - required - optional:
        raise ValueError("record must contain only native state/action/task and row identifiers")
    registry = validate_task_registry(task_registry)
    source_task = _integer(record["task_index"], "task_index")
    instruction = _text(record["task"], "native task instruction")
    matches = [row for row in registry.values() if row["source_task_index"] == source_task
               and row["task_instruction"] == instruction]
    if len(matches) != 1:
        raise ValueError("source task index/instruction does not match frozen Spatial mapping")
    state = np.asarray(record.get("observation.state", np.zeros(8)), dtype=np.float64)
    mask = np.asarray(record.get("observation.state_valid", np.ones(8, dtype=bool)))
    if state.shape != (8,) or mask.shape != (8,) or mask.dtype != np.bool_:
        raise ValueError("state must be 8D with a boolean 8D validity mask")
    if "observation.state" not in record and ("observation.state_valid" not in record or mask.any()):
        raise ValueError("missing state requires an explicit all-false validity mask")
    if not np.isfinite(state[mask]).all():
        raise ValueError("observed state must be finite")
    action = np.asarray(record["action"], dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("native action must be finite 7D within [-1,1]; no clipping")
    state = np.where(mask, state, 0).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("state is not representable as float32")
    return {
        "episode_index": _integer(record["episode_index"], "episode_index"),
        "frame_index": _integer(record["frame_index"], "frame_index"),
        "task_id": matches[0]["task_id"], "task_instruction": instruction,
        "state": state, "state_valid": mask.copy(), "action": action.astype(np.float32),
    }


@dataclass(frozen=True)
class NativeWindow:
    episode_index: int
    history: tuple[int, ...]
    actions: tuple[int, ...]
    targets: tuple[int, ...]


class LiberoFeaturePack:
    """Hash-bound, mmap-backed arrays with explicit source and episode registry."""

    def __init__(self, root: Path, *, allow_synthetic_fixture: bool = False):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        self.manifest_sha256 = sha256_file(self.root / "manifest.json")
        m = self.manifest
        _keys(m, {"schema", "benchmark", "source", "extractor", "layout", "fps",
                  "task_registry", "episodes", "arrays"}, "feature pack")
        if m["schema"] != SCHEMA or m["benchmark"] != "libero_spatial" or m["layout"] != LAYOUT:
            raise ValueError("wrong native LIBERO schema/layout; guidewire packs are not compatible")
        if isinstance(m["fps"], bool) or not isinstance(m["fps"], (float, int)) or not 0 < m["fps"] < 1000:
            raise ValueError("finite positive native fps required")
        _keys(m["source"], {"kind", "repo_id", "revision", "metadata_sha256"}, "source")
        source = m["source"]
        _hash(source["metadata_sha256"], "source metadata hash")
        self.synthetic = source["kind"] == "synthetic_fixture"
        if self.synthetic:
            if not allow_synthetic_fixture or source["repo_id"] != "synthetic/libero_contract_fixture":
                raise ValueError("synthetic fixture requires explicit smoke-only opt-in")
        elif source["kind"] != "public_demonstrations" or source["repo_id"] != DEMO_REPO or source["revision"] != DEMO_REVISION:
            raise ValueError("only pinned public demonstrations allowed; evaluation rollouts are excluded")
        _keys(m["extractor"], {"kind", "checkpoint_sha256", "preprocessing_sha256", "code_sha256"}, "extractor")
        expected_extractor = "synthetic_fixture" if self.synthetic else "frozen_pi05_native_multiview_v1"
        if m["extractor"]["kind"] != expected_extractor:
            raise ValueError("extractor does not match declared source scope")
        for name in ("checkpoint_sha256", "preprocessing_sha256", "code_sha256"):
            _hash(m["extractor"][name], name)
        self.tasks = validate_task_registry(m["task_registry"])
        _keys(m["arrays"], ARRAY_NAMES, "array allowlist")
        self.arrays = {}
        for name, entry in m["arrays"].items():
            _keys(entry, {"path", "sha256"}, f"array {name}")
            path = (self.root / _text(entry["path"], "array path")).resolve()
            if not path.is_relative_to(self.root) or path.suffix != ".npy":
                raise ValueError("array path must remain inside feature-pack root and be .npy")
            if sha256_file(path) != _hash(entry["sha256"], "array sha256"):
                raise ValueError(f"array hash mismatch: {name}")
            self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        self._validate_arrays()
        self._validate_episodes()
        self._validated_splits: dict[str, str] = {}

    def close(self) -> None:
        for value in self.arrays.values():
            if isinstance(value, np.memmap):
                value._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _validate_arrays(self) -> None:
        a = self.arrays
        n = len(a["episode_index"])
        if n == 0:
            raise ValueError("empty feature pack")
        for key in ("episode_index", "frame_index", "task_id"):
            if a[key].shape != (n,) or a[key].dtype.kind not in "iu" or np.any(a[key] < 0):
                raise ValueError(f"{key} must be a nonnegative integer vector")
        for key, shape in (("state", (n, 8)), ("action", (n, 7))):
            if a[key].shape != shape or a[key].dtype != np.float32:
                raise ValueError(f"{key} must be float32 {shape}")
        latent = a["visual_latent"]
        if latent.ndim != 3 or latent.shape[:2] != (n, 2) or latent.shape[2] <= 0 or latent.dtype != np.float32:
            raise ValueError("visual_latent must be float32 [N,2,D]")
        self.visual_dim = latent.shape[2]
        for key, shape in (("state_valid", (n, 8)), ("visual_valid", (n, 2))):
            if a[key].shape != shape or a[key].dtype != np.bool_:
                raise ValueError(f"{key} must be boolean {shape}")
        for start in range(0, n, 1024):
            batch = slice(start, start + 1024)
            if not np.isfinite(a["state"][batch][a["state_valid"][batch]]).all():
                raise ValueError("valid state values must be finite")
            if not np.isfinite(latent[batch][a["visual_valid"][batch]]).all():
                raise ValueError("valid visual latents must be finite")
            if not np.isfinite(a["action"][batch]).all() or np.any(np.abs(a["action"][batch]) > 1):
                raise ValueError("native actions must be finite and within [-1,1]")

    def _validate_episodes(self) -> None:
        self.episodes, self.indices = {}, {}
        rows = self.manifest["episodes"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("complete episode registry required")
        for row in rows:
            _keys(row, {"episode_index", "task_id", "source_trajectory_id", "leakage_group_id",
                        "record_count", "complete_episode"}, "episode")
            episode = _integer(row["episode_index"], "episode_index")
            task = _integer(row["task_id"], "task_id")
            count = _integer(row["record_count"], "record_count")
            if episode in self.episodes or task not in self.tasks or row["complete_episode"] is not True:
                raise ValueError("duplicate/unknown/incomplete episode")
            for key in ("source_trajectory_id", "leakage_group_id"):
                _text(row[key], key)
            indices = np.flatnonzero(self.arrays["episode_index"] == episode)
            indices = indices[np.argsort(self.arrays["frame_index"][indices], kind="stable")]
            frames = self.arrays["frame_index"][indices]
            if len(indices) != count or count < 2 or not np.array_equal(frames, np.arange(count)):
                raise ValueError("episode requires complete consecutive frames 0..record_count-1; no gaps/resets")
            if not np.all(self.arrays["task_id"][indices] == task):
                raise ValueError("task changes inside episode or disagrees with registry")
            self.episodes[episode], self.indices[episode] = dict(row), indices
        if set(np.unique(self.arrays["episode_index"])) != set(self.episodes):
            raise ValueError("unregistered episode rows")
        if {row["task_id"] for row in self.episodes.values()} != set(self.tasks):
            raise ValueError("task registry must match present episodes")

    def load_split(self, path: Path) -> dict:
        split = read_json(path)
        _keys(split, {"schema", "feature_pack_manifest_sha256", "train_episode_indices",
                      "validation_episode_indices"}, "split")
        if split["schema"] != SPLIT_SCHEMA or split["feature_pack_manifest_sha256"] != self.manifest_sha256:
            raise ValueError("split schema/hash does not match this pack")
        partitions = {}
        for name in ("train", "validation"):
            values = split[f"{name}_episode_indices"]
            if not isinstance(values, list) or not values:
                raise ValueError("split partitions must be nonempty lists")
            ids = [_integer(value, "split episode") for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate split episode")
            partitions[name] = set(ids)
        train, val = partitions["train"], partitions["validation"]
        if train & val or train | val != set(self.episodes):
            raise ValueError("split must be disjoint and cover known episodes exactly")
        for key in ("source_trajectory_id", "leakage_group_id"):
            groups = [{self.episodes[e][key] for e in partition} for partition in (train, val)]
            if groups[0] & groups[1]:
                raise ValueError(f"{key} leaks across train/validation")
        for partition in (train, val):
            if {self.episodes[e]["task_id"] for e in partition} != set(self.tasks):
                raise ValueError("both partitions must cover the frozen task registry")
        split["split_sha256"] = sha256_file(path)
        self._validated_splits[split["split_sha256"]] = json.dumps(split, sort_keys=True)
        return split

    def windows(self, episodes: set[int], context_len: int = 4, horizon: int = 3) -> list[NativeWindow]:
        if type(context_len) is not int or type(horizon) is not int or min(context_len, horizon) < 1:
            raise ValueError("context/horizon must be positive integers")
        if not episodes or any(_integer(e, "window episode") not in self.episodes for e in episodes):
            raise ValueError("unknown/empty window episode selection")
        windows = []
        for episode in sorted(episodes):
            rows = self.indices[episode]
            for t in range(context_len - 1, len(rows) - horizon):
                windows.append(NativeWindow(episode, tuple(map(int, rows[t-context_len+1:t+1])),
                                            tuple(map(int, rows[t:t+horizon])), tuple(map(int, rows[t+1:t+horizon+1]))))
        return windows


def fit_train_normalization(pack: LiberoFeaturePack, split: dict) -> dict:
    """World-model-only mean/std, NOT a replacement for PI05 checkpoint stats."""
    if (split.get("feature_pack_manifest_sha256") != pack.manifest_sha256
            or pack._validated_splits.get(split.get("split_sha256")) != json.dumps(split, sort_keys=True)):
        raise ValueError("normalization requires the validated hash-bound split")
    train = split["train_episode_indices"]
    if not train or len(train) != len(set(train)) or any(_integer(e, "train episode") not in pack.episodes for e in train):
        raise ValueError("unknown/duplicate/empty train episodes")
    state_rows = np.concatenate([pack.indices[e] for e in sorted(train)])
    action_rows = np.concatenate([pack.indices[e][:-1] for e in sorted(train)])
    state = pack.arrays["state"][state_rows].astype(np.float64)
    valid = pack.arrays["state_valid"][state_rows]
    count = valid.sum(axis=0)
    mean = np.where(valid, state, 0).sum(axis=0) / np.maximum(count, 1)
    centered = np.where(valid, state - mean, 0)
    std = np.sqrt((centered * centered).sum(axis=0) / np.maximum(count, 1))
    std = np.where((count > 0) & (std >= 1e-6), std, 1)
    action = pack.arrays["action"][action_rows].astype(np.float64)
    action_std = action.std(axis=0)
    return {
        "schema": "pi05_libero_native_normalization_v1",
        "feature_pack_manifest_sha256": pack.manifest_sha256,
        "split_sha256": split["split_sha256"],
        "train_episode_indices": sorted(train),
        "state_mean": mean.tolist(), "state_std": std.tolist(), "state_count": count.tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": np.where(action_std >= 1e-6, action_std, 1).tolist(),
        "state_record_count": len(state_rows), "action_record_count": len(action_rows),
        "final_row_actions_excluded": True,
    }


class LiberoWindowDataset:
    def __init__(self, pack: LiberoFeaturePack, split: dict, *, partition: str,
                 normalization: dict, context_len: int = 4, horizon: int = 3):
        if partition not in {"train", "validation"}:
            raise ValueError("unknown partition")
        if normalization != fit_train_normalization(pack, split):
            raise ValueError("normalization does not exactly match this pack's training split")
        self.pack, self.normalization = pack, deepcopy(normalization)
        self.windows = pack.windows(set(split[f"{partition}_episode_indices"]), context_len, horizon)
        if not self.windows:
            raise ValueError("no complete temporal windows in requested partition")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        w, a, n = self.windows[index], self.pack.arrays, self.normalization
        h, actions, targets = list(w.history), list(w.actions), list(w.targets)
        current = h[-1]
        supported = np.asarray(n["state_count"]) > 0
        history_mask = a["state_valid"][h] & supported
        target_mask = a["state_valid"][targets] & a["state_valid"][current] & supported
        mean, std = np.asarray(n["state_mean"]), np.asarray(n["state_std"])
        history_state = (np.where(history_mask, a["state"][h], mean) - mean) / std
        future = np.where(target_mask, a["state"][targets], 0).astype(np.float64)
        anchor = np.where(target_mask, a["state"][current], 0).astype(np.float64)
        state_delta = (future - anchor) / std
        history_visual_mask = a["visual_valid"][h]
        future_visual_mask = a["visual_valid"][targets]
        task = self.pack.tasks[self.pack.episodes[w.episode_index]["task_id"]]
        inputs = {
            "history_visual_latent": np.where(history_visual_mask[..., None], a["visual_latent"][h], 0),
            "history_visual_valid": history_visual_mask.copy(),
            "history_state": history_state.astype(np.float32), "history_state_valid": history_mask.copy(),
            "task_instruction": task["task_instruction"],
            "candidate_actions": ((a["action"][actions] - np.asarray(n["action_mean"])) /
                                  np.asarray(n["action_std"])).astype(np.float32)[None],
        }
        output = {
            "inputs": inputs,
            "targets": {"future_visual_latent": np.where(future_visual_mask[..., None], a["visual_latent"][targets], 0),
                        "future_visual_valid": future_visual_mask.copy(),
                        "state_delta": state_delta.astype(np.float32), "state_target_valid": target_mask.copy()},
            "metadata": {"episode_index": w.episode_index, "task_id": task["task_id"],
                         "source_task_index": task["source_task_index"],
                         "history_indices": h, "action_indices": actions, "target_indices": targets},
        }
        for section in (output["inputs"], output["targets"]):
            for value in section.values():
                if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                    raise ValueError("nonfinite model-facing window after masking/normalization")
        return output
