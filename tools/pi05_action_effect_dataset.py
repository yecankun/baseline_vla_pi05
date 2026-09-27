from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from pi05_action_effect_world_model import encode_active_action


FEATURE_PACK_SCHEMA = "project2026_action_effect_feature_pack_v1"
SPLIT_SCHEMA = "project2026_action_effect_episode_split_v1"
STATE_32_LAYOUT_SCHEMA = "project2026_pi05_observable_event_state_v1"
STATE_32_LAYOUT = {
    "schema": STATE_32_LAYOUT_SCHEMA,
    "dimension": 32,
    "elite_tcp_pose_6d": [0, 6],
    "base_piper_tactile_compat": [6, 14],
    "task_one_hot": [14, 16],
    "piper_busy": [16, 17],
    "piper_cooldown": [17, 18],
    "piper_request_accepted": [18, 19],
    "piper_executed_feed": [19, 20],
    "piper_event_age_steps_saturating": [20, 21],
    "piper_event_status_one_hot_idle_delayed_executing_completed": [21, 25],
    "piper_event_state_valid": [25, 26],
    "piper_event_age_steps_valid": [26, 27],
    "reserved": [27, 32],
}
REQUIRED_ARRAYS = {
    "visual_latent",
    "state_32",
    "elite_tcp_delta_6d",
    "piper_intent_id",
    "task_id",
    "episode_index",
    "frame_index",
    "domain_id",
}
OPTIONAL_TARGET_ARRAYS = {
    "guidance_effect_id",
    "branch_outcome_id",
    "invalid_feed_flag",
    "visual_valid",
    "state_valid_32",
    "degraded_visual_latent",
    "degradation_pair_valid",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TemporalWindow:
    episode_index: int
    domain_id: int
    task_id: int
    history_indices: tuple[int, ...]
    action_indices: tuple[int, ...]
    target_indices: tuple[int, ...]


def build_temporal_windows(
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    task_id: np.ndarray,
    domain_id: np.ndarray,
    *,
    context_len: int,
    horizon: int,
    allowed_episodes: set[int] | None = None,
) -> list[TemporalWindow]:
    if context_len <= 0 or horizon <= 0:
        raise ValueError("context_len and horizon must be positive")
    size = int(episode_index.shape[0])
    for name, values in (
        ("frame_index", frame_index),
        ("task_id", task_id),
        ("domain_id", domain_id),
    ):
        if values.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")

    windows: list[TemporalWindow] = []
    for episode in sorted(int(value) for value in np.unique(episode_index)):
        if allowed_episodes is not None and episode not in allowed_episodes:
            continue
        indices = np.flatnonzero(episode_index == episode)
        order = np.argsort(frame_index[indices], kind="stable")
        indices = indices[order]
        frames = frame_index[indices]
        if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
            raise ValueError(f"episode {episode} frame_index must be unique and increasing")
        episode_tasks = np.unique(task_id[indices])
        episode_domains = np.unique(domain_id[indices])
        if len(episode_tasks) != 1 or len(episode_domains) != 1:
            raise ValueError(f"episode {episode} changes task or domain inside an episode")
        required = context_len + horizon
        if len(indices) < required:
            continue
        for history_end in range(context_len - 1, len(indices) - horizon):
            history = indices[history_end - context_len + 1 : history_end + 1]
            actions = indices[history_end : history_end + horizon]
            targets = indices[history_end + 1 : history_end + horizon + 1]
            windows.append(
                TemporalWindow(
                    episode_index=episode,
                    domain_id=int(episode_domains[0]),
                    task_id=int(episode_tasks[0]),
                    history_indices=tuple(int(value) for value in history),
                    action_indices=tuple(int(value) for value in actions),
                    target_indices=tuple(int(value) for value in targets),
                )
            )
    return windows


def make_episode_stratified_real_folds(
    episode_rows: list[dict[str, Any]],
    *,
    folds: int = 5,
) -> list[dict[str, Any]]:
    if folds <= 1:
        raise ValueError("folds must be greater than one")
    by_task: dict[str, list[int]] = {"left": [], "right": []}
    seen: set[int] = set()
    for row in episode_rows:
        episode = int(row["episode_index"])
        task = str(row["task"])
        domain = str(row.get("domain", "real_current"))
        if episode in seen:
            raise ValueError(f"duplicate episode in fold request: {episode}")
        if task not in by_task:
            raise ValueError(f"unsupported task for fold request: {task!r}")
        if domain != "real_current":
            raise ValueError("real folds may contain only current real data")
        seen.add(episode)
        by_task[task].append(episode)
    for task in by_task:
        by_task[task].sort()
        if len(by_task[task]) < folds:
            raise ValueError(f"task {task} has {len(by_task[task])} episodes, fewer than {folds}")

    result: list[dict[str, Any]] = []
    for fold in range(folds):
        validation = sorted(
            by_task["left"][fold::folds] + by_task["right"][fold::folds]
        )
        train = sorted(seen - set(validation))
        result.append(
            {
                "fold": fold,
                "train_episode_indices": train,
                "validation_episode_indices": validation,
                "validation_left": [value for value in validation if value in by_task["left"]],
                "validation_right": [value for value in validation if value in by_task["right"]],
            }
        )
    validation_union = [
        episode
        for fold in result
        for episode in fold["validation_episode_indices"]
    ]
    if sorted(validation_union) != sorted(seen):
        raise AssertionError("each real episode must appear in validation exactly once")
    return result


def source_episode_balanced_weights(
    windows: list[TemporalWindow],
    *,
    real_draw_fraction: float,
    episode_source: dict[int, str] | None = None,
) -> torch.Tensor:
    if not 0.0 <= real_draw_fraction <= 1.0:
        raise ValueError("real_draw_fraction must be in [0, 1]")
    if not windows:
        raise ValueError("cannot weight an empty window set")
    domains = {window.domain_id for window in windows}
    if not domains <= {0, 1}:
        raise ValueError("domain_id must use sim=0 and real_current=1")
    requested_mass = {0: 1.0 - real_draw_fraction, 1: real_draw_fraction}
    present_domains = sorted(domains)
    if len(present_domains) == 1:
        requested_mass[present_domains[0]] = 1.0
    elif any(requested_mass[domain] <= 0.0 for domain in present_domains):
        raise ValueError("both domains are present but one requested draw mass is zero")

    source_by_episode = episode_source or {
        window.episode_index: f"episode:{window.episode_index}" for window in windows
    }
    missing_sources = sorted(
        {window.episode_index for window in windows} - set(source_by_episode)
    )
    if missing_sources:
        raise ValueError(f"episode_source missing episodes: {missing_sources[:8]}")
    episode_counts: dict[tuple[int, int], int] = {}
    episodes_by_domain_source: dict[tuple[int, str], set[int]] = {}
    sources_by_domain: dict[int, set[str]] = {0: set(), 1: set()}
    for window in windows:
        source = str(source_by_episode[window.episode_index]).strip()
        if not source:
            raise ValueError(f"empty source trajectory for episode {window.episode_index}")
        key = (window.domain_id, window.episode_index)
        episode_counts[key] = episode_counts.get(key, 0) + 1
        sources_by_domain[window.domain_id].add(source)
        episodes_by_domain_source.setdefault((window.domain_id, source), set()).add(
            window.episode_index
        )
    weights = []
    for window in windows:
        source = str(source_by_episode[window.episode_index]).strip()
        source_count = len(sources_by_domain[window.domain_id])
        source_episode_count = len(episodes_by_domain_source[(window.domain_id, source)])
        weight = requested_mass[window.domain_id] / (
            source_count
            * source_episode_count
            * episode_counts[(window.domain_id, window.episode_index)]
        )
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.double)
    tensor = tensor / tensor.sum()
    return tensor


class ActionEffectFeaturePack:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest = _read_json(self.root / "manifest.json")
        if self.manifest.get("schema") != FEATURE_PACK_SCHEMA:
            raise ValueError(f"unexpected feature-pack schema: {self.manifest.get('schema')!r}")
        arrays_name = str(self.manifest.get("outputs", {}).get("arrays", "arrays.npz"))
        arrays_path = (self.root / arrays_name).resolve()
        try:
            arrays_path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("feature-pack arrays path escapes pack root") from exc
        expected_arrays_sha256 = str(
            self.manifest.get("outputs", {}).get("arrays_sha256", "")
        ).lower()
        if len(expected_arrays_sha256) != 64:
            raise ValueError("feature pack must declare outputs.arrays_sha256")
        actual_arrays_sha256 = _sha256_file(arrays_path)
        if actual_arrays_sha256 != expected_arrays_sha256:
            raise ValueError("feature-pack arrays SHA256 mismatch")
        self.arrays_sha256 = actual_arrays_sha256
        self.manifest_sha256 = _sha256_file(self.root / "manifest.json")
        with np.load(arrays_path, allow_pickle=False) as loaded:
            keys = set(loaded.files)
            missing = sorted(REQUIRED_ARRAYS - keys)
            if missing:
                raise ValueError(f"feature pack missing arrays: {missing}")
            self.arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        self._validate()

    def _validate(self) -> None:
        arrays = self.arrays
        allowed_arrays = REQUIRED_ARRAYS | OPTIONAL_TARGET_ARRAYS
        extra_arrays = sorted(set(arrays) - allowed_arrays)
        if extra_arrays:
            raise ValueError(
                "feature pack contains undeclared arrays; keep diagnostic truth in a separate sidecar: "
                f"{extra_arrays}"
            )
        count = int(arrays["state_32"].shape[0])
        expected = int(self.manifest.get("counts", {}).get("records", -1))
        if expected != count:
            raise ValueError(f"manifest records={expected} but arrays contain {count}")
        expected_shapes = {
            "state_32": (count, 32),
            "elite_tcp_delta_6d": (count, 6),
            "piper_intent_id": (count,),
            "task_id": (count,),
            "episode_index": (count,),
            "frame_index": (count,),
            "domain_id": (count,),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} shape {arrays[name].shape} != {shape}")
        if arrays["visual_latent"].ndim != 3 or arrays["visual_latent"].shape[0] != count:
            raise ValueError("visual_latent must be [N, V, D]")
        if not np.isfinite(arrays["visual_latent"]).all() or not np.isfinite(arrays["state_32"]).all():
            raise ValueError("visual/state arrays contain non-finite values")
        if not np.isfinite(arrays["elite_tcp_delta_6d"]).all():
            raise ValueError("Elite action array contains non-finite values")
        if not set(np.unique(arrays["piper_intent_id"]).tolist()) <= {0, 1, 2}:
            raise ValueError("piper_intent_id must use retract=0, hold=1, feed=2")
        if not set(np.unique(arrays["task_id"]).tolist()) <= {0, 1}:
            raise ValueError("task_id must use left=0 and right=1")
        if not set(np.unique(arrays["domain_id"]).tolist()) <= {0, 1}:
            raise ValueError("domain_id must use sim=0 and real_current=1")
        if "visual_valid" in arrays and arrays["visual_valid"].shape != arrays["visual_latent"].shape[:2]:
            raise ValueError("visual_valid must be [N, V]")
        if "state_valid_32" in arrays and arrays["state_valid_32"].shape != (count, 32):
            raise ValueError("state_valid_32 must be [N, 32]")
        degradation_keys = {
            "degraded_visual_latent",
            "degradation_pair_valid",
        } & set(arrays)
        if degradation_keys and degradation_keys != {
            "degraded_visual_latent",
            "degradation_pair_valid",
        }:
            raise ValueError("degraded latent and pair-valid arrays must be provided together")
        if "degraded_visual_latent" in arrays:
            if arrays["degraded_visual_latent"].shape != arrays["visual_latent"].shape:
                raise ValueError("degraded_visual_latent must match visual_latent shape")
            if arrays["degradation_pair_valid"].shape != (count,):
                raise ValueError("degradation_pair_valid must be [N]")
            if not np.isfinite(arrays["degraded_visual_latent"]).all():
                raise ValueError("degraded visual latent contains non-finite values")
        for name, allowed in (
            ("guidance_effect_id", {-1, 0, 1, 2}),
            ("branch_outcome_id", {-1, 0, 1}),
            ("invalid_feed_flag", {-1, 0, 1}),
        ):
            if name in arrays:
                if arrays[name].shape != (count,):
                    raise ValueError(f"{name} must be [N]")
                if not set(np.unique(arrays[name]).tolist()) <= allowed:
                    raise ValueError(f"{name} contains unsupported labels")

        contract = self.manifest.get("contract", {})
        if contract.get("senior_historical_real_data_allowed") is not False:
            raise ValueError("feature pack must explicitly exclude senior historical real data")
        if contract.get("exact_truth_in_policy_input") is not False:
            raise ValueError("feature pack must declare exact truth absent from policy input")
        if contract.get("action_interface") != "elite_tcp_delta_6d + piper_intent_id":
            raise ValueError("feature pack changed the shared action interface")
        visual_source = contract.get("visual_latent_source", {})
        if visual_source.get("extractor") != "pi05_multiview_mean_patch_v1":
            raise ValueError("feature pack must declare the fixed PI05 visual latent extractor")
        if visual_source.get("backbone_frozen") is not True:
            raise ValueError("feature-pack visual backbone must remain frozen")
        if not str(visual_source.get("pretrained_revision", "")).strip():
            raise ValueError("feature pack must pin the PI05 pretrained revision")
        if not str(visual_source.get("pretrained_name_or_path", "")).strip():
            raise ValueError("feature pack must identify the PI05 pretrained source")
        if contract.get("state_32_layout") != STATE_32_LAYOUT:
            raise ValueError("feature pack does not declare the fixed observable state_32 layout")

        episode_values = {int(value) for value in np.unique(arrays["episode_index"])}
        expected_episodes = int(self.manifest.get("counts", {}).get("episodes", -1))
        if expected_episodes != len(episode_values):
            raise ValueError(
                f"manifest episodes={expected_episodes} but arrays contain {len(episode_values)}"
            )
        provenance_rows = self.manifest.get("episode_provenance")
        if not isinstance(provenance_rows, list):
            raise ValueError("feature pack must declare episode_provenance")
        provenance_by_episode: dict[int, dict[str, Any]] = {}
        for row in provenance_rows:
            if not isinstance(row, dict):
                raise ValueError("episode_provenance entries must be objects")
            episode = int(row.get("episode_index", -1))
            if episode in provenance_by_episode:
                raise ValueError(f"duplicate episode_provenance entry: {episode}")
            task = str(row.get("task"))
            domain = str(row.get("domain"))
            if task not in {"left", "right"} or domain not in {"sim", "real_current"}:
                raise ValueError(f"invalid episode provenance task/domain for {episode}")
            for key in ("episode_instance_id", "source_trajectory_id", "scenario_family_id"):
                if not str(row.get(key, "")).strip():
                    raise ValueError(f"episode {episode} missing provenance field {key}")
            provenance_by_episode[episode] = row
        if set(provenance_by_episode) != episode_values:
            raise ValueError("episode_provenance must cover every array episode exactly once")
        for episode, row in provenance_by_episode.items():
            indices = np.flatnonzero(arrays["episode_index"] == episode)
            expected_task = 0 if row["task"] == "left" else 1
            expected_domain = 0 if row["domain"] == "sim" else 1
            if set(np.unique(arrays["task_id"][indices]).tolist()) != {expected_task}:
                raise ValueError(f"episode {episode} task does not match provenance")
            if set(np.unique(arrays["domain_id"][indices]).tolist()) != {expected_domain}:
                raise ValueError(f"episode {episode} domain does not match provenance")
        self.episode_provenance = provenance_by_episode

    @property
    def visual_dim(self) -> int:
        return int(self.arrays["visual_latent"].shape[2])

    @property
    def views(self) -> int:
        return int(self.arrays["visual_latent"].shape[1])

    def validate_episode_split(self, split: dict[str, Any]) -> None:
        if str(split.get("feature_pack_manifest_sha256", "")).lower() != self.manifest_sha256:
            raise ValueError("episode split is not bound to this feature-pack manifest")
        train = {int(value) for value in split["train_episode_indices"]}
        validation = {int(value) for value in split["validation_episode_indices"]}
        all_episodes = set(self.episode_provenance)
        if train | validation != all_episodes:
            raise ValueError("split manifest must cover every feature-pack episode exactly once")
        partition = {episode: "train" for episode in train}
        partition.update({episode: "validation" for episode in validation})
        family_partitions: dict[str, set[str]] = {}
        source_partitions: dict[str, set[str]] = {}
        for episode, provenance in self.episode_provenance.items():
            family = str(provenance["scenario_family_id"])
            source = str(provenance["source_trajectory_id"])
            family_partitions.setdefault(family, set()).add(partition[episode])
            source_partitions.setdefault(source, set()).add(partition[episode])
        leaking_families = sorted(key for key, values in family_partitions.items() if len(values) > 1)
        leaking_sources = sorted(key for key, values in source_partitions.items() if len(values) > 1)
        if leaking_families or leaking_sources:
            raise ValueError(
                "source/family leakage across train and validation: "
                f"families={leaking_families[:5]} sources={leaking_sources[:5]}"
            )

    def episode_source_map(self) -> dict[int, str]:
        return {
            episode: str(row["source_trajectory_id"])
            for episode, row in self.episode_provenance.items()
        }


def compute_train_normalization(
    pack: ActionEffectFeaturePack,
    train_episodes: set[int],
    *,
    eps: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute state/action statistics from training episodes only."""

    if not train_episodes:
        raise ValueError("train normalization requires at least one episode")
    selected = np.isin(pack.arrays["episode_index"], sorted(train_episodes))
    if not bool(selected.any()):
        raise ValueError("train normalization selected no records")
    states = np.asarray(pack.arrays["state_32"], dtype=np.float64)
    state_valid = np.asarray(
        pack.arrays.get("state_valid_32", np.ones_like(states, dtype=np.bool_)),
        dtype=np.bool_,
    )
    state_valid = state_valid & selected[:, None]
    state_count = state_valid.sum(axis=0).astype(np.int64)
    state_mean = np.divide(
        np.where(state_valid, states, 0.0).sum(axis=0),
        np.maximum(state_count, 1),
    )
    centered = np.where(state_valid, states - state_mean, 0.0)
    state_std = np.sqrt(
        np.divide((centered * centered).sum(axis=0), np.maximum(state_count, 1))
    )
    state_mean = np.where(state_count > 0, state_mean, 0.0)
    state_std = np.where((state_count > 0) & (state_std >= eps), state_std, 1.0)

    elite = np.asarray(pack.arrays["elite_tcp_delta_6d"][selected], dtype=np.float64)
    elite_mean = elite.mean(axis=0)
    elite_std = elite.std(axis=0)
    elite_std = np.where(elite_std >= eps, elite_std, 1.0)
    return {
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "state_count": state_count,
        "elite_mean": elite_mean.astype(np.float32),
        "elite_std": elite_std.astype(np.float32),
        "record_count": np.asarray([int(selected.sum())], dtype=np.int64),
    }


def normalize_state_32_tensor(
    state_32: torch.Tensor,
    state_valid_32: torch.Tensor,
    normalization: dict[str, Any],
) -> torch.Tensor:
    if state_32.shape != state_valid_32.shape or state_32.shape[-1] != 32:
        raise ValueError("state and validity must share shape [..., 32]")
    mean = torch.as_tensor(
        normalization["state_mean"], dtype=state_32.dtype, device=state_32.device
    )
    std = torch.as_tensor(
        normalization["state_std"], dtype=state_32.dtype, device=state_32.device
    )
    return ((state_32 - mean) / std).masked_fill(~state_valid_32.bool(), 0.0)


def normalize_active_action_candidates(
    candidate_actions: torch.Tensor,
    normalization: dict[str, Any],
) -> torch.Tensor:
    if candidate_actions.shape[-1] != 9:
        raise ValueError("active candidate actions must end in nine dimensions")
    mean = torch.as_tensor(
        normalization["elite_mean"],
        dtype=candidate_actions.dtype,
        device=candidate_actions.device,
    )
    std = torch.as_tensor(
        normalization["elite_std"],
        dtype=candidate_actions.dtype,
        device=candidate_actions.device,
    )
    normalized = candidate_actions.clone()
    normalized[..., :6] = (normalized[..., :6] - mean) / std
    return normalized


class ActionEffectWindowDataset(Dataset):
    def __init__(
        self,
        pack: ActionEffectFeaturePack,
        windows: list[TemporalWindow],
        normalization: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.pack = pack
        self.windows = windows
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays = self.pack.arrays
        window = self.windows[index]
        history = np.asarray(window.history_indices, dtype=np.int64)
        actions = np.asarray(window.action_indices, dtype=np.int64)
        targets = np.asarray(window.target_indices, dtype=np.int64)
        elite_values = np.asarray(arrays["elite_tcp_delta_6d"][actions], dtype=np.float32)
        if self.normalization is not None:
            elite_values = (
                elite_values - self.normalization["elite_mean"]
            ) / self.normalization["elite_std"]
        elite = torch.from_numpy(elite_values).float()
        piper = torch.from_numpy(arrays["piper_intent_id"][actions]).long()
        candidate = encode_active_action(elite, piper).unsqueeze(0)
        all_state_valid = np.asarray(
            arrays.get("state_valid_32", np.ones_like(arrays["state_32"], dtype=np.bool_)),
            dtype=np.bool_,
        )
        history_state_valid = all_state_valid[history]
        target_state_valid = all_state_valid[targets] & all_state_valid[history[-1]][None, :]
        state_values = np.asarray(arrays["state_32"], dtype=np.float32)
        if self.normalization is not None:
            state_values = (
                state_values - self.normalization["state_mean"]
            ) / self.normalization["state_std"]
        state_history_values = np.where(history_state_valid, state_values[history], 0.0)
        current_state_values = np.where(
            all_state_valid[history[-1]], state_values[history[-1]], 0.0
        )
        future_state_values = np.where(all_state_valid[targets], state_values[targets], 0.0)
        state_history = torch.from_numpy(state_history_values).float()
        current_state = torch.from_numpy(current_state_values).float()
        future_state = torch.from_numpy(future_state_values).float()
        future_visual = torch.from_numpy(arrays["visual_latent"][targets]).float()
        if "visual_valid" in arrays:
            future_visual_valid = torch.from_numpy(arrays["visual_valid"][targets]).bool()
            view_weights = future_visual_valid.to(future_visual.dtype).unsqueeze(-1)
            next_visual = (future_visual * view_weights).sum(dim=1) / view_weights.sum(
                dim=1
            ).clamp_min(1.0)
        else:
            future_visual_valid = torch.ones(
                len(targets), self.pack.views, dtype=torch.bool
            )
            next_visual = future_visual.mean(dim=1)
        item: dict[str, torch.Tensor] = {
            "history_visual_latent": torch.from_numpy(arrays["visual_latent"][history]).float(),
            "history_state": state_history,
            "history_state_valid": torch.from_numpy(history_state_valid).bool(),
            "task_id": torch.tensor(window.task_id, dtype=torch.long),
            "candidate_actions": candidate,
            "next_visual_latent": next_visual,
            "state_delta": future_state - current_state.unsqueeze(0),
            "episode_index": torch.tensor(window.episode_index, dtype=torch.long),
            "domain_id": torch.tensor(window.domain_id, dtype=torch.long),
        }
        if "degraded_visual_latent" in arrays:
            item["history_degraded_visual_latent"] = torch.from_numpy(
                arrays["degraded_visual_latent"][history]
            ).float()
            item["degradation_pair_valid"] = torch.tensor(
                bool(np.asarray(arrays["degradation_pair_valid"][history]).all()),
                dtype=torch.bool,
            )
        else:
            item["history_degraded_visual_latent"] = item[
                "history_visual_latent"
            ].clone()
            item["degradation_pair_valid"] = torch.tensor(False, dtype=torch.bool)
        if "visual_valid" in arrays:
            item["history_visual_valid"] = torch.from_numpy(
                arrays["visual_valid"][history]
            ).bool()
            item["visual_target_valid"] = future_visual_valid.any(dim=1)
        else:
            item["history_visual_valid"] = torch.ones(
                len(history), self.pack.views, dtype=torch.bool
            )
            item["visual_target_valid"] = torch.ones(len(targets), dtype=torch.bool)
        if "state_valid_32" in arrays:
            item["state_target_valid"] = torch.from_numpy(target_state_valid).bool()
        else:
            item["state_target_valid"] = torch.ones(len(targets), 32, dtype=torch.bool)
        for name in ("guidance_effect_id", "branch_outcome_id", "invalid_feed_flag"):
            if name in arrays:
                item[name] = torch.from_numpy(arrays[name][targets]).long()
            else:
                item[name] = torch.full((len(targets),), -1, dtype=torch.long)
        return item


def load_episode_split(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"unexpected split schema: {value.get('schema')!r}")
    train = {int(item) for item in value.get("train_episode_indices", [])}
    validation = {int(item) for item in value.get("validation_episode_indices", [])}
    if not train or not validation or train & validation:
        raise ValueError("episode split must have non-empty, disjoint train/validation sets")
    return value


def iter_window_episode_ids(windows: list[TemporalWindow]) -> Iterator[int]:
    for window in windows:
        yield window.episode_index
