import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import models, transforms

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, desc=None, leave=True):
            self.iterable = iterable
            self.total = total
            self.desc = desc

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, total=None, desc=None, leave=True):
        return _TqdmFallback(iterable=iterable, total=total, desc=desc, leave=leave)


FULL_SIM_STATE_KEYS = [
    "tip_pos",
    "heading",
    "target_pos",
    "contact_normal",
    "distance_to_wall",
    "contact_strength",
    "contact_flag",
    "path_progress",
    "piper_step",
    "piper_insertion_length",
    "elirobot_pose",
    "lateral_offset",
    "path_tangent",
    "local_radius",
]

REAL_DIRECT_STATE_KEYS = [
    "piper_step",
    "piper_insertion_length",
    "elirobot_pose",
]

SENIOR_PIPER_REAL_LIKE_STATE_KEYS = [
    "elite_tcp_pose_6d",
    "piper_step",
]

SENIOR_PIPER_REAL_LIKE_WITH_PHASE_STATE_KEYS = SENIOR_PIPER_REAL_LIKE_STATE_KEYS

REAL_DIRECT_PLUS_ESTIMATED_TIP_CONTACT_STATE_KEYS = [
    "elite_tcp_pose_6d",
    "piper_step",
    "piper_insertion_length",
    "estimated_tip_pos_3d",
    "estimated_tip_heading_3d",
    "tip_estimator_confidence",
    "tip_estimator_visible",
    "estimated_contact_flag",
    "contact_estimator_confidence",
    "estimated_image_distance_px",
]

REAL_DIRECT_PLUS_ESTIMATED_TIP_REGISTERED_GEOMETRY_STATE_KEYS = [
    *REAL_DIRECT_PLUS_ESTIMATED_TIP_CONTACT_STATE_KEYS,
    "estimated_wall_margin",
    "estimated_wall_margin_fraction",
    "route_estimator_confidence",
    "estimated_magnet_wall_pull",
    "estimated_wall_side_risk",
    "estimated_wall_normal_3d",
    "estimated_route_tangent_3d",
]

SCALAR_STATE_KEYS = {
    "distance_to_wall",
    "contact_strength",
    "contact_flag",
    "path_progress",
    "piper_step",
    "piper_insertion_length",
    "local_radius",
    "tip_estimator_confidence",
    "tip_estimator_visible",
    "estimated_contact_flag",
    "contact_estimator_confidence",
    "estimated_image_distance_px",
    "estimated_wall_margin",
    "estimated_wall_margin_fraction",
    "route_estimator_confidence",
    "estimated_magnet_wall_pull",
    "estimated_wall_side_risk",
}

ESTIMATED_SCHEMA_VECTOR_DEFAULTS = {
    "estimated_tip_pos_3d": [0.0, 0.0, 0.0],
    "estimated_tip_heading_3d": [0.0, 0.0, 0.0],
    "estimated_wall_normal_3d": [0.0, 0.0, 0.0],
    "estimated_route_tangent_3d": [0.0, 0.0, 0.0],
}

ESTIMATED_SCHEMA_SCALAR_DEFAULTS = {
    "piper_step": 0.0,
    "piper_insertion_length": 0.0,
    "tip_estimator_confidence": 0.0,
    "tip_estimator_visible": 0.0,
    "estimated_contact_flag": 0.0,
    "contact_estimator_confidence": 0.0,
    "estimated_image_distance_px": 0.0,
    "estimated_wall_margin": 0.0,
    "estimated_wall_margin_fraction": 0.0,
    "route_estimator_confidence": 0.0,
    "estimated_magnet_wall_pull": 0.0,
    "estimated_wall_side_risk": 0.0,
}

ESTIMATED_OBSERVATION_SCHEMAS = {
    "real_direct_plus_estimated_tip_contact",
    "real_direct_plus_estimated_tip_registered_geometry",
}


def _elite_tcp_pose6d_from_state(state: dict) -> list[float]:
    pose = state.get("elite_tcp_pose_6d")
    if pose is None:
        pose = state.get("robot_state", {}).get("elite_tcp_pose_6d")
    if pose is not None:
        arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if arr.size != 6:
            raise ValueError(f"elite_tcp_pose_6d expected 6 values, got {arr.size}")
        return arr.astype(np.float32).tolist()

    # Legacy datasets only stored a 3D sim tool point. Use millimeter xyz plus
    # zero orientation as an explicit compatibility fallback.
    xyz = np.asarray(state.get("elirobot_pose", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    if xyz.size != 3:
        raise ValueError(f"elirobot_pose fallback expected 3 values, got {xyz.size}")
    return [float(v) for v in (xyz * 1000.0).tolist()] + [0.0, 0.0, 0.0]


def _estimated_schema_item_from_state(state: dict, key: str):
    if key == "elite_tcp_pose_6d":
        return _elite_tcp_pose6d_from_state(state)
    if key in ESTIMATED_SCHEMA_VECTOR_DEFAULTS:
        item = state.get(key)
        if item is None:
            item = ESTIMATED_SCHEMA_VECTOR_DEFAULTS[key]
        arr = np.asarray(item, dtype=np.float32).reshape(-1)
        expected = len(ESTIMATED_SCHEMA_VECTOR_DEFAULTS[key])
        if arr.size != expected:
            raise ValueError(f"{key} expected {expected} values, got {arr.size}")
        return arr.astype(np.float32).tolist()
    if key in ESTIMATED_SCHEMA_SCALAR_DEFAULTS:
        item = state.get(key, ESTIMATED_SCHEMA_SCALAR_DEFAULTS[key])
        if item is None:
            item = ESTIMATED_SCHEMA_SCALAR_DEFAULTS[key]
        return float(item)
    return state[key]


def _joint_names_from_mapping(mapping):
    if isinstance(mapping, dict):
        return list(mapping.keys())
    return []


def _joint_values_from_mapping(mapping, names):
    if not names:
        return np.asarray([], dtype=np.float32)
    if isinstance(mapping, dict):
        return np.asarray([float(mapping[name]) for name in names], dtype=np.float32)
    arr = np.asarray(mapping, dtype=np.float32).reshape(-1)
    if len(arr) != len(names):
        raise ValueError(f"Expected {len(names)} joint values, got {len(arr)}")
    return arr.astype(np.float32)


def build_state_vector(
    sample,
    max_steps: int,
    piper_joint_names=None,
    elite_joint_names=None,
    observation_schema: str = "full_sim_state",
    piper_command_period_for_state: int = 40,
):
    state = sample["state"]
    values = []
    if observation_schema == "full_sim_state":
        state_keys = FULL_SIM_STATE_KEYS
    elif observation_schema == "real_direct":
        state_keys = REAL_DIRECT_STATE_KEYS
    elif observation_schema == "senior_piper_real_like":
        state_keys = SENIOR_PIPER_REAL_LIKE_STATE_KEYS
    elif observation_schema == "senior_piper_real_like_with_phase":
        state_keys = SENIOR_PIPER_REAL_LIKE_WITH_PHASE_STATE_KEYS
    elif observation_schema == "real_direct_plus_estimated_tip_contact":
        state_keys = REAL_DIRECT_PLUS_ESTIMATED_TIP_CONTACT_STATE_KEYS
    elif observation_schema == "real_direct_plus_estimated_tip_registered_geometry":
        state_keys = REAL_DIRECT_PLUS_ESTIMATED_TIP_REGISTERED_GEOMETRY_STATE_KEYS
    else:
        raise ValueError(f"Unsupported observation_schema: {observation_schema}")
    for key in state_keys:
        if observation_schema in ESTIMATED_OBSERVATION_SCHEMAS:
            item = _estimated_schema_item_from_state(state, key)
        else:
            item = _elite_tcp_pose6d_from_state(state) if key == "elite_tcp_pose_6d" else state[key]
        if key in SCALAR_STATE_KEYS:
            values.append(float(item))
        else:
            values.extend(item)
    if observation_schema in {
        "senior_piper_real_like",
        "senior_piper_real_like_with_phase",
        *ESTIMATED_OBSERVATION_SCHEMAS,
    }:
        if observation_schema == "senior_piper_real_like_with_phase":
            period = max(int(piper_command_period_for_state), 1)
            phase = (float(sample.get("step", 0.0)) % period) / float(period)
            values.append(float(np.sin(2.0 * np.pi * phase)))
            values.append(float(np.cos(2.0 * np.pi * phase)))
        values.append(float(sample["task"] == "left"))
        values.append(float(sample["task"] == "right"))
        return np.asarray(values, dtype=np.float32)
    robot_state = state.get("robot_state", {})
    piper_joints = robot_state.get("piper_joints", {})
    elite_joints = robot_state.get("elite_joints", {})
    if piper_joint_names is None and isinstance(piper_joints, dict):
        piper_joint_names = list(piper_joints.keys())
    if elite_joint_names is None and isinstance(elite_joints, dict):
        elite_joint_names = list(elite_joints.keys())
    values.extend(_joint_values_from_mapping(piper_joints, piper_joint_names or []).tolist())
    values.extend(_joint_values_from_mapping(elite_joints, elite_joint_names or []).tolist())
    step_norm = float(sample["step"]) / max(max_steps, 1)
    values.append(step_norm)
    values.append(float(sample["task"] == "left"))
    values.append(float(sample["task"] == "right"))
    return np.asarray(values, dtype=np.float32)


PIPER_STEP_CLASS_VALUES = [-1, 0, 1]


def _piper_step_class_from_action(action: dict) -> int:
    if isinstance(action, dict) and "piper_step_command" in action:
        step_command = int(np.clip(round(float(action.get("piper_step_command", 0))), -1, 1))
    else:
        feed = float(np.asarray(action.get("piper_feed", 0.0), dtype=np.float32).reshape(-1)[0])
        step_command = int(np.sign(feed)) if abs(feed) > 0.05 else 0
    return PIPER_STEP_CLASS_VALUES.index(step_command)


def _elite_tcp_delta6d_from_sample(sample: dict) -> np.ndarray:
    action = sample.get("action", {})
    if "elite_tcp_delta_6d" in action:
        delta = np.asarray(action["elite_tcp_delta_6d"], dtype=np.float32).reshape(-1)
        if delta.size != 6:
            raise ValueError(f"elite_tcp_delta_6d expected 6 values, got {delta.size}")
        return delta.astype(np.float32)
    if "elite_tcp_pose_6d" in action:
        current = np.asarray(_elite_tcp_pose6d_from_state(sample.get("state", {})), dtype=np.float32)
        target = np.asarray(action["elite_tcp_pose_6d"], dtype=np.float32).reshape(-1)
        if target.size != 6:
            raise ValueError(f"elite_tcp_pose_6d expected 6 values, got {target.size}")
        return (target.astype(np.float32) - current.astype(np.float32)).astype(np.float32)
    raise ValueError(
        "elite_action_representation='tcp_delta' requires action.elite_tcp_delta_6d "
        "or action.elite_tcp_pose_6d. Recollect or backfill the dataset with TCP action fields."
    )


def build_action_vector(
    sample,
    piper_joint_names=None,
    elite_joint_names=None,
    elite_action_representation: str = "absolute",
    piper_head: str = "feed_regression",
):
    action = sample["action"]
    if isinstance(action, dict) and "piper_feed" in action and "elite_joints" in action:
        if elite_action_representation == "tcp_delta":
            elite_vec = _elite_tcp_delta6d_from_sample(sample)
        else:
            elite = action["elite_joints"]
            if elite_joint_names is None and isinstance(elite, dict):
                elite_joint_names = list(elite.keys())
            elite_vec = _joint_values_from_mapping(elite, elite_joint_names or [])
            if elite_action_representation == "delta":
                robot_state = sample.get("state", {}).get("robot_state", {})
                current_elite = _joint_values_from_mapping(robot_state.get("elite_joints", {}), elite_joint_names or [])
                elite_vec = elite_vec - current_elite
            elif elite_action_representation != "absolute":
                raise ValueError(f"Unsupported elite_action_representation: {elite_action_representation}")
        if piper_head == "feed_regression":
            piper_vec = [float(np.asarray(action["piper_feed"], dtype=np.float32).reshape(-1)[0])]
        elif piper_head == "step_classification":
            piper_vec = [0.0, 0.0, 0.0]
            piper_vec[_piper_step_class_from_action(action)] = 1.0
        else:
            raise ValueError(f"Unsupported piper_head: {piper_head}")
        return np.asarray([*piper_vec, *elite_vec.tolist()], dtype=np.float32)
    if isinstance(action, dict) and "piper_joints" in action and "elite_joints" in action:
        piper = action["piper_joints"]
        elite = action["elite_joints"]
        if piper_joint_names is None and isinstance(piper, dict):
            piper_joint_names = list(piper.keys())
        if elite_joint_names is None and isinstance(elite, dict):
            elite_joint_names = list(elite.keys())
        piper_vec = _joint_values_from_mapping(piper, piper_joint_names or [])
        elite_vec = _joint_values_from_mapping(elite, elite_joint_names or [])
        return np.asarray([*piper_vec.tolist(), *elite_vec.tolist()], dtype=np.float32)
    if isinstance(action, dict) and "piper" in action and "elirobot_delta" in action:
        piper = float(action["piper"][0] if isinstance(action["piper"], (list, tuple, np.ndarray)) else action["piper"])
        delta = np.asarray(action["elirobot_delta"], dtype=np.float32).reshape(-1)
        delta = np.tanh(delta).astype(np.float32)
        return np.asarray([piper, *delta.tolist()], dtype=np.float32)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    return arr.astype(np.float32)


def sample_action_weight(sample):
    state = sample.get("state", {})
    contact_strength = float(state.get("contact_strength", 0.0))
    distance_to_wall = float(state.get("distance_to_wall", 1.0))
    boundary_projection_count = int(state.get("boundary_projection_count", 0))
    last_boundary_projection = bool(state.get("last_boundary_projection", False))

    weight = 1.0
    if contact_strength >= 0.25:
        weight *= 1.8
    if contact_strength >= 0.45:
        weight *= 1.6
    if distance_to_wall <= 0.014:
        weight *= 1.4
    if boundary_projection_count > 0:
        weight *= 1.5
    if last_boundary_projection:
        weight *= 1.5
    if sample.get("scenario", "") == "on_policy_recovery":
        weight *= 1.2
    return float(weight)


class DualArmGuidewireDataset(Dataset):
    def __init__(
        self,
        manifest_paths,
        image_size: int = 224,
        max_steps: int = 280,
        strict_files: bool = False,
        manifest_weights=None,
        success_sample_weight: float = 1.0,
        scenario_weights=None,
        elite_action_representation: str = "absolute",
        piper_head: str = "feed_regression",
        observation_schema: str = "full_sim_state",
        piper_command_period_for_state: int = 40,
    ):
        if isinstance(manifest_paths, (str, Path)):
            manifest_paths = [manifest_paths]
        self.manifest_paths = [Path(p) for p in manifest_paths]
        if not self.manifest_paths:
            raise ValueError("At least one manifest path is required")
        if manifest_weights is None:
            manifest_weights = [1.0] * len(self.manifest_paths)
        elif len(manifest_weights) == 1 and len(self.manifest_paths) > 1:
            manifest_weights = list(manifest_weights) * len(self.manifest_paths)
        elif len(manifest_weights) != len(self.manifest_paths):
            raise ValueError("manifest_weights must match manifest_paths or contain a single broadcastable value")

        self.samples = []
        self.sample_weights = []
        missing = 0
        for manifest_path, manifest_weight in zip(self.manifest_paths, manifest_weights):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            episode_success = {episode["episode"]: bool(episode.get("success", False)) for episode in manifest.get("episodes", [])}
            samples = [s for s in manifest["samples"] if "side" in s["images"] and "top" in s["images"]]
            for sample in samples:
                side_path = self._resolve_path(sample["images"]["side"])
                top_path = self._resolve_path(sample["images"]["top"])
                if side_path.exists() and top_path.exists():
                    self.samples.append(sample)
                    sample_weight = float(manifest_weight)
                    if episode_success.get(sample.get("episode"), False):
                        sample_weight *= float(success_sample_weight)
                    if scenario_weights:
                        sample_weight *= float(scenario_weights.get(sample.get("scenario", ""), 1.0))
                    sample_weight *= sample_action_weight(sample)
                    self.sample_weights.append(sample_weight)
                else:
                    missing += 1
                    if strict_files:
                        missing_path = side_path if not side_path.exists() else top_path
                        raise FileNotFoundError(missing_path)
        if missing:
            print(f"Warning: skipped {missing} samples with missing side/top images")
        if not self.samples:
            raise RuntimeError("No dual-arm camera_pair samples found in manifests")
        self.sample_weights = np.asarray(self.sample_weights, dtype=np.float32)
        self.max_steps = max_steps
        self.elite_action_representation = str(elite_action_representation)
        self.piper_head = str(piper_head)
        self.observation_schema = str(observation_schema)
        self.piper_command_period_for_state = int(piper_command_period_for_state)
        first_state = self.samples[0].get("state", {})
        first_robot_state = first_state.get("robot_state", {})
        self.piper_joint_names = _joint_names_from_mapping(first_robot_state.get("piper_joints", {}))
        self.elite_joint_names = _joint_names_from_mapping(first_robot_state.get("elite_joints", {}))
        first_action = self.samples[0].get("action", {})
        if isinstance(first_action, dict) and "piper_feed" in first_action and "elite_joints" in first_action:
            self.action_mode = "piper_feed_elite_joint"
        elif isinstance(first_action, dict) and "piper_joints" in first_action and "elite_joints" in first_action:
            self.action_mode = "joint"
        else:
            self.action_mode = "legacy"
        if self.action_mode == "joint":
            if not self.piper_joint_names:
                self.piper_joint_names = _joint_names_from_mapping(first_action.get("piper_joints", {}))
            if not self.elite_joint_names:
                self.elite_joint_names = _joint_names_from_mapping(first_action.get("elite_joints", {}))
        elif self.action_mode == "piper_feed_elite_joint":
            if not self.elite_joint_names:
                self.elite_joint_names = _joint_names_from_mapping(first_action.get("elite_joints", {}))
        self.action_dim = int(
            build_action_vector(
                self.samples[0],
                self.piper_joint_names,
                self.elite_joint_names,
                self.elite_action_representation,
                self.piper_head,
            ).shape[0]
        )
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _resolve_path(path_str):
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def __len__(self):
        return len(self.samples)

    def _read_image(self, path_str):
        path = self._resolve_path(path_str)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        side = self._read_image(sample["images"]["side"])
        top = self._read_image(sample["images"]["top"])
        state = torch.tensor(
            build_state_vector(
                sample,
                self.max_steps,
                self.piper_joint_names,
                self.elite_joint_names,
                self.observation_schema,
                self.piper_command_period_for_state,
            ),
            dtype=torch.float32,
        )
        action = torch.tensor(
            build_action_vector(
                sample,
                self.piper_joint_names,
                self.elite_joint_names,
                self.elite_action_representation,
                self.piper_head,
            ),
            dtype=torch.float32,
        )
        return {"side": side, "top": top, "state": state, "action": action}


def build_temporal_pair_indices(dataset: DualArmGuidewireDataset, allowed_indices) -> list[tuple[int, int]]:
    allowed = {int(idx) for idx in allowed_indices}
    by_episode: dict[str, list[int]] = {}
    for idx in allowed:
        sample = dataset.samples[idx]
        by_episode.setdefault(str(sample.get("episode", "")), []).append(idx)

    pairs: list[tuple[int, int]] = []
    for indices in by_episode.values():
        indices.sort(key=lambda i: int(dataset.samples[i].get("step", 0)))
        for cur_idx, next_idx in zip(indices, indices[1:]):
            cur = dataset.samples[cur_idx]
            nxt = dataset.samples[next_idx]
            if cur.get("task") == nxt.get("task") and int(nxt.get("step", 0)) > int(cur.get("step", 0)):
                pairs.append((cur_idx, next_idx))
    return pairs


class TemporalPairDataset(Dataset):
    def __init__(self, base: DualArmGuidewireDataset, pairs: list[tuple[int, int]]):
        self.base = base
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cur_idx, next_idx = self.pairs[idx]
        cur = self.base[cur_idx]
        nxt = self.base[next_idx]
        return {
            "side": cur["side"],
            "top": cur["top"],
            "state": cur["state"],
            "action": cur["action"],
            "next_side": nxt["side"],
            "next_top": nxt["top"],
            "next_state": nxt["state"],
            "next_action": nxt["action"],
        }


class DualArmCameraStatePolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        side_resnet = models.resnet18(weights=weights)
        top_resnet = models.resnet18(weights=weights)
        self.side_encoder = nn.Sequential(*list(side_resnet.children())[:-1])
        self.top_encoder = nn.Sequential(*list(top_resnet.children())[:-1])
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(512 + 512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(128, action_dim)

    def forward(self, side, top, state):
        side_feat = self.side_encoder(side).flatten(1)
        top_feat = self.top_encoder(top).flatten(1)
        state_feat = self.state_encoder(state)
        feat = self.trunk(torch.cat([side_feat, top_feat, state_feat], dim=1))
        return self.action_head(feat)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion,
    piper_dim: int,
    action_mode: str,
    piper_feed_negative_loss_weight: float,
    piper_head: str,
):
    model.eval()
    total_loss = total_piper = total_elite = 0.0
    total_count = 0
    for batch in loader:
        side = batch["side"].to(device)
        top = batch["top"].to(device)
        state = batch["state"].to(device)
        target = batch["action"].to(device)
        pred = model(side, top, state)
        piper_t = target[:, :piper_dim]
        elite_t = target[:, piper_dim:]
        piper_p = pred[:, :piper_dim]
        elite_p = pred[:, piper_dim:]
        loss_piper = piper_loss(piper_p, piper_t, action_mode, piper_feed_negative_loss_weight, piper_head)
        loss_elite = criterion(elite_p, elite_t)
        loss = loss_piper + loss_elite
        bs = len(target)
        total_loss += float(loss.item()) * bs
        total_piper += float(loss_piper.item()) * bs
        total_elite += float(loss_elite.item()) * bs
        total_count += bs
    denom = max(total_count, 1)
    return total_loss / denom, total_piper / denom, total_elite / denom


def piper_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_mode: str,
    negative_weight: float,
    piper_head: str = "feed_regression",
) -> torch.Tensor:
    if action_mode == "piper_feed_elite_joint" and piper_head == "step_classification":
        return F.cross_entropy(pred, torch.argmax(target, dim=1))
    if action_mode != "piper_feed_elite_joint" or float(negative_weight) == 1.0:
        return F.smooth_l1_loss(pred, target)
    per_item = F.smooth_l1_loss(pred, target, reduction="none")
    weights = torch.where(target < -0.05, torch.full_like(target, float(negative_weight)), torch.ones_like(target))
    return (per_item * weights).sum() / weights.sum().clamp_min(1.0)


def elite_temporal_smoothness_loss(
    model,
    batch,
    device,
    piper_dim: int,
) -> tuple[torch.Tensor, int]:
    side = batch["side"].to(device)
    top = batch["top"].to(device)
    state = batch["state"].to(device)
    action = batch["action"].to(device)
    next_side = batch["next_side"].to(device)
    next_top = batch["next_top"].to(device)
    next_state = batch["next_state"].to(device)
    next_action = batch["next_action"].to(device)

    pred = model(side, top, state)
    next_pred = model(next_side, next_top, next_state)
    pred_delta = next_pred[:, piper_dim:] - pred[:, piper_dim:]
    target_delta = next_action[:, piper_dim:] - action[:, piper_dim:]
    return F.smooth_l1_loss(pred_delta, target_delta), len(action)


@torch.no_grad()
def evaluate_temporal_smoothness(model, loader, device, piper_dim: int) -> float:
    if loader is None:
        return 0.0
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        loss, bs = elite_temporal_smoothness_loss(model, batch, device, piper_dim)
        total += float(loss.item()) * bs
        count += bs
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="Train a simple dual-arm + dual-camera baseline policy.")
    parser.add_argument("--manifest", nargs="+", default=["simulation_output/dual_arm_dataset_aug/manifest.json"])
    parser.add_argument("--manifest-weights", nargs="+", type=float, default=[1.0])
    parser.add_argument("--success-sample-weight", type=float, default=1.0)
    parser.add_argument("--scenario-weight", action="append", default=[], help="Optional scenario weight, e.g. success_recovery=2.0")
    parser.add_argument("--out", default="simulation_output/baseline_dual_arm_camera_state")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--strict-files", action="store_true")
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument(
        "--piper-feed-negative-loss-weight",
        type=float,
        default=1.0,
        help="Extra loss weight for negative piper_feed samples in piper_feed_elite_joint mode.",
    )
    parser.add_argument(
        "--elite-smoothness-weight",
        type=float,
        default=0.0,
        help="Weight for adjacent-sample Elite joint delta loss. Keeps action schema unchanged.",
    )
    parser.add_argument(
        "--temporal-batch-size",
        type=int,
        default=0,
        help="Batch size for temporal smoothness pairs. Defaults to --batch-size when <= 0.",
    )
    parser.add_argument(
        "--elite-action-representation",
        choices=["absolute", "delta", "tcp_delta"],
        default="absolute",
        help=(
            "Train Elite outputs as absolute joint targets, deltas from current Elite joints, "
            "or real-style TCP 6D deltas from the current Elite TCP pose. Rollout still executes "
            "through the MuJoCo/robot joint layer."
        ),
    )
    parser.add_argument(
        "--observation-schema",
        choices=[
            "full_sim_state",
            "real_direct",
            "senior_piper_real_like",
            "senior_piper_real_like_with_phase",
            "real_direct_plus_estimated_tip_contact",
            "real_direct_plus_estimated_tip_registered_geometry",
        ],
        default="full_sim_state",
        help=(
            "Structured-state inputs for the policy. full_sim_state preserves legacy simulator fields; "
            "real_direct uses robot/Piper/task fields that have direct real-system counterparts; "
            "senior_piper_real_like uses Elite TCP 6D pose, Piper step, and task id to match the inherited Piper input; "
            "senior_piper_real_like_with_phase additionally includes an explicit controller phase sin/cos for "
            "diagnosing scheduled Piper labels; real_direct_plus_estimated_tip_contact adds estimator-derived "
            "guidewire tip/contact fields with defined real acquisition paths; "
            "real_direct_plus_estimated_tip_registered_geometry additionally includes registered-route wall-margin "
            "and wall-side guidance-risk estimator fields."
        ),
    )
    parser.add_argument(
        "--piper-command-period-for-state",
        type=int,
        default=40,
        help=(
            "Controller-period used only by senior_piper_real_like_with_phase to encode "
            "sin/cos(step modulo period). This is a real controller clock/state diagnostic, not simulator oracle feedback."
        ),
    )
    parser.add_argument(
        "--piper-head",
        choices=["feed_regression", "step_classification"],
        default="feed_regression",
        help=(
            "Use the legacy continuous piper_feed regression target or classify "
            "piper_step_command as retract/hold/feed while preserving piper_feed_elite_joint rollout actions."
        ),
    )
    parser.add_argument(
        "--piper-step-feed-value",
        type=float,
        default=0.7,
        help="Continuous piper_feed command emitted at rollout for a classified feed step.",
    )
    parser.add_argument(
        "--piper-step-retract-value",
        type=float,
        default=0.7,
        help="Absolute continuous piper_feed command emitted at rollout for a classified retract step.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(args.manifest_weights) == 1 and len(args.manifest) > 1:
        manifest_weights = args.manifest_weights * len(args.manifest)
    elif len(args.manifest_weights) == len(args.manifest):
        manifest_weights = args.manifest_weights
    else:
        raise ValueError("--manifest-weights must contain either one value or one value per manifest")
    scenario_weights = {}
    for item in args.scenario_weight:
        if "=" not in item:
            raise ValueError("--scenario-weight must use NAME=WEIGHT format")
        key, value = item.split("=", 1)
        scenario_weights[key] = float(value)
    dataset = DualArmGuidewireDataset(
        args.manifest,
        image_size=args.image_size,
        max_steps=args.max_steps,
        strict_files=args.strict_files,
        manifest_weights=manifest_weights,
        success_sample_weight=args.success_sample_weight,
        scenario_weights=scenario_weights,
        elite_action_representation=args.elite_action_representation,
        piper_head=args.piper_head,
        observation_schema=args.observation_schema,
        piper_command_period_for_state=args.piper_command_period_for_state,
    )
    state_dim = int(
        build_state_vector(
            dataset.samples[0],
            args.max_steps,
            dataset.piper_joint_names,
            dataset.elite_joint_names,
            args.observation_schema,
            args.piper_command_period_for_state,
        ).shape[0]
    )
    action_dim = int(dataset.action_dim)
    if dataset.action_mode == "joint":
        piper_dim = len(dataset.piper_joint_names)
    elif dataset.action_mode == "piper_feed_elite_joint" and args.piper_head == "step_classification":
        piper_dim = len(PIPER_STEP_CLASS_VALUES)
    else:
        piper_dim = 1
    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_weights = dataset.sample_weights[train_set.indices]
    train_sampler = WeightedRandomSampler(weights=torch.as_tensor(train_weights, dtype=torch.double), num_samples=len(train_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    temporal_batch_size = args.temporal_batch_size if args.temporal_batch_size > 0 else args.batch_size
    train_temporal_loader = None
    val_temporal_loader = None
    train_temporal_pairs = []
    val_temporal_pairs = []
    if args.elite_smoothness_weight > 0.0:
        train_temporal_pairs = build_temporal_pair_indices(dataset, train_set.indices)
        val_temporal_pairs = build_temporal_pair_indices(dataset, val_set.indices)
        if train_temporal_pairs:
            train_temporal_loader = DataLoader(
                TemporalPairDataset(dataset, train_temporal_pairs),
                batch_size=temporal_batch_size,
                shuffle=True,
                num_workers=0,
            )
        if val_temporal_pairs:
            val_temporal_loader = DataLoader(
                TemporalPairDataset(dataset, val_temporal_pairs),
                batch_size=temporal_batch_size,
                shuffle=False,
                num_workers=0,
            )
        print(
            f"temporal smoothness pairs: train={len(train_temporal_pairs)} "
            f"val={len(val_temporal_pairs)} weight={args.elite_smoothness_weight}"
        )

    model = DualArmCameraStatePolicy(state_dim=state_dim, action_dim=action_dim, pretrained=args.pretrained).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "manifest": args.manifest,
        "manifest_weights": manifest_weights,
        "success_sample_weight": args.success_sample_weight,
        "scenario_weights": scenario_weights,
        "device": str(device),
        "samples": len(dataset),
        "train_samples": train_len,
        "val_samples": val_len,
        "state_dim": state_dim,
        "observation_schema": args.observation_schema,
        "piper_command_period_for_state": args.piper_command_period_for_state,
        "action_dim": action_dim,
        "action_mode": dataset.action_mode,
        "piper_feed_negative_loss_weight": args.piper_feed_negative_loss_weight,
        "piper_head": args.piper_head,
        "piper_step_class_values": PIPER_STEP_CLASS_VALUES,
        "piper_step_feed_value": args.piper_step_feed_value,
        "piper_step_retract_value": args.piper_step_retract_value,
        "elite_smoothness_weight": args.elite_smoothness_weight,
        "elite_action_representation": args.elite_action_representation,
        "temporal_batch_size": temporal_batch_size,
        "train_temporal_pairs": len(train_temporal_pairs),
        "val_temporal_pairs": len(val_temporal_pairs),
        "piper_joint_names": dataset.piper_joint_names,
        "elite_joint_names": dataset.elite_joint_names,
        "epochs": [],
    }
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_piper = total_elite = total_smooth = 0.0
        total_count = 0
        total_smooth_count = 0
        temporal_iter = iter(train_temporal_loader) if train_temporal_loader is not None else None

        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch:03d} [train]", leave=False)
        for batch in train_pbar:
            side = batch["side"].to(device)
            top = batch["top"].to(device)
            state = batch["state"].to(device)
            target = batch["action"].to(device)
            piper_t = target[:, :piper_dim]
            elite_t = target[:, piper_dim:]

            pred = model(side, top, state)
            piper_p = pred[:, :piper_dim]
            elite_p = pred[:, piper_dim:]
            loss_piper = piper_loss(
                piper_p,
                piper_t,
                dataset.action_mode,
                args.piper_feed_negative_loss_weight,
                args.piper_head,
            )
            loss_elite = criterion(elite_p, elite_t)
            loss_smooth = torch.zeros((), device=device)
            smooth_bs = 0
            if temporal_iter is not None:
                try:
                    temporal_batch = next(temporal_iter)
                except StopIteration:
                    temporal_iter = iter(train_temporal_loader)
                    temporal_batch = next(temporal_iter)
                loss_smooth, smooth_bs = elite_temporal_smoothness_loss(model, temporal_batch, device, piper_dim)
            loss = loss_piper + loss_elite + float(args.elite_smoothness_weight) * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(target)
            total_loss += float(loss.item()) * bs
            total_piper += float(loss_piper.item()) * bs
            total_elite += float(loss_elite.item()) * bs
            if smooth_bs:
                total_smooth += float(loss_smooth.item()) * smooth_bs
                total_smooth_count += smooth_bs
            total_count += bs
            train_pbar.set_postfix(
                loss=f"{total_loss / max(total_count, 1):.4f}",
                piper=f"{total_piper / max(total_count, 1):.4f}",
                elite=f"{total_elite / max(total_count, 1):.4f}",
                smooth=f"{total_smooth / max(total_smooth_count, 1):.4f}",
            )
        train_pbar.close()

        train_loss = total_loss / max(total_count, 1)
        train_piper = total_piper / max(total_count, 1)
        train_elite = total_elite / max(total_count, 1)
        train_smooth = total_smooth / max(total_smooth_count, 1)
        val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch:03d} [val]", leave=False)
        model.eval()
        val_total_loss = val_total_piper = val_total_elite = 0.0
        val_total_count = 0
        with torch.no_grad():
            for batch in val_pbar:
                side = batch["side"].to(device)
                top = batch["top"].to(device)
                state = batch["state"].to(device)
                target = batch["action"].to(device)
                piper_t = target[:, :piper_dim]
                elite_t = target[:, piper_dim:]
                pred = model(side, top, state)
                piper_p = pred[:, :piper_dim]
                elite_p = pred[:, piper_dim:]
                loss_piper = piper_loss(
                    piper_p,
                    piper_t,
                    dataset.action_mode,
                    args.piper_feed_negative_loss_weight,
                    args.piper_head,
                )
                loss_elite = criterion(elite_p, elite_t)
                loss = loss_piper + loss_elite
                bs = len(target)
                val_total_loss += float(loss.item()) * bs
                val_total_piper += float(loss_piper.item()) * bs
                val_total_elite += float(loss_elite.item()) * bs
                val_total_count += bs
                val_pbar.set_postfix(
                    loss=f"{val_total_loss / max(val_total_count, 1):.4f}",
                    piper=f"{val_total_piper / max(val_total_count, 1):.4f}",
                    elite=f"{val_total_elite / max(val_total_count, 1):.4f}",
                )
        val_pbar.close()
        val_loss = val_total_loss / max(val_total_count, 1)
        val_piper = val_total_piper / max(val_total_count, 1)
        val_elite = val_total_elite / max(val_total_count, 1)
        val_smooth = evaluate_temporal_smoothness(model, val_temporal_loader, device, piper_dim)
        val_objective = val_loss + float(args.elite_smoothness_weight) * val_smooth
        log["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_piper_loss": train_piper,
                "train_elite_loss": train_elite,
                "train_elite_smoothness_loss": train_smooth,
                "val_loss": val_loss,
                "val_piper_loss": val_piper,
                "val_elite_loss": val_elite,
                "val_elite_smoothness_loss": val_smooth,
                "val_objective": val_objective,
            }
        )
        print(
            f"epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_piper={train_piper:.6f} val_piper={val_piper:.6f} "
            f"train_elite={train_elite:.6f} val_elite={val_elite:.6f} "
            f"train_smooth={train_smooth:.6f} val_smooth={val_smooth:.6f} "
            f"val_objective={val_objective:.6f}"
        )
        if val_objective < best_val:
            best_val = val_objective
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "state_dim": state_dim,
                    "observation_schema": args.observation_schema,
                    "piper_command_period_for_state": args.piper_command_period_for_state,
                    "image_size": args.image_size,
                    "max_steps": args.max_steps,
                    "val_loss": val_loss,
                    "val_objective": best_val,
                    "elite_smoothness_weight": args.elite_smoothness_weight,
                    "elite_action_representation": args.elite_action_representation,
                    "piper_head": args.piper_head,
                    "piper_step_class_values": PIPER_STEP_CLASS_VALUES,
                    "piper_step_feed_value": args.piper_step_feed_value,
                    "piper_step_retract_value": args.piper_step_retract_value,
                    "action_dim": action_dim,
                    "action_mode": dataset.action_mode,
                    "piper_joint_names": dataset.piper_joint_names,
                    "elite_joint_names": dataset.elite_joint_names,
                },
                out_dir / "best_model.pt",
            )

    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved best model and log to {out_dir}")


if __name__ == "__main__":
    main()
