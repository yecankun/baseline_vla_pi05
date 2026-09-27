from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ACTIVE_ACTION_DIMS = 9
PROJECT_ELITE_SLICE = (0, 6)
PROJECT_PIPER_SLICE = (6, 9)

FORBIDDEN_POLICY_KEY_FRAGMENTS = (
    "diagnostic_targets",
    "exact_contact",
    "exact_tip",
    "wall_distance",
    "route_truth",
    "raw_event_id",
    "event_id",
)


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    domain: str
    image_keys: tuple[str, ...]
    image_shape: tuple[int, int, int]
    state_dim: int
    action_dim: int
    task_conditioning: str
    action_contract: str
    evidence_scope: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_keys"] = list(self.image_keys)
        value["image_shape"] = list(self.image_shape)
        return value


BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "project_d0": BenchmarkProfile(
        name="project_d0",
        domain="project_2026_diagnostic_simulation",
        image_keys=("observation.images.side", "observation.images.top"),
        image_shape=(3, 224, 224),
        state_dim=32,
        action_dim=PROJECT_ACTIVE_ACTION_DIMS,
        task_conditioning="required_left_or_right_instruction",
        action_contract="project_compat_9d_decoded_to_elite_tcp_delta_6d_plus_piper_intent_id",
        evidence_scope="diagnostic_translation_plus_hold_feed_only",
    ),
    "project_r0": BenchmarkProfile(
        name="project_r0",
        domain="project_2026_current_real_pilot",
        image_keys=("observation.images.side", "observation.images.top"),
        image_shape=(3, 224, 224),
        state_dim=32,
        action_dim=PROJECT_ACTIVE_ACTION_DIMS,
        task_conditioning="required_left_or_right_instruction",
        action_contract="project_compat_9d_decoded_to_elite_tcp_delta_6d_plus_piper_intent_id",
        evidence_scope="data_audit_required_before_any_training_or_scoring",
    ),
    "pusht": BenchmarkProfile(
        name="pusht",
        domain="general_benchmark",
        image_keys=("observation.image",),
        image_shape=(3, 96, 96),
        state_dim=2,
        action_dim=2,
        task_conditioning="constant_instruction_allowed",
        action_contract="native_continuous_2d",
        evidence_scope="architecture_and_visual_robustness_benchmark",
    ),
    "libero_spatial": BenchmarkProfile(
        name="libero_spatial",
        domain="general_benchmark",
        image_keys=("observation.images.image", "observation.images.image2"),
        image_shape=(3, 224, 224),
        state_dim=8,
        action_dim=7,
        task_conditioning="required_native_task_instruction",
        action_contract="native_continuous_7d",
        evidence_scope="language_conditioned_architecture_and_visual_robustness_benchmark",
    ),
}


@dataclass(frozen=True)
class VisualCorruption:
    name: str
    severity: int

    def __post_init__(self) -> None:
        if self.name not in {"gaussian_sensor_noise", "low_contrast", "blur", "occlusion"}:
            raise ValueError(f"unsupported corruption: {self.name}")
        if self.severity not in {1, 2, 3}:
            raise ValueError("severity must be one of 1, 2, or 3")


def profile_fingerprint(profile: BenchmarkProfile) -> str:
    payload = repr(tuple(sorted(profile.to_dict().items()))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_rng(*, sample_key: str, view_key: str, corruption: VisualCorruption, seed: int) -> np.random.Generator:
    payload = f"{seed}|{sample_key}|{view_key}|{corruption.name}|{corruption.severity}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="little", signed=False))


def _to_hwc_float(image: np.ndarray) -> tuple[np.ndarray, bool, np.dtype, float]:
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError(f"expected a three-dimensional image, got {value.shape}")
    channel_first = value.shape[0] in {1, 3, 4} and value.shape[-1] not in {1, 3, 4}
    if channel_first:
        value = np.moveaxis(value, 0, -1)
    original_dtype = value.dtype
    scale = 255.0 if np.issubdtype(original_dtype, np.integer) or float(np.nanmax(value)) > 1.5 else 1.0
    value = value.astype(np.float32) / scale
    if not np.isfinite(value).all():
        raise ValueError("image contains non-finite values")
    return np.clip(value, 0.0, 1.0), channel_first, original_dtype, scale


def _restore_image(value: np.ndarray, *, channel_first: bool, dtype: np.dtype, scale: float) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    if scale == 255.0:
        value = np.rint(value * 255.0)
    value = value.astype(dtype)
    return np.moveaxis(value, -1, 0) if channel_first else value


def _blur_once(value: np.ndarray) -> np.ndarray:
    padded = np.pad(value, ((1, 1), (1, 1), (0, 0)), mode="edge")
    result = np.zeros_like(value)
    for dy in range(3):
        for dx in range(3):
            result += padded[dy : dy + value.shape[0], dx : dx + value.shape[1]]
    return result / 9.0


def apply_visual_corruption(
    image: np.ndarray,
    *,
    sample_key: str,
    view_key: str,
    corruption: VisualCorruption,
    seed: int = 20260911,
) -> np.ndarray:
    """Apply a deterministic, evaluation-only photometric corruption.

    The image shape and pixel geometry are preserved. The caller is responsible
    for retaining the clean sample and using the same episode/frame split.
    """

    value, channel_first, dtype, scale = _to_hwc_float(image)
    rng = _stable_rng(sample_key=sample_key, view_key=view_key, corruption=corruption, seed=seed)

    if corruption.name == "gaussian_sensor_noise":
        sigma = (0.02, 0.05, 0.10)[corruption.severity - 1]
        value = value + rng.normal(0.0, sigma, size=value.shape).astype(np.float32)
    elif corruption.name == "low_contrast":
        contrast = (0.70, 0.45, 0.25)[corruption.severity - 1]
        luminance = value.mean(axis=(0, 1), keepdims=True)
        value = luminance + contrast * (value - luminance)
    elif corruption.name == "blur":
        for _ in range(corruption.severity):
            value = _blur_once(value)
    elif corruption.name == "occlusion":
        fraction = (0.10, 0.20, 0.30)[corruption.severity - 1]
        height, width = value.shape[:2]
        box_h = max(1, int(round(height * fraction)))
        box_w = max(1, int(round(width * fraction)))
        top = int(rng.integers(0, max(1, height - box_h + 1)))
        left = int(rng.integers(0, max(1, width - box_w + 1)))
        value = value.copy()
        value[top : top + box_h, left : left + box_w] = value.mean(axis=(0, 1), keepdims=True)

    return _restore_image(value, channel_first=channel_first, dtype=dtype, scale=scale)


def encode_project_action(elite_tcp_delta_6d: Sequence[float], piper_intent_id: int) -> np.ndarray:
    elite = np.asarray(elite_tcp_delta_6d, dtype=np.float32)
    if elite.shape != (6,) or not np.isfinite(elite).all():
        raise ValueError("elite_tcp_delta_6d must be a finite six-vector")
    if int(piper_intent_id) not in {0, 1, 2}:
        raise ValueError("piper_intent_id must be 0, 1, or 2")
    action = np.zeros(PROJECT_ACTIVE_ACTION_DIMS, dtype=np.float32)
    action[slice(*PROJECT_ELITE_SLICE)] = elite
    action[PROJECT_PIPER_SLICE[0] + int(piper_intent_id)] = 1.0
    return action


def decode_project_action(action_9d: Sequence[float]) -> dict[str, Any]:
    action = np.asarray(action_9d, dtype=np.float32)
    if action.shape != (PROJECT_ACTIVE_ACTION_DIMS,) or not np.isfinite(action).all():
        raise ValueError("project compatibility action must be a finite nine-vector")
    piper_scores = action[slice(*PROJECT_PIPER_SLICE)]
    return {
        "elite_tcp_delta_6d": action[slice(*PROJECT_ELITE_SLICE)].astype(float).tolist(),
        "piper_intent_id": int(np.argmax(piper_scores)),
    }


def _flatten_mapping_keys(value: Mapping[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, child in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        keys.append(full_key)
        if isinstance(child, Mapping):
            keys.extend(_flatten_mapping_keys(child, full_key))
    return keys


def validate_policy_observation(observation: Mapping[str, Any]) -> None:
    for key in _flatten_mapping_keys(observation):
        lowered = key.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_POLICY_KEY_FRAGMENTS):
            raise ValueError(f"privileged or forbidden policy field: {key}")


def robustness_change(*, clean_value: float, corrupted_value: float, higher_is_better: bool) -> dict[str, float]:
    clean = float(clean_value)
    corrupted = float(corrupted_value)
    signed_drop = clean - corrupted if higher_is_better else corrupted - clean
    denominator = abs(clean) if abs(clean) > 1e-12 else 1.0
    return {
        "clean": clean,
        "corrupted": corrupted,
        "absolute_degradation": signed_drop,
        "relative_degradation": signed_drop / denominator,
    }
