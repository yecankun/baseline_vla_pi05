"""Evaluation-only visual corruption wrapper for public benchmark environments.

The wrapper sits between an environment's raw observation and LeRobot policy
preprocessing. It changes only declared image leaves and preserves the native
state, task, action, reward, and termination semantics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vla_benchmark_contract import (
    VisualCorruption,
    apply_visual_corruption,
    validate_policy_observation,
)


RAW_ENV_IMAGE_PATHS: dict[str, tuple[str, ...]] = {
    "pusht": ("pixels",),
    "libero": ("pixels.image", "pixels.image2"),
}


def get_mapping_path(value: Mapping[str, Any], path: str) -> Any:
    # LeRobot processed observations use literal dotted keys, while raw LIBERO
    # observations use nested mappings. Support both without reinterpreting the
    # observation schema.
    if path in value:
        return value[path]
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"missing observation path: {path}")
        current = current[part]
    return current


def _replace_mapping_path(value: Mapping[str, Any], path: str, replacement: Any) -> dict[str, Any]:
    parts = path.split(".")

    def replace(current: Mapping[str, Any], index: int) -> dict[str, Any]:
        key = parts[index]
        if key not in current:
            raise KeyError(f"missing observation path: {path}")
        result = dict(current)
        if index == len(parts) - 1:
            result[key] = replacement
            return result
        child = current[key]
        if not isinstance(child, Mapping):
            raise TypeError(f"observation path is not a mapping before its leaf: {path}")
        result[key] = replace(child, index + 1)
        return result

    return replace(value, 0)


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _corrupt_image_leaf(
    value: Any,
    *,
    sample_prefix: str,
    view_key: str,
    corruption: VisualCorruption,
    seed: int,
    verify_determinism: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean = np.asarray(value)
    if clean.ndim == 3:
        clean_batch = clean[None, ...]
        batched = False
    elif clean.ndim == 4:
        clean_batch = clean
        batched = True
    else:
        raise ValueError(f"image path {view_key} must be HWC/CHW or batched, got {clean.shape}")

    corrupted_items: list[np.ndarray] = []
    deterministic = True
    for batch_index, image in enumerate(clean_batch):
        sample_key = f"{sample_prefix}/batch-{batch_index:03d}"
        first = apply_visual_corruption(
            image,
            sample_key=sample_key,
            view_key=view_key,
            corruption=corruption,
            seed=seed,
        )
        if verify_determinism:
            second = apply_visual_corruption(
                image,
                sample_key=sample_key,
                view_key=view_key,
                corruption=corruption,
                seed=seed,
            )
            deterministic = deterministic and bool(np.array_equal(first, second))
        corrupted_items.append(first)

    noisy_batch = np.stack(corrupted_items, axis=0)
    noisy = noisy_batch if batched else noisy_batch[0]
    if noisy.shape != clean.shape or noisy.dtype != clean.dtype:
        raise AssertionError(f"corruption changed shape or dtype at {view_key}")
    delta = np.abs(noisy.astype(np.float32) - clean.astype(np.float32))
    return noisy, {
        "view_key": view_key,
        "shape": list(clean.shape),
        "dtype": str(clean.dtype),
        "clean_sha256": _sha256_array(clean),
        "corrupted_sha256": _sha256_array(noisy),
        "changed": bool(not np.array_equal(clean, noisy)),
        "mean_absolute_pixel_change": float(delta.mean()),
        "max_absolute_pixel_change": float(delta.max()),
        "deterministic_duplicate": deterministic,
    }


class EvaluationVisualObservationWrapper:
    """Proxy an environment while corrupting only selected raw image leaves."""

    def __init__(
        self,
        environment: Any,
        *,
        image_paths: Sequence[str],
        corruption: VisualCorruption,
        seed: int,
        episode_key: str,
        verify_determinism: bool = False,
    ) -> None:
        if not image_paths:
            raise ValueError("at least one image path is required")
        self.environment = environment
        self.image_paths = tuple(str(path) for path in image_paths)
        self.corruption = corruption
        self.seed = int(seed)
        self.episode_key = str(episode_key)
        self.verify_determinism = bool(verify_determinism)
        self.frame_index = -1
        self.frame_records: list[dict[str, Any]] = []
        self.last_clean_observation: Mapping[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)

    def _corrupt_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("environment observation must be a mapping")
        validate_policy_observation(observation)
        result: Mapping[str, Any] = observation
        view_records: list[dict[str, Any]] = []
        sample_prefix = f"{self.episode_key}/frame-{self.frame_index:06d}"
        for path in self.image_paths:
            clean_image = get_mapping_path(observation, path)
            noisy_image, record = _corrupt_image_leaf(
                clean_image,
                sample_prefix=sample_prefix,
                view_key=path,
                corruption=self.corruption,
                seed=self.seed,
                verify_determinism=self.verify_determinism,
            )
            result = _replace_mapping_path(result, path, noisy_image)
            view_records.append(record)
        self.frame_records.append(
            {
                "sample_prefix": sample_prefix,
                "frame_index": self.frame_index,
                "views": view_records,
            }
        )
        return dict(result)

    def reset(self, *args: Any, **kwargs: Any):
        observation, info = self.environment.reset(*args, **kwargs)
        self.frame_index = 0
        self.frame_records = []
        self.last_clean_observation = observation
        return self._corrupt_observation(observation), info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.environment.step(action)
        self.frame_index += 1
        self.last_clean_observation = observation
        return self._corrupt_observation(observation), reward, terminated, truncated, info

    def close(self) -> Any:
        return self.environment.close()
