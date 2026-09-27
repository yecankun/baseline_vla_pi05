#!/usr/bin/env python
"""Dependency-light contract smoke for the evaluation visual wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_benchmark_contract import VisualCorruption
from vla_visual_noise_wrapper import EvaluationVisualObservationWrapper, get_mapping_path


class _DummyEnv:
    def __init__(self) -> None:
        y, x = np.mgrid[0:12, 0:16]
        image = np.stack((x * 9, y * 13, x + y), axis=-1).astype(np.uint8)
        self._image = image[None, ...]
        self._step = 0
        self.native_marker = "preserved"

    def _observation(self) -> dict[str, object]:
        return {
            "pixels": {
                "image": self._image.copy(),
                "image2": np.flip(self._image, axis=2).copy(),
            },
            "robot_state": np.asarray([[1.0, 2.0, float(self._step)]], dtype=np.float32),
            "task": ["dummy task"],
        }

    def reset(self, *, seed: int):
        self._step = 0
        return self._observation(), {"seed": seed}

    def step(self, action: np.ndarray):
        self._step += 1
        return self._observation(), np.asarray([0.0]), np.asarray([False]), np.asarray([False]), {}

    def close(self) -> None:
        return None


def _run_once(name: str) -> dict[str, object]:
    flat = {"observation.images.image": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    if get_mapping_path(flat, "observation.images.image") is not flat["observation.images.image"]:
        raise AssertionError("literal dotted-key lookup failed")
    wrapper = EvaluationVisualObservationWrapper(
        _DummyEnv(),
        image_paths=("pixels.image", "pixels.image2"),
        corruption=VisualCorruption(name=name, severity=2),
        seed=20260911,
        episode_key="dummy-episode",
        verify_determinism=True,
    )
    noisy, _ = wrapper.reset(seed=7)
    clean = wrapper.last_clean_observation
    assert clean is not None
    assert wrapper.native_marker == "preserved"
    np.testing.assert_array_equal(noisy["robot_state"], clean["robot_state"])
    assert noisy["task"] == clean["task"]
    assert not np.array_equal(noisy["pixels"]["image"], clean["pixels"]["image"])
    assert not np.array_equal(noisy["pixels"]["image2"], clean["pixels"]["image2"])
    assert all(view["deterministic_duplicate"] for view in wrapper.frame_records[0]["views"])
    first_hashes = [view["corrupted_sha256"] for view in wrapper.frame_records[0]["views"]]
    next_noisy, *_ = wrapper.step(np.zeros((1, 7), dtype=np.float32))
    assert next_noisy["pixels"]["image"].shape == clean["pixels"]["image"].shape

    replay = EvaluationVisualObservationWrapper(
        _DummyEnv(),
        image_paths=("pixels.image", "pixels.image2"),
        corruption=VisualCorruption(name=name, severity=2),
        seed=20260911,
        episode_key="dummy-episode",
    )
    replay.reset(seed=7)
    replay_hashes = [view["corrupted_sha256"] for view in replay.frame_records[0]["views"]]
    if first_hashes != replay_hashes:
        raise AssertionError(f"wrapper replay is not deterministic for {name}")
    return {
        "status": "passed",
        "views": 2,
        "deterministic_replay": True,
        "non_image_fields_preserved": True,
        "shape_and_dtype_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = {
        name: _run_once(name)
        for name in ("gaussian_sensor_noise", "low_contrast", "blur", "occlusion")
    }
    report = {
        "schema_version": "vla_visual_noise_wrapper_smoke_v1",
        "status": "passed",
        "corruptions": results,
        "boundaries": {
            "evaluation_only": True,
            "policy_or_checkpoint_loaded": False,
            "dataset_loaded": False,
            "training_started": False,
            "optimizer_steps": 0,
        },
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("status=passed corruptions=4 wrapper_position=preprocess_input optimizer_steps=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
