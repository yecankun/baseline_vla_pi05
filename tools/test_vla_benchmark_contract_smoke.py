from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_benchmark_contract import (
    BENCHMARK_PROFILES,
    VisualCorruption,
    apply_visual_corruption,
    decode_project_action,
    encode_project_action,
    profile_fingerprint,
    robustness_change,
    validate_policy_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dependency-light smoke for the VLA benchmark/noise contract.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    expected = {"project_d0", "project_r0", "pusht", "libero_spatial"}
    if set(BENCHMARK_PROFILES) != expected:
        raise AssertionError(f"unexpected benchmark profiles: {sorted(BENCHMARK_PROFILES)}")

    elite = np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    encoded = encode_project_action(elite, 2)
    decoded = decode_project_action(encoded)
    np.testing.assert_allclose(decoded["elite_tcp_delta_6d"], elite, atol=0.0, rtol=0.0)
    if decoded["piper_intent_id"] != 2:
        raise AssertionError("Piper compatibility decode failed")

    y, x = np.mgrid[0:32, 0:40]
    image = np.stack((x / 39.0, y / 31.0, (x + y) / 70.0), axis=-1)
    image = np.rint(image * 255.0).astype(np.uint8)
    corruption_checks: dict[str, dict[str, object]] = {}
    for name in ("gaussian_sensor_noise", "low_contrast", "blur", "occlusion"):
        spec = VisualCorruption(name=name, severity=2)
        first = apply_visual_corruption(
            image,
            sample_key="episode-3/frame-17",
            view_key="observation.images.side",
            corruption=spec,
        )
        second = apply_visual_corruption(
            image,
            sample_key="episode-3/frame-17",
            view_key="observation.images.side",
            corruption=spec,
        )
        if first.shape != image.shape or first.dtype != image.dtype:
            raise AssertionError(f"{name} changed image shape or dtype")
        if not np.array_equal(first, second):
            raise AssertionError(f"{name} is not deterministic")
        if np.array_equal(first, image):
            raise AssertionError(f"{name} did not change the image")
        corruption_checks[name] = {
            "shape_preserved": True,
            "dtype_preserved": True,
            "deterministic": True,
            "mean_absolute_pixel_change": float(np.abs(first.astype(np.float32) - image).mean()),
        }

    validate_policy_observation(
        {
            "observation.images.side": image,
            "observation.images.top": image,
            "observation.state": np.zeros(32, dtype=np.float32),
            "task": "guide left",
        }
    )
    privileged_rejected = False
    try:
        validate_policy_observation({"observation": {"exact_tip_position": [0.0, 0.0, 0.0]}})
    except ValueError:
        privileged_rejected = True
    if not privileged_rejected:
        raise AssertionError("privileged observation was not rejected")

    accuracy_degradation = robustness_change(clean_value=0.8, corrupted_value=0.6, higher_is_better=True)
    error_degradation = robustness_change(clean_value=0.5, corrupted_value=0.7, higher_is_better=False)
    np.testing.assert_allclose(accuracy_degradation["absolute_degradation"], 0.2)
    np.testing.assert_allclose(error_degradation["absolute_degradation"], 0.2)

    report = {
        "status": "passed",
        "training_started": False,
        "optimizer_steps": 0,
        "profiles": {
            name: {
                **profile.to_dict(),
                "fingerprint": profile_fingerprint(profile),
            }
            for name, profile in BENCHMARK_PROFILES.items()
        },
        "project_action_round_trip": {
            "compatibility_action_dim": int(encoded.shape[0]),
            "preferred_output_interface": "elite_tcp_delta_6d + piper_intent_id",
            "piper_intent_id": decoded["piper_intent_id"],
        },
        "visual_corruptions": corruption_checks,
        "noise_protocol": {
            "evaluation_only": True,
            "paired_clean_corrupted": True,
            "fixed_episode_frame_view_seed": True,
            "geometric_warp": False,
        },
        "privileged_policy_fields_rejected": privileged_rejected,
        "robustness_metric_examples": {
            "higher_is_better": accuracy_degradation,
            "lower_is_better": error_degradation,
        },
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "status=passed profiles=4 corruptions=4 "
        "project_action_round_trip=passed privileged_fields=blocked optimizer_steps=0"
    )


if __name__ == "__main__":
    main()
