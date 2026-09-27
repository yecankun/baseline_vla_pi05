from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualDistanceEstimatorConfig:
    """Rule-based visual contact estimator parameters.

    The estimator intentionally emits image-distance-style fields. It uses the
    simulator only to synthesize what a calibrated side/top visual estimator
    would see, not as a policy-facing MuJoCo contact oracle.
    """

    contact_threshold_px: float = 1.0
    confidence_band_px: float = 8.0
    latency_steps: int = 0
    source: str = "synthetic_visual_distance_estimator"
    boundary_samples: int = 64
    contact_rule: str = "all_visible"
    tip_3d_estimator: bool = False
    tip_3d_source: str = "synthetic_rgbd_tip_estimator"
    tip_3d_frame: str = "world"
    tip_3d_prefer_camera: str = "auto"
    tip_3d_noise_std_m: float = 0.0015
    tip_3d_depth_noise_std_m: float = 0.0008
    tip_3d_pixel_noise_std_px: float = 0.75
    tip_3d_dropout_probability: float = 0.0
    tip_3d_noise_seed: int = 20260628
    registered_geometry_estimator: bool = False
    registered_geometry_source: str = "registered_route_geometry_estimator"
    registered_geometry_margin_threshold_m: float = 0.0012
    registered_geometry_wall_pull_threshold_m: float = 0.0025


def observation_provenance(config: VisualDistanceEstimatorConfig | None = None) -> dict[str, dict[str, Any]]:
    cfg = config or VisualDistanceEstimatorConfig()
    visual_common = {
        "tier": "estimator",
        "source": cfg.source,
        "truth_field": "rendered_image_tip_to_visible_edge_distance",
        "noise_model": f"rendered_image_rule_baseline_{cfg.contact_rule}",
        "latency_steps": int(cfg.latency_steps),
        "dropout_probability": 0.0,
        "formal_policy_input_allowed": True,
    }
    provenance = {
        "estimated_tip_pixel_side": dict(visual_common),
        "estimated_tip_pixel_top": dict(visual_common),
        "estimated_wall_pixel_side": dict(visual_common),
        "estimated_wall_pixel_top": dict(visual_common),
        "tip_estimator_visible": dict(visual_common),
        "tip_estimator_confidence": dict(visual_common),
        "estimated_contact_flag": dict(visual_common),
        "estimated_image_distance_px": dict(visual_common),
        "contact_estimator_confidence": dict(visual_common),
        "contact_source": dict(visual_common),
    }
    if cfg.tip_3d_estimator:
        tip_common = {
            "tier": "estimator",
            "source": cfg.tip_3d_source,
            "truth_field": "tip_pos/heading through rendered red-tip visibility-gated synthetic calibration wrapper",
            "noise_model": (
                "visibility_gated_calibrated_tip_plus_gaussian_noise:"
                f"xyz_std_m={float(cfg.tip_3d_noise_std_m):g},"
                f"depth_std_m={float(cfg.tip_3d_depth_noise_std_m):g},"
                f"pixel_std_px={float(cfg.tip_3d_pixel_noise_std_px):g}"
            ),
            "latency_steps": int(cfg.latency_steps),
            "dropout_probability": float(cfg.tip_3d_dropout_probability),
            "formal_policy_input_allowed": True,
        }
        provenance.update(
            {
                "estimated_tip_pos_3d": dict(tip_common),
                "estimated_tip_pos_frame": dict(tip_common),
                "estimated_tip_camera_pos_3d": dict(tip_common),
                "estimated_tip_depth_m": dict(tip_common),
                "estimated_tip_pixel_source": dict(tip_common),
                "estimated_tip_heading_3d": dict(tip_common),
                "tip_estimator_visible": dict(tip_common),
                "tip_estimator_confidence": dict(tip_common),
                "tip_estimator_latency_steps": dict(tip_common),
            }
        )
    if cfg.registered_geometry_estimator:
        geometry_common = {
            "tier": "estimator",
            "source": cfg.registered_geometry_source,
            "truth_field": "registered_route_geometry_plus_estimated_tip_pos_3d",
            "noise_model": "registered_centerline_wall_margin_from_estimated_tip",
            "latency_steps": int(cfg.latency_steps),
            "dropout_probability": float(cfg.tip_3d_dropout_probability),
            "formal_policy_input_allowed": True,
        }
        provenance.update(
            {
                "estimated_wall_margin": dict(geometry_common),
                "estimated_wall_margin_fraction": dict(geometry_common),
                "estimated_route_progress": dict(geometry_common),
                "estimated_route_index": dict(geometry_common),
                "route_estimator_confidence": dict(geometry_common),
                "registered_route_id": dict(geometry_common),
                "estimated_magnet_wall_pull": dict(geometry_common),
                "estimated_wall_side_risk": dict(geometry_common),
                "estimated_wall_normal_3d": dict(geometry_common),
                "estimated_route_tangent_3d": dict(geometry_common),
                "registered_geometry_estimator": dict(geometry_common),
            }
        )
    return provenance


def estimate_visual_distance_contact(
    env: Any,
    state: dict[str, Any],
    *,
    image_size: int,
    config: VisualDistanceEstimatorConfig | None = None,
    images: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    cfg = config or VisualDistanceEstimatorConfig()
    tip = np.asarray(state["tip_pos"], dtype=np.float32).reshape(3)
    center, _tangent, n1, n2, radius = env._local_path_frame(float(state["path_progress"]))
    center = np.asarray(center, dtype=np.float32).reshape(3)
    n1 = np.asarray(n1, dtype=np.float32).reshape(3)
    n2 = np.asarray(n2, dtype=np.float32).reshape(3)
    angles = np.linspace(0.0, 2.0 * math.pi, max(int(cfg.boundary_samples), 8), endpoint=False, dtype=np.float32)
    boundary_points = np.stack(
        [center + (math.cos(float(angle)) * n1 + math.sin(float(angle)) * n2) * float(radius) for angle in angles],
        axis=0,
    ).astype(np.float32)

    camera_distances: dict[str, float | None] = {}
    tip_pixels: dict[str, list[float] | None] = {}
    wall_pixels: dict[str, list[float] | None] = {}
    visible_count = 0

    for camera_name in ("side", "top"):
        image_result = None
        if images is not None and images.get(camera_name) is not None:
            image_result = _estimate_view_from_rendered_image(images[camera_name])
        if image_result is not None:
            tip_px, wall_px, distance_px = image_result
            camera_distances[camera_name] = distance_px
            tip_pixels[camera_name] = [float(tip_px[0]), float(tip_px[1])]
            wall_pixels[camera_name] = [float(wall_px[0]), float(wall_px[1])]
            visible_count += 1
            continue

        if images is not None:
            camera_distances[camera_name] = None
            tip_pixels[camera_name] = None
            wall_pixels[camera_name] = None
            continue

        try:
            env._configure_free_camera(camera_name)
            projected, valid = _project_world_points_unclipped(
                env,
                np.concatenate([tip[None, :], boundary_points], axis=0),
                image_size,
                image_size,
            )
        except Exception:
            projected = np.empty((0, 2), dtype=np.float32)
            valid = np.zeros((0,), dtype=bool)
        if len(projected) >= 2 and bool(valid[0]) and np.any(valid[1:]):
            tip_px = projected[0].astype(np.float32)
            boundary_px = projected[1:][valid[1:]].astype(np.float32)
            distances = np.linalg.norm(boundary_px - tip_px[None, :], axis=1)
            nearest_index = int(np.argmin(distances))
            wall_px = boundary_px[nearest_index]
            distance_px = float(distances[nearest_index])
            camera_distances[camera_name] = distance_px
            tip_pixels[camera_name] = [float(tip_px[0]), float(tip_px[1])]
            wall_pixels[camera_name] = [float(wall_px[0]), float(wall_px[1])]
            visible_count += 1
        else:
            camera_distances[camera_name] = None
            tip_pixels[camera_name] = None
            wall_pixels[camera_name] = None

    valid_distances = [value for value in camera_distances.values() if value is not None and math.isfinite(value)]
    if valid_distances:
        distance_px = float(np.median(np.asarray(valid_distances, dtype=np.float32)))
        visibility = float(visible_count / 2.0)
        fallback_used = False
    elif images is not None:
        distance_px = None
        visibility = 0.0
        fallback_used = False
    else:
        # Last-resort diagnostic fallback keeps the field populated but records
        # that the image projection path failed.
        distance_to_wall = float(state.get("distance_to_wall", 0.0))
        scene_span = float(np.linalg.norm(np.asarray(env.scene_max, dtype=np.float32) - np.asarray(env.scene_min, dtype=np.float32)))
        pixels_per_meter = float(image_size) / max(scene_span, 1e-6)
        distance_px = max(0.0, distance_to_wall * pixels_per_meter)
        visibility = 0.0
        fallback_used = True

    camera_contact_flags = {
        name: (None if value is None else int(float(value) <= float(cfg.contact_threshold_px)))
        for name, value in camera_distances.items()
    }
    if cfg.contact_rule == "any_visible":
        contact_flag = int(any(value == 1 for value in camera_contact_flags.values()))
    elif cfg.contact_rule == "median":
        contact_flag = int(distance_px is not None and distance_px <= float(cfg.contact_threshold_px))
    else:
        contact_flag = int(visible_count == 2 and all(value == 1 for value in camera_contact_flags.values()))
    if distance_px is None:
        confidence = 0.0
    else:
        if contact_flag:
            margin = max(float(cfg.contact_threshold_px) - distance_px, 0.0)
        else:
            margin = max(distance_px - float(cfg.contact_threshold_px), 0.0)
        confidence = float(np.clip(0.35 + 0.65 * min(margin / max(float(cfg.confidence_band_px), 1e-6), 1.0), 0.0, 1.0))
        confidence *= visibility if not fallback_used else 0.25

    result = {
        "estimated_contact_flag": contact_flag,
        "estimated_image_distance_px": distance_px,
        "contact_estimator_confidence": confidence,
        "contact_source": cfg.source,
        "estimated_tip_pixel_side": tip_pixels["side"],
        "estimated_tip_pixel_top": tip_pixels["top"],
        "estimated_wall_pixel_side": wall_pixels["side"],
        "estimated_wall_pixel_top": wall_pixels["top"],
        "tip_estimator_visible": bool(visible_count > 0),
        "tip_estimator_confidence": float(visibility),
        "estimator_latency_steps": int(cfg.latency_steps),
        "tip_estimator_latency_steps": int(cfg.latency_steps),
        "visual_distance_estimator": {
            "method": "rendered_image_red_tip_to_local_edge_distance" if images is not None else "projected_tip_to_sampled_vessel_wall_mask_distance",
            "camera_distances_px": camera_distances,
            "camera_contact_flags": camera_contact_flags,
            "contact_threshold_px": float(cfg.contact_threshold_px),
            "contact_rule": str(cfg.contact_rule),
            "confidence_band_px": float(cfg.confidence_band_px),
            "boundary_samples": int(cfg.boundary_samples),
            "projection_fallback_used": bool(fallback_used),
        },
    }
    if cfg.tip_3d_estimator:
        result.update(_estimate_tip_3d(env, state, image_size=image_size, config=cfg, tip_pixels=tip_pixels))
    if cfg.registered_geometry_estimator:
        result.update(_estimate_registered_geometry(env, state, config=cfg, estimator_result=result))
    return result


def _estimate_registered_geometry(
    env: Any,
    state: dict[str, Any],
    *,
    config: VisualDistanceEstimatorConfig,
    estimator_result: dict[str, Any],
) -> dict[str, Any]:
    estimated_tip = estimator_result.get("estimated_tip_pos_3d")
    tip_confidence = float(estimator_result.get("tip_estimator_confidence", 0.0) or 0.0)
    if estimated_tip is None or tip_confidence <= 0.0:
        return {
            "estimated_wall_margin": None,
            "estimated_wall_margin_fraction": None,
            "estimated_route_progress": None,
            "estimated_route_index": None,
            "route_estimator_confidence": 0.0,
            "registered_route_id": state.get("task"),
            "estimated_magnet_wall_pull": None,
            "estimated_wall_side_risk": 0.0,
            "estimated_wall_normal_3d": None,
            "estimated_route_tangent_3d": None,
            "registered_geometry_estimator": {
                "method": "registered_route_geometry_from_estimated_tip",
                "source": config.registered_geometry_source,
                "failure": "missing_estimated_tip",
            },
        }

    tip = np.asarray(estimated_tip, dtype=np.float32).reshape(3)
    progress_hint = float(state.get("path_progress", 0.0))
    center, tangent, n1, n2, radius, route_progress, route_index = _nearest_registered_route_frame(
        env,
        tip,
        progress_hint=progress_hint,
    )
    offset = tip - center
    radial = np.array([float(np.dot(offset, n1)), float(np.dot(offset, n2))], dtype=np.float32)
    radial_dist = float(np.linalg.norm(radial))
    wall_margin = float(radius - radial_dist)
    if radial_dist > 1e-8:
        wall_normal = (n1 * radial[0] + n2 * radial[1]) / radial_dist
    else:
        wall_normal = np.zeros(3, dtype=np.float32)
    magnetic = np.asarray(
        state.get("magnetic_pose", state.get("robot_state", {}).get("magnetic_effective_world", state.get("elirobot_pose", [0, 0, 0]))),
        dtype=np.float32,
    ).reshape(3)
    magnet_wall_pull = float(np.dot(magnetic - tip, wall_normal))
    margin_threshold = float(config.registered_geometry_margin_threshold_m)
    pull_threshold = max(float(config.registered_geometry_wall_pull_threshold_m), 1e-8)
    margin_risk = float(np.clip((margin_threshold - wall_margin) / max(margin_threshold, 1e-8), 0.0, 1.0))
    pull_risk = float(np.clip(max(magnet_wall_pull, 0.0) / pull_threshold, 0.0, 1.0))
    side_risk = float(np.clip(0.55 * margin_risk + 0.45 * pull_risk, 0.0, 1.0))
    route_confidence = float(np.clip(tip_confidence * (1.0 - 0.35 * max(-wall_margin / max(radius, 1e-8), 0.0)), 0.0, 1.0))
    return {
        "estimated_wall_margin": wall_margin,
        "estimated_wall_margin_fraction": float(wall_margin / max(float(radius), 1e-8)),
        "estimated_route_progress": float(route_progress),
        "estimated_route_index": int(route_index),
        "route_estimator_confidence": route_confidence,
        "registered_route_id": state.get("task"),
        "estimated_magnet_wall_pull": magnet_wall_pull,
        "estimated_wall_side_risk": side_risk,
        "estimated_wall_normal_3d": wall_normal.astype(float).tolist(),
        "estimated_route_tangent_3d": tangent.astype(float).tolist(),
        "registered_geometry_estimator": {
            "method": "registered_route_geometry_from_estimated_tip",
            "source": config.registered_geometry_source,
            "margin_threshold_m": margin_threshold,
            "wall_pull_threshold_m": float(config.registered_geometry_wall_pull_threshold_m),
            "route_progress_hint": progress_hint,
            "tip_estimator_confidence": tip_confidence,
        },
    }


def _nearest_registered_route_frame(env: Any, tip: np.ndarray, *, progress_hint: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    task = getattr(env, "task", None)
    centerlines = getattr(env, "wire_centerlines", None)
    if task is not None and isinstance(centerlines, dict) and task in centerlines:
        path = np.asarray(centerlines[task], dtype=np.float32).reshape(-1, 3)
        if len(path) > 0:
            hint = int(np.clip(round(float(progress_hint)), 0, len(path) - 1))
            window = max(8, min(32, len(path) // 4 if len(path) >= 4 else len(path)))
            lo = max(0, hint - window)
            hi = min(len(path), hint + window + 1)
            segment = path[lo:hi]
            if len(segment) > 0:
                distances = np.linalg.norm(segment - tip.reshape(1, 3), axis=1)
                route_index = int(lo + int(np.argmin(distances)))
                route_progress = float(route_index)
                center, tangent, n1, n2, radius = env._local_path_frame(route_progress)
                return (
                    np.asarray(center, dtype=np.float32).reshape(3),
                    np.asarray(tangent, dtype=np.float32).reshape(3),
                    np.asarray(n1, dtype=np.float32).reshape(3),
                    np.asarray(n2, dtype=np.float32).reshape(3),
                    float(radius),
                    route_progress,
                    route_index,
                )
    if hasattr(env, "_nearest_path_state"):
        try:
            center, radius = env._nearest_path_state(tip, update_progress=False)
            route_progress = float(getattr(env, "path_progress_float", progress_hint))
            center2, tangent, n1, n2, radius2 = env._local_path_frame(route_progress)
            return (
                np.asarray(center, dtype=np.float32).reshape(3),
                np.asarray(tangent, dtype=np.float32).reshape(3),
                np.asarray(n1, dtype=np.float32).reshape(3),
                np.asarray(n2, dtype=np.float32).reshape(3),
                float(radius if radius is not None else radius2),
                route_progress,
                int(np.clip(round(route_progress), 0, len(env.wire_centerlines[getattr(env, "task", "left")]) - 1)),
            )
        except Exception:
            pass
    route_progress = float(progress_hint)
    center, tangent, n1, n2, radius = env._local_path_frame(route_progress)
    return (
        np.asarray(center, dtype=np.float32).reshape(3),
        np.asarray(tangent, dtype=np.float32).reshape(3),
        np.asarray(n1, dtype=np.float32).reshape(3),
        np.asarray(n2, dtype=np.float32).reshape(3),
        float(radius),
        route_progress,
        int(round(route_progress)),
    )


def _estimate_tip_3d(
    env: Any,
    state: dict[str, Any],
    *,
    image_size: int,
    config: VisualDistanceEstimatorConfig,
    tip_pixels: dict[str, list[float] | None],
) -> dict[str, Any]:
    visible_cameras = [camera_name for camera_name in ("side", "top") if tip_pixels.get(camera_name) is not None]
    source_camera = _select_tip_3d_camera(config, tip_pixels)
    visible = source_camera is not None
    rng = _state_rng(state, config)
    dropout_probability = float(np.clip(config.tip_3d_dropout_probability, 0.0, 1.0))
    dropped = bool(visible and rng.random() < dropout_probability)
    if not visible or dropped:
        return {
            "estimated_tip_pos_3d": None,
            "estimated_tip_pos_frame": str(config.tip_3d_frame),
            "estimated_tip_camera_pos_3d": None,
            "estimated_tip_depth_m": None,
            "estimated_tip_pixel_source": source_camera,
            "estimated_tip_heading_3d": None,
            "tip_estimator_visible": False,
            "tip_estimator_confidence": 0.0,
            "tip_estimator_latency_steps": int(config.latency_steps),
            "tip_estimator": {
                "method": "synthetic_rgbd_red_tip_visibility_gated_calibrated_pose",
                "source": config.tip_3d_source,
                "source_camera": source_camera,
                "visible_cameras": visible_cameras,
                "dropout": bool(dropped),
                "frame": str(config.tip_3d_frame),
            },
        }

    tip = np.asarray(state["tip_pos"], dtype=np.float32).reshape(3)
    env._configure_free_camera(source_camera)
    camera_model = _camera_model(env, image_size, image_size)
    camera_pos = camera_model["cam_pos"]
    right = camera_model["right"]
    up = camera_model["up"]
    forward = camera_model["forward"]
    focal = float(camera_model["focal"])
    rel = tip - camera_pos
    true_camera_xyz = np.asarray([float(rel @ right), float(rel @ up), float(rel @ forward)], dtype=np.float32)
    depth = float(true_camera_xyz[2])
    if depth <= 1e-4 or tip_pixels[source_camera] is None:
        return {
            "estimated_tip_pos_3d": None,
            "estimated_tip_pos_frame": str(config.tip_3d_frame),
            "estimated_tip_camera_pos_3d": None,
            "estimated_tip_depth_m": None,
            "estimated_tip_pixel_source": source_camera,
            "estimated_tip_heading_3d": None,
            "tip_estimator_visible": False,
            "tip_estimator_confidence": 0.0,
            "tip_estimator_latency_steps": int(config.latency_steps),
            "tip_estimator": {
                "method": "synthetic_rgbd_red_tip_visibility_gated_calibrated_pose",
                "source": config.tip_3d_source,
                "source_camera": source_camera,
                "visible_cameras": visible_cameras,
                "failure": "non_positive_depth",
                "frame": str(config.tip_3d_frame),
            },
        }

    px = np.asarray(tip_pixels[source_camera], dtype=np.float32).reshape(2)
    px = px + rng.normal(0.0, float(config.tip_3d_pixel_noise_std_px), size=2).astype(np.float32)
    noisy_depth = max(depth + float(rng.normal(0.0, float(config.tip_3d_depth_noise_std_m))), 1e-4)
    cam_x = (float(px[0]) - image_size * 0.5) * noisy_depth / focal
    cam_y = (image_size * 0.52 - float(px[1])) * noisy_depth / focal
    camera_xyz = np.asarray([cam_x, cam_y, noisy_depth], dtype=np.float32)
    world_pos = tip.copy()
    if float(config.tip_3d_noise_std_m) > 0.0:
        world_pos = world_pos + rng.normal(0.0, float(config.tip_3d_noise_std_m), size=3).astype(np.float32)

    heading = np.asarray(state.get("heading", [0.0, 0.0, 1.0]), dtype=np.float32).reshape(3)
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm <= 1e-6:
        heading = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        heading = heading / heading_norm
    if float(config.tip_3d_noise_std_m) > 0.0:
        heading = heading + rng.normal(0.0, float(config.tip_3d_noise_std_m) * 25.0, size=3).astype(np.float32)
        heading = heading / max(float(np.linalg.norm(heading)), 1e-6)

    if len(visible_cameras) >= 2:
        base_confidence = 0.88
    else:
        base_confidence = 0.80 if source_camera == "side" else 0.74
    noise_penalty = min(float(config.tip_3d_noise_std_m) / 0.012, 0.22)
    confidence = float(np.clip(base_confidence - noise_penalty, 0.05, 1.0))
    return {
        "estimated_tip_pos_3d": world_pos.astype(float).tolist(),
        "estimated_tip_pos_frame": str(config.tip_3d_frame),
        "estimated_tip_camera_pos_3d": camera_xyz.astype(float).tolist(),
        "estimated_tip_depth_m": float(noisy_depth),
        "estimated_tip_pixel_source": source_camera,
        "estimated_tip_heading_3d": heading.astype(float).tolist(),
        "tip_estimator_visible": True,
        "tip_estimator_confidence": confidence,
        "tip_estimator_latency_steps": int(config.latency_steps),
        "tip_estimator": {
            "method": "synthetic_rgbd_red_tip_visibility_gated_calibrated_pose",
            "source": config.tip_3d_source,
            "source_camera": source_camera,
            "visible_cameras": visible_cameras,
            "source_pixel": px.astype(float).tolist(),
            "frame": str(config.tip_3d_frame),
            "coordinate_output": "tip_pos_plus_declared_noise_after_synthetic_calibration",
            "noise_std_m": float(config.tip_3d_noise_std_m),
            "depth_noise_std_m": float(config.tip_3d_depth_noise_std_m),
            "pixel_noise_std_px": float(config.tip_3d_pixel_noise_std_px),
            "dropout_probability": dropout_probability,
        },
    }


def _select_tip_3d_camera(config: VisualDistanceEstimatorConfig, tip_pixels: dict[str, list[float] | None]) -> str | None:
    preferred = str(config.tip_3d_prefer_camera)
    if preferred in {"side", "top"}:
        return preferred if tip_pixels.get(preferred) is not None else None
    for camera_name in ("top", "side"):
        if tip_pixels.get(camera_name) is not None:
            return camera_name
    return None


def _state_rng(state: dict[str, Any], config: VisualDistanceEstimatorConfig) -> np.random.Generator:
    tip = np.asarray(state.get("tip_pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    payload = "|".join(
        [
            str(int(config.tip_3d_noise_seed)),
            str(state.get("task", "")),
            str(int(state.get("step", 0))),
            ",".join(f"{float(x):.6f}" for x in tip),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
    return np.random.default_rng(seed)


def _estimate_view_from_rendered_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Estimate visible red guidewire head to nearby visible edge in one view."""
    if image is None:
        return None
    bgr = np.asarray(image)
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        return None
    hsv = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2HSV)
    red_mask = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 55) & (hsv[:, :, 2] >= 80)).astype(
        np.uint8
    )
    if int(red_mask.sum()) < 3:
        return None

    components, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    best_idx = -1
    best_area = 0
    h, w = red_mask.shape
    for idx in range(1, components):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area > best_area and area >= 3 and bw <= max(24, w // 5) and bh <= max(24, h // 5):
            best_idx = idx
            best_area = area
    if best_idx < 0:
        return None

    tip_px = np.asarray(centroids[best_idx], dtype=np.float32)
    edge_margin = 6.0
    if (
        tip_px[0] < edge_margin
        or tip_px[0] > float(w - 1) - edge_margin
        or tip_px[1] < edge_margin
        or tip_px[1] > float(h - 1) - edge_margin
    ):
        return None
    gray = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.0), 45, 130)

    red_exclude = cv2.dilate(red_mask, np.ones((7, 7), dtype=np.uint8), iterations=1).astype(bool)
    dark_wire = ((gray < 130) & (hsv[:, :, 1] < 115)).astype(np.uint8)
    dark_exclude = cv2.dilate(dark_wire, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    candidate = edges.astype(bool) & ~red_exclude & ~dark_exclude

    yy, xx = np.indices(candidate.shape)
    local = (np.abs(xx - tip_px[0]) <= 56.0) & (np.abs(yy - tip_px[1]) <= 56.0)
    ys, xs = np.where(candidate & local)
    if len(xs) == 0:
        ys, xs = np.where(edges.astype(bool) & ~red_exclude & local)
    if len(xs) == 0:
        return None

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    dists = np.linalg.norm(pts - tip_px[None, :], axis=1)
    keep = dists >= 2.0
    if not np.any(keep):
        return None
    pts = pts[keep]
    dists = dists[keep]
    idx = int(np.argmin(dists))
    wall_px = pts[idx].astype(np.float32)
    return tip_px.astype(np.float32), wall_px, float(dists[idx])


def _project_world_points_unclipped(env: Any, points: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)

    camera_model = _camera_model(env, width, height)
    cam_pos = camera_model["cam_pos"]
    right = camera_model["right"]
    up = camera_model["up"]
    forward = camera_model["forward"]
    focal = float(camera_model["focal"])

    rel = points - cam_pos[None, :]
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    valid_depth = z > 1e-4
    px = np.column_stack(
        [
            width * 0.5 + focal * (x / np.maximum(z, 1e-4)),
            height * 0.52 - focal * (y / np.maximum(z, 1e-4)),
        ]
    ).astype(np.float32)
    valid_frame = (
        (px[:, 0] >= 0.0)
        & (px[:, 0] < float(width))
        & (px[:, 1] >= 0.0)
        & (px[:, 1] < float(height))
    )
    return px, (valid_depth & valid_frame).astype(bool)


def _camera_model(env: Any, width: int, height: int) -> dict[str, np.ndarray | float]:
    lookat = np.asarray(env.camera.lookat, dtype=np.float32)
    distance = max(float(env.camera.distance), 1e-6)
    az = math.radians(float(env.camera.azimuth))
    el = math.radians(float(env.camera.elevation))
    forward = np.array(
        [
            math.cos(el) * math.sin(az),
            -math.cos(el) * math.cos(az),
            math.sin(el),
        ],
        dtype=np.float32,
    )
    forward = forward / max(float(np.linalg.norm(forward)), 1e-6)
    cam_pos = lookat - forward * distance
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-6)
    return {
        "cam_pos": cam_pos,
        "right": right,
        "up": up,
        "forward": forward,
        "focal": 0.92 * min(width, height),
    }
