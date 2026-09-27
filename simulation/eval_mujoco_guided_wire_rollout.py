from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from simulation.mujoco_guided_wire_env import MuJoCoGuidedWireConfig, MuJoCoGuidedWireEnv
from simulation.tip_guided_wire_env import TipGuidedWireConfig, TipGuidedWireEnv
from simulation.train_dual_arm_baseline import DualArmCameraStatePolicy, build_state_vector
from simulation.visual_distance_estimator import VisualDistanceEstimatorConfig, estimate_visual_distance_contact


def make_sample_from_obs(obs_dict: dict, max_steps: int, estimator_fields: dict | None = None) -> dict:
    estimator_fields = estimator_fields or {}
    return {
        "task": obs_dict["task"],
        "step": obs_dict["step"],
        "state": {
            "tip_pos": obs_dict["tip_pos"],
            "heading": obs_dict["heading"],
            "target_pos": obs_dict["target_pos"],
            "contact_normal": obs_dict["contact_normal"],
            "distance_to_wall": obs_dict["distance_to_wall"],
            "contact_strength": obs_dict["contact_strength"],
            "contact_flag": obs_dict["contact_flag"],
            "path_progress": obs_dict["path_progress"],
            "piper_step": obs_dict["piper_step"],
            "piper_insertion_length": obs_dict["piper_insertion_length"],
            "elirobot_pose": obs_dict["elirobot_pose"],
            "elite_tcp_pose_6d": obs_dict.get("elite_tcp_pose_6d"),
            "lateral_offset": obs_dict["lateral_offset"],
            "path_tangent": obs_dict["path_tangent"],
            "local_radius": obs_dict["local_radius"],
            "robot_state": obs_dict.get("robot_state", {}),
            "controller_state": obs_dict.get("controller_state", {}),
            **estimator_fields,
        },
    }


def image_to_tensor(img_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DualArmCameraStatePolicy(state_dim=int(ckpt["state_dim"]), action_dim=int(ckpt.get("action_dim", 4)), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "image_size": int(ckpt["image_size"]),
        "max_steps": int(ckpt.get("max_steps", 560)),
        "observation_schema": ckpt.get("observation_schema", "full_sim_state"),
        "piper_command_period_for_state": int(ckpt.get("piper_command_period_for_state", 40)),
        "action_dim": int(ckpt.get("action_dim", 4)),
        "action_mode": ckpt.get("action_mode", "legacy"),
        "piper_head": ckpt.get("piper_head", "feed_regression"),
        "piper_step_class_values": list(ckpt.get("piper_step_class_values", [-1, 0, 1])),
        "piper_step_feed_value": float(ckpt.get("piper_step_feed_value", 0.7)),
        "piper_step_retract_value": float(ckpt.get("piper_step_retract_value", 0.7)),
        "elite_action_representation": ckpt.get("elite_action_representation", "absolute"),
        "piper_joint_names": list(ckpt.get("piper_joint_names", [])),
        "elite_joint_names": list(ckpt.get("elite_joint_names", [])),
    }


def _policy_estimator_enabled(args: argparse.Namespace, model_bundle: dict) -> bool:
    mode = str(args.policy_estimator_mode)
    if mode == "off":
        return False
    observation_schema = str(model_bundle.get("observation_schema", ""))
    if mode == "auto":
        return observation_schema in {
            "real_direct_plus_estimated_tip_contact",
            "real_direct_plus_estimated_tip_registered_geometry",
        }
    return True


def _policy_estimator_config(args: argparse.Namespace, model_bundle: dict) -> VisualDistanceEstimatorConfig:
    mode = str(args.policy_estimator_mode)
    observation_schema = str(model_bundle.get("observation_schema", ""))
    tip_3d_enabled = bool(args.policy_estimated_tip_3d)
    if mode in {"auto", "estimated_tip_3d"} and observation_schema in {
        "real_direct_plus_estimated_tip_contact",
        "real_direct_plus_estimated_tip_registered_geometry",
    }:
        tip_3d_enabled = True
    if mode == "estimated_tip_3d":
        tip_3d_enabled = True
    registered_geometry_enabled = bool(args.policy_registered_geometry_estimator) or (
        mode == "auto" and observation_schema == "real_direct_plus_estimated_tip_registered_geometry"
    )
    if registered_geometry_enabled:
        tip_3d_enabled = True
    return VisualDistanceEstimatorConfig(
        contact_threshold_px=float(args.policy_visual_distance_contact_threshold_px),
        confidence_band_px=float(args.policy_visual_distance_confidence_band_px),
        latency_steps=int(args.policy_visual_distance_latency_steps),
        contact_rule=str(args.policy_visual_distance_contact_rule),
        tip_3d_estimator=tip_3d_enabled,
        tip_3d_frame=str(args.policy_estimated_tip_3d_frame),
        tip_3d_prefer_camera=str(args.policy_estimated_tip_3d_prefer_camera),
        tip_3d_noise_std_m=float(args.policy_estimated_tip_3d_noise_std_m),
        tip_3d_depth_noise_std_m=float(args.policy_estimated_tip_3d_depth_noise_std_m),
        tip_3d_pixel_noise_std_px=float(args.policy_estimated_tip_3d_pixel_noise_std_px),
        tip_3d_dropout_probability=float(args.policy_estimated_tip_3d_dropout_probability),
        tip_3d_noise_seed=int(args.policy_estimated_tip_3d_noise_seed),
        registered_geometry_estimator=registered_geometry_enabled,
        registered_geometry_margin_threshold_m=float(args.policy_registered_geometry_margin_threshold_m),
        registered_geometry_wall_pull_threshold_m=float(args.policy_registered_geometry_wall_pull_threshold_m),
    )


def _estimate_policy_observation_fields(
    env: MuJoCoGuidedWireEnv,
    obs_dict: dict,
    images: dict[str, np.ndarray],
    args: argparse.Namespace,
    model_bundle: dict,
) -> dict:
    if not _policy_estimator_enabled(args, model_bundle):
        return {}
    return estimate_visual_distance_contact(
        env,
        obs_dict,
        image_size=int(args.camera_size),
        config=_policy_estimator_config(args, model_bundle),
        images=images,
    )


def _joint_dict_from_vector(names: list[str], vector: np.ndarray) -> dict:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return {name: float(vector[idx]) for idx, name in enumerate(names)}


def _elite_tcp_pose6d_from_obs(obs_dict: dict) -> np.ndarray:
    pose = obs_dict.get("elite_tcp_pose_6d")
    if pose is None:
        pose = obs_dict.get("robot_state", {}).get("elite_tcp_pose_6d")
    if pose is None:
        raise ValueError("tcp_delta Elite rollout requires obs_dict.elite_tcp_pose_6d")
    arr = np.asarray(pose, dtype=np.float32).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"elite_tcp_pose_6d expected 6 values, got {arr.size}")
    return arr.astype(np.float32)


def _guard_piper_feed(env: MuJoCoGuidedWireEnv, piper_vec: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    feed = env._piper_feed_signal(piper_vec)
    guarded_feed = float(np.clip(feed, args.piper_feed_min, args.piper_feed_max))
    return env._piper_joint_vector_from_feed(guarded_feed)


def _scripted_piper_feed(
    env: MuJoCoGuidedWireEnv,
    obs_dict: dict,
    task: str,
    feed_direction: float,
) -> tuple[np.ndarray, float]:
    joint_signal_limit = min(
        float(env.piper_joint_limits.get("joint7", (0.0, 0.035))[1]),
        -float(env.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]),
    )
    insertion_limit = min(float(env.config.piper_insertion_window), max(joint_signal_limit - float(env.piper_feed_home_signal), 1e-6))
    current_insertion = float(obs_dict.get("piper_insertion_length", 0.0))
    contact_strength = float(obs_dict.get("contact_strength", 0.0))
    distance_to_wall = float(obs_dict.get("distance_to_wall", 1.0))
    hard_contact = contact_strength > 0.92 or distance_to_wall < -env.config.wall_margin * 0.20
    soft_contact = distance_to_wall < env.config.wall_margin * 0.20 or contact_strength > 0.80

    if hard_contact or current_insertion >= insertion_limit * 0.93:
        feed_direction = -1.0
    elif current_insertion <= insertion_limit * 0.55:
        feed_direction = 1.0

    if hard_contact:
        piper_cmd = 0.16
    elif soft_contact:
        piper_cmd = 0.30
    elif task == "left":
        piper_cmd = 0.86
    else:
        piper_cmd = 0.90

    feed_step = float(piper_cmd) * float(env.config.advance_step) * float(env.config.piper_advection_scale)
    feed_step = max(feed_step, insertion_limit * 0.12)
    if feed_direction < 0.0:
        feed_step = max(feed_step * 2.2, insertion_limit * 0.24)
    desired_insertion = float(np.clip(current_insertion + feed_direction * feed_step, 0.0, insertion_limit))
    return env._piper_joint_vector_from_feed(float(env.piper_feed_home_signal) + desired_insertion), feed_direction


def _elite_tip_anchor_joint_vector(
    env: MuJoCoGuidedWireEnv,
    elite_joint_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    center, tangent, n1, n2, radius = env._local_path_frame(env.path_progress_float)
    path = env.wire_centerlines[env.task]
    idx = int(np.clip(np.floor(env.path_progress_float), 0, len(path) - 1))
    idx = min(idx + max(int(args.elite_anchor_lookahead_points), 1), len(path) - 1)
    lookahead = path[idx]
    desired_direction = lookahead - env.tip
    if float(np.linalg.norm(desired_direction)) < 1e-6:
        desired_direction = tangent.copy()
    desired_direction = desired_direction / max(float(np.linalg.norm(desired_direction)), 1e-6)

    tip_offset = env.tip - center
    tip_local = np.array([float(np.dot(tip_offset, n1)), float(np.dot(tip_offset, n2))], dtype=np.float32)
    correction = -n1 * tip_local[0] - n2 * tip_local[1]
    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > 1e-6:
        correction = correction / correction_norm * min(radius * float(args.elite_anchor_lateral_gain), radius * 0.55)

    desired_elite = env.tip + desired_direction * float(args.elite_anchor_ahead) + correction
    anchor_joints = env._solve_robot_ik(
        "elite",
        desired_elite,
        max_iters=int(args.elite_anchor_ik_iters),
        damping=float(args.elite_anchor_ik_damping),
        step_limit=float(args.elite_anchor_ik_step_limit),
    )
    anchor_dict = env._robot_joint_dict(
        "elite",
        env._robot_joint_vector_from_action("elite", anchor_joints),
    )
    return np.asarray([float(anchor_dict[name]) for name in elite_joint_names], dtype=np.float32)


def _apply_elite_tip_anchor(
    env: MuJoCoGuidedWireEnv,
    elite_vec: np.ndarray,
    elite_joint_names: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    if not args.elite_tip_anchor:
        return elite_vec
    anchor_vec = _elite_tip_anchor_joint_vector(env, elite_joint_names, args)
    anchored = elite_vec.astype(np.float32)
    if float(args.elite_anchor_blend) > 0.0:
        blend = float(np.clip(args.elite_anchor_blend, 0.0, 1.0))
        anchored = ((1.0 - blend) * anchored + blend * anchor_vec).astype(np.float32)
    if float(args.elite_anchor_joint_limit) > 0.0:
        limit = float(args.elite_anchor_joint_limit)
        anchored = (anchor_vec + np.clip(anchored - anchor_vec, -limit, limit)).astype(np.float32)
    return anchored


def _estimated_tip_coupling_anchor(
    env: MuJoCoGuidedWireEnv,
    obs_dict: dict,
    elite_vec: np.ndarray,
    elite_joint_names: list[str],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    status = {
        "enabled": bool(args.estimated_tip_coupling_guard),
        "active": False,
        "reason": "disabled",
        "distance": None,
    }
    if not args.estimated_tip_coupling_guard:
        return elite_vec, status

    estimated_tip = obs_dict.get("estimated_tip_pos_3d")
    if estimated_tip is None:
        status["reason"] = "missing_estimated_tip"
        return elite_vec, status
    tip_confidence = float(obs_dict.get("tip_estimator_confidence", 0.0) or 0.0)
    if tip_confidence < float(args.estimated_tip_coupling_min_confidence):
        status["reason"] = "low_tip_confidence"
        return elite_vec, status

    tip = np.asarray(estimated_tip, dtype=np.float32).reshape(3)
    current_tcp = _elite_tcp_pose6d_from_obs(obs_dict)
    current_xyz = current_tcp[:3].astype(np.float32) / 1000.0
    distance = float(np.linalg.norm(current_xyz - tip))
    status["distance"] = distance
    if distance <= float(args.estimated_tip_coupling_distance):
        status["reason"] = "within_distance"
        return elite_vec, status

    heading = obs_dict.get("estimated_tip_heading_3d")
    if heading is None:
        heading_vec = np.zeros(3, dtype=np.float32)
    else:
        heading_vec = np.asarray(heading, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(heading_vec))
        heading_vec = heading_vec / max(norm, 1e-6) if norm > 1e-6 else np.zeros(3, dtype=np.float32)

    desired_xyz = tip + heading_vec * float(args.estimated_tip_coupling_ahead)
    desired_pose = current_tcp.copy()
    desired_pose[:3] = desired_xyz.astype(np.float32) * 1000.0
    anchor_vec = env._elite_joint_vector_from_tcp_action({"elite_tcp_pose_6d": desired_pose.astype(float).tolist()})

    anchored = elite_vec.astype(np.float32)
    blend = float(np.clip(args.estimated_tip_coupling_blend, 0.0, 1.0))
    if blend > 0.0:
        anchored = ((1.0 - blend) * anchored + blend * anchor_vec).astype(np.float32)
    if float(args.estimated_tip_coupling_joint_limit) > 0.0:
        limit = float(args.estimated_tip_coupling_joint_limit)
        anchored = (anchor_vec + np.clip(anchored - anchor_vec, -limit, limit)).astype(np.float32)
    status.update(
        {
            "active": True,
            "reason": "distance_exceeded",
            "target_tcp_pose_6d": desired_pose.astype(float).tolist(),
            "tip_estimator_confidence": tip_confidence,
        }
    )
    return anchored, status


def _registered_geometry_wall_safety_guard(
    env: MuJoCoGuidedWireEnv,
    obs_dict: dict,
    elite_vec: np.ndarray,
    elite_joint_names: list[str],
    tcp_action: dict | None,
    piper_step_command: int | None,
    piper_feed: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict | None, int | None, float, dict]:
    status = {
        "enabled": bool(args.registered_geometry_wall_safety_guard),
        "active": False,
        "reason": "disabled",
        "risk": None,
        "margin": None,
        "magnet_wall_pull": None,
        "route_estimator_confidence": None,
        "tip_estimator_confidence": None,
        "piper_held": False,
        "elite_corrected": False,
    }
    if not args.registered_geometry_wall_safety_guard:
        return elite_vec, tcp_action, piper_step_command, piper_feed, status

    route_confidence = float(obs_dict.get("route_estimator_confidence", 0.0) or 0.0)
    tip_confidence = float(obs_dict.get("tip_estimator_confidence", 0.0) or 0.0)
    status["route_estimator_confidence"] = route_confidence
    status["tip_estimator_confidence"] = tip_confidence
    min_confidence = float(args.registered_geometry_wall_min_confidence)
    if route_confidence < min_confidence or tip_confidence < min_confidence:
        status["reason"] = "low_estimator_confidence"
        return elite_vec, tcp_action, piper_step_command, piper_feed, status

    risk = obs_dict.get("estimated_wall_side_risk")
    margin = obs_dict.get("estimated_wall_margin")
    magnet_wall_pull = obs_dict.get("estimated_magnet_wall_pull")
    risk_value = float(risk) if risk is not None else 0.0
    margin_value = float(margin) if margin is not None else float("inf")
    pull_value = float(magnet_wall_pull) if magnet_wall_pull is not None else 0.0
    status.update(
        {
            "risk": risk_value,
            "margin": margin_value,
            "magnet_wall_pull": pull_value,
        }
    )

    risk_threshold = float(args.registered_geometry_wall_risk_threshold)
    margin_threshold = float(args.registered_geometry_wall_margin_threshold_m)
    pull_threshold = float(args.registered_geometry_wall_pull_threshold_m)
    risk_trigger = risk_value >= risk_threshold
    margin_trigger = margin_value <= margin_threshold
    pull_trigger = pull_value >= pull_threshold
    if not (risk_trigger or margin_trigger or pull_trigger):
        status["reason"] = "below_threshold"
        return elite_vec, tcp_action, piper_step_command, piper_feed, status

    reasons = []
    if risk_trigger:
        reasons.append("risk")
    if margin_trigger:
        reasons.append("margin")
    if pull_trigger:
        reasons.append("magnet_wall_pull")
    status["active"] = True
    status["reason"] = "+".join(reasons)

    severity_parts = []
    if risk_trigger:
        denom = max(1.0 - risk_threshold, 1e-6)
        severity_parts.append((risk_value - risk_threshold) / denom)
    if margin_trigger:
        severity_parts.append((margin_threshold - margin_value) / max(margin_threshold, 1e-6))
    if pull_trigger:
        severity_parts.append((pull_value - pull_threshold) / max(abs(pull_threshold), 1e-6))
    severity = float(np.clip(max(severity_parts) if severity_parts else 0.0, 0.0, 1.0))
    status["severity"] = severity

    if bool(args.registered_geometry_wall_hold_piper):
        piper_feed = min(float(piper_feed), 0.0)
        if piper_step_command is not None and int(piper_step_command) > 0:
            piper_step_command = 0
        status["piper_held"] = True

    normal = obs_dict.get("estimated_wall_normal_3d")
    if tcp_action is None or normal is None:
        status["elite_correction_reason"] = "missing_tcp_action_or_wall_normal"
        return elite_vec, tcp_action, piper_step_command, piper_feed, status

    normal_vec = np.asarray(normal, dtype=np.float32).reshape(3)
    normal_norm = float(np.linalg.norm(normal_vec))
    if normal_norm <= 1e-6:
        status["elite_correction_reason"] = "invalid_wall_normal"
        return elite_vec, tcp_action, piper_step_command, piper_feed, status
    normal_vec = normal_vec / normal_norm

    delta = np.asarray(tcp_action.get("elite_tcp_delta_6d", [0.0] * 6), dtype=np.float32).reshape(6).copy()
    original_delta_xyz = delta[:3].astype(np.float32).copy()
    toward_wall_mm = float(np.dot(original_delta_xyz, normal_vec))
    correction = np.zeros(3, dtype=np.float32)
    if bool(args.registered_geometry_wall_remove_toward_normal) and toward_wall_mm > 0.0:
        correction -= normal_vec * toward_wall_mm * float(args.registered_geometry_wall_correction_blend)
    away_delta_mm = float(args.registered_geometry_wall_away_delta_mm) * severity
    if away_delta_mm > 0.0:
        correction -= normal_vec * away_delta_mm
    max_correction_mm = float(args.registered_geometry_wall_max_correction_mm)
    correction_norm = float(np.linalg.norm(correction))
    if max_correction_mm > 0.0 and correction_norm > max_correction_mm:
        correction = correction * (max_correction_mm / max(correction_norm, 1e-6))

    if float(np.linalg.norm(correction)) <= 1e-9:
        status["elite_correction_reason"] = "no_toward_wall_component"
        return elite_vec, tcp_action, piper_step_command, piper_feed, status

    delta[:3] = original_delta_xyz + correction
    current_tcp = _elite_tcp_pose6d_from_obs(obs_dict)
    corrected_pose = current_tcp.copy()
    corrected_pose[:3] = corrected_pose[:3] + delta[:3]
    corrected_tcp_action = {
        "elite_tcp_delta_6d": delta.astype(float).tolist(),
        "elite_tcp_pose_6d": corrected_pose.astype(float).tolist(),
    }
    corrected_elite_vec = env._elite_joint_vector_from_tcp_action(corrected_tcp_action)
    status.update(
        {
            "elite_corrected": True,
            "toward_wall_mm": toward_wall_mm,
            "correction_delta_mm": correction.astype(float).tolist(),
            "original_elite_tcp_delta_6d": tcp_action.get("elite_tcp_delta_6d"),
            "corrected_elite_tcp_delta_6d": corrected_tcp_action["elite_tcp_delta_6d"],
        }
    )
    return corrected_elite_vec, corrected_tcp_action, piper_step_command, piper_feed, status


def _elite_route_plan_anchor_joint_vector(
    env: MuJoCoGuidedWireEnv,
    elite_joint_names: list[str],
    args: argparse.Namespace,
    route_plan_progress: float,
) -> np.ndarray:
    center, tangent, _n1, _n2, _radius = env._local_path_frame(route_plan_progress)
    desired_elite = center + tangent * float(args.elite_route_plan_anchor_ahead)
    anchor_joints = env._solve_robot_ik(
        "elite",
        desired_elite,
        max_iters=int(args.elite_route_plan_anchor_ik_iters),
        damping=float(args.elite_route_plan_anchor_ik_damping),
        step_limit=float(args.elite_route_plan_anchor_ik_step_limit),
    )
    anchor_dict = env._robot_joint_dict(
        "elite",
        env._robot_joint_vector_from_action("elite", anchor_joints),
    )
    return np.asarray([float(anchor_dict[name]) for name in elite_joint_names], dtype=np.float32)


def _apply_elite_route_plan_anchor(
    env: MuJoCoGuidedWireEnv,
    elite_vec: np.ndarray,
    elite_joint_names: list[str],
    args: argparse.Namespace,
    route_plan_progress: float,
) -> np.ndarray:
    if not args.elite_route_plan_anchor:
        return elite_vec
    anchor_vec = _elite_route_plan_anchor_joint_vector(env, elite_joint_names, args, route_plan_progress)
    anchored = elite_vec.astype(np.float32)
    if float(args.elite_route_plan_anchor_blend) > 0.0:
        blend = float(np.clip(args.elite_route_plan_anchor_blend, 0.0, 1.0))
        anchored = ((1.0 - blend) * anchored + blend * anchor_vec).astype(np.float32)
    if float(args.elite_route_plan_anchor_joint_limit) > 0.0:
        limit = float(args.elite_route_plan_anchor_joint_limit)
        anchored = (anchor_vec + np.clip(anchored - anchor_vec, -limit, limit)).astype(np.float32)
    return anchored


def _piper_insertion_limit(env: MuJoCoGuidedWireEnv) -> float:
    joint_signal_limit = min(
        float(env.piper_joint_limits.get("joint7", (0.0, 0.035))[1]),
        -float(env.piper_joint_limits.get("joint8", (-0.035, 0.0))[0]),
    )
    return min(float(env.config.piper_insertion_window), max(joint_signal_limit - float(env.piper_feed_home_signal), 1e-6))


def _guard_scalar_piper_feed(env: MuJoCoGuidedWireEnv, obs_dict: dict, piper_feed: float, phase_direction: float, args: argparse.Namespace) -> tuple[float, float]:
    insertion_limit = _piper_insertion_limit(env)
    current_insertion = float(obs_dict.get("piper_insertion_length", 0.0))
    upper = insertion_limit * float(args.piper_feed_phase_upper)
    lower = insertion_limit * float(args.piper_feed_phase_lower)
    if current_insertion >= upper:
        phase_direction = -1.0
    elif current_insertion <= lower:
        phase_direction = 1.0
    if phase_direction < 0.0:
        piper_feed = min(float(piper_feed), -abs(float(args.piper_feed_phase_retreat)))
    return float(np.clip(piper_feed, -1.0, 1.0)), phase_direction


def _apply_piper_step_controller(
    policy_step_command: int | None,
    piper_feed: float,
    last_executed_step: int,
    step: int,
    args: argparse.Namespace,
) -> tuple[int | None, float, int, bool]:
    if not args.piper_step_controller or policy_step_command is None:
        return policy_step_command, piper_feed, last_executed_step, False
    policy_step_command = int(np.clip(policy_step_command, -1, 1))
    if policy_step_command == 0:
        return 0, 0.0, last_executed_step, False
    cooldown = max(int(args.piper_step_cooldown), 0)
    if step - int(last_executed_step) < cooldown:
        return 0, 0.0, last_executed_step, True
    magnitude = abs(float(piper_feed))
    if args.piper_step_controller_feed_value >= 0.0 and policy_step_command > 0:
        magnitude = float(args.piper_step_controller_feed_value)
    if args.piper_step_controller_retract_value >= 0.0 and policy_step_command < 0:
        magnitude = float(args.piper_step_controller_retract_value)
    executed_feed = float(np.clip(np.sign(policy_step_command) * magnitude, -1.0, 1.0))
    return policy_step_command, executed_feed, int(step), False


def segment_min_distance_to_wall(env: MuJoCoGuidedWireEnv) -> float:
    return env.segment_min_distance_to_wall()


def build_env(args: argparse.Namespace, max_steps: int, seed: int) -> MuJoCoGuidedWireEnv:
    config_cls = TipGuidedWireConfig if args.env_type == "tip" else MuJoCoGuidedWireConfig
    wire_visual_mode = args.wire_visual_mode or ("line" if args.env_type == "tip" else "segments")
    wire_segments = args.wire_segments if args.wire_segments is not None else (160 if args.env_type == "tip" else 34)
    tip_tail_decay_segments = args.tip_tail_decay_segments if args.tip_tail_decay_segments is not None else 18.0
    config_kwargs = dict(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=max_steps,
        wire_segments=wire_segments,
        robot_visual_mode=args.robot_visual_mode,
        wire_visual_mode=wire_visual_mode,
        wire_visual_radius=args.wire_visual_radius,
        wire_visual_rgb=args.wire_visual_rgb,
        wire_tip_visual_rgb=args.wire_tip_visual_rgb,
        wire_tip_visual_segments=args.wire_tip_visual_segments,
        wire_tip_visual_radius_scale=args.wire_tip_visual_radius_scale,
        wire_tip_marker_radius=args.wire_tip_marker_radius,
        wire_tip_marker_alpha=args.wire_tip_marker_alpha,
        wire_visual_offset=tuple(float(x) for x in args.wire_visual_offset),
        wire_line_width=args.wire_line_width,
        wire_line_core_width=args.wire_line_core_width,
        show_vessel_mesh=not args.hide_vessel_mesh,
        show_path_tubes=args.show_path_tubes,
        show_tool_markers=args.show_tool_markers,
        render_overlay=False,
        render_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
        guidance_mode=args.guidance_mode,
        magnet_force=args.magnet_force,
        magnet_range=args.magnet_range,
        piper_force=args.piper_force,
        piper_advection_scale=args.piper_advection_scale,
        piper_primitive_steps=max(int(args.piper_primitive_steps), 1),
        piper_primitive_feed_value=float(args.piper_primitive_feed_value),
        piper_primitive_retract_value=float(args.piper_primitive_retract_value),
        spring_k=args.spring_k,
        bend_k=args.bend_k,
        damping=args.damping,
        sim_substeps=args.sim_substeps,
        wall_k=args.wall_k,
        gravity_z=args.gravity_z,
        wire_max_turn_degrees=args.wire_max_turn_degrees,
        physical_curvature_smoothing=args.physical_curvature_smoothing,
        physical_curvature_iterations=args.physical_curvature_iterations,
        physical_tip_stiff_segments=args.physical_tip_stiff_segments,
        physical_tip_curvature_smoothing=args.physical_tip_curvature_smoothing,
        physical_tip_curvature_iterations=args.physical_tip_curvature_iterations,
        wire_max_speed=args.wire_max_speed,
        elite_motion_smoothing=args.elite_motion_smoothing,
        elite_joint_rate_limit=args.elite_joint_rate_limit,
        elite_joint_accel_limit=args.elite_joint_accel_limit,
        elite_joint_jerk_limit=args.elite_joint_jerk_limit,
    )
    if args.env_type == "tip":
        config_kwargs.update(
            tip_progress_gain=args.tip_progress_gain,
            tip_retract_progress_gain=args.tip_retract_progress_gain,
            tip_magnetic_progress_gain=args.tip_magnetic_progress_gain,
            tip_lateral_magnetic_gain=args.tip_lateral_magnetic_gain,
            tip_lateral_centering_gain=args.tip_lateral_centering_gain,
            tip_contact_relief_gain=args.tip_contact_relief_gain,
            tip_max_lateral_radius_fraction=args.tip_max_lateral_radius_fraction,
            tip_tail_decay_segments=tip_tail_decay_segments,
        )
    config = config_cls(**config_kwargs)
    if args.env_type == "tip":
        return TipGuidedWireEnv(config, seed=seed)
    return MuJoCoGuidedWireEnv(config, seed=seed)


def rollout_validity_audit(args: argparse.Namespace) -> dict:
    interventions: list[str] = []
    real_observable_controllers: list[str] = []
    if args.elite_tip_anchor:
        interventions.append("elite_tip_anchor")
    if args.tactile_safety:
        interventions.append("tactile_safety")
    if args.estimated_tip_coupling_guard:
        interventions.append("estimated_tip_coupling_guard")
    if args.scripted_piper_feed:
        interventions.append("scripted_piper_feed")
    if args.piper_feed_phase_guard:
        interventions.append("piper_feed_phase_guard")
    if args.show_path_tubes:
        interventions.append("show_path_tubes")
    if args.show_tool_markers:
        interventions.append("show_tool_markers")
    if args.registered_geometry_wall_safety_guard:
        real_observable_controllers.append("registered_geometry_wall_safety_guard")

    allowed_for_formal = not interventions
    return {
        "formal_eval": bool(args.formal_eval),
        "route": "route_2_pure_sim_physics",
        "allowed_for_formal_eval": bool(allowed_for_formal),
        "diagnostic_or_oracle_interventions": interventions,
        "real_observable_controllers": real_observable_controllers,
        "note": (
            "Formal evaluation should not use oracle tip anchors, scripted "
            "Piper/control guards, tactile wall/contact feedback, or debug "
            "visual leakage. Joint rate/acceleration/jerk limits and scheduled "
            "route-plan anchors remain allowed when treated as real robot "
            "bandwidth/controller constraints. Registered-geometry wall safety "
            "is allowed only as a controller that consumes estimator/provenance "
            "fields rather than exact MuJoCo contact or wall truth."
        ),
    }


def validate_formal_eval_args(args: argparse.Namespace) -> None:
    audit = rollout_validity_audit(args)
    if args.formal_eval and not audit["allowed_for_formal_eval"]:
        blocked = ", ".join(audit["diagnostic_or_oracle_interventions"])
        raise ValueError(f"--formal-eval rejects diagnostic/oracle interventions: {blocked}")


def run_rollout(args: argparse.Namespace, task: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_bundle = load_model(args.checkpoint, device)
    model = model_bundle["model"]
    image_size = model_bundle["image_size"]
    checkpoint_max_steps = model_bundle["max_steps"]
    piper_joint_names = model_bundle["piper_joint_names"]
    elite_joint_names = model_bundle["elite_joint_names"]
    max_steps = int(args.max_steps or checkpoint_max_steps)
    env = build_env(args, max_steps=max_steps, seed=args.seed)
    _obs, info = env.reset(options={"task": task, "start_fraction": args.start_fraction})

    out_dir = Path(args.out) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        writer = cv2.VideoWriter(
            str(out_dir / "rollout.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.render_width, args.render_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {out_dir / 'rollout.mp4'}")

    states: list[dict] = []
    actions: list[dict] = []
    total_reward = 0.0
    min_tip_wall = float("inf")
    min_segment_wall = float("inf")
    max_contact = 0.0
    scripted_piper_direction = 1.0
    scalar_piper_phase_direction = 1.0
    cached_pred: np.ndarray | None = None
    cached_estimator_fields: dict = {}
    route_plan_progress = float(getattr(env, "path_progress_float", 0.0))
    last_piper_step_executed_at = -10**9

    try:
        while True:
            obs_dict = info["obs_dict"]
            step = int(obs_dict["step"])
            if args.progress_log_every > 0 and step % args.progress_log_every == 0:
                print(
                    f"  {task}: step={step}, progress={float(obs_dict['path_progress']):.2f}, "
                    f"dist={float(obs_dict['distance_to_target']):.4f}, "
                    f"insert={float(obs_dict.get('piper_insertion_length', 0.0)):.6f}, "
                    f"contact={float(obs_dict.get('contact_strength', 0.0)):.3f}",
                    flush=True,
                )
            if writer is not None and obs_dict["step"] % max(args.render_every, 1) == 0:
                frame = env.render_camera(
                    args.video_camera,
                    hide_robot_visuals=args.diagnostic_video_hide_robots,
                    diagnostic_overlay=args.diagnostic_video_overlay,
                )
                if frame.shape[:2] != (args.render_height, args.render_width):
                    frame = cv2.resize(frame, (args.render_width, args.render_height), interpolation=cv2.INTER_AREA)
                writer.write(frame)

            if cached_pred is None or step % max(args.policy_every, 1) == 0:
                images = env.render_camera_pair(size=args.camera_size)
                estimator_fields = _estimate_policy_observation_fields(env, obs_dict, images, args, model_bundle)
                cached_estimator_fields = estimator_fields
                side = image_to_tensor(images["side"], image_size, device)
                top = image_to_tensor(images["top"], image_size, device)
                state_vec = torch.tensor(
                    build_state_vector(
                        make_sample_from_obs(obs_dict, checkpoint_max_steps, estimator_fields=estimator_fields),
                        checkpoint_max_steps,
                        piper_joint_names,
                        elite_joint_names,
                        model_bundle.get("observation_schema", "full_sim_state"),
                        int(model_bundle.get("piper_command_period_for_state", 40)),
                    ),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    cached_pred = model(side, top, state_vec).squeeze(0).detach().cpu().numpy().astype(np.float32)
            if cached_estimator_fields:
                obs_dict = {**obs_dict, **cached_estimator_fields}
            pred = cached_pred

            if model_bundle["action_mode"] == "joint":
                piper_vec = pred[: len(piper_joint_names)]
                elite_vec = pred[len(piper_joint_names) : len(piper_joint_names) + len(elite_joint_names)]
                if args.scripted_piper_feed:
                    piper_vec, scripted_piper_direction = _scripted_piper_feed(env, obs_dict, task, scripted_piper_direction)
                elif args.piper_feed_guard:
                    piper_vec = _guard_piper_feed(env, piper_vec, args)
                if args.tactile_safety:
                    contact_strength = float(obs_dict.get("contact_strength", 0.0))
                    distance_to_wall = float(obs_dict.get("distance_to_wall", 1.0))
                    if contact_strength >= args.retreat_contact or contact_strength >= args.hold_contact or distance_to_wall <= args.hold_wall:
                        current = obs_dict.get("robot_state", {}).get("piper_joints", {})
                        piper_vec = np.asarray([float(current[name]) for name in piper_joint_names], dtype=np.float32)
            elif model_bundle["action_mode"] == "piper_feed_elite_joint":
                piper_step_command = None
                policy_piper_step_command = None
                piper_controller_delayed = False
                estimated_tip_coupling_status = {"enabled": bool(args.estimated_tip_coupling_guard), "active": False, "reason": "not_applicable"}
                registered_geometry_wall_safety_status = {
                    "enabled": bool(args.registered_geometry_wall_safety_guard),
                    "active": False,
                    "reason": "not_applicable",
                }
                if model_bundle.get("piper_head", "feed_regression") == "step_classification":
                    class_values = list(model_bundle.get("piper_step_class_values", [-1, 0, 1]))
                    piper_logits = pred[: len(class_values)]
                    class_idx = int(np.argmax(piper_logits))
                    piper_step_command = int(class_values[class_idx])
                    policy_piper_step_command = piper_step_command
                    if piper_step_command > 0:
                        piper_feed = float(model_bundle.get("piper_step_feed_value", 0.7))
                    elif piper_step_command < 0:
                        piper_feed = -float(model_bundle.get("piper_step_retract_value", 0.7))
                    else:
                        piper_feed = 0.0
                    elite_start = len(class_values)
                else:
                    piper_feed = float(np.clip(pred[0], -1.0, 1.0))
                    elite_start = 1
                elite_action_representation = model_bundle.get("elite_action_representation", "absolute")
                tcp_action: dict | None = None
                if elite_action_representation == "tcp_delta":
                    elite_tcp_delta = pred[elite_start : elite_start + 6].astype(np.float32)
                    # Match the inherited real inference path: apply translation
                    # deltas and keep the current TCP orientation for IK.
                    elite_tcp_delta = elite_tcp_delta.copy()
                    elite_tcp_delta[3:] = 0.0
                    current_tcp = _elite_tcp_pose6d_from_obs(obs_dict)
                    elite_tcp_pose = current_tcp.copy()
                    elite_tcp_pose[:3] = elite_tcp_pose[:3] + elite_tcp_delta[:3]
                    tcp_action = {
                        "elite_tcp_delta_6d": elite_tcp_delta.astype(float).tolist(),
                        "elite_tcp_pose_6d": elite_tcp_pose.astype(float).tolist(),
                    }
                    elite_vec = env._elite_joint_vector_from_tcp_action(tcp_action)
                else:
                    elite_vec = pred[elite_start : elite_start + len(elite_joint_names)]
                if elite_action_representation == "delta":
                    current_elite = obs_dict.get("robot_state", {}).get("elite_joints", {})
                    current_vec = np.asarray([float(current_elite[name]) for name in elite_joint_names], dtype=np.float32)
                    elite_vec = current_vec + elite_vec.astype(np.float32)
                elite_vec = _apply_elite_tip_anchor(env, elite_vec, elite_joint_names, args)
                if args.elite_route_plan_anchor:
                    route_plan_progress = float(
                        np.clip(
                            route_plan_progress + float(args.elite_route_plan_anchor_step),
                            0.0,
                            len(env.wire_centerlines[env.task]) - 1,
                        )
                    )
                    elite_vec = _apply_elite_route_plan_anchor(env, elite_vec, elite_joint_names, args, route_plan_progress)
                elite_vec, estimated_tip_coupling_status = _estimated_tip_coupling_anchor(
                    env,
                    obs_dict,
                    elite_vec,
                    elite_joint_names,
                    args,
                )
                if estimated_tip_coupling_status.get("active"):
                    piper_feed = 0.0
                    if piper_step_command is not None:
                        piper_step_command = 0
                elite_vec, tcp_action, piper_step_command, piper_feed, registered_geometry_wall_safety_status = (
                    _registered_geometry_wall_safety_guard(
                        env,
                        obs_dict,
                        elite_vec,
                        elite_joint_names,
                        tcp_action,
                        piper_step_command,
                        piper_feed,
                        args,
                    )
                )
                if args.piper_feed_phase_guard:
                    piper_feed, scalar_piper_phase_direction = _guard_scalar_piper_feed(
                        env,
                        obs_dict,
                        piper_feed,
                        scalar_piper_phase_direction,
                        args,
                    )
                if args.tactile_safety:
                    contact_strength = float(obs_dict.get("contact_strength", 0.0))
                    distance_to_wall = float(obs_dict.get("distance_to_wall", 1.0))
                    if contact_strength >= args.retreat_contact:
                        piper_feed = min(piper_feed, args.retreat_piper)
                    elif contact_strength >= args.hold_contact or distance_to_wall <= args.hold_wall:
                        piper_feed = min(piper_feed, args.hold_piper)

                piper_step_command, piper_feed, last_piper_step_executed_at, piper_controller_delayed = _apply_piper_step_controller(
                    piper_step_command,
                    piper_feed,
                    last_piper_step_executed_at,
                    step,
                    args,
                )
                action = {
                    "piper_feed": piper_feed,
                    "elite_joints": _joint_dict_from_vector(elite_joint_names, elite_vec),
                }
                if tcp_action is not None and not (args.elite_tip_anchor or args.elite_route_plan_anchor):
                    action.update(tcp_action)
                if piper_step_command is not None:
                    action["piper_step_command"] = piper_step_command
                    action["piper_command_label"] = {1: "feed", 0: "hold", -1: "retract"}[int(piper_step_command)]
                if policy_piper_step_command is not None:
                    action["policy_piper_step_command"] = int(policy_piper_step_command)
                    action["policy_piper_command_label"] = {1: "feed", 0: "hold", -1: "retract"}[int(policy_piper_step_command)]
                    action["piper_step_controller_enabled"] = bool(args.piper_step_controller)
                    action["piper_step_controller_delayed"] = bool(piper_controller_delayed)
                if args.estimated_tip_coupling_guard:
                    action["estimated_tip_coupling_guard"] = estimated_tip_coupling_status
                if args.registered_geometry_wall_safety_guard:
                    action["registered_geometry_wall_safety_guard"] = registered_geometry_wall_safety_status
                _obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step_obs = info["obs_dict"]
                min_tip_wall = min(min_tip_wall, float(step_obs["distance_to_wall"]))
                min_segment_wall = min(min_segment_wall, segment_min_distance_to_wall(env))
                max_contact = max(max_contact, float(step_obs["contact_strength"]))
                states.append(obs_dict)
                actions.append(
                    {
                        "step": int(obs_dict["step"]),
                        "action": action,
                        "reward": float(reward),
                    }
                )
                if terminated or truncated:
                    final_obs = info["obs_dict"]
                    if writer is not None:
                        for _ in range(max(args.fps // 2, 1)):
                            writer.write(
                                env.render_camera(
                                    args.video_camera,
                                    hide_robot_visuals=args.diagnostic_video_hide_robots,
                                    diagnostic_overlay=args.diagnostic_video_overlay,
                                )
                            )
                    states.append(final_obs)
                    break
                continue
            else:
                piper = float(np.clip(pred[0], -1.0, 1.0))
                delta = np.tanh(pred[1:4]).astype(np.float32)
                if args.tactile_safety:
                    contact_strength = float(obs_dict.get("contact_strength", 0.0))
                    distance_to_wall = float(obs_dict.get("distance_to_wall", 1.0))
                    if contact_strength >= args.retreat_contact:
                        piper = min(piper, args.retreat_piper)
                    elif contact_strength >= args.hold_contact or distance_to_wall <= args.hold_wall:
                        piper = min(piper, args.hold_piper)
                action = {"piper": np.array([piper], dtype=np.float32), "elirobot_delta": delta}
                _obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                step_obs = info["obs_dict"]
                min_tip_wall = min(min_tip_wall, float(step_obs["distance_to_wall"]))
                min_segment_wall = min(min_segment_wall, segment_min_distance_to_wall(env))
                max_contact = max(max_contact, float(step_obs["contact_strength"]))
                states.append(obs_dict)
                actions.append(
                    {
                        "step": int(obs_dict["step"]),
                        "action": {"piper": piper, "elirobot_delta": delta.astype(float).tolist()},
                        "reward": float(reward),
                    }
                )
                if terminated or truncated:
                    final_obs = info["obs_dict"]
                    if writer is not None:
                        for _ in range(max(args.fps // 2, 1)):
                            writer.write(
                                env.render_camera(
                                    args.video_camera,
                                    hide_robot_visuals=args.diagnostic_video_hide_robots,
                                    diagnostic_overlay=args.diagnostic_video_overlay,
                                )
                            )
                    states.append(final_obs)
                    break
                continue

            action = {
                "piper_joints": _joint_dict_from_vector(piper_joint_names, piper_vec),
                "elite_joints": _joint_dict_from_vector(elite_joint_names, elite_vec),
            }
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            step_obs = info["obs_dict"]
            min_tip_wall = min(min_tip_wall, float(step_obs["distance_to_wall"]))
            min_segment_wall = min(min_segment_wall, segment_min_distance_to_wall(env))
            max_contact = max(max_contact, float(step_obs["contact_strength"]))
            states.append(obs_dict)
            actions.append(
                {
                    "step": int(obs_dict["step"]),
                    "action": action,
                    "reward": float(reward),
                }
            )

            if terminated or truncated:
                final_obs = info["obs_dict"]
                if writer is not None:
                    for _ in range(max(args.fps // 2, 1)):
                        writer.write(
                            env.render_camera(
                                args.video_camera,
                                hide_robot_visuals=args.diagnostic_video_hide_robots,
                                diagnostic_overlay=args.diagnostic_video_overlay,
                            )
                        )
                states.append(final_obs)
                break
    finally:
        if writer is not None:
            writer.release()
        env.close()

    final_obs = info["obs_dict"]
    meta = {
        "task": task,
        "checkpoint": args.checkpoint,
        "env_type": args.env_type,
        "guidance_mode": args.guidance_mode,
        "success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "steps": int(final_obs["step"]),
        "distance_to_target": float(final_obs["distance_to_target"]),
        "path_progress": float(final_obs["path_progress"]),
        "max_contact_strength": max_contact,
        "min_tip_distance_to_wall": min_tip_wall,
        "min_segment_distance_to_wall": min_segment_wall,
        "boundary_projection_count": int(final_obs["boundary_projection_count"]),
        "total_reward": total_reward,
        "elite_motion_smoothing": float(args.elite_motion_smoothing),
        "elite_joint_rate_limit": float(args.elite_joint_rate_limit),
        "elite_joint_accel_limit": float(args.elite_joint_accel_limit),
        "elite_joint_jerk_limit": float(args.elite_joint_jerk_limit),
        "elite_tip_anchor": bool(args.elite_tip_anchor),
        "elite_anchor_joint_limit": float(args.elite_anchor_joint_limit),
        "elite_anchor_blend": float(args.elite_anchor_blend),
        "elite_anchor_lookahead_points": int(args.elite_anchor_lookahead_points),
        "elite_anchor_ahead": float(args.elite_anchor_ahead),
        "elite_anchor_lateral_gain": float(args.elite_anchor_lateral_gain),
        "elite_route_plan_anchor": bool(args.elite_route_plan_anchor),
        "elite_route_plan_anchor_step": float(args.elite_route_plan_anchor_step),
        "elite_route_plan_anchor_ahead": float(args.elite_route_plan_anchor_ahead),
        "elite_route_plan_anchor_joint_limit": float(args.elite_route_plan_anchor_joint_limit),
        "elite_route_plan_anchor_blend": float(args.elite_route_plan_anchor_blend),
        "policy_estimator_mode": str(args.policy_estimator_mode),
        "policy_estimator_enabled": bool(_policy_estimator_enabled(args, model_bundle)),
        "policy_estimator_config": (
            _policy_estimator_config(args, model_bundle).__dict__ if _policy_estimator_enabled(args, model_bundle) else None
        ),
        "piper_step_controller": bool(args.piper_step_controller),
        "piper_step_cooldown": int(args.piper_step_cooldown),
        "piper_step_controller_feed_value": float(args.piper_step_controller_feed_value),
        "piper_step_controller_retract_value": float(args.piper_step_controller_retract_value),
        "piper_primitive_steps": int(max(args.piper_primitive_steps, 1)),
        "piper_primitive_feed_value": float(args.piper_primitive_feed_value),
        "piper_primitive_retract_value": float(args.piper_primitive_retract_value),
        "estimated_tip_coupling_guard": bool(args.estimated_tip_coupling_guard),
        "estimated_tip_coupling_distance": float(args.estimated_tip_coupling_distance),
        "estimated_tip_coupling_min_confidence": float(args.estimated_tip_coupling_min_confidence),
        "estimated_tip_coupling_ahead": float(args.estimated_tip_coupling_ahead),
        "estimated_tip_coupling_joint_limit": float(args.estimated_tip_coupling_joint_limit),
        "estimated_tip_coupling_blend": float(args.estimated_tip_coupling_blend),
        "registered_geometry_wall_safety_guard": bool(args.registered_geometry_wall_safety_guard),
        "registered_geometry_wall_risk_threshold": float(args.registered_geometry_wall_risk_threshold),
        "registered_geometry_wall_margin_threshold_m": float(args.registered_geometry_wall_margin_threshold_m),
        "registered_geometry_wall_pull_threshold_m": float(args.registered_geometry_wall_pull_threshold_m),
        "registered_geometry_wall_min_confidence": float(args.registered_geometry_wall_min_confidence),
        "registered_geometry_wall_hold_piper": bool(args.registered_geometry_wall_hold_piper),
        "registered_geometry_wall_remove_toward_normal": bool(args.registered_geometry_wall_remove_toward_normal),
        "registered_geometry_wall_correction_blend": float(args.registered_geometry_wall_correction_blend),
        "registered_geometry_wall_away_delta_mm": float(args.registered_geometry_wall_away_delta_mm),
        "registered_geometry_wall_max_correction_mm": float(args.registered_geometry_wall_max_correction_mm),
        "validity_audit": rollout_validity_audit(args),
        "diagnostic_video_hide_robots": bool(args.diagnostic_video_hide_robots),
        "diagnostic_video_overlay": bool(args.diagnostic_video_overlay),
        "video": str((out_dir / "rollout.mp4").resolve()) if args.video else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as f:
        for state in states:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")
    with (out_dir / "actions.jsonl").open("w", encoding="utf-8") as f:
        for action in actions:
            f.write(json.dumps(action, ensure_ascii=False) + "\n")

    print(
        f"{task}: success={meta['success']} steps={meta['steps']} failure={meta['failure_reason']} "
        f"dist={meta['distance_to_target']:.4f} max_contact={meta['max_contact_strength']:.3f} "
        f"min_seg_wall={meta['min_segment_distance_to_wall']:.6f} projections={meta['boundary_projection_count']}"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dual-camera/state baseline in the MuJoCo guided-wire environment.")
    parser.add_argument("--checkpoint", default="simulation_output/baseline_mujoco_guided_wire_final_v1/best_model.pt")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/baseline_mujoco_guided_wire_rollout")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--env-type", choices=["polyline", "tip"], default="polyline")
    parser.add_argument(
        "--formal-eval",
        action="store_true",
        help="Reject diagnostic/oracle rollout interventions that are not valid for formal sim-to-real evaluation.",
    )
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--start-fraction", type=float, default=0.46)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wire-segments", type=int, default=None)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--wire-visual-mode", choices=["segments", "line", "both"], default=None)
    parser.add_argument("--wire-visual-radius", type=float, default=0.0025)
    parser.add_argument("--wire-visual-rgb", default="0.06 0.56 1.0")
    parser.add_argument("--wire-tip-visual-rgb", default="0.06 0.56 1.0")
    parser.add_argument("--wire-tip-visual-segments", type=int, default=0)
    parser.add_argument("--wire-tip-visual-radius-scale", type=float, default=1.0)
    parser.add_argument("--wire-tip-marker-radius", type=float, default=0.0)
    parser.add_argument("--wire-tip-marker-alpha", type=float, default=0.0)
    parser.add_argument("--wire-visual-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--wire-line-width", type=int, default=4)
    parser.add_argument("--wire-line-core-width", type=int, default=1)
    parser.add_argument("--guidance-mode", choices=["mvp_autopilot", "physical"], default="mvp_autopilot")
    parser.add_argument("--show-path-tubes", action="store_true")
    parser.add_argument("--show-tool-markers", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    parser.add_argument("--magnet-force", type=float, default=0.36)
    parser.add_argument("--magnet-range", type=float, default=0.17)
    parser.add_argument("--piper-force", type=float, default=0.17)
    parser.add_argument("--piper-advection-scale", type=float, default=0.11)
    parser.add_argument("--piper-primitive-steps", type=int, default=1)
    parser.add_argument("--piper-primitive-feed-value", type=float, default=0.70)
    parser.add_argument("--piper-primitive-retract-value", type=float, default=0.70)
    parser.add_argument("--spring-k", type=float, default=760.0)
    parser.add_argument("--bend-k", type=float, default=165.0)
    parser.add_argument("--damping", type=float, default=6.2)
    parser.add_argument("--sim-substeps", type=int, default=24)
    parser.add_argument("--wall-k", type=float, default=650.0)
    parser.add_argument("--gravity-z", type=float, default=-9.81)
    parser.add_argument("--wire-max-turn-degrees", type=float, default=24.0)
    parser.add_argument("--physical-curvature-smoothing", type=float, default=0.78)
    parser.add_argument("--physical-curvature-iterations", type=int, default=12)
    parser.add_argument("--physical-tip-stiff-segments", type=int, default=7)
    parser.add_argument("--physical-tip-curvature-smoothing", type=float, default=0.82)
    parser.add_argument("--physical-tip-curvature-iterations", type=int, default=4)
    parser.add_argument("--wire-max-speed", type=float, default=0.26)
    parser.add_argument("--elite-motion-smoothing", type=float, default=0.52)
    parser.add_argument("--elite-joint-rate-limit", type=float, default=0.0)
    parser.add_argument("--elite-joint-accel-limit", type=float, default=0.0)
    parser.add_argument("--elite-joint-jerk-limit", type=float, default=0.0)
    parser.add_argument("--elite-tip-anchor", action="store_true", help="Constrain Elite target near an expert-style tip-relative anchor before stepping the environment.")
    parser.add_argument("--elite-anchor-joint-limit", type=float, default=0.040, help="Max absolute joint deviation from the tip-relative anchor when --elite-tip-anchor is enabled; <=0 disables clipping.")
    parser.add_argument("--elite-anchor-blend", type=float, default=0.0, help="Blend model target toward the tip-relative anchor before clipping; 0 keeps model target within the anchor band.")
    parser.add_argument("--elite-anchor-lookahead-points", type=int, default=8)
    parser.add_argument("--elite-anchor-ahead", type=float, default=0.018)
    parser.add_argument("--elite-anchor-lateral-gain", type=float, default=0.75)
    parser.add_argument("--elite-anchor-ik-iters", type=int, default=7)
    parser.add_argument("--elite-anchor-ik-damping", type=float, default=0.04)
    parser.add_argument("--elite-anchor-ik-step-limit", type=float, default=0.20)
    parser.add_argument(
        "--elite-route-plan-anchor",
        action="store_true",
        help="Constrain Elite target near a scheduled route-plan IK anchor; uses no current tip/contact/wall oracle state.",
    )
    parser.add_argument("--elite-route-plan-anchor-step", type=float, default=0.20)
    parser.add_argument("--elite-route-plan-anchor-ahead", type=float, default=0.010)
    parser.add_argument("--elite-route-plan-anchor-joint-limit", type=float, default=0.020)
    parser.add_argument("--elite-route-plan-anchor-blend", type=float, default=0.0)
    parser.add_argument("--elite-route-plan-anchor-ik-iters", type=int, default=7)
    parser.add_argument("--elite-route-plan-anchor-ik-damping", type=float, default=0.04)
    parser.add_argument("--elite-route-plan-anchor-ik-step-limit", type=float, default=0.20)
    parser.add_argument("--tip-progress-gain", type=float, default=0.45)
    parser.add_argument("--tip-retract-progress-gain", type=float, default=0.015)
    parser.add_argument("--tip-magnetic-progress-gain", type=float, default=0.020)
    parser.add_argument("--tip-lateral-magnetic-gain", type=float, default=0.16)
    parser.add_argument("--tip-lateral-centering-gain", type=float, default=0.06)
    parser.add_argument("--tip-contact-relief-gain", type=float, default=0.42)
    parser.add_argument("--tip-max-lateral-radius-fraction", type=float, default=0.82)
    parser.add_argument("--tip-tail-decay-segments", type=float, default=None)
    parser.add_argument("--delta-scale", type=float, default=1.0)
    parser.add_argument("--tactile-safety", action="store_true")
    parser.add_argument("--hold-contact", type=float, default=0.35)
    parser.add_argument("--retreat-contact", type=float, default=0.70)
    parser.add_argument("--hold-wall", type=float, default=0.0004)
    parser.add_argument("--hold-piper", type=float, default=0.12)
    parser.add_argument("--retreat-piper", type=float, default=0.0)
    parser.add_argument("--piper-feed-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--piper-feed-min", type=float, default=0.0032)
    parser.add_argument("--piper-feed-max", type=float, default=0.0062)
    parser.add_argument("--scripted-piper-feed", action="store_true")
    parser.add_argument("--piper-feed-phase-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--piper-feed-phase-upper", type=float, default=0.88)
    parser.add_argument("--piper-feed-phase-lower", type=float, default=0.52)
    parser.add_argument("--piper-feed-phase-retreat", type=float, default=1.0)
    parser.add_argument(
        "--piper-step-controller",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For Piper step-classification checkpoints, treat the policy output as hold/feed/retract intent "
            "and let a real-style step controller execute at most one Piper primitive per cooldown window."
        ),
    )
    parser.add_argument("--piper-step-cooldown", type=int, default=5)
    parser.add_argument(
        "--piper-step-controller-feed-value",
        type=float,
        default=-1.0,
        help="Executed feed magnitude for --piper-step-controller; negative keeps the checkpoint's feed value.",
    )
    parser.add_argument(
        "--piper-step-controller-retract-value",
        type=float,
        default=-1.0,
        help="Executed retract magnitude for --piper-step-controller; negative keeps the checkpoint's retract value.",
    )
    parser.add_argument(
        "--estimated-tip-coupling-guard",
        action="store_true",
        help=(
            "Use estimator-visible guidewire tip position and current Elite TCP pose as a real-style controller "
            "guard: hold Piper and bias Elite back near the estimated tip when they drift apart."
        ),
    )
    parser.add_argument("--estimated-tip-coupling-distance", type=float, default=0.035)
    parser.add_argument("--estimated-tip-coupling-min-confidence", type=float, default=0.35)
    parser.add_argument("--estimated-tip-coupling-ahead", type=float, default=0.006)
    parser.add_argument("--estimated-tip-coupling-joint-limit", type=float, default=0.020)
    parser.add_argument("--estimated-tip-coupling-blend", type=float, default=0.35)
    parser.add_argument(
        "--registered-geometry-wall-safety-guard",
        action="store_true",
        help=(
            "Use estimated tip + registered-route geometry fields as a real-observable controller guard: "
            "hold Piper and correct Elite TCP delta away from the estimated wall side when wall risk is high."
        ),
    )
    parser.add_argument("--registered-geometry-wall-risk-threshold", type=float, default=0.85)
    parser.add_argument("--registered-geometry-wall-margin-threshold-m", type=float, default=0.0)
    parser.add_argument("--registered-geometry-wall-pull-threshold-m", type=float, default=0.008)
    parser.add_argument("--registered-geometry-wall-min-confidence", type=float, default=0.35)
    parser.add_argument("--registered-geometry-wall-hold-piper", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--registered-geometry-wall-remove-toward-normal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--registered-geometry-wall-correction-blend", type=float, default=1.0)
    parser.add_argument("--registered-geometry-wall-away-delta-mm", type=float, default=0.0)
    parser.add_argument("--registered-geometry-wall-max-correction-mm", type=float, default=0.6)
    parser.add_argument(
        "--policy-estimator-mode",
        choices=["auto", "off", "visual_distance", "estimated_tip_3d"],
        default="auto",
        help=(
            "Estimator fields supplied to the policy state. auto enables visual distance + estimated 3D tip "
            "for real_direct_plus_estimated_tip_contact checkpoints, and also enables registered geometry "
            "for real_direct_plus_estimated_tip_registered_geometry checkpoints."
        ),
    )
    parser.add_argument("--policy-visual-distance-contact-threshold-px", type=float, default=1.0)
    parser.add_argument("--policy-visual-distance-confidence-band-px", type=float, default=8.0)
    parser.add_argument("--policy-visual-distance-latency-steps", type=int, default=0)
    parser.add_argument(
        "--policy-visual-distance-contact-rule",
        choices=["all_visible", "any_visible", "median"],
        default="all_visible",
    )
    parser.add_argument("--policy-estimated-tip-3d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy-estimated-tip-3d-frame", choices=["world"], default="world")
    parser.add_argument("--policy-estimated-tip-3d-prefer-camera", choices=["auto", "side", "top"], default="auto")
    parser.add_argument("--policy-estimated-tip-3d-noise-std-m", type=float, default=0.0015)
    parser.add_argument("--policy-estimated-tip-3d-depth-noise-std-m", type=float, default=0.0008)
    parser.add_argument("--policy-estimated-tip-3d-pixel-noise-std-px", type=float, default=0.75)
    parser.add_argument("--policy-estimated-tip-3d-dropout-probability", type=float, default=0.0)
    parser.add_argument("--policy-estimated-tip-3d-noise-seed", type=int, default=20260628)
    parser.add_argument("--policy-registered-geometry-estimator", action="store_true")
    parser.add_argument("--policy-registered-geometry-margin-threshold-m", type=float, default=0.0012)
    parser.add_argument("--policy-registered-geometry-wall-pull-threshold-m", type=float, default=0.0025)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--policy-every", type=int, default=1, help="Run image/state policy every N env steps and reuse the previous action between calls.")
    parser.add_argument("--progress-log-every", type=int, default=0, help="Print rollout progress every N steps; 0 disables it.")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-camera", choices=["overview", "top", "side", "perspective"], default="overview")
    parser.add_argument("--diagnostic-video-hide-robots", action="store_true", help="Hide robot visual geoms only in saved rollout video frames; policy camera inputs remain unchanged.")
    parser.add_argument("--diagnostic-video-overlay", action="store_true", help="Draw guidewire, tip, magnetic point, and Elite tool markers only on saved rollout video frames.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()
    validate_formal_eval_args(args)

    summaries = [run_rollout(args, task) for task in args.tasks]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
