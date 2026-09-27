from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco_guided_wire_env import MuJoCoRoutePlanGuideExpert  # noqa: E402
from simulation.tip_guided_wire_env import TipGuidedWireConfig, TipGuidedWireEnv  # noqa: E402
from simulation.visual_distance_estimator import (  # noqa: E402
    VisualDistanceEstimatorConfig,
    estimate_visual_distance_contact,
    observation_provenance as visual_distance_observation_provenance,
)


ACTION_MODE = "piper_feed_elite_joint"
SCENARIO = "formal_tip_line_guidance"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_env(args: argparse.Namespace, seed: int) -> TipGuidedWireEnv:
    config = TipGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode="kinematic",
        wire_visual_mode="line",
        wire_visual_radius=args.wire_visual_radius,
        wire_visual_rgb=args.wire_visual_rgb,
        wire_tip_visual_rgb=args.wire_tip_visual_rgb,
        wire_tip_visual_segments=args.wire_tip_visual_segments,
        wire_tip_visual_radius_scale=args.wire_tip_visual_radius_scale,
        wire_tip_marker_radius=args.wire_tip_marker_radius,
        wire_tip_marker_alpha=args.wire_tip_marker_alpha,
        wire_visual_offset=tuple(float(x) for x in args.wire_visual_offset),
        wire_line_width=4,
        wire_line_core_width=1,
        show_vessel_mesh=True,
        show_path_tubes=False,
        show_tool_markers=False,
        render_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
        render_domain_preset=args.render_domain_preset,
        guidance_mode="physical",
        piper_advection_scale=0.11,
        piper_primitive_steps=max(int(getattr(args, "piper_primitive_steps", 1)), 1),
        piper_primitive_feed_value=float(getattr(args, "piper_primitive_feed_value", 0.70)),
        piper_primitive_retract_value=float(getattr(args, "piper_primitive_retract_value", 0.70)),
        elite_motion_smoothing=0.55,
        elite_joint_rate_limit=0.0,
        elite_joint_accel_limit=0.0,
        tip_progress_gain=0.45,
        tip_retract_progress_gain=0.015,
        tip_magnetic_progress_gain=0.020,
        tip_lateral_magnetic_gain=0.16,
        tip_lateral_centering_gain=0.06,
        tip_contact_relief_gain=0.42,
        tip_max_lateral_radius_fraction=0.82,
        tip_tail_decay_segments=18.0,
    )
    return TipGuidedWireEnv(config, seed=seed)


def build_expert(args: argparse.Namespace) -> MuJoCoRoutePlanGuideExpert:
    piper_primitive_steps = max(int(getattr(args, "piper_primitive_steps", 1)), 1)
    return MuJoCoRoutePlanGuideExpert(
        plan_step=float(args.route_plan_step),
        elite_ahead=float(args.elite_ahead),
        piper_cmd=float(args.piper_cmd),
        piper_command_period=int(args.piper_command_period),
        piper_command_width=int(args.piper_command_width),
        route_plan_command_phase_lock=True,
        piper_command_as_event=bool(getattr(args, "piper_command_as_event", False) or piper_primitive_steps > 1),
    )


def action_record(env: TipGuidedWireEnv, action: dict) -> dict:
    elite_joints = {name: float(value) for name, value in action["elite_joints"].items()}
    current_elite_tcp = env.robot_tool_pose6d("elite").astype(np.float32)
    target_elite_vec = env._robot_joint_vector_from_action("elite", elite_joints)
    target_elite_fk = env._robot_tool_pose6d_from_vector("elite", target_elite_vec).astype(np.float32)
    target_elite_tcp = current_elite_tcp.copy()
    target_elite_tcp[:3] = target_elite_fk[:3]
    elite_tcp_delta = target_elite_tcp - current_elite_tcp

    piper_vec = env._robot_joint_vector_from_action("piper", action["piper_joints"])
    target_signal = env._piper_feed_signal(piper_vec)
    current_signal = float(getattr(env, "piper_feed_signal", env._piper_feed_signal(env.piper_home_joint_vector)))
    feed_unit = max(float(env.config.advance_step) * float(env.config.piper_advection_scale), 1e-8)
    piper_sim_feed = float(np.clip((target_signal - current_signal) / feed_unit, -1.0, 1.0))
    piper_step_command = int(np.clip(np.sign(float(action.get("piper_step_command", 0))), -1, 1))
    piper_feed = float(np.clip(float(action.get("piper_feed_command", piper_step_command)), -1.0, 1.0))
    piper_command_label = {-1: "retract", 0: "hold", 1: "feed"}[piper_step_command]

    record = {
        "piper_feed": piper_feed,
        "piper_sim_feed": piper_sim_feed,
        "piper_step_command": piper_step_command,
        "piper_command_label": piper_command_label,
        "elite_joints": elite_joints,
        "elite_tcp_pose_6d": target_elite_tcp.astype(float).tolist(),
        "elite_tcp_delta_6d": elite_tcp_delta.astype(float).tolist(),
    }
    for key in ("elite_desired_pose_3d", "elite_plan_progress", "elite_plan_tangent"):
        if key in action:
            value = action[key]
            if hasattr(value, "astype"):
                value = value.astype(float).tolist()
            record[key] = value
    return record


def estimator_config(args: argparse.Namespace) -> VisualDistanceEstimatorConfig:
    return VisualDistanceEstimatorConfig(
        contact_threshold_px=args.visual_distance_contact_threshold_px,
        confidence_band_px=args.visual_distance_confidence_band_px,
        latency_steps=args.visual_distance_latency_steps,
        contact_rule=args.visual_distance_contact_rule,
        tip_3d_estimator=args.estimated_tip_3d,
        tip_3d_frame=args.estimated_tip_3d_frame,
        tip_3d_prefer_camera=args.estimated_tip_3d_prefer_camera,
        tip_3d_noise_std_m=args.estimated_tip_3d_noise_std_m,
        tip_3d_depth_noise_std_m=args.estimated_tip_3d_depth_noise_std_m,
        tip_3d_pixel_noise_std_px=args.estimated_tip_3d_pixel_noise_std_px,
        tip_3d_dropout_probability=args.estimated_tip_3d_dropout_probability,
        tip_3d_noise_seed=args.estimated_tip_3d_noise_seed,
        registered_geometry_estimator=args.registered_geometry_estimator,
        registered_geometry_margin_threshold_m=args.registered_geometry_margin_threshold_m,
        registered_geometry_wall_pull_threshold_m=args.registered_geometry_wall_pull_threshold_m,
    )


def maybe_estimator_fields(env: TipGuidedWireEnv, state: dict, args: argparse.Namespace, images: dict | None = None) -> dict:
    if not args.visual_distance_estimator and not args.estimated_tip_3d and not args.registered_geometry_estimator:
        return {}
    return estimate_visual_distance_contact(env, state, image_size=args.image_size, config=estimator_config(args), images=images)


def sample_state(env: TipGuidedWireEnv, obs: dict) -> dict:
    state = dict(obs)
    state["segment_min_distance_to_wall"] = env.segment_min_distance_to_wall()
    return state


def sample_record(episode_dir: Path, state: dict, action: dict, estimator_fields: dict) -> dict:
    step = int(state["step"])
    return {
        "episode": episode_dir.name,
        "task": state["task"],
        "scenario": SCENARIO,
        "step": step,
        "mode": "tip_guided_wire",
        "images": {
            "side": str((episode_dir / "frames" / "side" / f"{step:06d}.png").as_posix()),
            "top": str((episode_dir / "frames" / "top" / f"{step:06d}.png").as_posix()),
        },
        "instruction": state["instruction"],
        "state": {
            "tip_pos": state["tip_pos"],
            "heading": state["heading"],
            "target_pos": state["target_pos"],
            "contact_flag": state["contact_flag"],
            "contact_direction": state["contact_direction"],
            "contact_normal": state["contact_normal"],
            "contact_strength": state["contact_strength"],
            "distance_to_wall": state["distance_to_wall"],
            "segment_min_distance_to_wall": state["segment_min_distance_to_wall"],
            "path_progress": state["path_progress"],
            "piper_step": state["piper_step"],
            "piper_insertion_length": state["piper_insertion_length"],
            "boundary_projection_count": state["boundary_projection_count"],
            "boundary_projection_window": state["boundary_projection_window"],
            "last_boundary_projection": state["last_boundary_projection"],
            "elirobot_pose": state["elirobot_pose"],
            "elite_tcp_pose_6d": state.get("elite_tcp_pose_6d"),
            "magnetic_pose": state["magnetic_pose"],
            "lateral_offset": state["lateral_offset"],
            "path_tangent": state["path_tangent"],
            "local_radius": state["local_radius"],
            "robot_state": state["robot_state"],
            "controller_state": state.get("controller_state", {}),
            **estimator_fields,
        },
        "action": action,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def distance(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def collect_episode(
    args: argparse.Namespace,
    episode_dir: Path,
    task: str,
    seed: int,
    start_fraction: float,
) -> dict:
    env = build_env(args, seed)
    expert = build_expert(args)
    side_dir = episode_dir / "frames" / "side"
    top_dir = episode_dir / "frames" / "top"
    side_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    states: list[dict] = []
    actions: list[dict] = []
    samples: list[dict] = []
    total_reward = 0.0
    contact_values: list[float] = []
    tip_wall_values: list[float] = []
    segment_wall_values: list[float] = []
    progress_values: list[float] = []
    tip_to_magnetic_values: list[float] = []

    _gym_obs, info = env.reset(options={"task": task, "start_fraction": start_fraction})
    start_progress = float(info["obs_dict"]["path_progress"])
    final_obs = sample_state(env, info["obs_dict"])
    last_progress_log_step = -1

    try:
        while True:
            obs = sample_state(env, info["obs_dict"])
            step = int(obs["step"])
            contact_values.append(float(obs["contact_strength"]))
            tip_wall_values.append(float(obs["distance_to_wall"]))
            segment_wall_values.append(float(obs["segment_min_distance_to_wall"]))
            progress_values.append(float(obs["path_progress"]))
            tip_to_magnetic_values.append(distance(obs["tip_pos"], obs["magnetic_pose"]))

            action = action_record(env, expert.act(env))
            if bool(obs["success"]) or obs["failure_reason"] is not None or step >= args.max_steps:
                final_obs = obs
                break

            if args.progress_log_every > 0 and step - last_progress_log_step >= args.progress_log_every:
                last_progress_log_step = step
                progress_gain = float(obs["path_progress"]) - start_progress
                print(
                    f"  {episode_dir.name}: step={step}, gain={progress_gain:.2f}, "
                    f"samples={len(samples)}, min_tip_wall={min(tip_wall_values):.6f}, "
                    f"contact={max(contact_values):.3f}",
                    flush=True,
                )

            if step % max(args.sample_every, 1) == 0:
                images = env.render_camera_pair(size=args.image_size)
                cv2.imwrite(str(side_dir / f"{step:06d}.png"), images["side"])
                cv2.imwrite(str(top_dir / f"{step:06d}.png"), images["top"])
                estimator_fields = maybe_estimator_fields(env, obs, args, images=images)
                states.append(obs)
                actions.append({"step": step, "action": action})
                samples.append(sample_record(episode_dir, obs, action, estimator_fields))

            _gym_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                final_obs = sample_state(env, info["obs_dict"])
                contact_values.append(float(final_obs["contact_strength"]))
                tip_wall_values.append(float(final_obs["distance_to_wall"]))
                segment_wall_values.append(float(final_obs["segment_min_distance_to_wall"]))
                progress_values.append(float(final_obs["path_progress"]))
                tip_to_magnetic_values.append(distance(final_obs["tip_pos"], final_obs["magnetic_pose"]))
                break
    finally:
        env.close()

    progress_deltas = [progress_values[i] - progress_values[i - 1] for i in range(1, len(progress_values))]
    meta = {
        "episode": episode_dir.name,
        "task": task,
        "scenario": SCENARIO,
        "instruction": final_obs["instruction"],
        "seed": seed,
        "start_fraction": start_fraction,
        "start_progress": start_progress,
        "final_progress": float(final_obs["path_progress"]),
        "steps": int(final_obs["step"]),
        "samples": len(samples),
        "success": bool(final_obs["success"]),
        "success_reason": "env_success" if bool(final_obs["success"]) else None,
        "env_success": bool(final_obs["success"]),
        "failure_reason": final_obs["failure_reason"],
        "distance_to_target": float(final_obs["distance_to_target"]),
        "boundary_projection_count": int(final_obs["boundary_projection_count"]),
        "min_tip_distance_to_wall": min(tip_wall_values) if tip_wall_values else 0.0,
        "min_segment_distance_to_wall": min(segment_wall_values) if segment_wall_values else 0.0,
        "max_contact_strength": max(contact_values) if contact_values else 0.0,
        "contact_p95": percentile(contact_values, 95),
        "stall_fraction": float(sum(abs(x) < 1e-8 for x in progress_deltas) / max(len(progress_deltas), 1)),
        "slow_fraction": float(sum(0 <= x < args.slow_progress_threshold for x in progress_deltas) / max(len(progress_deltas), 1)),
        "tip_to_magnetic_median": percentile(tip_to_magnetic_values, 50),
        "tip_to_magnetic_p95": percentile(tip_to_magnetic_values, 95),
        "tip_to_magnetic_max": max(tip_to_magnetic_values) if tip_to_magnetic_values else 0.0,
        "total_reward": total_reward,
        "action_schema": action_schema(env),
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(episode_dir / "states.jsonl", states)
    write_jsonl(episode_dir / "actions.jsonl", actions)
    return {"meta": meta, "samples": samples}


def action_schema(env: TipGuidedWireEnv) -> dict:
    return {
        "type": ACTION_MODE,
        "piper": "signed_step_feed_hold_retract_plus_continuous_execution_value",
        "piper_feed": "real guidewire-level scalar command in [-1, 1]; current mainline route-plan uses 0=hold and 0.70=feed",
        "piper_sim_feed": "diagnostic simulator-side feeder stroke amount; not a real policy target",
        "piper_step_command": "-1=retract, 0=hold, 1=feed; current mainline emits feed/hold",
        "piper_command_label": "retract|hold|feed",
        "elite_tcp_pose_6d": "real-style Elite TCP target pose [xyz_mm, rpy_rad]; route-plan labels keep current orientation",
        "elite_tcp_delta_6d": "Elite TCP target minus current TCP pose [dxyz_mm, drpy_rad]; preferred VLA-facing Elite target",
        "elite_joints": "executed compatibility command after IK; kept for diagnostics/control logs",
        "piper_joint_names": env.piper_joint_names,
        "elite_joint_names": env.elite_joint_names,
    }


def effective_magnet_height_offset(args: argparse.Namespace) -> float:
    scene_path = REPO_ROOT / args.scene_config
    scene_scale = 1.0
    if scene_path.exists():
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        scene_scale = float(scene.get("vessel_scale", 1.0))
    return float(TipGuidedWireConfig().magnet_tip_height_offset) * scene_scale


def mainline_config(args: argparse.Namespace) -> dict:
    return {
        "route": "formal_tip_line_guidance_v1",
        "formal_data": True,
        "expert": "registered_route_plan_no_tip_contact_oracle",
        "env": "tip_guided_wire",
        "guidewire_assumption": "tip_centric_hard_elastic",
        "wire_visual_mode": "line",
        "wire_visual_radius": float(args.wire_visual_radius),
        "wire_visual_rgb": args.wire_visual_rgb,
        "wire_tip_visual_rgb": args.wire_tip_visual_rgb,
        "wire_tip_visual_segments": int(args.wire_tip_visual_segments),
        "wire_tip_visual_radius_scale": float(args.wire_tip_visual_radius_scale),
        "wire_tip_marker_radius": float(args.wire_tip_marker_radius),
        "wire_tip_marker_alpha": float(args.wire_tip_marker_alpha),
        "wire_visual_offset": [float(x) for x in args.wire_visual_offset],
        "wire_visual_tail": "piper_tcp_to_configured_via_points_then_inside_vessel",
        "wire_segments": int(args.wire_segments),
        "render_domain_preset": args.render_domain_preset,
        "tip_tail_decay_segments": 18.0,
        "show_tool_markers": False,
        "show_path_tubes": False,
        "action_mode": ACTION_MODE,
        "elite_action_target": "tcp_delta_6d_plus_ik_execution",
        "piper_schedule": {
            "piper_cmd": float(args.piper_cmd),
            "piper_command_period": int(args.piper_command_period),
            "piper_command_width": int(args.piper_command_width),
            "route_plan_command_phase_lock": True,
            "piper_command_as_event": bool(getattr(args, "piper_command_as_event", False) or int(max(getattr(args, "piper_primitive_steps", 1), 1)) > 1),
            "piper_primitive_steps": int(max(getattr(args, "piper_primitive_steps", 1), 1)),
            "piper_primitive_feed_value": float(getattr(args, "piper_primitive_feed_value", 0.70)),
            "piper_primitive_retract_value": float(getattr(args, "piper_primitive_retract_value", 0.70)),
        },
        "route_plan": {
            "route_plan_step": float(args.route_plan_step),
            "elite_ahead": float(args.elite_ahead),
            "elite_height_offset": effective_magnet_height_offset(args),
            "elite_target_semantics": "front_up_magnetic_target = route point + tangent * elite_ahead + world_z * elite_height_offset",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the current formal mainline dataset: tip-centric hard-elastic guidewire, "
            "continuous line visual, registered route-plan expert, hidden debug visuals."
        )
    )
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/formal_tip_line_guidance_dataset_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--max-attempts-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--start-fraction-min", type=float, default=0.42)
    parser.add_argument("--start-fraction-max", type=float, default=0.54)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--progress-log-every", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument(
        "--render-domain-preset",
        choices=["default", "branchs_like_v1"],
        default="default",
        help="Optional renderer/domain preset. branchs_like_v1 is a diagnostic preset for closer senior branchs-style images.",
    )
    parser.add_argument("--wire-segments", type=int, default=160)
    parser.add_argument("--wire-visual-radius", type=float, default=0.0008)
    parser.add_argument("--wire-visual-rgb", default="0.02 0.02 0.018")
    parser.add_argument("--wire-tip-visual-rgb", default="0.78 0.04 0.02")
    parser.add_argument("--wire-tip-visual-segments", type=int, default=8)
    parser.add_argument("--wire-tip-visual-radius-scale", type=float, default=2.2)
    parser.add_argument("--wire-tip-marker-radius", type=float, default=0.0020)
    parser.add_argument("--wire-tip-marker-alpha", type=float, default=0.95)
    parser.add_argument("--wire-visual-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--slow-progress-threshold", type=float, default=0.05)
    parser.add_argument("--route-plan-step", type=float, default=0.24)
    parser.add_argument("--elite-ahead", type=float, default=0.010)
    parser.add_argument("--piper-cmd", type=float, default=0.70)
    parser.add_argument("--piper-command-period", type=int, default=40)
    parser.add_argument("--piper-command-width", type=int, default=20)
    parser.add_argument(
        "--piper-primitive-steps",
        type=int,
        default=1,
        help="Number of simulator steps used to execute one real-style Piper feed/retract intent.",
    )
    parser.add_argument("--piper-primitive-feed-value", type=float, default=0.70)
    parser.add_argument("--piper-primitive-retract-value", type=float, default=0.70)
    parser.add_argument(
        "--piper-command-as-event",
        action="store_true",
        help="Emit one feed event per schedule period instead of a sustained feed window. Automatically enabled when --piper-primitive-steps > 1.",
    )
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument(
        "--formal-data",
        action="store_true",
        help="Compatibility no-op: this collector always uses the formal mainline configuration.",
    )
    parser.add_argument("--visual-distance-estimator", action="store_true")
    parser.add_argument("--visual-distance-contact-threshold-px", type=float, default=1.0)
    parser.add_argument("--visual-distance-confidence-band-px", type=float, default=8.0)
    parser.add_argument("--visual-distance-latency-steps", type=int, default=0)
    parser.add_argument(
        "--visual-distance-contact-rule",
        choices=["all_visible", "any_visible", "median"],
        default="all_visible",
    )
    parser.add_argument("--estimated-tip-3d", action="store_true")
    parser.add_argument("--estimated-tip-3d-frame", choices=["world"], default="world")
    parser.add_argument("--estimated-tip-3d-prefer-camera", choices=["auto", "side", "top"], default="auto")
    parser.add_argument("--estimated-tip-3d-noise-std-m", type=float, default=0.0015)
    parser.add_argument("--estimated-tip-3d-depth-noise-std-m", type=float, default=0.0008)
    parser.add_argument("--estimated-tip-3d-pixel-noise-std-px", type=float, default=0.75)
    parser.add_argument("--estimated-tip-3d-dropout-probability", type=float, default=0.0)
    parser.add_argument("--estimated-tip-3d-noise-seed", type=int, default=20260628)
    parser.add_argument("--registered-geometry-estimator", action="store_true")
    parser.add_argument("--registered-geometry-margin-threshold-m", type=float, default=0.0012)
    parser.add_argument("--registered-geometry-wall-pull-threshold-m", type=float, default=0.0025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_fraction_min > args.start_fraction_max:
        raise ValueError("--start-fraction-min must be <= --start-fraction-max")
    if args.registered_geometry_estimator and not args.estimated_tip_3d:
        raise ValueError("--registered-geometry-estimator requires --estimated-tip-3d")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mesh": args.mesh,
        "route_config": args.route_config,
        "scene_config": args.scene_config,
        "camera_config": args.camera_config,
        "mode": SCENARIO,
        "guidance_mode": "physical",
        "mainline_config": mainline_config(args),
        "validity_audit": {
            "formal_data": True,
            "route": "route_2_formal_tip_line_guidance",
            "expert_validity": "route_plan_no_tip_contact_oracle",
            "formal_data_allowed": True,
            "visual_leakage_guard": "tool_markers_and_path_tubes_hidden_by_construction",
        },
        "action_schema": {"type": ACTION_MODE},
        "episodes": [],
        "rejected_episodes": [],
        "samples": [],
    }
    if args.visual_distance_estimator or args.estimated_tip_3d or args.registered_geometry_estimator:
        manifest["observation_provenance"] = visual_distance_observation_provenance(estimator_config(args))
    if args.visual_distance_estimator:
        manifest["visual_distance_estimator"] = {
            "enabled": True,
            "source": "synthetic_visual_distance_estimator",
            "contact_threshold_px": float(args.visual_distance_contact_threshold_px),
            "confidence_band_px": float(args.visual_distance_confidence_band_px),
            "latency_steps": int(args.visual_distance_latency_steps),
            "contact_rule": args.visual_distance_contact_rule,
        }
    if args.estimated_tip_3d:
        manifest["estimated_tip_3d_estimator"] = {
            "enabled": True,
            "source": "synthetic_rgbd_tip_estimator",
            "frame": args.estimated_tip_3d_frame,
            "prefer_camera": args.estimated_tip_3d_prefer_camera,
            "noise_std_m": float(args.estimated_tip_3d_noise_std_m),
            "depth_noise_std_m": float(args.estimated_tip_3d_depth_noise_std_m),
            "pixel_noise_std_px": float(args.estimated_tip_3d_pixel_noise_std_px),
            "dropout_probability": float(args.estimated_tip_3d_dropout_probability),
            "latency_steps": int(args.visual_distance_latency_steps),
            "noise_seed": int(args.estimated_tip_3d_noise_seed),
        }
    if args.registered_geometry_estimator:
        manifest["registered_geometry_estimator"] = {
            "enabled": True,
            "source": "registered_route_geometry_estimator",
            "requires": "estimated_tip_pos_3d",
            "margin_threshold_m": float(args.registered_geometry_margin_threshold_m),
            "wall_pull_threshold_m": float(args.registered_geometry_wall_pull_threshold_m),
        }

    accepted = 0
    attempts = 0
    for task in args.tasks:
        for slot in range(1, args.episodes_per_task + 1):
            for retry in range(1, args.max_attempts_per_episode + 1):
                attempts += 1
                seed = args.seed + attempts
                start_fraction = float(rng.uniform(args.start_fraction_min, args.start_fraction_max))
                episode_dir = out_root / f"episode_{task}_{slot:04d}_try_{retry:02d}"
                result = collect_episode(args, episode_dir, task, seed, start_fraction)
                meta = result["meta"]
                print(
                    f"{meta['episode']}: task={task}, success={meta['success']}, "
                    f"steps={meta['steps']}, gain={meta['final_progress'] - meta['start_progress']:.2f}, "
                    f"samples={meta['samples']}, min_tip_wall={meta['min_tip_distance_to_wall']:.6f}, "
                    f"contact={meta['max_contact_strength']:.3f}, failure={meta['failure_reason']}",
                    flush=True,
                )
                if meta["success"] or args.keep_failures:
                    manifest["action_schema"] = meta["action_schema"]
                    manifest["episodes"].append(meta)
                    manifest["samples"].extend(result["samples"])
                else:
                    manifest["rejected_episodes"].append(meta)
                if meta["success"]:
                    accepted += 1
                    break
            else:
                print(f"WARNING: failed to collect successful {task} episode slot {slot}")

    manifest["summary"] = {
        "accepted_episodes": accepted,
        "attempts": attempts,
        "samples": len(manifest["samples"]),
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved formal tip-line guidance manifest to {manifest_path}")


if __name__ == "__main__":
    main()
