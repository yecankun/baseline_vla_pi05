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

from simulation.mujoco_guided_wire_env import (  # noqa: E402
    MuJoCoGuidedWireConfig,
    MuJoCoGuidedWireEnv,
    MuJoCoMagneticGuideExpert,
    MuJoCoRoutePlanGuideExpert,
)
from simulation.visual_distance_estimator import (  # noqa: E402
    VisualDistanceEstimatorConfig,
    estimate_visual_distance_contact,
    observation_provenance as visual_distance_observation_provenance,
)


LEGACY_NOTICE = (
    "LEGACY/DIAGNOSTIC ENTRYPOINT: simulation.collect_mujoco_physical_guidance "
    "keeps the older full-polyline physical guidewire route for comparison. New "
    "formal mainline data should use simulation.collect_formal_tip_line_guidance."
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_dict(env: MuJoCoGuidedWireEnv, action: dict, action_mode: str) -> dict:
    elite_joints = {name: float(value) for name, value in action["elite_joints"].items()}
    current_elite_tcp = env.robot_tool_pose6d("elite").astype(np.float32)
    target_elite_vec = env._robot_joint_vector_from_action("elite", elite_joints)
    target_elite_fk = env._robot_tool_pose6d_from_vector("elite", target_elite_vec).astype(np.float32)
    # Match the inherited Elite controller semantics: learn a TCP translation
    # delta and keep the current tool orientation for IK.
    target_elite_tcp = current_elite_tcp.copy()
    target_elite_tcp[:3] = target_elite_fk[:3]
    elite_tcp_delta = target_elite_tcp - current_elite_tcp
    if action_mode == "joint_target":
        return {
            "piper_joints": {name: float(value) for name, value in action["piper_joints"].items()},
            "elite_joints": elite_joints,
            "elite_tcp_pose_6d": target_elite_tcp.astype(float).tolist(),
            "elite_tcp_delta_6d": elite_tcp_delta.astype(float).tolist(),
        }
    piper_vec = env._robot_joint_vector_from_action("piper", action["piper_joints"])
    target_signal = env._piper_feed_signal(piper_vec)
    current_signal = float(getattr(env, "piper_feed_signal", env._piper_feed_signal(env.piper_home_joint_vector)))
    feed_unit = max(float(env.config.advance_step) * float(env.config.piper_advection_scale), 1e-8)
    piper_sim_feed = float(np.clip((target_signal - current_signal) / feed_unit, -1.0, 1.0))
    if "piper_step_command" in action:
        piper_step_command = int(np.clip(np.sign(float(action["piper_step_command"])), -1, 1))
        piper_feed = float(np.clip(float(action.get("piper_feed_command", piper_step_command)), -1.0, 1.0))
    else:
        # Negative simulator-side Piper stroke currently resets the feeder travel;
        # it is not a real guidewire retract command unless an expert emits one.
        piper_step_command = 1 if piper_sim_feed > 0.05 else 0
        piper_feed = float(piper_step_command)
    piper_command_label = {-1: "retract", 0: "hold", 1: "feed"}[piper_step_command]
    return {
        "piper_feed": piper_feed,
        "piper_sim_feed": piper_sim_feed,
        "piper_step_command": piper_step_command,
        "piper_command_label": piper_command_label,
        "elite_joints": elite_joints,
        "elite_tcp_pose_6d": target_elite_tcp.astype(float).tolist(),
        "elite_tcp_delta_6d": elite_tcp_delta.astype(float).tolist(),
    }


def build_env(args: argparse.Namespace, seed: int) -> MuJoCoGuidedWireEnv:
    config = MuJoCoGuidedWireConfig(
        mesh_path=args.mesh,
        route_config_path=args.route_config,
        scene_config_path=args.scene_config,
        camera_config_path=args.camera_config,
        max_steps=args.max_steps,
        wire_segments=args.wire_segments,
        robot_visual_mode=args.robot_visual_mode,
        wire_visual_mode=args.wire_visual_mode,
        wire_visual_radius=args.wire_visual_radius,
        wire_line_width=args.wire_line_width,
        wire_line_core_width=args.wire_line_core_width,
        show_vessel_mesh=not args.hide_vessel_mesh,
        show_path_tubes=not args.hide_path_tubes,
        show_tool_markers=not args.hide_tool_markers,
        render_observation=False,
        render_width=args.render_width,
        render_height=args.render_height,
        guidance_mode="physical",
        magnet_force=args.magnet_force,
        magnet_range=args.magnet_range,
        piper_force=args.piper_force,
        piper_advection_scale=args.piper_advection_scale,
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
    )
    return MuJoCoGuidedWireEnv(config, seed=seed)


def make_expert(args: argparse.Namespace, seed: int):
    if args.expert_mode == "route_plan":
        return MuJoCoRoutePlanGuideExpert(
            plan_step=args.route_plan_step,
            elite_ahead=args.elite_ahead,
            piper_cmd=args.piper_cmd,
            piper_command_period=args.piper_command_period,
            piper_command_width=args.piper_command_width,
            route_plan_command_phase_lock=args.route_plan_command_phase_lock,
        )
    return MuJoCoMagneticGuideExpert(
        rng_seed=seed,
        steer_noise=args.steer_noise,
        lookahead_points=args.lookahead_points,
        elite_ahead=args.elite_ahead,
        lateral_gain=args.lateral_gain,
        piper_cmd=args.piper_cmd,
    )


def sample_state(env: MuJoCoGuidedWireEnv, obs: dict) -> dict:
    state = dict(obs)
    state["segment_min_distance_to_wall"] = env.segment_min_distance_to_wall()
    return state


def estimator_config(args: argparse.Namespace) -> VisualDistanceEstimatorConfig:
    return VisualDistanceEstimatorConfig(
        contact_threshold_px=args.visual_distance_contact_threshold_px,
        confidence_band_px=args.visual_distance_confidence_band_px,
        latency_steps=args.visual_distance_latency_steps,
        contact_rule=args.visual_distance_contact_rule,
    )


def maybe_add_visual_distance_estimate(env: MuJoCoGuidedWireEnv, state: dict, args: argparse.Namespace) -> dict:
    if not args.visual_distance_estimator:
        return {}
    return estimate_visual_distance_contact(env, state, image_size=args.image_size, config=estimator_config(args))


def sample_record(episode_dir: Path, state: dict, action: dict, estimator_fields: dict | None = None) -> dict:
    step = int(state["step"])
    estimator_fields = estimator_fields or {}
    return {
        "episode": episode_dir.name,
        "task": state["task"],
        "scenario": "mujoco_physical_guidance",
        "step": step,
        "mode": "mujoco_physical_guidance",
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


def collect_episode(
    args: argparse.Namespace,
    episode_dir: Path,
    task: str,
    seed: int,
    start_fraction: float,
    progress_delta: float,
) -> dict:
    env = build_env(args, seed)
    expert = make_expert(args, seed)
    side_dir = episode_dir / "frames" / "side"
    top_dir = episode_dir / "frames" / "top"
    side_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    states: list[dict] = []
    actions: list[dict] = []
    samples: list[dict] = []
    total_reward = 0.0
    min_tip_wall = float("inf")
    min_segment_wall = float("inf")
    max_contact_strength = 0.0

    _gym_obs, info = env.reset(options={"task": task, "start_fraction": start_fraction})
    start_progress = float(info["obs_dict"]["path_progress"])
    target_progress = min(start_progress + progress_delta, len(env.paths[task]) - 1)
    final_obs = sample_state(env, info["obs_dict"])
    last_progress_log_step = -1

    try:
        while True:
            obs = sample_state(env, info["obs_dict"])
            step = int(obs["step"])
            min_tip_wall = min(min_tip_wall, float(obs["distance_to_wall"]))
            min_segment_wall = min(min_segment_wall, float(obs["segment_min_distance_to_wall"]))
            max_contact_strength = max(max_contact_strength, float(obs["contact_strength"]))

            action = expert.act(env)
            serial_action = action_dict(env, action, args.action_mode)
            reached_progress_target = float(obs["path_progress"]) >= target_progress
            env_success = bool(obs["success"])
            progress_success = reached_progress_target and args.success_mode == "progress_or_env"
            short_success = progress_success or env_success
            if short_success or obs["failure_reason"] is not None or step >= args.max_steps:
                final_obs = obs
                break

            if args.progress_log_every > 0 and step - last_progress_log_step >= args.progress_log_every:
                last_progress_log_step = step
                progress_gain = float(obs["path_progress"]) - start_progress
                print(
                    f"  {episode_dir.name}: step={step}, gain={progress_gain:.2f}/{progress_delta:.1f}, "
                    f"samples={len(samples)}, min_wall={min_segment_wall:.6f}, contact={max_contact_strength:.3f}",
                    flush=True,
                )

            if step % max(args.sample_every, 1) == 0:
                images = env.render_camera_pair(size=args.image_size)
                cv2.imwrite(str(side_dir / f"{step:06d}.png"), images["side"])
                cv2.imwrite(str(top_dir / f"{step:06d}.png"), images["top"])
                estimator_fields = maybe_add_visual_distance_estimate(env, obs, args)
                states.append(obs)
                actions.append({"step": step, "action": serial_action})
                samples.append(sample_record(episode_dir, obs, serial_action, estimator_fields))

            _gym_obs, reward, terminated, truncated, info = env.step(serial_action)
            total_reward += float(reward)
            if terminated or truncated:
                final_obs = sample_state(env, info["obs_dict"])
                min_tip_wall = min(min_tip_wall, float(final_obs["distance_to_wall"]))
                min_segment_wall = min(min_segment_wall, float(final_obs["segment_min_distance_to_wall"]))
                max_contact_strength = max(max_contact_strength, float(final_obs["contact_strength"]))
                break
    finally:
        env.close()

    reached_progress_target = float(final_obs["path_progress"]) >= target_progress
    env_success = bool(final_obs["success"])
    progress_success = reached_progress_target and args.success_mode == "progress_or_env"
    short_success = progress_success or env_success
    success_reason = "env_success" if env_success else ("progress_target" if progress_success else None)
    meta = {
        "episode": episode_dir.name,
        "task": task,
        "scenario": "mujoco_physical_guidance",
        "instruction": final_obs["instruction"],
        "seed": seed,
        "start_fraction": start_fraction,
        "progress_delta": progress_delta,
        "start_progress": start_progress,
        "target_progress": target_progress,
        "final_progress": float(final_obs["path_progress"]),
        "steps": int(final_obs["step"]),
        "samples": len(samples),
        "success": bool(short_success),
        "success_reason": success_reason,
        "reached_progress_target": bool(reached_progress_target),
        "env_success": env_success,
        "success_mode": args.success_mode,
        "failure_reason": final_obs["failure_reason"],
        "distance_to_target": float(final_obs["distance_to_target"]),
        "boundary_projection_count": int(final_obs["boundary_projection_count"]),
        "min_tip_distance_to_wall": min_tip_wall,
        "min_segment_distance_to_wall": min_segment_wall,
        "max_contact_strength": max_contact_strength,
        "total_reward": total_reward,
        "action_schema": {
            "type": args.action_mode,
            "piper": "normalized_feed_delta_with_signed_step_label" if args.action_mode == "piper_feed_elite_joint" else "joint_target",
            "piper_feed": "real guidewire-level scalar command in [-1, 1]; current route-plan uses 0=hold and 1=feed",
            "piper_sim_feed": "diagnostic simulator-side feeder stroke amount in [-1, 1]; negative values may be feeder reset, not necessarily guidewire retract",
            "piper_step_command": "-1=retract, 0=hold, 1=feed; real guidewire-level signed-step label",
            "piper_command_label": "retract|hold|feed",
            "elite_tcp_pose_6d": "real-style Elite TCP target pose [xyz_mm, rpy_rad]; current route-plan labels keep current orientation",
            "elite_tcp_delta_6d": "Elite TCP target minus current TCP pose [dxyz_mm, drpy_rad]; matches inherited pose-delta + IK control semantics",
            "piper_joint_names": env.piper_joint_names,
            "elite_joint_names": env.elite_joint_names,
        },
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(episode_dir / "states.jsonl", states)
    write_jsonl(episode_dir / "actions.jsonl", actions)
    return {"meta": meta, "samples": samples}


def physical_params(args: argparse.Namespace) -> dict:
    keys = [
        "magnet_force",
        "magnet_range",
        "piper_force",
        "piper_advection_scale",
        "spring_k",
        "bend_k",
        "damping",
        "sim_substeps",
        "wall_k",
        "gravity_z",
        "wire_max_turn_degrees",
        "physical_curvature_smoothing",
        "physical_curvature_iterations",
        "physical_tip_stiff_segments",
        "physical_tip_curvature_smoothing",
        "physical_tip_curvature_iterations",
        "wire_max_speed",
        "lookahead_points",
        "elite_ahead",
        "lateral_gain",
        "piper_cmd",
        "piper_command_period",
        "piper_command_width",
        "expert_mode",
        "route_plan_step",
    ]
    return {key: getattr(args, key) for key in keys}


def validity_audit(args: argparse.Namespace) -> dict:
    if args.expert_mode == "route_plan":
        return {
            "formal_data": bool(args.formal_data),
            "route": "route_2_pure_sim_physics",
            "expert_validity": "route_plan_no_tip_contact_oracle",
            "formal_data_allowed": True,
            "reason": (
                "Route-plan expert avoids exact tip position and exact wall/contact "
                "feedback. It uses a pre-registered vessel route, scheduled progress, "
                "Piper insertion state, and Elite IK as a formal-validity probe."
            ),
        }
    return {
        "formal_data": bool(args.formal_data),
        "route": "route_2_pure_sim_physics",
        "expert_validity": "diagnostic_oracle",
        "formal_data_allowed": False,
        "reason": (
            "MuJoCoMagneticGuideExpert uses exact simulated tip state, local "
            "vessel frame, centerline lookahead, exact contact/wall signals, "
            "and IK. Use for diagnostics only unless a matching real "
            "sensing/control path is explicitly defined."
        ),
    }


def validate_formal_data_args(args: argparse.Namespace) -> None:
    if args.formal_data and args.expert_mode != "route_plan":
        audit = validity_audit(args)
        raise ValueError(
            "--formal-data is not allowed for simulation.collect_mujoco_physical_guidance "
            f"with the current expert: {audit['reason']}"
        )
    if args.formal_data and not args.hide_tool_markers:
        raise ValueError("--formal-data requires --hide-tool-markers to avoid visual leakage.")
    if args.formal_data and not args.hide_path_tubes:
        raise ValueError("--formal-data requires --hide-path-tubes to avoid visual leakage.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy diagnostic full-polyline MuJoCo collection entrypoint.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/mujoco_physical_guidance_dataset_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-attempts-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--start-fraction-min", type=float, default=0.58)
    parser.add_argument("--start-fraction-max", type=float, default=0.70)
    parser.add_argument("--progress-deltas", nargs="+", type=float, default=[14.0, 20.0, 28.0, 40.0])
    parser.add_argument(
        "--success-mode",
        choices=["progress_or_env", "env_success"],
        default="progress_or_env",
        help="Accept episodes after reaching a progress target, or require the environment success condition.",
    )
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--wire-segments", type=int, default=34)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--progress-log-every", type=int, default=0, help="Print per-episode progress every N steps; 0 disables it.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--wire-visual-mode", choices=["segments", "line", "both"], default="segments")
    parser.add_argument("--wire-visual-radius", type=float, default=0.0025)
    parser.add_argument("--wire-line-width", type=int, default=4)
    parser.add_argument("--wire-line-core-width", type=int, default=1)
    parser.add_argument("--action-mode", choices=["piper_feed_elite_joint", "joint_target"], default="piper_feed_elite_joint")
    parser.add_argument("--hide-tool-markers", action="store_true")
    parser.add_argument("--hide-vessel-mesh", action="store_true")
    parser.add_argument("--hide-path-tubes", action="store_true")
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--expert-mode", choices=["oracle", "route_plan"], default="oracle")
    parser.add_argument(
        "--formal-data",
        action="store_true",
        help="Require formal sim-to-real data validity; rejects the current oracle expert.",
    )
    parser.add_argument("--steer-noise", type=float, default=0.0)
    parser.add_argument("--magnet-force", type=float, default=0.36)
    parser.add_argument("--magnet-range", type=float, default=0.17)
    parser.add_argument("--piper-force", type=float, default=0.17)
    parser.add_argument("--piper-advection-scale", type=float, default=0.11)
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
    parser.add_argument("--lookahead-points", type=int, default=7)
    parser.add_argument("--route-plan-step", type=float, default=0.32)
    parser.add_argument("--elite-ahead", type=float, default=0.018)
    parser.add_argument("--lateral-gain", type=float, default=0.75)
    parser.add_argument("--piper-cmd", type=float, default=0.17)
    parser.add_argument("--piper-command-period", type=int, default=125)
    parser.add_argument("--piper-command-width", type=int, default=1)
    parser.add_argument(
        "--route-plan-command-phase-lock",
        action="store_true",
        help="Advance route-plan progress only during feed-command phases, compensating by duty cycle.",
    )
    parser.add_argument(
        "--visual-distance-estimator",
        action="store_true",
        help="Write synthetic visual image-distance contact estimator fields and observation provenance.",
    )
    parser.add_argument("--visual-distance-contact-threshold-px", type=float, default=1.0)
    parser.add_argument("--visual-distance-confidence-band-px", type=float, default=8.0)
    parser.add_argument("--visual-distance-latency-steps", type=int, default=0)
    parser.add_argument(
        "--visual-distance-contact-rule",
        choices=["all_visible", "any_visible", "median"],
        default="all_visible",
        help="Rule for binary estimated_contact_flag. Continuous estimated_image_distance_px is still recorded.",
    )
    args = parser.parse_args()
    print(LEGACY_NOTICE, flush=True)
    validate_formal_data_args(args)

    if args.start_fraction_min > args.start_fraction_max:
        raise ValueError("--start-fraction-min must be <= --start-fraction-max")
    if not args.progress_deltas:
        raise ValueError("--progress-deltas must not be empty")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mesh": args.mesh,
        "route_config": args.route_config,
        "scene_config": args.scene_config,
        "camera_config": args.camera_config,
        "mode": "mujoco_physical_guidance",
        "guidance_mode": "physical",
        "physical_params": physical_params(args),
        "validity_audit": validity_audit(args),
        "action_schema": {"type": args.action_mode},
        "episodes": [],
        "rejected_episodes": [],
        "samples": [],
    }
    if args.visual_distance_estimator:
        manifest["observation_provenance"] = visual_distance_observation_provenance(estimator_config(args))
        manifest["visual_distance_estimator"] = {
            "enabled": True,
            "source": "synthetic_visual_distance_estimator",
            "contact_threshold_px": float(args.visual_distance_contact_threshold_px),
            "confidence_band_px": float(args.visual_distance_confidence_band_px),
            "latency_steps": int(args.visual_distance_latency_steps),
            "contact_rule": args.visual_distance_contact_rule,
        }

    accepted = 0
    attempts = 0
    for task in args.tasks:
        for slot in range(1, args.episodes_per_task + 1):
            for retry in range(1, args.max_attempts_per_episode + 1):
                attempts += 1
                seed = args.seed + attempts
                start_fraction = float(rng.uniform(args.start_fraction_min, args.start_fraction_max))
                progress_delta = float(rng.choice(np.asarray(args.progress_deltas, dtype=np.float32)))
                episode_dir = out_root / f"episode_{task}_{slot:04d}_try_{retry:02d}"
                result = collect_episode(args, episode_dir, task, seed, start_fraction, progress_delta)
                meta = result["meta"]
                print(
                    f"{meta['episode']}: task={task}, success={meta['success']}, reason={meta['success_reason']}, "
                    f"steps={meta['steps']}, gain={meta['final_progress'] - meta['start_progress']:.2f}/{progress_delta:.1f}, "
                    f"samples={meta['samples']}, min_wall={meta['min_segment_distance_to_wall']:.6f}, "
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
    print(f"Saved physical dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
