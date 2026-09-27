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

from simulation.collect_mujoco_physical_guidance import action_dict, write_jsonl  # noqa: E402
from simulation.mujoco_guided_wire_env import MuJoCoRoutePlanGuideExpert  # noqa: E402
from simulation.visual_distance_estimator import (  # noqa: E402
    VisualDistanceEstimatorConfig,
    estimate_visual_distance_contact,
    observation_provenance as visual_distance_observation_provenance,
)
from simulation.tip_guided_wire_env import (  # noqa: E402
    TipGuidedWireConfig,
    TipGuidedWireEnv,
    TipMagneticGuideExpert,
)


LEGACY_NOTICE = (
    "LEGACY/DIAGNOSTIC ENTRYPOINT: simulation.collect_tip_guided_wire keeps older "
    "oracle/route-plan switches for comparison. New formal data collection should "
    "use simulation.collect_formal_tip_line_guidance instead."
)


def build_env(args: argparse.Namespace, seed: int) -> TipGuidedWireEnv:
    config = TipGuidedWireConfig(
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
        piper_advection_scale=args.piper_advection_scale,
        elite_motion_smoothing=args.elite_motion_smoothing,
        elite_joint_rate_limit=args.elite_joint_rate_limit,
        elite_joint_accel_limit=args.elite_joint_accel_limit,
        tip_progress_gain=args.tip_progress_gain,
        tip_retract_progress_gain=args.tip_retract_progress_gain,
        tip_magnetic_progress_gain=args.tip_magnetic_progress_gain,
        tip_lateral_magnetic_gain=args.tip_lateral_magnetic_gain,
        tip_lateral_centering_gain=args.tip_lateral_centering_gain,
        tip_contact_relief_gain=args.tip_contact_relief_gain,
        tip_max_lateral_radius_fraction=args.tip_max_lateral_radius_fraction,
        tip_tail_decay_segments=args.tip_tail_decay_segments,
    )
    return TipGuidedWireEnv(config, seed=seed)


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
    return TipMagneticGuideExpert(
        rng_seed=seed,
        steer_noise=args.steer_noise,
        lookahead_points=args.lookahead_points,
        elite_ahead=args.elite_ahead,
        lateral_gain=args.lateral_gain,
        piper_cmd=args.piper_cmd,
    )


def sample_state(env: TipGuidedWireEnv, obs: dict) -> dict:
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


def maybe_add_visual_distance_estimate(env: TipGuidedWireEnv, state: dict, args: argparse.Namespace) -> dict:
    if not args.visual_distance_estimator:
        return {}
    return estimate_visual_distance_contact(env, state, image_size=args.image_size, config=estimator_config(args))


def sample_record(episode_dir: Path, state: dict, action: dict, estimator_fields: dict | None = None) -> dict:
    step = int(state["step"])
    estimator_fields = estimator_fields or {}
    return {
        "episode": episode_dir.name,
        "task": state["task"],
        "scenario": "tip_guided_wire",
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


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


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
    contact_values: list[float] = []
    tip_wall_values: list[float] = []
    segment_wall_values: list[float] = []
    progress_values: list[float] = []
    tip_to_magnetic_values: list[float] = []

    _gym_obs, info = env.reset(options={"task": task, "start_fraction": start_fraction})
    start_progress = float(info["obs_dict"]["path_progress"])
    target_progress = min(start_progress + progress_delta, len(env.paths[task]) - 1)
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

            action = expert.act(env)
            serial_action = action_dict(env, action, args.action_mode)
            reached_progress_target = float(obs["path_progress"]) >= target_progress
            env_success = bool(obs["success"])
            progress_success = reached_progress_target and args.success_mode == "progress_or_env"
            if progress_success or env_success or obs["failure_reason"] is not None or step >= args.max_steps:
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
                estimator_fields = maybe_add_visual_distance_estimate(env, obs, args)
                states.append(obs)
                actions.append({"step": step, "action": serial_action})
                samples.append(sample_record(episode_dir, obs, serial_action, estimator_fields))

            _gym_obs, reward, terminated, truncated, info = env.step(serial_action)
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

    reached_progress_target = float(final_obs["path_progress"]) >= target_progress
    env_success = bool(final_obs["success"])
    progress_success = reached_progress_target and args.success_mode == "progress_or_env"
    short_success = progress_success or env_success
    success_reason = "env_success" if env_success else ("progress_target" if progress_success else None)
    progress_deltas = [
        progress_values[i] - progress_values[i - 1] for i in range(1, len(progress_values))
    ]
    meta = {
        "episode": episode_dir.name,
        "task": task,
        "scenario": "tip_guided_wire",
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


def tip_params(args: argparse.Namespace) -> dict:
    keys = [
        "piper_advection_scale",
        "elite_motion_smoothing",
        "elite_joint_rate_limit",
        "elite_joint_accel_limit",
        "tip_progress_gain",
        "tip_retract_progress_gain",
        "tip_magnetic_progress_gain",
        "tip_lateral_magnetic_gain",
        "tip_lateral_centering_gain",
        "tip_contact_relief_gain",
        "tip_max_lateral_radius_fraction",
        "tip_tail_decay_segments",
        "lookahead_points",
        "elite_ahead",
        "lateral_gain",
        "piper_cmd",
        "piper_command_period",
        "piper_command_width",
        "expert_mode",
        "route_plan_step",
        "wire_visual_mode",
        "wire_visual_radius",
        "wire_line_width",
        "wire_line_core_width",
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
            "TipMagneticGuideExpert uses exact simulated tip state, local vessel "
            "frame, centerline lookahead, exact contact/wall signals, and IK. "
            "Use for diagnostics only unless a matching real sensing/control "
            "path is explicitly defined."
        ),
    }


def validate_formal_data_args(args: argparse.Namespace) -> None:
    if args.formal_data and args.expert_mode != "route_plan":
        audit = validity_audit(args)
        raise ValueError(
            "--formal-data is not allowed for simulation.collect_tip_guided_wire "
            f"with the current expert: {audit['reason']}"
        )
    if args.formal_data and not args.hide_tool_markers:
        raise ValueError("--formal-data requires --hide-tool-markers to avoid visual leakage.")
    if args.formal_data and not args.hide_path_tubes:
        raise ValueError("--formal-data requires --hide-path-tubes to avoid visual leakage.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy diagnostic tip-centric MuJoCo collection entrypoint.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation_output/mujoco_camera_config.json")
    parser.add_argument("--out", default="simulation_output/tip_guided_wire_dataset_v1")
    parser.add_argument("--tasks", nargs="+", default=["left", "right"], choices=["left", "right"])
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-attempts-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9801)
    parser.add_argument("--start-fraction-min", type=float, default=0.58)
    parser.add_argument("--start-fraction-max", type=float, default=0.70)
    parser.add_argument("--progress-deltas", nargs="+", type=float, default=[999.0])
    parser.add_argument(
        "--success-mode",
        choices=["progress_or_env", "env_success"],
        default="env_success",
        help="Accept episodes after reaching a progress target, or require the environment success condition.",
    )
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--wire-segments", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--progress-log-every", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--robot-visual-mode", choices=["kinematic", "static", "none"], default="kinematic")
    parser.add_argument("--wire-visual-mode", choices=["segments", "line", "both"], default="line")
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
    parser.add_argument("--lookahead-points", type=int, default=8)
    parser.add_argument("--route-plan-step", type=float, default=0.32)
    parser.add_argument("--elite-ahead", type=float, default=0.018)
    parser.add_argument("--lateral-gain", type=float, default=0.75)
    parser.add_argument("--piper-cmd", type=float, default=0.34)
    parser.add_argument("--piper-command-period", type=int, default=125)
    parser.add_argument("--piper-command-width", type=int, default=1)
    parser.add_argument(
        "--route-plan-command-phase-lock",
        action="store_true",
        help="Advance route-plan progress only during feed-command phases, compensating by duty cycle.",
    )
    parser.add_argument("--piper-advection-scale", type=float, default=0.11)
    parser.add_argument("--elite-motion-smoothing", type=float, default=0.55)
    parser.add_argument("--elite-joint-rate-limit", type=float, default=0.0)
    parser.add_argument("--elite-joint-accel-limit", type=float, default=0.0)
    parser.add_argument("--tip-progress-gain", type=float, default=0.45)
    parser.add_argument("--tip-retract-progress-gain", type=float, default=0.015)
    parser.add_argument("--tip-magnetic-progress-gain", type=float, default=0.020)
    parser.add_argument("--tip-lateral-magnetic-gain", type=float, default=0.16)
    parser.add_argument("--tip-lateral-centering-gain", type=float, default=0.06)
    parser.add_argument("--tip-contact-relief-gain", type=float, default=0.42)
    parser.add_argument("--tip-max-lateral-radius-fraction", type=float, default=0.82)
    parser.add_argument("--tip-tail-decay-segments", type=float, default=18.0)
    parser.add_argument("--slow-progress-threshold", type=float, default=0.05)
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
        "mode": "tip_guided_wire",
        "guidance_mode": "physical",
        "tip_params": tip_params(args),
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
                    f"steps={meta['steps']}, gain={meta['final_progress'] - meta['start_progress']:.2f}, "
                    f"samples={meta['samples']}, min_tip_wall={meta['min_tip_distance_to_wall']:.6f}, "
                    f"contact={meta['max_contact_strength']:.3f}, stall={meta['stall_fraction']:.3f}, "
                    f"failure={meta['failure_reason']}",
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
    print(f"Saved tip-guided-wire dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
