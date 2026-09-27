import argparse
import json
from pathlib import Path

import cv2

from simulation.dual_arm_guidewire_3d_env import DualArmGuidewire3DEnv, DualArmMesh3DEnvConfig, DualArmPathExpert, DualArmRecoveryExpert, DualArmSuccessRecoveryExpert


def make_expert(scenario: str, seed: int):
    if scenario == "normal":
        return DualArmRecoveryExpert(rng_seed=seed, steer_noise=0.0, push_strength=1.0)
    if scenario == "noisy":
        return DualArmRecoveryExpert(rng_seed=seed, steer_noise=0.02, push_strength=1.05)
    if scenario == "contact_recovery":
        return DualArmRecoveryExpert(rng_seed=seed, steer_noise=0.025, push_strength=1.15)
    if scenario == "success_recovery":
        return DualArmSuccessRecoveryExpert(rng_seed=seed, steer_noise=0.0, push_strength=0.35)
    raise ValueError(f"Unknown scenario: {scenario}")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_episode(env, expert, task: str, out_dir: Path, scenario: str, image_size: int, start_progress: float = 0.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    side_dir = out_dir / "frames" / "side"
    top_dir = out_dir / "frames" / "top"
    side_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    _obs, info = env.reset(options={"task": task})
    if start_progress > 0.0:
        env.set_path_progress(start_progress)
        info = env._info(env._obs_dict(done=False, success=False, failure_reason=None))
    states = []
    actions = []
    total_reward = 0.0
    while True:
        obs_dict = info["obs_dict"]
        images = env.render_camera_pair(size=image_size)
        step = obs_dict["step"]
        cv2.imwrite(str(side_dir / f"{step:06d}.png"), images["side"])
        cv2.imwrite(str(top_dir / f"{step:06d}.png"), images["top"])
        action = expert.act(env)
        _next_obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs_dict)
        actions.append({"step": step, "action": {"piper": float(action["piper"][0]), "elirobot_delta": action["elirobot_delta"].astype(float).tolist()}, "reward": float(reward)})
        total_reward += float(reward)
        if terminated or truncated:
            final_obs = info["obs_dict"]
            images = env.render_camera_pair(size=image_size)
            final_step = final_obs["step"]
            cv2.imwrite(str(side_dir / f"{final_step:06d}.png"), images["side"])
            cv2.imwrite(str(top_dir / f"{final_step:06d}.png"), images["top"])
            states.append(final_obs)
            break

    meta = {
        "task": task,
        "scenario": scenario,
        "instruction": env.instruction,
        "steps": states[-1]["step"],
        "success": states[-1]["success"],
        "failure_reason": states[-1]["failure_reason"],
        "total_reward": total_reward,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "states.jsonl", states)
    write_jsonl(out_dir / "actions.jsonl", actions)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Collect dual-arm guidewire episodes from the 3D simulator.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_route_v1.json")
    parser.add_argument("--out", default="simulation_output/dual_arm_dataset_v1")
    parser.add_argument("--episodes-per-combo", type=int, default=2)
    parser.add_argument("--scenarios", nargs="+", default=["normal", "noisy", "contact_recovery"])
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--start-progress-mode", choices=["entry", "mixed", "late"], default="entry")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mesh": args.mesh,
        "route_config": args.route_config,
        "mode": "dual_arm",
        "episodes": [],
        "samples": [],
    }

    episode_index = 1
    for scenario in args.scenarios:
        for task in args.tasks:
            for _ in range(args.episodes_per_combo):
                seed = args.seed + episode_index
                config_kwargs = {"mesh_path": args.mesh, "route_config_path": args.route_config}
                if args.max_steps is not None:
                    config_kwargs["max_steps"] = args.max_steps
                env = DualArmGuidewire3DEnv(DualArmMesh3DEnvConfig(**config_kwargs), seed=seed)
                expert = make_expert(scenario, seed)
                start_progress = 0.0
                if args.start_progress_mode != "entry":
                    path_len = len(env.paths[task]) - 1
                    rng = __import__("numpy").random.default_rng(seed)
                    if args.start_progress_mode == "late":
                        start_progress = float(rng.uniform(0.55, 0.86) * path_len)
                    elif args.start_progress_mode == "mixed":
                        if episode_index % 3 == 0:
                            start_progress = float(rng.uniform(0.55, 0.86) * path_len)
                        elif episode_index % 3 == 1:
                            start_progress = float(rng.uniform(0.25, 0.55) * path_len)
                if scenario == "success_recovery":
                    path_len = len(env.paths[task]) - 1
                    rng = __import__("numpy").random.default_rng(seed)
                    if args.start_progress_mode == "entry":
                        start_progress = 0.0
                    elif args.start_progress_mode == "late":
                        start_progress = float(rng.uniform(0.55, 0.86) * path_len)
                    else:
                        start_progress = float(rng.uniform(0.08, 0.52) * path_len)
                episode_dir = out_root / f"episode_{episode_index:06d}_{scenario}_{task}"
                meta = collect_episode(env, expert, task, episode_dir, scenario=scenario, image_size=args.image_size, start_progress=start_progress)
                meta["episode"] = episode_dir.name
                meta["start_progress"] = start_progress
                manifest["episodes"].append(meta)

                states = [json.loads(x) for x in (episode_dir / "states.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
                actions = [json.loads(x) for x in (episode_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
                action_by_step = {item["step"]: item for item in actions}
                for state in states:
                    step = state["step"]
                    if step not in action_by_step:
                        continue
                    manifest["samples"].append(
                        {
                            "episode": episode_dir.name,
                            "task": meta["task"],
                            "scenario": meta["scenario"],
                            "step": step,
                            "mode": "dual_arm",
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
                                "path_progress": state["path_progress"],
                                "piper_step": state["piper_step"],
                                "piper_insertion_length": state["piper_insertion_length"],
                                "elirobot_pose": state["elirobot_pose"],
                                "lateral_offset": state["lateral_offset"],
                                "path_tangent": state["path_tangent"],
                                "local_radius": state["local_radius"],
                            },
                            "action": action_by_step[step]["action"],
                            "reward": action_by_step[step]["reward"],
                        }
                    )

                print(
                    f"{episode_dir.name}: scenario={scenario}, task={task}, "
                    f"success={meta['success']}, steps={meta['steps']}, failure={meta['failure_reason']}"
                )
                episode_index += 1

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset manifest to {manifest_path}")
    print(f"Episodes: {len(manifest['episodes'])}, samples: {len(manifest['samples'])}")


if __name__ == "__main__":
    main()
