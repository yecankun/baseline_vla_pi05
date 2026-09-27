import argparse
import json
from pathlib import Path

import cv2

from simulation.collect_mesh_3d import collect_episode
from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, Mesh3DPathExpert, Mesh3DRecoveryExpert, MeshGuidewire3DEnv


def make_expert(scenario: str, seed: int):
    if scenario == "normal":
        return Mesh3DPathExpert(rng_seed=seed, steer_noise=0.0)
    if scenario == "noisy":
        return Mesh3DPathExpert(rng_seed=seed, steer_noise=0.035)
    if scenario == "contact_recovery":
        return Mesh3DRecoveryExpert(rng_seed=seed, steer_noise=0.025)
    raise ValueError(f"Unknown scenario: {scenario}")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_episode_camera_pair(env, expert, task: str, out_dir: Path, scenario: str, image_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    side_dir = out_dir / "frames" / "side"
    top_dir = out_dir / "frames" / "top"
    side_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    _obs, info = env.reset(options={"task": task})
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
        actions.append({"step": step, "action": action.astype(float).tolist(), "reward": reward})
        total_reward += reward
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


def append_samples(manifest_samples, episode_dir: Path, meta, image_mode: str):
    states = load_jsonl(episode_dir / "states.jsonl")
    actions = load_jsonl(episode_dir / "actions.jsonl")
    action_by_step = {item["step"]: item for item in actions}
    for state in states:
        step = state["step"]
        if step not in action_by_step:
            continue
        if image_mode == "camera_pair":
            images = {
                "side": str((episode_dir / "frames" / "side" / f"{step:06d}.png").as_posix()),
                "top": str((episode_dir / "frames" / "top" / f"{step:06d}.png").as_posix()),
            }
        else:
            images = {"debug": str((episode_dir / "frames" / f"{step:06d}.png").as_posix())}
        manifest_samples.append(
            {
                "episode": episode_dir.name,
                "task": meta["task"],
                "scenario": meta["scenario"],
                "step": step,
                "images": images,
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
                },
                "action": action_by_step[step]["action"],
                "reward": action_by_step[step]["reward"],
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Build a first VLA-style dataset from the 3D guidewire simulator.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_route_v1.json")
    parser.add_argument("--out", default="simulation_output/vla_dataset_v1")
    parser.add_argument("--episodes-per-combo", type=int, default=2)
    parser.add_argument("--scenarios", nargs="+", default=["normal", "noisy", "contact_recovery"])
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--image-mode", choices=["debug", "camera_pair"], default="camera_pair")
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mesh": args.mesh,
        "route_config": args.route_config,
        "episodes": [],
        "samples": [],
    }

    episode_index = 1
    for scenario in args.scenarios:
        for task in args.tasks:
            for i in range(args.episodes_per_combo):
                seed = args.seed + episode_index
                env = MeshGuidewire3DEnv(
                    Mesh3DEnvConfig(mesh_path=args.mesh, route_config_path=args.route_config),
                    seed=seed,
                )
                expert = make_expert(scenario, seed)
                episode_dir = out_root / f"episode_{episode_index:06d}_{scenario}_{task}"
                if args.no_render:
                    meta = collect_episode(env, expert, task, episode_dir, render=False, scenario=scenario)
                    image_mode = "debug"
                elif args.image_mode == "camera_pair":
                    meta = collect_episode_camera_pair(env, expert, task, episode_dir, scenario=scenario, image_size=args.image_size)
                    image_mode = "camera_pair"
                else:
                    meta = collect_episode(env, expert, task, episode_dir, render=True, scenario=scenario)
                    image_mode = "debug"
                meta["episode"] = episode_dir.name
                manifest["episodes"].append(meta)
                append_samples(manifest["samples"], episode_dir, meta, image_mode=image_mode)
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
