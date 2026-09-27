import argparse
import json
from pathlib import Path

import cv2

from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, Mesh3DPathExpert, Mesh3DRecoveryExpert, MeshGuidewire3DEnv


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_episode(env: MeshGuidewire3DEnv, expert: Mesh3DPathExpert, task: str, out_dir: Path, render: bool, scenario: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if render:
        frames_dir.mkdir(exist_ok=True)

    _obs, info = env.reset(options={"task": task})
    states = []
    actions = []
    total_reward = 0.0
    while True:
        obs_dict = info["obs_dict"]
        action = expert.act(env)
        if render:
            cv2.imwrite(str(frames_dir / f"{obs_dict['step']:06d}.png"), env.render())
        _next_obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs_dict)
        actions.append({"step": obs_dict["step"], "action": action.astype(float).tolist(), "reward": reward})
        total_reward += reward
        if terminated or truncated:
            final_obs = info["obs_dict"]
            states.append(final_obs)
            if render:
                cv2.imwrite(str(frames_dir / f"{final_obs['step']:06d}.png"), env.render())
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
    parser = argparse.ArgumentParser(description="Collect episodes from 3D STL-derived guidewire environment.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--out", default="simulation_output/mesh_3d")
    parser.add_argument("--task", choices=["left", "right", "both"], default="both")
    parser.add_argument("--scenario", choices=["normal", "noisy", "contact_recovery"], default="normal")
    parser.add_argument("--contact-side", choices=["auto", "left", "right", "upper", "lower"], default="auto")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    env = MeshGuidewire3DEnv(Mesh3DEnvConfig(mesh_path=args.mesh, route_config_path=args.route_config), seed=args.seed)
    if args.scenario == "normal":
        expert = Mesh3DPathExpert(rng_seed=args.seed, steer_noise=0.0)
    elif args.scenario == "noisy":
        expert = Mesh3DPathExpert(rng_seed=args.seed, steer_noise=0.035)
    else:
        expert = Mesh3DRecoveryExpert(rng_seed=args.seed, side=args.contact_side)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    env.save_debug_config(str(out_root / "mesh_3d_debug_config.json"))

    tasks = ["left", "right"] if args.task == "both" else [args.task]
    summaries = []
    for episode_idx in range(args.episodes):
        task = tasks[episode_idx % len(tasks)]
        episode_dir = out_root / f"episode_{episode_idx + 1:06d}"
        meta = collect_episode(env, expert, task, episode_dir, render=not args.no_render, scenario=args.scenario)
        summaries.append({"episode": episode_dir.name, **meta})
        print(
            f"{episode_dir.name}: task={task}, success={meta['success']}, "
            f"steps={meta['steps']}, failure={meta['failure_reason']}"
        )

    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(summaries)} 3D mesh episodes to {out_root}")


if __name__ == "__main__":
    main()
