import argparse
import json
import os
from pathlib import Path

import cv2

from simulation.guidewire_env import CenterlineExpert, Guidewire2DEnv


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_episode(env: Guidewire2DEnv, expert: CenterlineExpert, task: str, out_dir: Path, render: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if render:
        frames_dir.mkdir(exist_ok=True)

    obs = env.reset(task=task)
    states = []
    actions = []
    total_reward = 0.0

    while True:
        action = expert.act(obs, env)
        if render:
            cv2.imwrite(str(frames_dir / f"{obs['step']:06d}.png"), env.render())

        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        states.append(obs)
        actions.append({"step": obs["step"], "action": action, "reward": reward})
        obs = next_obs
        if done:
            if render:
                cv2.imwrite(str(frames_dir / f"{obs['step']:06d}.png"), env.render())
            states.append(obs)
            break

    meta = {
        "task": task,
        "instruction": env.instruction,
        "steps": obs["step"],
        "success": obs["success"],
        "failure_reason": obs["failure_reason"],
        "total_reward": total_reward,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "states.jsonl", states)
    write_jsonl(out_dir / "actions.jsonl", actions)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Collect MVP1 2D guidewire simulation episodes.")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--out", type=str, default="simulation_output/mvp1")
    parser.add_argument("--task", choices=["left", "right", "both"], default="both")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    env = Guidewire2DEnv(seed=args.seed)
    expert = CenterlineExpert()
    tasks = ["left", "right"] if args.task == "both" else [args.task]
    summaries = []

    for episode_idx in range(args.episodes):
        task = tasks[episode_idx % len(tasks)]
        episode_dir = out_root / f"episode_{episode_idx + 1:06d}"
        meta = collect_episode(env, expert, task, episode_dir, render=not args.no_render)
        summaries.append({"episode": episode_dir.name, **meta})
        print(
            f"{episode_dir.name}: task={task}, success={meta['success']}, "
            f"steps={meta['steps']}, failure={meta['failure_reason']}"
        )

    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(summaries)} episodes to {out_root}")


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")
    main()

