from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.collect_dual_arm_3d import collect_episode
from simulation.dual_arm_guided_start_env import (
    GuidedStartDualArmEnvConfig,
    GuidedStartDualArmExpert,
    GuidedStartDualArmGuidewire3DEnv,
)


def add_episode_samples(manifest: dict, episode_dir: Path, meta: dict) -> None:
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
                "mode": "guided_start_dual_arm",
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
                    "boundary_projection_count": state.get("boundary_projection_count", 0),
                    "last_boundary_projection": state.get("last_boundary_projection", False),
                },
                "action": action_by_step[step]["action"],
                "reward": action_by_step[step]["reward"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect episodes that start directly in the Elite-guided stage.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_route_v1.json")
    parser.add_argument("--out", default="simulation_output/dual_arm_guided_start_dataset_v1")
    parser.add_argument("--episodes-per-task", type=int, default=4)
    parser.add_argument("--tasks", nargs="+", default=["left", "right"])
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--start-min", type=float, default=0.50)
    parser.add_argument("--start-max", type=float, default=0.70)
    parser.add_argument("--noise", type=float, default=0.006)
    parser.add_argument("--steer-noise", type=float, default=0.006)
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mesh": args.mesh,
        "route_config": args.route_config,
        "mode": "guided_start_dual_arm",
        "start_fraction_range": [args.start_min, args.start_max],
        "episodes": [],
        "samples": [],
    }

    episode_index = 1
    for task in args.tasks:
        for repeat in range(args.episodes_per_task):
            seed = args.seed + episode_index
            rng = np.random.default_rng(seed)
            config = GuidedStartDualArmEnvConfig(
                mesh_path=args.mesh,
                route_config_path=args.route_config,
                max_steps=args.max_steps,
                guided_start_min_fraction=args.start_min,
                guided_start_max_fraction=args.start_max,
                guided_start_lateral_noise=args.noise,
            )
            env = GuidedStartDualArmGuidewire3DEnv(config, seed=seed)
            expert = GuidedStartDualArmExpert(rng_seed=seed, steer_noise=args.steer_noise)
            path_len = len(env.paths[task]) - 1
            start_fraction = float(rng.uniform(args.start_min, args.start_max))
            start_progress = start_fraction * path_len
            episode_dir = out_root / f"episode_{episode_index:06d}_guided_start_{task}"
            meta = collect_episode(
                env,
                expert,
                task,
                episode_dir,
                scenario="guided_start",
                image_size=args.image_size,
                start_progress=start_progress,
            )
            meta["episode"] = episode_dir.name
            meta["start_progress"] = start_progress
            meta["start_fraction"] = start_fraction
            meta["repeat"] = repeat
            manifest["episodes"].append(meta)
            add_episode_samples(manifest, episode_dir, meta)
            print(
                f"{episode_dir.name}: task={task}, start={start_fraction:.3f}, "
                f"success={meta['success']}, steps={meta['steps']}, failure={meta['failure_reason']}"
            )
            episode_index += 1

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved dataset manifest to {manifest_path}")
    print(f"Episodes: {len(manifest['episodes'])}, samples: {len(manifest['samples'])}")


if __name__ == "__main__":
    main()
