"""Verify and visualize three fixed recorded-action replays, without a policy."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/diffusion_pusht_benchmark_v1"
OUTPUT = ROOT / "simulation_output/diffusion_pusht_benchmark_replay_v1"
SEEDS = (100000, 100049, 100099)  # First, middle and last; never selected by outcome.
PINS = {
    "report.json": "c1d4c4e582af1116ec41d554c8ae8be76480cee73488acff4471097bafe314b9",
    "per_episode.jsonl": "d27f2869b7c482d9fc4358314ad7f488d862dc721a4ac7aedc3a1a3fcd159c07",
    "per_step.jsonl": "45b2c591d72eab18b88c4027514329e94f940c6a4ff132b9a601072013d122a7",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def inputs():
    for name, digest in PINS.items():
        require(sha(SOURCE / name) == digest, f"Source hash changed: {name}")
    report = json.loads((SOURCE / "report.json").read_text())
    episodes = [json.loads(line) for line in (SOURCE / "per_episode.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (SOURCE / "per_step.jsonl").read_text().splitlines()]
    require(report["status"] == "completed" and report["strict_protocol_pass"], "Source not a completed strict pass")
    require([item["seed"] for item in episodes] == list(range(100000, 100100)), "Seed plan changed")
    require(episodes == report["episodes"], "Episode sidecar differs")
    return report, episodes, rows


def draw_sheet(panels, output):
    from PIL import Image, ImageDraw, ImageFont
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title_font = ImageFont.truetype(font_path, 23)
    label_font = ImageFont.truetype(font_path, 17)
    panel_size, row_height, header = 384, 425, 105
    sheet = Image.new("RGB", (3 * panel_size, header + 3 * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 8), "Diffusion Policy / Push-T 基准核验：已记录动作回放", font=title_font, fill="black")
    draw.text((14, 43), "固定第 1 / 50 / 100 回合；不调用模型，不增加评分样本", font=label_font, fill="black")
    draw.text((14, 70), "列：初始状态 / 最高覆盖状态 / 最终状态；所有步骤数值与原记录核对", font=label_font, fill="black")
    for row_index, episode in enumerate(panels):
        for column, (label, item) in enumerate(zip(("初始", "最高覆盖", "最终"), episode["frames"])):
            x, y = column * panel_size, header + row_index * row_height
            text = f"seed {episode['seed']} | {label} | 步骤 {item['step']}"
            draw.text((x + 7, y + 4), text, font=label_font, fill="black")
            sheet.paste(Image.fromarray(item["frame"]), (x, y + 36))
    sheet.save(output)
    return {"path": str(output), "sha256": sha(output), "visual_status": "not_viewed", "font": font_path}


def run(output=OUTPUT):
    started = time.monotonic()
    report, episodes, rows = inputs()
    output.mkdir(exist_ok=False)
    result = {"status": "running", "selected_seeds": list(SEEDS), "selection": "fixed first/middle/last",
              "source_sha256": PINS, "script_sha256": sha(__file__), "policy_loads": 0,
              "policy_forwards": 0, "optimizer_steps": 0, "new_score_episodes": 0, "episodes": [],
              "replayed_environment_steps": 0}
    vector = None
    try:
        import numpy as np
        from smoke_lerobot_public_envs import _make_environment, _package_versions
        from run_diffusion_pusht_baseline import canonical_hash, validate_env_kwargs, info_metrics
        require(_package_versions() == report["package_versions"], "Environment package drift")
        for key, function in (("_make_environment", _make_environment),):
            require(sha(inspect.getfile(function)) == report["source_files"][key]["sha256"], "Environment helper drift")
        config, vector = _make_environment("pusht", argparse.Namespace(episode_length=300))
        validate_env_kwargs(config.gym_kwargs)
        env = vector.envs[0]
        require(canonical_hash(inspect.getsource(type(env.unwrapped).step)) == report["environment_step_source_sha256"], "Native step drift")
        panels = []
        for seed in SEEDS:
            recorded = next(item for item in episodes if item["seed"] == seed)
            steps = [row for row in rows if row["seed"] == seed and row["kind"] == "step"]
            reset_row = next(row for row in rows if row["seed"] == seed and row["kind"] == "reset")
            require(len(steps) == recorded["steps"], "Trace count differs")
            best_step = max(steps, key=lambda row: row["coverage"])["step"]
            random.seed(seed)
            np.random.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            raw, _ = env.reset(seed=seed)
            digest = hashlib.sha256(np.asarray(raw["pixels"]).tobytes() + np.asarray(raw["agent_pos"]).tobytes()).hexdigest()
            require(digest == reset_row["observation_sha256"] == recorded["reset_observation_sha256"], "Initial observation differs")
            frames = {0: {"step": 0, "frame": np.asarray(env.render()).copy()}}
            differences = []
            max_reward_error = max_coverage_error = 0.0
            for index, expected in enumerate(steps, 1):
                require(expected["step"] == index, "Noncontiguous source trace")
                action = np.asarray(expected["native_action"], dtype=np.float32)
                require(action.shape == (2,) and np.isfinite(action).all(), "Invalid source action")
                _, reward, terminated, truncated, info = env.step(action)
                result["replayed_environment_steps"] += 1
                metrics = info_metrics(info, terminal=bool(terminated or truncated))
                reward_error = abs(float(reward) - expected["reward"])
                coverage_error = abs(metrics["coverage"] - expected["coverage"])
                max_reward_error = max(max_reward_error, reward_error)
                max_coverage_error = max(max_coverage_error, coverage_error)
                if reward_error or coverage_error:
                    differences.append({"step": index, "source_reward": expected["reward"], "replay_reward": float(reward),
                                        "source_coverage": expected["coverage"], "replay_coverage": metrics["coverage"]})
                require(metrics["is_success"] == expected["is_success"] and bool(terminated) == expected["terminated"]
                        and bool(truncated) == expected["truncated"], "Success/done replay differs")
                require(not (terminated or truncated) or index == len(steps), "Steps after done")
                if index in (best_step, len(steps)):
                    frames[index] = {"step": index, "frame": np.asarray(env.render()).copy()}
            panels.append({"seed": seed, "frames": [frames[0], frames[best_step], frames[len(steps)]]})
            result["episodes"].append({"seed": seed, "steps": len(steps), "best_step": best_step,
                                       "source_success": recorded["is_success"], "initial_observation_exact": True,
                                       "all_success_done_exact": True,
                                       "all_rewards_coverage_success_done_exact": not differences,
                                       "numeric_mismatch_steps": len(differences),
                                       "max_abs_reward_error": max_reward_error, "max_abs_coverage_error": max_coverage_error,
                                       "first_numeric_difference": differences[0] if differences else None})
        result["visual_artifact"] = draw_sheet(panels, output / "recorded_action_replay_zh.png")
        result["status"] = ("passed" if all(e["all_rewards_coverage_success_done_exact"] for e in result["episodes"])
                            else "completed_with_numeric_differences")
    except Exception:
        result.update(status="failed", traceback=traceback.format_exc())
    finally:
        if vector is not None:
            vector.close()
        result["runtime_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("passed", "completed_with_numeric_differences") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--attempt", help="Explicit new diagnostic suffix; preserve all previous attempts")
    args = parser.parse_args()
    if args.attempt is not None and not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", args.attempt):
        parser.error("Invalid diagnostic attempt suffix")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if not args.execute:
        inputs()
        print(json.dumps({"status": "preflight_passed", "source_sha256": PINS, "selected_seeds": SEEDS,
                          "environment_constructed": False, "policy_loads": 0}))
        return 0
    return run(OUTPUT.with_name(OUTPUT.name + "_" + args.attempt) if args.attempt else OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
