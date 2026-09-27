"""Audit completed BC/ACT logs; optionally render first-seed recorded actions.

No checkpoint loading, policy inference, optimizer, new benchmark or hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import run_diffusion_pusht_baseline as common

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_model(root, name, reference):
    directory = root / "simulation_output/pusht_bc_act_benchmark_v1" / name
    report, status = read(directory / "report.json"), read(directory / "status.json")
    episodes = [json.loads(line) for line in (directory / "per_episode.jsonl").read_text().splitlines()]
    require(episodes == report["episodes"], f"{name}: per-episode/report disagreement")
    require(report["status"] == status["status"] == "completed" and status["episodes_completed"] == 100,
            f"{name}: incomplete benchmark")
    require(report["stage"] == "benchmark" and report["model"] == name, "wrong benchmark identity")
    require(not report["training_started"] and report["optimizer_steps"] == 0
            and report["model_state_unchanged"] and report["environment_processors_verified_identity"],
            f"{name}: inference-only contract failed")
    summary = common.aggregate(episodes, "benchmark")
    require(all(report[key] == value for key, value in summary.items()), f"{name}: aggregate mismatch")
    require(summary["strict_protocol_pass"] and status["strict_protocol_pass"], f"{name}: protocol failed")
    common.verify_environment_parity(reference, report)
    smoke = read(root / "simulation_output/pusht_bc_act_smoke_v1" / name / "report.json")
    for key in ("checkpoint_binding", "source_files", "protocol", "protocol_sha256"):
        require(report[key] == smoke[key], f"{name}: smoke binding mismatch: {key}")
    require(smoke["strict_protocol_pass"] and smoke["reset_replay_exact"], f"{name}: incomplete smoke")
    dp_resets = {item["seed"]: item["reset_observation_sha256"] for item in reference["episodes"]}
    first_steps, current, pending, step_count, reset_count = [], None, None, 0, 0

    def finish():
        if current is None:
            return
        expected = episodes[current["episode"]]
        for key in ("steps", "is_success", "max_coverage", "terminated", "truncated"):
            require(current[key] == expected[key], f"{name}: episode outcome mismatch: {key}")
        require(current["done"] or current["steps"] == 300, f"{name}: early incomplete episode")
        require(expected["stopped_at_stage_cap"] == (not current["done"]), f"{name}: cap mismatch")
        require(pending is None, f"{name}: unexecuted action record")

    with (directory / "per_step.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            require(row["phase"] == "primary", f"{name}: unexpected extra replay")
            if row["kind"] == "reset":
                finish()
                require(row["episode"] == reset_count < 100 and row["seed"] == 100000 + reset_count,
                        f"{name}: reset schedule mismatch")
                expected = episodes[reset_count]
                require(row["observation_sha256"] == expected["reset_observation_sha256"] == dp_resets[row["seed"]],
                        f"{name}: initial observation differs from DP")
                current = dict(episode=reset_count, seed=row["seed"], steps=0, is_success=False,
                               max_coverage=0.0, terminated=False, truncated=False, done=False,
                               next_observation=row["observation_sha256"])
                reset_count += 1
                continue
            require(current is not None and not current["done"], f"{name}: step outside active episode")
            require(row["episode"] == current["episode"] and row["seed"] == current["seed"]
                    and row["step"] == current["steps"] + 1 <= 300, f"{name}: step sequence mismatch")
            require(row["observation_sha256"] == current["next_observation"], f"{name}: broken observation chain")
            action = row["native_action"]
            require(len(action) == 2 and all(math.isfinite(x) and 0 <= x <= 512 for x in action)
                    and row["native_action_out_of_bounds_elements"] == 0, f"{name}: invalid native action")
            step = row["step"]
            before = (1 - step) % 8
            queue = {"before": before, "after": (8 - step) % 8, "chunk_generated": before == 0} if name == "act" else {
                "before": 0, "after": 0, "chunk_generated": True}
            require(row["queue"] == queue, f"{name}: action queue mismatch")
            if row["kind"] == "action":
                require(pending is None, f"{name}: duplicate action")
                pending = row
                continue
            require(row["kind"] == "step" and pending is not None, f"{name}: missing action record")
            require(all(row[key] == value for key, value in pending.items() if key != "kind"),
                    f"{name}: executed/requested action disagreement")
            pending = None
            require(all(isinstance(row[k], bool) for k in ("is_success", "terminated", "truncated"))
                    and row["source"] in ("info", "final_info") and math.isfinite(row["reward"])
                    and math.isfinite(row["coverage"]) and 0 <= row["coverage"] <= 1,
                    f"{name}: invalid environment metric")
            current.update(steps=step, is_success=current["is_success"] or row["is_success"],
                           max_coverage=max(current["max_coverage"], row["coverage"]),
                           terminated=row["terminated"], truncated=row["truncated"],
                           done=row["terminated"] or row["truncated"], next_observation=row["next_observation_sha256"])
            step_count += 1
            if current["episode"] == 0:
                first_steps.append(row)
    finish()
    require(reset_count == 100 and step_count == report["environment_steps"] == report["environment_step_calls"]
            == sum(item["steps"] for item in episodes), f"{name}: execution count mismatch")
    return {**summary, "environment_steps": step_count, "runtime_seconds": report["runtime_seconds"],
            "log_audit_passed": True, "reset_parity_with_dp": True, "action_queue_verified": True}, first_steps, report


def replay_first_episode(first_steps, out):
    """Replay logged actions, not a second model evaluation or additional score."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from smoke_lerobot_public_envs import _make_environment

    font_path = next(path for path in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                                     Path("C:/Windows/Fonts/msyh.ttc")) if path.exists())
    font, title_font = (ImageFont.truetype(str(font_path), size) for size in (21, 25))
    canvas = Image.new("RGB", (1536, 982), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "BC / ACT：完整基准首个测试场景（相同 seed=100000）", font=title_font, fill="black")
    draw.text((12, 46), "按已记录动作重放；逐步核对观测和结果；不是重新推理或新增基准成绩", font=font, fill="black")
    replay_steps = 0
    for model_index, (name, rows) in enumerate(first_steps.items()):
        common._seed(100000, {"np": np, "torch": torch})
        config, vector = _make_environment("pusht", argparse.Namespace(episode_length=300))
        try:
            common.validate_env_kwargs(config.gym_kwargs)
            env = vector.envs[0]
            env.action_space.seed(100000)
            env.observation_space.seed(100000)
            raw, _ = env.reset(seed=100000)
            snapshots = [(0, common._render(env, np))]
            selected = {len(rows) // 3, 2 * len(rows) // 3, len(rows)}
            for row in rows:
                def obs_hash(value):
                    return hashlib.sha256(value["pixels"].tobytes() + value["agent_pos"].tobytes()).hexdigest()
                require(obs_hash(raw) == row["observation_sha256"], f"{name}: replay observation mismatch")
                raw, reward, terminated, truncated, info = env.step(np.asarray(row["native_action"], dtype=np.float32))
                actual = {"observation": obs_hash(raw), "terminated": bool(terminated),
                          "truncated": bool(truncated), "reward": float(reward)}
                expected = {"observation": row["next_observation_sha256"], "terminated": row["terminated"],
                            "truncated": row["truncated"], "reward": row["reward"]}
                require(actual == expected, f"{name}: replay step {row['step']} outcome mismatch: actual={actual}, expected={expected}")
                metrics = common.info_metrics(info, terminal=bool(terminated or truncated))
                require(all(row[key] == value for key, value in metrics.items()), f"{name}: replay info mismatch")
                replay_steps += 1
                if row["step"] in selected:
                    snapshots.append((row["step"], common._render(env, np)))
            top = 92 + model_index * 440
            succeeded = any(row["is_success"] for row in rows)
            draw.text((12, top), f'{name.upper()}｜该场景{"成功" if succeeded else "未成功"}｜最大覆盖率 {max(r["coverage"] for r in rows):.3f}',
                      font=font, fill="black")
            for column, (step, frame) in enumerate(snapshots):
                draw.text((column * 384 + 12, top + 29), f"步骤 {step}", font=font, fill="black")
                canvas.paste(Image.fromarray(frame), (column * 384, top + 56))
        finally:
            vector.close()
    destination = out / "first_seed_replay_zh.png"
    canvas.save(destination)
    return {"path": str(destination), "seed": 100000, "recorded_action_replay_steps": replay_steps,
            "all_step_observations_and_info_exact": True, "model_forward_calls": 0, "visual_status": "not_viewed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_bc_act_benchmark_audit_v1")
    parser.add_argument("--replay-first", action="store_true")
    args = parser.parse_args()
    require(not args.out.exists(), f"preserve existing output: {args.out}")
    started = time.perf_counter()
    reference = read(args.root / "simulation_output/diffusion_pusht_benchmark_v1/report.json")
    require(reference["status"] == "completed" and reference["strict_protocol_pass"], "DP reference incomplete")
    dp_summary = common.aggregate(reference["episodes"], "benchmark")
    require(all(reference[key] == value for key, value in dp_summary.items()), "DP aggregate mismatch")
    models, first_steps, reports = {}, {}, {}
    for name in ("bc", "act"):
        models[name], first_steps[name], reports[name] = audit_model(args.root, name, reference)
    require(reports["bc"]["protocol"] == reports["act"]["protocol"], "BC/ACT protocol mismatch")
    bc, act = models["bc"], models["act"]
    delta_coverage = act["mean_episode_max_coverage"] - bc["mean_episode_max_coverage"]
    paired = {"success_change_percentage_points": 100 * (act["observed_success_rate"] - bc["observed_success_rate"]),
              "relative_success_change": None, "relative_success_note": "undefined because BC success rate is zero",
              "mean_max_coverage_change": delta_coverage,
              "mean_max_coverage_relative_change": delta_coverage / bc["mean_episode_max_coverage"],
              "both_success": 0, "bc_only_success": 0, "act_only_success": 0, "both_failure": 0}
    for left, right in zip(reports["bc"]["episodes"], reports["act"]["episodes"]):
        key = {(True, True): "both_success", (True, False): "bc_only_success",
               (False, True): "act_only_success", (False, False): "both_failure"}[(left["is_success"], right["is_success"])]
        paired[key] += 1
    args.out.mkdir(parents=True, exist_ok=False)
    visual = replay_first_episode(first_steps, args.out) if args.replay_first else None
    report = {"schema": "pusht_bc_act_benchmark_audit_v1", "status": "passed", "models": models,
              "paired_act_minus_bc": paired, "dp_external_reference": dp_summary,
              "source_root": str(args.root / "simulation_output/pusht_bc_act_benchmark_v1"),
              "visual_replay": visual, "elapsed_seconds": time.perf_counter() - started,
              "checkpoints_loaded": 0, "optimizer_updates": 0, "new_benchmark_episodes": 0,
              "scope": "single-training-seed clean Push-T; DP is external pretrained, not same-data/budget; no guidewire/real-system claim"}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
