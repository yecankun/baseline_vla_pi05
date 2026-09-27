"""Offline B4b audit: preserve observed results, fail strict protocol separately.

No policy, simulator, training, or LeRobot imports. Original run files are read
only. All generated reports and decoded-frame sheets go to a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def inventory(root: Path):
    return {p.relative_to(root).as_posix(): {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(root.rglob("*")) if p.is_file()}


def audit_trace(events: list[dict]) -> tuple[list[dict], dict]:
    """Check actual applied payload -> reset -> completion chains, not a fake schedule."""
    active = pending = None
    complete, starts, automatic, payloads = [], [], [], {}
    for event in events:
        kind = event["kind"]
        if kind == "state_applied":
            if pending is not None:
                raise ValueError("unpaired applied-state payload")
            digest = event["state_sha256"]
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("malformed payload hash")
            key = (event["task_id"], event["init_state_index"])
            if key in payloads and payloads[key] != digest:
                raise ValueError("same task/init index changed state payload")
            if any(t == key[0] and i != key[1] and h == digest for (t, i), h in payloads.items()):
                raise ValueError("distinct init indices share one payload")
            payloads[key] = digest
            pending = event
        elif kind == "reset":
            if pending is None or any(event[k] != pending[k] for k in ("task_id", "seed")):
                raise ValueError("reset has no matching actual payload")
            if event["applied_init_state_index"] != pending["init_state_index"]:
                raise ValueError("reset index differs from actual payload")
            if event["reset_kind"] == "episode_start":
                if active is not None or type(event["seed"]) is not int:
                    raise ValueError("overlapping episode or absent seed")
                active = {"task_id": event["task_id"], "seed": event["seed"],
                          "init_state_index": event["applied_init_state_index"],
                          "state_sha256": pending["state_sha256"], "automatic_resets": []}
                starts.append(dict(active))
            elif event["reset_kind"] == "automatic":
                if active is None or event["seed"] is not None or event["task_id"] != active["task_id"]:
                    raise ValueError("automatic reset outside active episode")
                automatic.append(event)
                active["automatic_resets"].append(event["applied_init_state_index"])
            else:
                raise ValueError("unknown reset type")
            pending = None
        elif kind == "episode_complete":
            if pending is not None or active is None:
                raise ValueError("completion without a paired start")
            if any(event[k] != active[k] for k in ("task_id", "seed", "init_state_index")):
                raise ValueError("completion/start identity mismatch")
            if type(event["success"]) is not bool or type(event["steps"]) is not int or not 1 <= event["steps"] <= 280:
                raise ValueError("invalid success/step count")
            low, high = event["action_min"], event["action_max"]
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("invalid action range")
            complete.append({**event, "state_sha256": active["state_sha256"],
                             "automatic_resets": list(active["automatic_resets"])})
            active = None
        else:
            raise ValueError(f"unknown trace event: {kind}")
    if active is not None or pending is not None:
        raise ValueError("unfinished trace")
    expected = [(task, 1000 + index, index) for task in range(10) for index in range(10)]
    actual = [(row["task_id"], row["seed"], row["init_state_index"]) for row in complete]
    if actual != expected:
        raise ValueError("actual completed task/seed/init schedule differs from frozen 100 episodes")
    return complete, {"events": len(events), "seeded_starts": len(starts),
                      "automatic_resets": len(automatic), "unique_task_index_payloads": len(payloads)}


def decode_video(path: Path, expected_steps: int):
    import cv2
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unreadable video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pinned lerobot_eval.py slices ep_frames[:done_index+1]. Since frame 0 is
    # the reset observation, its saved video has S frames for S actions, not
    # S+1: the terminal post-action render is omitted (including auto-reset).
    pick = set(round(i * (expected_steps - 1) / 7) for i in range(8))
    frames, count, size = {}, 0, None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            current_size = [int(bgr.shape[1]), int(bgr.shape[0])]
            if size is not None and current_size != size:
                raise ValueError("video dimensions change inside episode")
            size = current_size
            if count in pick:
                frames[count] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            count += 1
    finally:
        capture.release()
    if count != declared or count != expected_steps:
        raise ValueError(f"video frame count {count}/{declared} differs from pinned saved-frame count {expected_steps}: {path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid video FPS")
    return {"decoded_frames": count, "size_wh": size, "fps": fps,
            "terminal_post_action_frame_saved": False}, frames


def make_sheet(rows: list[dict], out: Path, *, font_path: Path, title: str, columns: int = 4):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    title_font = ImageFont.truetype(str(font_path), 25)
    label_font = ImageFont.truetype(str(font_path), 19)
    panels = [(row, index, frame) for row in rows for index, frame in sorted(row["frames"].items())]
    cell_w, cell_h, header = 420, 260, 90
    canvas = Image.new("RGB", (columns * cell_w, header + math.ceil(len(panels) / columns) * cell_h), "#f5f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), title, font=title_font, fill="#162033")
    draw.text((16, 50), "原视频等间隔取样｜保持宽高比｜末帧不是最后动作后的终态", font=label_font, fill="#445066")
    for panel, (row, index, frame) in enumerate(panels):
        x, y = panel % columns * cell_w, header + panel // columns * cell_h
        label = f"任务 {row['task_id']} / seed {row['seed']} / 帧 {index}"
        draw.text((x + 9, y + 5), label, font=label_font, fill="#172135")
        image = ImageOps.contain(Image.fromarray(frame), (cell_w - 16, cell_h - 40))
        canvas.paste(image, (x + (cell_w - image.width) // 2, y + 35))
    canvas.save(out)
    return {"file": out.name, "sha256": sha256(out), "visual_status": "not_viewed",
            "panels": len(panels), "size_wh": list(canvas.size)}


def audit(args):
    root, out = args.result_root.resolve(), args.out.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("audit output must be new and outside the immutable run directory")
    before = inventory(root)
    status, info, manifest, env, policy = [load(root / name) for name in
        ("status.json", "eval_info.json", "run_manifest.json", "effective_env_config.json", "effective_policy_config.json")]
    protocol, preflight = load(args.protocol), load(args.preflight)
    events = [json.loads(line) for line in (root / "episode_trace.jsonl").read_text().splitlines() if line.strip()]
    completed, trace_info = audit_trace(events)
    task_rows = info["per_task"]
    checks = {
        "status_completed": status["status"] == "completed" and status["exit_code"] == 0,
        "eval_hash_matches_status": sha256(root / "eval_info.json") == status["eval_info_sha256"],
        "actual_payload_seeded_reset_complete_chain": len(completed) == 100,
        "protocol_hash_matches_prerun": manifest["protocol_sha256"] == preflight["protocol_sha256"] == sha256(args.protocol),
        "all_checkpoint_files_match_prerun": manifest["checkpoint_hashes"] == preflight["checkpoint_hashes"],
        "model_hash_matches_frozen_route": manifest["checkpoint_hashes"]["model.safetensors"] == protocol["checkpoint_routes"]["pi05_primary"]["model_sha256"],
        "wrapper_hash_matches_prerun_and_local": manifest["wrapper_sha256"] == preflight["wrapper_sha256"] == sha256(args.runner),
        "pinned_versions_match": manifest["versions"] == preflight["versions"] and all(
            protocol["software_contract"][key] == value for key, value in manifest["versions"].items()),
        "ten_tasks_exactly": len(task_rows) == 10 and sorted(t["task_id"] for t in task_rows) == list(range(10)),
        "env_parameters": env["task"] == "libero_spatial" and env["task_ids"] == list(range(10)) and
            env["episode_length"] == 280 and env["control_mode"] == "relative" and env["init_states"] is True and env["max_parallel_tasks"] == 1,
        "policy_state_action_dimensions": policy["input_features"]["observation.state"]["shape"] == [8] and policy["output_features"]["action"]["shape"] == [7],
        "checkpoint_inference_configuration": policy["n_action_steps"] == 10 and policy["num_inference_steps"] == 10 and policy["chunk_size"] == 50,
        "synchronous_one_env_argv": all(a in manifest["official_eval_argv"] for a in
            ("--eval.batch_size=1", "--eval.use_async_envs=false", "--eval.n_episodes=10", "--seed=1000")),
        "no_duplicate_cli_options": len(manifest["official_eval_argv"]) == len({a.split("=", 1)[0] for a in manifest["official_eval_argv"]}),
        "automatic_reset_matches_observed_terminal_outcome": all(
            r["automatic_resets"] == ([r["init_state_index"] + 1] if r["success"] else []) for r in completed),
    }
    by_task = []
    videos, failing_frames, reference_frames = [], [], []
    expected_paths = set()
    for task_id in range(10):
        selected = [t for t in task_rows if t["task_id"] == task_id]
        if len(selected) != 1:
            raise ValueError("task metrics absent/duplicated")
        task = selected[0]
        rows = [r for r in completed if r["task_id"] == task_id]
        metrics = task["metrics"]
        checks[f"task_{task_id}_success_vector"] = task["task_group"] == "libero_spatial" and metrics["successes"] == [r["success"] for r in rows]
        checks[f"task_{task_id}_reward_vectors"] = len(metrics["sum_rewards"]) == len(metrics["max_rewards"]) == 10 and all(
            math.isfinite(float(x)) for x in metrics["sum_rewards"] + metrics["max_rewards"])
        if len(metrics["video_paths"]) != 10:
            raise ValueError("missing video mappings")
        for i, row in enumerate(rows):
            relative = f"videos/libero_spatial_{task_id}/eval_episode_{i}.mp4"
            if not metrics["video_paths"][i].endswith("/" + relative):
                raise ValueError("video mapping disagrees with episode order")
            path = root / relative
            expected_paths.add(relative)
            video, frames = decode_video(path, row["steps"])
            videos.append({"task_id": task_id, "seed": row["seed"], "init_state_index": i,
                           "success": row["success"], "file": relative, "sha256": sha256(path), **video})
            if not row["success"]:
                failing_frames.append({**row, "frames": frames})
            elif i == 0:
                reference_frames.append({**row, "frames": {max(frames): frames[max(frames)]}})
        count = sum(r["success"] for r in rows)
        by_task.append({"task_id": task_id, "successes": count, "episodes": 10, "success_rate_percent": count * 10,
                        "failed_seeds": [r["seed"] for r in rows if not r["success"]],
                        "steps": [r["steps"] for r in rows]})
    total = sum(r["success"] for r in completed)
    checks["video_paths_exactly_100"] = expected_paths == {p.relative_to(root).as_posix() for p in root.rglob("*.mp4")}
    checks["all_100_videos_fully_decoded"] = len(videos) == 100
    checks["aggregate_arithmetic"] = info["overall"]["n_episodes"] == status["episodes"] == 100 and math.isclose(info["overall"]["pc_success"], total) and status["success_rate_percent"] == total
    low, high = min(r["action_min"] for r in completed), max(r["action_max"] for r in completed)
    out_of_bounds = [r for r in completed if r["action_min"] < -1 or r["action_max"] > 1]
    declared_shape = protocol["observation_contract"]["image_features"]["observation.images.image"]
    native_policy_shapes = all(policy["input_features"][key]["shape"] == shape for key, shape in protocol["observation_contract"]["image_features"].items())
    strict = {
        "native_actions_all_within_declared_bounds": len(out_of_bounds) == 0,
        "saved_main_view_resolution_matches_declared_image_features": all(
            video["size_wh"] == list(reversed(declared_shape[1:])) for video in videos),
        "policy_declared_image_shapes_match": native_policy_shapes,
    }
    after = inventory(root)
    checks["original_artifacts_unchanged"] = before == after
    if not all(checks.values()):
        raise ValueError(f"artifact/result checks failed: {[k for k,v in checks.items() if not v]}")
    out.mkdir(parents=True, exist_ok=False)
    sheets = []
    for row in failing_frames:
        path = out / f"failure_task{row['task_id']}_seed{row['seed']}_zh.png"
        sheets.append(make_sheet([row], path, font_path=args.font,
                                 title=f"B4b 失败视频取样｜任务 {row['task_id']}｜seed {row['seed']}｜{row['steps']} 步"))
    if reference_frames:
        sheets.append(make_sheet(reference_frames, out / "successful_reference_finals_zh.png", font_path=args.font,
                                 title="B4b 成功参考末帧｜每任务第 0 次（成功者）", columns=5))
    report = {
        "schema": "pi05_libero_b4b_offline_audit_v1", "status": "audited_with_protocol_deviations" if not all(strict.values()) else "audited",
        "artifact_and_schedule_checks": checks, "strict_protocol_checks": strict,
        "strict_frozen_protocol_passed": all(strict.values()), "observed_successes": total, "episodes": 100,
        "observed_success_rate_percent": total, "task_macro_success_rate_percent": sum(t["success_rate_percent"] for t in by_task) / 10,
        "per_task": by_task, "trace": trace_info, "runtime_overall": {k:v for k,v in info["overall"].items() if k != "video_paths"},
        "actions": {"min": low, "max": high, "episodes_with_any_bound_violation": len(out_of_bounds),
                    "violating_episode_ids": [[r["task_id"], r["seed"]] for r in out_of_bounds],
                    "per_dimension_or_per_step_violation_count_available": False},
        "images": {"env_config_declared_resolution_hw": [env["observation_height"], env["observation_width"]],
                   "saved_main_view_resolution_wh": sorted({tuple(v["size_wh"]) for v in videos}),
                   "policy_feature_declared_chw": policy["input_features"]["observation.images.image"]["shape"],
                   "policy_internal_resolution_hw": policy["image_resolution"], "actual_policy_boundary_tensor_logged": False,
                   "config_metadata_is_not_runtime_tensor_evidence": True},
        "failed_episodes": [{k:v for k,v in r.items() if k != "frames"} for r in failing_frames],
        "videos": videos, "visual_sheets": sheets, "original_inventory": before,
        "policy_loaded": False, "simulation_steps": 0, "optimizer_steps": 0,
        "claim_scope": "observed pinned PI05 LIBERO-Spatial 10x10; not strict protocol pass or published-score equivalence",
        "limitations": ["init hashes identify runtime payloads, not execution-time hashes of every BDDL/asset file",
                        "video frames are representative visual evidence, not a replacement success classifier",
                        "pinned evaluator saves S video frames for S actions; final post-action/terminal frame is absent",
                        "env config declares360 but saved raw-main-view videos are256; config values alone are not executed tensor shapes",
                        "raw min/max cannot identify which action dimensions or exact steps saturated",
                        "global policy RNG remained upstream; later noisy rollouts need not share all action-noise draws",
                        "no project guidewire, world-model improvement, real-system or formal-data conclusion"],
        "audit_code_sha256": sha256(Path(__file__)),
    }
    write(out / "report.json", report)
    print(json.dumps({"status": report["status"], "successes": total, "episodes": 100,
                      "checks": len(checks), "strict_protocol_passed": all(strict.values()),
                      "out_of_bounds_episodes": len(out_of_bounds), "out": str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("docs/libero-spatial-score-protocol-v1.json"))
    parser.add_argument("--preflight", type=Path, default=Path("simulation_output/pi05_libero_spatial_b4b_preflight_v1.json"))
    parser.add_argument("--runner", type=Path, default=Path("tools/run_pi05_libero_b4b.py"))
    parser.add_argument("--font", type=Path, required=True)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
