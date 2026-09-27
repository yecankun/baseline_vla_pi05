from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.collect_formal_tip_line_guidance import build_env  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in str(text).replace(",", " ").split() if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in str(text).replace(",", " ").split() if item.strip()]


def parse_optional_float_list(text: str | None, fallback: list[float | None]) -> list[float | None]:
    if text is None or not str(text).strip():
        return fallback
    values: list[float | None] = []
    for item in str(text).replace(",", " ").split():
        token = item.strip().lower()
        if token in {"none", "default", "current"}:
            values.append(None)
        else:
            values.append(float(token))
    return values


def red_tip_centroid(image: np.ndarray) -> tuple[float | None, float | None, int]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 70]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 80, 70]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, None, 0
    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        candidates.append((area, label))
    if not candidates:
        return None, None, 0
    area, label = max(candidates)
    cx, cy = centroids[label]
    return float(cx), float(cy), int(area)


def put_label(image: np.ndarray, text: str, point: tuple[float | None, float | None] | None = None) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if point is not None and point[0] is not None and point[1] is not None:
        cv2.circle(out, (int(round(point[0])), int(round(point[1]))), 5, (0, 255, 255), 2)
    return out


def make_sheet(before: np.ndarray, after: np.ndarray, label: str, before_pt: tuple[float | None, float | None], after_pt: tuple[float | None, float | None]) -> np.ndarray:
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
    before_labeled = put_label(before, f"{label} before", before_pt)
    after_labeled = put_label(after, f"{label} after", after_pt)
    diff = cv2.absdiff(before, after)
    diff_labeled = put_label(cv2.addWeighted(before, 0.5, diff, 2.0, 0.0), f"{label} diff")
    return np.concatenate([before_labeled, after_labeled, diff_labeled], axis=1)


def action_for_feed(env: Any, piper_cmd: float) -> dict[str, Any]:
    step_command = int(np.sign(float(piper_cmd))) if abs(float(piper_cmd)) > 0.05 else 0
    return {
        "piper_feed": float(piper_cmd),
        "piper_step_command": step_command,
        "elite_joints": env._robot_joint_dict("elite", env._robot_joint_vector("elite")),
    }


def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    sheets_dir = out_dir / "contact_sheets"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    piper_cmds = parse_float_list(args.piper_cmds)
    piper_advection_scales = parse_float_list(args.piper_advection_scales)
    commanded_progress_gains = parse_optional_float_list(args.commanded_progress_gains, [None])
    piper_forces = parse_optional_float_list(args.piper_forces, [None])
    primitive_steps_values = parse_int_list(args.primitive_steps)
    bursts = parse_int_list(args.bursts)
    for task in args.tasks:
        for piper_advection_scale in piper_advection_scales:
            for commanded_progress_gain in commanded_progress_gains:
                for piper_force in piper_forces:
                    for primitive_steps in primitive_steps_values:
                        for seed_offset in range(args.repeats):
                            seed = int(args.seed + seed_offset)
                            env = build_env(args, seed=seed)
                            env.config.piper_advection_scale = float(piper_advection_scale)
                            env.config.piper_primitive_steps = max(int(primitive_steps), 1)
                            env.config.piper_primitive_feed_value = 1.0
                            env.config.piper_primitive_retract_value = 1.0
                            if args.advance_step is not None:
                                env.config.advance_step = float(args.advance_step)
                            if commanded_progress_gain is not None:
                                env.config.commanded_progress_gain = float(commanded_progress_gain)
                            if piper_force is not None:
                                env.config.piper_force = float(piper_force)
                            try:
                                for piper_cmd in piper_cmds:
                                    for burst in bursts:
                                        row = run_one_probe_case(
                                            args,
                                            env,
                                            task,
                                            piper_cmd,
                                            burst,
                                            primitive_steps,
                                            seed,
                                            frames_dir,
                                            sheets_dir,
                                        )
                                        rows.append(row)
                                        write_outputs(rows, out_dir, quiet=True)
                            finally:
                                close = getattr(env, "close", None)
                                if callable(close):
                                    close()
    return rows


def run_one_probe_case(
    args: argparse.Namespace,
    env: Any,
    task: str,
    piper_cmd: float,
    burst: int,
    primitive_steps: int,
    seed: int,
    frames_dir: Path,
    sheets_dir: Path,
) -> dict[str, Any]:
    _obs, info = env.reset(seed=seed, options={"task": task})
    env.config.piper_primitive_feed_value = abs(float(piper_cmd))
    env.config.piper_primitive_retract_value = abs(float(piper_cmd))
    for _ in range(max(args.settle_steps, 0)):
        _obs, _reward, _terminated, _truncated, info = env.step(action_for_feed(env, 0.0))
    before_info = info
    before = env.render_camera(args.camera)
    before_x, before_y, before_area = red_tip_centroid(before)
    for _ in range(max(int(burst), 0)):
        _obs, _reward, _terminated, _truncated, info = env.step(action_for_feed(env, piper_cmd))
        for _ in range(max(int(primitive_steps) - 1, 0)):
            _obs, _reward, _terminated, _truncated, info = env.step(action_for_feed(env, 0.0))
    for _ in range(max(args.after_settle_steps, 0)):
        _obs, _reward, _terminated, _truncated, info = env.step(action_for_feed(env, 0.0))
    after_info = info
    after = env.render_camera(args.camera)
    after_x, after_y, after_area = red_tip_centroid(after)

    dx = dy = euclidean = None
    if before_x is not None and after_x is not None:
        dx = float(after_x - before_x)
        dy = float(after_y - before_y)
        euclidean = float(np.hypot(dx, dy))

    stem = (
        f"{task}_adv{env.config.piper_advection_scale:.3f}"
        f"_cpg{env.config.commanded_progress_gain:.3f}"
        f"_pf{env.config.piper_force:.3f}"
        f"_cmd{piper_cmd:.3f}_prim{primitive_steps}_burst{burst}_seed{seed}"
    )
    before_path = frames_dir / f"{stem}_before.png"
    after_path = frames_dir / f"{stem}_after.png"
    sheet_path = sheets_dir / f"{stem}.png"
    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)
    sheet = make_sheet(
        before,
        after,
        f"{task} adv={env.config.piper_advection_scale:.3f} cmd={piper_cmd:.3f} prim={primitive_steps} burst={burst}",
        (before_x, before_y),
        (after_x, after_y),
    )
    cv2.imwrite(str(sheet_path), sheet)
    before_obs = before_info.get("obs_dict", {}) if isinstance(before_info, dict) else {}
    after_obs = after_info.get("obs_dict", {}) if isinstance(after_info, dict) else {}
    return {
        "task": task,
        "seed": seed,
        "piper_advection_scale": float(env.config.piper_advection_scale),
        "advance_step": float(env.config.advance_step),
        "feed_unit": float(env.config.advance_step * env.config.piper_advection_scale),
        "commanded_progress_gain": float(env.config.commanded_progress_gain),
        "piper_force": float(env.config.piper_force),
        "piper_cmd": float(piper_cmd),
        "primitive_steps": int(primitive_steps),
        "burst": int(burst),
        "feed_steps": int(max(burst, 0) * max(primitive_steps, 0)),
        "camera": args.camera,
        "before_x": before_x,
        "before_y": before_y,
        "before_red_area": before_area,
        "after_x": after_x,
        "after_y": after_y,
        "after_red_area": after_area,
        "dx_px": dx,
        "dy_px": dy,
        "euclidean_px": euclidean,
        "before_path_progress": before_obs.get("path_progress"),
        "after_path_progress": after_obs.get("path_progress"),
        "path_progress_delta": (
            float(after_obs["path_progress"]) - float(before_obs["path_progress"])
            if "path_progress" in before_obs and "path_progress" in after_obs
            else None
        ),
        "before_piper_insertion_length": before_obs.get("piper_insertion_length"),
        "after_piper_insertion_length": after_obs.get("piper_insertion_length"),
        "piper_insertion_delta": (
            float(after_obs["piper_insertion_length"]) - float(before_obs["piper_insertion_length"])
            if "piper_insertion_length" in before_obs and "piper_insertion_length" in after_obs
            else None
        ),
        "before_tip_pos": before_obs.get("tip_pos"),
        "after_tip_pos": after_obs.get("tip_pos"),
        "before_image": str(before_path.as_posix()),
        "after_image": str(after_path.as_posix()),
        "contact_sheet": str(sheet_path.as_posix()),
    }


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, quiet: bool = False) -> None:
    write_jsonl(out_dir / "measurements.jsonl", rows)
    if rows:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with (out_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    summary: dict[str, Any] = {"rows": len(rows), "groups": {}}
    groups: dict[tuple[str, float, float, float, float, int, int], list[float]] = {}
    for row in rows:
        value = row.get("euclidean_px")
        if isinstance(value, (int, float)) and np.isfinite(value):
            groups.setdefault(
                (
                    str(row["task"]),
                    float(row.get("piper_advection_scale", 0.0)),
                    float(row.get("commanded_progress_gain", 0.0)),
                    float(row.get("piper_force", 0.0)),
                    float(row["piper_cmd"]),
                    int(row.get("primitive_steps", 1)),
                    int(row["burst"]),
                ),
                [],
            ).append(float(value))
    for key, values in sorted(groups.items()):
        task, adv_scale, commanded_progress_gain, piper_force, cmd, primitive_steps, burst = key
        ordered = sorted(values)
        summary["groups"][f"{task}|adv{adv_scale:.3f}|cpg{commanded_progress_gain:.3f}|pf{piper_force:.3f}|cmd{cmd:.3f}|prim{primitive_steps}|burst{burst}"] = {
            "n": len(ordered),
            "median_euclidean_px": float(np.median(ordered)),
            "min_euclidean_px": float(ordered[0]),
            "max_euclidean_px": float(ordered[-1]),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not quiet:
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe simulated Piper-only visible forward scale.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--scene-config", default="simulation_output/robot_scene_mvp/scene_config.json")
    parser.add_argument("--camera-config", default="simulation/camera_configs/mujoco_camera_top_manual_v1.json")
    parser.add_argument("--out", default="simulation_output/sim_piper_forward_probe")
    parser.add_argument("--tasks", nargs="+", choices=["left", "right"], default=["left", "right"])
    parser.add_argument("--piper-cmds", default="0.4 0.6 0.8 1.0")
    parser.add_argument("--piper-advection-scales", default="0.11")
    parser.add_argument("--commanded-progress-gains", default=None)
    parser.add_argument("--piper-forces", default=None)
    parser.add_argument("--primitive-steps", default="1")
    parser.add_argument("--bursts", default="1 2 3")
    parser.add_argument("--advance-step", type=float, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--camera", choices=["side", "top"], default="side")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--settle-steps", type=int, default=3)
    parser.add_argument("--after-settle-steps", type=int, default=3)
    parser.add_argument("--wire-segments", type=int, default=240)
    parser.add_argument("--wire-visual-radius", type=float, default=0.0008)
    parser.add_argument("--wire-visual-rgb", default="0.01 0.01 0.01")
    parser.add_argument("--wire-tip-visual-rgb", default="0.90 0.02 0.02")
    parser.add_argument("--wire-tip-visual-segments", type=int, default=8)
    parser.add_argument("--wire-tip-visual-radius-scale", type=float, default=2.2)
    parser.add_argument("--wire-tip-marker-radius", type=float, default=0.0020)
    parser.add_argument("--wire-tip-marker-alpha", type=float, default=0.95)
    parser.add_argument("--wire-visual-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--render-domain-preset", default="branchs_like_v1")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    rows = run_probe(args)
    write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
