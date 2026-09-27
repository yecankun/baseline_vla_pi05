from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.tip_guided_wire_env import TipGuidedWireConfig, TipGuidedWireEnv


WINDOW = "Adjust wire visual route"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _point_from_config(route_config: dict) -> np.ndarray:
    item = route_config.get("wire_visual_piper_exit_point")
    if isinstance(item, dict) and "point" in item:
        return np.asarray(item["point"], dtype=np.float32)
    if item is not None:
        return np.asarray(item, dtype=np.float32)
    return np.asarray([-0.10, -0.12, -0.03], dtype=np.float32)


def _entry_progress_from_config(route_config: dict) -> float:
    return float(route_config.get("wire_visual_entry_progress", 0.0))


def _entry_point_from_config(route_config: dict, env: TipGuidedWireEnv, entry_progress: float) -> np.ndarray:
    item = route_config.get("wire_visual_entry_point") or route_config.get("wire_visual_vessel_entry_point")
    if isinstance(item, dict) and "point" in item:
        return np.asarray(item["point"], dtype=np.float32)
    if item is not None:
        return np.asarray(item, dtype=np.float32)
    return env._route_center_at(entry_progress)


def _route_points_from_config(route_config: dict) -> list[np.ndarray]:
    rows = route_config.get("wire_visual_route_points")
    if rows is None:
        rows = route_config.get("wire_visual_centerline_points")
    if rows is None:
        return []
    points: list[np.ndarray] = []
    for item in rows:
        point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
        points.append(point.astype(np.float32))
    return points


def _dynamic_points_from_config(route_config: dict, task: str | None = None, task_specific: bool = False) -> list[np.ndarray]:
    key = f"{task}_wire_visual_dynamic_route_points" if task and task_specific else "wire_visual_dynamic_route_points"
    rows = route_config.get(key)
    if rows is None and task and task_specific:
        rows = route_config.get("wire_visual_dynamic_route_points")
    if rows is None:
        return []
    points: list[np.ndarray] = []
    for item in rows:
        point = np.asarray(item.get("point", item) if isinstance(item, dict) else item, dtype=np.float32)
        points.append(point.astype(np.float32))
    return points


def _route_point_rows(route_points: list[np.ndarray]) -> list[dict]:
    return [
        {
            "frame": "scene",
            "point": [float(v) for v in point],
            "progress": float(index),
            "source": "manual_outlet_adjuster",
        }
        for index, point in enumerate(route_points)
    ]


def _set_route_values(
    env: TipGuidedWireEnv,
    outlet: np.ndarray,
    entry_progress: float,
    entry_point: np.ndarray,
    route_points: list[np.ndarray],
    dynamic_points: list[np.ndarray],
    dynamic_key: str = "wire_visual_dynamic_route_points",
) -> None:
    route_config = env.reference_env.route_config
    route_config["wire_visual_piper_exit_point"] = {
        "frame": "scene",
        "point": [float(v) for v in outlet],
        "source": "manual_outlet_adjuster",
    }
    route_config["wire_visual_entry_progress"] = float(max(entry_progress, 0.0))
    route_config["wire_visual_entry_point"] = {
        "frame": "scene",
        "point": [float(v) for v in entry_point],
        "source": "manual_outlet_adjuster",
    }
    route_config["wire_visual_route_points"] = _route_point_rows(route_points)
    route_config[dynamic_key] = _route_point_rows(dynamic_points)
    env._rebuild_visual_tail()
    env._sync_mujoco()


def _set_robot_alpha(env: TipGuidedWireEnv, visible: bool) -> None:
    if not hasattr(env, "_wire_outlet_adjuster_original_rgba"):
        env._wire_outlet_adjuster_original_rgba = env.model.geom_rgba.copy()
    original = env._wire_outlet_adjuster_original_rgba
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(env.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        text = f"{geom_name} {body_name}".lower()
        if "piper" in text or "elite" in text or "elirobot" in text:
            env.model.geom_rgba[geom_id, 3] = original[geom_id, 3] if visible else 0.0


def _active_point(
    outlet: np.ndarray,
    entry_point: np.ndarray,
    route_points: list[np.ndarray],
    dynamic_points: list[np.ndarray],
    active_kind: str,
    active_route_index: int,
) -> np.ndarray:
    if active_kind == "entry":
        return entry_point
    if active_kind == "route" and route_points:
        return route_points[int(np.clip(active_route_index, 0, len(route_points) - 1))]
    if active_kind == "dynamic" and dynamic_points:
        return dynamic_points[int(np.clip(active_route_index, 0, len(dynamic_points) - 1))]
    return outlet


def _active_label(active_kind: str, active_route_index: int) -> str:
    if active_kind == "route":
        return f"route{active_route_index}"
    if active_kind == "dynamic":
        return f"dynamic{active_route_index}"
    return active_kind


def _set_scene_markers(
    env: TipGuidedWireEnv,
    outlet: np.ndarray,
    entry_point: np.ndarray,
    route_points: list[np.ndarray],
    dynamic_points: list[np.ndarray],
    active_kind: str,
    active_route_index: int,
) -> None:
    active = _active_point(outlet, entry_point, route_points, dynamic_points, active_kind, active_route_index)
    marker_specs = [
        ("diagnostic_magnetic_marker", "diagnostic_magnetic_marker_geom", active, [1.0, 0.50, 0.00, 0.98], 0.0075),
        ("diagnostic_elite_marker", "diagnostic_elite_marker_geom", entry_point, [1.0, 0.95, 0.00, 0.98], 0.0085),
        ("diagnostic_tip_marker", "diagnostic_tip_marker_geom", env.tip, [0.95, 0.02, 0.02, 0.98], 0.0060),
    ]
    for body_name, geom_name, point, rgba, radius in marker_specs:
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            env.model.geom_rgba[geom_id, :] = np.asarray(rgba, dtype=np.float32)
            env.model.geom_size[geom_id, 0] = float(radius)
        joint_name = f"{body_name}_free"
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_addr = int(env.model.jnt_qposadr[joint_id])
        env.data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(point, dtype=np.float32)
        env.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mujoco.mj_forward(env.model, env.data)


def _draw_text(
    image: np.ndarray,
    *,
    camera: str,
    outlet: np.ndarray,
    entry_progress: float,
    entry_point: np.ndarray,
    active_label: str,
    route_points: list[np.ndarray],
    dynamic_points: list[np.ndarray],
    step: float,
    robots_visible: bool,
    saved: bool,
) -> np.ndarray:
    out = image.copy()
    lines = [
        f"camera={camera} | outlet=({outlet[0]:+.4f}, {outlet[1]:+.4f}, {outlet[2]:+.4f})",
        f"entry=({entry_point[0]:+.4f}, {entry_point[1]:+.4f}, {entry_point[2]:+.4f}) progress={entry_progress:.3f}",
        f"active={active_label} prefix_points={len(route_points)} dynamic_points={len(dynamic_points)} step={step * 1000.0:.1f}mm robots={'on' if robots_visible else 'off'}",
        "MuJoCo markers: orange=active point  yellow=entry  red=tip",
        "e: cycle active  n: add prefix point  g: add dynamic point  x: delete active point",
        "dynamic points shape the post-attach visual segment directly",
        "r: reset entry to route",
        "j/l: x  i/k: y  u/o: z  ,/.: route progress",
        "[/]: step  1/2/3: side/top/overview  h: robots  s: save  q: quit",
    ]
    if saved:
        lines.append("saved route config")
    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactively adjust the visual guidewire outlet, entry, and manual route points.")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--camera-config", default="simulation/camera_configs/mujoco_camera_top_manual_v1.json")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--camera", choices=["side", "top", "overview"], default="side")
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--wire-segments", type=int, default=180)
    parser.add_argument("--step", type=float, default=0.002)
    parser.add_argument("--show-robots", action="store_true")
    parser.add_argument(
        "--task-specific",
        action="store_true",
        help="Read/write task-specific dynamic points, e.g. right_wire_visual_dynamic_route_points.",
    )
    args = parser.parse_args()

    route_path = Path(args.route_config)
    route_config = _load_json(route_path)
    outlet = _point_from_config(route_config)
    entry_progress = _entry_progress_from_config(route_config)
    step = float(args.step)
    camera = str(args.camera)
    robots_visible = bool(args.show_robots)
    active_kind = "outlet"
    active_route_index = 0
    route_points = _route_points_from_config(route_config)
    dynamic_key = f"{args.task}_wire_visual_dynamic_route_points" if args.task_specific else "wire_visual_dynamic_route_points"
    dynamic_points = _dynamic_points_from_config(route_config, args.task, args.task_specific)
    saved = False

    cfg = TipGuidedWireConfig(
        route_config_path=str(route_path),
        camera_config_path=args.camera_config,
        render_width=args.render_width,
        render_height=args.render_height,
        render_observation=False,
        wire_segments=args.wire_segments,
        wire_visual_mode="line",
        wire_visual_radius=0.0008,
        wire_visual_rgb="0.02 0.02 0.018",
        wire_tip_visual_rgb="0.78 0.04 0.02",
        wire_tip_visual_segments=8,
        wire_tip_visual_radius_scale=2.2,
        wire_tip_marker_radius=0.0020,
        wire_tip_marker_alpha=0.0,
        show_tool_markers=False,
        show_path_tubes=False,
    )
    env = TipGuidedWireEnv(cfg, seed=7)
    try:
        env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
        entry_point = _entry_point_from_config(route_config, env, entry_progress)
        _set_route_values(env, outlet, entry_progress, entry_point, route_points, dynamic_points, dynamic_key)
        _set_robot_alpha(env, robots_visible)
        _set_scene_markers(env, outlet, entry_point, route_points, dynamic_points, active_kind, active_route_index)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, int(args.render_width), int(args.render_height))
        frame = None
        while True:
            _set_route_values(env, outlet, entry_progress, entry_point, route_points, dynamic_points, dynamic_key)
            _set_robot_alpha(env, robots_visible)
            _set_scene_markers(env, outlet, entry_point, route_points, dynamic_points, active_kind, active_route_index)
            image = env.render_camera(camera)
            env._configure_free_camera(camera)
            frame = _draw_text(
                image,
                camera=camera,
                outlet=outlet,
                entry_progress=entry_progress,
                entry_point=entry_point,
                active_label=_active_label(active_kind, active_route_index),
                route_points=route_points,
                dynamic_points=dynamic_points,
                step=step,
                robots_visible=robots_visible,
                saved=saved,
            )
            if frame is not None:
                cv2.imshow(WINDOW, frame)
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), 27):
                break
            delta = np.zeros(3, dtype=np.float32)
            if key == ord("j"):
                delta[0] -= step
            elif key == ord("l"):
                delta[0] += step
            elif key == ord("i"):
                delta[1] += step
            elif key == ord("k"):
                delta[1] -= step
            elif key == ord("u"):
                delta[2] += step
            elif key == ord("o"):
                delta[2] -= step
            elif key == ord("e"):
                if active_kind == "outlet":
                    active_kind = "entry"
                elif active_kind == "entry" and route_points:
                    active_kind = "route"
                    active_route_index = 0
                elif active_kind == "route" and active_route_index < len(route_points) - 1:
                    active_route_index += 1
                elif active_kind in {"entry", "route"} and dynamic_points:
                    active_kind = "dynamic"
                    active_route_index = 0
                elif active_kind == "dynamic" and active_route_index < len(dynamic_points) - 1:
                    active_route_index += 1
                else:
                    active_kind = "outlet"
                    active_route_index = 0
            elif key == ord("n"):
                if route_points:
                    previous = route_points[-1]
                else:
                    previous = entry_point
                new_point = ((previous + env.tip.astype(np.float32)) * 0.5).astype(np.float32)
                route_points.append(new_point)
                active_kind = "route"
                active_route_index = len(route_points) - 1
                saved = False
            elif key == ord("g"):
                if dynamic_points:
                    previous = dynamic_points[-1]
                elif route_points:
                    previous = route_points[-1]
                else:
                    previous = entry_point
                new_point = ((previous + env.tip.astype(np.float32)) * 0.5).astype(np.float32)
                dynamic_points.append(new_point)
                active_route_index = len(dynamic_points) - 1
                active_kind = "dynamic"
                saved = False
            elif key == ord("x"):
                if active_kind == "route" and route_points:
                    route_points.pop(active_route_index)
                    if route_points:
                        active_route_index = int(np.clip(active_route_index, 0, len(route_points) - 1))
                    else:
                        active_kind = "entry"
                        active_route_index = 0
                    saved = False
                elif active_kind == "dynamic" and dynamic_points:
                    dynamic_points.pop(active_route_index)
                    if dynamic_points:
                        active_route_index = int(np.clip(active_route_index, 0, len(dynamic_points) - 1))
                    else:
                        active_kind = "entry"
                        active_route_index = 0
                    saved = False
            elif key == ord("["):
                step = max(step * 0.5, 0.00025)
            elif key == ord("]"):
                step = min(step * 2.0, 0.02)
            elif key == ord(","):
                entry_progress = max(entry_progress - 0.5, 0.0)
            elif key == ord("."):
                entry_progress = min(entry_progress + 0.5, float(env.path_progress_float))
            elif key == ord("r"):
                entry_point = env._route_center_at(entry_progress)
                saved = False
            elif key == ord("1"):
                camera = "side"
            elif key == ord("2"):
                camera = "top"
            elif key == ord("3"):
                camera = "overview"
            elif key == ord("h"):
                robots_visible = not robots_visible
            elif key == ord("s"):
                route_config = _load_json(route_path)
                route_config["wire_visual_piper_exit_point"] = {
                    "frame": "scene",
                    "point": [float(v) for v in outlet],
                    "source": "manual_outlet_adjuster",
                }
                route_config["wire_visual_entry_progress"] = float(entry_progress)
                route_config["wire_visual_entry_point"] = {
                    "frame": "scene",
                    "point": [float(v) for v in entry_point],
                    "source": "manual_outlet_adjuster",
                }
                route_config["wire_visual_route_points"] = _route_point_rows(route_points)
                route_config[dynamic_key] = _route_point_rows(dynamic_points)
                route_config["_wire_visual_via_note"] = (
                    "Visual-only guidewire tail route. wire_visual_piper_exit_point is the manually calibrated visual "
                    "guidewire outlet; wire_visual_entry_point is the manually calibrated vessel entry center; rendering "
                    "uses wire_visual_route_points as a fixed entry prefix; the final prefix point is the attach marker. "
                    "wire_visual_dynamic_route_points optionally shape the post-attach visual segment directly; "
                    "task-prefixed dynamic point keys override the global dynamic points for a single branch."
                )
                _save_json(route_path, route_config)
                saved = True
                print(
                    f"saved outlet={outlet.astype(float).tolist()} "
                    f"entry={entry_point.astype(float).tolist()} entry_progress={entry_progress:.3f} to {route_path}"
                )
            if float(np.linalg.norm(delta)) > 0.0:
                if active_kind == "entry":
                    entry_point = (entry_point + delta).astype(np.float32)
                elif active_kind == "route" and route_points:
                    route_points[active_route_index] = (route_points[active_route_index] + delta).astype(np.float32)
                elif active_kind == "dynamic" and dynamic_points:
                    dynamic_points[active_route_index] = (dynamic_points[active_route_index] + delta).astype(np.float32)
                else:
                    outlet = (outlet + delta).astype(np.float32)
                saved = False
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
