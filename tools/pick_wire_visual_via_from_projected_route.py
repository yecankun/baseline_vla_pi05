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


WINDOW = "Pick projected route center points"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_name_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).replace(",", " ").split() if item.strip()]


def _parse_vec3(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in str(text).replace(",", " ").split() if item.strip()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)} from {text!r}")
    return np.asarray(values, dtype=np.float32)


def _camera_axes(env: TipGuidedWireEnv) -> tuple[np.ndarray, np.ndarray]:
    az = np.radians(float(env.camera.azimuth))
    right = np.array([np.cos(az), np.sin(az), 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    return right, np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _camera_forward(env: TipGuidedWireEnv) -> np.ndarray:
    az = np.radians(float(env.camera.azimuth))
    el = np.radians(float(env.camera.elevation))
    forward = np.array(
        [-np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)],
        dtype=np.float32,
    )
    return forward / max(float(np.linalg.norm(forward)), 1e-6)


def _camera_position(env: TipGuidedWireEnv) -> np.ndarray:
    return np.asarray(env.camera.lookat, dtype=np.float32) - _camera_forward(env) * float(env.camera.distance)


def _set_camera_from_position(env: TipGuidedWireEnv, position: np.ndarray) -> None:
    lookat = np.asarray(env.camera.lookat, dtype=np.float32)
    direction = lookat - np.asarray(position, dtype=np.float32)
    distance = max(float(np.linalg.norm(direction)), 1e-6)
    unit = direction / distance
    env.camera.distance = distance
    env.camera.elevation = float(np.degrees(np.arcsin(float(np.clip(unit[2], -1.0, 1.0)))))
    env.camera.azimuth = float(np.degrees(np.arctan2(float(-unit[0]), float(unit[1]))))


def _mouse_wheel_delta(flags: int) -> int:
    packed_delta = (int(flags) >> 16) & 0xFFFF
    if packed_delta & 0x8000:
        packed_delta -= 0x10000
    return int(packed_delta)


def _set_picker_visibility(env: TipGuidedWireEnv, mode: str) -> None:
    if mode == "all":
        return
    if not hasattr(env, "_projected_picker_original_rgba"):
        env._projected_picker_original_rgba = env.model.geom_rgba.copy()
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(env.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        text = f"{geom_name} {body_name}".lower()
        is_vessel = geom_name == "vessel_mesh"
        is_route_marker = geom_name.startswith("route_pick_marker_")
        is_piper = "piper" in text
        visible = mode == "all" or is_vessel or is_route_marker or (mode == "vessel-piper" and is_piper)
        if visible:
            if is_route_marker:
                env.model.geom_rgba[geom_id, :] = np.asarray([1.0, 0.52, 0.02, 0.92], dtype=np.float32)
            elif is_vessel:
                env.model.geom_rgba[geom_id, 3] = 0.24
            else:
                env.model.geom_rgba[geom_id, 3] = max(float(env.model.geom_rgba[geom_id, 3]), 0.72 if is_piper else 0.88)
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0


def _route_marker_progress_from_name(name: str, task: str) -> float | None:
    prefix = f"route_pick_marker_{task}_p"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    digits = rest.split("_", 1)[0]
    if not digits.isdigit():
        return None
    return float(int(digits)) / 1000.0


def _set_route_marker_range(
    env: TipGuidedWireEnv,
    *,
    task: str,
    max_progress: float,
    picked_geom_ids: set[int],
) -> None:
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        progress = _route_marker_progress_from_name(geom_name, task)
        if progress is None:
            continue
        if progress > max_progress + 1e-6:
            env.model.geom_rgba[geom_id, 3] = 0.0
        elif int(geom_id) in picked_geom_ids:
            env.model.geom_rgba[geom_id, :] = np.asarray([0.0, 0.90, 1.0, 0.98], dtype=np.float32)
        else:
            env.model.geom_rgba[geom_id, :] = np.asarray([1.0, 0.52, 0.02, 0.92], dtype=np.float32)


def _select_route_marker(
    env: TipGuidedWireEnv,
    click_xy: tuple[int, int],
    scene_option: mujoco.MjvOption,
    *,
    task: str,
) -> tuple[int, float, np.ndarray, str] | None:
    if env.renderer is None:
        return None
    x, y = click_xy
    original_alpha = env.model.geom_rgba[:, 3].copy()
    original_group = env.model.geom_group.copy()
    select_option = mujoco.MjvOption()
    select_option.geomgroup[:] = 0
    select_option.geomgroup[0] = 1
    selpnt = np.zeros(3, dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    flexid = np.array([-1], dtype=np.int32)
    skinid = np.array([-1], dtype=np.int32)
    try:
        for marker_geom_id in range(env.model.ngeom):
            geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, marker_geom_id) or ""
            if not geom_name.startswith(f"route_pick_marker_{task}_"):
                env.model.geom_rgba[marker_geom_id, 3] = 0.0
                env.model.geom_group[marker_geom_id] = 5
            else:
                env.model.geom_group[marker_geom_id] = 0
        env.renderer.update_scene(env.data, camera=env.camera)
        bodyid = mujoco.mjv_select(
            env.model,
            env.data,
            select_option,
            float(env.renderer.width) / float(env.renderer.height),
            float(x) / float(env.renderer.width),
            1.0 - float(y) / float(env.renderer.height),
            env.renderer.scene,
            selpnt,
            geomid,
            flexid,
            skinid,
        )
    finally:
        env.model.geom_rgba[:, 3] = original_alpha
        env.model.geom_group[:] = original_group
        env.renderer.update_scene(env.data, camera=env.camera)
    if int(bodyid) < 0 or int(geomid[0]) < 0:
        print(f"No MuJoCo geom selected near pixel ({x}, {y}).")
        return None
    geom_id = int(geomid[0])
    geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    progress = _route_marker_progress_from_name(geom_name, task)
    if progress is None:
        print(f"Selected geom is {geom_name!r}, not a route marker; click an orange marker.")
        return None
    return geom_id, progress, selpnt.astype(np.float32), geom_name


def _auto_piper_exit_point(
    env: TipGuidedWireEnv,
    mode: str,
    body_names: list[str],
    geom_names: list[str],
    offset: np.ndarray,
) -> tuple[np.ndarray, str, list[str]]:
    env._sync_mujoco()
    if mode == "body-midpoint":
        points = []
        for name in body_names:
            body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Unknown MuJoCo body: {name}")
            points.append(np.asarray(env.data.xpos[body_id], dtype=np.float32))
        return (np.mean(points, axis=0) + offset).astype(np.float32), mode, body_names
    if mode == "geom-midpoint":
        points = []
        for name in geom_names:
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Unknown MuJoCo geom: {name}")
            points.append(np.asarray(env.data.geom_xpos[geom_id], dtype=np.float32))
        return (np.mean(points, axis=0) + offset).astype(np.float32), mode, geom_names
    if mode == "robot-tool":
        return (np.asarray(env.robot_tool_world("piper"), dtype=np.float32) + offset).astype(np.float32), mode, ["piper_tool"]
    raise ValueError(f"Unsupported auto Piper exit mode: {mode}")


def _route_candidates(
    env: TipGuidedWireEnv,
    *,
    max_progress: float,
    stride: float,
) -> dict[str, np.ndarray]:
    progresses = np.arange(0.0, max_progress + 1e-6, max(float(stride), 1e-3), dtype=np.float32)
    if len(progresses) == 0 or float(progresses[-1]) < max_progress:
        progresses = np.concatenate([progresses, np.asarray([max_progress], dtype=np.float32)])
    points = []
    for progress in progresses:
        center, _tangent, _n1, _n2, _radius = env._local_path_frame(float(progress))
        points.append(np.asarray(center, dtype=np.float32))
    points_arr = np.asarray(points, dtype=np.float32)
    return {"progresses": progresses, "points": points_arr}


def _draw_overlay(
    image: np.ndarray,
    candidates: dict[str, np.ndarray],
    picks: list[dict],
    piper_exit: dict | None,
    view_id: int,
    text: str,
) -> np.ndarray:
    out = image.copy()
    if picks:
        pick_text = "picks: " + ", ".join(f"{index}:{row['progress']:.1f}" for index, row in enumerate(picks, start=1))
        cv2.putText(out, pick_text[:96], (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(out, pick_text[:96], (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    if piper_exit is not None:
        point = piper_exit.get("point", [0.0, 0.0, 0.0])
        cv2.putText(out, f"Piper exit {point[0]:.3f},{point[1]:.3f},{point[2]:.3f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(out, f"Piper exit {point[0]:.3f},{point[1]:.3f},{point[2]:.3f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(out, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, "Left: orange route marker | G: Piper exit | Backspace: undo | Enter: save | Right/Middle/Wheel: camera", (12, out.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(out, "Left: orange route marker | G: Piper exit | Backspace: undo | Enter: save | Right/Middle/Wheel: camera", (12, out.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick visual guidewire via points from projected registered route centers.")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--out-picks", default="simulation/routes/wire_visual_via_projected_route_picks.json")
    parser.add_argument("--camera", choices=["top", "side", "overview", "perspective"], default="top")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--camera-config", default="simulation/camera_configs/mujoco_camera_top_manual_v1.json")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--route-stride", type=float, default=0.25)
    parser.add_argument("--max-pick-distance-px", type=float, default=35.0)
    parser.add_argument("--picker-visibility", choices=["vessel-piper", "vessel-only", "all"], default="vessel-piper")
    parser.add_argument("--azimuth-step", type=float, default=8.0)
    parser.add_argument("--elevation-step", type=float, default=5.0)
    parser.add_argument("--pan-step", type=float, default=0.012)
    parser.add_argument("--mouse-rotate-scale", type=float, default=0.22)
    parser.add_argument("--mouse-pan-scale", type=float, default=0.00045)
    parser.add_argument("--wheel-zoom-factor", type=float, default=1.08)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--max-distance", type=float, default=2.5)
    parser.add_argument("--auto-piper-exit-mode", choices=["body-midpoint", "geom-midpoint", "robot-tool"], default="body-midpoint")
    parser.add_argument("--auto-piper-exit-bodies", default="piper_link7,piper_link8")
    parser.add_argument("--auto-piper-exit-geoms", default="piper_link7_vis0_geom,piper_link8_vis0_geom")
    parser.add_argument("--auto-piper-exit-offset", default="0 0 0")
    parser.add_argument("--sort-by-progress", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    route_path = Path(args.route_config)
    route_config = _load_json(route_path)
    cfg = TipGuidedWireConfig(
        route_config_path=str(route_path),
        camera_config_path=args.camera_config,
        render_width=args.render_width,
        render_height=args.render_height,
        render_observation=False,
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
        route_picker_markers=True,
        route_picker_marker_task=args.task,
        route_picker_marker_stride=args.route_stride,
        route_picker_marker_size=0.0038,
        route_picker_marker_alpha=0.92,
    )
    env = TipGuidedWireEnv(cfg, seed=7)
    picks: list[dict] = []
    piper_exit: dict | None = None
    view_id = 0
    picked_geom_ids: set[int] = set()
    mouse_state = {"right": False, "middle": False, "last": (0, 0), "dirty": False}
    auto_bodies = _parse_name_list(args.auto_piper_exit_bodies)
    auto_geoms = _parse_name_list(args.auto_piper_exit_geoms)
    auto_offset = _parse_vec3(args.auto_piper_exit_offset)

    try:
        env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
        _set_picker_visibility(env, args.picker_visibility)
        _set_route_marker_range(env, task=args.task, max_progress=float(env.path_progress_float), picked_geom_ids=picked_geom_ids)
        env._configure_free_camera(args.camera)
        scene_option = mujoco.MjvOption()

        def set_distance(distance: float) -> None:
            env.camera.distance = float(np.clip(distance, float(args.min_distance), float(args.max_distance)))

        def pan(dx: int, dy: int) -> None:
            right, up = _camera_axes(env)
            scale = float(args.mouse_pan_scale) * max(float(env.camera.distance), 0.01)
            env.camera.lookat[:] = np.asarray(env.camera.lookat, dtype=np.float32) - right * float(dx) * scale + up * float(dy) * scale

        def render() -> tuple[np.ndarray, dict[str, np.ndarray]]:
            env._sync_mujoco()
            _set_route_marker_range(env, task=args.task, max_progress=float(env.path_progress_float), picked_geom_ids=picked_geom_ids)
            image = env.render_camera(args.camera)
            env._configure_free_camera(args.camera)
            candidates = _route_candidates(
                env,
                max_progress=float(env.path_progress_float),
                stride=float(args.route_stride),
            )
            return image, candidates

        image, candidates = render()

        def mark_dirty() -> None:
            mouse_state["dirty"] = True

        def on_mouse(event, x, y, flags, _param):
            nonlocal view_id, image, candidates
            if event == cv2.EVENT_RBUTTONDOWN:
                mouse_state["right"] = True
                mouse_state["last"] = (int(x), int(y))
                return
            if event == cv2.EVENT_RBUTTONUP:
                mouse_state["right"] = False
                return
            if event == cv2.EVENT_MBUTTONDOWN:
                mouse_state["middle"] = True
                mouse_state["last"] = (int(x), int(y))
                return
            if event == cv2.EVENT_MBUTTONUP:
                mouse_state["middle"] = False
                return
            if event == cv2.EVENT_MOUSEMOVE and (mouse_state["right"] or mouse_state["middle"]):
                last_x, last_y = mouse_state["last"]
                dx = int(x) - int(last_x)
                dy = int(y) - int(last_y)
                mouse_state["last"] = (int(x), int(y))
                if mouse_state["right"]:
                    env.camera.azimuth = float(env.camera.azimuth) + float(dx) * float(args.mouse_rotate_scale)
                    env.camera.elevation = float(np.clip(float(env.camera.elevation) - float(dy) * float(args.mouse_rotate_scale), -179.0, 179.0))
                else:
                    pan(dx, dy)
                view_id += 1
                mark_dirty()
                return
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = _mouse_wheel_delta(flags)
                factor = max(float(args.wheel_zoom_factor), 1.01)
                if delta > 0:
                    set_distance(float(env.camera.distance) / factor)
                elif delta < 0:
                    set_distance(float(env.camera.distance) * factor)
                view_id += 1
                mark_dirty()
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                selected = _select_route_marker(env, (int(x), int(y)), scene_option, task=args.task)
                if selected is None:
                    return
                geom_id, progress, selected_point, geom_name = selected
                if progress > float(env.path_progress_float) + 1e-6:
                    print(f"Selected route marker progress={progress:.3f} is beyond current tip progress={float(env.path_progress_float):.3f}.")
                    return
                point = env._route_center_at(progress).astype(np.float32)
                picked_geom_ids.add(int(geom_id))
                picks.append(
                    {
                        "pixel": [int(x), int(y)],
                        "view_id": int(view_id),
                        "progress": progress,
                        "point": point.astype(float).tolist(),
                        "frame": "scene",
                        "source": "mujoco_route_pick_marker",
                        "selected_geom": geom_name,
                        "selected_scene_point": selected_point.astype(float).tolist(),
                    }
                )
                print(f"picked #{len(picks)} progress={progress:.3f} geom={geom_name} center={point.tolist()}")
                mark_dirty()

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, int(args.render_width), int(args.render_height))
        cv2.setMouseCallback(WINDOW, on_mouse)

        while True:
            if mouse_state["dirty"]:
                image, candidates = render()
                mouse_state["dirty"] = False
            status = (
                f"az={float(env.camera.azimuth):.1f} elev={float(env.camera.elevation):.1f} "
                f"dist={float(env.camera.distance):.3f} picks={len(picks)}"
            )
            cv2.imshow(WINDOW, _draw_overlay(image, candidates, picks, piper_exit, view_id, status))
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), 27):
                break
            if key in (8, 127) and picks:
                removed = picks.pop()
                if "selected_geom" in removed:
                    geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, str(removed["selected_geom"]))
                    if geom_id >= 0:
                        picked_geom_ids.discard(int(geom_id))
                print(f"removed progress={removed['progress']:.3f}")
                mark_dirty()
            if key in (ord("g"), ord("G")):
                try:
                    point, source_mode, source_names = _auto_piper_exit_point(
                        env,
                        mode=args.auto_piper_exit_mode,
                        body_names=auto_bodies,
                        geom_names=auto_geoms,
                        offset=auto_offset,
                    )
                    piper_exit = {
                        "frame": "scene",
                        "point": point.astype(float).tolist(),
                        "source": f"auto_{source_mode}",
                        "source_names": source_names,
                        "offset": auto_offset.astype(float).tolist(),
                    }
                    print(f"set Piper visual exit from {source_mode}: {point.astype(float).tolist()}")
                except Exception as exc:
                    print(f"Failed to compute automatic Piper exit: {exc}")
            rerender = False
            if key == ord("a"):
                env.camera.azimuth = float(env.camera.azimuth) - float(args.azimuth_step)
                rerender = True
            elif key == ord("d"):
                env.camera.azimuth = float(env.camera.azimuth) + float(args.azimuth_step)
                rerender = True
            elif key == ord("w"):
                env.camera.elevation = float(np.clip(float(env.camera.elevation) + float(args.elevation_step), -179.0, 179.0))
                rerender = True
            elif key == ord("s"):
                env.camera.elevation = float(np.clip(float(env.camera.elevation) - float(args.elevation_step), -179.0, 179.0))
                rerender = True
            if rerender:
                view_id += 1
                mark_dirty()
            if key in (10, 13):
                if not picks:
                    print("No points selected; route config not changed.")
                    continue
                saved_picks = sorted(picks, key=lambda row: float(row["progress"])) if args.sort_by_progress else list(picks)
                route_config["wire_visual_via_points"] = [
                    {
                        "frame": "scene",
                        "point": row["point"],
                        "progress": row["progress"],
                        "source": row["source"],
                    }
                    for row in saved_picks
                ]
                if piper_exit is not None:
                    route_config["wire_visual_piper_exit_point"] = piper_exit
                route_config["_wire_visual_via_note"] = (
                    "Visual-only guidewire tail points picked from native MuJoCo route-center marker geoms. "
                    "No custom world-to-screen projection or vessel/table surface point is used."
                )
                _save_json(route_path, route_config)
                _save_json(
                    Path(args.out_picks),
                    {
                        "camera": args.camera,
                        "task": args.task,
                        "piper_exit": piper_exit,
                        "sort_by_progress": bool(args.sort_by_progress),
                        "picks": saved_picks,
                    },
                )
                print(f"saved {len(saved_picks)} projected route points to {route_path}")
                print(f"saved pick log to {args.out_picks}")
                break
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
