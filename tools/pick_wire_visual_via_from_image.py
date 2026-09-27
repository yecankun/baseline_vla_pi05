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
from simulation.visual_distance_estimator import _project_world_points_unclipped


WINDOW = "Pick wire visual via points"


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


def _named_body_position(env: TipGuidedWireEnv, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, str(name))
    if body_id < 0:
        raise ValueError(f"Unknown MuJoCo body: {name}")
    return np.asarray(env.data.xpos[body_id], dtype=np.float32)


def _named_geom_position(env: TipGuidedWireEnv, name: str) -> np.ndarray:
    geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
    if geom_id < 0:
        raise ValueError(f"Unknown MuJoCo geom: {name}")
    return np.asarray(env.data.geom_xpos[geom_id], dtype=np.float32)


def _auto_piper_exit_point(
    env: TipGuidedWireEnv,
    mode: str,
    body_names: list[str],
    geom_names: list[str],
    offset: np.ndarray,
) -> tuple[np.ndarray, str, list[str]]:
    env._sync_mujoco()
    if mode == "body-midpoint":
        if len(body_names) < 2:
            raise ValueError("--auto-piper-exit-bodies must contain at least two body names")
        points = [_named_body_position(env, name) for name in body_names]
        return (np.mean(points, axis=0) + offset).astype(np.float32), mode, body_names
    if mode == "geom-midpoint":
        if len(geom_names) < 2:
            raise ValueError("--auto-piper-exit-geoms must contain at least two geom names")
        points = [_named_geom_position(env, name) for name in geom_names]
        return (np.mean(points, axis=0) + offset).astype(np.float32), mode, geom_names
    if mode == "robot-tool":
        return (np.asarray(env.robot_tool_world("piper"), dtype=np.float32) + offset).astype(np.float32), mode, ["piper_tool"]
    raise ValueError(f"Unsupported auto Piper exit mode: {mode}")


def _set_picker_visibility(env: TipGuidedWireEnv, mode: str) -> None:
    mode = str(mode)
    if mode == "all":
        return
    if not hasattr(env, "_picker_original_geom_rgba"):
        env._picker_original_geom_rgba = env.model.geom_rgba.copy()
        env._picker_original_geom_group = env.model.geom_group.copy()
    for geom_id in range(env.model.ngeom):
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(env.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        text = f"{geom_name} {body_name}".lower()
        is_vessel = geom_name == "vessel_mesh"
        is_piper = "piper" in text
        if mode == "vessel-only":
            visible = is_vessel
        elif mode == "vessel-piper":
            visible = is_vessel or is_piper
        else:
            raise ValueError(f"Unsupported picker visibility mode: {mode}")
        if visible:
            env.model.geom_group[geom_id] = 0
            if is_vessel:
                env.model.geom_rgba[geom_id, 3] = max(float(env.model.geom_rgba[geom_id, 3]), 0.92)
            else:
                env.model.geom_rgba[geom_id, 3] = max(float(env.model.geom_rgba[geom_id, 3]), 0.75)
        else:
            env.model.geom_rgba[geom_id, 3] = 0.0
            env.model.geom_group[geom_id] = 5


def _restore_visibility(env: TipGuidedWireEnv) -> None:
    original = getattr(env, "_picker_original_geom_rgba", None)
    if original is not None:
        env.model.geom_rgba[:] = original
    original_group = getattr(env, "_picker_original_geom_group", None)
    if original_group is not None:
        env.model.geom_group[:] = original_group


def _select_visible_point(
    env: TipGuidedWireEnv,
    click_xy: tuple[int, int],
    scene_option: mujoco.MjvOption,
) -> tuple[np.ndarray, int, int, str] | None:
    if env.renderer is None:
        return None
    x, y = click_xy
    selpnt = np.zeros(3, dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    flexid = np.array([-1], dtype=np.int32)
    skinid = np.array([-1], dtype=np.int32)
    bodyid = mujoco.mjv_select(
        env.model,
        env.data,
        scene_option,
        float(env.renderer.width) / float(env.renderer.height),
        float(x) / float(env.renderer.width),
        1.0 - float(y) / float(env.renderer.height),
        env.renderer.scene,
        selpnt,
        geomid,
        flexid,
        skinid,
    )
    if int(bodyid) < 0:
        return None
    geom_name = ""
    if int(geomid[0]) >= 0:
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) or ""
    return selpnt.astype(np.float32), int(bodyid), int(geomid[0]), geom_name


def _reprojection_error_px(
    env: TipGuidedWireEnv,
    point: np.ndarray,
    click_xy: tuple[int, int],
    width: int,
    height: int,
) -> tuple[float | None, list[float] | None]:
    try:
        projected, valid = _project_world_points_unclipped(
            env,
            np.asarray(point, dtype=np.float32).reshape(1, 3),
            width,
            height,
        )
    except Exception:
        return None, None
    if len(projected) == 0 or not bool(valid[0]):
        return None, None
    px = projected[0].astype(np.float32)
    err = float(np.linalg.norm(px - np.asarray(click_xy, dtype=np.float32)))
    return err, [float(px[0]), float(px[1])]


def _draw_overlay(
    image: np.ndarray,
    clicks: list[dict],
    piper_exit: dict | None,
    view_id: int,
    camera_text: str,
) -> np.ndarray:
    out = image.copy()
    if piper_exit is not None and int(piper_exit.get("view_id", -1)) == int(view_id) and "pixel" in piper_exit:
        x, y = piper_exit["pixel"]
        cv2.circle(out, (int(x), int(y)), 8, (255, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 11, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(out, "P", (int(x) + 11, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    for idx, row in enumerate(clicks, start=1):
        if int(row.get("view_id", -1)) != int(view_id):
            continue
        x, y = row["pixel"]
        cv2.circle(out, (int(x), int(y)), 7, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(x), int(y)), 9, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(out, str(idx), (int(x) + 10, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    line1 = "Left click: add via | G: gripper-center Piper exit | P click: surface exit | C click: center | F: orbit/free-look"
    line2 = "Right drag: rotate | Middle drag: pan | Wheel: zoom | Backspace: undo via | X: clear Piper exit | Enter: save"
    cv2.putText(out, line1, (12, out.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(out, line1, (12, out.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, line2, (12, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(out, line2, (12, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.putText(out, camera_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.putText(out, camera_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    if piper_exit is not None and "pixel" not in piper_exit:
        point = piper_exit.get("point", [0.0, 0.0, 0.0])
        source = str(piper_exit.get("source", "auto"))
        text = f"Piper exit: {source} [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]"
        cv2.putText(out, text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(out, text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def _camera_axes(env: TipGuidedWireEnv) -> tuple[np.ndarray, np.ndarray]:
    az = np.radians(float(env.camera.azimuth))
    right = np.array([np.cos(az), np.sin(az), 0.0], dtype=np.float32)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = right / max(float(np.linalg.norm(right)), 1e-6)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return right, up


def _camera_forward(env: TipGuidedWireEnv) -> np.ndarray:
    az = np.radians(float(env.camera.azimuth))
    el = np.radians(float(env.camera.elevation))
    forward = np.array(
        [
            -np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return forward / norm


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
    flags = int(flags)
    packed_delta = (flags >> 16) & 0xFFFF
    if packed_delta & 0x8000:
        packed_delta -= 0x10000
    if packed_delta != 0:
        return int(packed_delta)
    if flags > 0:
        return 1
    if flags < 0:
        return -1
    return 0


def _route_center_from_surface_point(
    env: TipGuidedWireEnv,
    surface_point: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    progress = env._nearest_route_progress_to_point(surface_point, max_progress=float(env.path_progress_float))
    center, _tangent, _n1, _n2, _radius = env._local_path_frame(progress)
    center = np.asarray(center, dtype=np.float32)
    distance = float(np.linalg.norm(np.asarray(surface_point, dtype=np.float32) - center))
    return float(progress), center, distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick visual-only guidewire via points from a rendered MuJoCo image.")
    parser.add_argument("--route-config", default="simulation/routes/vessel_0422_wire_route_v1.json")
    parser.add_argument("--out-picks", default="simulation/routes/wire_visual_via_image_picks.json")
    parser.add_argument("--camera", choices=["top", "side", "overview", "perspective"], default="top")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--camera-config", default="simulation/camera_configs/mujoco_camera_top_manual_v1.json")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--wire-segments", type=int, default=160)
    parser.add_argument("--vessel-only", action=argparse.BooleanOptionalAction, default=None, help="Deprecated alias for --picker-visibility vessel-only/all.")
    parser.add_argument(
        "--picker-visibility",
        choices=["vessel-piper", "vessel-only", "all"],
        default="vessel-piper",
        help="Visible geoms while picking. Default keeps vessel and Piper visible, hiding Elite and other clutter.",
    )
    parser.add_argument(
        "--allowed-geoms",
        default="vessel_mesh",
        help="Comma/space-separated MuJoCo geom names allowed for picks. Use '*' to allow any visible geom.",
    )
    parser.add_argument("--azimuth-step", type=float, default=8.0)
    parser.add_argument("--elevation-step", type=float, default=5.0)
    parser.add_argument("--pan-step", type=float, default=0.012)
    parser.add_argument("--zoom-factor", type=float, default=1.12)
    parser.add_argument("--mouse-rotate-scale", type=float, default=0.22, help="Camera degrees per dragged pixel.")
    parser.add_argument("--mouse-pan-scale", type=float, default=0.00045, help="Lookat translation per dragged pixel at distance 1.0.")
    parser.add_argument("--wheel-zoom-factor", type=float, default=1.08)
    parser.add_argument("--min-distance", type=float, default=0.03)
    parser.add_argument("--max-distance", type=float, default=2.5)
    parser.add_argument(
        "--via-point-mode",
        choices=["route-center", "surface"],
        default="route-center",
        help="Save picked vessel-surface points directly, or map them to the nearest registered route center.",
    )
    parser.add_argument(
        "--auto-piper-exit-mode",
        choices=["body-midpoint", "geom-midpoint", "robot-tool"],
        default="body-midpoint",
        help="How G computes the non-surface visual guidewire outlet.",
    )
    parser.add_argument(
        "--auto-piper-exit-bodies",
        default="piper_link7,piper_link8",
        help="Body names averaged by --auto-piper-exit-mode body-midpoint.",
    )
    parser.add_argument(
        "--auto-piper-exit-geoms",
        default="piper_link7_vis0_geom,piper_link8_vis0_geom",
        help="Geom names averaged by --auto-piper-exit-mode geom-midpoint.",
    )
    parser.add_argument(
        "--auto-piper-exit-offset",
        default="0 0 0",
        help="Scene-frame XYZ offset added to the automatic Piper visual exit point.",
    )
    args = parser.parse_args()

    route_path = Path(args.route_config)
    route_config = _load_json(route_path)
    config = TipGuidedWireConfig(
        route_config_path=str(route_path),
        camera_config_path=args.camera_config,
        render_width=args.render_width,
        render_height=args.render_height,
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
    env = TipGuidedWireEnv(config, seed=7)
    clicks: list[dict] = []
    piper_exit: dict | None = None
    pick_mode = "via"
    camera_mode = "orbit"
    view_id = 0
    allowed_geoms = {
        item.strip()
        for item in args.allowed_geoms.replace(",", " ").split()
        if item.strip()
    }
    allow_any_geom = "*" in allowed_geoms
    auto_piper_exit_bodies = _parse_name_list(args.auto_piper_exit_bodies)
    auto_piper_exit_geoms = _parse_name_list(args.auto_piper_exit_geoms)
    auto_piper_exit_offset = _parse_vec3(args.auto_piper_exit_offset)
    try:
        env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
        visibility_mode = args.picker_visibility
        if args.vessel_only is not None:
            visibility_mode = "vessel-only" if bool(args.vessel_only) else "all"
        _set_picker_visibility(env, visibility_mode)
        scene_option = mujoco.MjvOption()
        if visibility_mode != "all":
            scene_option.geomgroup[:] = 0
            scene_option.geomgroup[0] = 1
        env._configure_free_camera(args.camera)

        def render_current_view() -> np.ndarray:
            env._sync_mujoco()
            if env.renderer is None:
                env.renderer = mujoco.Renderer(env.model, height=args.render_height, width=args.render_width)
            env.renderer.update_scene(env.data, camera=env.camera, scene_option=scene_option)
            rgb = env.renderer.render()
            image_now = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return image_now

        image = render_current_view()
        mouse_state = {
            "right_dragging": False,
            "middle_dragging": False,
            "last_xy": (0, 0),
            "right_start_position": None,
            "right_start_azimuth": 0.0,
            "right_start_elevation": 0.0,
            "dirty": False,
        }

        def mark_dirty() -> None:
            mouse_state["dirty"] = True

        def set_camera_distance(distance: float) -> None:
            env.camera.distance = float(np.clip(float(distance), float(args.min_distance), float(args.max_distance)))

        def pan_camera_pixels(dx: int, dy: int) -> None:
            right, up = _camera_axes(env)
            scale = float(args.mouse_pan_scale) * max(float(env.camera.distance), 0.01)
            env.camera.lookat[:] = (
                np.asarray(env.camera.lookat, dtype=np.float32)
                - right * float(dx) * scale
                + up * float(dy) * scale
            )

        def on_mouse(event, x, y, _flags, _param):
            nonlocal view_id, piper_exit, pick_mode
            if event == cv2.EVENT_RBUTTONDOWN:
                mouse_state["right_dragging"] = True
                mouse_state["last_xy"] = (int(x), int(y))
                mouse_state["right_start_position"] = _camera_position(env).copy()
                mouse_state["right_start_azimuth"] = float(env.camera.azimuth)
                mouse_state["right_start_elevation"] = float(env.camera.elevation)
                return
            if event == cv2.EVENT_RBUTTONUP:
                mouse_state["right_dragging"] = False
                return
            if event == cv2.EVENT_MBUTTONDOWN:
                mouse_state["middle_dragging"] = True
                mouse_state["last_xy"] = (int(x), int(y))
                return
            if event == cv2.EVENT_MBUTTONUP:
                mouse_state["middle_dragging"] = False
                return
            if event == cv2.EVENT_MOUSEMOVE and bool(mouse_state["right_dragging"]):
                last_x, last_y = mouse_state["last_xy"]
                dx = int(x) - int(last_x)
                dy = int(y) - int(last_y)
                mouse_state["last_xy"] = (int(x), int(y))
                if dx or dy:
                    scale = float(args.mouse_rotate_scale)
                    if camera_mode == "free-look":
                        camera_position = np.asarray(mouse_state["right_start_position"], dtype=np.float32)
                        env.camera.azimuth = float(env.camera.azimuth) + float(dx) * scale
                        env.camera.elevation = float(np.clip(float(env.camera.elevation) - float(dy) * scale, -179.0, 179.0))
                        env.camera.lookat[:] = camera_position + _camera_forward(env) * float(env.camera.distance)
                    else:
                        env.camera.azimuth = float(env.camera.azimuth) + float(dx) * scale
                        env.camera.elevation = float(np.clip(float(env.camera.elevation) - float(dy) * scale, -179.0, 179.0))
                    view_id += 1
                    mark_dirty()
                return
            if event == cv2.EVENT_MOUSEMOVE and bool(mouse_state["middle_dragging"]):
                last_x, last_y = mouse_state["last_xy"]
                dx = int(x) - int(last_x)
                dy = int(y) - int(last_y)
                mouse_state["last_xy"] = (int(x), int(y))
                if dx or dy:
                    pan_camera_pixels(dx, dy)
                    view_id += 1
                    mark_dirty()
                return
            if event == cv2.EVENT_MOUSEWHEEL:
                delta = _mouse_wheel_delta(_flags)
                factor = max(float(args.wheel_zoom_factor), 1.01)
                if delta > 0:
                    set_camera_distance(float(env.camera.distance) / factor)
                elif delta < 0:
                    set_camera_distance(float(env.camera.distance) * factor)
                view_id += 1
                mark_dirty()
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                picked = _select_visible_point(env, (x, y), scene_option)
                if picked is None:
                    print(f"No visible MuJoCo geometry at pixel ({x}, {y}); rotate/zoom and click on the vessel surface.")
                    return
                surface_point, bodyid, geomid, geom_name = picked
                reproj_error_px, reproj_pixel = _reprojection_error_px(
                    env,
                    surface_point,
                    (int(x), int(y)),
                    int(args.render_width),
                    int(args.render_height),
                )
                reproj_text = (
                    "reproject=unavailable"
                    if reproj_error_px is None
                    else f"reproject_error={reproj_error_px:.1f}px projected={reproj_pixel}"
                )
                if pick_mode == "center":
                    old_position = _camera_position(env)
                    env.camera.lookat[:] = surface_point.astype(np.float32)
                    _set_camera_from_position(env, old_position)
                    pick_mode = "via"
                    view_id += 1
                    mark_dirty()
                    print(f"set camera center geom={geom_name or geomid} point={surface_point.tolist()} {reproj_text}")
                    return
                if pick_mode == "piper_exit":
                    piper_exit = {
                        "pixel": [int(x), int(y)],
                        "view_id": int(view_id),
                        "body_id": int(bodyid),
                        "geom_id": int(geomid),
                        "geom_name": geom_name,
                        "source": "mujoco_mjv_select",
                        "point": surface_point.astype(float).tolist(),
                        "frame": "scene",
                        "reproject_error_px": reproj_error_px,
                        "reproject_pixel": reproj_pixel,
                    }
                    pick_mode = "via"
                    print(f"set Piper visual exit geom={geom_name or geomid} point={surface_point.tolist()} {reproj_text}")
                    return
                if not allow_any_geom and geom_name not in allowed_geoms:
                    print(
                        f"Rejected geom={geom_name or geomid} at pixel ({x}, {y}); "
                        f"allowed={sorted(allowed_geoms)}. Click directly on the visible vessel mesh."
                    )
                    return
                progress, center_point, surface_to_center_distance = _route_center_from_surface_point(env, surface_point)
                point = center_point if args.via_point_mode == "route-center" else surface_point
                clicks.append(
                    {
                        "pixel": [int(x), int(y)],
                        "view_id": int(view_id),
                        "progress": float(progress),
                        "body_id": int(bodyid),
                        "geom_id": int(geomid),
                        "geom_name": geom_name,
                        "source": "mujoco_mjv_select",
                        "point": point.astype(float).tolist(),
                        "point_semantics": args.via_point_mode,
                        "surface_point": surface_point.astype(float).tolist(),
                        "center_point": center_point.astype(float).tolist(),
                        "surface_to_center_distance": float(surface_to_center_distance),
                        "reproject_error_px": reproj_error_px,
                        "reproject_pixel": reproj_pixel,
                        "frame": "scene",
                    }
                )
                print(
                    f"picked #{len(clicks)} geom={geom_name or geomid} progress={progress:.3f} "
                    f"saved={args.via_point_mode} surface_to_center={surface_to_center_distance:.4f} "
                    f"point={point.tolist()} {reproj_text}"
                )

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, args.render_width, args.render_height)
        cv2.setMouseCallback(WINDOW, on_mouse)
        while True:
            if bool(mouse_state["dirty"]):
                image = render_current_view()
                mouse_state["dirty"] = False
            camera_text = (
                f"az={float(env.camera.azimuth):.1f} elev={float(env.camera.elevation):.1f} "
                f"dist={float(env.camera.distance):.3f} cam={camera_mode} mode={pick_mode} "
                f"piper_exit={piper_exit is not None} picks={len(clicks)}"
            )
            cv2.imshow(
                WINDOW,
                _draw_overlay(
                    image,
                    clicks,
                    piper_exit,
                    view_id,
                    camera_text=camera_text,
                ),
            )
            key = cv2.waitKeyEx(30)
            if key in (ord("q"), 27):
                break
            if key in (8, 127) and clicks:
                removed = clicks.pop()
                print(f"removed progress={removed['progress']:.3f}")
            if key in (ord("p"), ord("P")):
                pick_mode = "piper_exit"
                print("Next left click sets wire_visual_piper_exit_point from a visible surface. Prefer G for the gripper-center outlet.")
            if key in (ord("g"), ord("G")):
                try:
                    point, source_mode, source_names = _auto_piper_exit_point(
                        env,
                        mode=args.auto_piper_exit_mode,
                        body_names=auto_piper_exit_bodies,
                        geom_names=auto_piper_exit_geoms,
                        offset=auto_piper_exit_offset,
                    )
                except Exception as exc:
                    print(f"Failed to compute automatic Piper exit: {exc}")
                else:
                    piper_exit = {
                        "view_id": int(view_id),
                        "source": f"auto_{source_mode}",
                        "source_names": source_names,
                        "offset": auto_piper_exit_offset.astype(float).tolist(),
                        "point": point.astype(float).tolist(),
                        "frame": "scene",
                    }
                    pick_mode = "via"
                    print(
                        f"set Piper visual exit from {source_mode} {source_names}: "
                        f"point={point.astype(float).tolist()} offset={auto_piper_exit_offset.astype(float).tolist()}"
                    )
            if key in (ord("c"), ord("C")):
                pick_mode = "center"
                print("Next left click sets camera rotation center/lookat.")
            if key in (ord("f"), ord("F")):
                camera_mode = "free-look" if camera_mode == "orbit" else "orbit"
                print(f"camera mode: {camera_mode}")
            if key in (ord("x"), ord("X")):
                piper_exit = None
                pick_mode = "via"
                print("cleared Piper visual exit point.")
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
            elif key in (ord("+"), ord("=")):
                set_camera_distance(float(env.camera.distance) / max(float(args.zoom_factor), 1.01))
                rerender = True
            elif key in (ord("-"), ord("_")):
                set_camera_distance(float(env.camera.distance) * max(float(args.zoom_factor), 1.01))
                rerender = True
            elif key in (2424832, 81):  # left arrow
                right, _up = _camera_axes(env)
                env.camera.lookat[:] = np.asarray(env.camera.lookat, dtype=np.float32) - right * float(args.pan_step)
                rerender = True
            elif key in (2555904, 83):  # right arrow
                right, _up = _camera_axes(env)
                env.camera.lookat[:] = np.asarray(env.camera.lookat, dtype=np.float32) + right * float(args.pan_step)
                rerender = True
            elif key in (2490368, 82):  # up arrow
                _right, up = _camera_axes(env)
                env.camera.lookat[:] = np.asarray(env.camera.lookat, dtype=np.float32) + up * float(args.pan_step)
                rerender = True
            elif key in (2621440, 84):  # down arrow
                _right, up = _camera_axes(env)
                env.camera.lookat[:] = np.asarray(env.camera.lookat, dtype=np.float32) - up * float(args.pan_step)
                rerender = True
            if rerender:
                view_id += 1
                mark_dirty()
            if key in (10, 13):
                if not clicks:
                    print("No points selected; route config not changed.")
                    continue
                route_config["wire_visual_via_points"] = [
                    {
                        "frame": "scene",
                        "point": row["point"],
                        "progress": row["progress"],
                        "surface_point": row.get("surface_point"),
                        "surface_frame": "scene",
                    }
                    for row in clicks
                ]
                if piper_exit is not None:
                    route_config["wire_visual_piper_exit_point"] = {
                        "frame": "scene",
                        "point": piper_exit["point"],
                    }
                    if "source" in piper_exit:
                        route_config["wire_visual_piper_exit_point"]["source"] = piper_exit["source"]
                    if "source_names" in piper_exit:
                        route_config["wire_visual_piper_exit_point"]["source_names"] = piper_exit["source_names"]
                    if "offset" in piper_exit:
                        route_config["wire_visual_piper_exit_point"]["offset"] = piper_exit["offset"]
                else:
                    route_config.pop("wire_visual_piper_exit_point", None)
                route_config["_wire_visual_via_note"] = (
                    "Visual-only guidewire tail points picked from rendered MuJoCo image. "
                    "wire_visual_piper_exit_point is the visual guidewire outlet, distinct from robot Piper TCP. "
                    "Clicked vessel-surface points are saved only as selection evidence; rendering uses the mapped "
                    "registered-route center/progress points, including the first vessel entry point."
                )
                _save_json(route_path, route_config)
                _save_json(Path(args.out_picks), {"camera": args.camera, "task": args.task, "piper_exit": piper_exit, "picks": clicks})
                print(f"saved {len(clicks)} via points to {route_path}")
                print(f"saved pick log to {args.out_picks}")
                break
    finally:
        _restore_visibility(env)
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
