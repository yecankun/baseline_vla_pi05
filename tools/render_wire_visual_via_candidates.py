from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.tip_guided_wire_env import TipGuidedWireConfig, TipGuidedWireEnv
from simulation.visual_distance_estimator import _project_world_points_unclipped


def _parse_progresses(value: str, current_progress: float) -> list[float]:
    if value:
        result = [float(item) for item in value.replace(",", " ").split()]
    else:
        result = list(np.arange(0.0, current_progress + 1e-6, 5.0))
        if result[-1] < current_progress:
            result.append(float(current_progress))
    return sorted({round(float(np.clip(item, 0.0, current_progress)), 3) for item in result})


def _draw_labels(image: np.ndarray, points_px: np.ndarray, valid: np.ndarray, labels: list[str]) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    for point, ok, label in zip(points_px, valid, labels):
        if not bool(ok):
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        cv2.circle(out, (x, y), 5, (0, 220, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render numbered route-progress candidates for wire visual via-point selection.")
    parser.add_argument("--out", default="docs/_wire_visual_via_candidates")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--progresses", default="", help="Optional comma/space-separated progress values. Default: every 5 indices to current tip.")
    parser.add_argument("--camera-config", default="simulation/camera_configs/mujoco_camera_top_manual_v1.json")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--wire-segments", type=int, default=160)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = TipGuidedWireConfig(
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
    try:
        env.reset(options={"task": args.task, "start_fraction": args.start_fraction})
        progresses = _parse_progresses(args.progresses, float(env.path_progress_float))
        route_points = []
        rows = []
        for progress in progresses:
            center, _tangent, _n1, _n2, _radius = env._local_path_frame(progress)
            route_points.append(center)
            rows.append({"label": str(int(round(progress))), "progress": float(progress), "point": center.astype(float).tolist()})
        points = np.asarray(route_points, dtype=np.float32)
        labels = [row["label"] for row in rows]
        for camera in ("top", "side", "overview"):
            image = env.render_camera(camera)
            env._configure_free_camera(camera)
            projected, valid = _project_world_points_unclipped(env, points, args.render_width, args.render_height)
            annotated = _draw_labels(image, projected, valid, labels)
            cv2.imwrite(str(out_dir / f"{camera}_candidates.png"), annotated)
        meta = {
            "task": args.task,
            "start_fraction": float(args.start_fraction),
            "path_progress": float(env.path_progress_float),
            "piper_tcp": np.asarray(env.piper_pose, dtype=np.float32).astype(float).tolist(),
            "tip": np.asarray(env.tip, dtype=np.float32).astype(float).tolist(),
            "candidates": rows,
        }
        (out_dir / "candidates.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved candidates to {out_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
