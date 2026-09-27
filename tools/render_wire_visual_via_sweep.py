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


def _parse_values(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split() if item.strip()]


def _caption(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (8, 8), (210, 42), (255, 255, 255), -1)
    cv2.rectangle(out, (8, 8), (210, 42), (0, 0, 0), 1)
    cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return out


def _sheet(images: list[np.ndarray], cols: int) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    rows = int(np.ceil(len(images) / cols))
    height, width = images[0].shape[:2]
    blank = np.full_like(images[0], 245)
    padded = images + [blank] * (rows * cols - len(images))
    row_images = []
    for row in range(rows):
        row_images.append(np.hstack(padded[row * cols : (row + 1) * cols]))
    return np.vstack(row_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render visual guidewire tail via-progress candidates as contact sheets.")
    parser.add_argument("--out", default="docs/_wire_visual_via_sweep")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--start-fraction", type=float, default=0.58)
    parser.add_argument("--progresses", default="20 30 40 50 60")
    parser.add_argument("--camera-config", default="simulation/camera_configs/mujoco_camera_top_manual_v1.json")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--wire-segments", type=int, default=160)
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progresses = _parse_values(args.progresses)
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
        meta = {
            "task": args.task,
            "start_fraction": float(args.start_fraction),
            "path_progress": float(env.path_progress_float),
            "piper_tcp": np.asarray(env.piper_pose, dtype=np.float32).astype(float).tolist(),
            "tip": np.asarray(env.tip, dtype=np.float32).astype(float).tolist(),
            "candidates": [],
        }
        sheets: dict[str, list[np.ndarray]] = {"top": [], "side": [], "overview": []}
        for progress in progresses:
            clipped = float(np.clip(progress, 0.0, env.path_progress_float))
            env.reference_env.route_config["wire_visual_via_progresses"] = [clipped]
            env._rebuild_visual_tail()
            env._sync_mujoco()
            progress_dir = out_dir / f"progress_{clipped:.1f}".replace(".", "p")
            progress_dir.mkdir(exist_ok=True)
            candidate = {"progress": clipped, "images": {}}
            for camera in ("top", "side", "overview"):
                image = env.render_camera(camera)
                image = _caption(image, f"via={clipped:.1f}")
                image_path = progress_dir / f"{camera}.png"
                cv2.imwrite(str(image_path), image)
                sheets[camera].append(image)
                candidate["images"][camera] = str(image_path)
            meta["candidates"].append(candidate)
        for camera, images in sheets.items():
            cv2.imwrite(str(out_dir / f"{camera}_sheet.png"), _sheet(images, cols=max(int(args.cols), 1)))
        (out_dir / "sweep.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"saved via sweep to {out_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
