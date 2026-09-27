import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def list_episode_frames(episode_dir: Path):
    side_dir = episode_dir / "frames" / "side"
    top_dir = episode_dir / "frames" / "top"
    if not side_dir.exists() or not top_dir.exists():
        raise FileNotFoundError(f"Missing frame dirs under {episode_dir}")
    steps = sorted(int(p.stem) for p in side_dir.glob("*.png"))
    return [(step, side_dir / f"{step:06d}.png", top_dir / f"{step:06d}.png") for step in steps]


def make_panel(side_img, top_img, width=512, height=512):
    side = cv2.resize(side_img, (width, height), interpolation=cv2.INTER_AREA)
    top = cv2.resize(top_img, (width, height), interpolation=cv2.INTER_AREA)
    return np.concatenate([side, top], axis=1)


def draw_header(canvas, text):
    bar_h = 56
    out = np.full((canvas.shape[0] + bar_h, canvas.shape[1], 3), 248, dtype=np.uint8)
    out[bar_h:, :] = canvas
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, bar_h - 1), (238, 238, 238), -1)
    cv2.putText(out, text, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def build_summary_image(episodes, width=384, height=384):
    rows = []
    for epi in episodes:
        frames = list_episode_frames(epi["dir"])
        picks = [frames[0], frames[len(frames) // 2], frames[-1]]
        cells = []
        for step, side_path, top_path in picks:
            side = cv2.imread(str(side_path), cv2.IMREAD_COLOR)
            top = cv2.imread(str(top_path), cv2.IMREAD_COLOR)
            panel = make_panel(side, top, width=width, height=height)
            header = f"{epi['episode']}  {epi['task']}  step={step}  success={epi['success']}"
            cells.append(draw_header(panel, header))
        rows.append(np.concatenate(cells, axis=0))
    return np.concatenate(rows, axis=1)


def write_episode_video(episode, out_path: Path, fps: int = 18, max_frames: int = 180):
    frames = list_episode_frames(episode["dir"])
    if not frames:
        raise RuntimeError(f"No frames found for {episode['episode']}")
    stride = max(1, math.ceil(len(frames) / max_frames))
    selected = frames[::stride]
    first_side = cv2.imread(str(selected[0][1]), cv2.IMREAD_COLOR)
    first_top = cv2.imread(str(selected[0][2]), cv2.IMREAD_COLOR)
    panel = draw_header(make_panel(first_side, first_top), f"{episode['episode']}  task={episode['task']}  success={episode['success']}")
    h, w = panel.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    for idx, (step, side_path, top_path) in enumerate(selected):
        side = cv2.imread(str(side_path), cv2.IMREAD_COLOR)
        top = cv2.imread(str(top_path), cv2.IMREAD_COLOR)
        frame = make_panel(side, top)
        label = f"{episode['episode']}  task={episode['task']}  step={step}/{frames[-1][0]}  frame={idx+1}/{len(selected)}"
        writer.write(draw_header(frame, label))
    writer.release()
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description="Randomly visualize a few dual-arm dataset episodes.")
    parser.add_argument("--manifest", default="simulation_output/dual_arm_dataset_success_recovery_v3/manifest.json")
    parser.add_argument("--out-dir", default="simulation_output/dual_arm_dataset_success_recovery_v3_visuals")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=180)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    episode_meta = []
    for epi in manifest["episodes"]:
        episode_dir = manifest_path.parent / epi["episode"]
        if episode_dir.exists():
            item = dict(epi)
            item["dir"] = episode_dir
            episode_meta.append(item)
    if not episode_meta:
        raise RuntimeError("No episode directories found for visualization")

    rng = random.Random(args.seed)
    chosen = rng.sample(episode_meta, k=min(args.episodes, len(episode_meta)))
    print("Selected episodes:")
    for epi in chosen:
        print(f"  {epi['episode']} task={epi['task']} success={epi['success']} steps={epi['steps']}")

    summary = build_summary_image(chosen)
    summary_path = out_dir / "summary.png"
    cv2.imwrite(str(summary_path), summary)

    video_paths = []
    for epi in chosen:
        video_path = out_dir / f"{epi['episode']}.mp4"
        nframes = write_episode_video(epi, video_path, fps=args.fps, max_frames=args.max_frames)
        video_paths.append((video_path, nframes))

    print(f"Saved summary: {summary_path}")
    for path, nframes in video_paths:
        print(f"Saved video: {path} ({nframes} frames)")


if __name__ == "__main__":
    main()
