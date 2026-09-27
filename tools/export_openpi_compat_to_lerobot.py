from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPECTED_SCHEMA = "project_2026_openpi_compat_pack_v0"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def import_lerobot_dataset():
    errors: list[str] = []
    try:
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        errors.append(f"lerobot.datasets: {exc!r}")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        errors.append(f"lerobot.datasets.lerobot_dataset: {exc!r}")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except Exception as exc:
        errors.append(f"lerobot.common.datasets.lerobot_dataset: {exc!r}")
    raise ImportError(
        "Could not import LeRobotDataset. Tried: "
        "lerobot.datasets.LeRobotDataset, lerobot.datasets.lerobot_dataset.LeRobotDataset, "
        "and lerobot.common.datasets.lerobot_dataset.LeRobotDataset. "
        f"Errors: {errors}"
    )


def resolve_pack(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(f"unexpected schema: {manifest.get('schema')!r}")
    return manifest_path.parent, manifest


def resolve_ref(base: Path, ref: str) -> Path:
    path = Path(ref.replace("\\", "/"))
    if path.is_absolute():
        return path
    direct = base / path
    if direct.exists():
        return direct
    return path


def load_rgb(path: Path, image_size: int | None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        image = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    return image.astype(np.uint8, copy=False)


def source_episode(row: dict[str, Any], fallback: int) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    episode = source.get("episode") or row.get("sample_id") or f"episode_{fallback:06d}"
    return str(episode)


def source_episode_split(rows: list[tuple[int, dict[str, Any]]]) -> str | None:
    splits = {str(row.get("split")) for _, row in rows if row.get("split") is not None}
    if len(splits) > 1:
        raise ValueError(f"source episode contains multiple split labels: {sorted(splits)}")
    return next(iter(splits)) if splits else None


def task_text(row: dict[str, Any]) -> str:
    text = row.get("language_instruction")
    if isinstance(text, str) and text.strip():
        return text.strip()
    task = row.get("task")
    if task == "left":
        return "Use Piper insertion and Elite magnetic guidance to steer the guidewire into the left branch."
    if task == "right":
        return "Use Piper insertion and Elite magnetic guidance to steer the guidewire into the right branch."
    return "Use Piper insertion and Elite magnetic guidance to steer the guidewire."


def make_features(image_size: int) -> dict[str, Any]:
    return {
        "observation.images.side": {
            "dtype": "image",
            "shape": (int(image_size), int(image_size), 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.top": {
            "dtype": "image",
            "shape": (int(image_size), int(image_size), 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (32,),
            "names": ["project2026_state_32"],
        },
        "action": {
            "dtype": "float32",
            "shape": (32,),
            "names": ["project2026_action_32"],
        },
        "elite_tcp_delta_6d": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["elite_tcp_delta_6d"],
        },
        "piper_intent_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["piper_intent_id"],
        },
    }


def create_lerobot_dataset(*, repo_id: str, root: Path, fps: int, image_size: int, use_videos: bool, force: bool):
    if root.exists():
        if not force:
            raise FileExistsError(f"output root exists; pass --force to overwrite: {root}")
        shutil.rmtree(root)
    LeRobotDataset = import_lerobot_dataset()
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=int(fps),
        robot_type="project2026_guidewire_dual_arm",
        features=make_features(image_size),
        use_videos=bool(use_videos),
    )


def add_episode_frames(
    dataset: Any,
    *,
    episode_rows: list[tuple[int, dict[str, Any]]],
    arrays: dict[str, np.ndarray],
    image_root: Path,
    image_size: int,
) -> int:
    for idx, row in episode_rows:
        side_path = image_root / str(row["observation.images.side"]).replace("\\", "/")
        top_path = image_root / str(row["observation.images.top"]).replace("\\", "/")
        frame = {
            "observation.images.side": load_rgb(side_path, image_size),
            "observation.images.top": load_rgb(top_path, image_size),
            "observation.state": arrays["state_32"][idx].astype(np.float32),
            "action": arrays["action_32"][idx].astype(np.float32),
            "elite_tcp_delta_6d": arrays["elite_tcp_delta_6d"][idx].astype(np.float32),
            "piper_intent_id": np.asarray([int(arrays["piper_intent_id"][idx])], dtype=np.int64),
            "task": task_text(row),
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    return len(episode_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a project_2026 OpenPI-compatible pack to an official LeRobotDataset directory. "
            "This preserves project action semantics by keeping action_32 as compatibility action and "
            "storing piper_intent_id / elite_tcp_delta_6d as explicit extra fields."
        )
    )
    parser.add_argument("pack", type=Path, help="OpenPI-compatible pack directory or manifest.")
    parser.add_argument("--out", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--repo-id", default="project2026/guidewire-openpi-compat-smoke")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-videos", action="store_true", help="Create image-backed dataset without video encoding.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-frames-per-episode", type=int)
    args = parser.parse_args()

    pack_dir, manifest = resolve_pack(args.pack)
    # Real captures have an irregular physical clock and an explicit train-only
    # use scope. Do not accidentally relabel them as 15 FPS or truncate events.
    real_prototype = manifest.get("training_use", {}).get("scope") == "user_authorized_real10_train_only_prototype"
    if real_prototype:
        if args.max_episodes is not None or args.max_frames_per_episode is not None:
            raise ValueError("real10 prototype export uses all complete episode transitions")
        if args.fps != manifest["timing"]["lerobot_index_fps"]:
            raise ValueError("real10 requires --fps 5 as an indexing clock, not a physical sample rate")
        if args.image_size != manifest["input_preprocessing"]["image_size"]:
            raise ValueError("keep the real10 pack image size unchanged during export")
    arrays_path = pack_dir / str(manifest.get("output", {}).get("arrays_npz", "openpi_arrays.npz"))
    index_path = pack_dir / str(manifest.get("output", {}).get("index_jsonl", "index.jsonl"))
    image_root = resolve_ref(pack_dir, str(manifest.get("output", {}).get("image_root", "")))
    if not arrays_path.exists():
        raise FileNotFoundError(f"arrays not found: {arrays_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"image root not found: {image_root}")

    arrays = {name: value for name, value in np.load(arrays_path).items()}
    rows = read_jsonl(index_path)
    if len(rows) != int(arrays["state_32"].shape[0]):
        raise ValueError(f"index rows {len(rows)} != array rows {arrays['state_32'].shape[0]}")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[source_episode(row, idx)].append((idx, row))
    episode_keys = list(grouped.keys())
    if args.max_episodes is not None:
        episode_keys = episode_keys[: max(int(args.max_episodes), 0)]

    dataset = create_lerobot_dataset(
        repo_id=args.repo_id,
        root=args.out,
        fps=args.fps,
        image_size=args.image_size,
        use_videos=not args.no_videos,
        force=args.force,
    )

    episode_lengths: list[int] = []
    episode_exports: list[dict[str, Any]] = []
    try:
        for episode_key in episode_keys:
            episode_rows = grouped[episode_key]
            if args.max_frames_per_episode is not None:
                episode_rows = episode_rows[: max(int(args.max_frames_per_episode), 0)]
            if not episode_rows:
                continue
            frame_count = add_episode_frames(
                dataset,
                episode_rows=episode_rows,
                arrays=arrays,
                image_root=image_root,
                image_size=args.image_size,
            )
            episode_lengths.append(frame_count)
            lerobot_episode_index = len(episode_exports)
            episode_exports.append(
                {
                    "source_episode": episode_key,
                    "lerobot_episode_index": int(lerobot_episode_index),
                    "split": source_episode_split(episode_rows),
                    "frames": int(frame_count),
                }
            )
        if hasattr(dataset, "finalize"):
            dataset.finalize()
    except Exception:
        if hasattr(dataset, "cleanup_interrupted_episode"):
            dataset.cleanup_interrupted_episode()
        raise

    export_manifest = {
        "schema": "project_2026_lerobot_export_v0",
        "source_pack": str(args.pack),
        "source_pack_schema": manifest.get("schema"),
        "repo_id": args.repo_id,
        "root": str(args.out),
        "fps": int(args.fps),
        "image_size": int(args.image_size),
        "use_videos": not args.no_videos,
        "episodes": len(episode_lengths),
        "frames": int(sum(episode_lengths)),
        "episode_length_min": int(min(episode_lengths)) if episode_lengths else 0,
        "episode_length_max": int(max(episode_lengths)) if episode_lengths else 0,
        "episode_exports": episode_exports,
        "train_episode_indices": [
            item["lerobot_episode_index"] for item in episode_exports if item["split"] == "train"
        ],
        "val_episode_indices": [
            item["lerobot_episode_index"] for item in episode_exports if item["split"] == "val"
        ],
        "features": make_features(args.image_size),
        "semantics": {
            "action": "OpenPI/LeRobot compatibility action_32; padded dims are not robot controls",
            "preferred_project_target": "Elite TCP delta regression plus Piper intent classification",
            "extra_fields": ["elite_tcp_delta_6d", "piper_intent_id"],
        },
    }
    if real_prototype:
        for key in ("adapter_version", "training_use", "timing", "input_preprocessing", "openpi_compat"):
            export_manifest[key] = manifest[key]
        shutil.copy2(index_path, args.out / "project2026_source_index.jsonl")
        export_manifest["source_timing_index"] = "project2026_source_index.jsonl"
    write_json(args.out / "project2026_lerobot_export_manifest.json", export_manifest)
    print(f"wrote LeRobot dataset: {args.out}")
    print(f"episodes={len(episode_lengths)} frames={sum(episode_lengths)} use_videos={not args.no_videos}")


if __name__ == "__main__":
    main()
