"""Pinned public Push-T acquisition/audit only; no optimizer or environment.

Source bytes are immutable. Split and train-only numeric statistics live in a
separate generated output directory. No project/real datasets are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs/pusht-dataset-source-v1.json"
DEFAULT_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_7628202a")
REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
REQUIRED_FILES = {".gitattributes", "README.md", "data/chunk-000/file-000.parquet",
                  "meta/episodes/chunk-000/file-000.parquet", "meta/info.json", "meta/stats.json",
                  "meta/tasks.parquet", "videos/observation.image/chunk-000/file-000.mp4"}
SPLIT_SALT = "pusht_bc_act_episode_split_v1:20260913"
IMAGE_STATS = {"mean": [[[0.485]], [[0.456]], [[0.406]]],
               "std": [[[0.229]], [[0.224]], [[0.225]]]}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(path=SOURCE):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    if source["repo_id"] != "lerobot/pusht" or source["revision"] != REVISION:
        raise ValueError("unexpected public dataset/revision")
    if (set(source["files"]) != REQUIRED_FILES or source["file_count"] != 8
            or source["total_bytes"] != 7686801
            or sum(x["size"] for x in source["files"].values()) != source["total_bytes"]):
        raise ValueError("incomplete source manifest")
    for name, spec in source["files"].items():
        safe_path(Path("."), name)
        if len(spec["sha256"]) != 64 or spec["size"] < 0:
            raise ValueError("invalid file pin")
    return source


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("unsafe source path")
    target = (Path(root) / relative).resolve()
    if not target.is_relative_to(Path(root).resolve()) or target == Path(root).resolve():
        raise ValueError("source path escapes root")
    return target


def verify_files(root, source):
    verified = {}
    for directory in ("data", "meta", "videos"):
        base = Path(root) / directory
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and path.relative_to(root).as_posix() not in source["files"]:
                    raise ValueError(f"unlisted source file: {path.relative_to(root)}")
    for name, spec in source["files"].items():
        path = safe_path(root, name)
        if not path.is_file() or path.stat().st_size != spec["size"]:
            raise ValueError(f"missing/size mismatch: {name}")
        digest = file_hash(path)
        if digest != spec["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
        verified[name] = {"size": path.stat().st_size, "sha256": digest}
    return verified


def acquire(root, source):
    """Explicit stage only. Valid files reused; mismatches/partials never replaced."""
    total = sum(item["size"] for item in source["files"].values())
    if total > 100_000_000 or any(item["size"] > 100_000_000 for item in source["files"].values()):
        raise ValueError("large transfer is user-run; this entrypoint is small-download only")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < total * 2 + 100_000_000:
        raise ValueError("insufficient free space")
    for name, spec in source["files"].items():
        path = safe_path(root, name)
        if path.exists():
            if path.stat().st_size != spec["size"] or file_hash(path) != spec["sha256"]:
                raise ValueError(f"existing source mismatch: {name}")
            continue
        partial = path.with_name(path.name + ".partial")
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/lerobot/pusht/resolve/{REVISION}/{name}"
        # Exclusive creation retains interrupted transfer evidence for inspection.
        with partial.open("xb") as out, urllib.request.urlopen(url, timeout=30) as response:
            received = 0
            while block := response.read(1024 * 1024):
                received += len(block)
                if received > spec["size"]:
                    raise ValueError(f"oversize payload: {name}")
                out.write(block)
        if partial.stat().st_size != spec["size"] or file_hash(partial) != spec["sha256"]:
            raise ValueError(f"download pin mismatch: {name}; partial retained")
        if path.exists():
            raise FileExistsError(path)
        partial.rename(path)
        print(f"verified download {name} bytes={spec['size']}", flush=True)
    return verify_files(root, source)


def episode_split(episode_ids, val_count=20):
    ids = list(episode_ids)
    if len(ids) != len(set(ids)) or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("episode IDs must be unique nonnegative integers")
    if not 0 < val_count < len(ids):
        raise ValueError("both splits must be nonempty")
    ranked = sorted(ids, key=lambda e: hashlib.sha256(f"{SPLIT_SALT}:{e}".encode()).hexdigest())
    val = sorted(ranked[:val_count])
    return {"method": "sha256_salted_episode_rank", "salt": SPLIT_SALT,
            "train_episodes": sorted(set(ids) - set(val)), "val_episodes": val}


def train_statistics(states, actions, episodes, split):
    """Float64 population moments; future processors must use these, not meta/stats."""
    episodes = np.asarray(episodes)
    train = np.isin(episodes, split["train_episodes"])
    if set(split["train_episodes"]) & set(split["val_episodes"]):
        raise ValueError("split overlap")
    if not train.any() or not np.isin(episodes, split["train_episodes"] + split["val_episodes"]).all():
        raise ValueError("missing/unassigned records")
    stats = {"observation.image": IMAGE_STATS}
    for key, values in (("observation.state", states), ("action", actions)):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(episodes), 2) or not np.isfinite(values).all():
            raise ValueError(f"invalid numeric feature: {key}")
        selected = values[train]
        std = selected.std(axis=0, ddof=0)
        if np.any(std <= 1e-8):
            raise ValueError(f"degenerate training normalization: {key}")
        stats[key] = {"mean": selected.mean(axis=0).tolist(), "std": std.tolist()}
    return stats, int(train.sum())


def validate_records(table, episode_table, info):
    """Structural and timing validation, no hidden-state interpretation."""
    episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    meta = episode_table.to_pylist()
    if len(episodes) != 25650 or info["total_frames"] != len(episodes) or len(meta) != 206:
        raise ValueError("unexpected record/episode count")
    if sorted(row["episode_index"] for row in meta) != list(range(206)):
        raise ValueError("metadata episode coverage/duplicates")
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("record index not consecutive")
    for row in meta:
        start, end, e = row["dataset_from_index"], row["dataset_to_index"], row["episode_index"]
        selected = np.flatnonzero(episodes == e)
        if not np.array_equal(selected, np.arange(start, end)) or len(selected) != row["length"]:
            raise ValueError(f"episode range mismatch: {e}")
        if not np.array_equal(frames[selected], np.arange(len(selected))):
            raise ValueError(f"frame reset mismatch: {e}")
        if not np.allclose(timestamps[selected], frames[selected] / 10, atol=1e-4, rtol=0):
            raise ValueError(f"timestamp mismatch: {e}")
    if set(episodes.tolist()) != set(range(206)):
        raise ValueError("data episode coverage")
    if not np.isfinite(actions).all() or np.any(actions < 0) or np.any(actions > 512):
        raise ValueError("native action bounds mismatch")
    return states, actions, episodes


def decode_smoke(root, split):
    """Six deterministic first/last frame probes using the future loader backend."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("lerobot/pusht", root=root, revision=REVISION,
                       delta_timestamps={"action": [i / 10 for i in range(16)]},
                       video_backend="pyav")
    probes = []
    # Fixed before observing images/outcomes, covering train/val and boundary padding.
    for e in [split["train_episodes"][0], split["val_episodes"][0], 205]:
        row = ds.meta.episodes[e]
        for index in [row["dataset_from_index"], row["dataset_to_index"] - 1]:
            sample = ds[index]
            image = sample["observation.image"]
            action = sample["action"]
            pad = sample["action_is_pad"]
            if list(image.shape) != [3, 96, 96] or list(action.shape) != [16, 2]:
                raise ValueError("decoded shape mismatch")
            if not np.isfinite(image.numpy()).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("decoded image not finite [0,1]")
            if (list(pad.shape) != [16] or pad[0].item()
                    or (index == row["dataset_from_index"] and pad.sum().item() != 0)
                    or (index == row["dataset_to_index"] - 1 and pad.sum().item() != 15)):
                raise ValueError("chunk crosses episode or incorrect padding")
            probes.append({"episode": e, "index": index, "image_shape": list(image.shape),
                           "image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                           "action_shape": list(action.shape), "padded_actions": int(pad.sum())})
    return {"status": "passed", "backend": "pyav", "probes": probes,
            "all_images_decoded": False}


def audit(root, source, output, decode=False):
    # Check source bytes before importing the dataset loader or creating derived artifacts.
    verified = verify_files(root, source)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("derived outputs must be outside immutable source root")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        import pyarrow.parquet as pq
        info = json.loads((root / "meta/info.json").read_text())
        if info != source["metadata_expected"]:
            raise ValueError("info differs from pinned metadata")
        if info["codebase_version"] != "v3.0" or info["fps"] != 10 or info["total_episodes"] != 206:
            raise ValueError("metadata compatibility failure")
        table = pq.read_table(root / "data/chunk-000/file-000.parquet",
                              columns=["episode_index", "frame_index", "index", "timestamp",
                                       "observation.state", "action"])
        metadata = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet",
                                 columns=["episode_index", "dataset_from_index", "dataset_to_index", "length"])
        states, actions, episodes = validate_records(table, metadata, info)
        split = episode_split(list(range(206)))
        stats, train_frames = train_statistics(states, actions, episodes, split)
        split.update({"source_revision": REVISION, "train_frames": train_frames,
                      "val_frames": len(episodes) - train_frames})
        decode_result = decode_smoke(root, split) if decode else {"status": "not_run"}
        report = {"status": "passed", "stage": "data_preparation_only", "source_revision": REVISION,
                  "source_manifest_sha256": canonical_hash(source), "source_files": verified,
                  "split_sha256": canonical_hash(split), "stats_sha256": canonical_hash(stats),
                  "split": split, "normalization": stats,
                  "normalization_source": "train_episode_numeric_population_moments_and_fixed_ImageNet_image_stats",
                  "global_meta_stats_used": False, "policy_observation_keys": ["observation.image", "observation.state"],
                  "target_keys": ["action"], "decode_smoke": decode_result,
                  "optimizer_steps": 0, "environment_steps": 0,
                  "training_started": False, "training_authorized_by_report": False,
                  "guidewire_or_real_capability_claim_allowed": False}
        for name, value in [("split.json", split), ("normalization.json", stats)]:
            (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                  "optimizer_steps": 0, "training_started": False}
    report["runtime_seconds"] = time.monotonic() - started
    report["source_path"] = str(root.resolve())
    report["tool_sha256"] = file_hash(__file__)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["acquire", "audit"], required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=REPO / "simulation_output/pusht_bc_act_data_gate_v1")
    parser.add_argument("--decode-smoke", action="store_true")
    args = parser.parse_args()
    source = source_manifest()
    if args.stage == "acquire":
        if args.decode_smoke:
            parser.error("decode smoke is audit-only")
        result = {"status": "passed", "verified_files": acquire(args.root, source), "training_started": False}
    else:
        result = audit(args.root, source, args.output, args.decode_smoke)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
