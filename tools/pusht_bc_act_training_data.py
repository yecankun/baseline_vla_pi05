"""Pinned image/state/action-only Push-T adapter; no training or downloads.

The production loader validates the saved data gate and decodes the one pinned
AV1 video exactly once to an in-memory uint8 cache. Dataset diagnostics, rewards,
success flags and terminal labels are never read as observations or boundaries.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    from prepare_pusht_bc_act_data import (
        DEFAULT_ROOT, IMAGE_STATS, REPO, REVISION, SOURCE, canonical_hash,
        episode_split, file_hash, source_manifest, train_statistics,
        validate_records, verify_files,
    )
else:
    from .prepare_pusht_bc_act_data import (
        DEFAULT_ROOT, IMAGE_STATS, REPO, REVISION, SOURCE, canonical_hash,
        episode_split, file_hash, source_manifest, train_statistics,
        validate_records, verify_files,
    )


DEFAULT_GATE_DIR = REPO / "simulation_output/pusht_bc_act_data_gate_v1"
SCHEMA = "pusht_bc_act_training_data_v1"
FRAME_COUNT = 25650
CHUNK_SIZE = 16
RAW_IMAGE_SHAPE = (96, 96, 3)
NUMERIC_COLUMNS = ("episode_index", "frame_index", "index", "timestamp", "observation.state", "action")
EPISODE_COLUMNS = (
    "episode_index", "dataset_from_index", "dataset_to_index", "length",
    "videos/observation.image/chunk_index", "videos/observation.image/file_index",
    "videos/observation.image/from_timestamp", "videos/observation.image/to_timestamp",
)
GATE_PINS = {
    "report.json": "607f8b81bef37bbbb1b74ec00646a6062357cf06e501e32f0163834a7646eafb",
    "split.json": "24131e9146576d1aaa050b7726e8ae7c178a20ca92a63490ca9f1c74c5b22d59",
    "normalization.json": "01bb762f7174920e778326ac5585bc90eca64bd7d5374ef6e327a43baf781c89",
}
CODE_PINS = {
    "tools/prepare_pusht_bc_act_data.py": "363db452dd729f5642ea93dedc8848ffc847cbf563d3f31de9c58ca254d12e00",
    "tools/pusht_bc_act_models.py": "6e35404670e9e13c8f425f0c9d5be77091360401507eef8a08a6eb9b36888332",
    "docs/pusht-dataset-source-v1.json": "937cf36380eaf6c7623c4afc29a1120e00a461a61af78fea0be42146d116f016",
}


def verify_gate(gate_dir: Path = DEFAULT_GATE_DIR) -> tuple[dict, dict, dict, dict]:
    """Read only hash-pinned sidecars; never infer or rewrite split/statistics."""
    gate_dir = Path(gate_dir)
    for name, expected in GATE_PINS.items():
        if file_hash(gate_dir / name) != expected:
            raise ValueError(f"saved data gate SHA256 mismatch: {name}")
    for name, expected in CODE_PINS.items():
        if file_hash(REPO / name) != expected:
            raise ValueError(f"data/model preparation source SHA256 mismatch: {name}")
    report, split, stats = [json.loads((gate_dir / name).read_text(encoding="utf-8"))
                            for name in ("report.json", "split.json", "normalization.json")]
    if (report["status"] != "passed" or report["source_revision"] != REVISION
            or report["split"] != split or report["normalization"] != stats
            or report["split_sha256"] != canonical_hash(split)
            or report["stats_sha256"] != canonical_hash(stats)
            or report["global_meta_stats_used"] is not False
            or report["policy_observation_keys"] != ["observation.image", "observation.state"]
            or report["target_keys"] != ["action"]
            or report["decode_smoke"]["status"] != "passed"
            or report["decode_smoke"]["backend"] != "pyav"
            or len(report["decode_smoke"]["probes"]) != 6):
        raise ValueError("saved gate contract differs from the accepted interface")
    source = source_manifest(SOURCE)
    if (report["source_manifest_sha256"] != canonical_hash(source)
            or report["source_files"] != source["files"]
            or report["tool_sha256"] != CODE_PINS["tools/prepare_pusht_bc_act_data.py"]):
        raise ValueError("gate/source provenance binding mismatch")
    return report, split, stats, source


def _integer_vector(value: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu" or (not allow_empty and value.size == 0):
        raise ValueError(f"{name} must be a nonempty 1D integer array")
    if value.size and (np.any(value < 0) or np.any(value > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must contain nonnegative int64-compatible values")
    return value.astype(np.int64, copy=False)


def episode_end_indices(episodes: Any, frames: Any) -> np.ndarray:
    episodes = _integer_vector(episodes, "episode indices")
    frames = _integer_vector(frames, "frame indices")
    if frames.shape != episodes.shape:
        raise ValueError("episode/frame arrays differ in length")
    starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    ends = np.r_[starts[1:], len(episodes)]
    if len(np.unique(episodes[starts])) != len(starts):
        raise ValueError("one episode occurs in multiple noncontiguous blocks")
    result = np.empty(len(episodes), dtype=np.int64)
    for start, end in zip(starts, ends, strict=True):
        if not np.array_equal(frames[start:end], np.arange(end - start)):
            raise ValueError("frame indices must start at zero and be consecutive per episode")
        result[start:end] = end
    return result


def chunk_indices(indices: Any, end_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchors = _integer_vector(indices, "batch indices")
    end_indices = _integer_vector(end_indices, "episode end indices")
    if np.any(anchors >= len(end_indices)):
        raise ValueError("batch index is outside the dataset")
    if np.any(end_indices <= np.arange(len(end_indices))) or np.any(end_indices > len(end_indices)):
        raise ValueError("invalid exclusive episode-end bounds")
    future = anchors[:, None] + np.arange(CHUNK_SIZE, dtype=np.int64)[None, :]
    ends = end_indices[anchors, None]
    return np.minimum(future, ends - 1), future >= ends


def validate_statistics(stats: dict[str, Any]) -> None:
    if not isinstance(stats, dict) or set(stats) != {"observation.image", "observation.state", "action"}:
        raise ValueError("statistics must include only image/state/action")
    if stats["observation.image"] != IMAGE_STATS:
        raise ValueError("fixed ImageNet image statistics changed")
    for name in ("observation.state", "action"):
        group = stats[name]
        if not isinstance(group, dict) or set(group) != {"mean", "std"}:
            raise ValueError("numeric statistics require exactly mean/std")
        for key in ("mean", "std"):
            value = np.asarray(group[key], dtype=np.float64)
            if value.shape != (2,) or not np.isfinite(value).all() or (key == "std" and np.any(value <= 1e-8)):
                raise ValueError(f"invalid saved statistics: {name}.{key}")


def split_indices(episodes: np.ndarray, split: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train = _integer_vector(split["train_episodes"], "train episodes")
    val = _integer_vector(split["val_episodes"], "val episodes")
    if (len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val)
            or np.intersect1d(train, val).size
            or set(np.r_[train, val].tolist()) != set(episodes.tolist())):
        raise ValueError("episode split duplicates, overlaps, or omits data")
    train_rows = np.flatnonzero(np.isin(episodes, train)).astype(np.int64)
    val_rows = np.flatnonzero(np.isin(episodes, val)).astype(np.int64)
    if len(train_rows) != split["train_frames"] or len(val_rows) != split["val_frames"]:
        raise ValueError("saved split frame counts disagree with the actual episodes")
    return train_rows, val_rows


def float32_image_hash(image: np.ndarray) -> str:
    import torch
    if image.dtype != np.uint8 or image.shape != RAW_IMAGE_SHAPE:
        raise ValueError("probe image must be uint8 HWC96")
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255).contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def check_video_frame(frame: Any, index: int, count: int) -> np.ndarray:
    if index >= count or frame.pts is None or frame.time_base is None:
        raise ValueError("video frame count or PTS missing/out of bounds")
    timestamp = Fraction(frame.pts) * Fraction(frame.time_base)
    if timestamp != Fraction(index, 10):
        raise ValueError(f"video PTS does not equal global index/10: frame {index}")
    pixels = frame.to_ndarray(format="rgb24")
    if pixels.dtype != np.uint8 or pixels.shape != RAW_IMAGE_SHAPE:
        raise ValueError(f"decoded RGB shape/dtype mismatch at frame {index}")
    return pixels


def verify_image_probes(images: np.ndarray, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified = []
    seen = set()
    for probe in probes:
        index = probe["index"]
        if type(index) is not int or not 0 <= index < len(images) or index in seen:
            raise ValueError("duplicate or invalid pinned probe frame")
        seen.add(index)
        actual = float32_image_hash(images[index])
        if actual != probe["image_sha256"]:
            raise ValueError(f"sequential decode differs from the pinned LeRobot PyAV image: {index}")
        verified.append({"index": index, "image_sha256": actual, "exact_match": True})
    return verified


def _memory_check(cache_bytes: int) -> dict[str, Any]:
    required = cache_bytes + 256 * 1024 * 1024
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {"status": "available_memory_unknown", "required_bytes": required}
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip().split()[0]
              for line in path.read_text().splitlines() if ":" in line}
    available = int(fields["MemAvailable"]) * 1024
    if available < required:
        raise MemoryError(f"insufficient RAM for uint8 image cache: {available} < {required}")
    return {"status": "passed", "required_bytes": required, "available_bytes": available}


def decode_image_cache(video: Path, probes: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    import av
    cache_bytes = FRAME_COUNT * int(np.prod(RAW_IMAGE_SHAPE))
    memory = _memory_check(cache_bytes)
    images = np.empty((FRAME_COUNT, *RAW_IMAGE_SHAPE), dtype=np.uint8)
    with av.open(str(video), mode="r") as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
            raise ValueError("expected exactly one video stream and no audio")
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        if codec not in {"av1", "libdav1d", "libaom-av1"}:
            raise ValueError(f"pinned video is not decoded as AV1: {codec}")
        if Fraction(stream.average_rate) != Fraction(10, 1):
            raise ValueError("pinned video FPS must be 10")
        if (stream.codec_context.width, stream.codec_context.height) != (96, 96):
            raise ValueError("pinned video dimensions changed")
        if stream.frames not in (0, FRAME_COUNT):
            raise ValueError("container frame count disagrees with pinned dataset")
        count = 0
        for frame in container.decode(stream):
            images[count] = check_video_frame(frame, count, FRAME_COUNT)
            count += 1
        if count != FRAME_COUNT:
            raise ValueError(f"decoded frame count differs from {FRAME_COUNT}: {count}")
        stream_info = {"codec": codec, "fps": 10, "time_base": str(stream.time_base),
                       "container_frames": stream.frames, "frame_count": count}
    probes_verified = verify_image_probes(images, probes)
    digest = hashlib.sha256(memoryview(images).cast("B")).hexdigest()
    return images, {
        **stream_info, "layout": "uint8_NHWC_RGB", "shape": list(images.shape),
        "bytes": images.nbytes, "sha256": digest, "backend": "sequential_pyav_rgb24",
        "av_version": av.__version__, "all_frames_decoded": True, "all_frame_pts_exact": True,
        "six_saved_lerobot_probes": probes_verified, "memory_check": memory,
        "written_to_disk": False,
    }


class PushTTrainingData:
    """Validated arrays with explicit action chunks and metadata isolation.

    Direct construction is useful for synthetic tests; only load_training_data
    establishes the full pinned production binding. Input arrays are retained
    in RAM, with private read-only views, not copied into a second image cache.
    """

    def __init__(self, *, images: np.ndarray, states: np.ndarray, actions: np.ndarray,
                 episodes: np.ndarray, frames: np.ndarray, split: dict, stats: dict,
                 binding: dict | None = None, load_diagnostics: dict | None = None) -> None:
        if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != RAW_IMAGE_SHAPE or not images.flags.c_contiguous or len(images) == 0:
            raise ValueError("image cache must be nonempty contiguous uint8 NHWC96 RGB")
        for name, array in (("states", states), ("actions", actions)):
            if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != (len(images), 2) or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite float32 [N,2]")
        if np.any(actions < 0) or np.any(actions > 512):
            raise ValueError("native action targets exceed [0,512]")
        episodes = _integer_vector(episodes, "episodes")
        frames = _integer_vector(frames, "frames")
        if len(episodes) != len(images):
            raise ValueError("numeric and image record counts differ")
        self._ends = episode_end_indices(episodes, frames)
        validate_statistics(stats)
        self.train_indices, self.val_indices = split_indices(episodes, split)
        self._images = images.view()
        self._states, self._actions = states.view(), actions.view()
        self.episode_indices, self.frame_indices = episodes.view(), frames.view()
        for array in (self._images, self._states, self._actions, self.episode_indices,
                      self.frame_indices, self.train_indices, self.val_indices, self._ends):
            array.flags.writeable = False
        self._stats, self._split = copy.deepcopy(stats), copy.deepcopy(split)
        self._binding = copy.deepcopy(binding or {"schema": SCHEMA, "source_kind": "synthetic_constructor_no_production_gate"})
        self.load_diagnostics = copy.deepcopy(load_diagnostics or {})

    @property
    def stats(self) -> dict:
        return copy.deepcopy(self._stats)

    @property
    def split(self) -> dict:
        return copy.deepcopy(self._split)

    @property
    def binding(self) -> dict:
        return copy.deepcopy(self._binding)

    def __len__(self) -> int:
        return len(self._images)

    def batch(self, indices: Any, device: str = "cuda") -> dict[str, Any]:
        import torch
        indices = _integer_vector(indices, "batch indices")
        future, padding = chunk_indices(indices, self._ends)
        image = torch.from_numpy(self._images[indices]).permute(0, 3, 1, 2).contiguous().float().div_(255)
        return {
            "observation": {
                "observation.image": image.to(device),
                "observation.state": torch.from_numpy(self._states[indices]).to(device),
            },
            "action": torch.from_numpy(self._actions[future]).to(device),
            "action_is_pad": torch.from_numpy(padding).to(device),
            "indices": indices.tolist(),
        }


def load_training_data(root: Path = DEFAULT_ROOT, gate_dir: Path = DEFAULT_GATE_DIR,
                       cache_images: bool = True) -> PushTTrainingData:
    """Verify pinned bytes, preserve saved split/stats, then decode once; no network."""
    if cache_images is not True:
        raise ValueError("only the verified single-pass in-memory image-cache path is supported")
    root, gate_dir = Path(root), Path(gate_dir)
    started = time.monotonic()
    report, split, stats, source = verify_gate(gate_dir)
    source_files = verify_files(root, source)
    import pyarrow.parquet as pq
    info = json.loads((root / "meta/info.json").read_text())
    if info != source["metadata_expected"]:
        raise ValueError("source metadata differs from the pinned snapshot")
    table = pq.read_table(root / "data/chunk-000/file-000.parquet", columns=list(NUMERIC_COLUMNS))
    metadata = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet", columns=list(EPISODE_COLUMNS))
    states64, actions64, episodes = validate_records(table, metadata, info)
    for row in metadata.to_pylist():
        for name in ("chunk_index", "file_index"):
            if row[f"videos/observation.image/{name}"] != 0:
                raise ValueError("episode points outside the pinned single video")
        for endpoint, index_name in (("from_timestamp", "dataset_from_index"), ("to_timestamp", "dataset_to_index")):
            actual = float(row[f"videos/observation.image/{endpoint}"])
            if not np.isfinite(actual) or abs(actual - row[index_name] / 10) > 1e-3:
                raise ValueError("episode video offset disagrees with global frame order")
    expected_split = episode_split(list(range(206)))
    if any(split[key] != value for key, value in expected_split.items()):
        raise ValueError("saved salted episode split changed")
    recomputed, count = train_statistics(states64, actions64, episodes, split)
    if canonical_hash(recomputed) != report["stats_sha256"] or count != 23488:
        raise ValueError("actual train-only moments differ from saved normalization")
    images, cache = decode_image_cache(root / "videos/observation.image/chunk-000/file-000.mp4", report["decode_smoke"]["probes"])
    memory_check = cache.pop("memory_check")
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    binding = {
        "schema": SCHEMA, "source_kind": "pinned_public_push_t", "revision": REVISION,
        "source_root": str(root.resolve()), "gate_dir": str(gate_dir.resolve()),
        "source_files": source_files, "source_manifest_sha256": report["source_manifest_sha256"],
        "gate_file_sha256": GATE_PINS, "split_sha256": report["split_sha256"],
        "stats_sha256": report["stats_sha256"], "normalization_source": report["normalization_source"],
        "source_code_sha256": {**CODE_PINS, "tools/pusht_bc_act_training_data.py": file_hash(__file__)},
        "numeric_columns_read": list(NUMERIC_COLUMNS), "episode_columns_read": list(EPISODE_COLUMNS),
        "policy_observation_keys": ["observation.image", "observation.state"],
        "image_cache": cache, "chunk_size": CHUNK_SIZE, "chunk_padding": "repeat_last_action_with_true_padding_mask",
        "train_frames": 23488, "val_frames": 2162, "record_count": FRAME_COUNT,
        "normalization_recomputed_from_train_matches_saved": True,
        "global_meta_stats_used": False, "training_started": False, "optimizer_steps": 0,
    }
    data = PushTTrainingData(images=images, states=states64.astype(np.float32), actions=actions64.astype(np.float32),
                            episodes=episodes, frames=frames, split=split, stats=stats, binding=binding,
                            load_diagnostics={"memory_check": memory_check})
    if len(data.train_indices) != 23488 or len(data.val_indices) != 2162:
        raise ValueError("final train/val record coverage mismatch")
    if verify_files(root, source) != source_files:
        raise ValueError("source changed during cache construction")
    verify_gate(gate_dir)
    data.load_diagnostics["load_seconds"] = time.monotonic() - started
    return data
