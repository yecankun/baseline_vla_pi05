"""Check published Push-T rewards against original recorded numerical poses.

Downloads only small numerical Zarr members from the pinned official raw repo.
Static geometry only: no environment construction/reset/step, model or training.
Original data, reward sidecar and feature cache are never changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numcodecs
import numpy as np
import pyarrow.parquet as pq
import requests

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from gym_pusht.envs.pusht import PushTEnv
import pymunk

ROOT = Path(__file__).resolve().parents[1]
RAW_REPO = "lerobot-raw/pusht_raw"
RAW_REVISION = "f93fc57921866e96d8ae00efbe5bb146a04b088f"
DATA_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
ZARR = "pusht_cchi_v7_replay.zarr"
ARRAYS = ("data/state", "data/action", "meta/episode_ends")
RAW_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_raw_numeric_f93fc579")
DATA_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/datasets/pusht_7628202a")
OUT = ROOT / "simulation_output/pusht_reward_geometry_audit_v1"
TOLERANCE = 5e-7  # comparison to stored float32, not a fitted threshold
SOURCES = {
    "raw": f"https://huggingface.co/datasets/{RAW_REPO}/tree/{RAW_REVISION}",
    "historical_converter": "https://github.com/huggingface/lerobot/blob/8e7d697/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py",
    "historical_add_tee": "https://github.com/huggingface/gym-pusht/blob/e0684ff988d223808c0a9dcfaba9dc4991791370/gym_pusht/envs/pusht.py#L417",
    "raw_collection": "https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/demo_pusht.py",
    "raw_observation": "https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/env/pusht/pusht_env.py#L140",
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_numeric(root):
    """Resume completed tiny source members by pinned path/expected size."""
    api = f"https://huggingface.co/api/datasets/{RAW_REPO}/tree/{RAW_REVISION}"
    members = []
    for key in ARRAYS:
        response = requests.get(f"{api}/{ZARR}/{key}", params={"limit": 1000}, timeout=30)
        response.raise_for_status()
        members.extend(x for x in response.json() if x["type"] == "file")
    if any(x["size"] > 100_000_000 for x in members):
        raise ValueError("large-file transfer is user-owned")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def fetch(item):
        path = root / item["path"]
        if path.exists() and path.stat().st_size == item["size"]:
            return
        response = requests.get(
            f"https://huggingface.co/datasets/{RAW_REPO}/resolve/{RAW_REVISION}/{item['path']}",
            timeout=30,
        )
        response.raise_for_status()
        if len(response.content) != item["size"]:
            raise ValueError(f"incomplete numeric member: {item['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)

    print(f"numeric fetch: {len(members)} files, {sum(x['size'] for x in members)} bytes; no images", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, _ in enumerate(pool.map(fetch, members), 1):
            if count % 80 == 0:
                print(f"numeric members ready: {count}/{len(members)}", flush=True)
    provenance = {"repo": RAW_REPO, "revision": RAW_REVISION,
                  "files": [{"path": x["path"], "size": x["size"]} for x in members],
                  "total_bytes": sum(x["size"] for x in members),
                  "images_downloaded": 0, "elapsed_seconds": time.monotonic() - started}
    write(root / "numeric_source.json", provenance)
    return provenance


def read_zarr_array(root, key):
    """Decode the fixed Blosc/C-order arrays without installing Zarr."""
    folder = root / ZARR / key
    meta = json.loads((folder / ".zarray").read_text())
    if meta["zarr_format"] != 2 or meta["filters"] or meta["order"] != "C":
        raise ValueError("unexpected numerical source encoding")
    shape, chunks = tuple(meta["shape"]), tuple(meta["chunks"])
    if shape[1:] != chunks[1:]:
        raise ValueError("expected source chunking only on first dimension")
    out = np.empty(shape, dtype=meta["dtype"])
    codec = numcodecs.get_codec(meta["compressor"])
    for i in range(math.ceil(shape[0] / chunks[0])):
        name = ".".join([str(i)] + ["0"] * (len(shape) - 1))
        decoded = np.frombuffer(codec.decode((folder / name).read_bytes()), dtype=out.dtype).reshape(chunks)
        lo, hi = i * chunks[0], min((i + 1) * chunks[0], shape[0])
        out[lo:hi] = decoded[:hi - lo]
    return out


def compare(predicted, recorded):
    difference = np.asarray(predicted, dtype=np.float64) - np.asarray(recorded, dtype=np.float64)
    return {"mae": float(np.abs(difference).mean()), "max_abs": float(np.abs(difference).max()),
            "within_5e_7": int((np.abs(difference) <= TOLERANCE).sum()),
            "float32_exact_equal": int((np.asarray(predicted, dtype=np.float32) == recorded).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--local-only", action="store_true", help="reuse completed pinned numerical download")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse overwrite: {args.output}")
    started = time.monotonic()
    source = (json.loads((args.raw_root / "numeric_source.json").read_text())
              if args.local_only else fetch_numeric(args.raw_root))
    if source["repo"] != RAW_REPO or source["revision"] != RAW_REVISION:
        raise ValueError("wrong original numerical source")
    states, actions, ends = (read_zarr_array(args.raw_root, key) for key in ARRAYS)
    table = pq.read_table(DATA_ROOT / "data/chunk-000/file-000.parquet", columns=[
        "observation.state", "action", "episode_index", "frame_index", "next.reward", "next.success", "next.done"])
    recorded = {k: np.asarray(table[k].to_pylist()) for k in table.column_names}
    n = len(states)
    episode = np.repeat(np.arange(len(ends)), np.diff(np.r_[0, ends]))
    frames = np.concatenate([np.arange(length) for length in np.diff(np.r_[0, ends])])
    alignment = {"rows": n, "episodes": len(ends),
                 "agent_state_exact": np.array_equal(states[:, :2], recorded["observation.state"]),
                 "actions_exact": np.array_equal(actions, recorded["action"]),
                 "episode_boundaries_exact": np.array_equal(episode, recorded["episode_index"]),
                 "frame_indices_exact": np.array_equal(frames, recorded["frame_index"])}
    if states.shape != (25650, 5) or not all(alignment[k] for k in (
            "agent_state_exact", "actions_exact", "episode_boundaries_exact", "frame_indices_exact")):
        raise ValueError(f"original numerical source does not align: {alignment}")

    # Static bodies only. Do not use reset/_set_state: they shift legacy poses
    # and run physics. State already contains the recorded body position.
    space = pymunk.Space()
    body, _ = PushTEnv.add_tee(space, [0, 0], 0)
    holder = SimpleNamespace(block=body, goal_pose=np.array([256, 256, np.pi / 4]),
                             get_goal_pose_body=PushTEnv.get_goal_pose_body)
    native, legacy = np.empty(n), np.empty(n)
    displacement = np.empty(n)
    geometry_started = time.monotonic()
    for i, state in enumerate(states):
        position, angle = state[2:4].tolist(), float(state[4])
        body.angle = angle
        body.position = position
        native[i] = PushTEnv._get_coverage(holder)
        # Historical add_tee set position before angle with nonzero CoG.
        body.angle = 0
        body.position = position
        body.angle = angle
        legacy[i] = PushTEnv._get_coverage(holder)
        displacement[i] = np.linalg.norm(np.asarray(body.position) - state[2:4])
    geometry_seconds = time.monotonic() - geometry_started
    next_rows = np.minimum(np.arange(n) + 1, np.repeat(ends - 1, np.diff(np.r_[0, ends])))
    native_reward, legacy_reward = np.clip(native / .95, 0, 1), np.clip(legacy / .95, 0, 1)
    variants = {"native_pose_shift1": compare(native_reward[next_rows], recorded["next.reward"]),
                "legacy_pose_shift1": compare(legacy_reward[next_rows], recorded["next.reward"]),
                "native_pose_no_shift": compare(native_reward, recorded["next.reward"]),
                "legacy_pose_no_shift": compare(legacy_reward, recorded["next.reward"])}
    legacy_match = variants["legacy_pose_shift1"]["within_5e_7"] == n
    native_match = variants["native_pose_shift1"]["within_5e_7"] == n
    ids = np.argsort(np.abs(native_reward[next_rows] - recorded["next.reward"]))[-5:][::-1]
    examples = [{"index": int(i), "episode": int(episode[i]), "frame": int(frames[i]),
                 "outcome_row": int(next_rows[i]), "recorded_reward": float(recorded["next.reward"][i]),
                 "native_pose_reward": float(native_reward[next_rows[i]]),
                 "legacy_pose_reward": float(legacy_reward[next_rows[i]]),
                 "legacy_pose_offset_pixels": float(displacement[next_rows[i]])} for i in ids]
    report = {
        "schema": "pusht_reward_geometry_audit_v1", "sources": SOURCES,
        "data_revision": DATA_REVISION, "raw_revision": RAW_REVISION,
        "numerical_alignment": {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in alignment.items()},
        "versions": {k: importlib.metadata.version(k) for k in ("gym-pusht", "pymunk", "shapely", "numcodecs")},
        "static_geometry": {"native": "angle_then_position_preserves_recorded_pose",
                            "historical": "position_then_angle_from_angle0_rotates_about_nonzero_CoG",
                            "recorded_pose_origin": "body.position_and_body.angle_observed_before_action",
                            "goal_pose": holder.goal_pose.tolist(), "cog": list(body.center_of_gravity),
                            "legacy_position_offset_mean_pixels": float(displacement.mean()),
                            "legacy_position_offset_max_pixels": float(displacement.max()),
                            "geometry_seconds": geometry_seconds},
        "reward_comparisons": variants,
        "legacy_shift1_reproduces_all_recorded_rewards": legacy_match,
        "native_shift1_reproduces_all_recorded_rewards": native_match,
        "success": {"recorded": int(recorded["next.success"].sum()),
                    "legacy_shift1": int((legacy[next_rows] > .95).sum()),
                    "native_shift1": int((native[next_rows] > .95).sum()),
                    "native_terminal_episodes": int((native[ends - 1] > .95).sum())},
        "raw_initial_outcomes_recoverable": True,
        "native_coverage": {"min": float(native.min()), "max": float(native.max())},
        "largest_reward_discrepancies": examples,
        "decision": ("legacy_pose_conversion_reproduced_do_not_train_native_task_score_on_recorded_reward"
                     if legacy_match and not native_match else
                     "native_reward_equivalence_verified" if native_match else "provenance_unresolved"),
        "target_use": "isolated_offline_geometry_only_never_policy_input_not_real_system_truth",
        "execution": {"model_forward_calls": 0, "optimizer_steps": 0, "environment_steps": 0,
                      "environment_resets": 0, "environment_instances": 0, "hardware_actions": 0,
                      "video_frames_decoded": 0, "checkpoint_loads": 0},
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.mkdir(parents=True)
    write(args.output / "report.json", report)
    np.savez_compressed(args.output / "geometry_diagnostic_targets.npz",
                        native_coverage=native, native_reward=native_reward,
                        legacy_coverage=legacy, legacy_reward=legacy_reward,
                        next_rows=next_rows, episodes=episode, frames=frames)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
