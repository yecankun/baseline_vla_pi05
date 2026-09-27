"""Prepare/audit recorded reward labels without training or environment calls.

Reads the pinned numeric Parquet and existing RGB-object feature cache only.
Does not decode video, load weights, change source labels or use inspected
candidate outcomes. An existing output directory is never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

import pusht_object_dynamics as base
import pusht_recorded_reward_targets as target


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    return {"count": len(v), "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "std": float(v.std()),
            "zero": int((v == 0).sum()), "unique": int(len(np.unique(v)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=base.CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=target.TARGET_ROOT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse overwrite: {args.output}")
    start = time.monotonic()
    data = base.ObjectData(args.cache_root, device="cpu")
    if data.manifest["source_revision"] != target.REVISION:
        raise ValueError("only the already approved pinned public dataset is supported")
    source = Path(data.manifest["source_root"]) / "data/chunk-000/file-000.parquet"
    columns = ("index", "episode_index", "frame_index", "observation.state", "action",
               "next.reward", "next.done", "next.success")
    table = pq.read_table(source, columns=list(columns))
    if any(table[k].null_count for k in columns):
        raise ValueError("missing source fields; no fill or label synthesis")
    arrays = {k: np.asarray(table[k].to_pylist()) for k in columns}
    n = len(data.episodes)
    for name, expected in (("index", np.arange(n)), ("episode_index", data.episodes),
                           ("frame_index", data.frames), ("observation.state", data.states.numpy()),
                           ("action", data.actions.numpy())):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"pinned source/cache mismatch: {name}")
    reward = arrays["next.reward"].astype(np.float32)
    done, success = arrays["next.done"], arrays["next.success"]
    if not np.isfinite(reward).all() or ((reward < 0) | (reward > 1)).any():
        raise ValueError("recorded reward must be finite and bounded; never repair labels")
    starts = np.r_[0, np.flatnonzero(np.diff(data.episodes)) + 1]
    ends = np.r_[starts[1:] - 1, n - 1]
    expected_done = np.zeros(n, dtype=bool)
    expected_done[ends] = expected_done[ends - 1] = True
    tail_checks = {
        "episodes": len(starts),
        "done_exactly_last_two_rows_in_every_episode": bool(np.array_equal(done, expected_done)),
        "reward_last_two_rows_equal_every_episode": bool(np.array_equal(reward[ends], reward[ends - 1])),
        "success_last_two_rows_equal_every_episode": bool(np.array_equal(success[ends], success[ends - 1])),
        "done_true_count": int(done.sum()), "success_true_count": int(success.sum()),
    }
    if not all(tail_checks[k] for k in (
            "done_exactly_last_two_rows_in_every_episode", "reward_last_two_rows_equal_every_episode",
            "success_last_two_rows_equal_every_episode")):
        raise ValueError("snapshot no longer agrees with documented historical tail convention")

    # Do not cross an episode to invent reward at its first observation.
    current = np.full(n, np.nan, dtype=np.float32)
    available = data.frames > 0
    current[available] = reward[np.flatnonzero(available) - 1]
    splits = {}
    for name, anchors in (("train", data.train), ("val", data.val)):
        endpoint, label_row = anchors + base.HORIZON, anchors + base.HORIZON - 1
        if not (np.array_equal(data.episodes[endpoint], data.episodes[anchors])
                and np.array_equal(data.frames[endpoint], data.frames[anchors] + base.HORIZON)):
            raise ValueError("target endpoint crosses an episode")
        valid_delta = available[anchors]
        terminal = reward[label_row]
        delta = terminal[valid_delta] - current[anchors[valid_delta]]
        splits[name] = {
            "original_windows": len(anchors), "terminal_targets": len(terminal),
            "terminal_reward": summary(terminal), "delta_available": int(valid_delta.sum()),
            "missing_initial_reward": int((~valid_delta).sum()), "delta_reward": summary(delta),
            "delta_strata": {"negative": int((delta < -1e-7).sum()),
                             "near_zero": int((np.abs(delta) <= 1e-7).sum()),
                             "positive": int((delta > 1e-7).sum())},
            "uses_repeat_padded_last_row": bool(np.isin(label_row, ends).any()),
            "windows_ending_at_real_last_frame": int(np.isin(endpoint, ends).sum()),
        }
    stats = splits["train"]["terminal_reward"]
    if stats["std"] <= 1e-8:
        raise ValueError("constant target cannot provide scalar scoring supervision")
    target_stats = {"mean": stats["mean"], "std": stats["std"], "samples": stats["count"],
                    "population_std": True, "source": "original_training_windows_only"}
    manifest = {
        "schema": target.SCHEMA, "status": "prepared_recorded_reward_only",
        "contract": target.CONTRACT, "provenance": target.PROVENANCE,
        "source_parquet": str(source), "columns_read": list(columns),
        "cache_identity": data.identity, "target_stats": target_stats,
        "readiness": {"recorded_reward_interface": True, "exact_current_coverage_labels": False,
                      "controlled_target_only_comparison_ready": False, "training_started": False},
    }
    args.output.mkdir(parents=True)
    write(args.output / "manifest.json", manifest)
    np.savez_compressed(args.output / "targets.npz", next_reward=reward, next_done=done,
                        next_success=success, current_reward=current, current_reward_valid=available,
                        episodes=data.episodes, frames=data.frames, train=data.train, val=data.val)

    # One real batch through the new adapter, no model construction or forward.
    wrapped = target.RecordedRewardData(data, args.output)
    chosen = np.unique(np.concatenate((data.train[:4], data.train[-4:], data.val[:4], data.val[-4:])))
    inputs, targets = wrapped.batch(chosen)
    old_current, old_state, old_actions, _ = data.batch(chosen)
    inputs_equal = all(torch.equal(a, b) for a, b in zip(inputs, (old_current, old_state, old_actions)))
    targets_equal = np.array_equal(targets.numpy()[:, 0], reward[chosen + base.HORIZON - 1])
    standardized = (targets - target_stats["mean"]) / target_stats["std"]
    if not inputs_equal or not targets_equal or not bool(torch.isfinite(standardized).all()):
        raise ValueError("real batch must preserve inputs and return separately aligned finite targets")
    rows = [{"index": int(i), "episode": int(data.episodes[i]), "frame": int(data.frames[i]),
             "last_action_row": int(i + base.HORIZON - 1), "outcome_frame_row": int(i + base.HORIZON),
             "target": float(reward[i + base.HORIZON - 1]),
             "current_reward": float(current[i]) if available[i] else None,
             "current_reward_available": bool(available[i])} for i in chosen]
    report = {
        "schema": target.SCHEMA, "status": "adapter_verified_not_training",
        "rows": n, "episodes": len(starts), "published_reward": summary(reward),
        "source_has_exact_coverage_or_block_pose": False,
        "historical_next_frame_mapping_consistent": tail_checks, "splits": splits,
        "target_stats": target_stats,
        "batch_check": {"samples": len(chosen), "input_shapes": [list(x.shape) for x in inputs],
                        "target_shape": list(targets.shape), "old_inputs_exactly_equal": inputs_equal,
                        "target_rows_exactly_equal": targets_equal,
                        "standardized_target_finite": True,
                        "target_not_in_input_tuple": len(inputs) == 3},
        "batch_examples": rows,
        "training_blockers": [
            "Recorded reward is not yet verified against native coverage from the current environment; "
            "the snapshot lacks raw block pose and exact conversion runtime provenance.",
            "Terminal reward preserves all windows, but DS0 learned visual improvement. A clean "
            "target-only comparison must match terminal-vs-delta parametrization as well as budget. "
            "Delta reward lacks the first-frame baseline; do not fill zero or silently change cohorts.",
        ],
        "decision": "interface_ready_recorded_reward_only_no_training_entrypoint_or_run",
        "execution": {"optimizer_steps": 0, "model_forward_calls": 0, "checkpoint_loads": 0,
                      "environment_steps": 0, "hardware_actions": 0, "video_frames_decoded": 0},
        "elapsed_seconds": time.monotonic() - start,
    }
    write(args.output / "report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "published_reward", "splits", "batch_check",
                                           "decision", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
