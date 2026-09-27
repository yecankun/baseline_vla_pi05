from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from probe_pi05_lerobot_adapter import import_lerobot_dataset
from train_pi05_lerobot_adapter import split_by_episode


def as_numpy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def image_chw(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError(f"expected three-dimensional image tensor, got {tuple(tensor.shape)}")
    if tensor.shape[0] not in {1, 3, 4} and tensor.shape[-1] in {1, 3, 4}:
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    if tensor.max().item() > 1.5:
        tensor = tensor / 255.0
    return tensor.clamp(0.0, 1.0).contiguous()


def image_signature(tensor: torch.Tensor) -> np.ndarray:
    pooled = functional.adaptive_avg_pool2d(tensor.unsqueeze(0), (8, 8)).squeeze(0)
    channel_mean = tensor.mean(dim=(1, 2))
    channel_std = tensor.std(dim=(1, 2), unbiased=False)
    grayscale = pooled.mean(dim=0).flatten()
    return torch.cat([grayscale, channel_mean, channel_std]).numpy().astype(np.float32)


def image_hash(tensor: torch.Tensor) -> str:
    encoded = (tensor * 255.0).round().to(torch.uint8).numpy().tobytes()
    return hashlib.sha1(encoded).hexdigest()


def scalar_percentiles(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "p01": np.quantile(values, 0.01, axis=0).tolist(),
        "p50": np.quantile(values, 0.50, axis=0).tolist(),
        "p99": np.quantile(values, 0.99, axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "zero_fraction": np.mean(np.isclose(values, 0.0, atol=1e-8), axis=0).tolist(),
        "positive_fraction": np.mean(values > 1e-8, axis=0).tolist(),
        "negative_fraction": np.mean(values < -1e-8, axis=0).tolist(),
        "unique_rows_exact": int(np.unique(values, axis=0).shape[0]),
        "unique_rows_rounded_3dp": int(np.unique(np.round(values, 3), axis=0).shape[0]),
    }


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.abs(np.asarray(prediction) - np.asarray(target))
    squared = np.square(np.asarray(prediction) - np.asarray(target))
    target_variance = np.var(target, axis=0)
    r2 = 1.0 - np.mean(squared, axis=0) / np.maximum(target_variance, 1e-12)
    return {
        "records": int(target.shape[0]),
        "mae_mean": float(error.mean()),
        "mae_by_dim": error.mean(axis=0).tolist(),
        "median_abs_error": float(np.median(error)),
        "p95_abs_error": float(np.quantile(error, 0.95)),
        "r2_by_dim": r2.tolist(),
    }


def ridge_predict(
    train_features: np.ndarray,
    train_target: np.ndarray,
    val_features: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    active = std > 1e-8
    train = (train_features[:, active] - mean[active]) / std[active]
    val = (val_features[:, active] - mean[active]) / std[active]
    train = np.concatenate([train, np.ones((train.shape[0], 1))], axis=1)
    val = np.concatenate([val, np.ones((val.shape[0], 1))], axis=1)
    gram = train.T @ train
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, train.T @ train_target)
    return val @ weights, {
        "input_dims": int(train_features.shape[1]),
        "active_dims": int(active.sum()),
        "alpha": float(alpha),
    }


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(int(value) for value in target.tolist()) | set(int(value) for value in prediction.tolist()))
    confusion = [[int(np.sum((target == row) & (prediction == col))) for col in labels] for row in labels]
    recalls = []
    for label in labels:
        support = int(np.sum(target == label))
        if support:
            recalls.append(float(np.mean(prediction[target == label] == label)))
    return {
        "labels": labels,
        "accuracy": float(np.mean(target == prediction)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "support": {str(label): int(np.sum(target == label)) for label in labels},
        "predicted_support": {str(label): int(np.sum(prediction == label)) for label in labels},
        "confusion_target_rows_pred_cols": confusion,
    }


def lag_one_correlation(values: np.ndarray, episodes: np.ndarray) -> list[float | None]:
    result: list[float | None] = []
    same_episode = episodes[1:] == episodes[:-1]
    for dim in range(values.shape[1]):
        left = values[:-1, dim][same_episode]
        right = values[1:, dim][same_episode]
        if left.size < 2 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
            result.append(None)
        else:
            result.append(float(np.corrcoef(left, right)[0, 1]))
    return result


def run_lengths(values: np.ndarray, episodes: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    previous_value: int | None = None
    previous_episode: int | None = None
    for value, episode in zip(values.tolist(), episodes.tolist()):
        value = int(value)
        episode = int(episode)
        if previous_value is None or value != previous_value or episode != previous_episode:
            if current:
                lengths.append(current)
            current = 1
        else:
            current += 1
        previous_value = value
        previous_episode = episode
    if current:
        lengths.append(current)
    return lengths


def nearest_train_distance(train: np.ndarray, val: np.ndarray, *, chunk_size: int = 64) -> np.ndarray:
    train = np.asarray(train, dtype=np.float32)
    val = np.asarray(val, dtype=np.float32)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    active = scale > 1e-8
    train = (train[:, active] - mean[active]) / scale[active]
    val = (val[:, active] - mean[active]) / scale[active]
    distances: list[np.ndarray] = []
    for start in range(0, val.shape[0], chunk_size):
        block = val[start : start + chunk_size]
        squared = np.square(block[:, None, :] - train[None, :, :]).mean(axis=2)
        distances.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(distances)


def make_contact_sheet(
    dataset: Any,
    indices: list[int],
    out_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    rows: list[Image.Image] = []
    for idx in indices:
        item = dataset[idx]
        side = image_chw(item["observation.images.side"])
        top = image_chw(item["observation.images.top"])
        tiles = []
        for tensor in (side, top):
            array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            tiles.append(Image.fromarray(array).resize((224, 224)))
        row = Image.new("RGB", (448, 250), "white")
        row.paste(tiles[0], (0, 26))
        row.paste(tiles[1], (224, 26))
        elite = as_numpy(item["elite_tcp_delta_6d"], dtype=np.float64).reshape(-1)
        piper = int(as_numpy(item["piper_intent_id"]).reshape(-1)[0])
        task = str(item.get("task", "missing"))
        route = "left" if "left branch" in task else "right" if "right branch" in task else "unknown"
        label = (
            f"idx={idx} ep={int(item['episode_index'])} frame={int(item['frame_index'])} "
            f"route={route} piper={piper} dxyz={np.round(elite[:3], 2).tolist()}"
        )
        ImageDraw.Draw(row).text((4, 6), label, fill=(0, 0, 0))
        rows.append(row)
    sheet = Image.new("RGB", (448, 250 * len(rows)), "white")
    for row_idx, row in enumerate(rows):
        sheet.paste(row, (0, 250 * row_idx))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PI05 LeRobot training data quality and learnability.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--split-manifest", type=Path, help="Use the authoritative LeRobot export episode split.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--image-audit", action="store_true")
    args = parser.parse_args()
    if args.max_records is None and args.split_manifest is None:
        args.max_records = 4096

    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
    train_indices, val_indices, split_mode = split_by_episode(
        dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
        max_records=args.max_records,
        split_manifest=args.split_manifest,
    )
    indices = sorted(train_indices + val_indices)
    position_by_index = {idx: position for position, idx in enumerate(indices)}

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    elite_actions: list[np.ndarray] = []
    piper_ids: list[int] = []
    episodes: list[int] = []
    frames: list[int] = []
    tasks: list[str] = []
    image_signatures: list[np.ndarray] = []
    combined_image_hashes: list[str] = []
    consecutive_image_mae: list[float] = []
    previous_images: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    for count, idx in enumerate(indices, start=1):
        item = dataset[idx]
        state = as_numpy(item["observation.state"], dtype=np.float64).reshape(-1)
        action = as_numpy(item["action"], dtype=np.float64).reshape(-1)
        elite = as_numpy(item["elite_tcp_delta_6d"], dtype=np.float64).reshape(-1)
        piper = int(as_numpy(item["piper_intent_id"]).reshape(-1)[0])
        episode = int(item["episode_index"])
        frame = int(item["frame_index"])
        task = str(item.get("task", "missing"))
        states.append(state)
        actions.append(action)
        elite_actions.append(elite)
        piper_ids.append(piper)
        episodes.append(episode)
        frames.append(frame)
        tasks.append(task)
        if args.image_audit:
            side = image_chw(item["observation.images.side"])
            top = image_chw(item["observation.images.top"])
            image_signatures.append(np.concatenate([image_signature(side), image_signature(top)]))
            combined_image_hashes.append(f"{image_hash(side)}:{image_hash(top)}")
            previous = previous_images.get(episode)
            if previous is not None and frame > 0:
                consecutive_image_mae.append(
                    float(((side - previous[0]).abs().mean() + (top - previous[1]).abs().mean()).item() / 2.0)
                )
            previous_images[episode] = (side, top)
        if count % 500 == 0:
            print(f"loaded_records={count}", flush=True)

    state_values = np.stack(states)
    action_values = np.stack(actions)
    elite_values = np.stack(elite_actions)
    piper_values = np.asarray(piper_ids, dtype=np.int64)
    episode_values = np.asarray(episodes, dtype=np.int64)
    frame_values = np.asarray(frames, dtype=np.int64)
    task_values = np.asarray(tasks)
    train_positions = np.asarray([position_by_index[idx] for idx in train_indices], dtype=np.int64)
    val_positions = np.asarray([position_by_index[idx] for idx in val_indices], dtype=np.int64)

    schema_checks = {
        "finite_state": bool(np.isfinite(state_values).all()),
        "finite_action": bool(np.isfinite(action_values).all()),
        "explicit_elite_matches_action_0_6": bool(np.array_equal(elite_values, action_values[:, :6])),
        "piper_one_hot_matches_explicit": bool(
            np.array_equal(np.argmax(action_values[:, 6:9], axis=1), piper_values)
        ),
        "padded_action_dims_9_32_zero": bool(np.allclose(action_values[:, 9:], 0.0)),
    }

    episode_tasks: dict[int, Counter[str]] = defaultdict(Counter)
    episode_lengths: Counter[int] = Counter()
    for episode, task in zip(episode_values.tolist(), task_values.tolist()):
        episode_lengths[int(episode)] += 1
        episode_tasks[int(episode)][str(task)] += 1

    train_elite = elite_values[train_positions, :3]
    val_elite = elite_values[val_positions, :3]
    zero_prediction = np.zeros_like(val_elite)
    mean_prediction = np.repeat(train_elite.mean(axis=0, keepdims=True), len(val_positions), axis=0)
    task_prediction = np.empty_like(val_elite)
    for task in np.unique(task_values[val_positions]):
        train_task = train_elite[task_values[train_positions] == task]
        task_prediction[task_values[val_positions] == task] = train_task.mean(axis=0)

    previous_positions: list[int] = []
    current_positions: list[int] = []
    for position in val_positions.tolist():
        if position > 0 and episode_values[position - 1] == episode_values[position] and frame_values[position - 1] + 1 == frame_values[position]:
            previous_positions.append(position - 1)
            current_positions.append(position)
    previous_positions_array = np.asarray(previous_positions, dtype=np.int64)
    current_positions_array = np.asarray(current_positions, dtype=np.int64)

    state_prediction, state_ridge = ridge_predict(
        state_values[train_positions],
        train_elite,
        state_values[val_positions],
        alpha=args.ridge_alpha,
    )
    state_class_scores, state_class_ridge = ridge_predict(
        state_values[train_positions],
        np.eye(3, dtype=np.float64)[piper_values[train_positions]],
        state_values[val_positions],
        alpha=args.ridge_alpha,
    )

    piper_majority = Counter(piper_values[train_positions].tolist()).most_common(1)[0][0]
    piper_baselines = {
        "majority": classification_metrics(
            piper_values[val_positions],
            np.full(len(val_positions), piper_majority, dtype=np.int64),
        ),
        "state_ridge": {
            **classification_metrics(piper_values[val_positions], state_class_scores.argmax(axis=1)),
            "ridge": state_class_ridge,
        },
        "previous_label": classification_metrics(
            piper_values[current_positions_array],
            piper_values[previous_positions_array],
        ),
    }

    image_report: dict[str, Any] = {"enabled": bool(args.image_audit)}
    if args.image_audit:
        image_values = np.stack(image_signatures).astype(np.float64)
        image_prediction, image_ridge = ridge_predict(
            image_values[train_positions],
            train_elite,
            image_values[val_positions],
            alpha=args.ridge_alpha,
        )
        image_class_scores, image_class_ridge = ridge_predict(
            image_values[train_positions],
            np.eye(3, dtype=np.float64)[piper_values[train_positions]],
            image_values[val_positions],
            alpha=args.ridge_alpha,
        )
        train_hashes = set(combined_image_hashes[position] for position in train_positions.tolist())
        val_hashes = [combined_image_hashes[position] for position in val_positions.tolist()]
        image_report.update(
            {
                "signature_dims": int(image_values.shape[1]),
                "exact_duplicate_rows": int(len(image_values) - len(set(combined_image_hashes))),
                "val_exact_hash_match_in_train": int(sum(value in train_hashes for value in val_hashes)),
                "consecutive_pixel_mae": scalar_percentiles(np.asarray(consecutive_image_mae)),
                "nearest_train_signature_distance": scalar_percentiles(
                    nearest_train_distance(image_values[train_positions], image_values[val_positions])
                ),
                "elite_translation_image_ridge": {
                    **regression_metrics(val_elite, image_prediction),
                    "ridge": image_ridge,
                },
                "piper_image_ridge": {
                    **classification_metrics(piper_values[val_positions], image_class_scores.argmax(axis=1)),
                    "ridge": image_class_ridge,
                },
            }
        )

    report = {
        "scope": "read-only PI05 training-data audit; experimental simulation data, not real-system validation",
        "root": str(args.root),
        "repo_id": args.repo_id,
        "records": len(indices),
        "dataset_records": len(dataset),
        "split": {
            "mode": split_mode,
            "manifest": str(args.split_manifest) if args.split_manifest is not None else None,
            "seed": args.seed,
            "train_records": len(train_positions),
            "val_records": len(val_positions),
            "train_episodes": sorted(set(int(value) for value in episode_values[train_positions].tolist())),
            "val_episodes": sorted(set(int(value) for value in episode_values[val_positions].tolist())),
            "train_tasks": dict(Counter(task_values[train_positions].tolist())),
            "val_tasks": dict(Counter(task_values[val_positions].tolist())),
        },
        "schema_checks": schema_checks,
        "episodes": {
            "count": len(episode_lengths),
            "length": scalar_percentiles(np.asarray(list(episode_lengths.values()))),
            "task_count_per_episode": {str(key): dict(value) for key, value in sorted(episode_tasks.items())},
        },
        "state": {
            "shape": list(state_values.shape),
            "constant_dims": np.flatnonzero(state_values.std(axis=0) < 1e-8).tolist(),
            "varying_dims": np.flatnonzero(state_values.std(axis=0) >= 1e-8).tolist(),
            "stats": vector_stats(state_values),
        },
        "elite_action": {
            "shape": list(elite_values.shape),
            "stats": vector_stats(elite_values),
            "lag_one_correlation": lag_one_correlation(elite_values, episode_values),
            "consecutive_exact_match_fraction": float(
                np.mean(
                    np.all(np.isclose(elite_values[1:], elite_values[:-1], atol=1e-8), axis=1)[
                        episode_values[1:] == episode_values[:-1]
                    ]
                )
            ),
        },
        "piper": {
            "counts": {str(key): int(value) for key, value in sorted(Counter(piper_values.tolist()).items())},
            "run_length": scalar_percentiles(np.asarray(run_lengths(piper_values, episode_values))),
            "transition_fraction": float(
                np.mean((piper_values[1:] != piper_values[:-1])[episode_values[1:] == episode_values[:-1]])
            ),
            "baselines": piper_baselines,
        },
        "elite_translation_baselines": {
            "zero": regression_metrics(val_elite, zero_prediction),
            "train_mean": regression_metrics(val_elite, mean_prediction),
            "task_mean": regression_metrics(val_elite, task_prediction),
            "previous_action": regression_metrics(
                elite_values[current_positions_array, :3],
                elite_values[previous_positions_array, :3],
            ),
            "state_ridge": {
                **regression_metrics(val_elite, state_prediction),
                "ridge": state_ridge,
            },
        },
        "image_audit": image_report,
        "known_interface_limits": [
            "LeRobot PI05 0.4.4 diffusion currently ignores observation.state in the project baseline",
            "state_32 contains only Elite pose, Piper state, three tactile values and validity, and task one-hot",
            "the source expert uses estimated-tip and registered-route geometry not exported as explicit state_32 fields",
            "current dataset contains simulation episodes only and does not establish real-system validity",
        ],
    }

    if args.contact_sheet:
        sheet_indices: list[int] = []
        for subset in (train_indices, val_indices):
            if subset:
                positions = np.linspace(0, len(subset) - 1, num=min(6, len(subset)), dtype=int)
                sheet_indices.extend(subset[int(position)] for position in positions)
        make_contact_sheet(dataset, sheet_indices, args.contact_sheet)
        report["contact_sheet"] = str(args.contact_sheet)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": report["records"],
        "split": report["split"],
        "schema_checks": report["schema_checks"],
        "state_constant_dims": report["state"]["constant_dims"],
        "elite_action": report["elite_action"],
        "piper": report["piper"],
        "elite_translation_baselines": report["elite_translation_baselines"],
        "image_audit": report["image_audit"],
        "out": str(args.out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
