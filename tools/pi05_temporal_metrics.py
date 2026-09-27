from __future__ import annotations

from typing import Any

import torch


PIPER_CLASSES = 3


def tensor_error_stats(error: torch.Tensor) -> dict[str, Any]:
    values = error.detach().float().cpu()
    if values.numel() == 0:
        return {"records": 0, "mean": None, "median": None, "p95": None, "max": None, "by_dim": []}
    matrix = values.reshape(values.shape[0], -1)
    flat = matrix.flatten()
    return {
        "records": int(matrix.shape[0]),
        "mean": float(flat.mean().item()),
        "median": float(flat.median().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
        "by_dim": [
            {
                "mean": float(matrix[:, dim].mean().item()),
                "median": float(matrix[:, dim].median().item()),
                "p95": float(torch.quantile(matrix[:, dim], 0.95).item()),
                "max": float(matrix[:, dim].max().item()),
            }
            for dim in range(matrix.shape[1])
        ],
    }


def regression_metrics(target: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = target.detach().float().cpu()
    pred = pred.detach().float().cpu()
    if target.shape != pred.shape:
        raise ValueError(f"regression target/pred shape mismatch: {tuple(target.shape)} != {tuple(pred.shape)}")
    return tensor_error_stats((pred - target).abs())


def piper_metrics(target: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    target = target.detach().long().cpu().flatten()
    pred = pred.detach().long().cpu().flatten()
    if target.shape != pred.shape:
        raise ValueError(f"Piper target/pred shape mismatch: {tuple(target.shape)} != {tuple(pred.shape)}")
    matrix = [[0 for _ in range(PIPER_CLASSES)] for _ in range(PIPER_CLASSES)]
    for target_id, pred_id in zip(target.tolist(), pred.tolist(), strict=True):
        if target_id not in range(PIPER_CLASSES) or pred_id not in range(PIPER_CLASSES):
            raise ValueError(f"invalid Piper target/pred pair: {target_id}/{pred_id}")
        matrix[target_id][pred_id] += 1
    support = [sum(row) for row in matrix]
    recalls = [matrix[idx][idx] / support[idx] for idx in range(PIPER_CLASSES) if support[idx] > 0]
    total = sum(support)
    return {
        "records": int(total),
        "accuracy": float((target == pred).float().mean().item()) if total else None,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else None,
        "majority_baseline": float(max(support) / total) if total else None,
        "support": support,
        "predicted_support": [sum(matrix[row][col] for row in range(PIPER_CLASSES)) for col in range(PIPER_CLASSES)],
        "per_class_recall": [
            float(matrix[idx][idx] / support[idx]) if support[idx] else None for idx in range(PIPER_CLASSES)
        ],
        "confusion_target_rows_pred_cols": matrix,
    }


def temporal_stratified_metrics(
    *,
    episode_indices: torch.Tensor,
    frame_indices: torch.Tensor,
    piper_target: torch.Tensor,
    piper_pred: torch.Tensor,
    elite_translation_target: torch.Tensor | None = None,
    elite_translation_pred: torch.Tensor | None = None,
) -> dict[str, Any]:
    episode = episode_indices.detach().long().cpu().flatten()
    frame = frame_indices.detach().long().cpu().flatten()
    target = piper_target.detach().long().cpu().flatten()
    pred = piper_pred.detach().long().cpu().flatten()
    record_count = int(target.numel())
    if not (episode.numel() == frame.numel() == target.numel() == pred.numel()):
        raise ValueError("temporal metric inputs must have equal record counts")
    order = sorted(range(record_count), key=lambda idx: (int(episode[idx]), int(frame[idx])))
    order_tensor = torch.tensor(order, dtype=torch.long)
    episode = episode[order_tensor]
    frame = frame[order_tensor]
    target = target[order_tensor]
    pred = pred[order_tensor]

    eligible = torch.zeros(record_count, dtype=torch.bool)
    transition = torch.zeros(record_count, dtype=torch.bool)
    previous_target = torch.full((record_count,), -1, dtype=torch.long)
    previous_position: dict[int, int] = {}
    for index in range(record_count):
        episode_id = int(episode[index])
        prior = previous_position.get(episode_id)
        if prior is not None:
            eligible[index] = True
            previous_target[index] = target[prior]
            transition[index] = target[index] != target[prior]
        previous_position[episode_id] = index
    steady = eligible & ~transition
    first = ~eligible

    report: dict[str, Any] = {
        "records": record_count,
        "episodes": len(set(int(value) for value in episode.tolist())),
        "first_frame_records": int(first.sum().item()),
        "eligible_temporal_records": int(eligible.sum().item()),
        "transition_records": int(transition.sum().item()),
        "steady_records": int(steady.sum().item()),
        "model_piper_overall": piper_metrics(target, pred),
        "model_piper_transition": piper_metrics(target[transition], pred[transition]),
        "model_piper_steady": piper_metrics(target[steady], pred[steady]),
        "previous_piper_baseline": piper_metrics(target[eligible], previous_target[eligible]),
    }

    if (elite_translation_target is None) != (elite_translation_pred is None):
        raise ValueError("provide both Elite translation target and prediction or neither")
    if elite_translation_target is not None and elite_translation_pred is not None:
        elite_target = elite_translation_target.detach().float().cpu()[order_tensor]
        elite_pred = elite_translation_pred.detach().float().cpu()[order_tensor]
        if elite_target.shape != elite_pred.shape or elite_target.shape != (record_count, 3):
            raise ValueError(
                f"Elite translation target/pred must be shape ({record_count}, 3), "
                f"got {tuple(elite_target.shape)}/{tuple(elite_pred.shape)}"
            )
        previous_elite = torch.zeros_like(elite_target)
        previous_position.clear()
        for index in range(record_count):
            episode_id = int(episode[index])
            prior = previous_position.get(episode_id)
            if prior is not None:
                previous_elite[index] = elite_target[prior]
            previous_position[episode_id] = index
        report.update(
            {
                "model_elite_translation_overall": regression_metrics(elite_target, elite_pred),
                "model_elite_translation_transition": regression_metrics(
                    elite_target[transition], elite_pred[transition]
                ),
                "model_elite_translation_steady": regression_metrics(elite_target[steady], elite_pred[steady]),
                "previous_elite_translation_baseline": regression_metrics(
                    elite_target[eligible], previous_elite[eligible]
                ),
            }
        )
    return report
