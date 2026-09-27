from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_equal(label: str, baseline: Any, overlay: Any) -> None:
    if baseline != overlay:
        raise ValueError(f"paired comparison mismatch for {label}: {baseline!r} != {overlay!r}")


def require_sampler_pair(label: str, baseline: dict[str, Any], overlay: dict[str, Any]) -> None:
    for field in ("mode", "seed", "optimizer_steps", "draws", "base_records", "base_weight_sum"):
        require_equal(f"{label}.{field}", baseline.get(field), overlay.get(field))
    if baseline.get("mode") != "weighted_with_replacement":
        raise ValueError(f"{label} baseline did not use paired weighted sampling")
    if int(baseline.get("overlay_records", -1)) != 0:
        raise ValueError(f"{label} baseline unexpectedly contains overlay records")
    if int(overlay.get("overlay_records", 0)) <= 0:
        raise ValueError(f"{label} overlay run does not contain overlay records")


def metric_change(baseline: float, overlay: float, *, lower_is_better: bool) -> dict[str, Any]:
    absolute = float(overlay - baseline)
    relative = float(absolute / baseline) if baseline != 0.0 else None
    improvement = -absolute if lower_is_better else absolute
    relative_improvement = -relative if lower_is_better and relative is not None else relative
    return {
        "baseline": float(baseline),
        "overlay": float(overlay),
        "overlay_minus_baseline": absolute,
        "relative_change": relative,
        "improvement": improvement,
        "relative_improvement": relative_improvement,
        "lower_is_better": lower_is_better,
    }


def nested(report: dict[str, Any], *keys: str) -> Any:
    value: Any = report
    for key in keys:
        value = value[key]
    return value


def hold_feed_confusion(matrix: list[list[int]]) -> dict[str, Any]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"expected 3x3 Piper confusion matrix, got {matrix!r}")
    return {
        "class_order": ["hold", "feed"],
        "target_rows_pred_cols": [[matrix[row][col] for col in (1, 2)] for row in (1, 2)],
        "predicted_retract_from_hold_feed": int(matrix[1][0] + matrix[2][0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed paired comparison of PI05 baseline and E2 weighted-overlay held-out reports."
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--overlay-report", type=Path, required=True)
    parser.add_argument("--baseline-role-report", type=Path)
    parser.add_argument("--overlay-role-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.baseline_role_report is None) != (args.overlay_role_report is None):
        raise ValueError("provide both --baseline-role-report and --overlay-role-report or neither")

    baseline = read_report(args.baseline_report)
    overlay = read_report(args.overlay_report)
    for field in (
        "root",
        "repo_id",
        "split",
        "split_details",
        "train_samples",
        "val_samples",
        "num_inference_steps",
        "evaluation",
    ):
        require_equal(field, baseline.get(field), overlay.get(field))
    require_equal("validation records", nested(baseline, "result", "records"), nested(overlay, "result", "records"))
    require_equal(
        "Piper validation support",
        nested(baseline, "result", "piper", "support"),
        nested(overlay, "result", "piper", "support"),
    )
    temporal_fields = ("records", "episodes", "transition_records", "steady_records")
    require_equal(
        "temporal validation partition",
        {key: nested(baseline, "result", "temporal_stratification", key) for key in temporal_fields},
        {key: nested(overlay, "result", "temporal_stratification", key) for key in temporal_fields},
    )
    require_equal(
        "Elite checkpoint step",
        baseline["policy_initialization"].get("checkpoint_step"),
        overlay["policy_initialization"].get("checkpoint_step"),
    )
    require_equal(
        "Piper checkpoint step",
        baseline.get("mixed_head_checkpoint_step"),
        overlay.get("mixed_head_checkpoint_step"),
    )
    for field in (
        "mixed_head_train_piper_class_counts",
        "mixed_head_class_weights",
        "mixed_head_class_weight_source",
    ):
        require_equal(field, baseline.get(field), overlay.get(field))
    if baseline.get("mixed_head_class_weight_source") != "base_training_split_only":
        raise ValueError("Piper class weights were not derived from the base training split only")

    base_elite_sampler = baseline["policy_initialization"].get("training_sampler") or {}
    overlay_elite_sampler = overlay["policy_initialization"].get("training_sampler") or {}
    base_piper_sampler = baseline.get("mixed_head_training_sampler") or {}
    overlay_piper_sampler = overlay.get("mixed_head_training_sampler") or {}
    require_sampler_pair("elite_sampler", base_elite_sampler, overlay_elite_sampler)
    require_sampler_pair("piper_sampler", base_piper_sampler, overlay_piper_sampler)
    if baseline["policy_initialization"].get("training_overlay") is not None:
        raise ValueError("baseline Elite checkpoint unexpectedly declares an overlay")
    if baseline.get("mixed_head_training_overlay") is not None:
        raise ValueError("baseline Piper checkpoint unexpectedly declares an overlay")
    elite_overlay = overlay["policy_initialization"].get("training_overlay") or {}
    piper_overlay = overlay.get("mixed_head_training_overlay") or {}
    for field in ("schema", "pack", "records", "episodes", "weight_sum", "role_counts"):
        require_equal(f"overlay provenance.{field}", elite_overlay.get(field), piper_overlay.get(field))
    if elite_overlay.get("policy_training_ready") is not False or elite_overlay.get("formal_data_allowed") is not False:
        raise ValueError("overlay readiness flags changed; comparison rejected")

    temporal_base = nested(baseline, "result", "temporal_stratification")
    temporal_overlay = nested(overlay, "result", "temporal_stratification")
    metrics = {
        "elite_translation_mae": metric_change(
            nested(baseline, "result", "elite_translation_0_3_abs_error", "mean"),
            nested(overlay, "result", "elite_translation_0_3_abs_error", "mean"),
            lower_is_better=True,
        ),
        "elite_translation_transition_mae": metric_change(
            nested(temporal_base, "model_elite_translation_transition", "mean"),
            nested(temporal_overlay, "model_elite_translation_transition", "mean"),
            lower_is_better=True,
        ),
        "elite_translation_steady_mae": metric_change(
            nested(temporal_base, "model_elite_translation_steady", "mean"),
            nested(temporal_overlay, "model_elite_translation_steady", "mean"),
            lower_is_better=True,
        ),
        "piper_accuracy": metric_change(
            nested(baseline, "result", "piper", "accuracy"),
            nested(overlay, "result", "piper", "accuracy"),
            lower_is_better=False,
        ),
        "piper_balanced_accuracy": metric_change(
            nested(baseline, "result", "piper", "balanced_accuracy"),
            nested(overlay, "result", "piper", "balanced_accuracy"),
            lower_is_better=False,
        ),
        "piper_transition_accuracy": metric_change(
            nested(temporal_base, "model_piper_transition", "accuracy"),
            nested(temporal_overlay, "model_piper_transition", "accuracy"),
            lower_is_better=False,
        ),
        "piper_steady_accuracy": metric_change(
            nested(temporal_base, "model_piper_steady", "accuracy"),
            nested(temporal_overlay, "model_piper_steady", "accuracy"),
            lower_is_better=False,
        ),
    }
    base_matrix = nested(baseline, "result", "piper", "confusion_target_rows_pred_cols")
    overlay_matrix = nested(overlay, "result", "piper", "confusion_target_rows_pred_cols")
    role_comparison = None
    if args.baseline_role_report is not None and args.overlay_role_report is not None:
        baseline_roles = read_report(args.baseline_role_report)
        overlay_roles = read_report(args.overlay_role_report)
        if nested(baseline_roles, "evaluation", "set") != "e2_training_overlay":
            raise ValueError("baseline role report is not an E2 training-overlay diagnostic")
        require_equal(
            "role diagnostic evaluation contract",
            baseline_roles.get("offline_overlay_evaluation"),
            overlay_roles.get("offline_overlay_evaluation"),
        )
        base_role_metrics = nested(baseline_roles, "result", "offline_role_stratification")
        overlay_role_metrics = nested(overlay_roles, "result", "offline_role_stratification")
        require_equal("offline role names", sorted(base_role_metrics), sorted(overlay_role_metrics))
        role_comparison = {}
        for role in sorted(base_role_metrics):
            base_role = base_role_metrics[role]
            overlay_role = overlay_role_metrics[role]
            require_equal(f"{role}.records", base_role["records"], overlay_role["records"])
            require_equal(f"{role}.Piper support", base_role["piper"]["support"], overlay_role["piper"]["support"])
            role_comparison[role] = {
                "records": base_role["records"],
                "elite_translation_mae": metric_change(
                    base_role["elite_translation_0_3_abs_error"]["mean"],
                    overlay_role["elite_translation_0_3_abs_error"]["mean"],
                    lower_is_better=True,
                ),
                "piper_accuracy": metric_change(
                    base_role["piper"]["accuracy"],
                    overlay_role["piper"]["accuracy"],
                    lower_is_better=False,
                ),
                "piper_balanced_accuracy": metric_change(
                    base_role["piper"]["balanced_accuracy"],
                    overlay_role["piper"]["balanced_accuracy"],
                    lower_is_better=False,
                ),
            }
    report = {
        "scope": "paired held-out simulation-data algorithm diagnostic; not formal training or real-system validation",
        "status": "compared",
        "baseline_report": str(args.baseline_report),
        "overlay_report": str(args.overlay_report),
        "controlled_contract": {
            "same_base_split_and_validation": True,
            "same_seeds_and_evaluation": True,
            "same_optimizer_steps": True,
            "overlay_train_only": True,
            "overlay_not_used_for_model_selection": True,
            "offline_weights_not_policy_observations": True,
        },
        "metrics": metrics,
        "piper_confusion": {
            "class_order": ["retract", "hold", "feed"],
            "baseline_target_rows_pred_cols": base_matrix,
            "overlay_target_rows_pred_cols": overlay_matrix,
            "baseline_hold_feed": hold_feed_confusion(base_matrix),
            "overlay_hold_feed": hold_feed_confusion(overlay_matrix),
        },
        "offline_training_overlay_role_comparison": role_comparison,
        "overlay_provenance": elite_overlay,
        "interpretation_limit": (
            "One paired seed and two same-task overlay families cannot establish stability. "
            "Use held-out deltas to decide whether a minimal repeat-seed check is warranted; do not infer "
            "rotation, contact/tactile, formal readiness, or real-system effectiveness."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"elite_mae_delta={metrics['elite_translation_mae']['overlay_minus_baseline']:.6f} "
        f"piper_acc_delta={metrics['piper_accuracy']['overlay_minus_baseline']:.6f} "
        f"piper_balanced_delta={metrics['piper_balanced_accuracy']['overlay_minus_baseline']:.6f} "
        f"wrote={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
