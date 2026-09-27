from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_log(path: Path, expected_mode: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    mode = report.get("piper_head_feature_contract", {}).get("mode")
    if mode != expected_mode:
        raise ValueError(f"expected {expected_mode!r} in {path}, got {mode!r}")
    initialization = report.get("policy_initialization", {})
    if initialization.get("status") != "loaded":
        raise ValueError(f"run did not use verified pretrained loading: {path}")
    if float(initialization.get("loaded_parameter_fraction", 0.0)) < 0.99:
        raise ValueError(f"pretrained unique-parameter coverage below 0.99: {path}")
    return report


def comparison_contract(report: dict[str, Any]) -> dict[str, Any]:
    args = report["args"]
    initialization = report["policy_initialization"]
    return {
        "root": report["root"],
        "repo_id": report["repo_id"],
        "split": report["split"],
        "train_samples": report["train_samples"],
        "val_samples": report["val_samples"],
        "train_piper_class_counts": report["train_piper_class_counts"],
        "seed": args["seed"],
        "eval_seed": args["eval_seed"],
        "max_records": args["max_records"],
        "val_fraction": args["val_fraction"],
        "val_batches": args["val_batches"],
        "class_weighting": args["class_weighting"],
        "head_feature_dim": args["head_feature_dim"],
        "pretrained_source": initialization["resolved"]["source"],
        "pretrained_revision": initialization["resolved"]["resolved_revision"],
        "pretrained_unique_parameter_coverage": initialization["loaded_parameter_fraction"],
    }


def summarize_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation["piper"]
    confusion = metrics["confusion_target_rows_pred_cols"]
    predicted_support = [sum(row[column] for row in confusion) for column in range(len(confusion))]
    present_target_classes = sum(value > 0 for value in metrics["support"])
    chance_balanced_accuracy = 1.0 / max(present_target_classes, 1)
    return {
        "piper_ce": evaluation["piper_ce"],
        **metrics,
        "predicted_support": predicted_support,
        "predicted_class_count": sum(value > 0 for value in predicted_support),
        "diagnostic_gates": {
            "beats_majority_accuracy": metrics["accuracy"] > metrics["majority_baseline"],
            "balanced_accuracy_above_present_class_chance": (
                metrics["balanced_accuracy"] > chance_balanced_accuracy
            ),
            "non_degenerate_predictions": sum(value > 0 for value in predicted_support) > 1,
        },
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    validation_points = [
        {"step": int(row["step"]), "evaluation": row["val"]}
        for row in report["steps"]
        if row.get("val") is not None
    ]
    if not validation_points:
        raise ValueError("run contains no step-level validation points")
    best_point = max(
        validation_points,
        key=lambda point: (
            point["evaluation"]["piper"]["balanced_accuracy"],
            point["evaluation"]["piper"]["accuracy"],
            -point["evaluation"]["piper_ce"],
            -point["step"],
        ),
    )
    final = summarize_metrics(report["final_val"])
    best = summarize_metrics(best_point["evaluation"])
    return {
        "feature_contract": report["piper_head_feature_contract"],
        "best_validation": {"step": best_point["step"], **best},
        "final_validation": final,
        "best_minus_final": {
            "accuracy": best["accuracy"] - final["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"] - final["balanced_accuracy"],
            "piper_ce": best["piper_ce"] - final["piper_ce"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare controlled state_only and dimension-balanced prefix_state Piper-head runs."
    )
    parser.add_argument("--state-only-log", type=Path, required=True)
    parser.add_argument("--prefix-state-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    state_only_log = load_log(args.state_only_log, "state_only")
    prefix_state_log = load_log(args.prefix_state_log, "prefix_state")
    state_contract = comparison_contract(state_only_log)
    prefix_contract = comparison_contract(prefix_state_log)
    if state_contract != prefix_contract:
        differences = {
            key: {"state_only": state_contract.get(key), "prefix_state": prefix_contract.get(key)}
            for key in sorted(set(state_contract) | set(prefix_contract))
            if state_contract.get(key) != prefix_contract.get(key)
        }
        raise ValueError(f"runs are not a controlled comparison: {differences}")

    state_result = summarize(state_only_log)
    prefix_result = summarize(prefix_state_log)
    report = {
        "scope": "controlled Piper-head algorithm ablation; not real-system validation",
        "status": "compared",
        "contract": state_contract,
        "state_only": state_result,
        "prefix_state": prefix_result,
        "prefix_state_minus_state_only": {
            "best_validation": {
                metric: (
                    prefix_result["best_validation"][metric]
                    - state_result["best_validation"][metric]
                )
                for metric in ("accuracy", "balanced_accuracy", "piper_ce")
            },
            "final_validation": {
                metric: (
                    prefix_result["final_validation"][metric]
                    - state_result["final_validation"][metric]
                )
                for metric in ("accuracy", "balanced_accuracy", "piper_ce")
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"state_best_step={state_result['best_validation']['step']} "
        f"state_best_balanced_acc={state_result['best_validation']['balanced_accuracy']:.4f} "
        f"prefix_best_step={prefix_result['best_validation']['step']} "
        f"prefix_best_balanced_acc={prefix_result['best_validation']['balanced_accuracy']:.4f} "
        f"delta={report['prefix_state_minus_state_only']['best_validation']['balanced_accuracy']:.4f} "
        f"wrote={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
