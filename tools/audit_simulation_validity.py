from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIAGNOSTIC_ROLLOUT_KEYS = {
    "elite_tip_anchor": "oracle tip-relative Elite anchor",
    "tactile_safety": "exact simulated tactile/contact safety feedback",
    "scripted_piper_feed": "scripted Piper control",
    "piper_feed_phase_guard": "scripted Piper phase guard",
    "show_path_tubes": "debug path visual leakage",
    "show_tool_markers": "debug tool marker leakage",
    "diagnostic_video_overlay": "diagnostic video overlay",
    "diagnostic_video_hide_robots": "diagnostic-only video rendering change",
}

ORACLE_DATASET_MODES = {
    "mujoco_physical_guidance": "MuJoCoMagneticGuideExpert uses exact simulated tip, local vessel frame, lookahead, contact/wall state, and IK.",
    "tip_guided_wire": "TipMagneticGuideExpert uses exact simulated tip, local vessel frame, lookahead, contact/wall state, and IK.",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def path_kind(path: Path) -> str:
    if path.is_file() and path.name == "manifest.json":
        return "dataset"
    if path.is_file() and path.name == "meta.json":
        return "rollout_task"
    if path.is_dir() and (path / "manifest.json").exists():
        return "dataset"
    if path.is_dir() and (path / "summary.json").exists():
        return "rollout"
    if path.is_dir() and (path / "meta.json").exists():
        return "rollout_task"
    return "unknown"


def verdict(status: str, formal_allowed: bool, reasons: list[str], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "formal_allowed": bool(formal_allowed),
        "reasons": reasons,
        "warnings": warnings or [],
    }


def audit_dataset(path: Path) -> dict[str, Any]:
    manifest_path = path if path.is_file() else path / "manifest.json"
    manifest = load_json(manifest_path)
    reasons: list[str] = []
    warnings: list[str] = []

    embedded = manifest.get("validity_audit")
    if isinstance(embedded, dict):
        formal_allowed = bool(embedded.get("formal_data_allowed", embedded.get("allowed_for_formal_data", False)))
        status = "formal_valid" if formal_allowed else "diagnostic_oracle"
        reason = str(embedded.get("reason", "manifest validity_audit marks this dataset as not formal-valid"))
        reasons.append(reason)
        return {
            "path": str(manifest_path),
            "kind": "dataset",
            "mode": manifest.get("mode"),
            "verdict": verdict(status, formal_allowed, reasons, warnings),
            "embedded_validity_audit": embedded,
        }

    mode = str(manifest.get("mode", ""))
    if mode in ORACLE_DATASET_MODES:
        reasons.append(ORACLE_DATASET_MODES[mode])
        reasons.append("legacy manifest has no validity_audit field; inferred from known collection script semantics")
        status = "diagnostic_oracle"
        formal_allowed = False
    else:
        reasons.append("no validity_audit field and mode is not recognized by this tool")
        status = "unknown_legacy"
        formal_allowed = False

    action_type = manifest.get("action_schema", {}).get("type") if isinstance(manifest.get("action_schema"), dict) else None
    if action_type and action_type != "piper_feed_elite_joint":
        warnings.append(f"action schema is {action_type}, not current piper_feed_elite_joint")

    return {
        "path": str(manifest_path),
        "kind": "dataset",
        "mode": manifest.get("mode"),
        "episodes": len(manifest.get("episodes", [])) if isinstance(manifest.get("episodes"), list) else None,
        "samples": len(manifest.get("samples", [])) if isinstance(manifest.get("samples"), list) else None,
        "verdict": verdict(status, formal_allowed, reasons, warnings),
    }


def audit_rollout_meta(meta_path: Path) -> dict[str, Any]:
    meta = load_json(meta_path)
    reasons: list[str] = []
    warnings: list[str] = []
    embedded = meta.get("validity_audit")
    if isinstance(embedded, dict):
        formal_allowed = bool(embedded.get("allowed_for_formal_eval", False))
        status = "formal_valid" if formal_allowed else "diagnostic_oracle"
        interventions = embedded.get("diagnostic_or_oracle_interventions", [])
        if interventions:
            reasons.append("diagnostic/oracle interventions: " + ", ".join(str(x) for x in interventions))
        else:
            reasons.append(str(embedded.get("note", "embedded validity_audit present")))
        return {
            "path": str(meta_path),
            "kind": "rollout_task",
            "task": meta.get("task"),
            "env_type": meta.get("env_type"),
            "success": meta.get("success"),
            "verdict": verdict(status, formal_allowed, reasons, warnings),
            "embedded_validity_audit": embedded,
        }

    interventions = []
    for key, label in DIAGNOSTIC_ROLLOUT_KEYS.items():
        if bool(meta.get(key, False)):
            interventions.append((key, label))
    if interventions:
        reasons.extend(f"{key}: {label}" for key, label in interventions)
        reasons.append("legacy rollout meta has no validity_audit field; inferred from recorded flags")
        status = "diagnostic_oracle"
        formal_allowed = False
    else:
        reasons.append("legacy rollout meta has no validity_audit field and no known diagnostic/oracle flags were recorded")
        status = "unknown_legacy"
        formal_allowed = False

    if meta.get("env_type") == "tip":
        warnings.append("tip-centric dynamics are plausible but still an assumption; see simulation-expert-validity-audit.md")

    return {
        "path": str(meta_path),
        "kind": "rollout_task",
        "task": meta.get("task"),
        "env_type": meta.get("env_type"),
        "success": meta.get("success"),
        "verdict": verdict(status, formal_allowed, reasons, warnings),
    }


def audit_rollout(path: Path) -> dict[str, Any]:
    task_reports = []
    for meta_path in sorted(path.glob("*/meta.json")):
        task_reports.append(audit_rollout_meta(meta_path))
    if not task_reports and (path / "meta.json").exists():
        return audit_rollout_meta(path / "meta.json")
    if not task_reports:
        return {
            "path": str(path),
            "kind": "rollout",
            "verdict": verdict("unknown_legacy", False, ["rollout directory has no task meta.json files"]),
            "tasks": [],
        }

    formal_allowed = all(bool(r["verdict"]["formal_allowed"]) for r in task_reports)
    statuses = sorted({str(r["verdict"]["status"]) for r in task_reports})
    status = "formal_valid" if formal_allowed else ("diagnostic_oracle" if "diagnostic_oracle" in statuses else "unknown_legacy")
    reasons = []
    for report in task_reports:
        task = report.get("task") or Path(str(report["path"])).parent.name
        for reason in report["verdict"]["reasons"]:
            reasons.append(f"{task}: {reason}")
    return {
        "path": str(path),
        "kind": "rollout",
        "verdict": verdict(status, formal_allowed, reasons),
        "tasks": task_reports,
    }


def audit_path(path: Path) -> dict[str, Any]:
    kind = path_kind(path)
    if kind == "dataset":
        return audit_dataset(path)
    if kind == "rollout":
        return audit_rollout(path)
    if kind == "rollout_task":
        meta_path = path if path.is_file() else path / "meta.json"
        return audit_rollout_meta(meta_path)
    return {
        "path": str(path),
        "kind": "unknown",
        "verdict": verdict("unknown", False, ["path is not a recognized manifest or rollout directory"]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Simulation Validity Audit",
        "",
        f"generated_at: {report['generated_at']}",
        "",
        "| Path | Kind | Status | Formal allowed | Reasons |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["items"]:
        v = item["verdict"]
        reasons = "<br>".join(str(x) for x in v["reasons"])
        lines.append(
            f"| `{item['path']}` | {item['kind']} | {v['status']} | {v['formal_allowed']} | {reasons} |"
        )
    lines.append("")
    lines.append("Status meanings:")
    lines.append("")
    lines.append("- `formal_valid`: metadata says the artifact passes the current formal-validity guard; this does not imply task success or physical quality.")
    lines.append("- `diagnostic_oracle`: useful for diagnostics/ablations, not formal data-generation evidence.")
    lines.append("- `unknown_legacy`: old artifact lacks enough metadata; treat as not formal-valid until rerun or reviewed.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit simulation datasets/rollouts for formal sim-to-real validity.")
    parser.add_argument("paths", nargs="+", help="Dataset manifest/path or rollout directory/meta path.")
    parser.add_argument("--out", help="Optional output directory for JSON and Markdown reports.")
    args = parser.parse_args()

    items = [audit_path(Path(p)) for p in args.paths]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "simulation_validity_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "simulation_validity_audit.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
