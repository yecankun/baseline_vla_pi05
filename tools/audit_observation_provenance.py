from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIRECT_FIELDS = {
    "task",
    "instruction",
    "piper_step",
    "piper_insertion_length",
    "elite_tcp_pose_6d",
    "robot_state.elite_tcp_pose_6d",
    "robot_state.elite_joints",
    "robot_state.piper_joints",
}

CONTROLLER_FIELDS = {
    "controller_state.piper_intent",
    "controller_state.piper_executed_command",
    "controller_state.piper_requested_feed",
    "controller_state.piper_executed_feed",
    "controller_state.piper_motion_state",
    "controller_state.piper_step_count",
    "controller_state.piper_insertion_length",
    "controller_state.piper_busy",
    "controller_state.piper_cooldown",
    "controller_state.elite_action_type",
    "controller_state.elite_tcp_delta_6d",
    "controller_state.elite_requested_tcp_pose_6d",
    "controller_state.elite_executed_tcp_pose_6d",
    "controller_state.elite_requested_joints",
    "controller_state.elite_executed_joints",
    "controller_state.elite_target_limited",
}

ESTIMATOR_FIELDS = {
    "estimated_tip_pos_3d",
    "estimated_tip_pos_frame",
    "estimated_tip_camera_pos_3d",
    "estimated_tip_depth_m",
    "estimated_tip_pixel_source",
    "estimated_tip_heading_3d",
    "tip_estimator_confidence",
    "tip_estimator_visible",
    "tip_estimator_latency_steps",
    "estimated_tip_pixel_side",
    "estimated_tip_pixel_top",
    "estimated_wall_pixel_side",
    "estimated_wall_pixel_top",
    "estimated_contact_flag",
    "estimated_contact_strength",
    "estimated_contact_direction_2d",
    "estimated_image_distance_px",
    "contact_estimator_confidence",
    "contact_source",
    "estimated_wall_margin",
    "estimated_wall_margin_fraction",
    "estimated_route_progress",
    "estimated_route_index",
    "route_estimator_confidence",
    "registered_route_id",
    "estimated_magnet_wall_pull",
    "estimated_wall_side_risk",
    "estimated_wall_normal_3d",
    "estimated_route_tangent_3d",
    "registered_geometry_estimator",
}

PRIVILEGED_FIELDS = {
    "tip_pos",
    "heading",
    "target_pos",
    "contact_flag",
    "contact_direction",
    "contact_normal",
    "contact_strength",
    "distance_to_wall",
    "segment_min_distance_to_wall",
    "path_progress",
    "boundary_projection_count",
    "boundary_projection_window",
    "last_boundary_projection",
    "magnetic_pose",
    "lateral_offset",
    "path_tangent",
    "local_radius",
}

SCHEMA_INPUTS = {
    "senior_piper_real_like": {
        "elite_tcp_pose_6d",
        "piper_step",
        "task",
    },
    "senior_piper_real_like_with_phase": {
        "elite_tcp_pose_6d",
        "piper_step",
        "task",
        "controller_phase",
    },
    "real_direct": {
        "piper_step",
        "piper_insertion_length",
        "elirobot_pose",
        "robot_state.piper_joints",
        "robot_state.elite_joints",
        "task",
        "step_norm",
    },
    "real_direct_plus_estimated_tip_contact": {
        "elite_tcp_pose_6d",
        "piper_step",
        "piper_insertion_length",
        "task",
        "estimated_tip_pos_3d",
        "estimated_tip_heading_3d",
        "tip_estimator_confidence",
        "tip_estimator_visible",
        "estimated_contact_flag",
        "contact_estimator_confidence",
        "contact_source",
    },
    "real_direct_plus_estimated_tip_registered_geometry": {
        "elite_tcp_pose_6d",
        "piper_step",
        "piper_insertion_length",
        "task",
        "estimated_tip_pos_3d",
        "estimated_tip_heading_3d",
        "tip_estimator_confidence",
        "tip_estimator_visible",
        "estimated_contact_flag",
        "estimated_image_distance_px",
        "contact_estimator_confidence",
        "contact_source",
        "estimated_wall_margin",
        "estimated_wall_margin_fraction",
        "route_estimator_confidence",
        "estimated_magnet_wall_pull",
        "estimated_wall_side_risk",
        "estimated_wall_normal_3d",
        "estimated_route_tangent_3d",
    },
    "real_direct_plus_visual_contact": {
        "elite_tcp_pose_6d",
        "piper_step",
        "piper_insertion_length",
        "task",
        "estimated_contact_flag",
        "estimated_image_distance_px",
        "contact_estimator_confidence",
        "contact_source",
    },
    "full_sim_state": {
        "tip_pos",
        "heading",
        "target_pos",
        "contact_normal",
        "distance_to_wall",
        "contact_strength",
        "contact_flag",
        "path_progress",
        "piper_step",
        "piper_insertion_length",
        "elirobot_pose",
        "lateral_offset",
        "path_tangent",
        "local_radius",
        "robot_state.piper_joints",
        "robot_state.elite_joints",
        "task",
        "step_norm",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def manifest_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "manifest.json"
    if path.name != "manifest.json":
        raise ValueError(f"Expected manifest.json or dataset directory: {path}")
    return path


def flatten_state(state: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for key, value in state.items():
        fields.add(str(key))
        if isinstance(value, dict):
            for subkey in value:
                fields.add(f"{key}.{subkey}")
    return fields


def sample_states(manifest: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        return []
    states = []
    for sample in samples[: max(limit, 0)]:
        if isinstance(sample, dict) and isinstance(sample.get("state"), dict):
            states.append(sample["state"])
    return states


def field_tier(field: str) -> str:
    if field in DIRECT_FIELDS or field in {"task", "step_norm", "controller_phase"}:
        return "direct_or_controller_clock"
    if field in CONTROLLER_FIELDS:
        return "controller"
    if field in ESTIMATOR_FIELDS:
        return "estimator"
    if field in PRIVILEGED_FIELDS:
        return "privileged"
    if field == "elirobot_pose":
        return "direct_if_calibrated"
    if field.startswith("robot_state."):
        return "direct_robot_state"
    if field.startswith("controller_state."):
        return "controller"
    if field.startswith("estimated_") or field.endswith("_confidence"):
        return "estimator"
    return "unknown"


def load_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("observation_provenance", {})
    return provenance if isinstance(provenance, dict) else {}


def audit_schema(schema: str, provenance: dict[str, Any]) -> dict[str, Any]:
    inputs = SCHEMA_INPUTS.get(schema)
    if inputs is None:
        return {
            "schema": schema,
            "known_schema": False,
            "formal_allowed": False,
            "errors": [f"unknown observation schema: {schema}"],
            "warnings": [],
            "inputs": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    input_rows = []
    for field in sorted(inputs):
        tier = field_tier(field)
        prov = provenance.get(field, {})
        if tier == "privileged":
            errors.append(f"{schema} consumes privileged simulator field: {field}")
        if tier == "estimator":
            if not isinstance(prov, dict) or not prov:
                errors.append(f"{schema} consumes estimator field without observation_provenance: {field}")
            elif not bool(prov.get("formal_policy_input_allowed", False)):
                errors.append(f"{schema} estimator field is not marked formal_policy_input_allowed: {field}")
            for required_key in ["tier", "source", "formal_policy_input_allowed"]:
                if isinstance(prov, dict) and required_key not in prov:
                    errors.append(f"{field} provenance missing required key: {required_key}")
        if tier == "direct_if_calibrated":
            warnings.append(f"{field} is allowed only if calibrated to the real robot/camera frame")
        if field == "controller_phase":
            warnings.append("controller_phase is allowed only if it is real controller state, not hidden route-plan phase")
        input_rows.append({"field": field, "tier": tier, "provenance": prov})

    return {
        "schema": schema,
        "known_schema": True,
        "formal_allowed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "inputs": input_rows,
    }


def audit_manifest(path: Path, schema: str, sample_limit: int) -> dict[str, Any]:
    mp = manifest_path(path)
    manifest = load_json(mp)
    provenance = load_provenance(manifest)
    states = sample_states(manifest, sample_limit)
    observed_fields: set[str] = set()
    for state in states:
        observed_fields.update(flatten_state(state))

    privileged_present = sorted(field for field in observed_fields if field_tier(field) == "privileged")
    estimator_present = sorted(field for field in observed_fields if field_tier(field) == "estimator")
    controller_present = sorted(field for field in observed_fields if field_tier(field) == "controller")
    schema_report = audit_schema(schema, provenance)

    warnings = list(schema_report["warnings"])
    if privileged_present:
        warnings.append(
            "samples store privileged diagnostic fields; this is acceptable only if the selected policy schema does not consume them"
        )
    if estimator_present and not provenance:
        warnings.append("samples contain estimator-looking fields but manifest has no observation_provenance block")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(mp),
        "mode": manifest.get("mode"),
        "samples_total": len(manifest.get("samples", [])) if isinstance(manifest.get("samples"), list) else None,
        "sampled_states": len(states),
        "observation_schema": schema,
        "formal_allowed_for_schema": bool(schema_report["formal_allowed"]),
        "schema_report": schema_report,
        "observed_field_summary": {
            "privileged_present": privileged_present,
            "estimator_present": estimator_present,
            "controller_present": controller_present,
        },
        "warnings": warnings,
        "recommendation": recommendation(schema_report),
    }


def recommendation(schema_report: dict[str, Any]) -> str:
    if schema_report["formal_allowed"]:
        return "Schema passes this provenance audit. Still verify dataset quality and sim-to-real assumptions separately."
    return "Keep this run diagnostic. Do not treat it as formal sim-to-real training evidence until errors are resolved."


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Observation Provenance Audit",
        "",
        f"generated_at: `{report['generated_at']}`",
        f"manifest: `{report['manifest']}`",
        f"observation_schema: `{report['observation_schema']}`",
        f"formal_allowed_for_schema: `{report['formal_allowed_for_schema']}`",
        "",
        "## Schema Inputs",
        "",
        "| Field | Tier | Provenance source | Formal allowed |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["schema_report"]["inputs"]:
        prov = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        source = prov.get("source", "") if prov else ""
        allowed = prov.get("formal_policy_input_allowed", "") if prov else ""
        lines.append(f"| `{row['field']}` | {row['tier']} | {source} | {allowed} |")

    errors = report["schema_report"]["errors"]
    warnings = report["warnings"]
    lines.extend(["", "## Errors", ""])
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- none")

    summary = report["observed_field_summary"]
    lines.extend(
        [
            "",
            "## Observed Stored Fields",
            "",
            "These fields were found in sampled states. Stored fields are not automatically policy inputs.",
            "",
            f"- privileged_present: `{', '.join(summary['privileged_present']) if summary['privileged_present'] else 'none'}`",
            f"- estimator_present: `{', '.join(summary['estimator_present']) if summary['estimator_present'] else 'none'}`",
            f"- controller_present: `{', '.join(summary['controller_present']) if summary['controller_present'] else 'none'}`",
            "",
            "## Recommendation",
            "",
            report["recommendation"],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether a dataset manifest's selected observation schema is real-observable/provenance-safe."
    )
    parser.add_argument("manifest", help="Dataset manifest.json or dataset directory.")
    parser.add_argument(
        "--observation-schema",
        default="senior_piper_real_like",
        help="Observation schema to audit as policy input.",
    )
    parser.add_argument("--sample-limit", type=int, default=200, help="Number of manifest samples to inspect.")
    parser.add_argument("--out", help="Optional output directory for JSON and Markdown reports.")
    args = parser.parse_args()

    report = audit_manifest(Path(args.manifest), args.observation_schema, args.sample_limit)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "observation_provenance_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "observation_provenance_audit.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
