from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.train_dual_arm_baseline import DualArmCameraStatePolicy, build_state_vector


PIPER_LABELS = {-1: "retract", 0: "hold", 1: "feed"}


def image_to_tensor(img_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


def load_model(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DualArmCameraStatePolicy(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt.get("action_dim", 4)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "image_size": int(ckpt["image_size"]),
        "max_steps": int(ckpt.get("max_steps", 560)),
        "observation_schema": ckpt.get("observation_schema", "full_sim_state"),
        "piper_command_period_for_state": int(ckpt.get("piper_command_period_for_state", 40)),
        "action_mode": ckpt.get("action_mode", "legacy"),
        "piper_head": ckpt.get("piper_head", "feed_regression"),
        "piper_step_class_values": list(ckpt.get("piper_step_class_values", [-1, 0, 1])),
        "piper_step_feed_value": float(ckpt.get("piper_step_feed_value", 0.7)),
        "piper_step_retract_value": float(ckpt.get("piper_step_retract_value", 0.7)),
        "elite_action_representation": ckpt.get("elite_action_representation", "absolute"),
    }


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def state_from_record(record: dict[str, Any], *, task: str, step: int) -> dict[str, Any]:
    state = dict(record.get("state", {}))
    for key in (
        "elite_tcp_pose_6d",
        "piper_step",
        "piper_insertion_length",
        "estimated_tip_pos_3d",
        "estimated_tip_heading_3d",
        "tip_estimator_confidence",
        "tip_estimator_visible",
        "estimated_contact_flag",
        "contact_estimator_confidence",
        "estimated_image_distance_px",
        "estimated_wall_margin",
        "estimated_wall_margin_fraction",
        "route_estimator_confidence",
        "estimated_magnet_wall_pull",
        "estimated_wall_side_risk",
        "estimated_wall_normal_3d",
        "estimated_route_tangent_3d",
    ):
        if key in record and key not in state:
            state[key] = record[key]
    return {
        "task": record.get("task", task),
        "step": int(record.get("step", step)),
        "state": state,
    }


def decode_policy_output(pred: np.ndarray, model_bundle: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if model_bundle["action_mode"] != "piper_feed_elite_joint":
        raise ValueError(f"Shadow adapter expects piper_feed_elite_joint, got {model_bundle['action_mode']}")

    piper_step_command: int | None = None
    piper_feed = 0.0
    elite_start = 1
    if model_bundle.get("piper_head", "feed_regression") == "step_classification":
        class_values = list(model_bundle.get("piper_step_class_values", [-1, 0, 1]))
        piper_logits = pred[: len(class_values)]
        class_idx = int(np.argmax(piper_logits))
        piper_step_command = int(class_values[class_idx])
        if piper_step_command > 0:
            piper_feed = float(model_bundle.get("piper_step_feed_value", 0.7))
        elif piper_step_command < 0:
            piper_feed = -float(model_bundle.get("piper_step_retract_value", 0.7))
        elite_start = len(class_values)
    else:
        piper_feed = float(np.clip(pred[0], -1.0, 1.0))
        if piper_feed > 0.35:
            piper_step_command = 1
        elif piper_feed < -0.35:
            piper_step_command = -1
        else:
            piper_step_command = 0

    elite_action_representation = str(model_bundle.get("elite_action_representation", "absolute"))
    if elite_action_representation != "tcp_delta":
        raise ValueError(f"Shadow adapter expects tcp_delta Elite action, got {elite_action_representation}")

    elite_tcp_delta = pred[elite_start : elite_start + 6].astype(np.float32).copy()
    elite_tcp_delta[3:] = 0.0
    current_tcp = np.asarray(state["elite_tcp_pose_6d"], dtype=np.float32).reshape(6)
    elite_tcp_pose = current_tcp.copy()
    elite_tcp_pose[:3] = elite_tcp_pose[:3] + elite_tcp_delta[:3]
    return {
        "raw_prediction": pred.astype(float).tolist(),
        "piper_feed": float(piper_feed),
        "piper_step_command": int(piper_step_command),
        "piper_command_label": PIPER_LABELS[int(piper_step_command)],
        "elite_tcp_delta_6d": elite_tcp_delta.astype(float).tolist(),
        "elite_tcp_pose_6d": elite_tcp_pose.astype(float).tolist(),
    }


def apply_elite_only_wallguard(action: dict[str, Any], state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    status: dict[str, Any] = {
        "enabled": bool(args.wallguard),
        "active": False,
        "elite_corrected": False,
        "reason": "disabled",
        "piper_modified": False,
    }
    guarded = dict(action)
    if not args.wallguard:
        return {"action": guarded, "status": status}

    route_conf = float(state.get("route_estimator_confidence", 0.0) or 0.0)
    tip_conf = float(state.get("tip_estimator_confidence", 0.0) or 0.0)
    status["route_estimator_confidence"] = route_conf
    status["tip_estimator_confidence"] = tip_conf
    if route_conf < float(args.wallguard_min_confidence) or tip_conf < float(args.wallguard_min_confidence):
        status["reason"] = "low_estimator_confidence"
        return {"action": guarded, "status": status}

    risk = float(state.get("estimated_wall_side_risk", 0.0) or 0.0)
    margin = state.get("estimated_wall_margin")
    margin = float(margin) if margin is not None else float("inf")
    pull = float(state.get("estimated_magnet_wall_pull", 0.0) or 0.0)
    status.update({"risk": risk, "margin": margin, "magnet_wall_pull": pull})

    triggers = []
    if risk >= float(args.wallguard_risk_threshold):
        triggers.append("risk")
    if margin <= float(args.wallguard_margin_threshold_m):
        triggers.append("margin")
    if pull >= float(args.wallguard_pull_threshold_m):
        triggers.append("magnet_wall_pull")
    if not triggers:
        status["reason"] = "below_threshold"
        return {"action": guarded, "status": status}

    normal = state.get("estimated_wall_normal_3d")
    if normal is None:
        status["reason"] = "+".join(triggers)
        status["elite_correction_reason"] = "missing_wall_normal"
        return {"action": guarded, "status": status}
    normal_vec = np.asarray(normal, dtype=np.float32).reshape(3)
    normal_norm = float(np.linalg.norm(normal_vec))
    if normal_norm <= 1e-6:
        status["reason"] = "+".join(triggers)
        status["elite_correction_reason"] = "invalid_wall_normal"
        return {"action": guarded, "status": status}
    normal_vec = normal_vec / normal_norm

    delta = np.asarray(action["elite_tcp_delta_6d"], dtype=np.float32).reshape(6).copy()
    original_xyz = delta[:3].copy()
    toward_wall_mm = float(np.dot(original_xyz, normal_vec))
    status["active"] = True
    status["reason"] = "+".join(triggers)
    status["toward_wall_mm"] = toward_wall_mm
    if toward_wall_mm <= 0.0:
        status["elite_correction_reason"] = "no_toward_wall_component"
        return {"action": guarded, "status": status}

    correction = -normal_vec * toward_wall_mm * float(args.wallguard_correction_blend)
    correction_norm = float(np.linalg.norm(correction))
    max_correction = float(args.wallguard_max_correction_mm)
    if max_correction > 0.0 and correction_norm > max_correction:
        correction = correction * (max_correction / max(correction_norm, 1e-6))
    delta[:3] = original_xyz + correction
    current_tcp = np.asarray(state["elite_tcp_pose_6d"], dtype=np.float32).reshape(6)
    target = current_tcp.copy()
    target[:3] = target[:3] + delta[:3]
    guarded["elite_tcp_delta_6d"] = delta.astype(float).tolist()
    guarded["elite_tcp_pose_6d"] = target.astype(float).tolist()
    status.update(
        {
            "elite_corrected": True,
            "correction_delta_mm": correction.astype(float).tolist(),
            "original_elite_tcp_delta_6d": action["elite_tcp_delta_6d"],
            "corrected_elite_tcp_delta_6d": guarded["elite_tcp_delta_6d"],
        }
    )
    return {"action": guarded, "status": status}


def iter_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_jsonl:
        records = []
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    record = json.loads(line)
                    record["_line_no"] = line_no
                    records.append(record)
        return records
    if not args.side_image or not args.top_image or not args.state_json:
        raise ValueError("Provide either --input-jsonl or --side-image --top-image --state-json")
    record = json.loads(Path(args.state_json).read_text(encoding="utf-8"))
    record["side_image"] = args.side_image
    record["top_image"] = args.top_image
    return [record]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current dual-camera/state policy in real-system shadow mode. "
            "This script records raw and guarded actions only; it never commands robots."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-jsonl", help="JSONL records with side_image, top_image, task, step, and state fields.")
    parser.add_argument("--side-image", help="Single-record side image path.")
    parser.add_argument("--top-image", help="Single-record top image path.")
    parser.add_argument("--state-json", help="Single-record state JSON path.")
    parser.add_argument("--image-base-dir", default=".", help="Base directory for relative image paths in JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N records; 0 means all records.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wallguard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wallguard-risk-threshold", type=float, default=0.50)
    parser.add_argument("--wallguard-margin-threshold-m", type=float, default=0.0015)
    parser.add_argument("--wallguard-pull-threshold-m", type=float, default=0.0025)
    parser.add_argument("--wallguard-min-confidence", type=float, default=0.35)
    parser.add_argument("--wallguard-correction-blend", type=float, default=1.0)
    parser.add_argument("--wallguard-max-correction-mm", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_bundle = load_model(Path(args.checkpoint), device)
    model = model_bundle["model"]
    image_size = int(model_bundle["image_size"])
    max_steps = int(args.max_steps or model_bundle["max_steps"])
    base_dir = Path(args.image_base_dir)
    records = iter_records(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "records": 0,
        "feed": 0,
        "hold": 0,
        "retract": 0,
        "wallguard_active": 0,
        "wallguard_corrected": 0,
        "reference_records": 0,
        "piper_reference_mismatch": 0,
    }
    with out_path.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if args.limit and args.limit > 0 and idx >= int(args.limit):
                break
            task = str(record.get("task", args.task))
            step = int(record.get("step", idx))
            sample = state_from_record(record, task=task, step=step)
            side_path = resolve_path(str(record["side_image"]), base_dir)
            top_path = resolve_path(str(record["top_image"]), base_dir)
            side = image_to_tensor(read_image(side_path), image_size, device)
            top = image_to_tensor(read_image(top_path), image_size, device)
            state_vec = torch.tensor(
                build_state_vector(
                    sample,
                    max_steps,
                    None,
                    None,
                    model_bundle.get("observation_schema", "full_sim_state"),
                    int(model_bundle.get("piper_command_period_for_state", 40)),
                ),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                pred = model(side, top, state_vec).squeeze(0).detach().cpu().numpy().astype(np.float32)
            raw_action = decode_policy_output(pred, model_bundle, sample["state"])
            guarded = apply_elite_only_wallguard(raw_action, sample["state"], args)
            label = raw_action["piper_command_label"]
            counts["records"] += 1
            counts[label] += 1
            counts["wallguard_active"] += int(bool(guarded["status"].get("active")))
            counts["wallguard_corrected"] += int(bool(guarded["status"].get("elite_corrected")))
            reference_action = record.get("reference_action")
            if reference_action is not None:
                counts["reference_records"] += 1
                reference_step = int(reference_action.get("piper_step_command", 0))
                counts["piper_reference_mismatch"] += int(reference_step != int(raw_action["piper_step_command"]))
            f.write(
                json.dumps(
                    {
                        "index": idx,
                        "source_line": record.get("_line_no"),
                        "task": task,
                        "step": step,
                        "side_image": str(side_path),
                        "top_image": str(top_path),
                        "observation_schema": model_bundle.get("observation_schema"),
                        "input_quality": record.get("input_quality"),
                        "reference_action": reference_action,
                        "raw_policy_action": raw_action,
                        "shadow_controller_action": guarded["action"],
                        "wallguard": guarded["status"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"wrote shadow predictions to {out_path}")


if __name__ == "__main__":
    main()
