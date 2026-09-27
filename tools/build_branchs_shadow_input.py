from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np


PIPER_LABELS = {0: "hold", 1: "feed"}


def numeric_suffix(path: Path, prefix: str) -> int:
    text = path.stem if path.is_file() else path.name
    if not text.startswith(prefix):
        raise ValueError(f"Expected {prefix}* name, got {path.name}")
    return int(text[len(prefix) :])


def read_pose_file(path: Path) -> list[list[float]]:
    poses: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            pose = ast.literal_eval(line)
            if len(pose) != 6:
                raise ValueError(f"{path}:{line_no} expected 6D pose, got {len(pose)}")
            poses.append([float(v) for v in pose])
    return poses


def read_label_file(path: Path) -> list[int]:
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = int(float(line))
            if value not in PIPER_LABELS:
                raise ValueError(f"{path}:{line_no} expected Piper label 0/1, got {value}")
            labels.append(value)
    return labels


def branch_task(branch_name: str, args: argparse.Namespace) -> str:
    if branch_name == "branch1":
        return args.branch1_task
    if branch_name == "branch2":
        return args.branch2_task
    return args.default_task


def build_record(
    *,
    branch_name: str,
    path_name: str,
    frame_number: int,
    image_path: Path,
    pose: list[float],
    piper_step_before_action: int,
    piper_label: int,
    task: str,
    branch_root: Path,
    visual_contact: dict | None = None,
) -> dict:
    image_text = str(image_path.as_posix())
    visual_contact = visual_contact or {}
    estimated_contact_flag = float(visual_contact.get("estimated_contact_flag", 0.0))
    contact_estimator_confidence = float(visual_contact.get("contact_estimator_confidence", 0.0))
    estimated_image_distance_px = float(visual_contact.get("estimated_image_distance_px", 0.0))
    estimated_tip_pixel_side = visual_contact.get("estimated_tip_pixel_side")
    return {
        "task": task,
        "step": int(frame_number - 1),
        "branch": branch_name,
        "path": path_name,
        "frame_number": int(frame_number),
        "side_image": image_text,
        "top_image": image_text,
        "input_quality": {
            "source": "branchs_senior_real_data",
            "single_camera_duplicated_as_top": True,
            "registered_geometry_available": False,
            "estimated_tip_3d_available": False,
            "visual_contact_estimator": visual_contact.get("method"),
            "note": (
                "Diagnostic shadow input only: branchs has one camera, Elite 6D pose, "
                "and Piper 0/1 labels, but no top camera or registered-geometry estimator fields."
            ),
        },
        "state": {
            "elite_tcp_pose_6d": pose,
            "piper_step": float(piper_step_before_action),
            "piper_insertion_length": 0.0,
            "estimated_tip_pos_3d": [0.0, 0.0, 0.0],
            "estimated_tip_heading_3d": [0.0, 0.0, 0.0],
            "tip_estimator_confidence": 0.0,
            "tip_estimator_visible": 0.0,
            "estimated_contact_flag": estimated_contact_flag,
            "contact_estimator_confidence": contact_estimator_confidence,
            "estimated_image_distance_px": estimated_image_distance_px,
            "estimated_wall_margin": 0.0,
            "estimated_wall_margin_fraction": 0.0,
            "route_estimator_confidence": 0.0,
            "estimated_magnet_wall_pull": 0.0,
            "estimated_wall_side_risk": 0.0,
            "estimated_wall_normal_3d": [0.0, 0.0, 0.0],
            "estimated_route_tangent_3d": [0.0, 0.0, 0.0],
            "estimated_tip_pixel_side": estimated_tip_pixel_side,
        },
        "reference_action": {
            "piper_step_command": int(piper_label),
            "piper_command_label": PIPER_LABELS[int(piper_label)],
        },
    }


def estimate_visual_contact(
    image_path: Path,
    *,
    contact_threshold_px: float,
    confidence_when_visible: float,
    red_min_area: float,
    red_max_area: float,
) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "method": "branchs_hsv_red_tip_to_edge_distance",
            "failure": "image_read_failed",
            "estimated_contact_flag": 0.0,
            "contact_estimator_confidence": 0.0,
            "estimated_image_distance_px": 0.0,
            "estimated_tip_pixel_side": None,
        }

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Real branchs images render the guidewire head as dark red/orange rather
    # than the saturated red used in MuJoCo.
    red_mask = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([25, 255, 220]))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(red_min_area) or area > float(red_max_area):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if cy < 100 or cy > image.shape[0] - 30:
            continue
        candidates.append((area, cx, cy, contour))
    if not candidates:
        return {
            "method": "branchs_hsv_red_tip_to_edge_distance",
            "failure": "red_tip_not_detected",
            "estimated_contact_flag": 0.0,
            "contact_estimator_confidence": 0.0,
            "estimated_image_distance_px": 0.0,
            "estimated_tip_pixel_side": None,
        }

    # Prefer the largest valid red/orange blob. This usually selects the guidewire
    # head over small colored support markers in branchs.
    _area, cx, cy, contour = max(candidates, key=lambda item: item[0])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # Exclude the detected red tip itself so its contour is not counted as a wall.
    red_dilated = cv2.dilate(red_mask, np.ones((9, 9), np.uint8), iterations=1)
    edges[red_dilated > 0] = 0
    # Distance transform expects zero pixels as targets.
    edge_targets = np.where(edges > 0, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(edge_targets, cv2.DIST_L2, 3)
    pts = contour.reshape(-1, 2)
    distances = [float(distance[int(np.clip(y, 0, distance.shape[0] - 1)), int(np.clip(x, 0, distance.shape[1] - 1))]) for x, y in pts]
    min_distance = float(min(distances)) if distances else float(distance[int(cy), int(cx)])
    return {
        "method": "branchs_hsv_red_tip_to_edge_distance",
        "estimated_contact_flag": float(min_distance <= float(contact_threshold_px)),
        "contact_estimator_confidence": float(confidence_when_visible),
        "estimated_image_distance_px": min_distance,
        "estimated_tip_pixel_side": [float(cx), float(cy)],
        "red_tip_area_px": float(_area),
        "contact_threshold_px": float(contact_threshold_px),
        "note": "Uses image edge distance as a temporary vessel-wall proxy because the original frunet mask is absent.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert senior branchs data to real shadow-mode JSONL input.")
    parser.add_argument("--branch-root", default="branchs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--branch1-task", choices=["left", "right"], default="left")
    parser.add_argument("--branch2-task", choices=["left", "right"], default="right")
    parser.add_argument("--default-task", choices=["left", "right"], default="left")
    parser.add_argument("--max-frames-per-path", type=int, default=0)
    parser.add_argument("--visual-contact-estimator", action="store_true")
    parser.add_argument("--contact-threshold-px", type=float, default=5.0)
    parser.add_argument("--contact-confidence", type=float, default=0.45)
    parser.add_argument("--red-min-area", type=float, default=5.0)
    parser.add_argument("--red-max-area", type=float, default=500.0)
    args = parser.parse_args()

    branch_root = Path(args.branch_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"records": 0, "branches": {}, "skipped_missing": 0}
    with out_path.open("w", encoding="utf-8") as f:
        for branch_dir in sorted(branch_root.glob("branch*"), key=lambda p: p.name):
            if not branch_dir.is_dir():
                continue
            task = branch_task(branch_dir.name, args)
            pose_files = {numeric_suffix(p, "pose"): p for p in (branch_dir / "path").glob("pose*.txt")}
            label_files = {numeric_suffix(p, "label"): p for p in (branch_dir / "piper").glob("label*.txt")}
            image_dirs = {numeric_suffix(p, "path"): p for p in (branch_dir / "image1").glob("path*") if p.is_dir()}
            branch_count = 0
            for path_idx in sorted(set(pose_files) & set(label_files) & set(image_dirs)):
                poses = read_pose_file(pose_files[path_idx])
                labels = read_label_file(label_files[path_idx])
                images = sorted(image_dirs[path_idx].glob("*.png"), key=lambda p: numeric_suffix(p, ""))
                n = min(len(poses), len(labels), len(images))
                if args.max_frames_per_path and args.max_frames_per_path > 0:
                    n = min(n, int(args.max_frames_per_path))
                piper_step = 0
                for i in range(n):
                    frame_number = numeric_suffix(images[i], "")
                    visual_contact = None
                    if args.visual_contact_estimator:
                        visual_contact = estimate_visual_contact(
                            images[i],
                            contact_threshold_px=float(args.contact_threshold_px),
                            confidence_when_visible=float(args.contact_confidence),
                            red_min_area=float(args.red_min_area),
                            red_max_area=float(args.red_max_area),
                        )
                    record = build_record(
                        branch_name=branch_dir.name,
                        path_name=f"path{path_idx}",
                        frame_number=frame_number,
                        image_path=images[i],
                        pose=poses[i],
                        piper_step_before_action=piper_step,
                        piper_label=labels[i],
                        task=task,
                        branch_root=branch_root,
                        visual_contact=visual_contact,
                    )
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["records"] += 1
                    branch_count += 1
                    if labels[i] > 0:
                        piper_step += 1
            counts["branches"][branch_dir.name] = branch_count
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"wrote branchs shadow input to {out_path}")


if __name__ == "__main__":
    main()
