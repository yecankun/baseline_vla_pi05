"""Review manual tip/target points against passive captures; no hardware API."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from probe_real10_seeded_wire_tracking import PARAMETERS, track


def run(args):
    annotation = json.loads(args.annotations.read_text().splitlines()[-1])
    capture = json.loads((args.capture / "capture.json").read_text())
    if capture["robot_motion_commands"] != 0 or capture["feeder_packets"] != 0:
        raise ValueError("This review expects a passive sequence")
    args.out.mkdir(parents=True, exist_ok=False)
    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    report = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "annotation_source": str(args.annotations.resolve()),
        "annotation_revision": annotation["revision"],
        "capture_directory": str(args.capture.resolve()),
        "parameters": PARAMETERS, "views": {},
        "robot_motion_commands": 0, "feeder_packets": 0,
        "valid_for_hardware_control": False,
        "scope": "Offline local image correspondence at a manually identified tip, not semantic tip detection",
        "limitations": [
            "An unobserved interval separates the annotated snapshot and passive sequence.",
            "Static local appearance may belong to the vessel or reflection rather than the moving wire.",
            "No moving-wire tracking or current camera-to-robot registration is established.",
            "Pixel separation is not vessel arc length, millimetres or a planned straight-line path.",
        ],
        "visual_status": "not_viewed",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    for row, view in enumerate(("side", "top")):
        points = annotation["views"][view]
        if any(points[name]["status"] != "visible" for name in ("tip", "target")):
            raise ValueError(f"Missing visible tip/target in {view}")
        seed, target = (np.asarray(points[name]["xy"], float) for name in ("tip", "target"))
        paths = [Path(annotation["capture_directory"]) / f"{view}_latest.jpg"]
        paths += [args.capture / frame["views"][view]["file"] for frame in capture["frames"]]
        images = [cv2.imread(str(path)) for path in paths]
        if any(im is None or im.shape[:2] != (1080, 1920) for im in images):
            raise ValueError("Missing or incorrect original image")
        rows = track([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images], seed)
        accepted = [r for r in rows[1:] if r["output_xy"] is not None]
        distances = [float(np.linalg.norm(np.asarray(r["output_xy"])-seed)) for r in accepted]
        report["views"][view] = {
            "manual_tip_xy": seed.tolist(), "manual_target_xy": target.tolist(),
            "target_minus_tip_px": (target-seed).tolist(),
            "straight_pixel_separation": float(np.linalg.norm(target-seed)),
            "passive_frames": len(rows)-1, "emitted_local_points": len(accepted),
            "maximum_local_point_drift_px": max(distances) if distances else None,
            "last_local_point_xy": rows[-1]["output_xy"],
            "first_rejection": rows[-1]["first_rejection"], "frames": rows,
        }
        for col, (label, im) in enumerate((("人工标记原图", images[0]), ("被动采集末帧", images[-1]))):
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            for xy, color, name in ((seed, "cyan", "人工尖端"), (target, "magenta", "目标")):
                ax.scatter(*xy, s=100, marker="+", color=color, linewidths=2)
                ax.annotate(name, xy, xytext=(10, 16), textcoords="offset points", color=color,
                            fontproperties=font, fontsize=12,
                            bbox={"facecolor": "black", "alpha": .65, "edgecolor": "none"})
            if col and rows[-1]["output_xy"] is not None:
                ax.scatter(*rows[-1]["output_xy"], s=110, facecolors="none", edgecolors="yellow")
            # Equal scale within each camera pair; no line implies a valid vessel path.
            lo = np.minimum(seed, target)-[95, 115]
            hi = np.maximum(seed, target)+[95, 95]
            ax.set_xlim(max(0, lo[0]), min(1920, hi[0]))
            ax.set_ylim(min(1080, hi[1]), max(0, lo[1]))
            state = f"局部跟踪 {len(accepted)}/{len(rows)-1}" if col else "青色：尖端；紫色：目标"
            ax.set_title(f"{view.upper()} · {label} · {state}", fontproperties=font)
            ax.set_xlabel("原图横坐标 / 像素", fontproperties=font)
            ax.set_ylabel("原图纵坐标 / 像素", fontproperties=font)
    fig.suptitle("用户标记复核：仅像素对应，尚未验证运动中的导丝跟踪或机器人坐标", fontproperties=font, fontsize=16)
    fig.savefig(args.out / "marked_points_review.png", dpi=130)
    plt.close(fig)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({v: {k: val for k, val in data.items() if k != "frames"}
                      for v, data in report["views"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
