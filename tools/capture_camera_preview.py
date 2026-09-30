"""Save fresh dual-RealSense color frames; no robot, feeder, or model imports."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1]


def capture(args):
    args.out.mkdir(parents=True, exist_ok=False)
    pipelines = []
    report = {"source": "live_camera_capture", "status": "starting",
              "hardware_executed": False, "motion_commands": 0, "feeder_packets": 0,
              "visual_status": "not_viewed", "warmup_frames": args.warmup_frames,
              "profile": "1920x1080 BGR8 15fps color only", "cameras": {},
              "timing": "host receive timestamps; not exposure-synchronized", "cleanup_errors": []}
    images = {}
    try:
        available = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        for role, serial in (("side", args.side_serial), ("top", args.top_serial)):
            if serial not in available:
                raise RuntimeError(f"camera {serial} is absent; available={sorted(available)}")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
            pipeline.start(config)
            pipelines.append((role, serial, pipeline))
        for _ in range(args.warmup_frames):
            for role, serial, pipeline in pipelines:
                color = pipeline.wait_for_frames(5000).get_color_frame()
                if not color:
                    raise RuntimeError(f"{serial}: missing color frame")
        for role, serial, pipeline in pipelines:
            color = pipeline.wait_for_frames(5000).get_color_frame()
            if not color:
                raise RuntimeError(f"{serial}: missing final color frame")
            timestamp = time.time()
            image = np.asanyarray(color.get_data()).copy()
            if image.shape != (1080, 1920, 3):
                raise RuntimeError(f"{serial}: unexpected shape {image.shape}")
            image_path = args.out / f"{role}_{serial}.png"
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"failed to write {image_path}")
            images[role] = image
            report["cameras"][role] = {"serial": serial, "path": image_path.name,
                "host_timestamp": timestamp, "captured_at": datetime.fromtimestamp(timestamp).astimezone().isoformat(),
                "device_frame_number": color.get_frame_number(), "shape": list(image.shape)}
        font = ImageFont.truetype(str(args.font), 22)
        sheet = Image.new("RGB", (1920, 620), "#142232")
        draw = ImageDraw.Draw(sheet)
        for index, role in enumerate(("side", "top")):
            camera = report["cameras"][role]
            draw.text((index*960+14, 9), f"{role.upper()} · {camera['serial']}", font=font, fill="white")
            rgb = cv2.cvtColor(cv2.resize(images[role], (960, 540)), cv2.COLOR_BGR2RGB)
            sheet.paste(Image.fromarray(rgb), (index*960, 45))
            draw.text((index*960+14, 588), camera["captured_at"][:19].replace("T", " "), font=font, fill="white")
        sheet.save(args.out / "preview.png")
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for role, serial, pipeline in reversed(pipelines):
            try:
                pipeline.stop()
            except Exception as exc:
                report["cleanup_errors"].append(f"{serial}: {exc}")
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side-serial", default="317222071938")
    parser.add_argument("--top-serial", default="317222072584")
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output" / ("camera_preview_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    if args.side_serial == args.top_serial or args.warmup_frames < 1:
        parser.error("two distinct camera serials and positive warmup-frames are required")
    capture(args)


if __name__ == "__main__":
    main()
