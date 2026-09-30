"""Collect manual tip/target locations on two fixed photos; no hardware interfaces."""
from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlsplit


def validate(payload):
    views = {}
    for role in ("side", "top"):
        supplied = payload.get("views", {}).get(role, {})
        points = {}
        for name in ("tip", "target"):
            point = supplied.get(name, {"status": "unreviewed", "xy": None})
            status, xy = point.get("status"), point.get("xy")
            if status not in ("visible", "not_visible", "unreviewed"):
                raise ValueError("位置状态无效")
            if status == "visible":
                if (not isinstance(xy, list) or len(xy) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
                        or not 0 <= xy[0] < 1920 or not 0 <= xy[1] < 1080):
                    raise ValueError("请在图像范围内点击位置")
            elif xy is not None:
                raise ValueError("不可见或未标记的位置不能包含坐标")
            points[name] = {"status": status, "xy": xy}
        views[role] = points
    if all(p["status"] == "unreviewed" for view in views.values() for p in view.values()):
        raise ValueError("请标记位置，或注明看不清的位置")
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--host", default="192.168.5.11")
    parser.add_argument("--port", default=8767, type=int)
    parser.add_argument("--resume", action="store_true", help="reuse the same capture, URL and saved annotations")
    parser.add_argument("--review-image", type=Path, help="optional fixed review PNG; no directory serving")
    parser.add_argument("--reference-targets", type=Path,
                        help="reuse previous manually selected targets as an explicit reference; tips remain unreviewed")
    args = parser.parse_args()
    photos = {role: (args.capture_dir / f"{role}_latest.jpg").read_bytes() for role in ("side", "top")}
    page_source = Path(__file__).with_name("left_target_annotation.html").read_text()
    reference_views = None
    if args.reference_targets:
        reference = json.loads(args.reference_targets.read_text().splitlines()[-1])
        reference_views = validate(reference)
        for role in ("side", "top"):
            if reference_views[role]["target"]["status"] != "visible":
                raise ValueError("Reference targets must be visible in both views")
            reference_views[role]["tip"] = {"status": "unreviewed", "xy": None}
        page_source = page_source.replace(
            "先看 Side 左侧相机：选择“导丝尖端”并点击尖端，再选择“左端目标出口”并点击目标位置。Top 可补充同一点的另一视角。",
            "紫色目标沿用前次人工标记。请只补标 Side、Top 当前的导丝尖端并保存；若目标位置已变，可重新标目标。看不清的尖端请选“看不清”。")
    if args.resume:
        prior = json.loads((args.out / "server.json").read_text())
        if prior["capture_directory"] != str(args.capture_dir.resolve()):
            raise ValueError("Cannot reuse annotations on a different capture")
        token = urlsplit(prior["url"]).path.strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_-]{32}", token):
            raise ValueError("Invalid saved annotation URL")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        token = secrets.token_urlsafe(24)
    prefix = "/" + token
    page = page_source.replace("__PREFIX__", prefix).encode()
    lock = threading.Lock()
    record_path = args.out / "annotations.jsonl"
    records = [json.loads(line) for line in record_path.read_text().splitlines()] if args.resume and record_path.exists() else []
    review = args.review_image.read_bytes() if args.review_image else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, status, mime, body):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route in (prefix, prefix + "/"):
                self.send_body(200, "text/html; charset=utf-8", page)
            elif route in (prefix + "/side.jpg", prefix + "/top.jpg"):
                role = route.rsplit("/", 1)[1].split(".")[0]
                self.send_body(200, "image/jpeg", photos[role])
            elif route == prefix + "/annotation":
                with lock:
                    saved = {"views": records[-1]["views"], "revision": records[-1]["revision"]} if records else None
                    if saved is None and reference_views is not None:
                        saved = {"views": reference_views, "revision": 0, "initial_reference": True}
                self.send_body(200, "application/json", json.dumps(saved).encode())
            elif route == prefix + "/review.png" and review is not None:
                self.send_body(200, "image/png", review)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlsplit(self.path).path != prefix + "/annotation":
                self.send_error(404)
                return
            if self.headers.get("X-Annotation-Key") != token:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("提交长度无效")
                views = validate(json.loads(self.rfile.read(length)))
                if reference_views is not None and all(views[v]["tip"]["status"] == "unreviewed" for v in ("side", "top")):
                    raise ValueError("请标记当前尖端，或注明看不清；仅沿用目标不能作为新尖端定位")
                with lock:
                    record = {"schema": "current_scene_tip_target_annotation_v1",
                              "recorded_at": datetime.now().astimezone().isoformat(),
                              "revision": len(records) + 1, "source": "manual_browser_annotation",
                              "capture_directory": str(args.capture_dir.resolve()),
                              "coordinates": "original_1920x1080_pixels_top_left_origin",
                              "views": views, "hardware_action": False,
                              "robot_coordinate_registration": None,
                              "target_reference_source": str(args.reference_targets.resolve()) if args.reference_targets else None}
                    with (args.out / "annotations.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                body = json.dumps({"saved": True, "revision": record["revision"],
                                   "message": "标记已保存，设备没有动作。"}, ensure_ascii=False).encode()
                self.send_body(200, "application/json", body)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                self.send_body(400, "text/plain; charset=utf-8", str(exc).encode())

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    (args.out / "server.json").write_text(json.dumps({
        "url": f"http://{args.host}:{args.port}{prefix}/",
        "scope": "two fixed camera images and user-provided pixel annotations only",
        "capture_directory": str(args.capture_dir.resolve()),
        "hardware_interface": False, "pid": os.getpid(),
        "review_image": str(args.review_image.resolve()) if args.review_image else None,
        "target_reference_source": str(args.reference_targets.resolve()) if args.reference_targets else None,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"Annotation page ready on port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
