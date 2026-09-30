"""Local-network live camera page. GET-only; no robot or feeder connections."""
from __future__ import annotations

import argparse
from datetime import datetime
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>双相机实时预览</title><style>
body{margin:24px;background:#142232;color:#edf4fa;font:16px sans-serif}
h1{font-size:25px}main{display:grid;grid-template-columns:1fr 1fr;gap:20px}
section{background:#203449;padding:16px;border-radius:12px}h2{font-size:19px;margin:0 0 12px}
img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#101822}
a{color:#9bd5ff}.state{min-height:24px;margin-top:12px}.bad{color:#ffada7}
@media(max-width:900px){main{grid-template-columns:1fr}}</style>
<h1>双相机实时预览</h1><p>调整相机，使血管主体、分叉与导丝入口清晰入镜。相机名称需要结合实际视角确认。</p>
<main><section><h2>相机 A · Side · __SIDE__</h2><img id="side" alt="相机 A 实时图像">
<div id="side-state" class="state">正在获取画面…</div><a href="/side.jpg" target="_blank">打开最新原图</a></section>
<section><h2>相机 B · Top · __TOP__</h2><img id="top" alt="相机 B 实时图像">
<div id="top-state" class="state">正在获取画面…</div><a href="/top.jpg" target="_blank">打开最新原图</a></section></main>
<p>仅查看相机，不控制机械臂或递丝装置。页面自动刷新。</p>
<script>
const urls={};
async function refresh(role){
 const label=document.getElementById(role+'-state'), img=document.getElementById(role);
 try{
  const response=await fetch('/'+role+'.jpg?t='+Date.now(),{cache:'no-store',signal:AbortSignal.timeout(4000)});
  if(!response.ok) throw new Error(await response.text());
  const stamp=Number(response.headers.get('X-Capture-Timestamp'));
  const blob=await response.blob(), previous=urls[role];
  urls[role]=URL.createObjectURL(blob); img.src=urls[role];
  if(previous) URL.revokeObjectURL(previous);
  label.textContent='拍摄时间：'+new Date(stamp*1000).toLocaleTimeString('zh-CN',{hour12:false});
  label.className='state'; img.style.opacity=1;
 }catch(error){label.textContent='画面未更新：'+error.message;label.className='state bad';img.style.opacity=.35;}
 setTimeout(()=>refresh(role),400);
}
refresh('side');refresh('top');
</script></html>"""


class PreviewCamera:
    """Own the color pipeline in one thread and retry after USB disconnects."""

    def __init__(self, serial):
        self.serial = serial
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.frame_count = 0
        self.error = None
        self.reconnect_count = 0
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        import pyrealsense2 as rs
        while not self.stop.is_set():
            pipeline = rs.pipeline()
            started = False
            try:
                config = rs.config()
                config.enable_device(self.serial)
                config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
                pipeline.start(config)
                started = True
                warmup = 0
                while not self.stop.is_set():
                    color = pipeline.wait_for_frames(1500).get_color_frame()
                    if not color:
                        continue
                    warmup += 1
                    if warmup < 45:
                        continue
                    frame = (np.asanyarray(color.get_data()).copy(), time.time())
                    with self.lock:
                        self.frame = frame
                        self.frame_count += 1
                        self.error = None
            except Exception as exc:
                with self.lock:
                    self.error = exc
                    self.frame = None
                    self.reconnect_count += 1
            finally:
                if started:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
            self.stop.wait(2)

    def read(self):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with self.lock:
                if self.error is not None:
                    raise RuntimeError(f"相机 {self.serial} 正在重新连接：{self.error}")
                if self.frame is not None:
                    image, timestamp = self.frame
                    return image.copy(), timestamp
            time.sleep(.02)
        raise TimeoutError(f"相机 {self.serial} 暂无画面")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError(f"camera {self.serial} cleanup timed out")


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"pid": os.getpid(), "status": "starting", "hardware_executed": False,
              "motion_commands": 0, "feeder_packets": 0, "cleanup_errors": [],
              "started_at": datetime.now().astimezone().isoformat(),
              "url": f"http://{args.host}:{args.port}/",
              "side_serial": args.side_serial, "top_serial": args.top_serial,
              "profile": "1920x1080 BGR8 15fps; 45-frame warmup; independent host timestamps"}
    def save():
        (args.out / "server.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    cameras = {}
    server = None
    page = PAGE.replace("__SIDE__", html.escape(args.side_serial)).replace("__TOP__", html.escape(args.top_serial)).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def send_body(self, status, mime, body, timestamp=None):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            if timestamp is not None:
                self.send_header("X-Capture-Timestamp", str(timestamp))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                path = urlsplit(self.path).path
                if path == "/":
                    self.send_body(200, "text/html; charset=utf-8", page)
                elif path == "/status.json":
                    status = {}
                    for role, camera in cameras.items():
                        with camera.lock:
                            stamp = camera.frame[1] if camera.frame else None
                            status[role] = {"serial": camera.serial, "frame_count": camera.frame_count,
                                            "reconnect_count": camera.reconnect_count,
                                            "host_timestamp": stamp, "age_s": time.time()-stamp if stamp else None,
                                            "error": str(camera.error) if camera.error else None}
                    self.send_body(200, "application/json", json.dumps(status).encode())
                elif path in ("/side.jpg", "/top.jpg"):
                    image, timestamp = cameras[path[1:-4]].read()
                    if not 0 <= time.time()-timestamp <= 2:
                        raise RuntimeError("相机图像已过期，请检查连接")
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise RuntimeError("图像编码失败")
                    self.send_body(200, "image/jpeg", encoded.tobytes(), timestamp)
                else:
                    self.send_body(404, "text/plain", b"Not found")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                try:
                    self.send_body(503, "text/plain; charset=utf-8", str(exc).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        for role, serial in (("side", args.side_serial), ("top", args.top_serial)):
            cameras[role] = PreviewCamera(serial)
        for role, camera in cameras.items():
            image, timestamp = camera.read()
            cv2.imwrite(str(args.out / f"{role}_startup.png"), image)
        report["status"] = "running"
        save()
        print(json.dumps(report, ensure_ascii=False), flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        report["status"] = "stopped"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if server is not None:
            server.server_close()
        for role, camera in cameras.items():
            try:
                camera.close()
            except Exception as exc:
                report["cleanup_errors"].append(f"{role}: {exc}")
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.5.11")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--side-serial", default="317222071938")
    parser.add_argument("--top-serial", default="317222072584")
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output" / ("camera_live_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    if args.side_serial == args.top_serial:
        parser.error("two distinct serial numbers are required")
    run(args)


if __name__ == "__main__":
    main()
