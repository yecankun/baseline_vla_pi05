"""Serve only a fixed camera-photo/video allowlist; never expose diagnostic files."""
import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", required=True, type=Path)
    parser.add_argument("--host", default="192.168.5.11")
    parser.add_argument("--port", default=8766, type=int)
    parser.add_argument("--title", default="机械臂相机录像与原图")
    parser.add_argument("--description", default="本轮录像及运动前后相机原图，供核对现场观察。")
    parser.add_argument("--before-label", default="运动前")
    parser.add_argument("--after-label", default="运动后")
    parser.add_argument("--side-crop-x", type=int, default=300)
    parser.add_argument("--side-crop-y", type=int, default=85)
    parser.add_argument("--review-image", type=Path, help="optional fixed PNG comparison figure")
    args = parser.parse_args()
    root = args.arm_dir.resolve()
    # Exact routes only: no directory listings, path joins from requests, JSON,
    # controller logs, calibration records, control endpoints or uploads.
    routes = {"/motion.mp4": (root / "motion.mp4", "video/mp4")}
    if args.review_image:
        routes["/review.png"] = (args.review_image.resolve(), "image/png")
    for role in ("side", "top"):
        for stage in ("before", "after"):
            name = f"{role}_{stage}.jpg"
            routes["/" + name] = (root / name, "image/jpeg")
    for path, _ in routes.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or symbolic media path: {path}")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>机械臂相机录像与原图</title><style>body{max-width:1200px;margin:24px auto;padding:0 16px;font:17px/1.7 sans-serif;background:#f4f6f8}video,img{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}section{background:white;padding:20px;margin:18px 0}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>机械臂相机录像与原图</h1><p>本轮录像及运动前后相机原图，供核对现场观察。</p><p><a href="http://192.168.5.11:8765/">实时双相机</a></p><video controls preload="metadata" src="/motion.mp4"></video><section><h2>Side 磁铁局部切换</h2><select id="phase"><option value="before">运动前</option><option value="after">运动后</option></select><div style="position:relative;width:390px;height:300px;max-width:100%;overflow:hidden"><img id="zoom" src="/side_before.jpg" style="position:absolute;width:1920px;max-width:none;left:-300px;top:-85px"></div></section>'''
    for role in ("side", "top"):
        page += f'<section><h2>{role} 原图</h2><div class="pair"><div>运动前<img src="/{role}_before.jpg"></div><div>运动后<img src="/{role}_after.jpg"></div></div></section>'
    if args.review_image:
        page += '<section><h2>导丝响应对比</h2><a href="/review.png"><img src="/review.png"></a></section>'
    page += "<script>document.getElementById('phase').onchange=e=>document.getElementById('zoom').src='/side_'+e.target.value+'.jpg';</script></html>"
    page = page.replace("机械臂相机录像与原图", html.escape(args.title))
    page = page.replace("本轮录像及运动前后相机原图，供核对现场观察。", html.escape(args.description))
    page = page.replace("运动前", html.escape(args.before_label))
    page = page.replace("运动后", html.escape(args.after_label))
    page = page.replace("left:-300px;top:-85px", f"left:{-args.side_crop_x}px;top:{-args.side_crop_y}px")
    page = page.encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body, mime = page, "text/html; charset=utf-8"
            elif self.path in routes:
                path, mime = routes[self.path]
                body = path.read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
