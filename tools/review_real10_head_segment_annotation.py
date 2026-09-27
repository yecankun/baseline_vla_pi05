"""Independent six-window head-segment annotation; no model or old-label writes."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import threading
from urllib.parse import parse_qs, urlsplit


SCHEMA = "real10_head_segment_annotation_v1"
CASES = (4, 39, 62, 65, 77, 100)
VIEWS = ("side", "top")
STATES = ("unreviewed", "visible", "ambiguous", "not_visible")
COVERAGE = ("complete", "partial", "uncertain")
HTML = Path(__file__).with_name("real10_head_segment_annotation.html")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def prepare(args):
    sources = {"source_manifest.json": args.points / "manifest.json",
               "source_points.jsonl": args.points / "annotations_v2.jsonl",
               "source_responses.jsonl": args.responses}
    before = {name: path.read_bytes() for name, path in sources.items()}
    original = json.loads(before["source_manifest.json"])
    cases = deepcopy(original["cases"])
    if tuple(c["ui_index"] for c in cases) != CASES:
        raise ValueError("only the original six windows are authorized")
    points = {r["case_id"]: r for r in rows(args.points / "annotations_v2.jsonl")}
    responses = {r["ui_index"]: r for r in rows(args.responses)}
    for case in cases:
        response = responses[case["ui_index"]]
        if response["source_episode"] != case["source_episode"]:
            raise ValueError("response snapshot episode mismatch")
        if len(case["keyframe_indices"]) != 3 or any(
                set(points[case["id"]]["views"][v]["frames"]) != {str(i) for i in case["keyframe_indices"]}
                for v in VIEWS):
            raise ValueError("preserve the existing three keyframes per view")
        case["read_only_response"] = deepcopy(response["human_annotation"])
        case["read_only_points"] = deepcopy(points[case["id"]])
    manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases, "original_image_wh": [1920, 1080],
        "display_roi_xyxy": original["display_roi_xyxy"], "slots": 36,
        "source_files": {name: path.as_posix() for name, path in sources.items()},
        "annotation_file": "head_annotations.jsonl", "is_ui_fixture": args.fixture,
        "scope": "visible head segment only; box or disconnected short polylines; no occlusion completion",
        "coordinate_space": "original_image_pixels", "head_length_mm_user_description": 25,
        "calibrated_pixel_length": None, "old_labels_read_only": True,
        "point_identity": "polyline vertices/endpoints are not tip or same-material-point labels",
        "missing": "unreviewed/ambiguous/not_visible have null geometry and coverage; views independent",
        "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False}
    args.pack.mkdir(parents=True, exist_ok=False)
    for name, data in before.items():
        (args.pack / name).write_bytes(data)
    write_json(args.pack / "manifest.json", manifest)
    shutil.copy2(__file__, args.pack / "entrypoint_snapshot.py")
    shutil.copy2(HTML, args.pack / "interface_snapshot.html")
    if any(path.read_bytes() != before[name] for name, path in sources.items()):
        raise ValueError("source changed during preparation; inspect before use")
    print(json.dumps({"pack": str(args.pack), "slots": 36, "new_annotations": 0,
                      "is_ui_fixture": args.fixture, "source_bytes_unchanged": True}))


def blank(case):
    return {"annotation_schema": SCHEMA, "case_id": case["id"], "revision": 0,
        "reviewer": "", "notes": "", "views": {v: {"frames": {
            str(i): {"status": "unreviewed", "coverage": None, "geometry": None, "notes": ""}
            for i in case["keyframe_indices"]}} for v in VIEWS}}


def validate_xy(xy):
    if (not isinstance(xy, list) or len(xy) != 2 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in xy) or
            not 0 <= xy[0] < 1920 or not 0 <= xy[1] < 1080):
        raise ValueError("坐标必须位于原图 1920×1080 内")
    return deepcopy(xy)


def validate_frame(frame):
    status, coverage, geometry = (frame.get(k) for k in ("status", "coverage", "geometry"))
    if status not in STATES:
        raise ValueError("未知头段可见状态")
    result = {"status": status, "coverage": coverage, "geometry": None,
              "notes": str(frame.get("notes", ""))[:2000]}
    if status != "visible":
        if geometry is not None or coverage is not None:
            raise ValueError("未标/不确定/不可见须保留 null，不得补零或补画")
        return result
    if coverage not in COVERAGE or not isinstance(geometry, dict):
        raise ValueError("已画头段须选择完整性，并提供粗框或短折线")
    if geometry.get("kind") == "bbox":
        box = geometry.get("xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("粗框须有四个原图坐标")
        a, b = validate_xy(box[:2]), validate_xy(box[2:])
        if b[0]-a[0] < 1 or b[1]-a[1] < 1:
            raise ValueError("粗框宽高须至少 1 原图像素")
        result["geometry"] = {"kind": "bbox", "xyxy": a+b}
    elif geometry.get("kind") == "polyline":
        segments = geometry.get("segments")
        if not isinstance(segments, list) or not 1 <= len(segments) <= 12:
            raise ValueError("折线须含 1–12 段可见部分；不要跨遮挡连线")
        checked = []
        for segment in segments:
            if not isinstance(segment, list) or not 2 <= len(segment) <= 64:
                raise ValueError("每段短折线至少 2 点；完成或取消正在绘制的线段")
            points = [validate_xy(p) for p in segment]
            if sum(math.dist(a, b) for a, b in zip(points, points[1:])) < 1:
                raise ValueError("折线不能退化为一个点")
            checked.append(points)
        result["geometry"] = {"kind": "polyline", "segments": checked}
    else:
        raise ValueError("仅接受粗框或短折线")
    return result


def progress(annotation):
    counts = Counter(f["status"] for v in annotation["views"].values() for f in v["frames"].values())
    return {"reviewed": 6-counts["unreviewed"], "visible": counts["visible"], "slots": 6}


class HeadStore:
    def __init__(self, pack, raw_root):
        self.pack, self.raw_root = Path(pack).resolve(), Path(raw_root).resolve()
        self.manifest = read_json(self.pack / "manifest.json")
        if self.manifest["schema"] != SCHEMA or tuple(c["ui_index"] for c in self.manifest["cases"]) != CASES:
            raise ValueError("wrong head annotation pack")
        self.cases = {c["id"]: c for c in self.manifest["cases"]}
        self.annotations = {key: blank(c) for key, c in self.cases.items()}
        self.path = self.pack / "head_annotations.jsonl"
        self.lock = threading.Lock()
        self.revisions = 0
        if self.path.exists():
            for row in rows(self.path):
                self.validate(row)
                self.annotations[row["case_id"]] = row
                self.revisions += 1

    def summary(self):
        counts = Counter(f["status"] for a in self.annotations.values()
                         for v in a["views"].values() for f in v["frames"].values())
        return {"slots": 36, "reviewed": 36-counts["unreviewed"], "status_counts": dict(counts),
                "saved_revisions": self.revisions, "is_ui_fixture": self.manifest["is_ui_fixture"]}

    def index(self):
        return {"schema": SCHEMA, "manifest": self.manifest, "annotations": self.annotations,
                "summary": self.summary(), "progress": {k: progress(a) for k, a in self.annotations.items()}}

    def image_path(self, case_id, view, index):
        if view not in VIEWS or not 0 <= index < len(self.cases[case_id]["frames"][view]):
            raise ValueError("unknown view/frame")
        path = (self.raw_root / self.cases[case_id]["frames"][view][index]["path"]).resolve()
        if not path.is_relative_to(self.raw_root):
            raise ValueError("image outside raw capture directory")
        return path

    def validate(self, payload):
        if payload.get("annotation_schema") != SCHEMA:
            raise ValueError("标注版本不匹配，请保留草稿后刷新")
        result = blank(self.cases[payload["case_id"]])
        reviewer = str(payload.get("reviewer", "")).strip()
        if not reviewer or len(reviewer) > 120:
            raise ValueError("请填写标注人姓名或缩写")
        result.update(reviewer=reviewer, notes=str(payload.get("notes", ""))[:4000])
        if set(payload["views"]) != set(VIEWS):
            raise ValueError("保留 Side/Top 独立状态；未知允许留空")
        for v in VIEWS:
            source = payload["views"][v]["frames"]
            if set(source) != set(result["views"][v]["frames"]):
                raise ValueError("仅允许原来的首/中/末三帧")
            result["views"][v]["frames"] = {key: validate_frame(f) for key, f in source.items()}
        return result

    def save(self, payload):
        result = self.validate(payload)
        with self.lock:
            old = self.annotations[result["case_id"]]
            if payload.get("revision") != old["revision"]:
                raise ValueError("另一页面已有新保存；请保留草稿后刷新，不覆盖它")
            case = self.cases[result["case_id"]]
            result.update(revision=old["revision"]+1, saved_at_utc=datetime.now(timezone.utc).isoformat(),
                source_episode=case["source_episode"], coordinate_space="original_image_pixels",
                annotation_source="ui_fixture_not_human" if self.manifest["is_ui_fixture"] else "human_head_segment_review",
                policy_input_allowed=False, deployable=False, formal_data_allowed=False)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
            self.annotations[result["case_id"]] = result
            self.revisions += 1
        return {"annotation": result, "summary": self.summary(), "progress": progress(result)}


def make_server(store, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, content, content_type="application/json; charset=utf-8", status=200):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url, q = urlsplit(self.path), parse_qs(urlsplit(self.path).query)
            try:
                if url.path == "/":
                    self.send(HTML.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/index":
                    self.send(store.index())
                elif url.path == "/image":
                    self.send(store.image_path(q["case"][0], q["view"][0], int(q["frame"][0])).read_bytes(), "image/png")
                else:
                    self.send({"error": "not found"}, status=404)
            except (ValueError, KeyError, OSError, IndexError) as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if self.headers.get("Origin") not in (None, *origins):
                return self.send({"error": "cross-origin writes forbidden"}, status=403)
            try:
                if self.path != "/api/annotation":
                    return self.send({"error": "not found"}, status=404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 262144:
                    raise ValueError("invalid request size")
                self.send(store.save(json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--fixture", action="store_true", help="prepare a separate UI-test pack, never human labels")
    p.add_argument("--points", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    p.add_argument("--responses", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1/source/annotation_snapshot.jsonl"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_head_segment_annotation_v1"))
    p.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    p.add_argument("--port", type=int, default=8795)
    args = p.parse_args()
    if args.fixture and "fixture" not in args.pack.name:
        p.error("fixture tests require an explicitly separate pack name containing 'fixture'")
    if args.prepare:
        return prepare(args)
    store = HeadStore(args.pack, args.raw_root)
    if args.summary:
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        return
    server = make_server(store, args.port)
    print(f"Head segment UI: http://127.0.0.1:{server.server_port}\nNew labels only: {store.path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
