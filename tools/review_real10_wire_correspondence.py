"""Loopback-only sparse guidewire/vessel correspondence pilot for six Real10 cases.

Read original images and frozen case identities; append only a NEW diagnostic
annotation file. No model, optical flow, old-label update, or hardware operation.
"""
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

from prepare_real10_event_windows import read_json, write_json


LEGACY_SCHEMA = "real10_sparse_wire_correspondence_v1"
SCHEMA = "real10_sparse_wire_correspondence_v2"
REFERENCE_SCHEMA = "real10_static_vessel_four_point_v1"
CASES = (4, 39, 62, 65, 77, 100)
VIEWS = ("side", "top")
POINTS = ("wire",)
REFERENCE_POINTS = ("p0", "p1", "pl", "pr")
STATES = ("unreviewed", "visible", "ambiguous", "not_visible")
FEATURES = (None, "tip", "distinct_shaft_landmark", "untrackable")


def prepare(args):
    if args.pack.exists():
        raise FileExistsError("preserve prior pilot/annotations; choose another --pack only for a new pilot")
    source = read_json(args.audit / "selection.json")
    chosen = [r for r in source if r["source_kind"] == "human_window" and r["ui_index"] in CASES]
    if tuple(r["ui_index"] for r in chosen) != CASES:
        raise ValueError("expected the same six preselected human-review cases")
    roi = read_json(args.audit / "protocol.json")["parameters"]["roi_xyxy_original"]
    cases = []
    for row in chosen:
        item = {k: deepcopy(row[k]) for k in ("id", "ui_index", "source_episode", "task", "human_response", "human_notes", "frames")}
        n = len(item["frames"]["side"])
        if n != len(item["frames"]["top"]) or n < 3:
            raise ValueError("paired context sequence needs at least three frames")
        item["keyframe_indices"] = [0, n//2, n-1]
        item["required_indices"] = [0, n-1]
        cases.append(item)
    manifest = {"schema": SCHEMA, "source_audit": args.audit.as_posix(), "cases": cases,
        "original_image_wh": [1920, 1080], "display_roi_xyxy": roi,
        "coordinate_space": "original per-view image pixels, top-left origin; no resize/ROI coordinates stored",
        "required_images": 24, "optional_middle_images": 12,
        "annotation_scope": "human sparse 2D correspondence; no contact, calibrated mm, model input or action relabeling",
        "identity": "same tip or distinctive material landmark within one view/window; shaft silhouette crossings are not material landmarks",
        "reference": "one static four-point vessel reference per view: p0 entry, p1 pre-bifurcation, pl left, pr right",
        "missing": "unreviewed is not missing; ambiguous/not_visible have null xy; never fill zero or infer from the other view",
        "axis_limit": "p0->p1->pl/pr are coarse projected polylines, not calibrated centerlines or insertion-distance labels",
        "annotation_file": "annotations_v2.jsonl", "reference_file": "vessel_reference.jsonl", "old_response_labels_read_only": True,
        "policy_input_allowed": False, "formal_data_allowed": False, "real_system_validated": False}
    args.pack.mkdir(parents=True)
    write_json(args.pack / "manifest.json", manifest)
    shutil.copyfile(__file__, args.pack / "entrypoint_snapshot.py")
    shutil.copyfile(Path(__file__).with_name("real10_wire_correspondence.html"), args.pack / "interface_snapshot.html")
    print(json.dumps({"pack": str(args.pack), "cases": len(cases), "required_images": 24,
                      "optional_middle_images": 12, "annotations_created": False}, ensure_ascii=False))


def blank(case):
    return {"annotation_schema": SCHEMA, "case_id": case["id"], "revision": 0, "reviewer": "", "notes": "",
        "views": {v: {"feature_type": None, "feature_description": "",
                      "correspondence_confirmed": False,
                      "frames": {str(i): {p: {"status": "unreviewed", "xy": None} for p in POINTS}
                                 for i in case["keyframe_indices"]}} for v in VIEWS}}


def frame_reviewed(frame):
    return frame["wire"]["status"] != "unreviewed"


def blank_reference():
    return {"annotation_schema": REFERENCE_SCHEMA, "revision": 0, "reviewer": "",
        "views": {v: {"confirmed": False, "notes": "",
                      "points": {p: {"status": "unreviewed", "xy": None, "source": None}
                                 for p in REFERENCE_POINTS}} for v in VIEWS}}


def validate_point(point):
    status, xy = point.get("status"), point.get("xy")
    if status not in STATES:
        raise ValueError("invalid point visibility")
    if status == "visible":
        if (not isinstance(xy, list) or len(xy) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in xy)
                or not 0 <= xy[0] < 1920 or not 0 <= xy[1] < 1080):
            raise ValueError("可定位点必须有有效的原图像素坐标")
    elif xy is not None:
        raise ValueError("未标注/看不清/不可见必须保留null坐标，不能填零")
    return {"status": status, "xy": deepcopy(xy)}


def reviewer_name(payload):
    reviewer = str(payload.get("reviewer", "")).strip()
    if not reviewer or len(reviewer) > 120:
        raise ValueError("请填写标注人姓名或缩写")
    return reviewer


def progress(case, annotation):
    reviewed, ready = 0, 0
    for view in VIEWS:
        a = annotation["views"][view]
        endpoints = [a["frames"][str(i)] for i in case["required_indices"]]
        reviewed += sum(frame_reviewed(f) for f in endpoints)
        ready += int(a["correspondence_confirmed"] and a["feature_type"] in FEATURES[1:3]
                     and all(f[p]["status"] == "visible" for f in endpoints for p in POINTS))
    return {"required_images_reviewed": reviewed, "required_images": 4, "paired_2d_ready_views": ready}


class CorrespondenceStore:
    def __init__(self, pack, raw_root):
        self.pack, self.raw_root = Path(pack).resolve(), Path(raw_root).resolve()
        self.manifest = read_json(self.pack / "manifest.json")
        if self.manifest["schema"] not in (LEGACY_SCHEMA, SCHEMA):
            raise ValueError("wrong correspondence schema")
        self.cases = {r["id"]: r for r in self.manifest["cases"]}
        self.annotations = {key: blank(case) for key, case in self.cases.items()}
        self.path = self.pack / "annotations_v2.jsonl"
        self.reference_path = self.pack / "vessel_reference.jsonl"
        self.reference = blank_reference()
        self.lock = threading.Lock()
        self.revisions = 0
        self.legacy_imported_cases = 0
        legacy_path = self.pack / "annotations.jsonl"
        if legacy_path.exists():
            # Read-only compatibility: carry W and its human identity, never turn
            # arbitrary old vessel A/B landmarks into the new center-path anchors.
            legacy = {}
            for line in legacy_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["annotation_schema"] != LEGACY_SCHEMA or row["case_id"] not in self.cases:
                    raise ValueError("legacy annotation belongs to a different pilot")
                legacy[row["case_id"]] = row
            for key, old in legacy.items():
                new = self.annotations[key]
                new.update(reviewer=old.get("reviewer", ""), notes=old.get("notes", ""),
                           legacy_source={"file": "annotations.jsonl", "revision": old["revision"]})
                for v in VIEWS:
                    for name in ("feature_type", "feature_description", "correspondence_confirmed"):
                        new["views"][v][name] = deepcopy(old["views"][v][name])
                    for i in new["views"][v]["frames"]:
                        new["views"][v]["frames"][i]["wire"] = deepcopy(old["views"][v]["frames"][i]["wire"])
            self.legacy_imported_cases = len(legacy)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["annotation_schema"] != SCHEMA or row["case_id"] not in self.cases:
                    raise ValueError("annotation belongs to a different pilot")
                self.annotations[row["case_id"]] = row
                self.revisions += 1
        if self.reference_path.exists():
            for line in self.reference_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["annotation_schema"] != REFERENCE_SCHEMA:
                    raise ValueError("wrong static reference schema")
                self.reference = row

    def summary(self):
        counts = [progress(case, self.annotations[key]) for key, case in self.cases.items()]
        statuses = Counter(p["status"] for a in self.annotations.values() for v in a["views"].values()
                           for f in v["frames"].values() for p in f.values())
        return {"schema": SCHEMA, "cases": len(counts), "saved_revisions": self.revisions,
            "required_images_reviewed": sum(c["required_images_reviewed"] for c in counts), "required_images": 24,
            "paired_2d_ready_views": sum(c["paired_2d_ready_views"] for c in counts), "possible_view_pairs": 12,
            "point_statuses_including_optional": dict(statuses), "annotation_path": str(self.path),
            "annotation_file_exists": self.path.exists(), "policy_input_allowed": False,
            "legacy_imported_cases": self.legacy_imported_cases,
            "reference_revision": self.reference["revision"], "reference_path": str(self.reference_path),
            "reference_visible_points": sum(p["status"] == "visible" for v in self.reference["views"].values()
                                            for p in v["points"].values()),
            "reference_ready_views": sum(v["confirmed"] and all(p["status"] == "visible" for p in v["points"].values())
                                         for v in self.reference["views"].values())}

    def index(self):
        return {"schema": SCHEMA, "manifest": self.manifest, "summary": self.summary(),
                "annotations": self.annotations, "reference": self.reference,
                "progress": {key: progress(case, self.annotations[key]) for key, case in self.cases.items()}}

    def image_path(self, case_id, view, index):
        if view not in VIEWS:
            raise ValueError("unknown view")
        refs = self.cases[case_id]["frames"][view]
        if not 0 <= index < len(refs):
            raise ValueError("frame index outside context sequence")
        path = (self.raw_root / refs[index]["path"]).resolve()
        if not path.is_relative_to(self.raw_root):
            raise ValueError("image path leaves original capture directory")
        return path

    def validate(self, payload):
        if payload.get("annotation_schema") != SCHEMA:
            raise ValueError("标注页版本不匹配，请先保留草稿再刷新")
        case = self.cases[payload["case_id"]]
        result = blank(case)
        result.update(reviewer=reviewer_name(payload), notes=str(payload.get("notes", ""))[:4000])
        if set(payload["views"]) != set(VIEWS):
            raise ValueError("both views are required; unknown is allowed")
        for view in VIEWS:
            source, target = payload["views"][view], result["views"][view]
            if source.get("feature_type") not in FEATURES:
                raise ValueError("invalid feature identity")
            target.update(feature_type=source["feature_type"],
                feature_description=str(source.get("feature_description", ""))[:1000],
                correspondence_confirmed=source.get("correspondence_confirmed") is True)
            if set(source["frames"]) != set(target["frames"]):
                raise ValueError("only the fixed anchor/middle/end keyframes can be annotated")
            for key in target["frames"]:
                if set(source["frames"][key]) != set(POINTS):
                    raise ValueError("unexpected point slots")
                for name in POINTS:
                    target["frames"][key][name] = validate_point(source["frames"][key][name])
                f = target["frames"][key]
                if f["wire"]["status"] == "visible":
                    if target["feature_type"] not in FEATURES[1:3]:
                        raise ValueError("请先说明标的是尖端，还是可辨识的同一材料标记")
                    if target["feature_type"] == "distinct_shaft_landmark" and not target["feature_description"].strip():
                        raise ValueError("导丝段标记需说明如何保证跨帧是同一物理点")
        return result

    def save_reference(self, payload):
        if payload.get("annotation_schema") != REFERENCE_SCHEMA or set(payload["views"]) != set(VIEWS):
            raise ValueError("固定血管参照版本不匹配")
        result = blank_reference()
        result["reviewer"] = reviewer_name(payload)
        for view in VIEWS:
            source, target = payload["views"][view], result["views"][view]
            if set(source["points"]) != set(REFERENCE_POINTS):
                raise ValueError("血管参照必须是P0/P1/PL/PR四个点")
            target.update(confirmed=source.get("confirmed") is True, notes=str(source.get("notes", ""))[:1000])
            for name in REFERENCE_POINTS:
                point = validate_point(source["points"][name])
                point["source"] = None
                if point["status"] == "visible":
                    origin = source["points"][name]["source"]
                    case_id, index = origin["case_id"], origin["frame_index"]
                    if type(index) is not int:
                        raise ValueError("invalid reference source frame")
                    self.image_path(case_id, view, index)
                    point["source"] = {"case_id": case_id, "frame_index": index,
                        "path": self.cases[case_id]["frames"][view][index]["path"]}
                target["points"][name] = point
            for a, b in (("p0", "p1"), ("p1", "pl"), ("p1", "pr")):
                pa, pb = target["points"][a], target["points"][b]
                if pa["status"] == pb["status"] == "visible" and math.dist(pa["xy"], pb["xy"]) < 1:
                    raise ValueError("相邻血管参照点不能重合；看不清请记缺失")
        with self.lock:
            if payload.get("revision") != self.reference["revision"]:
                raise ValueError("另一页面已更新血管参照，请保留草稿后刷新")
            result.update(revision=self.reference["revision"]+1, saved_at_utc=datetime.now(timezone.utc).isoformat(),
                annotation_source="human_static_four_point_reference", coordinate_space="original_image_pixels",
                scope="this six-window pack; user-declared fixed camera and vessel setup",
                policy_input_allowed=False, formal_data_allowed=False)
            with self.reference_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
            self.reference = result
        return {"reference": result, "summary": self.summary()}

    def save(self, payload):
        result = self.validate(payload)
        with self.lock:
            old = self.annotations[result["case_id"]]
            if payload.get("revision") != old["revision"]:
                raise ValueError("另一页面已保存较新版本；请保留草稿后刷新，不覆盖它")
            result.update(revision=old["revision"]+1, saved_at_utc=datetime.now(timezone.utc).isoformat(),
                annotation_source="human_sparse_correspondence", coordinate_space="original_image_pixels",
                source_episode=self.cases[result["case_id"]]["source_episode"], policy_input_allowed=False,
                reference_revision_at_save=self.reference["revision"])
            if old.get("legacy_source"):
                result["legacy_source"] = deepcopy(old["legacy_source"])
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
            self.annotations[result["case_id"]] = result
            self.revisions += 1
        return {"annotation": result, "progress": progress(self.cases[result["case_id"]], result), "summary": self.summary()}


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
            url, query = urlsplit(self.path), parse_qs(urlsplit(self.path).query)
            try:
                if url.path == "/":
                    self.send(Path(__file__).with_name("real10_wire_correspondence.html").read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/index":
                    self.send(store.index())
                elif url.path == "/image":
                    self.send(store.image_path(query["case"][0], query["view"][0], int(query["frame"][0])).read_bytes(), "image/png")
                else:
                    self.send({"error": "not found"}, status=404)
            except (ValueError, KeyError, OSError, IndexError) as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if self.headers.get("Origin") not in (None, *allowed):
                return self.send({"error": "cross-origin writes forbidden"}, status=403)
            try:
                if self.path not in ("/api/annotation", "/api/reference"):
                    return self.send({"error": "not found"}, status=404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                self.send(store.save_reference(payload) if self.path == "/api/reference" else store.save(payload))
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--summary", action="store_true", help="read saved progress only; never create labels")
    p.add_argument("--audit", type=Path, default=Path("simulation_output/real10_local_supervision_semantic_audit_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    p.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    p.add_argument("--port", type=int, default=8794)
    args = p.parse_args()
    if args.prepare:
        prepare(args)
        return
    store = CorrespondenceStore(args.pack, args.raw_root)
    if args.summary:
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        return
    server = make_server(store, args.port)
    print(f"Sparse correspondence: http://127.0.0.1:{server.server_port}\nNew labels only: {store.path}\nCtrl+C stops server; appended labels persist.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
