"""Loopback-only review of recorded real10 images; append human weak labels.

Reads original RGB on demand. No camera/robot/network client/model is imported.
Only annotations_joint_v2.jsonl is written; legacy annotations stay read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from prepare_real10_event_windows import SCHEMA, VIEWS, VISIBILITY, MOTION, read_json, read_jsonl


ANNOTATION_SCHEMA = "real10_joint_response_annotation_v2"
MOTION_EVIDENCE = ("side", "top", "both", "insufficient")


def annotation_status(row):
    complete_visibility = all(row["anchor_visibility"][v] is not None for v in VIEWS)
    if complete_visibility and row["joint_motion_response"] is not None:
        return "reviewed" if row["motion_evidence"] is not None else "needs_evidence"
    values = [*row["anchor_visibility"].values(), row["joint_motion_response"], row["motion_evidence"]]
    return "partial" if any(v is not None for v in values) else "unreviewed"


def project_legacy(row, source_line=None):
    """Read-only projection: agreed responses survive, evidence is not guessed."""
    per_view = row["future_motion_response"]
    agreed = per_view["side"] is not None and per_view["side"] == per_view["top"]
    result = {"annotation_schema": ANNOTATION_SCHEMA,
              **{k: row[k] for k in ("window_id", "source_episode", "split_group", "source_split")},
              "reviewer": row.get("reviewer"), "annotation_source": "legacy_projection",
              "saved_at_utc": None, "anchor_visibility": dict(row["anchor_visibility"]),
              "joint_motion_response": per_view["side"] if agreed else None,
              "motion_evidence": None, "notes": row.get("notes", ""), "legacy_migration": None}
    if source_line is not None:
        result["legacy_migration"] = {
            "source_file": "annotations.jsonl", "source_line": source_line,
            "source_saved_at_utc": row.get("saved_at_utc"), "source_reviewer": row.get("reviewer"),
            "per_view_motion": dict(per_view),
            "rule": "equal_nonnull_response_to_joint_else_manual_review",
            "result": "agreed" if agreed else "needs_response_review", "evidence_inferred": False,
        }
    result["status"] = annotation_status(result)
    return result


class ReviewStore:
    def __init__(self, pack, source_root):
        self.pack = Path(pack).resolve()
        self.source_root = Path(source_root).resolve()
        self.manifest = read_json(self.pack / "manifest.json")
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("unsupported event-window schema")
        self.windows = read_jsonl(self.pack / "windows.jsonl")
        self.by_id = {w["window_id"]: w for w in self.windows}
        self.frames = {r["frame_id"]: r for r in read_jsonl(self.pack / "observations.jsonl")}
        self.targets = {r["window_id"]: r for r in read_jsonl(self.pack / "window_targets.jsonl")}
        self.logs = {r["frame_id"]: r for r in read_jsonl(self.pack / "execution_log.jsonl")}
        self.annotations = {r["window_id"]: project_legacy(r) for r in read_jsonl(self.pack / "annotation_template.jsonl")}
        self.legacy_path = self.pack / "annotations.jsonl"
        self.path = self.pack / "annotations_joint_v2.jsonl"
        self.legacy_revisions = read_jsonl(self.legacy_path) if self.legacy_path.exists() else []
        for line, row in enumerate(self.legacy_revisions, 1):
            if row["window_id"] not in self.by_id:
                raise ValueError("legacy annotation belongs to another window pack")
            self.annotations[row["window_id"]] = project_legacy(row, line)
        if self.path.exists():
            for row in read_jsonl(self.path):
                if row.get("annotation_schema") != ANNOTATION_SCHEMA:
                    raise ValueError("unsupported annotation schema")
                if row["window_id"] not in self.by_id:
                    raise ValueError("annotation belongs to another window pack")
                self.annotations[row["window_id"]] = row
        self.lock = threading.Lock()

    def image_path(self, identifier, view):
        if view not in VIEWS:
            raise ValueError("unknown view")
        path = (self.source_root / self.frames[identifier]["images"][view]["path"]).resolve()
        if not path.is_relative_to(self.source_root):
            raise ValueError("image path leaves source root")
        return path

    def index(self):
        return {"schema": SCHEMA, "windows": [{**w, "status": self.annotations[w["window_id"]]["status"]} for w in self.windows],
                "annotation_schema": ANNOTATION_SCHEMA, "annotation_path": str(self.path),
                "legacy_annotation_path_read_only": str(self.legacy_path), "manifest": self.manifest,
                "summary": self.summary()}

    def summary(self):
        annotations = list(self.annotations.values())
        return {"annotation_schema": ANNOTATION_SCHEMA, "windows": len(annotations),
                "legacy_revisions": len(self.legacy_revisions),
                "legacy_windows": len({r["window_id"] for r in self.legacy_revisions}),
                "status": dict(Counter(a["status"] for a in annotations)),
                "joint_motion_response": dict(Counter(a["joint_motion_response"] for a in annotations if a["joint_motion_response"] is not None)),
                "motion_evidence": dict(Counter(a["motion_evidence"] for a in annotations if a["motion_evidence"] is not None)),
                "annotation_path": str(self.path), "legacy_file_modified": False}

    def window(self, identifier):
        w = self.by_id[identifier]
        target = self.targets[identifier]
        ids = w["history_frame_ids"] + target["future_frame_ids"]
        return {"window": w, "target_audit_only": target,
                "frames": [self.frames[key] for key in ids],
                "execution_logs_audit_only": [self.logs[key] for key in ids],
                "annotation": self.annotations[identifier]}

    def save(self, payload):
        if payload.get("annotation_schema") != ANNOTATION_SCHEMA:
            raise ValueError("标注页版本已更新，请保留未保存内容后刷新页面；旧版提交未写入")
        identifier = payload["window_id"]
        w = self.by_id[identifier]
        reviewer = str(payload.get("reviewer", "")).strip()
        if not reviewer or len(reviewer) > 120:
            raise ValueError("请填写标注人姓名或缩写")
        result = {"annotation_schema": ANNOTATION_SCHEMA,
                  "window_id": identifier, "source_episode": w["source_episode"],
                  "split_group": w["split_group"], "source_split": w["source_split"],
                  "reviewer": reviewer, "annotation_source": "human_visual_review",
                  "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "label_scope": "per_view_anchor_visibility_joint_future_response_not_contact_truth",
                  "legacy_migration": self.annotations[identifier].get("legacy_migration")}
        values = payload.get("anchor_visibility", {})
        if set(values) != set(VIEWS) or any(values[v] not in (*VISIBILITY, None) for v in VIEWS):
            raise ValueError("invalid anchor_visibility")
        result["anchor_visibility"] = {v: values[v] for v in VIEWS}
        for field, allowed in (("joint_motion_response", MOTION), ("motion_evidence", MOTION_EVIDENCE)):
            if field not in payload or payload[field] not in (*allowed, None):
                raise ValueError(f"invalid {field}")
            result[field] = payload[field]
        if result["motion_evidence"] == "insufficient" and result["joint_motion_response"] not in (None, "uncertain"):
            raise ValueError("证据不足时，联合响应应为无法判断或未标注")
        if not self.targets[identifier]["future_frame_ids"] and result["joint_motion_response"] not in (None, "uncertain"):
            raise ValueError("no future observations: motion must be uncertain or unreviewed")
        notes = str(payload.get("notes", ""))
        if len(notes) > 4000:
            raise ValueError("notes too long")
        result["notes"] = notes
        result["status"] = annotation_status(result)
        # Append-only human revisions. Source records and previous labels remain.
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            self.annotations[identifier] = result
        return result


def make_server(store, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, content, content_type="application/json; charset=utf-8", status=200):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self.send(Path(__file__).with_name("real10_event_review.html").read_bytes(), "text/html; charset=utf-8")
                elif parsed.path == "/api/index":
                    self.send(store.index())
                elif parsed.path == "/api/window":
                    self.send(store.window(query["id"][0]))
                elif parsed.path == "/image":
                    path = store.image_path(query["id"][0], query["view"][0])
                    self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/png")
                else:
                    self.send({"error": "not found"}, status=404)
            except (KeyError, ValueError, OSError) as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if origin is not None and origin not in allowed:
                return self.send({"error": "cross-origin writes forbidden"}, status=403)
            try:
                if self.path != "/api/annotation":
                    return self.send({"error": "not found"}, status=404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("invalid request size")
                self.send(store.save(json.loads(self.rfile.read(length))))
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--source-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--check-images", action="store_true", help="Check referenced file existence, print counts and exit; no decode")
    parser.add_argument("--summary", action="store_true", help="Read-only v2 projection/counts; never writes or changes annotations")
    args = parser.parse_args()
    store = ReviewStore(args.pack, args.source_root)
    if args.summary:
        print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        return 0
    if args.check_images:
        paths = [store.image_path(key, view) for key in store.frames for view in VIEWS]
        missing = [str(p) for p in paths if not p.is_file()]
        print(json.dumps({"references": len(paths), "present": len(paths) - len(missing),
                          "missing_first10": missing[:10], "decoded": 0}, ensure_ascii=False, indent=2))
        return int(bool(missing))
    server = make_server(store, args.port)
    print(f"Offline review: http://127.0.0.1:{server.server_port}\nLabels: {store.path}\nCtrl+C to stop; saved revisions persist.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
