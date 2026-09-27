"""Six-window, read-only 2D wire motion diagnostic from confirmed human points.

Coarse four-point references are not centerlines, physical insertion distance,
contact, response labels or policy input. No model, flow, training or hardware.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import shutil
import time

from prepare_real10_event_windows import read_json, write_json, write_jsonl
from review_real10_wire_correspondence import CASES, VIEWS, CorrespondenceStore, validate_point


SCHEMA = "real10_wire_path_motion_diagnostic_v1"
PARALLEL = "#007d7a"
TRANSVERSE = "#9e409e"
WIRE = "#ffdb4d"
TEXT = "#203345"
BACKGROUND = "#f3f6f8"
FRAME_NAMES = ("首帧", "中帧", "末帧")
STATUS = {"visible": "已定位", "ambiguous": "可见但定位不可靠",
          "not_visible": "不可见", "unreviewed": "原状态未标注"}


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1]]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def norm(a):
    return math.hypot(*a)


def project(point, vertices):
    """Nearest finite segment; exact distance ties choose the earlier segment."""
    candidates, offset = [], 0.0
    for i, (a, b) in enumerate(zip(vertices, vertices[1:])):
        edge = sub(b, a)
        length = norm(edge)
        if length == 0:
            raise ValueError("zero-length reference segment")
        u = [x / length for x in edge]
        n = [-u[1], u[0]]  # Image x right, y down; image-clockwise +90 degrees.
        raw_t = dot(sub(point, a), u) / length
        t = min(1.0, max(0.0, raw_t))
        q = [a[j] + t * edge[j] for j in (0, 1)]
        candidates.append({"segment": i, "t_raw": raw_t, "t_clamped": t,
                           "endpoint_clamped": raw_t < 0 or raw_t > 1,
                           "projected_xy": q, "distance_px": norm(sub(point, q)),
                           "signed_normal_px": dot(sub(point, q), n),
                           "s_px": offset + t * length, "unit_tangent": u})
        offset += length
    best = min(candidates, key=lambda c: (c["distance_px"], c["segment"]))
    return {**best, "candidates": candidates,
            "nearest_distance_margin_px": abs(candidates[0]["distance_px"] - candidates[1]["distance_px"])}


def decompose(start, end, u):
    delta = sub(end, start)
    n = [-u[1], u[0]]
    parallel, transverse = dot(delta, u), dot(delta, n)
    error = abs(dot(delta, delta) - parallel**2 - transverse**2)
    if error > 1e-8:
        raise AssertionError("orthogonal decomposition does not preserve squared displacement")
    return {"dx_px": delta[0], "dy_px": delta[1], "net_displacement_px": norm(delta),
            "parallel_px": parallel, "transverse_px": transverse,
            "unit_tangent": u, "unit_normal": n, "squared_norm_error": error}


def geometry_checks():
    p = project([3, 2], [[0, 0], [10, 0], [10, 10]])
    d = decompose([3, 2], [7, 5], p["unit_tangent"])
    assert (p["s_px"], p["distance_px"], d["parallel_px"], d["transverse_px"]) == (3, 2, 4, 3)
    assert project([12, 6], [[0, 0], [10, 0], [10, 10]])["s_px"] == 16
    q = project([-2, 1], [[0, 0], [10, 0], [10, 10]])
    assert q["endpoint_clamped"] and q["s_px"] == 0
    assert project([10, 0], [[0, 0], [10, 0], [10, 10]])["segment"] == 0
    return "passed: along/normal, segment selection, endpoint clamp, deterministic tie"


def calculate(store):
    if tuple(c["ui_index"] for c in store.cases.values()) != CASES:
        raise ValueError("only the existing six fixed cases are in scope")
    if (store.pack / "TEST_ONLY.json").exists():
        raise ValueError("UI test fixture is not human annotation")
    for reference in store.reference["views"].values():
        if not reference["confirmed"]:
            raise ValueError("static reference has not been confirmed")
        for p in reference["points"].values():
            if validate_point(p)["status"] != "visible":
                raise ValueError("all four static points must be present")
    rows = []
    for case_id, case in store.cases.items():
        annotation = store.annotations[case_id]
        for view in VIEWS:
            a = annotation["views"][view]
            branch = {"left": "pl", "right": "pr"}[case["task"]]
            points = store.reference["views"][view]["points"]
            vertices = [points[name]["xy"] for name in ("p0", "p1", branch)]
            frames = []
            for i in case["keyframe_indices"]:
                w = validate_point(a["frames"][str(i)]["wire"])
                frame = case["frames"][view][i]
                frames.append({"index": i, "path": frame["path"], "time_s": frame["time_s"],
                               "relative_time_s": frame["time_s"] - case["frames"][view][0]["time_s"],
                               "wire": w,
                               "projection": project(w["xy"], vertices) if w["status"] == "visible" else None,
                               "from_first_fixed_axis": None})
            first, last = frames[0], frames[-1]
            if [first["index"], last["index"]] != case["required_indices"]:
                raise ValueError("endpoints must remain the original required frames")
            reasons = []
            if not a["correspondence_confirmed"]:
                reasons.append("human_same_point_confirmation_absent")
            if a["feature_type"] not in ("tip", "distinct_shaft_landmark"):
                reasons.append("no_trackable_feature_identity")
            if first["projection"] is None or last["projection"] is None:
                reasons.append("required_endpoint_missing_no_substitution")
            motion = None
            if not reasons:
                p0, p1 = first["projection"], last["projection"]
                motion = decompose(first["wire"]["xy"], last["wire"]["xy"], p0["unit_tangent"])
                motion.update({"dt_s": last["time_s"] - first["time_s"],
                               "start_segment": p0["segment"], "end_segment": p1["segment"],
                               "segment_switch": p0["segment"] != p1["segment"],
                               "endpoint_projection_clamped": p0["endpoint_clamped"] or p1["endpoint_clamped"],
                               "delta_s_px": p1["s_px"] - p0["s_px"],
                               "start_path_distance_px": p0["distance_px"],
                               "end_path_distance_px": p1["distance_px"],
                               "delta_s_minus_fixed_parallel_px": p1["s_px"] - p0["s_px"] - motion["parallel_px"],
                               "illustrative_click_sensitivity": [
                                   {"per_endpoint_radius_px": r,
                                    "parallel_interval_px": [motion["parallel_px"] - 2*r, motion["parallel_px"] + 2*r],
                                    "transverse_interval_px": [motion["transverse_px"] - 2*r, motion["transverse_px"] + 2*r]}
                                   for r in (2, 5)]})
                for f in frames:
                    if f["projection"] is not None:
                        f["from_first_fixed_axis"] = decompose(first["wire"]["xy"], f["wire"]["xy"], p0["unit_tangent"])
            rows.append({"case_id": case_id, "ui_index": case["ui_index"], "view": view,
                         "source_episode": case["source_episode"], "task": case["task"],
                         "human_response_unchanged": case["human_response"], "human_notes": case["human_notes"],
                         "annotation_revision": annotation["revision"], "feature_type": a["feature_type"],
                         "feature_description": a["feature_description"],
                         "human_same_point_confirmed": a["correspondence_confirmed"],
                         "reference_revision": store.reference["revision"],
                         "reference_names": ["p0", "p1", branch], "reference_vertices": vertices,
                         "pair_ready": not reasons, "exclusion_reasons": reasons,
                         "frames": frames, "endpoint_motion": motion})
    return rows


def arrow(draw, a, b, color, width=3):
    d = sub(b, a)
    length = norm(d)
    if length < 1:
        return
    draw.line([tuple(a), tuple(b)], fill=color, width=width)
    u = [x / length for x in d]
    head = min(9, length * .4)
    left = [b[0] - head*u[0] - head*.45*u[1], b[1] - head*u[1] + head*.45*u[0]]
    right = [b[0] - head*u[0] + head*.45*u[1], b[1] - head*u[1] - head*.45*u[0]]
    draw.polygon([tuple(b), tuple(left), tuple(right)], fill=color)


def render(rows, store, out, font_path):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)
    title = ImageFont.truetype(str(font_path), 30)

    def picture(path, box, wh, point=None, vertices=None, branch_label="PL/PR"):
        with Image.open(path) as im:
            if im.size != (1920, 1080):
                raise ValueError("original image dimensions changed")
            result = im.convert("RGB").crop(box).resize(wh, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(result)
        sx, sy = wh[0]/(box[2]-box[0]), wh[1]/(box[3]-box[1])
        def xy(p):
            return ((p[0]-box[0])*sx, (p[1]-box[1])*sy)
        if vertices:
            draw.line([xy(p) for p in vertices], fill="#61e5e1", width=2)
            for name, p in zip(("P0", "P1", branch_label), vertices):
                x, y = xy(p)
                draw.ellipse((x-3, y-3, x+3, y+3), fill="#61e5e1")
                draw.text((max(0, min(wh[0]-60, x+5)), max(0, y-25)), name, font=small,
                          fill="#b9fffc", stroke_width=1, stroke_fill="#102b38")
        if point is not None:
            x, y = xy(point)
            draw.ellipse((x-5, y-5, x+5, y+5), outline=WIRE, width=2)
            draw.text((x+8, y-28), "W", font=small, fill=WIRE, stroke_width=1, stroke_fill="#102b38")
        return result

    # Show the static reference on its actual source image, including both branches.
    canvas = Image.new("RGB", (1500, 1260), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), "固定四点：粗方向参照，不是精确中心线", font=title, fill=TEXT)
    draw.text((25, 63), "P1位于分叉前；折线穿过或偏离弯曲管腔均不改点。它不能提供接触或毫米推进真值。", font=font, fill=TEXT)
    for vi, view in enumerate(VIEWS):
        y = 112 + vi * 555
        points = store.reference["views"][view]["points"]
        roi = store.manifest["display_roi_xyxy"][view]
        height = round((roi[3]-roi[1]) * 1440/(roi[2]-roi[0]))
        draw.text((25, y), f"{view.upper()} · 参照来源 #4 首帧 · revision {store.reference['revision']}", font=font, fill=TEXT)
        im = picture(store.raw_root / points["p0"]["source"]["path"], roi, (1440, height))
        d = ImageDraw.Draw(im)
        def xy(p):
            return ((p[0]-roi[0])*1440/(roi[2]-roi[0]), (p[1]-roi[1])*height/(roi[3]-roi[1]))
        for a, b in (("p0", "p1"), ("p1", "pl"), ("p1", "pr")):
            d.line([xy(points[a]["xy"]), xy(points[b]["xy"])], fill="#61e5e1", width=3)
        for name, label in (("p0", "P0 起点"), ("p1", "P1 分叉前"), ("pl", "PL 左支"), ("pr", "PR 右支")):
            x, py = xy(points[name]["xy"])
            d.ellipse((x-5, py-5, x+5, py+5), fill="#61e5e1")
            d.text((max(0, min(1300, x+10)), max(0, py-35)), label, font=font,
                   fill="#b9fffc", stroke_width=2, stroke_fill="#102b38")
        canvas.paste(im, (25, y+33))
    canvas.save(out / "reference_overview.png")

    for case_id, case in store.cases.items():
        canvas = Image.new("RGB", (1500, 1420), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        response = {"advance": "推进", "stationary": "不推进/静止标签"}[case["human_response"]]
        task = {"left": "左支", "right": "右支"}[case["task"]]
        draw.text((25, 18), f"#{case['ui_index']} · 目标{task} · 人工响应：{response}（原标签不变）", font=title, fill=TEXT)
        draw.text((25, 64), "仅原图二维像素。黄圈是人工W；首/中/末同裁剪。青色粗折线不是管腔分割。", font=font, fill=TEXT)
        for vi, view in enumerate(VIEWS):
            r = next(r for r in rows if r["case_id"] == case_id and r["view"] == view)
            y = 112 + vi * 620
            feature = "尖端" if r["feature_type"] == "tip" else (r["feature_description"] or "无可靠跟踪点")
            draw.text((25, y), f"{view.upper()} · {feature} · {'首末同点已确认' if r['pair_ready'] else '首末不可算，保留缺失'}", font=font, fill=TEXT)
            coords = [f["wire"]["xy"] for f in r["frames"] if f["wire"]["xy"] is not None]
            if coords:
                cx = (min(p[0] for p in coords) + max(p[0] for p in coords))/2
                cy = (min(p[1] for p in coords) + max(p[1] for p in coords))/2
                x0, y0 = max(0, min(1680, round(cx-120))), max(0, min(950, round(cy-65)))
                box = [x0, y0, x0+240, y0+130]
            else:
                box = store.manifest["display_roi_xyxy"][view]
            r["display_crop_xyxy_original"] = box
            for fi, f in enumerate(r["frames"]):
                x = 25 + 490*fi
                draw.text((x, y+34), f"{FRAME_NAMES[fi]} +{f['relative_time_s']:.3f}s · {STATUS[f['wire']['status']]}", font=small, fill=TEXT)
                # Untrackable views show the whole ROI, letterboxed rather than stretched.
                h = round((box[3]-box[1])*480/(box[2]-box[0]))
                im = picture(store.image_path(case_id, view, f["index"]), box, (480, h), point=f["wire"]["xy"])
                canvas.paste(im, (x, y+64+(260-h)//2))
            draw.text((25, y+333), "首帧参照全景（只读显示裁剪）", font=small, fill=TEXT)
            roi = store.manifest["display_roi_xyxy"][view]
            h = round((roi[3]-roi[1])*660/(roi[2]-roi[0]))
            canvas.paste(picture(store.image_path(case_id, view, r["frames"][0]["index"]), roi, (660, h),
                                 point=r["frames"][0]["wire"]["xy"], vertices=r["reference_vertices"],
                                 branch_label=r["reference_names"][-1].upper()), (25, y+363))
            m = r["endpoint_motion"]
            if m is None:
                draw.text((715, y+365), "没有首末数值；不补零、不换成中帧。", font=font, fill=TEXT)
                draw.text((715, y+400), "用户已说明留空原因，原status与null均保留。", font=small, fill=TEXT)
                continue
            draw.text((715, y+333), f"沿向 {m['parallel_px']:+.2f}   横向 {m['transverse_px']:+.2f}   Δs {m['delta_s_px']:+.2f} px", font=font, fill=TEXT)
            draw.text((715, y+367), f"距粗折线 首 {m['start_path_distance_px']:.1f} / 末 {m['end_path_distance_px']:.1f} px（不是壁距）", font=small, fill=TEXT)
            draw.text((715, y+397), "分解示意：黑=净位移，绿=沿向，紫=横向；1原图px画作4px", font=small, fill=TEXT)
            # Fixed scale across all views; translation only, no rotated drawing axes.
            a = [0.0, 0.0]
            b = [4*m["parallel_px"]*u for u in m["unit_tangent"]]
            c = [4*m["dx_px"], 4*m["dy_px"]]
            offset = [1070-(min(p[0] for p in (a,b,c))+max(p[0] for p in (a,b,c)))/2,
                      y+500-(min(p[1] for p in (a,b,c))+max(p[1] for p in (a,b,c)))/2]
            a, b, c = [[p[j]+offset[j] for j in (0,1)] for p in (a,b,c)]
            arrow(draw, a, c, TEXT, 2)
            arrow(draw, a, b, PARALLEL, 4)
            arrow(draw, b, c, TRANSVERSE, 4)
            draw.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3), fill=TEXT)
            draw.text((715, y+586), "固定首帧参照方向；横向正号 n=(-uy,ux)，不是解剖方向。", font=small, fill=TEXT)
        draw.text((25, 1371), "不是独立测试集；不自动判推进/静止，不改响应标签，不融合两相机数值，不训练。", font=font, fill=TEXT)
        canvas.save(out / f"{case_id}.png")


def report_text(rows):
    lines = ["# 六窗二维沿向/横向诊断", "", "只读人工同点与固定四点；原始响应标签保留，无模型训练。",
             "单位为各视角原图像素（x向右、y向下），不同相机的数值不平均、不相加。",
             "", "| 窗口 | 视角 | 人工响应 | 沿向 | 横向 | Δs | 首/末距粗折线 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        m = r["endpoint_motion"]
        nums = (f"{m['parallel_px']:+.3f} | {m['transverse_px']:+.3f} | {m['delta_s_px']:+.3f} | "
                f"{m['start_path_distance_px']:.2f}/{m['end_path_distance_px']:.2f}") if m else "缺失 | 缺失 | 缺失 | —"
        lines.append(f"| #{r['ui_index']} | {r['view']} | {r['human_response_unchanged']} | {nums} |")
    lines += ["", "## 计算约定", "",
              "1. task只选P0→P1→PL/PR粗参照，不证明实际走向；P1在分叉前，不是精确分叉点。",
              "2. 首帧W到两条有限线段的最近段确定单位方向u，全窗固定；沿向=ΔW·u，横向=ΔW·(-uy,ux)。",
              "3. Δs另按各帧最近有限线段计算；报告segment switch、clamp和候选距离，不能隐藏与固定方向的差异。",
              "4. report.json包含每个可见中帧的位置/投影和首帧到该帧的分解；不以净首末位移代替完整轨迹。",
              "5. ±2/±5px端点圆形扰动仅是说明性敏感性：固定轴分量各为±4/±10px范围，不是置信区间。",
              "   它不包括粗路径、参照方向、系统性标点误差或跨视角误差，不用于分类阈值。",
              "6. #4跟踪材料交界处，其余可用对跟踪tip；材料点运动不能直接冒充尖端推进。",
              "7. 未定位/未确认首末保持null；不由另一视角补点，不将缺失当静止。",
              "8. 本批是已用于开发的六个难例，不报告准确率、泛化收益或contact/tactile结论。", ""]
    return "\n".join(lines)


def run(args):
    started = time.perf_counter()
    if args.render_only:
        # Display-only fixes use the frozen output inputs and numbers, never live annotations.
        report = read_json(args.out/"report.json")
        if report["schema"] != SCHEMA:
            raise ValueError("not this diagnostic report")
        store = CorrespondenceStore(args.out, args.raw_root)
        render(report["rows"], store, args.out, args.font)
        shutil.copyfile(__file__, args.out/"renderer_snapshot.py")
        print(json.dumps({"render_only": True, "numeric_report_unchanged": True,
                          "elapsed_seconds": time.perf_counter()-started}))
        return
    if args.out.exists():
        raise FileExistsError("preserve the previous diagnostic; choose a fresh --out")
    store = CorrespondenceStore(args.pack, args.raw_root)
    inputs = [store.pack/name for name in ("manifest.json", "annotations_v2.jsonl", "vessel_reference.jsonl",
                                           "interface_protocol_v2.json", "chat_confirmation_20260921_v1.json")]
    original = {p: p.read_bytes() for p in inputs}
    checks = geometry_checks()
    rows = calculate(store)
    args.out.mkdir(parents=True)
    for path in inputs:
        shutil.copyfile(path, args.out/path.name)
    shutil.copyfile(__file__, args.out/"entrypoint_snapshot.py")
    protocol = {"schema": SCHEMA, "case_ids": list(CASES), "selected_before_analysis": True,
                "reference": "task selects P0-P1-PL/PR; same static reference per view; no point fitting",
                "axis": "start point nearest finite segment, fixed within the window; distance ties choose earlier segment",
                "parallel": "dot(last-first, u_start)", "transverse": "dot(last-first, [-uy,ux])",
                "secondary_delta_s": "difference of nearest-segment cumulative projected coordinates; no extrapolation",
                "coordinate_space": "original 1920x1080 per-view pixels, x right, y down; not mm, not 3D",
                "click_sensitivity": "illustrative Euclidean endpoint radii 2,5px => component bound +/-2r at fixed u; NOT CI or calibrated uncertainty; excludes reference-axis error",
                "display": "unchanged global ROI plus same 240x130 original-pixel crop across each visible-point view/window; no masking",
                "missing": "null stays null; no middle-frame substitution or cross-view imputation",
                "human_label_used_for_geometry": False, "training_executed": False, "labels_modified": False,
                "policy_input_allowed": False, "formal_data_allowed": False, "real_system_validated": False}
    if not args.numeric_only:
        render(rows, store, args.out, args.font)
    ready = [r for r in rows if r["pair_ready"]]
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol, "input_summary": store.summary(), "rows": rows,
              "summary": {"cases": len(store.cases), "view_pairs": len(rows), "ready_pairs": len(ready),
                          "missing_pairs": len(rows)-len(ready),
                          "ready_feature_counts": dict(Counter(r["feature_type"] for r in ready)),
                          "segment_switch_pairs": sum(r["endpoint_motion"]["segment_switch"] for r in ready),
                          "clamped_endpoint_pairs": sum(r["endpoint_motion"]["endpoint_projection_clamped"] for r in ready)},
              "verification": {"geometry_checks": checks,
                               "input_bytes_unchanged": all(p.read_bytes() == b for p,b in original.items())},
              "images_rendered": not args.numeric_only, "elapsed_seconds": time.perf_counter()-started,
              "visual_status": "not_viewed", "user_result_acceptance": "pending"}
    if not report["verification"]["input_bytes_unchanged"]:
        raise RuntimeError("inputs changed during the diagnostic; retain output for inspection")
    write_json(args.out/"protocol.json", protocol)
    write_json(args.out/"report.json", report)
    write_jsonl(args.out/"view_motion.jsonl", rows)
    (args.out/"README.md").write_text(report_text(rows), encoding="utf-8")
    if not args.numeric_only:
        cards = "\n".join(f'<section><h2>#{c["ui_index"]}</h2><a href="{c["id"]}.png"><img src="{c["id"]}.png" alt="窗口{c["ui_index"]}二维诊断"></a></section>' for c in store.cases.values())
        page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>六窗二维方向诊断</title>'
                '<style>body{font:18px system-ui;max-width:1500px;margin:30px auto;background:#f3f6f8;color:#203345}'
                'img{width:100%}section{margin:40px 0}pre{white-space:pre-wrap}</style>'
                '<h1>二维方向诊断 · 人工点已确认，投影结果待审阅</h1>'
                '<p>仅6个开发难例；不是训练标签、contact或毫米推进真值。点击图片看原尺寸。</p>'
                '<a href="report.json">完整数值与来源</a> · <a href="README.md">口径说明</a>'
                '<section><img src="reference_overview.png" alt="粗参照适用性"></section>' + cards +
                '<pre>' + html.escape(report_text(rows)) + '</pre></html>')
        (args.out/"index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"out": str(args.out), **report["summary"], **report["verification"],
                      "images_rendered": report["images_rendered"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    parser.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_wire_path_motion_diagnostic_v1"))
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--numeric-only", action="store_true", help="Compute the exact same geometry without raw images or Pillow (remote check).")
    parser.add_argument("--render-only", action="store_true", help="Redraw an existing output from its frozen points/report; no geometry or annotation changes.")
    run(parser.parse_args())
