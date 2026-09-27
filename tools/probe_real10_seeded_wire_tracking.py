"""Fixed six-window manual-seed tracking/abstention probe; no training or hardware.

Only the first-frame point enters the tracker. Later human points, response
labels, task and static vessel reference are offline evaluation/display only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from audit_real10_wire_path_motion import decompose, project
from prepare_real10_event_windows import read_json, write_json, write_jsonl
from review_real10_wire_correspondence import CASES, VIEWS, CorrespondenceStore


SCHEMA = "real10_seeded_wire_tracking_probe_v1"
# One predeclared configuration, no case-wise tuning or threshold selection.
PARAMETERS = {"lk_window_original_px": 41, "lk_levels": 3, "lk_max_iterations": 30,
              "lk_epsilon": .01, "lk_min_eigenvalue": 1e-4,
              "fb_max_original_px": 2.0, "step_max_original_px": 64.0,
              "patch_width_original_px": 21, "patch_std_min_gray": 2.0,
              "adjacent_ncc_min": .70, "initial_ncc_min": .60,
              "reject_latched": True, "automatic_reinitialization": False}
REASONS = {"no_first_frame_seed": "首帧无点，不初始化", "seed_texture_low": "首帧纹理不足",
           "lk_forward_failed": "正向追踪失败", "lk_backward_failed": "反向追踪失败",
           "patch_out_of_bounds": "局部块越界", "forward_backward": "前后向不一致",
           "step_too_large": "单步过大", "current_texture_low": "当前纹理不足",
           "adjacent_appearance": "相邻外观不一致", "initial_appearance": "偏离首帧外观",
           "raw_backend_lost": "候选追踪已丢失"}
NAMES = ("首帧", "中帧", "末帧")


def patch(gray, xy):
    radius = PARAMETERS["patch_width_original_px"] // 2
    if xy is None or not np.isfinite(xy).all():
        return None
    x, y = xy
    if not (radius <= x < gray.shape[1]-radius and radius <= y < gray.shape[0]-radius):
        return None
    return cv2.getRectSubPix(gray, (2*radius+1, 2*radius+1), (float(x), float(y))).astype(np.float32)


def correlation(a, b):
    if a is None or b is None:
        return None
    x, y = (a-a.mean()).ravel(), (b-b.mean()).ravel()
    denominator = float(np.linalg.norm(x)*np.linalg.norm(y))
    return float(np.clip(np.dot(x, y)/denominator, -1, 1)) if denominator > 1e-8 else None


def track(images, first_xy):
    """Causal inputs only. Rejected raw shadow is logged but never re-emitted."""
    if first_xy is None:
        return [{"index": i, "candidate_xy": None, "output_xy": None, "state": "uninitialized",
                 "quality": {}, "step_reasons": ["no_first_frame_seed"], "first_rejection": None}
                for i in range(len(images))]
    seed = [float(x) for x in first_xy]
    initial = patch(images[0], seed)
    std = float(initial.std()) if initial is not None else None
    first_rejection = None
    if std is None or std < PARAMETERS["patch_std_min_gray"]:
        first_rejection = {"index": 0, "reasons": ["seed_texture_low"]}
    rows = [{"index": 0, "candidate_xy": seed, "output_xy": seed if first_rejection is None else None,
             "state": "manual_seed" if first_rejection is None else "abstained",
             "quality": {"patch_std_gray": std}, "step_reasons": [] if first_rejection is None else ["seed_texture_low"],
             "first_rejection": first_rejection}]
    raw = seed
    p = PARAMETERS
    lk = {"winSize": (p["lk_window_original_px"],)*2, "maxLevel": p["lk_levels"],
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, p["lk_max_iterations"], p["lk_epsilon"]),
          "minEigThreshold": p["lk_min_eigenvalue"]}
    for i in range(1, len(images)):
        reasons, quality, candidate = [], {}, None
        if raw is None:
            reasons.append("raw_backend_lost")
        else:
            point = np.array([[raw]], dtype=np.float32)
            forward, status, _ = cv2.calcOpticalFlowPyrLK(images[i-1], images[i], point, None, **lk)
            if forward is None or not bool(status[0, 0]) or not np.isfinite(forward).all():
                reasons.append("lk_forward_failed")
            else:
                candidate = forward[0, 0].astype(float).tolist()
                backward, reverse, _ = cv2.calcOpticalFlowPyrLK(images[i], images[i-1], forward, None, **lk)
                fb = (float(np.linalg.norm(backward[0, 0]-point[0, 0]))
                      if backward is not None and bool(reverse[0, 0]) and np.isfinite(backward).all() else None)
                before, after = patch(images[i-1], raw), patch(images[i], candidate)
                step = float(np.linalg.norm(np.asarray(candidate)-np.asarray(raw)))
                std = float(after.std()) if after is not None else None
                adjacent, anchored = correlation(before, after), correlation(initial, after)
                quality = {"fb_error_px": fb, "step_px": step, "patch_std_gray": std,
                           "adjacent_ncc": adjacent, "initial_ncc": anchored}
                if after is None or before is None:
                    reasons.append("patch_out_of_bounds")
                if fb is None:
                    reasons.append("lk_backward_failed")
                elif fb > p["fb_max_original_px"]:
                    reasons.append("forward_backward")
                if step > p["step_max_original_px"]:
                    reasons.append("step_too_large")
                if std is None or std < p["patch_std_min_gray"]:
                    reasons.append("current_texture_low")
                if adjacent is None or adjacent < p["adjacent_ncc_min"]:
                    reasons.append("adjacent_appearance")
                if anchored is None or anchored < p["initial_ncc_min"]:
                    reasons.append("initial_appearance")
        if reasons and first_rejection is None:
            first_rejection = {"index": i, "reasons": reasons.copy()}
        rows.append({"index": i, "candidate_xy": candidate,
                     "output_xy": candidate if first_rejection is None else None,
                     "state": "tracked" if first_rejection is None else "abstained",
                     "quality": quality, "step_reasons": reasons, "first_rejection": first_rejection})
        raw = candidate  # Diagnostic shadow only after a gate rejection.
    return rows


def check():
    rng = np.random.default_rng(20260922)
    gray = cv2.GaussianBlur(rng.integers(0, 256, (192, 256), dtype=np.uint8), (5, 5), 1)
    moved = cv2.warpAffine(gray, np.float32([[1, 0, 5], [0, 1, -3]]), (256, 192))
    rows = track([gray, moved, np.zeros_like(gray), moved], [128, 96])
    error = float(np.linalg.norm(np.asarray(rows[1]["output_xy"])-[133, 93]))
    assert error < .2 and rows[1]["state"] == "tracked"
    assert rows[2]["output_xy"] is None and rows[3]["output_xy"] is None
    assert all(r["state"] == "uninitialized" for r in track([gray, moved], None))
    return {"known_translation_error_px": error, "blank_frame_abstention_and_latch": True,
            "missing_seed_not_initialized": True, "opencv": cv2.__version__}


def stats(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    return {"n": int(a.size), "mean_px": float(a.mean()) if a.size else None,
            "median_px": float(np.median(a)) if a.size else None,
            "max_px": float(a.max()) if a.size else None}


def evaluate(store, predictions):
    """Only called AFTER all predictions have been generated and saved."""
    rows = []
    for item in predictions:
        case, view = store.cases[item["case_id"]], item["view"]
        ann = store.annotations[item["case_id"]]["views"][view]
        points = store.reference["views"][view]["points"]
        end_name = {"left": "pl", "right": "pr"}[case["task"]]
        vertices = [points[n]["xy"] for n in ("p0", "p1", end_name)]
        u = project(item["seed_xy"], vertices)["unit_tangent"] if item["seed_xy"] is not None else None
        for i in case["keyframe_indices"][1:]:
            truth = ann["frames"][str(i)]["wire"]
            pred = item["frames"][i]
            eligible = (item["seed_xy"] is not None and ann["correspondence_confirmed"]
                        and truth["status"] == "visible" and ann["feature_type"] in ("tip", "distinct_shaft_landmark"))
            row = {"case_id": item["case_id"], "ui_index": case["ui_index"], "view": view, "index": i,
                   "is_endpoint": i == case["required_indices"][-1], "feature_type": ann["feature_type"],
                   "human_wire": truth, "human_same_point_confirmed": ann["correspondence_confirmed"],
                   "initialized": item["seed_xy"] is not None, "point_error_eligible": bool(eligible),
                   "prediction": pred, "raw_error_px": None, "output_error_px": None,
                   "repeat_seed_error_px": None, "human_motion": None, "raw_motion": None, "output_motion": None}
            if eligible:
                seed, target = item["seed_xy"], truth["xy"]
                row["repeat_seed_error_px"] = float(np.linalg.norm(np.asarray(seed)-target))
                row["human_motion"] = decompose(seed, target, u)
                for key, prefix in (("candidate_xy", "raw"), ("output_xy", "output")):
                    if pred[key] is not None:
                        row[prefix+"_error_px"] = float(np.linalg.norm(np.asarray(pred[key])-target))
                        row[prefix+"_motion"] = decompose(seed, pred[key], u)
            rows.append(row)
    return rows


def summarize(rows, predictions):
    def group(selected):
        eligible = [r for r in selected if r["point_error_eligible"]]
        accepted = [r for r in eligible if r["output_error_px"] is not None]
        raw = [r for r in eligible if r["raw_error_px"] is not None]
        return {"human_evaluable_points": len(eligible), "raw_candidate_points": len(raw),
                "output_points": len(accepted), "coverage": len(accepted)/len(eligible) if eligible else None,
                "raw_error": stats(r["raw_error_px"] for r in eligible),
                "repeat_seed_on_raw_subset": stats(r["repeat_seed_error_px"] for r in raw),
                "output_error": stats(r["output_error_px"] for r in accepted),
                "repeat_seed_on_output_subset": stats(r["repeat_seed_error_px"] for r in accepted),
                "repeat_seed_all": stats(r["repeat_seed_error_px"] for r in eligible),
                "output_error_exceed_5px_diagnostic_only": sum(r["output_error_px"] > 5 for r in accepted),
                "output_error_exceed_10px_diagnostic_only": sum(r["output_error_px"] > 10 for r in accepted)}
    explicit = [r for r in rows if r["initialized"] and r["human_wire"]["status"] in ("ambiguous", "not_visible")]
    return {"view_sequences": len(predictions), "initialized_views": sum(p["seed_xy"] is not None for p in predictions),
            "frame_slots": sum(len(p["frames"]) for p in predictions),
            "all_evaluable_nonseed_keyframes": group(rows), "endpoints": group([r for r in rows if r["is_endpoint"]]),
            "tip_only_endpoints": group([r for r in rows if r["is_endpoint"] and r["feature_type"] == "tip"]),
            "explicit_ambiguous_or_not_visible_initialized_points": len(explicit),
            "abstained_on_explicit_unlocalizable_points": sum(r["prediction"]["output_xy"] is None for r in explicit),
            "unreviewed_keyframes_not_scored_as_invisible": sum(r["human_wire"]["status"] == "unreviewed" for r in rows),
            "first_rejection_reasons": dict(Counter(reason for p in predictions if p["frames"][-1]["first_rejection"]
                                                     for reason in p["frames"][-1]["first_rejection"]["reasons"]))}


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def render(store, predictions, evaluations, out, font_path):
    from PIL import Image, ImageDraw, ImageFont
    title = ImageFont.truetype(str(font_path), 29)
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)
    bg, text = "#f3f6f8", "#203345"
    for case_id, case in store.cases.items():
        canvas = Image.new("RGB", (1500, 1190), bg)
        draw = ImageDraw.Draw(canvas)
        draw.text((25, 18), f"#{case['ui_index']} · 首帧锚定追踪：人工点仅初始化，后续点只核对", font=title, fill=text)
        draw.text((25, 62), "黄圈=人工点；青×=追踪输出；红×=已拒判的原始候选（不是输出）；灰圈=不动对照。", font=font, fill=text)
        for vi, view in enumerate(VIEWS):
            item = next(p for p in predictions if p["case_id"] == case_id and p["view"] == view)
            ann = store.annotations[case_id]["views"][view]
            feature = "尖端" if ann["feature_type"] == "tip" else ann["feature_description"] or "无可靠点"
            y = 116 + 505*vi
            draw.text((25, y), f"{view.upper()} · {feature} · {'首帧人工初始化' if item['seed_xy'] is not None else '首帧缺失，不初始化/不自动重获'}", font=font, fill=text)
            # Display box may use later annotations ONLY after tracking, never as tracker input.
            coords = [ann["frames"][str(i)]["wire"]["xy"] for i in case["keyframe_indices"]]
            coords += [item["frames"][i]["candidate_xy"] for i in case["keyframe_indices"]]
            coords = [p for p in coords if p is not None and 0 <= p[0] < 1920 and 0 <= p[1] < 1080]
            if coords:
                lo, hi = np.min(coords, axis=0), np.max(coords, axis=0)
                width, height = min(1920, max(240, int(hi[0]-lo[0])+80)), min(1080, max(150, int(hi[1]-lo[1])+80))
                x0 = max(0, min(1920-width, int((lo[0]+hi[0]-width)/2)))
                y0 = max(0, min(1080-height, int((lo[1]+hi[1]-height)/2)))
                box = [x0, y0, x0+width, y0+height]
            else:
                box = store.manifest["display_roi_xyxy"][view]
            for fi, i in enumerate(case["keyframe_indices"]):
                x, pred = 25+fi*490, item["frames"][i]
                truth = ann["frames"][str(i)]["wire"]
                state = {"uninitialized": "未初始化", "manual_seed": "人工初值", "tracked": "输出", "abstained": "拒判"}[pred["state"]]
                draw.text((x, y+35), f"{NAMES[fi]} · 帧{i} · {state} · 人工{truth['status']}", font=small, fill=text)
                with Image.open(store.image_path(case_id, view, i)) as image:
                    im = image.convert("RGB").crop(box)
                    scale = min(480/im.width, 300/im.height)
                    im = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS)
                mark = ImageDraw.Draw(im)
                for p, color, cross in ((item["seed_xy"], "#c4c4c4", False), (truth["xy"], "#ffdf40", False),
                                        (pred["candidate_xy"], "#55ffff" if pred["output_xy"] is not None else "#ff5e77", True)):
                    if p is None:
                        continue
                    px, py = (p[0]-box[0])*scale, (p[1]-box[1])*scale
                    if cross:
                        mark.line((px-6,py-6,px+6,py+6), fill=color, width=2)
                        mark.line((px-6,py+6,px+6,py-6), fill=color, width=2)
                    else:
                        mark.ellipse((px-7,py-7,px+7,py+7), outline=color, width=2)
                canvas.paste(im, (x+(480-im.width)//2, y+65+(300-im.height)//2))
                ev = next((r for r in evaluations if r["case_id"] == case_id and r["view"] == view and r["index"] == i), None)
                line = (f"原始误差 {fmt(ev['raw_error_px'])} / 输出 {fmt(ev['output_error_px'])} / 不动 {fmt(ev['repeat_seed_error_px'])} px"
                        if ev else "初值不计入追踪误差或覆盖率")
                draw.text((x, y+375), line, font=small, fill=text)
                if i > 0 and pred["quality"]:
                    q = pred["quality"]
                    draw.text((x, y+405), f"FB {fmt(q['fb_error_px'])}  相邻NCC {fmt(q['adjacent_ncc'])}  初帧NCC {fmt(q['initial_ncc'])}", font=small, fill=text)
            reject = item["frames"][-1]["first_rejection"]
            reason = (f"首次拒判：帧{reject['index']}；" + "、".join(REASONS[k] for k in reject["reasons"])) if reject else "本视角没有门限拒判（不等于同点身份或可见性已经可靠）"
            if item["seed_xy"] is None:
                reason = "首帧缺失造成不初始化，不计为自动识别不可见的成功；后续人工点不参与重初始化。"
            draw.text((25, y+445), reason, font=small, fill=text)
        draw.text((25, 1140), "6个开发难例；门限是未标定启发式，不是可见性概率。不补点、不改原标签、不进入训练/控制。", font=font, fill=text)
        canvas.save(out/f"{case_id}.png")


def write_readme(report, out):
    s = report["summary"]
    lines = ["# 六窗首帧锚定追踪与拒判", "", "固定配置，无训练、调阈值、重标注或自动重初始化。",
             "只有首帧人工点和截至当前帧的图像进入追踪；后续人工点、任务、四点参照仅离线评估/显示。",
             "", "| 口径 | 可评人工点 | 输出数 | 覆盖率 | 原始候选误差px | 输出子集误差px | 同子集不动误差px |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, name in (("all_evaluable_nonseed_keyframes", "中末帧"), ("endpoints", "仅末帧"), ("tip_only_endpoints", "仅尖端末帧")):
        g = s[key]
        lines.append(f"| {name} | {g['human_evaluable_points']} | {g['output_points']} | {g['coverage']:.1%} | {fmt(g['raw_error']['mean_px'])} | {fmt(g['output_error']['mean_px'])} | {fmt(g['repeat_seed_on_output_subset']['mean_px'])} |")
    lines += ["", "上述误差为人工点的欧氏像素偏差，不是动作MAE、毫米推进、接触误差或独立测试成绩。",
              "拒判后只读原始候选继续记录以暴露漂移，实际output保持null；不能通过拒掉难例隐藏覆盖率。",
              "", "## 逐窗末帧", "", "| 窗口/视角 | 状态 | 人工状态 | 原始误差 | 输出误差 | 不动误差 |",
              "|---|---|---|---:|---:|---:|"]
    for r in report["evaluation"]:
        if r["is_endpoint"]:
            lines.append(f"| #{r['ui_index']} {r['view']} | {r['prediction']['state']} | {r['human_wire']['status']} | {fmt(r['raw_error_px'])} | {fmt(r['output_error_px'])} | {fmt(r['repeat_seed_error_px'])} |")
    lines += ["", "## 限制", "", "- 初值由人工给出，本次不是自动尖端检测/重获；Side与Top独立，不互相补点。",
              "- #4跟踪材料交界处，不能混称所有点为尖端。", "- unreviewed不当不可见；首帧缺失不初始化不等于可见性检测成功。",
              f"- 已初始化视角只有{s['explicit_ambiguous_or_not_visible_initialized_points']}个明确ambiguous/not_visible后续标记，其中{s['abstained_on_explicit_unlocalizable_points']}个拒判；不足以估计不可见检测率。",
              "- 质量门限未标定；静态血管纹理也可能通过FB/NCC。5/10px仅显示误差量级，不用于训练或调整门限。",
              "- 未自动判推进/静止，没有改动原响应或人工点标。", ""]
    (out/"README.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve previous probe; choose a fresh --out")
    cv2.setNumThreads(1)
    store = CorrespondenceStore(args.pack, args.raw_root)
    if tuple(c["ui_index"] for c in store.cases.values()) != CASES or (store.pack/"TEST_ONLY.json").exists():
        raise ValueError("only the existing six human cases are allowed")
    input_names = ("manifest.json", "annotations_v2.jsonl", "vessel_reference.jsonl", "interface_protocol_v2.json", "chat_confirmation_20260921_v1.json")
    inputs = {n: (store.pack/n).read_bytes() for n in input_names}
    args.out.mkdir(parents=True)
    for name in input_names:
        shutil.copyfile(store.pack/name, args.out/name)
    shutil.copyfile(__file__, args.out/"entrypoint_snapshot.py")
    protocol = {"schema": SCHEMA, "frozen_before_predictions_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": PARAMETERS, "cases": list(CASES), "source_pack": args.pack.as_posix(),
                "tracker_inputs": ["first-frame human point if visible", "current and previous grayscale original images", "initial patch"],
                "never_tracker_inputs": ["future human point/status", "response label", "task", "vessel reference", "other view", "robot state"],
                "confidence_semantics": "uncalibrated FB/appearance/texture gates, not visibility probability",
                "output": "latched null after first rejection; raw candidate remains separate diagnostic shadow",
                "geometry": "fixed start axis from four-point reference is OFFLINE comparison only; not centerline/mm/contact",
                "training_executed": False, "labels_modified": False, "policy_input_allowed": False,
                "formal_data_allowed": False, "real_system_validated": False}
    write_json(args.out/"protocol.json", protocol)
    predictions = []
    for case_id, case in store.cases.items():
        for view in VIEWS:
            # Do not pass a case/annotation/store into track: only images and the first point.
            first = store.annotations[case_id]["views"][view]["frames"][str(case["required_indices"][0])]["wire"]
            seed = first["xy"] if first["status"] == "visible" else None
            images = [cv2.imread(str(store.image_path(case_id, view, i)), cv2.IMREAD_GRAYSCALE)
                      for i in range(len(case["frames"][view]))]
            if any(im is None or im.shape != (1080, 1920) for im in images):
                raise ValueError("missing/non-original input image")
            t = time.perf_counter()
            result = track(images, seed)
            predictions.append({"case_id": case_id, "view": view, "seed_xy": seed,
                                "image_refs": case["frames"][view], "frames": result,
                                "tracking_seconds": time.perf_counter()-t})
    write_jsonl(args.out/"predictions.jsonl", predictions)
    # All tracker outputs are frozen before later human coordinates are consumed for evaluation.
    evaluation = evaluate(store, predictions)
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol, "input_summary": store.summary(), "opencv": cv2.__version__,
              "summary": summarize(evaluation, predictions), "evaluation": evaluation,
              "raw_images_available_on": "local Windows; remote check is synthetic, not a repeat of these captures",
              "visual_status": "not_viewed", "user_result_acceptance": "pending"}
    render(store, predictions, evaluation, args.out, args.font)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["input_bytes_unchanged"] = all((store.pack/n).read_bytes() == b for n, b in inputs.items())
    if not report["input_bytes_unchanged"]:
        raise RuntimeError("input annotation changed during probe; do not treat mixed revisions as one result")
    write_jsonl(args.out/"evaluation.jsonl", evaluation)
    write_json(args.out/"report.json", report)
    write_readme(report, args.out)
    cards = "\n".join(f'<section><h2>#{c["ui_index"]}</h2><a href="{c["id"]}.png"><img src="{c["id"]}.png"></a></section>' for c in store.cases.values())
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>六窗首帧锚定追踪</title>'
            '<style>body{font:18px system-ui;max-width:1500px;margin:25px auto;background:#f3f6f8;color:#203345}img{width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>首帧锚定追踪 / 拒判 · 固定轻量基线</h1><p>后续人工点只核对，不进入追踪。红色候选已拒判，不能当输出。</p>'
            '<a href="report.json">完整结果</a> · <a href="protocol.json">冻结口径</a><pre>'
            + html.escape((args.out/"README.md").read_text(encoding="utf-8")) + '</pre>' + cards + '</html>')
    (args.out/"index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": report["summary"],
                      "input_bytes_unchanged": report["input_bytes_unchanged"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    parser.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_seeded_wire_tracking_v1"))
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--check", action="store_true", help="One small translated/blank image and missing-seed check, no captures.")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check()))
    else:
        run(args)
