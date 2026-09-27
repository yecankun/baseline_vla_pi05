"""Six-window BootsTAPIR adapter: prepare locally, infer remotely, report locally.

Only RGB crops and one first-frame query enter the pretrained tracker. This is
an offline diagnostic, not causal perception, automatic tip detection or training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from review_real10_wire_correspondence import CASES, VIEWS, CorrespondenceStore


SCHEMA = "real10_bootstapir_tracking_v1"
UPSTREAM = "https://github.com/google-deepmind/tapnet"
REVISION = "730cda1c730877cfedbe01bf87fb1cadb78a565d"
WEIGHT_URL = "https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt"
WEIGHT_BYTES = 218886140
WEIGHT_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1")
SNAPSHOTS = ("manifest.json", "annotations_v2.jsonl", "vessel_reference.jsonl",
             "interface_protocol_v2.json", "chat_confirmation_20260921_v1.json")
SIDE = 512


def crop_box(seed, image_wh=(1920, 1080)):
    """One fixed native-resolution square, determined from the initial query only."""
    x = max(0, min(image_wh[0] - SIDE, int(np.floor(seed[0] - SIDE / 2))))
    y = max(0, min(image_wh[1] - SIDE, int(np.floor(seed[1] - SIDE / 2))))
    return [x, y, x + SIDE, y + SIDE]


def local_query(seed, box):
    # Official TAPIR query order is time, y, x; predictions return x, y.
    return [0.0, float(seed[1] - box[1]), float(seed[0] - box[0])]


def unpack_predictions(tracks, occlusion_logits, expected_dist_logits, seed, box):
    """Keep raw candidates separate; per-frame abstention can subsequently recover."""
    visibility = 1 / (1 + np.exp(np.clip(occlusion_logits, -80, 80)))
    certainty = 1 / (1 + np.exp(np.clip(expected_dist_logits, -80, 80)))
    score = visibility * certainty
    rows, first_rejection = [], None
    for i, (xy, value) in enumerate(zip(tracks, score)):
        finite = bool(np.isfinite(xy).all() and np.isfinite(value))
        candidate = (xy + np.asarray(box[:2])).astype(float).tolist() if np.isfinite(xy).all() else None
        inside = bool(np.isfinite(xy).all() and 0 <= xy[0] < SIDE and 0 <= xy[1] < SIDE)
        reasons = []
        if not finite:
            reasons.append("nonfinite_model_output")
        elif value <= .5:
            reasons.append("official_visibility_uncertainty_gate")
        if not inside:
            reasons.append("outside_fixed_query_crop")
        if i == 0:
            # The user-specified initial query is not an automatic detection.
            output, state = list(seed), "manual_seed"
        else:
            output = candidate if not reasons else None
            state = "tracked" if output is not None else "abstained"
            if reasons and first_rejection is None:
                first_rejection = {"index": i, "reasons": reasons.copy()}
        quality = {"model_score_uncalibrated": float(value) if np.isfinite(value) else None,
                   "occlusion_logit": float(occlusion_logits[i]) if np.isfinite(occlusion_logits[i]) else None,
                   "expected_dist_logit": float(expected_dist_logits[i]) if np.isfinite(expected_dist_logits[i]) else None,
                   "inside_fixed_crop": inside}
        rows.append({"index": i, "candidate_xy": candidate, "output_xy": output, "state": state,
                     "quality": quality, "step_reasons": reasons if i else [],
                     "first_rejection": first_rejection,
                     "resumed_after_abstention": bool(i > 0 and output is not None and rows[-1]["output_xy"] is None)})
    return rows


def missing_frames(n):
    return [{"index": i, "candidate_xy": None, "output_xy": None, "state": "uninitialized",
             "quality": {}, "step_reasons": ["no_first_frame_seed"], "first_rejection": None,
             "resumed_after_abstention": False} for i in range(n)]


def prepare(args):
    from PIL import Image, ImageDraw, ImageFont
    if args.work.exists():
        raise FileExistsError("preserve prior inputs; choose a fresh --work")
    store = CorrespondenceStore(args.pack, args.raw_root)
    if tuple(c["ui_index"] for c in store.cases.values()) != CASES or (args.pack / "TEST_ONLY.json").exists():
        raise ValueError("only the same six human windows are allowed")
    started = time.perf_counter()
    inference = args.work / "inference_inputs"
    source = args.work / "evaluation_source"
    inference.mkdir(parents=True)
    source.mkdir()
    for name in SNAPSHOTS:
        shutil.copyfile(args.pack / name, source / name)
    shutil.copyfile(args.lk / "predictions.jsonl", source / "lk_predictions.jsonl")
    shutil.copyfile(args.lk / "report.json", source / "lk_report.json")
    shutil.copyfile(__file__, args.work / "entrypoint_snapshot.py")
    protocol = {"schema": SCHEMA, "frozen_before_predictions_utc": datetime.now(timezone.utc).isoformat(),
                "upstream": UPSTREAM, "source_revision": REVISION, "checkpoint_url": WEIGHT_URL,
                "checkpoint_bytes": WEIGHT_BYTES, "cases": list(CASES),
                "model": "BootsTAPIR v2 PyTorch, pyramid_level=1, extra_convs=True, use_casual_conv=False",
                "inference": "offline full observed window; future images relative to intermediate frames ARE used",
                "input": ["RGB uint8 -> [-1,1]", "first-frame human query [t,y,x]"],
                "never_model_input": ["future human coordinates/status", "task/response labels", "four-point vessel reference",
                                      "other view", "robot state", "contact/wall/route truth"],
                "crop": "512x512 native-pixel square centered on first query, clamped to image; fixed across window",
                "crop_caveat": "outside-crop points abstain; this is not full-image re-detection",
                "resize": "none in adapter; official model uses 256 initialization + explicit 512 refinement",
                "no_first_query": "all frames uninitialized; no later human point used for re-seeding",
                "gating": "official (1-sigmoid(occlusion))*(1-sigmoid(expected_dist)) > 0.5; in-crop finite point",
                "score_semantics": "uncalibrated model score, not confirmed same-point identity or clinical confidence",
                "abstention": "per frame, non-latched; later valid prediction can resume without new human query",
                "added_step_limit_px": None, "added_smoothing_or_interpolation": False,
                "timestamps": "retained for timing report, not an input to TAPIR; irregular spacing not modeled",
                "seed": 20260922, "causal": False, "policy_input_allowed": False,
                "training_executed": False, "labels_modified": False, "formal_data_allowed": False,
                "real_system_validated": False, "frozen_lk_baseline": str(args.lk),
                "comparison_limit": "LK was causal full-frame grayscale; this probe is offline RGB query-crop. Not an architecture-only ablation."}
    write_json(inference / "protocol.json", protocol)
    arrays, sequences = {}, []
    for case_id, case in store.cases.items():
        for view in VIEWS:
            first = store.annotations[case_id]["views"][view]["frames"]["0"]["wire"]
            seed = first["xy"] if first["status"] == "visible" else None
            box = crop_box(seed) if seed is not None else None
            key = f"{case_id}_{view}"
            if box is not None:
                images = []
                for i in range(len(case["frames"][view])):
                    with Image.open(store.image_path(case_id, view, i)) as image:
                        if image.size != (1920, 1080):
                            raise ValueError("expected original 1920x1080 images")
                        images.append(np.asarray(image.convert("RGB").crop(box)))
                arrays[key] = np.stack(images)
            sequences.append({"case_id": case_id, "view": view, "seed_xy": seed,
                              "image_refs": case["frames"][view], "crop_xyxy": box,
                              "video_key": key if seed is not None else None,
                              "query_tyx": local_query(seed, box) if seed is not None else None})
    np.savez_compressed(inference / "frames.npz", **arrays)
    write_json(inference / "manifest.json", {"schema": SCHEMA, "original_image_wh": [1920, 1080], "sequences": sequences})
    # Prepared-image check, not a prediction. The final frame never determines the crop.
    key = "human_077_top"
    canvas = Image.new("RGB", (1064, 634), "#f3f6f8")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(args.font), 23)
    draw.text((20, 15), "#77 TOP · 冻结输入裁剪：只由首帧人工点确定512像素方框", font=font, fill="#203345")
    item = next(s for s in sequences if s["video_key"] == key)
    for j, frame in enumerate((arrays[key][0], arrays[key][-1])):
        tile = Image.fromarray(frame)
        if j == 0:
            q = item["query_tyx"]
            ImageDraw.Draw(tile).ellipse((q[2]-6, q[1]-6, q[2]+6, q[1]+6), outline="#ffdf40", width=2)
        canvas.paste(tile, (20 + 520*j, 85))
        draw.text((20+520*j, 52), "首帧：黄色圈为人工查询点" if j == 0 else "末帧：原图裁剪，无预测/人工末点", font=font, fill="#203345")
    draw.text((20, 603), "不缩放、不补帧；图片不是模型输出，追踪效果尚待真实权重运行。", font=font, fill="#203345")
    canvas.save(args.work / "input_preview.png")
    result = {"schema": SCHEMA, "sequences": len(sequences), "initialized_sequences": len(arrays),
              "frame_slots": sum(len(s["image_refs"]) for s in sequences),
              "model_input_frames": sum(len(a) for a in arrays.values()),
              "compressed_input_bytes": (inference / "frames.npz").stat().st_size,
              "source_snapshots_unchanged": all((args.pack/n).read_bytes() == (source/n).read_bytes() for n in SNAPSHOTS),
              "model_inference_executed": False, "visual_status": "not_viewed", "elapsed_seconds": time.perf_counter()-started}
    write_json(args.work / "preparation_report.json", result)
    print(json.dumps(result, ensure_ascii=False))


def import_model(args):
    revision = subprocess.check_output(["git", "-C", str(args.tapnet), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"use the frozen upstream revision {REVISION}, got {revision}")
    sys.path.insert(0, str(args.tapnet.resolve()))
    from tapnet.torch.tapir_model import TAPIR
    return TAPIR


def check(args):
    """One adapter contract check; the random-weight forward proves shapes only."""
    import torch
    seed = [1092.1, 739.9]
    box = crop_box(seed)
    q = local_query(seed, box)
    assert np.allclose([q[2]+box[0], q[1]+box[1]], seed)
    rows = unpack_predictions(np.array([[256., 256.], [256., 256.], [356., 256.]]),
                              np.array([-10., 10., -10.]), np.array([-10., -10., -10.]), seed, box)
    assert rows[1]["output_xy"] is None and rows[2]["resumed_after_abstention"]
    assert abs(rows[2]["output_xy"][0] - rows[0]["candidate_xy"][0] - 100) < 1e-6
    assert all(r["output_xy"] is None for r in missing_frames(3))
    torch.manual_seed(20260922)
    model = import_model(args)(pyramid_level=1).eval().to(args.device)
    with torch.inference_mode():
        outputs = model(torch.zeros((1, 2, 256, 256, 3), device=args.device),
                        torch.tensor([[[0., 128., 128.]]], device=args.device),
                        refinement_resolutions=[(256, 256)])
    assert tuple(outputs["tracks"].shape) == (1, 1, 2, 2)
    assert torch.isfinite(outputs["tracks"]).all()
    result = {"coordinate_roundtrip": True, "null_then_resume_100px_without_added_step_gate": True,
              "missing_seed_stays_null": True, "official_forward_shapes_valid": True,
              "random_weights_only_no_accuracy_claim": True, "pretrained_weights_loaded": False,
              "source_revision": REVISION, "torch": torch.__version__, "device": args.device}
    print(json.dumps(result))


def infer(args):
    import torch
    inp = args.work / "inference_inputs"
    protocol, manifest = read_json(inp / "protocol.json"), read_json(inp / "manifest.json")
    if protocol["source_revision"] != REVISION or manifest["schema"] != SCHEMA:
        raise ValueError("input pack does not match the frozen adapter")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size != WEIGHT_BYTES:
        raise FileNotFoundError(f"download the official {WEIGHT_BYTES}-byte checkpoint first: {args.checkpoint}")
    if args.result.exists():
        raise FileExistsError("preserve previous/partial predictions; choose a fresh --result")
    torch.manual_seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    model = import_model(args)(pyramid_level=1).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(args.device)
    args.result.mkdir(parents=True)
    shutil.copyfile(inp / "protocol.json", args.result / "protocol.json")
    shutil.copyfile(__file__, args.result / "entrypoint_snapshot.py")
    started, predictions = time.perf_counter(), []
    with np.load(inp / "frames.npz", allow_pickle=False) as arrays, (args.result / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for item in manifest["sequences"]:
            t = time.perf_counter()
            if item["seed_xy"] is None:
                rows = missing_frames(len(item["image_refs"]))
            else:
                rgb = arrays[item["video_key"]]
                if rgb.shape != (len(item["image_refs"]), SIDE, SIDE, 3) or rgb.dtype != np.uint8:
                    raise ValueError("input must be the frozen native-pixel RGB sequence")
                video = torch.from_numpy(rgb).float().to(args.device)[None] / 127.5 - 1
                query = torch.tensor([[item["query_tyx"]]], device=args.device, dtype=torch.float32)
                with torch.inference_mode():
                    out = model(video, query, refinement_resolutions=[(SIDE, SIDE)])
                rows = unpack_predictions(out["tracks"][0, 0].cpu().numpy(),
                                          out["occlusion"][0, 0].cpu().numpy(),
                                          out["expected_dist"][0, 0].cpu().numpy(),
                                          item["seed_xy"], item["crop_xyxy"])
            for row in rows:
                row["latest_input_frame_index"] = len(rows)-1 if item["seed_xy"] is not None else None
                row["image_lookahead_s"] = item["image_refs"][-1]["time_s"] - item["image_refs"][row["index"]]["time_s"]
            record = dict(item, frames=rows, tracking_seconds=time.perf_counter()-t, causal=False)
            predictions.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            print(json.dumps({"done": len(predictions), "of": len(manifest["sequences"]),
                              "sequence": item["video_key"], "seconds": record["tracking_seconds"]}), flush=True)
    report = {"schema": SCHEMA, "sequences": len(predictions), "elapsed_seconds": time.perf_counter()-started,
              "source_revision": REVISION, "checkpoint_path": str(args.checkpoint),
              "checkpoint_bytes": args.checkpoint.stat().st_size, "torch": torch.__version__,
              "device": args.device, "causal": False, "model_inference_executed": True,
              "evaluation_executed": False, "training_executed": False,
              "policy_input_allowed": False, "visual_status": "not_viewed"}
    write_json(args.result / "inference_report.json", report)
    print(json.dumps(report))


def render(store, predictions, evaluations, lk, result, font_path):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), 23)
    small = ImageFont.truetype(str(font_path), 18)
    def fmt(x):
        return "—" if x is None else f"{x:.2f}"
    for case_id, case in store.cases.items():
        canvas = Image.new("RGB", (1510, 1320), "#f3f6f8")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), f"#{case['ui_index']} · BootsTAPIR离线同点追踪（不是实时部署结果）", font=font, fill="#203345")
        draw.text((20, 52), "黄圈=人工点；青×=模型输出；红×=被拒判候选；紫+=旧LK候选；灰圈=不动对照。", font=font, fill="#203345")
        for vi, view in enumerate(VIEWS):
            item = next(p for p in predictions if p["case_id"] == case_id and p["view"] == view)
            old = next(p for p in lk if p["case_id"] == case_id and p["view"] == view)
            ann = store.annotations[case_id]["views"][view]
            y = 100 + 588*vi
            feature = "尖端" if ann["feature_type"] == "tip" else ann["feature_description"] or "无初值"
            draw.text((20, y), f"{view.upper()} · {feature} · " + ("固定首帧查询裁剪" if item["seed_xy"] else "无首帧点：全窗不初始化"), font=font, fill="#203345")
            for j, i in enumerate(case["keyframe_indices"]):
                x, pred = 20+498*j, item["frames"][i]
                target = ann["frames"][str(i)]["wire"]
                box = item["crop_xyxy"] or [0, 0, 1920, 1080]
                with Image.open(store.image_path(case_id, view, i)) as image:
                    tile = image.convert("RGB").crop(box)
                scale = min(472/tile.width, 450/tile.height)
                tile = tile.resize((round(tile.width*scale), round(tile.height*scale)), Image.Resampling.LANCZOS)
                marks = ImageDraw.Draw(tile)
                for point, color, kind in ((item["seed_xy"], "#c4c4c4", "o"),
                                           (old["frames"][i]["candidate_xy"], "#db76ff", "+"),
                                           (target["xy"], "#ffdf40", "o"),
                                           (pred["candidate_xy"], "#55ffff" if pred["output_xy"] else "#ff5e77", "x")):
                    if point is None:
                        continue
                    px, py = (point[0]-box[0])*scale, (point[1]-box[1])*scale
                    if kind == "o":
                        marks.ellipse((px-6,py-6,px+6,py+6), outline=color, width=2)
                    elif kind == "+":
                        marks.line((px-7,py,px+7,py), fill=color, width=2)
                        marks.line((px,py-7,px,py+7), fill=color, width=2)
                    else:
                        marks.line((px-6,py-6,px+6,py+6), fill=color, width=2)
                        marks.line((px-6,py+6,px+6,py-6), fill=color, width=2)
                canvas.paste(tile, (x+(472-tile.width)//2, y+58+(450-tile.height)//2))
                state = {"manual_seed": "人工初值", "tracked": "输出", "abstained": "拒判", "uninitialized": "未初始化"}[pred["state"]]
                draw.text((x, y+31), f"帧{i} · {state} · 人工{target['status']}", font=small, fill="#203345")
                ev = next((r for r in evaluations if r["case_id"] == case_id and r["view"] == view and r["index"] == i), None)
                label = "初值不计入误差" if ev is None else f"候选/输出/不动偏差 {fmt(ev['raw_error_px'])}/{fmt(ev['output_error_px'])}/{fmt(ev['repeat_seed_error_px'])}px"
                draw.text((x, y+514), label, font=small, fill="#203345")
                draw.text((x, y+540), f"模型分数 {fmt(pred['quality'].get('model_score_uncalibrated'))}（未标定）", font=small, fill="#203345")
        draw.text((20, 1285), "仅6个开发难例；整段图像离线输入，无训练、自动标签或控制。紫色旧LK使用的输入口径不同。", font=font, fill="#203345")
        canvas.save(result / f"{case_id}.png")


def report(args):
    from probe_real10_seeded_wire_tracking import evaluate, summarize
    if (args.result / "report.json").exists():
        raise FileExistsError("report already exists; preserve the completed result")
    info = read_json(args.result / "inference_report.json")  # Require completed inference, not a partial file.
    manifest = read_json(args.work / "inference_inputs/manifest.json")
    predictions = read_jsonl(args.result / "predictions.jsonl")
    if len(predictions) != len(manifest["sequences"]) or info["sequences"] != len(predictions):
        raise ValueError("incomplete predictions")
    for pred, item in zip(predictions, manifest["sequences"]):
        if any(pred[k] != item[k] for k in ("case_id", "view", "seed_xy", "image_refs", "crop_xyxy")):
            raise ValueError("predictions belong to a different prepared input")
    store = CorrespondenceStore(args.work / "evaluation_source", args.raw_root)
    lk = read_jsonl(args.work / "evaluation_source/lk_predictions.jsonl")
    for current, previous in zip(predictions, lk):
        if any(current[k] != previous[k] for k in ("case_id", "view", "seed_xy", "image_refs")):
            raise ValueError("LK comparison uses different seeds/frames")
    rows = evaluate(store, predictions)
    baseline = summarize(evaluate(store, lk), lk)
    summary = summarize(rows, predictions)
    paired = [r for r in rows if r["point_error_eligible"] and r["raw_error_px"] is not None]
    old_rows = {(r["case_id"], r["view"], r["index"]): r for r in evaluate(store, lk)}
    for r in paired:
        r["frozen_lk_raw_error_px"] = old_rows[(r["case_id"], r["view"], r["index"])]["raw_error_px"]
    result = {"schema": SCHEMA, "summary": summary, "frozen_lk_summary": baseline, "inference": info,
              "evaluation": rows, "protocol": read_json(args.result / "protocol.json"),
              "visual_status": "not_viewed", "user_result_acceptance": "pending",
              "warning": "Offline RGB crop vs causal grayscale LK is a mechanism probe, not an architecture-only comparison."}
    write_jsonl(args.result / "evaluation.jsonl", rows)
    lines = ["# 六窗BootsTAPIR离线追踪", "", result["warning"], "",
             "| 范围 | 可评点 | 输出点 | 候选偏差px | 输出偏差px | 同输出子集不动px | 旧LK候选px |", "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "—" if x is None else f"{x:.3f}"
    for key, name in (("all_evaluable_nonseed_keyframes", "中末帧"), ("endpoints", "末帧"), ("tip_only_endpoints", "尖端末帧")):
        g, b = summary[key], baseline[key]
        lines.append(f"| {name} | {g['human_evaluable_points']} | {g['output_points']} | {fmt(g['raw_error']['mean_px'])} | {fmt(g['output_error']['mean_px'])} | {fmt(g['repeat_seed_on_output_subset']['mean_px'])} | {fmt(b['raw_error']['mean_px'])} |")
    lines += ["", "原图像素误差，不是毫米推进/接触。#4为材料特征，其他可评点为尖端。",
              "首帧缺失不初始化不等于自动识别不可见；unreviewed不作为不可见真值。",
              "模型分数未在导丝图像上标定；拒判后恢复输出不代表已确认恢复同点身份。",
              "人工中末点只在全部预测落盘后用于本报告。没有改标、训练或自动接入世界模型。", ""]
    (args.result / "README.md").write_text("\n".join(lines), encoding="utf-8")
    render(store, predictions, rows, lk, args.result, args.font)
    write_json(args.result / "report.json", result)
    cards = "".join(f'<h2>#{c["ui_index"]}</h2><img src="{c["id"]}.png">' for c in store.cases.values())
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>BootsTAPIR六窗诊断</title><style>body{max-width:1510px;margin:24px auto;background:#f3f6f8;font:18px system-ui}img{width:100%}pre{white-space:pre-wrap}</style><pre>'
    (args.result / "index.html").write_text(page+html.escape("\n".join(lines))+"</pre>"+cards+"</html>", encoding="utf-8")
    print(json.dumps({"result": str(args.result), "summary": summary}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "check", "infer", "report"), required=True)
    p.add_argument("--work", type=Path, default=Path("simulation_output/real10_bootstapir_tracking_v1"))
    p.add_argument("--result", type=Path, help="Default: WORK/result_v1. A fresh directory is required for inference.")
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    p.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    p.add_argument("--lk", type=Path, default=Path("simulation_output/real10_seeded_wire_tracking_v1"))
    p.add_argument("--tapnet", type=Path, default=WEIGHT_ROOT / "third_party/tapnet_730cda1")
    p.add_argument("--checkpoint", type=Path, default=WEIGHT_ROOT / "point_tracking/bootstapir_checkpoint_v2.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = p.parse_args()
    args.result = args.result or args.work / "result_v1"
    {"prepare": prepare, "check": check, "infer": infer, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
