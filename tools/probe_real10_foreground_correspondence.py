"""Frozen six-window foreground/identity probe, without training or relabeling.

Reuses the prepared BootsTAPIR crops. Raw-template and background-suppressed
matching share a fixed first-frame template, ambiguity and round-trip checks.
The foreground arm uses the whole clip and is strictly an OFFLINE diagnostic.
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

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from probe_real10_seeded_wire_tracking import evaluate, stats, summarize
from review_real10_wire_correspondence import CASES, VIEWS, CorrespondenceStore


SCHEMA = "real10_foreground_correspondence_probe_v1"
MODES = ("raw_template", "foreground_anchor")
PARAMETERS = {"template_side_px": 25, "gaussian_sigma_px": 6.0,
              "background_percentile": 90, "noise_sigma_multiplier": 3.0,
              "noise_floor_gray": 2.0, "template_std_min_gray": 1.0,
              "foreground_support_min_pixels": 8, "foreground_nearest_max_px": 4.0,
              "ncc_min": .75, "separated_peak_gap_min": .05,
              "second_peak_exclusion_radius_px": 16, "roundtrip_max_px": 3.0,
              "extra_step_limit_px": None, "template_update": False,
              "geometric_registration": False, "display_residual_saturation_gray": 20.0}
INPUT_NAMES = ("manifest.json", "annotations_v2.jsonl", "vessel_reference.jsonl",
               "interface_protocol_v2.json", "chat_confirmation_20260921_v1.json")


def window_mask():
    radius = PARAMETERS["template_side_px"] // 2
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    return np.exp(-(xx*xx+yy*yy)/(2*PARAMETERS["gaussian_sigma_px"]**2)).astype(np.float32)


def patch(image, xy):
    side = PARAMETERS["template_side_px"]
    return cv2.getRectSubPix(image, (side, side), tuple(float(v) for v in xy))


def corrected_gray(rgb):
    gray = np.stack([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in rgb]).astype(np.float32)
    offsets = np.median((gray-gray[0]).reshape(len(gray), -1), axis=1)
    return gray-offsets[:, None, None], offsets


def foreground(gray):
    # Hypothesis: moving dark wire can leave a brighter observation at each pixel.
    # This also responds to other moving objects; it is NOT wire segmentation.
    differences = np.diff(gray, axis=0)
    sigma = float(np.median(np.abs(differences-np.median(differences))) / (.67448975*np.sqrt(2)))
    threshold = max(PARAMETERS["noise_floor_gray"], PARAMETERS["noise_sigma_multiplier"]*sigma)
    bg = np.percentile(gray, PARAMETERS["background_percentile"], axis=0).astype(np.float32)
    residual = np.maximum(bg[None]-gray-threshold, 0).astype(np.float32)
    return residual, {"noise_sigma_gray": sigma, "subtraction_threshold_gray": threshold}


def foreground_support(template):
    center = (PARAMETERS["template_side_px"]-1)/2
    yy, xx = np.mgrid[:template.shape[0], :template.shape[1]]
    distances = np.hypot(xx-center, yy-center)
    active = (template > 0) & (window_mask() >= .2)
    count = int(active.sum())
    nearest = float(distances[active].min()) if count else None
    return {"pixels": count, "nearest_px": nearest,
            "sufficient": count >= PARAMETERS["foreground_support_min_pixels"] and
                          nearest is not None and nearest <= PARAMETERS["foreground_nearest_max_px"]}


def locate(image, template):
    """Anchor appearance only: no motion prior and no previous-frame template update."""
    mask = window_mask()
    mean = float((template*mask).sum()/mask.sum())
    stdev = float(np.sqrt(((template-mean)**2*mask).sum()/mask.sum()))
    if stdev < PARAMETERS["template_std_min_gray"]:
        return {"xy": None, "score": None, "second_score": None, "gap": None, "template_std_gray": stdev}
    scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    # A masked zero-variance patch can give a FINITE spurious high score too.
    # Use the documented masked mean/energy, rather than only checking NaNs.
    mean_image = cv2.matchTemplate(image, mask, cv2.TM_CCORR)/mask.sum()
    mask2 = mask*mask
    energy = (cv2.matchTemplate(image*image, mask2, cv2.TM_CCORR)
              - 2*mean_image*cv2.matchTemplate(image, mask2, cv2.TM_CCORR)
              + mean_image*mean_image*mask2.sum())
    candidate_std = np.sqrt(np.maximum(energy, 0)/mask2.sum())
    valid = (np.isfinite(scores) & (scores >= -1.001) & (scores <= 1.001)
             & (candidate_std >= PARAMETERS["template_std_min_gray"]))
    if not valid.any():
        return {"xy": None, "score": None, "second_score": None, "gap": None, "template_std_gray": stdev}
    scores = np.where(valid, np.clip(scores, -1, 1), -np.inf)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    best = float(scores[y, x])
    yy, xx = np.ogrid[:scores.shape[0], :scores.shape[1]]
    separated = np.where((xx-x)**2+(yy-y)**2 > PARAMETERS["second_peak_exclusion_radius_px"]**2, scores, -np.inf)
    second = float(separated.max()) if np.isfinite(separated).any() else None
    radius = PARAMETERS["template_side_px"]//2
    return {"xy": [float(x+radius), float(y+radius)], "score": best, "second_score": second,
            "gap": best-second if second is not None else None, "template_std_gray": stdev,
            "candidate_std_gray": float(candidate_std[y, x])}


def track(images, seed, require_foreground):
    template = patch(images[0], seed)
    seed_support = foreground_support(template) if require_foreground else None
    rows, first_rejection = [], None
    for i, image in enumerate(images):
        if i == 0:
            rows.append({"index": 0, "candidate_xy": list(seed), "output_xy": list(seed), "state": "manual_seed",
                         "quality": {"seed_support": seed_support}, "step_reasons": [], "first_rejection": None})
            continue
        found = locate(image, template)
        reasons, roundtrip, reverse_score, current_support = [], None, None, None
        candidate = found["xy"]
        if candidate is None:
            reasons.append("template_or_candidate_uninformative")
        else:
            current = patch(image, candidate)
            back = locate(images[0], current)
            reverse_score = back["score"]
            if back["xy"] is not None:
                roundtrip = float(np.linalg.norm(np.asarray(back["xy"])-seed))
            if found["score"] < PARAMETERS["ncc_min"]:
                reasons.append("anchor_appearance_low")
            if found["gap"] is None or found["gap"] < PARAMETERS["separated_peak_gap_min"]:
                reasons.append("spatial_match_ambiguous")
            if roundtrip is None or roundtrip > PARAMETERS["roundtrip_max_px"]:
                reasons.append("anchor_roundtrip_failed")
            if require_foreground:
                current_support = foreground_support(current)
                if not current_support["sufficient"]:
                    reasons.append("candidate_foreground_insufficient")
        if require_foreground and not seed_support["sufficient"]:
            reasons.append("seed_foreground_insufficient")
        if reasons and first_rejection is None:
            first_rejection = {"index": i, "reasons": reasons.copy()}
        rows.append({"index": i, "candidate_xy": candidate, "output_xy": candidate if not reasons else None,
                     "state": "tracked" if not reasons else "abstained", "step_reasons": reasons,
                     "first_rejection": first_rejection,
                     "quality": {**{k: v for k, v in found.items() if k != "xy"},
                                 "best_xy_in_crop": candidate, "roundtrip_error_px": roundtrip, "reverse_score": reverse_score,
                                 "seed_support": seed_support, "candidate_support": current_support}})
    return rows


def missing(n):
    return [{"index": i, "candidate_xy": None, "output_xy": None, "state": "uninitialized",
             "quality": {}, "step_reasons": ["no_first_frame_seed"], "first_rejection": None} for i in range(n)]


def check():
    rng = np.random.default_rng(20260922)
    background = rng.uniform(180, 220, (240, 280)).astype(np.float32)
    shape = rng.uniform(30, 90, (17, 17)).astype(np.float32)
    images = []
    for x in (64, 64, 164):
        image = background.copy()
        image[112:129, x-8:x+9] = shape
        images.append(image)
    fg, _ = foreground(np.stack(images))
    rows = track(fg, [64., 120.], True)
    assert rows[-1]["output_xy"] is not None, rows[-1]
    error = float(np.linalg.norm(np.array(rows[-1]["output_xy"])-[164, 120]))
    assert error < 1
    blank = track(np.zeros((3, 240, 280), np.float32), [64., 120.], True)
    assert all(r["output_xy"] is None for r in blank[1:])
    assert all(r["candidate_xy"] is None for r in missing(3))
    return {"synthetic_100px_jump_error_px": error, "unchanged_foreground_unknown_not_zero_motion": True,
            "missing_seed_null": True, "opencv": cv2.__version__, "real_accuracy_claim": False}


def infer(args):
    if args.out.exists():
        raise FileExistsError("preserve previous/partial result; choose a fresh --out")
    cv2.setNumThreads(1)
    manifest = read_json(args.source / "inference_inputs/manifest.json")
    if tuple(dict.fromkeys(int(s["case_id"].split("_")[-1]) for s in manifest["sequences"])) != CASES:
        raise ValueError("only the same six windows")
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    shutil.copyfile(args.source / "inference_inputs/manifest.json", args.out / "input_manifest.json")
    protocol = {"schema": SCHEMA, "frozen_before_predictions_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(args.source), "parameters": PARAMETERS, "modes": list(MODES),
                "source_pixels": "unchanged 512x512 native RGB crops from prior BootsTAPIR preparation",
                "input": "first human query and image sequence only; grayscale conversion and spatial-median brightness correction",
                "foreground": "max(temporal percentile90 brightness - current brightness - robust noise threshold, 0)",
                "assumptions": ["camera/vessel approximately static", "wire locally darker than some observed background",
                                "fixed local appearance approximates identity under translation; no rotation/deformation invariance"],
                "foreground_is_wire_segmentation": False,
                "identity_constraints": "fixed 25px Gaussian-masked NCC template, separated second-peak gap, round-trip to initial query",
                "foreground_gate": "first and candidate query neighborhoods have sufficient residual pixels close to query",
                "never_input": ["later human coordinates/status", "task/response labels", "four-point reference",
                                "other view", "robot state", "contact/wall/route truth", "prior model predictions"],
                "causality": "foreground uses ALL clip frames including later images; raw arm needs only first/current frame",
                "abstention": "per-frame null; not stationary/invisible/contact; later frames may resume without a new query",
                "confidence": "heuristic scores/gates uncalibrated, fixed before inference; no threshold sweep",
                "comparison": "same crop/window/query, but foreground preprocessing has future-image information; diagnostic not causal equivalence",
                "references": ["https://docs.opencv.org/4.13.0/df/dfb/group__imgproc__object.html",
                               "https://docs.opencv.org/4.10.0/d1/dc5/tutorial_background_subtraction.html"],
                "training_executed": False, "labels_modified": False, "policy_input_allowed": False,
                "formal_data_allowed": False, "real_system_validated": False}
    write_json(args.out / "protocol.json", protocol)
    predictions = {m: [] for m in MODES}
    maps, support = {}, []
    start = time.perf_counter()
    with np.load(args.source / "inference_inputs/frames.npz", allow_pickle=False) as data:
        for item in manifest["sequences"]:
            n = len(item["image_refs"])
            metadata = {}
            if item["video_key"] is not None:
                rgb = data[item["video_key"]]
                gray, offsets = corrected_gray(rgb)
                fg, metadata = foreground(gray)
                seed = np.array(item["seed_xy"])-item["crop_xyxy"][:2]
                maps[item["video_key"]+"_first"] = fg[0]
                maps[item["video_key"]+"_last"] = fg[-1]
                metadata.update(case_id=item["case_id"], view=item["view"],
                                brightness_offsets_gray=offsets.tolist(), seed_support=foreground_support(patch(fg[0], seed)))
                support.append(metadata)
            for mode in MODES:
                rows = missing(n) if item["seed_xy"] is None else track(gray if mode == MODES[0] else fg, seed, mode == MODES[1])
                for row in rows:
                    for key in ("candidate_xy", "output_xy"):
                        if row[key] is not None:
                            row[key] = (np.array(row[key])+item["crop_xyxy"][:2]).tolist()
                    row["latest_input_frame_index"] = (row["index"] if mode == MODES[0] else n-1) if item["seed_xy"] is not None else None
                predictions[mode].append(dict(item, frames=rows, preprocessing=metadata))
            print(json.dumps({"case": item["case_id"], "view": item["view"], "finished": len(predictions[MODES[0]])}), flush=True)
    # Future human labels are unavailable in this inference stage and never read.
    for mode, rows in predictions.items():
        write_jsonl(args.out / f"predictions_{mode}.jsonl", rows)
    write_jsonl(args.out / "foreground_support.jsonl", support)
    np.savez_compressed(args.out / "diagnostic_maps.npz", **maps)
    info = {"schema": SCHEMA, "sequences_per_arm": len(predictions[MODES[0]]), "opencv": cv2.__version__,
            "elapsed_seconds": time.perf_counter()-start, "training_executed": False, "human_future_points_read": False,
            "foreground_clip_lookahead": True, "raw_prediction_missing_is_not_zero": True}
    write_json(args.out / "inference_report.json", info)
    print(json.dumps(info))


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def render(args, store, arms, rows):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(args.font), 22)
    small = ImageFont.truetype(str(args.font), 18)
    with np.load(args.source / "inference_inputs/frames.npz", allow_pickle=False) as images, np.load(args.out / "diagnostic_maps.npz", allow_pickle=False) as maps:
        for case_id, case in store.cases.items():
            canvas = Image.new("RGB", (1510, 1310), "#f3f6f8")
            d = ImageDraw.Draw(canvas)
            d.text((20, 15), f"#{case['ui_index']} · 首帧外观锚定 + 背景抑制（离线诊断）", font=font, fill="#203345")
            d.text((20, 48), "黄圈=人工点；紫+=普通模板候选；青×=前景输出；红×=前景拒判候选；灰圈=初始位置。", font=font, fill="#203345")
            for vi, view in enumerate(VIEWS):
                item = next(p for p in arms[MODES[1]] if p["case_id"] == case_id and p["view"] == view)
                raw = next(p for p in arms[MODES[0]] if p["case_id"] == case_id and p["view"] == view)
                ann = store.annotations[case_id]["views"][view]
                y, last = 95+580*vi, len(item["frames"])-1
                feature = "尖端" if ann["feature_type"] == "tip" else ann["feature_description"] or "无可靠点"
                d.text((20, y), f"{view.upper()} · {feature}", font=font, fill="#203345")
                if item["seed_xy"] is None:
                    d.text((20, y+40), "首帧无点：不初始化；不使用后来人工点，不记为可见性判断成功。", font=font, fill="#203345")
                    continue
                key, box = item["video_key"], item["crop_xyxy"]
                first_rgb, last_rgb = images[key][0], images[key][-1]
                fg = maps[key+"_last"]
                alpha = np.clip(fg/PARAMETERS["display_residual_saturation_gray"], 0, 1)*.8
                overlay = (last_rgb*(1-alpha[..., None])+np.array([0, 235, 240])*alpha[..., None]).astype(np.uint8)
                panels = [("首帧RGB / 黄色查询点", first_rgb, 0),
                          ("末帧变化支持（青色≠导丝分割）", overlay, last),
                          ("末帧候选与人工点核对", last_rgb, last)]
                for j, (label, rgb, index) in enumerate(panels):
                    x = 20+498*j
                    tile = Image.fromarray(rgb.copy()).resize((450, 450), Image.Resampling.LANCZOS)
                    mark = ImageDraw.Draw(tile)
                    human = ann["frames"][str(index)]["wire"]["xy"]
                    marks = [(item["seed_xy"], "#c4c4c4", "o"), (human, "#ffdf40", "o")]
                    if j == 2:
                        pred = item["frames"][index]
                        marks += [(raw["frames"][index]["candidate_xy"], "#db76ff", "+"),
                                  (pred["candidate_xy"], "#55ffff" if pred["output_xy"] is not None else "#ff5e77", "x")]
                    for point, color, kind in marks:
                        if point is None:
                            continue
                        px, py = (point[0]-box[0])*450/512, (point[1]-box[1])*450/512
                        if kind == "o":
                            mark.ellipse((px-6,py-6,px+6,py+6), outline=color, width=2)
                        elif kind == "+":
                            mark.line((px-7,py,px+7,py), fill=color, width=2)
                            mark.line((px,py-7,px,py+7), fill=color, width=2)
                        else:
                            mark.line((px-6,py-6,px+6,py+6), fill=color, width=2)
                            mark.line((px-6,py+6,px+6,py-6), fill=color, width=2)
                    d.text((x, y+32), label, font=small, fill="#203345")
                    canvas.paste(tile, (x+10, y+59))
                support = item["preprocessing"]["seed_support"]
                q = item["frames"][last]["quality"]
                ev = next(r for r in rows[MODES[1]] if r["case_id"] == case_id and r["view"] == view and r["is_endpoint"])
                d.text((20, y+517), f"初值支持 {support['pixels']}px；最近 {fmt(support['nearest_px'])}px；噪声扣除 {fmt(item['preprocessing']['subtraction_threshold_gray'])}", font=small, fill="#203345")
                d.text((518, y+517), f"NCC {fmt(q.get('score'))}；峰差 {fmt(q.get('gap'))}；往返 {fmt(q.get('roundtrip_error_px'))}px", font=small, fill="#203345")
                d.text((1016, y+517), f"前景候选/输出误差 {fmt(ev['raw_error_px'])}/{fmt(ev['output_error_px'])}px", font=small, fill="#203345")
                d.text((20, y+544), "末帧状态：" + ("输出（不等于身份已证实）" if item["frames"][last]["output_xy"] is not None else "无法确定；不解释为静止或不可见"), font=small, fill="#203345")
            d.text((20, 1275), "首帧模板不更新，无步长限制；全窗背景估计用到后续图像。六窗开发诊断，不作监督/碰壁标签。", font=font, fill="#203345")
            canvas.save(args.out / f"{case_id}.png")


def report(args):
    if (args.out / "report.json").exists():
        raise FileExistsError("preserve completed report")
    info = read_json(args.out / "inference_report.json")
    manifest = read_json(args.out / "input_manifest.json")
    store = CorrespondenceStore(args.source / "evaluation_source", args.raw_root)
    boots = read_jsonl(args.source / "result_v1/predictions.jsonl")
    boot_eval = evaluate(store, boots)
    prior = {(r["case_id"], r["view"], r["index"]): r for r in boot_eval}
    arms, evaluations, summaries = {}, {}, {}
    for mode in MODES:
        preds = read_jsonl(args.out / f"predictions_{mode}.jsonl")
        if len(preds) != len(manifest["sequences"]):
            raise ValueError("incomplete predictions")
        for pred, item, old in zip(preds, manifest["sequences"], boots):
            if any(pred[k] != item[k] or pred[k] != old[k] for k in ("case_id", "view", "seed_xy", "image_refs", "crop_xyxy")):
                raise ValueError("mismatched baseline/input identity")
        rows = evaluate(store, preds)
        for row in rows:
            row["bootstapir_error_px"] = prior[(row["case_id"], row["view"], row["index"])]["raw_error_px"]
        summary = summarize(rows, preds)
        for group, selected in (("all_evaluable_nonseed_keyframes", rows),
                                ("endpoints", [r for r in rows if r["is_endpoint"]]),
                                ("tip_only_endpoints", [r for r in rows if r["is_endpoint"] and r["feature_type"] == "tip"])):
            summary[group]["bootstapir_on_raw_subset"] = stats(r["bootstapir_error_px"] for r in selected if r["raw_error_px"] is not None)
            summary[group]["bootstapir_on_output_subset"] = stats(r["bootstapir_error_px"] for r in selected if r["output_error_px"] is not None)
        summary["all_nonseed_reasons"] = dict(Counter(reason for p in preds for f in p["frames"][1:] for reason in f["step_reasons"]))
        arms[mode], evaluations[mode], summaries[mode] = preds, rows, summary
        write_jsonl(args.out / f"evaluation_{mode}.jsonl", rows)
    result = {"schema": SCHEMA, "summary": summaries, "inference": info,
              "protocol": read_json(args.out / "protocol.json"), "visual_status": "not_viewed",
              "user_result_acceptance": "pending", "policy_input_allowed": False,
              "no_absent_foreground_implies_stationary": True,
              "input_snapshot_bytes_unchanged": all((args.pack/n).read_bytes() == (args.source/"evaluation_source"/n).read_bytes() for n in INPUT_NAMES)}
    lines = ["# 前景与首帧身份约束：固定六窗诊断", "", "没有训练、阈值搜索或改标。前景采用整窗后续图像，不是实时因果方法。",
             "变化区域不是导丝分割；无证据是unknown，不能标成静止/碰壁。", "",
             "| 同9个可评末点 | 原始候选数 | 候选偏差px | 同候选子集BootsTAPIR | 输出数 | 输出偏差px | 同输出子集BootsTAPIR | 同输出子集不动 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        g = summaries[mode]["endpoints"]
        lines.append(f"| {mode} | {g['raw_candidate_points']} | {fmt(g['raw_error']['mean_px'])} | {fmt(g['bootstapir_on_raw_subset']['mean_px'])} | {g['output_points']} | {fmt(g['output_error']['mean_px'])} | {fmt(g['bootstapir_on_output_subset']['mean_px'])} | {fmt(g['repeat_seed_on_output_subset']['mean_px'])} |")
    lines += ["", "候选和拒判后的输出分开；分母改变时只能按相同子集比较，不能把拒绝难点叫作精度提升。",
              "#4是材料交界处，其余可评点是尖端；不是毫米推进、接触或实机成功率。",
              "平移模板不具旋转/形变不变性；前景依赖近似静态相机和至少一次露出的较亮背景。", ""]
    (args.out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    render(args, store, arms, evaluations)
    write_json(args.out / "report.json", result)
    cards = "".join(f'<h2>#{c["ui_index"]}</h2><img src="{c["id"]}.png">' for c in store.cases.values())
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>导丝前景与身份诊断</title><style>body{max-width:1510px;margin:24px auto;font:18px system-ui;background:#f3f6f8}img{width:100%}pre{white-space:pre-wrap}</style><pre>'
    (args.out / "index.html").write_text(page+html.escape("\n".join(lines))+"</pre>"+cards+"</html>", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "endpoints": {m: summaries[m]["endpoints"] for m in MODES},
                      "input_snapshot_bytes_unchanged": result["input_snapshot_bytes_unchanged"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("check", "infer", "report"))
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_bootstapir_tracking_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_foreground_correspondence_v1"))
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    parser.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if args.stage == "check":
        print(json.dumps(check()))
    elif args.stage == "infer":
        infer(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
