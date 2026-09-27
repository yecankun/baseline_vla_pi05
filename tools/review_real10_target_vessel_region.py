"""Read-only target-vessel envelope proposal, not a new feature/model experiment.

prepare: freeze small review index on project4090. render: use original RGB and
the old calibration masks on Windows; never write labels, features or weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path
import time
from urllib.parse import urlencode

import cv2
import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json


SCHEMA = "real10_target_vessel_region_review_v1"
VIEWS = ("side", "top")
# Hand-drawn generous envelopes on the SAME unlabeled calibration RGB used in
# v1, in its 720-wide ROI coordinates. Both branches retained for both tasks.
# These are review proposals, not wire labels, precise lumen masks or deployed ROIs.
POLYGONS = {
    "side": [[0,14],[50,14],[110,35],[210,56],[300,60],[390,70],[475,78],
             [555,101],[610,135],[655,168],[719,168],[719,240],[665,230],
             [600,203],[570,170],[505,148],[460,130],[390,119],[330,112],
             [275,115],[218,143],[178,153],[138,143],[90,123],[45,105],[0,94]],
    "top": [[0,195],[50,165],[100,141],[150,115],[220,89],[290,75],[360,73],
            [416,61],[444,37],[474,29],[520,41],[575,37],[636,17],[654,26],
            [617,52],[660,72],[693,89],[690,103],[647,90],[592,72],[540,84],
            [480,91],[431,111],[399,123],[371,132],[348,128],[320,124],
            [285,120],[240,126],[202,148],[154,167],[113,190],[72,219],
            [36,247],[0,259]],
}
FOCUS = (10, 12, 13, 24, 39, 44, 65)
# Approximate visual callouts only, never fed into the proposed mask.
CALLOUTS = {39: ("side", [0,0,215,92]), 65: ("top", [205,25,330,115])}


def prepare(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("keep existing review; do not overwrite a frozen proposal")
    source = read_json(args.reference / "protocol.json")
    rows = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    if source["schema"] != "real10_vessel_relative_local_motion_v1" or len(rows) != 45:
        raise ValueError("expected the current frozen 45-window local-motion version")
    roi = source["parameters"]["roi_xyxy_original"]
    width = source["parameters"]["width"]
    sizes = {v: [width, round((roi[v][3]-roi[v][1]) * width / (roi[v][2]-roi[v][0]))] for v in VIEWS}
    for view in VIEWS:
        p = np.asarray(POLYGONS[view])
        if not ((p >= 0).all() and (p[:, 0] < sizes[view][0]).all() and (p[:, 1] < sizes[view][1]).all()):
            raise ValueError("polygon outside frozen ROI pixel coordinates")
    groups = {}
    for row in rows:
        groups.setdefault(row["source_episode"], []).append(row)
    if len(groups) != 10:
        raise ValueError("expected the same ten source episodes")
    pairs = []
    for episode, group in sorted(groups.items()):
        group.sort(key=lambda r: r["anchor_frame_id"])
        pairs.append({"id": "episode_" + group[0]["task"] + "_" + episode[-3:], "kind": "episode",
            "task": group[0]["task"], "source_episode": episode,
            "first": group[0]["anchor_frame_id"], "last": group[-1]["future_frame_ids"][-1],
            "first_window": group[0]["ui_index"], "last_window": group[-1]["ui_index"]})
    for row in rows:
        if row["ui_index"] in FOCUS:
            pairs.append({"id": f"window_{row['ui_index']:03d}", "kind": "window", "ui_index": row["ui_index"],
                "task": row["task"], "source_episode": row["source_episode"],
                "first": row["anchor_frame_id"], "last": row["future_frame_ids"][-1]})
    frame_ids = {source["calibration_frame"]} | {p[k] for p in pairs for k in ("first", "last")}
    index = {f: {v: observations[f]["images"][v]["path"] for v in VIEWS} for f in sorted(frame_ids)}
    protocol = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": args.reference.as_posix(), "calibration_frame": source["calibration_frame"],
        "original_image_wh": [1920,1080], "existing_roi_xyxy": roi, "review_roi_wh": sizes,
        "proposal_polygon_xy_review_roi": POLYGONS,
        "coordinate_mapping": "original_xy = roi_left_top + review_xy * (roi_width / 720); raster pixels, not physical coordinates",
        "candidate_definition": "old frozen interior AND proposed envelope, for display/area accounting only; not applied to feature extractor",
        "reference_rim": "unchanged in existing model; this review does not modify motion compensation",
        "polygon_source": "manual broad foreground-vessel outline from left002 frame000000 RGB; both branches included; same per-view proposal for every task/episode",
        "review_selection": "earliest selected anchor and latest selected future per episode, plus seven prior failure-localization windows; not full-episode or all-frame coverage",
        "not_blind": "previous errors already inspected; neither this geometry review nor the reused 45 windows are independent final validation",
        "no_response_labels_or_scores_loaded": "only identity/frame metadata read from frozen rows; no OOF score file or live annotation file loaded",
        "callout_boxes_review_only": {str(k): {"view": v, "box_xyxy": b} for k, (v,b) in CALLOUTS.items()},
        "forbidden_claims": ["wire segmentation recall", "precise lumen", "all occlusion removed", "physical advance/contact", "classifier gain"],
        "proposal_status": "draft_not_user_accepted", "features_modified": False, "labels_modified": False,
        "training_executed": False, "policy_input_allowed": False, "formal_data_allowed": False}
    args.out.mkdir(parents=True)
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "review_index.json", {"pairs": pairs, "frames": index})
    print({"schema": SCHEMA, "episodes": len(groups), "pairs": len(pairs), "unique_source_frames": len(index),
           "training_executed": False, "source_images_read": False}, flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    started = time.perf_counter()
    protocol = read_json(args.out / "protocol.json")
    index = read_json(args.out / "review_index.json")
    if protocol["schema"] != SCHEMA:
        raise ValueError("incompatible review protocol")
    polygons = protocol["proposal_polygon_xy_review_roi"]
    roi = protocol["existing_roi_xyxy"]
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
    masks, calibration, statistics = {}, {}, {}
    with np.load(args.reference / "preview_arrays.npz", allow_pickle=False) as cache:
        for view in VIEWS:
            original = cache[f"interior_{view}"]
            calibration[view] = cv2.cvtColor(cache[f"calibration_{view}"], cv2.COLOR_BGR2RGB)
            w, h = protocol["review_roi_wh"][view]
            envelope = np.zeros((h, w), np.uint8)
            cv2.fillPoly(envelope, [np.asarray(polygons[view], dtype=np.int32)], 1)
            kept, removed = original & (envelope > 0), original & (envelope == 0)
            masks[view] = {"old": original, "envelope": envelope > 0, "kept": kept, "removed": removed}
            statistics[view] = {"old_interior_pixels": int(original.sum()), "kept_pixels": int(kept.sum()),
                "removed_pixels": int(removed.sum()), "kept_fraction_of_old_mask": float(kept.sum()/original.sum()),
                "removed_fraction_of_old_mask": float(removed.sum()/original.sum()),
                "warning": "pixel-area accounting, NOT guidewire recall/precision or motion-feature importance"}
    np.savez_compressed(args.out / "review_only_masks.npz", **{f"{v}_{k}": m for v, ms in masks.items() for k, m in ms.items()})
    cache = {}
    def source(frame, view):
        key = (frame, view)
        if key not in cache:
            path = args.source_root / index["frames"][frame][view]
            with Image.open(path) as im:
                if im.size != (1920, 1080):
                    raise ValueError(f"not original RGB: {path}")
                bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            x0,y0,x1,y1 = roi[view]
            rgb = cv2.cvtColor(cv2.resize(bgr[y0:y1,x0:x1], tuple(protocol["review_roi_wh"][view]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            cache[key] = rgb
        return cache[key]
    def overlay(rgb, view, old=False, box=None):
        result = rgb.copy()
        layers = [(masks[view]["old"], [40,205,225], .28)] if old else [
            (masks[view]["kept"], [25,220,110], .17), (masks[view]["removed"], [245,45,65], .43)]
        for selected, color, weight in layers:
            result[selected] = ((1-weight)*result[selected] + weight*np.asarray(color)).astype(np.uint8)
        if not old:
            cv2.polylines(result, [np.asarray(polygons[view], np.int32)], True, (10,245,120), 2)
        if box is not None:
            cv2.rectangle(result, tuple(box[:2]), tuple(box[2:]), (255,175,0), 2)
        return Image.fromarray(result)

    overview = Image.new("RGB", (2240, 865), "#f5f6f8")
    d = ImageDraw.Draw(overview)
    d.text((20,15), "固定目标区域草案：先排除旁侧模型，不把磁铁重叠区硬遮掉", font=title, fill="#202939")
    d.text((20,60), "绿色：拟保留的原颜色候选像素与包络线；红色：拟排除。包络不是精确管腔或导丝标签。", font=font, fill="#202939")
    for j, view in enumerate(VIEWS):
        y = 112 + j*350
        for k, (name, im) in enumerate((("原图固定ROI", Image.fromarray(calibration[view])),
                                       ("原颜色掩膜（青色）", overlay(calibration[view], view, old=True)),
                                       ("区域草案：保留绿 / 排除红", overlay(calibration[view], view)))):
            x = 20+k*740
            d.text((x,y), f"{view} · {name}", font=font, fill="#202939")
            overview.paste(im, (x,y+35))
        st = statistics[view]
        d.text((20,y+305), f"{view}：原颜色候选像素拟保留 {st['kept_fraction_of_old_mask']:.2%}，拟排除 {st['removed_fraction_of_old_mask']:.2%}；不是导丝召回率。", font=font, fill="#202939")
    d.text((20,817), "两任务/全部轨迹共用此草案；未修改旧掩膜、LK参照带、特征、人工标签或模型。", font=font, fill="#9b3c21")
    overview.save(args.out / "calibration_regions.jpg", quality=94)

    # Exact calibrated RGB cross-check, without new feature extraction or fitting.
    for view in VIEWS:
        np.testing.assert_array_equal(source(protocol["calibration_frame"], view), calibration[view])
    for p in index["pairs"]:
        sheet = Image.new("RGB", (1500, 835), "#f5f6f8")
        d = ImageDraw.Draw(sheet)
        task = "左" if p["task"] == "left" else "右"
        name = f"第{p['ui_index']}窗" if p["kind"] == "window" else f"{task}任务 / 轨迹{p['source_episode'][-3:]}"
        d.text((20,12), f"{name}：相同固定包络，不是运动响应重标", font=title, fill="#202939")
        d.text((20,56), "绿=拟保留；红=拟排除；橙框仅提示器械/血管邻接或重叠风险，不参与候选掩膜。", font=small, fill="#202939")
        note = "首帧冻结掩膜的固定位置，不是当前帧物体检测；移动器械可经过保留区，或留下空位置色块。"
        if p["kind"] == "episode":
            note = "已选窗口的最早锚点/最晚末帧，非全轨迹首末；色块是首帧固定位置，非逐帧物体检测。"
        d.text((20,85), note, font=small, fill="#9b3c21")
        for j, view in enumerate(VIEWS):
            y = 123 + j*337
            for k, field in enumerate(("first", "last")):
                frame = p[field]
                box = CALLOUTS.get(p.get("ui_index"))
                im = overlay(source(frame,view), view, box=box[1] if box and box[0] == view else None)
                x = 20+k*740
                d.text((x,y), f"{view} · {'起' if field == 'first' else '末'} · 原帧 {frame.rsplit('/',1)[1]}", font=font, fill="#202939")
                sheet.paste(im, (x,y+35))
        d.text((20,793), "请检查：是否漏掉目标血管/导丝？红区是否只含非目标结构？磁铁重叠不能等同于可直接删除。", font=small, fill="#9b3c21")
        sheet.save(args.out / (p["id"]+".jpg"), quality=93)

    # Readable trajectory coverage contact sheets, with full-resolution links on page.
    for task in ("left", "right"):
        pairs = [p for p in index["pairs"] if p["kind"] == "episode" and p["task"] == task]
        sheet = Image.new("RGB", (1500, 1050), "#f5f6f8")
        d = ImageDraw.Draw(sheet)
        d.text((20,12), ("左" if task == "left" else "右") + "任务五条轨迹：固定区域首末抽查", font=title, fill="#202939")
        d.text((20,56), "每行：Side起 / Side末 / Top起 / Top末；色块来自首帧固定掩膜，不是当前帧物体检测。", font=small, fill="#202939")
        for j, p in enumerate(pairs):
            y = 100+j*185
            d.text((20,y), f"轨迹{p['source_episode'][-3:]} · 已选窗{p['first_window']}—{p['last_window']}", font=small, fill="#202939")
            for k,(view,field) in enumerate((('side','first'),('side','last'),('top','first'),('top','last'))):
                im = overlay(source(p[field],view),view)
                sheet.paste(im.resize((360, im.height//2), Image.Resampling.LANCZOS), (20+k*370,y+31))
        sheet.save(args.out / f"coverage_{task}.jpg", quality=94)

    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>目标血管区域只读审查</title>",
        "<style>body{font:18px 'Microsoft YaHei',sans-serif;max-width:1550px;margin:24px auto;background:#f5f6f8;color:#202939}img{max-width:100%;height:auto}section,details{background:white;margin:20px 0;padding:16px}summary{cursor:pointer}a{color:#175ea8}.warning{color:#9b3c21}</style>",
        "<h1>目标血管区域：只读草案，尚未应用于模型</h1>",
        "<p>重点确认：绿色包络是否保留两条目标分支及导丝所在区域？红色拟删除区域是否只含旁侧模型/器械？</p>",
        "<p class='warning'>固定区域能排除空间分离的干扰，却不能消除器械与血管同像素重叠。绿色不表示导丝真值，红色不表示逐像素人工认证。</p>",
        "<p>红绿区域是校准首帧冻结的颜色掩膜位置，不是逐帧物体检测。器械离开后可能留下空位置色块，也可能移入保留区。请特别检查Top左下与Side右下裁剪边缘，以及血管/支架交界，不能先认定没有误删。</p>",
        "<p>没有训练、重提运动特征、改标注或更换模型。原图只读；本页无标注提交接口，也不运行分类器。</p>",
        "<section><h2>同一无标签校准帧上的草案</h2><a href='calibration_regions.jpg'><img src='calibration_regions.jpg'></a></section>"]
    for task, zh in (("left","左"),("right","右")):
        parts.append(f"<section><h2>{zh}任务：五条轨迹覆盖</h2><a href='coverage_{task}.jpg'><img src='coverage_{task}.jpg'></a></section>")
    for p in index["pairs"]:
        name = f"第{p['ui_index']}窗" if p["kind"] == "window" else ("左" if p["task"] == "left" else "右") + "任务轨迹" + p["source_episode"][-3:]
        name = html.escape(name)
        links = []
        for field in ("first","last"):
            for view in VIEWS:
                url = "http://127.0.0.1:8792/image?" + urlencode({"id":p[field],"view":view})
                links.append(f"<a href='{html.escape(url, quote=True)}'>{field}/{view}原始完整图</a>")
        opened = " open" if p.get("ui_index") in (39,65) else ""
        parts.append(f"<details{opened}><summary>{name} · 原尺寸覆盖图/原图</summary><a href='{p['id']}.jpg'><img loading='lazy' src='{p['id']}.jpg'></a><p>{' · '.join(links)}</p></details>")
    parts.append("<p>原图链接只GET现有8792服务；服务未启动时可先看本页静态拼图。<a href='protocol.json'>坐标与协议</a> · <a href='report.json'>像素面积记账</a></p></html>")
    (args.out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    report = {"schema": SCHEMA, "source_unique_frames": len(index["frames"]), "source_images_read": len(cache),
        "episode_coverage_pairs": 10, "focus_windows": list(FOCUS), "generated_sheets": len(index["pairs"])+3,
        "mask_area_statistics": statistics, "calibration_rgb_matches_frozen_preview": True,
        "training_executed": False, "motion_features_extracted": False, "original_masks_modified": False,
        "labels_modified": False, "policy_input_allowed": False, "formal_data_allowed": False,
        "visual_status": "not_viewed", "user_visual_acceptance": None,
        "render_seconds": time.perf_counter()-started,
        "limitation": "partial endpoint image coverage, no all-frame guarantee, no wire recall or removal precision; mask refinement is a pending human-reviewed proposal"}
    write_json(args.out / "report.json", report)
    print(report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare","render"), required=True)
    p.add_argument("--reference", type=Path, default=Path("simulation_output/real10_response_local_motion_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--source-root", type=Path, default=Path("collected_data"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_target_vessel_region_review_v1"))
    args = p.parse_args()
    cv2.setNumThreads(1)
    {"prepare": prepare, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
