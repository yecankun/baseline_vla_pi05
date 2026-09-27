"""Read-only six-case head-segment review; no detection, labels or training.

The fixed display crop can use existing future HUMAN points. It is an offline
inspection aid, never a policy crop, head bounding box or segmentation target.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageFont


CASES = (4, 39, 62, 65, 77, 100)
ROI = {"side": (0, 540, 1440, 1040), "top": (480, 560, 1920, 1080)}
STATUS = {"visible": "可定位", "ambiguous": "模糊", "not_visible": "不可见", "unreviewed": "未标"}
FEATURE = {"tip": "尖端", "distinct_shaft_landmark": "头/杆交界材料点", "untrackable": "无可靠特征"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(args):
    source_names = ("manifest.json", "annotations_v2.jsonl")
    before = {name: (args.points / name).read_bytes() for name in source_names}
    cases = read_json(args.points / "manifest.json")["cases"]
    if tuple(c["ui_index"] for c in cases) != CASES:
        raise ValueError("only the existing six cases are in scope")
    latest = {r["case_id"]: r for r in read_rows(args.points / "annotations_v2.jsonl")}
    responses = {r["ui_index"]: r for r in read_rows(args.response_snapshot)}
    font_path = args.font or next((p for p in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")) if p.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("Chinese font required; pass --font")
    title, body, small = (ImageFont.truetype(str(font_path), size) for size in (27, 21, 18))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()
    for name in source_names:
        shutil.copy2(args.points / name, args.out / f"source_{name}")
    shutil.copy2(args.response_snapshot, args.out / "source_response_snapshot.jsonl")
    shutil.copy2(__file__, args.out / "entrypoint_snapshot.py")
    html = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>头段与尖端：六窗只读复核</title>",
        "<style>body{max-width:1500px;margin:32px auto;padding:0 20px;font:17px/1.7 system-ui;background:#f6f8fa;color:#203040}"
        "section{background:white;border:1px solid #dbe1e8;border-radius:12px;padding:22px;margin:24px 0}img{width:100%}"
        "a{color:#176b9b}small{color:#556879}.note{border-left:4px solid #176b9b;padding-left:16px}</style>",
        "<h1>先看红色头段，再看端点是否可定位</h1><p class='note'>只复核原六窗，未检测、分割或训练；"
        "约25mm来自用户描述，不是图中毫米标尺。头段边界、可见性与运动均未生成新标签。"
        "已标点不覆盖原图；未标不等于不可见，头段移动不等于沿管推进。</p>",
        "<p>有点视角使用全部原关键帧点的包围范围中心，固定640×280原像素展示窗；所有帧保持同一裁剪。"
        "使用后续人工点仅为离线审阅，不是在线定位。无点视角显示原固定场景ROI缩略图。展示窗不是头部框。</p>"]
    frames, counts = [], Counter()
    for case in cases:
        ui, cid = case["ui_index"], case["id"]
        ann = latest[cid]
        human = responses[ui]["human_annotation"]
        response = {"advance": "推进", "stationary": "无推进"}.get(human["joint_motion_response"], human["joint_motion_response"])
        canvas = Image.new("RGB", (2000, 950), "#ffffff")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 16), f"#{ui}　{'左任务' if case['task']=='left' else '右任务'}　原窗口响应：{response}　｜头段/端点只读复核", fill="#203040", font=title)
        draw.text((24, 59), "不新增标签；固定展示窗不是头段框；约25mm不是像素标尺。原人工点仅在文字中列出，未画在原图上。", fill="#556879", font=small)
        links = []
        for row, view in enumerate(("side", "top")):
            av = ann["views"][view]
            coords = [p["wire"]["xy"] for p in av["frames"].values() if p["wire"]["xy"] is not None]
            if coords:
                cx = (min(p[0] for p in coords)+max(p[0] for p in coords))/2
                cy = (min(p[1] for p in coords)+max(p[1] for p in coords))/2
                x, y = max(0, min(1280, round(cx-320))), max(0, min(800, round(cy-140)))
                crop, mode = (x, y, x+640, y+280), "原像素展示"
            else:
                crop, mode = ROI[view], "无点：全ROI缩略，非头段定位"
            y0 = 105 + row*390
            draw.text((24, y0), f"{view.capitalize()}　原特征：{FEATURE[av['feature_type']]}　｜{mode}　｜原点标revision {ann['revision']}", fill="#176b9b", font=body)
            for col, index in enumerate(case["keyframe_indices"]):
                wire = av["frames"][str(index)]["wire"]
                ref = case["frames"][view][index]
                path = args.raw_root / ref["path"]
                with Image.open(path) as im:
                    if im.size != (1920, 1080):
                        raise ValueError("original 1920x1080 image required")
                    patch = im.convert("RGB").crop(crop)
                if not coords:
                    patch.thumbnail((640, 280), Image.Resampling.LANCZOS)
                x0 = 24 + col*660
                elapsed = ref["time_s"]-case["frames"][view][0]["time_s"]
                draw.text((x0, y0+37), f"{'首中末'[col]}帧 {index}　+{elapsed:.3f}s　原状态：{STATUS[wire['status']]}", fill="#203040", font=small)
                canvas.paste(patch, (x0, y0+69))
                xy = "null" if wire["xy"] is None else ", ".join(f"{v:.1f}" for v in wire["xy"])
                draw.text((x0, y0+351), f"原人工点：({xy})", fill="#556879", font=small)
                rel = Path(os.path.relpath(path.resolve(), args.out.resolve())).as_posix()
                links.append(f"<a href='{escape(rel, quote=True)}' target='_blank'>{view} 帧{index} 原图</a>")
                frames.append({"ui_index": ui, "view": view, "frame_index": index,
                    "source_image": path.as_posix(), "elapsed_s": elapsed, "display_crop_xyxy": list(crop),
                    "display_uses_human_points": bool(coords), "display_mode": mode,
                    "original_annotation_revision": ann["revision"], "original_feature_type": av["feature_type"],
                    "original_wire": wire, "original_correspondence_confirmed": av["correspondence_confirmed"]})
                counts[wire["status"]] += 1
        draw.text((24, 904), "原图颜色未增强；未将缺失补成静止；未画自动头段/尖端。整个头段的可辨性仍需人工判断，不作监督或模型输入。", fill="#556879", font=small)
        name = f"head_review_{ui:03d}.png"
        canvas.save(args.out / "figures" / name)
        html.extend([f"<section id='case{ui}'><h2>#{ui} · 原窗口响应：{response}</h2>",
            f"<p>{escape(case['human_notes'])}</p>", f"<a href='figures/{name}' target='_blank'><img src='figures/{name}'></a>",
            "<p>"+" · ".join(links)+"</p></section>"])
    if any((args.points / name).read_bytes() != data for name, data in before.items()):
        raise ValueError("source annotation files changed during export")
    report = {"schema": "real10_head_segment_readonly_review_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": list(CASES), "keyframe_images": len(frames), "original_point_status_counts": dict(counts),
        "source_annotations_unchanged": True, "new_labels_created": False, "segmentation_performed": False,
        "training_executed": False, "policy_input_allowed": False, "deployable": False,
        "display_crops_are_targets": False, "visual_status": "not_viewed", "frames": frames}
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (args.out / "index.html").write_text("\n".join(html)+"\n</html>\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    p.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    p.add_argument("--response-snapshot", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1/source/annotation_snapshot.jsonl"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_head_segment_review_v1"))
    p.add_argument("--font", type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
