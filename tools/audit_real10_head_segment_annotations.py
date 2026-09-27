"""Read back the six human head-segment windows; never alter labels or train.

Optional raw/overlay sheets use future HUMAN geometry for offline display only.
Boxes are weak extents; line endpoints have no tip/material correspondence role.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil

from review_real10_head_segment_annotation import HeadStore, VIEWS, rows, write_json


STATUS = {"visible": "可圈定", "not_visible": "不可见", "ambiguous": "无法可靠圈定", "unreviewed": "未标"}
COVERAGE = {"complete": "完整", "partial": "部分", "uncertain": "完整性不确定", None: "无几何"}
KIND = {"bbox": "粗框", "polyline": "短线", None: "无"}


def geometry_points(g):
    if g is None:
        return []
    if g["kind"] == "bbox":
        return [g["xyxy"][:2], g["xyxy"][2:]]
    return [p for segment in g["segments"] for p in segment]


def render(store, report, raw_root, out, font):
    from PIL import Image, ImageDraw, ImageFont

    font = font or next((p for p in (Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")) if p.is_file()), None)
    if font is None:
        raise FileNotFoundError("Chinese font required; use --font")
    title, body, small = (ImageFont.truetype(str(font), size) for size in (27, 22, 18))
    (out / "figures").mkdir()
    html = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>头段粗标回读 · 六窗</title>",
        "<style>body{max-width:1550px;margin:24px auto;padding:0 20px;font:17px/1.7 system-ui;background:#f3f6f7;color:#233d4a}"
        "section{background:white;padding:20px;margin:20px 0;border:1px solid #ccd9de;border-radius:12px}img{width:100%}"
        "a{color:#087b83}table{border-collapse:collapse}td,th{padding:7px 16px;border:1px solid #ccd9de}</style>",
        "<h1>可见头段粗标：原图与人工几何对照</h1><p>只读回读，不训练；原始标注和旧响应不变。"
        "青色是人工粗框/短线，不是模型预测；每帧上图原图、下图叠加。遮挡不补全，25mm不是像素标尺。</p>",
        "<p>各视角固定展示窗使用该窗口全部人工几何，仅供离线核对，不是在线模型ROI。"
        "折线无方向，端点不自动成为尖端；粗框不是分割mask。头段位置变化不等于推进。</p>",
        "<table><tr><th>窗口</th><th>可圈定/6</th><th>完整</th><th>部分</th><th>不确定</th><th>不可见</th><th>无法圈定</th></tr>"]
    for c in report["cases"]:
        s, k = c["status_counts"], c["coverage_counts"]
        html.append("<tr>"+"".join(f"<td>{v}</td>" for v in (c["ui_index"],s.get("visible",0),k.get("complete",0),
            k.get("partial",0),k.get("uncertain",0),s.get("not_visible",0),s.get("ambiguous",0)))+"</tr>")
    html.append("</table>")
    crops, opened = [], 0
    for cid, case in store.cases.items():
        a = store.annotations[cid]
        response = {"advance": "推进", "stationary": "无推进"}.get(case["read_only_response"]["joint_motion_response"], "未知")
        sheet = Image.new("RGB", (2000, 1560), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((24, 15), f"#{case['ui_index']}　原响应：{response}　｜头段标注 revision {a['revision']}　｜上原图 / 下人工叠加", font=title, fill="#233d4a")
        draw.text((24, 59), "青色仅为人工粗框/短线；不推断隐藏部分，不把线端点当尖端，不把头段运动当推进。无模型预测或训练。", font=small, fill="#526e7b")
        for row, view in enumerate(VIEWS):
            points = [p for f in a["views"][view]["frames"].values() for p in geometry_points(f["geometry"])]
            if points:
                cx = (min(p[0] for p in points)+max(p[0] for p in points))/2
                cy = (min(p[1] for p in points)+max(p[1] for p in points))/2
                x, y = max(0,min(1280,round(cx-320))), max(0,min(800,round(cy-140)))
                crop, mode = (x,y,x+640,y+280), "固定原像素展示窗"
            else:
                crop, mode = store.manifest["display_roi_xyxy"][view], "无头段几何：全ROI缩略"
            y0 = 106+row*690
            draw.text((24,y0), f"{view.capitalize()}　{mode}（仅离线展示）", font=body, fill="#087b83")
            for col,index in enumerate(case["keyframe_indices"]):
                f = a["views"][view]["frames"][str(index)]
                g = f["geometry"]
                with Image.open(raw_root / case["frames"][view][index]["path"]) as image:
                    if image.size != (1920,1080):
                        raise ValueError("original image must be 1920x1080")
                    clean = image.convert("RGB").crop(crop)
                opened += 1
                if not points:
                    clean.thumbnail((640,280),Image.Resampling.LANCZOS)
                overlay = clean.copy()
                od = ImageDraw.Draw(overlay)
                def xy(p):
                    return (p[0]-crop[0],p[1]-crop[1])
                if g and g["kind"] == "bbox":
                    od.rectangle((*xy(g["xyxy"][:2]),*xy(g["xyxy"][2:])),outline="#00d4cf",width=2)
                elif g:
                    for segment in g["segments"]:
                        od.line([xy(p) for p in segment],fill="#00d4cf",width=2)
                x0 = 24+col*660
                desc = f"{'首中末'[col]}帧 {index}　{STATUS[f['status']]} / {COVERAGE[f['coverage']]}"
                draw.text((x0,y0+36),desc,font=small,fill="#233d4a")
                sheet.paste(clean,(x0,y0+68))
                sheet.paste(overlay,(x0,y0+368))
                draw.text((x0,y0+649),f"人工{KIND[g['kind'] if g else None]}；原坐标 / 无补画",font=small,fill="#526e7b")
                crops.append({"case_id":cid,"view":view,"frame_index":index,
                    "display_crop_xyxy":list(crop),"uses_human_future_geometry":bool(points),"policy_input_allowed":False})
        draw.text((24,1510),"六窗是反复分析过的诊断案例，不是独立测试集；图像没有增强，标注不因本次绘图自动获得训练/部署许可。",font=small,fill="#526e7b")
        name = f"head_annotation_{case['ui_index']:03d}.png"
        sheet.save(out / "figures" / name)
        html.append(f"<section id='case{case['ui_index']}'><h2>#{case['ui_index']} · 原响应：{escape(response)}</h2>"
                    f"<a href='figures/{name}' target='_blank'><img src='figures/{name}' alt='原图与人工头段对照'></a></section>")
    (out / "index.html").write_text("\n".join(html)+"\n</html>\n",encoding="utf-8")
    write_json(out / "render_manifest.json",{"original_images_opened":opened,"sheets":6,
        "visual_status":"not_viewed","display_only":True,"frames":crops})


def run(args):
    before = {name:(args.pack/name).read_bytes() for name in ("manifest.json","head_annotations.jsonl")}
    store = HeadStore(args.pack,args.raw_root)
    if store.manifest["is_ui_fixture"]:
        raise ValueError("never audit synthetic UI fixture labels as human review")
    revisions = Counter()
    for r in rows(store.path):
        cid = r["case_id"]
        revisions[cid] += 1
        if (r["revision"] != revisions[cid] or r["annotation_source"] != "human_head_segment_review"
                or r["source_episode"] != store.cases[cid]["source_episode"]
                or any(r.get(k) is not False for k in ("policy_input_allowed","formal_data_allowed","deployable"))):
            raise ValueError("unexpected annotation provenance/revision or policy promotion")
    statuses,coverages,geometries,vertices,cross = (Counter() for _ in range(5))
    cases,frames = [],[]
    for cid,case in store.cases.items():
        a = store.annotations[cid]
        cs,cc,cg = Counter(),Counter(),Counter()
        for v in VIEWS:
            for index in case["keyframe_indices"]:
                f = a["views"][v]["frames"][str(index)]
                old = case["read_only_points"]["views"][v]
                point = old["frames"][str(index)]["wire"]
                statuses[f["status"]]+=1; cs[f["status"]]+=1
                cross[f"point_{point['status']}__head_{f['status']}"]+=1
                if f["geometry"]:
                    g = f["geometry"]
                    coverages[f["coverage"]]+=1; cc[f["coverage"]]+=1
                    geometries[g["kind"]]+=1; cg[g["kind"]]+=1
                    if g["kind"] == "polyline":
                        for segment in g["segments"]:
                            vertices[str(len(segment))]+=1
                frames.append({"case_id":cid,"ui_index":case["ui_index"],"view":v,"frame_index":index,
                    "head":f,"old_point":point,"old_point_feature_type":old["feature_type"],
                    "old_point_revision":case["read_only_points"]["revision"],"head_revision":a["revision"]})
        cases.append({"case_id":cid,"ui_index":case["ui_index"],"revision":a["revision"],
            "status_counts":dict(cs),"coverage_counts":dict(cc),"geometry_counts":dict(cg),
            "old_response_read_only":case["read_only_response"]["joint_motion_response"]})
    if any((args.pack/name).read_bytes()!=data for name,data in before.items()):
        raise ValueError("annotations changed while reading; rerun after edits stop")
    report = {"schema":"real10_head_segment_annotation_audit_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_pack":args.pack.as_posix(),"summary":store.summary(),"cases":cases,
        "coverage_counts":dict(coverages),"geometry_counts":dict(geometries),"polyline_vertex_counts":dict(vertices),
        "old_point_vs_new_head_status":dict(cross),"cross_statuses_have_different_target_semantics":True,
        "source_bytes_unchanged":True,"labels_changed":False,"training_executed":False,
        "policy_input_allowed":False,"formal_data_allowed":False,"deployable":False,
        "frames":frames}
    args.out.mkdir(parents=True,exist_ok=False)
    for name,data in before.items():
        (args.out / ("source_"+name)).write_bytes(data)
    shutil.copy2(__file__,args.out / "entrypoint_snapshot.py")
    write_json(args.out / "report.json",report)
    if not args.metadata_only:
        render(store,report,args.raw_root,args.out,args.font)
    if any((args.pack/name).read_bytes()!=data for name,data in before.items()):
        raise ValueError("source changed during rendering; preserve output and inspect")
    print(json.dumps({k:v for k,v in report.items() if k not in ("frames","cases")},ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack",type=Path,default=Path("simulation_output/real10_head_segment_annotation_v1"))
    p.add_argument("--raw-root",type=Path,default=Path("collected_data"))
    p.add_argument("--out",type=Path,default=Path("simulation_output/real10_head_segment_annotation_audit_v1"))
    p.add_argument("--metadata-only",action="store_true",help="schema/readback only, no images required")
    p.add_argument("--font",type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
