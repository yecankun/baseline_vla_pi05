"""Read-only audit of saved right-task stationary OOF cases; NEVER fit/relabel.

--stage summarize: saved features, fold coefficients and observable metadata only.
--stage render: local original RGB -> labeled contact sheets / read-only replay page.
Derived audit output is separate from annotations, model artifacts and policy data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json
from run_real10_response_baseline import predict_ridge
from run_real10_response_roi_comparison import METHODS, NAMES, ROI


def decompose(x, model):
    """Exact additive bookkeeping relative to train-fold centers, not causal XAI."""
    finite = np.isfinite(x)
    z = np.concatenate((np.where(finite, (x - model["mean"]) / model["std"], 0), (~finite).astype(float)))
    z /= model["feature_scale"]
    c = (z - model["center"]) * model["coef"]
    d = len(x)
    per_raw_feature = c[:d] + c[d:]
    groups = {"controller": float(per_raw_feature[:40].sum())}
    if d > 40:
        groups["anchor_visual"] = float(per_raw_feature[40:552].sum() + per_raw_feature[1576:2088].sum())
        groups["change_visual"] = float(per_raw_feature[552:1576].sum() + per_raw_feature[2088:].sum())
    score = float(predict_ridge(model, x[None])[0])
    center = float(model["target_center"])
    np.testing.assert_allclose(center + sum(groups.values()), score, atol=1e-10, rtol=0)
    return {"score_not_probability": score, "center": center, "group_contributions": groups}, per_raw_feature


def summarize(args):
    started = time.perf_counter()
    rows = read_jsonl(args.run / "annotation_snapshot.jsonl")
    predictions = {(r["window_id"], r["method"]): r for r in read_jsonl(args.run / "oof_predictions.jsonl")}
    folds = read_json(args.run / "folds.json")
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    names = read_json(args.baseline / "protocol.json")["controller_feature_names"]
    with np.load(args.run / "features.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    cases = []
    for i, row in enumerate(rows):
        if row["task"] != "right" or row["human_annotation"]["joint_motion_response"] != "stationary":
            continue
        fold = next(j for j, f in enumerate(folds) if f["heldout_episode"] == row["source_episode"])
        frames = [observations[f] for f in [row["anchor_frame_id"], *row["future_frame_ids"]]]
        values = dict(zip(names, arrays["controller_only"][i]))
        selected_features = {k: float(values[k]) if np.isfinite(values[k]) else None for k in
                             ("response_span_s", "observed_elite_xyz_path_mm", "busy_fraction",
                              "feed_command_history_fraction", "logged_counter_change_not_physical_feed",
                              "controller_observed_fraction")}
        case = {"ui_index": row["ui_index"], "window_id": row["window_id"], "episode": row["source_episode"],
                "human_annotation": row["human_annotation"], "selection_audit_only": row["selection_reason_audit_only"],
                "observable_features": selected_features, "models": {}, "frames": []}
        for frame in frames:
            ctl = frame["controller_history"]
            case["frames"].append({"frame_id": frame["frame_id"],
                                   "relative_s": frame["observation_available_at_s"] - frames[0]["observation_available_at_s"],
                                   "images": frame["images"], "controller_valid": ctl["valid"],
                                   "controller_history_source_step": ctl["source_step"],
                                   "controller_history": ctl["values"], "elite": frame["elite"]})
        for method in METHODS:
            with np.load(args.run / "fold_models" / method / f"fold_{fold:02d}.npz", allow_pickle=False) as z:
                model = {k: z[k] for k in z.files}
            detail, terms = decompose(arrays[method][i], model)
            saved = predictions[(row["window_id"], method)]
            np.testing.assert_allclose(detail["score_not_probability"], saved["score_not_probability"], rtol=0, atol=1e-10)
            detail["prediction"] = saved["prediction"]
            detail["top_controller_terms"] = sorted(
                [{"name": name, "contribution": float(terms[j])} for j, name in enumerate(names)],
                key=lambda p: abs(p["contribution"]), reverse=True)[:5]
            case["models"][method] = detail
        cases.append(case)
    right = [r for r in rows if r["task"] == "right"]
    coverage = {episode: dict(Counter(r["human_annotation"]["joint_motion_response"] for r in right if r["source_episode"] == episode))
                for episode in sorted({r["source_episode"] for r in right})}
    fp = [c for c in cases if c["models"][METHODS[2]]["prediction"] == "advance"]
    report = {"schema": "real10_right_stationary_readonly_audit_v1", "source_run": args.run.as_posix(),
              "selection": "ALL seven right-task human-stationary windows, including the correctly classified control",
              "cases": cases, "right_episode_label_coverage": coverage,
              "roi_false_positives": len(fp),
              "false_positives_with_any_human_clear_view": sum("clear" in c["human_annotation"]["anchor_visibility"].values() for c in fp),
              "decomposition": "score = train-fold target center + controller + anchor visual + temporal-change visual; algebraic terms, not calibrated probabilities or causal feature attribution",
              "audit_seconds": time.perf_counter() - started,
              "training_executed": False, "labels_modified": False, "roi_modified": False,
              "hardware_executed": False, "policy_input_allowed": False, "visual_status": "not_viewed"}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "audit.json", report)
    print({"cases": len(cases), "false_positives": len(fp), "saved_score_checks": len(cases) * len(METHODS),
           "any_clear_false_positives": report["false_positives_with_any_human_clear_view"], "seconds": report["audit_seconds"]}, flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    audit = read_json(args.out / "audit.json")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    large = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 29)
    summary_rows, sections = [], []
    for case in audit["cases"]:
        index, frames = case["ui_index"], case["frames"]
        scores = [case["models"][m]["score_not_probability"] for m in METHODS]
        visible = case["human_annotation"]["anchor_visibility"]
        v = case["observable_features"]
        sheet = Image.new("RGB", (1944, 1370), "#f5f6f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((18, 10), f"窗口{index}｜人工：无明显推进｜控制器/全图/ROI分数：" + " / ".join(f"{s:.3f}" for s in scores), font=large, fill="#202939")
        draw.text((18, 52), f"原标注可见性 side={visible['side']}，top={visible['top']}；Elite路径 {v['observed_elite_xyz_path_mm']:.3f} mm", font=font, fill="#202939")
        draw.text((18, 87), "原图固定ROI放大审查：不更改模型裁剪；左列Side，右列Top；分数不是概率。", font=font, fill="#202939")
        for row_index, frame_index in enumerate((0, len(frames) // 2, len(frames) - 1)):
            f = frames[frame_index]
            y = 130 + 410 * row_index
            draw.text((18, y), f"第{frame_index}帧  t=+{f['relative_s']:.3f}s  {f['frame_id'].rsplit('/', 1)[-1]}", font=font, fill="#202939")
            for col, view in enumerate(("side", "top")):
                with Image.open(args.source_root / f["images"][view]["path"]) as im:
                    patch = im.convert("RGB").crop(ROI[view])
                patch.thumbnail((950, 365), Image.Resampling.LANCZOS)
                sheet.paste(patch, (18 + col * 966, y + 35))
        sheet.save(args.out / f"window_{index:03d}.jpg", quality=95)
        summary_rows.append(f'<tr><td><a href="#w{index}">{index}</a></td><td>{html.escape(case["episode"].rsplit("_",1)[-1])}</td>'
                            f'<td>{visible["side"]} / {visible["top"]}</td><td>{v["observed_elite_xyz_path_mm"]:.3f}</td>'
                            + ''.join(f'<td>{s:.4f}</td>' for s in scores) + '</tr>')
        sections.append(f'<section id="w{index}"><h2>窗口{index} · 原标注：无明显推进</h2>'
                        f'<img class="sheet" src="window_{index:03d}.jpg"><details><summary>查看原图短时序列（只读，不写标签）</summary>'
                        f'<div class="player" data-window="{index}"><div><button type="button" class="play">播放/暂停</button> '
                        f'<input class="step" type="range" min="0" max="{len(frames)-1}" value="0"> <span class="stamp"></span></div>'
                        '<div class="views"><figure><img data-view="side"><figcaption>Side原图</figcaption></figure>'
                        '<figure><img data-view="top"><figcaption>Top原图</figcaption></figure></div></div></details></section>')
    data = [{"index": c["ui_index"], "frames": [{"id": f["frame_id"], "t": f["relative_s"]} for f in c["frames"]]} for c in audit["cases"]]
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>右任务无明显推进误判审查</title>'
    page += '<style>body{font:16px "Microsoft YaHei",sans-serif;max-width:1280px;margin:25px auto;background:#f5f6f8;color:#202939}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd3dc;padding:9px}section{background:white;margin:22px 0;padding:18px}img.sheet{width:100%}.views{display:flex;gap:12px}figure{width:50%;margin:12px 0}figure img{width:100%}summary{cursor:pointer;padding:12px}input{width:45%}</style>'
    page += '<h1>右任务：“无明显推进”误判只读审查</h1><p>包含全部7个stationary窗口：6个ROI误判＋1个正确对照。分数≥0.5判为推进，但不是概率。未重训、未修改标注或ROI。</p>'
    page += '<p>时序图保留既有固定ROI，仅作为放大的人工审查视图；展开后可访问本机8792服务中的原始双视角序列。两路采用记录配对，不宣称硬件同步。这里不提交任何标注。</p>'
    page += '<table><tr><th>窗号</th><th>右轨迹</th><th>Side/Top原标注可见性</th><th>Elite路径/mm</th><th>控制器分数</th><th>全图分数</th><th>ROI分数</th></tr>' + ''.join(summary_rows) + '</table>' + ''.join(sections)
    page += '<script>const cases=' + json.dumps(data) + ';document.querySelectorAll(".player").forEach(p=>{const c=cases.find(c=>c.index===Number(p.dataset.window));const input=p.querySelector(".step");let timer=null;function show(){const f=c.frames[Number(input.value)];p.querySelector(".stamp").textContent=`第${Number(input.value)}帧 · +${f.t.toFixed(3)}秒`;p.querySelectorAll("img").forEach(im=>im.src="http://127.0.0.1:8792/image?id="+encodeURIComponent(f.id)+"&view="+im.dataset.view);}function next(){input.value=(Number(input.value)+1)%c.frames.length;show();if(timer!==null){const i=Number(input.value),dt=i<c.frames.length-1?c.frames[i+1].t-c.frames[i].t:.6;timer=setTimeout(next,Math.max(50,dt*1000));}}input.addEventListener("input",show);p.querySelector(".play").onclick=()=>{if(timer!==null){clearTimeout(timer);timer=null;}else{timer=setTimeout(next,250);}};p.closest("details").addEventListener("toggle",e=>{if(e.target.open)show();else if(timer!==null){clearTimeout(timer);timer=null;}});});</script></html>'
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print({"contact_sheets": len(audit["cases"]), "read_only_page": str(args.out / "index.html")}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("summarize", "render"), required=True)
    p.add_argument("--run", type=Path, default=Path("simulation_output/real10_response_matched_roi_v1"))
    p.add_argument("--baseline", type=Path, default=Path("simulation_output/real10_response_baseline_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--source-root", type=Path, default=Path("collected_data"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_response_error_audit_v1"))
    args = p.parse_args()
    summarize(args) if args.stage == "summarize" else render(args)


if __name__ == "__main__":
    main()
