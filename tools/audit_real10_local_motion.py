"""Read-only localization of frozen local-motion errors. No fit or feature edits.

Summarize on project4090, then render source RGB locally. Saved ridge terms are
algebraic score contributions, NOT causal attribution or semantic wire tracking.
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
from run_real10_response_local_motion import FEATURE_NAMES, METHODS, NAMES, GRID, ROI


def terms_for(x, model):
    finite = np.isfinite(x)
    z = np.concatenate((np.where(finite, (x - model["mean"]) / model["std"], 0), (~finite).astype(float)))
    z /= model["feature_scale"]
    terms = (z - model["center"]) * model["coef"]
    n = len(x)
    terms = terms[:n] + terms[n:]
    score = float(predict_ridge(model, x[None])[0])
    np.testing.assert_allclose(float(model["target_center"]) + terms.sum(), score, atol=1e-10, rtol=0)
    return score, terms


def summarize(args):
    started = time.perf_counter()
    rows = read_jsonl(args.run / "annotation_snapshot.jsonl")
    folds = read_json(args.run / "folds.json")
    predictions = {(p["window_id"], p["method"]): p for p in read_jsonl(args.run / "oof_predictions.jsonl")}
    control_names = read_json(args.baseline / "protocol.json")["controller_feature_names"]
    quality = {q["window_id"]: q for q in read_jsonl(args.run / "motion_quality.jsonl")}
    observations = {o["frame_id"]: o for o in read_jsonl(args.pack / "observations.jsonl")}
    with np.load(args.run / "features.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    cases = []
    for i, row in enumerate(rows):
        if row["task"] != "left" and row["ui_index"] != 65:
            continue
        fold_index = next(j for j, f in enumerate(folds) if f["heldout_episode"] == row["source_episode"])
        frames = [observations[f] for f in [row["anchor_frame_id"], *row["future_frame_ids"]]]
        values = dict(zip(control_names, arrays["controller_only"][i]))
        case = {"ui_index": row["ui_index"], "task": row["task"], "episode": row["source_episode"],
                "window_id": row["window_id"], "annotation": row["human_annotation"], "models": {},
                "observable_features": {k: float(values[k]) if np.isfinite(values[k]) else None for k in
                    ("response_span_s", "observed_elite_xyz_path_mm", "busy_fraction", "feed_command_history_fraction",
                     "logged_counter_change_not_physical_feed")},
                "frames": [{"frame_id": f["frame_id"], "relative_s": f["observation_available_at_s"] - frames[0]["observation_available_at_s"],
                            "images": f["images"]} for f in frames], "reference_quality": {}}
        for view in ("side", "top"):
            q = quality[row["window_id"]]["by_view"][view]
            case["reference_quality"][view] = {
                "pairs": len(q), "valid_pairs": sum(p["valid"] for p in q),
                "max_reference_translation_px": max(float(np.linalg.norm(p["reference_translation_px"])) for p in q if p["valid"]),
                "max_fb_error_px": max(p["median_reference_fb_error_px"] for p in q if p["valid"])}
        for method in METHODS:
            with np.load(args.run / "fold_models" / method / f"fold_{fold_index:02d}.npz", allow_pickle=False) as z:
                model = {k: z[k] for k in z.files}
            score, terms = terms_for(arrays[method][i], model)
            saved = predictions[(row["window_id"], method)]
            np.testing.assert_allclose(score, saved["score_not_probability"], atol=1e-10, rtol=0)
            ncontrol = 0 if method == "local_motion_only" else 40
            named = list(zip(control_names[:ncontrol] + ([] if method == "controller_only" else FEATURE_NAMES), terms))
            controller = {"task": 0., "absolute_elite_pose": 0., "observed_elite_motion": 0., "other_controller": 0.}
            for name, value in named[:ncontrol]:
                group = "task" if name == "task_right" else "absolute_elite_pose" if name.startswith("anchor_elite_") else "observed_elite_motion" if name.startswith("observed_elite_") else "other_controller"
                controller[group] += float(value)
            grids = {v: np.zeros(GRID) for v in ("side", "top")}
            motion_family = {"flow": 0., "ridge_appearance_change": 0.}
            for name, value in named[ncontrol:]:
                view, _, cell, _ = name.split("_", 3)
                r, c = int(cell[1]), int(cell[3])
                grids[view][r, c] += value
                motion_family["ridge_appearance_change" if "ridge_" in name else "flow"] += float(value)
            groups = {**controller, **{v + "_motion": float(g.sum()) for v, g in grids.items()}}
            np.testing.assert_allclose(float(model["target_center"]) + sum(groups.values()), score, rtol=0, atol=1e-10)
            x = arrays[method][i]
            top_indices = sorted(range(len(terms)), key=lambda j: abs(terms[j]), reverse=True)[:8]
            top_terms = [{"name": named[j][0], "contribution": float(terms[j]),
                          "raw_value": float(x[j]) if np.isfinite(x[j]) else None,
                          "train_mean": float(model["mean"][j]), "train_std": float(model["std"][j]),
                          "train_standardized_value": float((x[j] - model["mean"][j]) / model["std"][j]) if np.isfinite(x[j]) else None}
                         for j in top_indices]
            case["models"][method] = {"score_not_probability": score, "prediction": saved["prediction"], "correct": saved["correct"],
                "center": float(model["target_center"]), "groups": groups, "motion_family": motion_family,
                "motion_grids": {v: g.tolist() for v, g in grids.items()},
                "top_terms": top_terms}
        case["regression_vs_controller"] = case["models"][METHODS[0]]["correct"] and not case["models"][METHODS[1]]["correct"]
        case["improvement_vs_controller"] = not case["models"][METHODS[0]]["correct"] and case["models"][METHODS[1]]["correct"]
        cases.append(case)
    left = [c for c in cases if c["task"] == "left"]
    regressions = [c for c in left if c["regression_vs_controller"]]
    improvements = [c for c in left if c["improvement_vs_controller"]]
    # Both-label correct controls from affected episodes, kept explicit, not picked by a new score threshold.
    preview = sorted({c["ui_index"] for c in regressions} | {11, 15, 21, 32, 33, 65})
    report = {"schema": "real10_local_motion_readonly_audit_v1", "cases": cases,
              "selection": "all 25 left-task windows plus right window 65; preview every new left regression plus explicit same-episode controls and fusion-only regression 33",
              "left_regressions": [c["ui_index"] for c in regressions], "left_improvements": [c["ui_index"] for c in improvements],
              "preview_windows": preview, "regression_episode_counts": dict(Counter(c["episode"] for c in regressions)),
              "saved_scores_checked": len(cases) * len(METHODS), "score_atol": 1e-10,
              "interpretation": "Train-fold centered algebraic contributions; not causal, not localization ground truth; cross-fold coefficients have different centers/scales.",
              "audit_seconds": time.perf_counter() - started, "training_executed": False, "features_modified": False,
              "labels_modified": False, "roi_modified": False, "hardware_executed": False, "policy_input_allowed": False}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "audit.json", report)
    print({k: report[k] for k in ("left_regressions", "left_improvements", "preview_windows", "saved_scores_checked", "audit_seconds")}, flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    audit = read_json(args.out / "audit.json")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    large = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    selected = [c for c in audit["cases"] if c["ui_index"] in audit["preview_windows"]]
    summary, sections = [], []
    for case in selected:
        index, frames = case["ui_index"], case["frames"]
        scores = [case["models"][m]["score_not_probability"] for m in METHODS]
        label = "推进" if case["annotation"]["joint_motion_response"] == "advance" else "无明显推进"
        sheet = Image.new("RGB", (2240, 1120), "#f5f6f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 12), f"窗口{index}｜人工：{label}｜控制器/仅运动/融合分数：" + " / ".join(f"{s:.4f}" for s in scores), font=large, fill="#202939")
        obs = case["observable_features"]
        draw.text((20, 58), f"Elite路径 {obs['observed_elite_xyz_path_mm']:.3f}mm；帧窗 {obs['response_span_s']:.3f}s；网格仅为冻结分类器的分数记账，不是导丝追踪。", font=font, fill="#202939")
        draw.text((20, 98), "左/中：原始RGB固定ROI首末帧；右：仅运动模型的网格贡献，红色偏推进、蓝色偏无明显推进。", font=font, fill="#202939")
        for j, view in enumerate(("side", "top")):
            y = 143 + 380 * j
            patches = []
            for f in (frames[0], frames[-1]):
                with Image.open(args.source_root / f["images"][view]["path"]) as im:
                    patch = im.convert("RGB").crop(ROI[view])
                patch = patch.resize((720, round(patch.height / patch.width * 720)), Image.Resampling.LANCZOS)
                patches.append(patch)
            marked = patches[0].copy()
            overlay = Image.new("RGBA", marked.size)
            painter = ImageDraw.Draw(overlay)
            grid = case["models"]["local_motion_only"]["motion_grids"][view]
            w, h = marked.size
            for r in range(GRID[0]):
                for c in range(GRID[1]):
                    value = grid[r][c]
                    box = (c*w//6, r*h//2, (c+1)*w//6-1, (r+1)*h//2-1)
                    color = (235, 65, 45) if value >= 0 else (45, 95, 230)
                    painter.rectangle(box, fill=(*color, min(150, round(abs(value)*800))), outline=(255,255,255,150))
                    painter.rectangle((box[0]+2, box[1]+2, box[0]+117, box[1]+27), fill=(255,255,255,230))
                    painter.text((box[0]+3, box[1]+3), f"{r},{c} {value:+.3f}", font=small, fill=(20,30,45,255))
            marked = Image.alpha_composite(marked.convert("RGBA"), overlay).convert("RGB")
            for col, (patch, title) in enumerate(zip((*patches, marked), (f"{view} 首帧", f"{view} 末帧", f"{view} 网格贡献"))):
                x = 20 + col*740
                draw.text((x, y), title, font=font, fill="#202939")
                sheet.paste(patch, (x, y+32))
            m = case["models"]["local_motion_only"]["groups"][view + "_motion"]
            f = case["models"]["controller_local_motion"]["groups"][view + "_motion"]
            draw.text((20, y+302), f"该视角分数贡献：仅运动 {m:+.4f}；融合 {f:+.4f}（不同模型，不能作为因果差值）。", font=font, fill="#202939")
        local = case["models"]["local_motion_only"]
        fused = case["models"]["controller_local_motion"]["groups"]
        fam = local["motion_family"]
        draw.text((20, 938), f"仅运动分数 = 0.5 + 光流项 {fam['flow']:+.4f} + 暗线外观变化项 {fam['ridge_appearance_change']:+.4f}", font=font, fill="#202939")
        draw.text((20, 984), f"融合控制器项：task {fused['task']:+.4f}；绝对姿态 {fused['absolute_elite_pose']:+.4f}；实测Elite变化 {fused['observed_elite_motion']:+.4f}；其余 {fused['other_controller']:+.4f}", font=font, fill="#202939")
        draw.text((20, 1030), "未改人工标签/区域/参数，未训练。粗网格可能同时包含导丝、血管纹理和器械，不能据此确证语义来源。", font=font, fill="#202939")
        name = f"window_{index:03d}.jpg"
        sheet.save(args.out / name, quality=94)
        summary.append(f'<tr><td><a href="#w{index}">{index}</a></td><td>{label}</td><td>{obs["observed_elite_xyz_path_mm"]:.3f}</td>' + ''.join(f'<td>{s:.4f}</td>' for s in scores) + '</tr>')
        sections.append(f'<section id="w{index}"><h2>窗口{index} · 原人工标注：{label}</h2><img class="sheet" src="{name}">'
            f'<details><summary>查看完整原图短时序列（只读）</summary><div class="player" data-window="{index}">'
            f'<input type="range" min="0" max="{len(frames)-1}" value="0"><span></span><div class="views"><img data-view="side"><img data-view="top"></div></div></details></section>')
    data = [{"index": c["ui_index"], "frames": [{"id": f["frame_id"], "t": f["relative_s"]} for f in c["frames"]]} for c in selected]
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>局部运动误判定位</title><style>body{font:17px sans-serif;max-width:1600px;margin:24px auto;background:#f5f6f8;color:#202939}section{background:white;margin:22px 0;padding:14px}img.sheet{width:100%}.views{display:flex;gap:1%}.views img{width:49%}td,th{border:1px solid #ccd3dc;padding:10px}table{border-collapse:collapse}input{width:50%}summary{cursor:pointer;padding:12px}</style>'
    page += '<h1>左任务退化与第65窗：只读定位</h1><p>数字审查覆盖全部25个左任务窗口及第65窗；展示全部6个左侧退化窗口、同轨迹正确对照、第33窗融合退化和第65窗。分数不是概率，贡献不是因果解释。</p><p>不训练、不改标签或掩膜；这里不能裁定导丝接触或自动重标。原图序列仅GET本地8792服务，不提交标签。</p>'
    page += '<table><tr><th>窗号</th><th>人工响应</th><th>Elite路径/mm</th><th>控制器</th><th>仅运动</th><th>融合</th></tr>' + ''.join(summary) + '</table>' + ''.join(sections)
    page += '<script>const cases=' + json.dumps(data) + ';document.querySelectorAll(".player").forEach(p=>{const c=cases.find(c=>c.index===Number(p.dataset.window));const input=p.querySelector("input");function show(){const f=c.frames[Number(input.value)];p.querySelector("span").textContent=`第${Number(input.value)}帧 +${f.t.toFixed(3)}秒`;p.querySelectorAll("img").forEach(im=>im.src="http://127.0.0.1:8792/image?id="+encodeURIComponent(f.id)+"&view="+im.dataset.view);}input.addEventListener("input",show);p.closest("details").addEventListener("toggle",e=>{if(e.target.open)show();});});</script></html>'
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print({"sheets": len(selected), "page": str(args.out / "index.html"), "visual_status": "not_viewed"}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("summarize", "render"), required=True)
    p.add_argument("--run", type=Path, default=Path("simulation_output/real10_response_local_motion_v1"))
    p.add_argument("--baseline", type=Path, default=Path("simulation_output/real10_response_baseline_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--source-root", type=Path, default=Path("collected_data"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_local_motion_error_audit_v1"))
    args = p.parse_args()
    (summarize if args.stage == "summarize" else render)(args)


if __name__ == "__main__":
    main()
