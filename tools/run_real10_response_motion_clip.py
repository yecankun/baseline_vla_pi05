"""One fixed +/-5 sigma motion-only ablation on frozen real10 development OOF.

No feature extraction, label revision, clip search, fusion or policy training.
Evaluate on the 4090; render the Chinese comparison locally without refitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from pathlib import Path
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_baseline import ALPHA, SEED, LABELS, LABEL_ZH, fit_ridge, folds_for, metrics, predict_ridge


SCHEMA = "real10_response_motion_fixed_clip5_v1"
BASE = "local_motion_only"
CLIPPED = "local_motion_clip5"
CLIP = 5.0  # Fixed before this run; intentionally not a sweep/CLI argument.
NAMES = {BASE: "原仅运动", CLIPPED: "固定 ±5σ 裁剪"}


def standardized(x, model):
    return np.where(np.isfinite(x), (x - model["mean"]) / model["std"], 0)


def clipping_summary(x, model):
    z = np.abs(standardized(x, model))
    clipped = z > CLIP
    return {"rows": len(x), "rows_with_clipping": int(clipped.any(axis=1).sum()),
            "clipped_raw_values": int(clipped.sum()), "finite_raw_values": int(np.isfinite(x).sum()),
            "raw_dimensions_with_clipping": int(clipped.any(axis=0).sum()),
            "max_abs_z_before": float(z.max()), "max_abs_z_after": float(np.minimum(z, CLIP).max())}


def evaluate(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("preserve completed/partial artifacts; inspect them, or use an explicitly new --out")
    source = read_json(args.reference / "protocol.json")
    rows = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    folds = read_json(args.reference / "folds.json")
    if (source["schema"] != "real10_vessel_relative_local_motion_v1" or
            source["ridge_alpha"] != ALPHA or source["seed"] != SEED or source["threshold"] != .5 or
            folds != folds_for(rows) or len(rows) != 45 or len(folds) != 10):
        raise ValueError("not the approved frozen 45-window/10-fold experiment")
    with np.load(args.reference / "features.npz", allow_pickle=False) as z:
        x, y = z[BASE], z["y"]
    names = source["feature_names"]
    if x.shape != (45, 288) or len(names) != 288:
        raise ValueError("expected unchanged 288-dimensional motion-only features")
    np.testing.assert_array_equal(y, [LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])
    old = {p["window_id"]: p for p in read_jsonl(args.reference / "oof_predictions.jsonl") if p["method"] == BASE}
    if set(old) != {r["window_id"] for r in rows}:
        raise ValueError("frozen OOF predictions do not cover the same windows")
    for row in rows:
        p = old[row["window_id"]]
        if p["target"] != row["human_annotation"]["joint_motion_response"] or p["heldout_episode"] != row["source_episode"]:
            raise ValueError("frozen OOF labels or episode identity changed")

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "protocol.json", {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": args.reference.as_posix(), "source_protocol": source,
        "methods": [BASE, CLIPPED], "standardized_clip": CLIP,
        "only_change": "train-fold z scores of raw motion dimensions clipped to [-5,5] in BOTH fit and predict, before unchanged block scaling; re-fit weighted center and ridge normally",
        "unchanged": ["raw features and NaNs", "288-d motion-only input", "45 frozen labels", "10 whole-episode folds",
                      "training-fold mean/std", "missing indicators", "block scaling [288]", "class balancing",
                      "alpha=1", "threshold=0.5", "seed=20260920", "ROI/mask", "PI05 split and weights"],
        "reference_refit": "no-clip refit must reproduce all frozen scores at rtol=0, atol=1e-10",
        "selection": "single user-approved development ablation after error inspection; no threshold/clip/seed search",
        "policy_input_allowed": False, "formal_data_allowed": False,
    })
    write_jsonl(args.out / "annotation_snapshot.jsonl", rows)
    write_json(args.out / "folds.json", folds)
    started = time.perf_counter()
    scores = {BASE: np.full(len(rows), np.nan), CLIPPED: np.full(len(rows), np.nan)}
    fold_audit, window_audit = [], []
    folder = args.out / "fold_models" / CLIPPED
    folder.mkdir(parents=True)
    for k, fold in enumerate(folds):
        train, test = np.array(fold["train_indices"]), np.array(fold["test_indices"])
        plain = fit_ridge(x[train], y[train], block_sizes=[288])
        scores[BASE][test] = predict_ridge(plain, x[test])
        np.testing.assert_allclose(scores[BASE][test], [old[rows[i]["window_id"]]["score_not_probability"] for i in test], rtol=0, atol=1e-10)
        clipped = fit_ridge(x[train], y[train], block_sizes=[288], standardized_clip=CLIP)
        for key in ("mean", "std", "feature_scale", "train_class_counts", "alpha"):
            np.testing.assert_array_equal(clipped[key], plain[key])
        scores[CLIPPED][test] = predict_ridge(clipped, x[test])
        model_path = folder / f"fold_{k:02d}.npz"
        np.savez_compressed(model_path, **clipped)
        with np.load(model_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(predict_ridge(saved, x[test]), scores[CLIPPED][test])
        fold_audit.append({"heldout_episode": fold["heldout_episode"],
                           "train": clipping_summary(x[train], clipped), "test": clipping_summary(x[test], clipped)})
        for i in test:
            z = standardized(x[i:i+1], clipped)[0]
            affected = np.flatnonzero(np.abs(z) > CLIP)
            extreme = sorted(affected, key=lambda j: abs(z[j]), reverse=True)[:5]
            window_audit.append({"window_id": rows[i]["window_id"], "ui_index": rows[i]["ui_index"],
                "heldout_episode": fold["heldout_episode"], "missing_raw_dimensions": int((~np.isfinite(x[i])).sum()),
                **clipping_summary(x[i:i+1], clipped),
                "largest_clipped_features": [{"name": names[j], "raw_value": float(x[i, j]),
                    "train_mean": float(clipped["mean"][j]), "train_std": float(clipped["std"][j]),
                    "z_before": float(z[j]), "z_after": float(np.clip(z[j], -CLIP, CLIP))} for j in extreme]})

    results, predictions = {}, []
    for method, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError("incomplete OOF predictions")
        guess = (values >= .5).astype(int)
        episodes = []
        for fold in folds:
            ix = np.array(fold["test_indices"])
            episodes.append({"episode": fold["heldout_episode"], **metrics(y[ix], guess[ix])})
        tasks = {}
        for task in ("left", "right"):
            ix = np.array([i for i, r in enumerate(rows) if r["task"] == task])
            tasks[task] = metrics(y[ix], guess[ix])
        results[method] = {"pooled": metrics(y, guess), "by_task": tasks, "episodes": episodes,
                           "episode_macro_accuracy": float(np.mean([e["accuracy"] for e in episodes]))}
        for i, row in enumerate(rows):
            predictions.append({"window_id": row["window_id"], "ui_index": row["ui_index"], "task": row["task"],
                "heldout_episode": row["source_episode"], "method": method, "target": LABELS[y[i]],
                "prediction": LABELS[guess[i]], "score_not_probability": float(values[i]), "correct": bool(y[i] == guess[i])})
    a, b = (scores[BASE] >= .5).astype(int), (scores[CLIPPED] >= .5).astype(int)
    paired = [{"window_id": r["window_id"], "ui_index": r["ui_index"], "task": r["task"],
               "episode": r["source_episode"], "target": LABELS[y[i]],
               "baseline_score": float(scores[BASE][i]), "clipped_score": float(scores[CLIPPED][i]),
               "baseline_prediction": LABELS[a[i]], "clipped_prediction": LABELS[b[i]],
               "effect": "unchanged" if a[i] == b[i] else ("helped" if b[i] == y[i] else "hurt")}
              for i, r in enumerate(rows)]
    changes = {}
    for scope in ("pooled", "left", "right"):
        before = results[BASE]["pooled"] if scope == "pooled" else results[BASE]["by_task"][scope]
        after = results[CLIPPED]["pooled"] if scope == "pooled" else results[CLIPPED]["by_task"][scope]
        changes[scope] = {key + "_pp": 100 * (after[key] - before[key]) for key in ("accuracy", "balanced_accuracy")}
    ed = [b["accuracy"] - a["accuracy"] for a, b in zip(results[BASE]["episodes"], results[CLIPPED]["episodes"])]
    report = {"schema": SCHEMA, "samples": len(rows), "episodes": len(folds), "label_order": list(LABELS),
        "methods": results, "changes_clipped_minus_baseline": changes,
        "helped_windows": [p["ui_index"] for p in paired if p["effect"] == "helped"],
        "hurt_windows": [p["ui_index"] for p in paired if p["effect"] == "hurt"],
        "episodes_improved_tied_worse": [sum(d > 0 for d in ed), sum(d == 0 for d in ed), sum(d < 0 for d in ed)],
        "key_windows": [p for p in paired if p["ui_index"] in (39, 65)], "fold_clipping_audit": fold_audit,
        "test_clipping_summary": {"windows_with_clipping": sum(w["rows_with_clipping"] for w in window_audit),
            "raw_values_clipped": sum(w["clipped_raw_values"] for w in window_audit),
            "finite_raw_values": sum(w["finite_raw_values"] for w in window_audit)},
        "baseline_score_max_abs_difference": float(np.max(np.abs(scores[BASE] - [old[r["window_id"]]["score_not_probability"] for r in rows]))),
        "baseline_scores_reproduced_atol": 1e-10, "saved_candidate_models_reloaded": len(folds),
        "fit_evaluate_seconds": time.perf_counter() - started, "recognizer_fit_executed": True,
        "PI05_training_executed": False, "hardware_executed": False, "policy_input_allowed": False,
        "formal_data_allowed": False, "independent_final_test": False,
        "interpretation": "development OOF after prior error inspection; no stable gain, contact or deployability claim",
        "visual_status": "not_viewed"}
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    write_jsonl(args.out / "paired_windows.jsonl", paired)
    write_jsonl(args.out / "clipping_audit.jsonl", sorted(window_audit, key=lambda w: w["ui_index"]))
    write_json(args.out / "report.json", report)
    print({"pooled": {k: v["pooled"] for k, v in results.items()}, "changes": changes,
           "helped": report["helped_windows"], "hurt": report["hurt_windows"],
           "key_windows": report["key_windows"], "seconds": report["fit_evaluate_seconds"]}, flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    report = read_json(args.out / "report.json")
    paired = read_jsonl(args.out / "paired_windows.jsonl")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 31)
    sheet = Image.new("RGB", (1400, 750), "#f4f6fa")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 20), "固定 ±5σ 裁剪：仅局部运动响应识别", font=title, fill="#182c43")
    draw.text((30, 78), "同45窗 / 10个整轨迹留一折；只改标准化后裁剪，训练与测试一致；阈值仍为0.5。", font=font, fill="#34475c")
    cols = [30, 355, 560, 770, 980, 1190]
    for x, text in zip(cols, ["方法", "准确率", "平衡准确率", "左任务BA", "右任务BA", "轨迹宏准确率"]):
        draw.text((x, 140), text, font=small, fill="#34475c")
    for i, method in enumerate((BASE, CLIPPED)):
        r = report["methods"][method]
        values = [r["pooled"]["accuracy"], r["pooled"]["balanced_accuracy"], r["by_task"]["left"]["balanced_accuracy"],
                  r["by_task"]["right"]["balanced_accuracy"], r["episode_macro_accuracy"]]
        for x, text in zip(cols, [NAMES[method], *[f"{v:.2%}" for v in values]]):
            draw.text((x, 190 + i * 55), text, font=font, fill="#182c43")
    cm = report["methods"][CLIPPED]["pooled"]["confusion_matrix_true_rows_predicted_columns"]
    draw.text((30, 320), f"裁剪后混淆矩阵：{cm}  （行=人工，列=预测；顺序：无明显推进 / 推进）", font=small, fill="#34475c")
    draw.text((30, 368), f"改对窗口：{report['helped_windows']}     改错窗口：{report['hurt_windows']}     轨迹改善/不变/退化：{report['episodes_improved_tied_worse']}", font=font, fill="#182c43")
    for i, p in enumerate(report["key_windows"]):
        draw.text((30, 430 + 53 * i), f"第{p['ui_index']}窗 · 人工{LABEL_ZH[p['target']]}：{p['baseline_score']:.4f} → {p['clipped_score']:.4f}；"
                  f"{LABEL_ZH[p['baseline_prediction']]} → {LABEL_ZH[p['clipped_prediction']]}", font=font, fill="#182c43")
    draw.text((30, 570), "分数不是概率；本轮没有重新拟合融合组、改区域、改标注或重训PI05。", font=font, fill="#34475c")
    draw.text((30, 622), "这是已用于开发的45窗诊断，不是独立最终验证；局部运动仍不等于导丝推进或碰壁。", font=font, fill="#a13b23")
    sheet.save(args.out / "comparison.png")
    parts = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>固定5σ裁剪消融</title>",
             "<style>body{font:18px 'Microsoft YaHei',sans-serif;max-width:1400px;margin:24px auto;background:#f4f6fa;color:#182c43}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #ccd5e0}.helped{background:#e2f5e9}.hurt{background:#ffe9e1}</style>",
             "<h1>仅局部运动：固定±5σ裁剪</h1><img src='comparison.png' alt='固定裁剪与原模型比较'>",
             "<p>下面列出全部45窗，不只列有利样本。浅绿=改对，浅红=改错。当前不更换部署模型、不继续搜索阈值。</p>",
             "<p><a href='../real10_local_motion_error_audit_v1/index.html'>先前只读图像审查</a> · <a href='report.json'>完整指标与裁剪审计</a></p>",
             "<table><tr><th>窗口</th><th>任务/轨迹</th><th>人工标签</th><th>原分数</th><th>裁剪分数</th><th>原预测</th><th>裁剪预测</th></tr>"]
    for p in paired:
        task = "左" if p["task"] == "left" else "右"
        cells = [str(p["ui_index"]), task + p["episode"][-3:], LABEL_ZH[p["target"]], f"{p['baseline_score']:.4f}",
                 f"{p['clipped_score']:.4f}", LABEL_ZH[p["baseline_prediction"]], LABEL_ZH[p["clipped_prediction"]]]
        parts.append(f"<tr class='{p['effect']}'>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    parts.append("</table></html>")
    (args.out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print({"page": str(args.out / "index.html"), "image": str(args.out / "comparison.png"), "refit": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("evaluate", "render"), required=True)
    parser.add_argument("--reference", type=Path, default=Path("simulation_output/real10_response_local_motion_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_response_motion_clip5_v1"))
    args = parser.parse_args()
    {"evaluate": evaluate, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
