"""Three predeclared new initializations of the unchanged window-response pair.

Reuse frozen features and folds; report EVERY seed. No model/threshold selection.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
import run_real10_window_response as base


SCHEMA = "real10_window_response_seed_retest_v1"
NEW_SEEDS = (20261020, 20261120, 20261220)
FOCUS = (4, 39, 45, 56, 59, 62, 65, 77, 100)
COPIED = ("features.npz", "annotation_snapshot.jsonl", "folds.json", "prepared.json")


def validate_reference(args):
    import torch
    protocol = read_json(args.source / "protocol.json")
    reference = read_json(args.source / "report.json")
    if (protocol["parameters"] != base.PARAMETERS or protocol["schema"] != base.SCHEMA
            or reference["status"] != "completed" or reference["improved_ui_indices"] != [45, 56, 59, 77]):
        raise ValueError("expected frozen completed discovery run")
    rows = read_jsonl(args.source / "annotation_snapshot.jsonl")
    folds = read_json(args.source / "folds.json")
    current = {r["window_id"]: r for r in read_jsonl(args.pack / "annotations_joint_v2.jsonl")}
    if len(rows) != 45 or len(folds) != 10 or folds != base.folds_for(rows) or current != {r["window_id"]: r["human_annotation"] for r in rows}:
        raise ValueError("original human supervision / episode split changed")
    previous = ast.parse((args.source / "entrypoint_snapshot.py").read_text(encoding="utf-8"))
    latest = ast.parse(Path(base.__file__).read_text(encoding="utf-8"))
    for name in ("make_model", "pool_logits", "normalize", "grouped"):
        old = next(n for n in previous.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in latest.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(old) != ast.dump(new):
            raise ValueError(f"out-of-scope computation change: {name}")
    with np.load(args.source / "features.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    oof = read_jsonl(args.source / "oof_predictions.jsonl")
    error = 0.
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for number, fold in enumerate(folds):
        tr, te = np.array(fold["train_indices"]), np.array(fold["test_indices"])
        x, dt, mask, norm = base.normalize(data["features"], data["dt"], data["lengths"], tr)
        tensors = [torch.as_tensor(a[te], device=args.device) for a in (x, dt, data["task"], mask)]
        for arm in base.ARMS:
            saved = torch.load(args.source / "fold_models" / arm / f"fold_{number:02d}.pt", map_location=args.device, weights_only=True)
            for key, value in norm.items():
                if not np.array_equal(np.asarray(value), saved["normalization"][key].cpu().numpy()):
                    raise ValueError("frozen normalization no longer matches")
            model = base.make_model().to(args.device).eval()
            model.load_state_dict(saved["state_dict"], strict=True)
            with torch.inference_mode():
                score = torch.sigmoid(model(*tensors, arm)[0]).cpu().numpy()
            old = np.array([oof[i]["methods"][arm]["score"] for i in te])
            error = max(error, float(np.abs(score-old).max()))
            if not np.allclose(score, old, atol=1e-7, rtol=0) or not np.array_equal(score >= .5, old >= .5):
                raise ValueError("original saved models no longer reproduce OOF")
    return {"reference_oof_replay_max_score_error": error, "models_replayed": 20,
            "core_functions_unchanged": True, "human_snapshot_matches_current": True,
            "original_fold_normalization_matches": True}


def descriptives(values):
    a = np.array(values, dtype=np.float64)
    return {"n_seeds": len(a), "mean": float(a.mean()), "sample_std_ddof1": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max())}


def summarize(reports):
    summary = {"methods": {}}
    for arm in base.ARMS:
        summary["methods"][arm] = {key: descriptives([r["methods"][arm]["pooled"][key] for r in reports])
                                   for key in ("accuracy", "balanced_accuracy", "recall_stationary", "recall_advance")}
        summary["methods"][arm]["episode_macro_accuracy"] = descriptives([r["methods"][arm]["episode_macro_accuracy"] for r in reports])
        summary["methods"][arm]["training_accuracy"] = descriptives([r["mean_training_accuracy_by_arm"][arm] for r in reports])
        summary["methods"][arm]["repeated_anchor_ba"] = descriptives([r["repeat_anchor_diagnostic"][arm]["metrics"]["pooled"]["balanced_accuracy"] for r in reports])
    summary["paired_delta_pp"] = {key: descriptives([r["paired_logmeanexp_minus_mean_pp"][key] for r in reports])
                                   for key in ("accuracy", "balanced_accuracy")}
    macro = [100*(r["methods"][base.ARMS[1]]["episode_macro_accuracy"]-r["methods"][base.ARMS[0]]["episode_macro_accuracy"]) for r in reports]
    summary["paired_delta_pp"]["episode_macro_accuracy"] = descriptives(macro)
    summary["positive_ba_delta_seeds"] = sum(r["paired_logmeanexp_minus_mean_pp"]["balanced_accuracy"] > 0 for r in reports)
    return summary


def run(args):
    import torch
    if args.out.exists():
        raise FileExistsError("preserve previous/partial retest; choose a fresh --out")
    validation = validate_reference(args)
    args.out.mkdir(parents=True)
    original_protocol = read_json(args.source / "protocol.json")
    protocol = {"schema": SCHEMA, "frozen_before_new_fits_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(args.source), "discovery_seed_read_only": base.SEED, "new_seeds": list(NEW_SEEDS),
                "parameters_except_seed": {k: v for k, v in base.PARAMETERS.items() if k != "seed"},
                "primary_confirmation_group": "new_seeds_only", "seed_ranges_nonoverlapping": True,
                "full_training_budget": "60 new models x 200 updates = 12000; no extra checkpoint/seed selection",
                "data": "same frozen 45 windows, 10 episodes, revised labels, cached features; no re-encoding",
                "summary": "arithmetic mean and sample SD(ddof=1) of metrics, NOT an ensemble or additional independent samples",
                "direction_rule": "all 3 new paired BA deltas > 0 AND mean episode-macro accuracy delta >= 0",
                "focus_ui_indices": list(FOCUS), "scope": "initialization sensitivity, not independent validation",
                "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False, "real_system_validated": False}
    write_json(args.out / "protocol.json", protocol)
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    shutil.copyfile(base.__file__, args.out / "training_entrypoint_snapshot.py")
    shutil.copyfile("docs/algorithm-real10-window-response-seed-protocol-20260922.md", args.out / "design_protocol.md")
    shutil.copyfile(args.source / "annotation_snapshot.jsonl", args.out / "annotation_snapshot.jsonl")
    shutil.copyfile(args.source / "folds.json", args.out / "folds.json")
    started = time.perf_counter()
    all_reports = [read_json(args.source / "report.json")]
    all_oof = [read_jsonl(args.source / "oof_predictions.jsonl")]
    checks = []
    for seed in NEW_SEEDS:
        destination = args.out / f"seed_{seed}"
        destination.mkdir()
        for name in COPIED:
            shutil.copyfile(args.source / name, destination / name)
        per_seed = {**original_protocol, "parameters": {**base.PARAMETERS, "seed": seed},
                    "seed_retest": {"schema": SCHEMA, "source": str(args.source), "changes": "initialization base_seed only",
                                    "frozen_before_fit_utc": datetime.now(timezone.utc).isoformat()}}
        write_json(destination / "protocol.json", per_seed)
        print(json.dumps({"start_base_seed": seed}), flush=True)
        base.train(SimpleNamespace(out=destination, device=args.device), base_seed=seed)
        result, predictions = read_json(destination / "report.json"), read_jsonl(destination / "oof_predictions.jsonl")
        if result["methods"]["controller_ridge"] != all_reports[0]["methods"]["controller_ridge"]:
            raise ValueError("seed-independent reference changed")
        if any((destination/n).read_bytes() != (args.source/n).read_bytes() for n in COPIED):
            raise ValueError("copied input changed during fitting")
        if any(any(a[k] != b[k] for k in ("window_id", "ui_index", "label", "episode", "task")) for a, b in zip(predictions, all_oof[0])):
            raise ValueError("OOF row identity differs from discovery")
        fold_logs = read_jsonl(destination / "fold_log.jsonl")
        if len(fold_logs) != 20 or any(r["seed"] != seed+r["fold"] or r["updates"] != 200 or not r["same_pair_initialization"] for r in fold_logs):
            raise ValueError("unmatched seed or training budget")
        for number in range(10):
            for arm in base.ARMS:
                relative = Path("fold_models")/arm/f"fold_{number:02d}.pt"
                before = torch.load(args.source/relative, map_location="cpu", weights_only=True)
                after = torch.load(destination/relative, map_location="cpu", weights_only=True)
                if any(not torch.equal(before["normalization"][k], after["normalization"][k]) for k in before["normalization"]):
                    raise ValueError("seed unexpectedly changes normalization")
        checks.append({"seed": seed, "inputs_unchanged": True, "normalization_unchanged": True,
                       "controller_reference_unchanged": True, "paired_initialization_and_budget": True,
                       "maximum_model_replay_error": max(r["replay_max_logit_error"] for r in fold_logs)})
        all_reports.append(result)
        all_oof.append(predictions)
    seed_ids = [base.SEED, *NEW_SEEDS]
    per_seed = []
    for seed, report in zip(seed_ids, all_reports):
        per_seed.append({"base_seed": seed, "new_confirmation_seed": seed in NEW_SEEDS,
                         "methods": report["methods"], "paired_delta_pp": report["paired_logmeanexp_minus_mean_pp"],
                         "improved_ui_indices": report["improved_ui_indices"], "worsened_ui_indices": report["worsened_ui_indices"],
                         "episode_win_tie_loss": report["episode_win_tie_loss"],
                         "mean_training_accuracy_by_arm": report["mean_training_accuracy_by_arm"],
                         "repeat_anchor_diagnostic": report["repeat_anchor_diagnostic"]})
    stability = []
    for i, original in enumerate(all_oof[0]):
        versions = []
        for seed, predictions in zip(seed_ids, all_oof):
            row = predictions[i]
            correct = [row["methods"][a]["prediction"] == row["label"] for a in base.ARMS]
            versions.append({"seed": seed, "methods": row["methods"], "correct": correct,
                             "improvement": not correct[0] and correct[1], "regression": correct[0] and not correct[1]})
        stability.append({"window_id": original["window_id"], "ui_index": original["ui_index"],
                          "label": original["label"], "episode": original["episode"], "task": original["task"],
                          "new_seed_improvement_count": sum(r["improvement"] for r in versions[1:]),
                          "new_seed_regression_count": sum(r["regression"] for r in versions[1:]), "per_seed": versions})
    primary = summarize(all_reports[1:])
    consistent = primary["positive_ba_delta_seeds"] == 3 and primary["paired_delta_pp"]["episode_macro_accuracy"]["mean"] >= 0
    report = {"schema": SCHEMA, "status": "completed", "per_seed": per_seed,
              "new_seeds_only": primary, "all_four_seeds_descriptive": summarize(all_reports),
              "direction_consistent_under_predeclared_rule": consistent,
              "new_training_seconds_including_checks": time.perf_counter()-started,
              "new_models": 60, "new_optimizer_updates": 12000, "images_encoded": 0,
              "visual_status": "not_viewed", "policy_input_allowed": False, "formal_data_allowed": False,
              "deployable": False, "real_system_validated": False}
    write_jsonl(args.out / "window_stability.jsonl", stability)
    write_json(args.out / "validation.json", {**validation, "new_seed_checks": checks})
    write_json(args.out / "report.json", report)
    print(json.dumps({"status": "completed", "new_seeds_only": primary, "consistent": consistent}), flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    result = read_json(args.out / "report.json")
    stability = read_jsonl(args.out / "window_stability.jsonl")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    image = Image.new("RGB", (1500, 850), "#f4f6f9")
    d = ImageDraw.Draw(image)
    d.text((30, 20), "固定随机种子复核：所有结果保留，不挑最好的一次", font=title, fill="#173049")
    d.text((30, 74), "同45窗 / 10折 / 4937可训参数 / 每折200步；新增3次为主复核，原探索结果只读参照", font=font, fill="#34495d")
    headings = [(30, "基础种子"), (270, "平均：准确率 / BA"), (600, "突变：准确率 / BA"),
                (950, "BA差(pp)"), (1150, "改对 / 改错")]
    for x, heading in headings:
        d.text((x, 132), heading, font=font, fill="#173049")
    lines = ["# 固定种子复核", "", "相同45个人工窗口，不是新增独立样本或集成模型。", "",
             "| seed | 范围 | 平均准确率 | 平均BA | 突变准确率 | 突变BA | BA差pp | 改对/改错 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for i, run in enumerate(result["per_seed"]):
        a, b = [run["methods"][arm]["pooled"] for arm in base.ARMS]
        label = "新增" if run["new_confirmation_seed"] else "原探索"
        values = [f"{run['base_seed']} ({label})", f"{a['accuracy']:.2%} / {a['balanced_accuracy']:.2%}",
                  f"{b['accuracy']:.2%} / {b['balanced_accuracy']:.2%}", f"{run['paired_delta_pp']['balanced_accuracy']:+.2f}",
                  f"{len(run['improved_ui_indices'])} / {len(run['worsened_ui_indices'])}"]
        for (x, _), value in zip(headings, values):
            d.text((x, 190+64*i), value, font=small, fill="#173049")
        lines.append(f"| {run['base_seed']} | {label} | {a['accuracy']:.2%} | {a['balanced_accuracy']:.2%} | {b['accuracy']:.2%} | {b['balanced_accuracy']:.2%} | {run['paired_delta_pp']['balanced_accuracy']:+.2f} | {values[-1]} |")
    primary = result["new_seeds_only"]
    details = []
    for arm in base.ARMS:
        m = primary["methods"][arm]
        details.append(f"新增3次 {base.NAMES[arm]}：BA {m['balanced_accuracy']['mean']:.2%} ± {100*m['balanced_accuracy']['sample_std_ddof1']:.2f} pp；轨迹宏准确率 {m['episode_macro_accuracy']['mean']:.2%}")
    delta = primary["paired_delta_pp"]["balanced_accuracy"]
    details += [f"新增3次配对BA差：{delta['mean']:+.2f} ± {delta['sample_std_ddof1']:.2f} pp；范围 [{delta['min']:+.2f}, {delta['max']:+.2f}] pp",
                f"新增正向seed：{primary['positive_ba_delta_seeds']}/3；预设方向一致条件：{'满足' if result['direction_consistent_under_predeclared_rule'] else '未满足'}",
                "控制器参照固定BA67.00%、轨迹宏65.83%；输入不同，不作纯架构归因。",
                "±为跨seed样本标准差，不是置信区间。三个seed仍只覆盖原45窗/10条轨迹。",
                "保留全部成功与失败；不是动作前预测、碰壁识别、策略收益或实机验证。"]
    for i, value in enumerate(details):
        d.text((30, 480+47*i), value, font=small, fill="#34495d")
    image.save(args.out / "seed_comparison_zh.png")
    lines += ["", *details, ""]
    sheet = Image.new("RGB", (1500, 850), "#f4f6f9")
    d = ImageDraw.Draw(sheet)
    d.text((30, 20), "固定难例与原4个改对窗口：逐seed分类稳定性", font=title, fill="#173049")
    d.text((30, 75), "每格：平均汇聚 → 突变汇聚；绿色=改对，红色=改错，灰色=分类相同。不是重新标注。", font=small, fill="#34495d")
    d.text((30, 126), "窗口 / 人工真值", font=font, fill="#173049")
    for j, run in enumerate(result["per_seed"]):
        d.text((325+j*280, 126), str(run["base_seed"]), font=font, fill="#173049")
    short = {"stationary": "无推进", "advance": "推进"}
    focus = [r for r in stability if r["ui_index"] in FOCUS]
    for i, row in enumerate(focus):
        y = 186+66*i
        d.text((30, y), f"#{row['ui_index']} / {short[row['label']]}", font=font, fill="#173049")
        for j, version in enumerate(row["per_seed"]):
            x = 305+j*280
            fill = "#d8eee1" if version["improvement"] else "#f6deda" if version["regression"] else "#e7ebf0"
            d.rounded_rectangle((x, y-5, x+264, y+42), radius=6, fill=fill)
            text = " → ".join(short[version["methods"][arm]["prediction"]] for arm in base.ARMS)
            d.text((x+15, y+2), text, font=small, fill="#173049")
    d.text((30, 802), "原改对窗口为#45/#56/#59/#77；所有45窗逐seed分数见window_stability.jsonl，不丢弃退化样本。", font=small, fill="#34495d")
    sheet.save(args.out / "case_stability_zh.png")
    lines += ["## 原四个改对窗口的新增seed复现", ""]
    for row in stability:
        if row["ui_index"] in (45, 56, 59, 77):
            lines.append(f"- #{row['ui_index']}：新增3次中改对{row['new_seed_improvement_count']}次，改错{row['new_seed_regression_count']}次。")
    (args.out / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>响应汇聚随机种子复核</title><style>body{max-width:1500px;margin:24px auto;background:#f4f6f9;font:18px system-ui}img{width:100%}pre{white-space:pre-wrap}</style>'
    (args.out / "index.html").write_text(page+'<img src="seed_comparison_zh.png"><img src="case_stability_zh.png"><pre>'+html.escape("\n".join(lines))+"</pre></html>", encoding="utf-8")
    print(json.dumps({"rendered": ["seed_comparison_zh.png", "case_stability_zh.png"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("run", "render"), required=True)
    p.add_argument("--source", type=Path, default=Path("simulation_output/real10_window_response_mil_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_window_response_seed_retest_v1"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    if args.stage == "run":
        run(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
