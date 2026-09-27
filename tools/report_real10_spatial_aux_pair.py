"""Read back the completed fixed experiment and render its real OOF exports.

No model execution, checkpoint changes, threshold search or new supervision.
Run locally with the repository .venv after fetching the small exported files.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


ARMS = ("spatial_window", "spatial_window_tip_aux")
SEEDS = (20261020, 20261120, 20261220)
COLORS = ("#0077BB", "#EE7733")
NAMES = ("仅窗口监督", "+ 尖端辅助监督")
VIEWS = ("side", "top")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def verify(out):
    report = read_json(out / "report.json")
    config = read_json(out / "run_protocol.json")
    rows = read_jsonl(out / "source/annotation_snapshot.jsonl")
    targets = read_jsonl(out / "source/supervision.jsonl")
    folds = read_json(out / "source/folds.json")
    predictions = read_jsonl(out / "oof_predictions.jsonl")
    points = read_jsonl(out / "oof_tip_locations.jsonl")
    fits = read_jsonl(out / "fit_records.jsonl")
    audit = {(r["window_id"], r["frame_index"], r["view"]): r
             for r in read_jsonl(out / "source/point_audit.jsonl") if r["tip_target_valid"]}
    if report["status"] != "complete_fixed_pair" or len(rows) != 45 or len(folds) != 10:
        raise ValueError("expected the completed fixed 45-window experiment")
    for frozen, current in (("runner_snapshot.py", "tools/train_real10_spatial_aux_pair.py"),
                            ("model_snapshot.py", "tools/real10_spatial_response.py")):
        if (out / frozen).read_bytes() != Path(current).read_bytes():
            raise ValueError("current training code differs from the saved run")
    for name in ("model_inputs.jsonl", "supervision.jsonl", "annotation_snapshot.jsonl",
                 "point_audit.jsonl", "folds.json"):
        if (out / "source" / name).read_bytes() != (Path(config["source"]) / name).read_bytes():
            raise ValueError(f"frozen source differs: {name}")
    latest = {r["window_id"]: r for r in read_jsonl(
        Path("simulation_output/real10_event_windows_v1/annotations_joint_v2.jsonl"))}
    if latest != {r["window_id"]: r["human_annotation"] for r in rows}:
        raise ValueError("latest human responses differ from this run's snapshot")
    expected_fits = {(s, f, a) for s in SEEDS for f in range(10) for a in ARMS}
    if len(fits) != 60 or {(r["base_seed"], r["fold_index"], r["arm"]) for r in fits} != expected_fits:
        raise ValueError("missing or duplicated fit record")
    for fit in fits:
        indices = folds[fit["fold_index"]]["train_indices"]
        positives = sum(targets[i]["response_y"] for i in indices)
        tip_count = sum(sum(sum(v) for v in targets[i]["tip_valid"]) for i in indices)
        if (fit["optimizer_steps"] != 200 or fit["trainable_parameters"] != 1554 or
                fit["train_windows"] != len(indices) or fit["train_tip_points"] != tip_count):
            raise ValueError("fit budget or training-only target count differs")
        if not np.isclose(fit["train_pos_weight"], (len(indices)-positives)/positives):
            raise ValueError("class weight does not match training fold")
        if any(v != 0 for v in fit["saved_replay_max_abs_error"].values()):
            raise ValueError("saved-model replay is not exact")
    if len(predictions) != 270 or len(points) != 150 or len(audit) != 25:
        raise ValueError("unexpected response or localization coverage")
    seed_lookup = {r["base_seed"]: r for r in report["per_seed"]}
    changed, localization = [], {view: {} for view in VIEWS}
    csv_rows = []
    for seed in SEEDS:
        arm_preds = {}
        for arm in ARMS:
            batch = sorted((r for r in predictions if r["base_seed"] == seed and r["arm"] == arm),
                           key=lambda r: r["sample_index"])
            if [r["sample_index"] for r in batch] != list(range(45)):
                raise ValueError("every window must occur once per seed/arm")
            matrix = np.zeros((2, 2), dtype=int)
            for r in batch:
                i = r["sample_index"]
                if (r["window_id"] != rows[i]["window_id"] or r["response_y"] != targets[i]["response_y"] or
                        r["source_episode"] != folds[r["fold_index"]]["heldout_episode"] or
                        i not in folds[r["fold_index"]]["test_indices"] or
                        r["prediction"] != int(r["logit"] >= 0)):
                    raise ValueError("OOF identity, target, threshold or fold mismatch")
                matrix[r["response_y"], r["prediction"]] += 1
            accuracy = np.trace(matrix)/45
            ba = np.mean(matrix.diagonal()/matrix.sum(1))
            macro = np.mean([np.mean([r["response_y"] == r["prediction"] for r in batch
                                     if r["source_episode"] == f["heldout_episode"]]) for f in folds])
            reported = seed_lookup[seed]["arms"][arm]
            if matrix.tolist() != reported["confusion_matrix_true_rows_predicted_columns"]:
                raise ValueError("reported confusion matrix differs from OOF rows")
            for metric, value in (("accuracy", accuracy), ("balanced_accuracy", ba),
                                  ("episode_macro_accuracy", macro)):
                if not np.isclose(value, reported[metric], atol=1e-12):
                    raise ValueError("reported response metric differs from OOF rows")
                csv_rows.append({"base_seed": seed, "arm": arm, "metric": metric,
                                 "value": value*100, "unit": "percent"})
            arm_points = [r for r in points if r["base_seed"] == seed and r["arm"] == arm]
            if len(arm_points) != 25 or {(r["window_id"], r["frame_index"], r["view"])
                                       for r in arm_points} != set(audit):
                raise ValueError("heldout tip coverage differs from existing human points")
            for p in arm_points:
                label = audit[p["window_id"], p["frame_index"], p["view"]]
                if (p["target_xy_original"] != label["wire"]["xy"] or
                        label["source_episode"] != folds[p["fold_index"]]["heldout_episode"]):
                    raise ValueError("tip target/fold mismatch")
                error = np.linalg.norm(np.array(p["predicted_xy_original"])-p["target_xy_original"])
                if not np.isclose(error, p["error_px"], atol=1e-5):
                    raise ValueError("original-pixel point error mismatch")
            for view in VIEWS:
                vals = [r for r in arm_points if r["view"] == view]
                value = np.mean([np.mean([r["error_px"] for r in vals if r["window_id"] == w])
                                 for w in sorted({r["window_id"] for r in vals})])
                if not np.isclose(value, reported["localization"][view]["window_macro_error_px"]):
                    raise ValueError("per-window localization summary mismatch")
                localization[view].setdefault(arm, []).append(float(value))
                csv_rows.append({"base_seed": seed, "arm": arm, "metric": f"{view}_tip_window_macro",
                                 "value": value, "unit": "original_image_px"})
            arm_preds[arm] = batch
        delta = {metric: seed_lookup[seed]["arms"][ARMS[1]][metric]-seed_lookup[seed]["arms"][ARMS[0]][metric]
                 for metric in ("accuracy", "balanced_accuracy", "episode_macro_accuracy")}
        for metric, value in delta.items():
            if not np.isclose(value, seed_lookup[seed]["paired_aux_minus_window"][metric]):
                raise ValueError("paired delta mismatch")
        improved, regressed = [], []
        for base, aux in zip(arm_preds[ARMS[0]], arm_preds[ARMS[1]]):
            b, a = base["prediction"] == base["response_y"], aux["prediction"] == aux["response_y"]
            if a and not b:
                improved.append(base["ui_index"])
            if b and not a:
                regressed.append(base["ui_index"])
        changed.append({"base_seed": seed, "improved_ui": improved, "regressed_ui": regressed})
    for view, values in localization.items():
        a, b = np.array(values[ARMS[0]]), np.array(values[ARMS[1]])
        values["mean_aux_minus_window_px"] = float((b-a).mean())
        values["relative_mean_error_change_percent"] = float((b.mean()/a.mean()-1)*100)
    result = {"status": "verified_from_exported_oof_rows", "fits": 60, "steps_per_fit": 200,
              "distinct_windows": 45, "trajectory_groups": 10, "response_rows": 270, "point_rows": 150,
              "training_source_snapshots_unchanged": True, "human_responses_current": True,
              "replay_evidence": "60 saved fit records report exact checkpoint replay; no reinference this turn",
              "all_reported_response_and_localization_metrics_recomputed": True,
              "changed_classifications": changed, "localization_by_seed_px": localization,
              "decision": "no stable gain; do not adopt as automatic response supervision or policy/world-model input",
              "formal_training_this_readback": False, "new_supervision_created": False,
              "cluster_note": "45 windows nested in 10 trajectories, not 45 independent trajectories"}
    write_json(out / "readback.json", result)
    return report, rows, points, csv_rows, result


def render(out, report, rows, points, csv_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from PIL import Image

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.is_file():
        raise FileNotFoundError("local rendering expects the existing Microsoft YaHei font")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 450, "svg.fonttype": "path"})
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "data-manifest.md").write_text(
        "# 真实结果绘图清单\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "|---|---|---|---|---|---|\n"
        "| 固定配对指标 | paired_metrics.csv | real | ../report.json；逐行OOF回算通过 | "
        "tools/report_real10_spatial_aux_pair.py | spatial_aux_results_zh.png/.svg |\n"
        "| 全部已有tip目标对照 | ../oof_tip_locations.jsonl | real | "
        "../source/model_inputs.jsonl、point_audit.jsonl与旧ROI224 | 同上 | tip_case_*_zh.png/.svg |\n\n"
        "固定三seed全部展示，不选最佳seed；602图不重编码，仅读取25个有标注点对应的旧ROI。\n"
        "25点覆盖5个开发窗，Side 13点/5窗，Top 12点/4窗。定位按原图像素，不合成毫米。\n"
        "图中坐标为注意力空间期望读出，不是heatmap峰值、跟踪轨迹、可见性判断或运动向量。\n"
        "#4只有材料点，不纳入tip图；#77 Side仅末帧有tip，不填补缺失帧。\n"
        "两臂结构/预算相同，仅定位损失权重0/0.1不同；反复使用的开发集，不是独立实机测试。\n",
        encoding="utf-8")
    with (figures / "paired_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("base_seed", "arm", "metric", "value", "unit"))
        writer.writeheader()
        writer.writerows(csv_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for i, seed_report in enumerate(report["per_seed"]):
        values = [seed_report["arms"][a]["balanced_accuracy"]*100 for a in ARMS]
        axes[0].plot([i-.12, i+.12], values, color="#AAAAAA", lw=1.2, zorder=1)
        for a in range(2):
            axes[0].scatter(i+(-.12 if a == 0 else .12), values[a], color=COLORS[a], s=42, zorder=2)
            axes[0].annotate(f"{values[a]:.1f}", (i+(-.12 if a == 0 else .12), values[a]),
                             xytext=(-8 if a == 0 else 8, -14 if a == 0 else 7),
                             textcoords="offset points", ha="center", fontsize=8, color=COLORS[a])
        axes[0].text(i, 80, f"差值 {values[1]-values[0]:+.1f} pp", ha="center", fontsize=9)
    axes[0].axhline(50, color="#777777", ls="--", lw=.8)
    axes[0].set(xticks=range(3), xticklabels=[str(s) for s in SEEDS], ylim=(0, 100),
                ylabel="整轨迹留出 · 平衡准确率（%）", xlabel="固定随机种子",
                title="响应识别：平均 +2.5 pp，但方向不一致")
    axes[0].text(.02, .03, "虚线：恒定类别预测的 BA = 50%", transform=axes[0].transAxes,
                 fontsize=8, color="#555555")
    for v, view in enumerate(VIEWS):
        for i, seed_report in enumerate(report["per_seed"]):
            x = v + (i-1)*.19
            values = [seed_report["arms"][a]["localization"][view]["window_macro_error_px"] for a in ARMS]
            axes[1].plot([x-.045, x+.045], values, color="#AAAAAA", lw=1)
            for a in range(2):
                axes[1].scatter(x+(-.045 if a == 0 else .045), values[a], color=COLORS[a],
                                marker=("o", "s", "^")[i], s=30, zorder=2)
    axes[1].set(xticks=range(2), xticklabels=["Side：13点 / 5窗", "Top：12点 / 4窗"],
                ylim=(0, 450), ylabel="留出尖端误差（原图 px，先逐窗平均）",
                title="定位读出：两个视角均未一致改善")
    axes[1].text(.03, .03, "圆 / 方 / 三角：三个固定种子；越低越好", transform=axes[1].transAxes,
                 fontsize=8, color="#555555")
    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", alpha=.5)
        ax.set_axisbelow(True)
    legend = [Line2D([0], [0], marker="o", color=c, label=n, lw=0) for c, n in zip(COLORS, NAMES)]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
    fig.suptitle("Real10 空间定位辅助监督：固定同结构配对结果", y=.995, fontsize=14)
    fig.text(.5, .025, "45窗 / 10条轨迹 · 3种子 · 两组各200步 · 开发集重复留出，不是PI05或实机成功率", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .87), w_pad=2)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"spatial_aux_results_zh.{ext}", dpi=450)
    plt.close(fig)

    inputs = {r["window_id"]: r for r in read_jsonl(out / "source/model_inputs.jsonl")}
    roi = read_json(out / "source/protocol.json")["roi_xyxy_original"]
    point_lookup = {(p["window_id"], p["frame_index"], p["view"], p["base_seed"], p["arm"]): p for p in points}
    cases = sorted({(p["ui_index"], p["window_id"]) for p in points})
    cards = []
    for ui, window in cases:
        keys = sorted({(p["frame_index"], p["view"]) for p in points if p["window_id"] == window},
                      key=lambda x: (VIEWS.index(x[1]), x[0]))
        fig, axes = plt.subplots(len(keys), 3, figsize=(8.8, len(keys)*2.5+1.35), squeeze=False)
        for r, (frame_index, view) in enumerate(keys):
            path = inputs[window]["model_input"]["sequence"][frame_index]["images"][view]
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            x0, y0, x1, y1 = roi[view]
            def plot_xy(xy):
                return (np.array(xy)-[x0, y0]+.5)/[x1-x0, y1-y0]*224-.5
            for c, seed in enumerate(SEEDS):
                ax = axes[r, c]
                pair = [point_lookup[window, frame_index, view, seed, a] for a in ARMS]
                ax.imshow(pixels)
                truth = plot_xy(pair[0]["target_xy_original"])
                ax.scatter(*truth, marker="x", color="#009988", linewidths=2.1, s=95, zorder=4)
                for a, p in enumerate(pair):
                    xy = plot_xy(p["predicted_xy_original"])
                    ax.scatter(*xy, facecolors="none", edgecolors="white", linewidths=2.5, s=100)
                    ax.scatter(*xy, marker=("o", "s")[a], facecolors="none", edgecolors=COLORS[a],
                               linewidths=1.7, s=80, zorder=5)
                ax.set(xticks=[], yticks=[], xlim=(-.5, 223.5), ylim=(223.5, -.5))
                if r == 0:
                    ax.set_title(f"seed {seed}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{view.capitalize()} · 帧{frame_index}", fontsize=10)
                ax.set_xlabel(f"无辅助 {pair[0]['error_px']:.0f} px  |  有辅助 {pair[1]['error_px']:.0f} px", fontsize=9)
        handles = [Line2D([0], [0], marker="x", color="#009988", lw=0, label="人工尖端")]
        handles += [Line2D([0], [0], marker=m, markerfacecolor="none", color=c, lw=0, label=n)
                    for m, c, n in zip(("o", "s"), COLORS, NAMES)]
        row = next(x for x in rows if x["window_id"] == window)
        label = "推进" if row["human_annotation"]["joint_motion_response"] == "advance" else "无推进"
        fig.suptitle(f"#{ui} · {label} · 全部已标注tip帧的留出坐标读出", y=.995, fontsize=13)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .975), ncol=3, frameon=False)
        fig.text(.5, .012, "使用模型原ROI224；误差按原图像素计算。读出不是跟踪轨迹，不补缺失点。", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, .947), h_pad=1, w_pad=.9)
        name = f"tip_case_{ui}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=450)
        plt.close(fig)
        cards.append(f'<section><h2>#{ui}：{label}</h2><p>{html.escape(window)}</p>'
                     f'<a href="figures/{name}.svg">SVG</a><img src="figures/{name}.png"></section>')
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>空间辅助监督配对回读</title>'
        '<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1100px;margin:30px auto;'
        'background:#f5f7fa;color:#223}section{background:white;padding:20px;margin:24px 0}'
        'img{width:100%;height:auto}p{overflow-wrap:anywhere;line-height:1.7}</style>'
        '<h1>空间辅助监督：完整配对结果</h1><p>60次拟合全部完成。平均BA +2.5个百分点，但'
        '三个种子为 +12.5 / +7.0 / −12.0，未满足一致性条件；不采用为自动响应标签或PI05/world model输入。</p>'
        '<p>所有图来自真实OOF结果，不挑seed。原45窗属于反复使用的开发集；25点仅覆盖5个窗。'
        '#4材料点排除，#77缺失帧不回填。此页面只读，无标注写回接口。</p>'
        '<img src="figures/spatial_aux_results_zh.png"><a href="figures/spatial_aux_results_zh.svg">结果SVG</a>'
        + ''.join(cards) + '</html>', encoding="utf-8")
    review = out / "review.json"
    if not review.exists():
        write_json(review, {"visual_status": "not_viewed", "user_acceptance": "pending",
                            "scope": "new paired result plot and all five tip-window comparison sheets"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    args = parser.parse_args()
    report, rows, points, csv_rows, result = verify(args.out)
    render(args.out, report, rows, points, csv_rows)
    print(json.dumps({"status": result["status"], "changed": result["changed_classifications"],
                      "localization": result["localization_by_seed_px"], "index": str(args.out / "index.html")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
