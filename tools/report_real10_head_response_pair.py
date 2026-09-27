"""Render saved developmental LOEO predictions locally; no training or inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from fit_real10_head_region import read_json, read_jsonl, write_json

ARMS = ("tip_frozen", "head_frozen")
SEEDS = (20261020, 20261120, 20261220)
CASES = (4, 39, 62, 65, 77, 100)
NAMES = {"tip_frozen": "尖端定位→冻结→响应", "head_frozen": "头段定位→冻结→响应"}
COLORS = ("#0077BB", "#EE7733")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_head_response_pair_v1"))
    args = parser.parse_args()
    report, protocol = (read_json(args.out / f"{name}.json") for name in ("report", "protocol"))
    predictions = read_jsonl(args.out / "oof_predictions.jsonl")
    locations = read_jsonl(args.out / "heldout_localization.jsonl")
    if len(predictions) != 270 or len(locations) != 150:
        raise ValueError("require the complete fixed three-seed pair")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    from PIL import Image

    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "svg.fonttype": "path"})
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 冻结头段表示响应对照：真实开发集OOF结果\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "|---|---|---|---|---|---|\n"
        "| 三seed配对性能 | ../report.json, ../oof_predictions.jsonl | real | 原45窗、10整轨迹LOEO | "
        "tools/report_real10_head_response_pair.py | response_pair_summary_zh.png/.svg |\n"
        "| 六个固定难例 | ../oof_predictions.jsonl | real | #4/#39/#62/#65/#77/#100，全部seed | "
        "同上 | response_pair_hardcases_zh.png/.svg |\n"
        "| 25个共同标注留出帧 | ../heldout_localization.jsonl, ../heldout_heatmaps.npz | real | "
        "固定最小seed、原ROI224图 | 同上 | response_pair_case_*_zh.png/.svg |\n\n"
        "45窗嵌套于10轨迹，3seed不是135个独立样本。训练侧几何隔离，未跨fold加载拟合权重。\n"
        "响应标签仍推进/未推进；小幅推进和仅移动是原人工说明，不是新增类别。\n"
        "仍是反复使用的开发集，不是独立测试、动作前预测、真实接触或策略收益。\n"
        "图像热图是log10(p/均匀概率)，固定显示[-1,2]；粗线/框仅离线叠加，不是模型输入。\n",
        encoding="utf-8")
    lookup = {(r["base_seed"], r["arm"]): r for r in report["seed_reports"]}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.1))
    for ax, key, title in zip(axes, ("balanced_accuracy", "accuracy", "episode_macro_accuracy"),
                              ("平衡准确率 BA", "窗口准确率", "整轨迹宏准确率")):
        for k, seed in enumerate(SEEDS):
            values = [100*lookup[seed, arm][key] for arm in ARMS]
            ax.plot([0, 1], values, "o-", color=("#0077BB", "#EE7733", "#009988")[k],
                    label=f"seed {seed}", lw=1.3, ms=5)
        ax.set(title=title, xticks=[0, 1], xticklabels=["尖端冻结", "头段冻结"], ylabel="%",
               ylim=(0, 100), xlim=(-.2, 1.2))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .91), frameon=False, fontsize=8)
    fig.suptitle("冻结定位表示能否改善响应识别？原45窗整轨迹LOEO", y=.99, fontsize=12)
    fig.text(.5, .018, "相同25帧定位监督预算、2000定位+200响应更新；三次拟合不是新增独立数据。",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.75, bottom=.18, wspace=.34)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"response_pair_summary_zh.{ext}", dpi=450)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.8))
    hardcases = []
    for ax, ui in zip(axes.flat, CASES):
        chosen = [r for r in predictions if r["ui_index"] == ui]
        truth = chosen[0]["response_y"]
        for j, arm in enumerate(ARMS):
            values = [next(r["probability_advance"] for r in chosen
                           if r["base_seed"] == seed and r["arm"] == arm) for seed in SEEDS]
            ax.plot(range(3), values, "o-" if j == 0 else "s--", color=COLORS[j],
                    label=NAMES[arm], lw=1.2, ms=4)
            hardcases.append({"ui_index": ui, "truth": truth, "arm": arm, "seeds": list(SEEDS),
                              "probability_advance": values})
        note = {4: "小幅推进", 65: "有移动但未推进", 100: "小幅推进"}.get(ui, "")
        ax.axhline(.5, color="#888888", ls=":", lw=1)
        ax.set(title=f"#{ui} 真值：{'推进' if truth else '未推进'}"+(f"（{note}）" if note else ""),
               xticks=range(3), xticklabels=["seed 1", "seed 2", "seed 3"],
               ylabel="推进分数", ylim=(-.03, 1.03))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .94), frameon=False, fontsize=9)
    fig.suptitle("固定六个难例：全部为整轨迹留出预测，不选择最好seed", y=.99, fontsize=12)
    fig.text(.5, .018, "虚线为固定0.5阈值；输出不是校准置信度。原人工标签未改，不由头段位移生成推进标签。",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.82, bottom=.10, hspace=.45, wspace=.32)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"response_pair_hardcases_zh.{ext}", dpi=450)
    plt.close(fig)
    write_json(args.out / "hardcase_readback.json", hardcases)

    maps = np.load(args.out / "heldout_heatmaps.npz")
    links = []
    for ui in sorted({r["ui_index"] for r in locations}):
        chosen = sorted([r for r in locations if r["ui_index"] == ui and
                         r["base_seed"] == SEEDS[0] and r["arm"] == "head_frozen"],
                        key=lambda r: (r["view"], r["frame_index"]))
        result = {r["arm"]: r for r in predictions if r["ui_index"] == ui and r["base_seed"] == SEEDS[0]}
        fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 2.35*len(chosen)), squeeze=False)
        for i, row in enumerate(chosen):
            roi = protocol["roi_xyxy_original"][row["view"]]
            wh, origin = np.subtract(roi[2:], roi[:2]), np.asarray(roi[:2])-.5
            pair = {r["arm"]: r for r in locations if
                    (r["ui_index"], r["base_seed"], r["view"], r["frame_index"]) ==
                    (ui, SEEDS[0], row["view"], row["frame_index"])}
            for j, arm in enumerate(("raw", *ARMS)):
                ax = axes[i, j]
                with Image.open(row["image_path"]) as img:
                    ax.imshow(img.convert("RGB"), extent=(0, 224, 224, 0))
                title = f"{row['view'].capitalize()} 帧{row['frame_index']} · {row['coverage']}"
                if arm != "raw":
                    current = pair[arm]
                    ax.imshow(np.log10(np.maximum(maps[current["array_key"]]*3136, 1e-10)),
                              extent=(-1.5, 222.5, 222.5, -1.5), cmap="magma",
                              vmin=-1, vmax=2, alpha=.48, interpolation="nearest")
                    xy = (np.asarray(current["peak_xy_original"])-origin)/wh*224
                    ax.plot(*xy, "x", color="#EE7733", ms=8, mew=1.5)
                    title = ("尖端冻结" if arm == "tip_frozen" else "头段冻结")+(
                        f" · 域内{100*current['support_mass']:.1f}%\n距离{current['peak_distance_px']:.1f}px")
                geom = row["geometry"]
                if geom["kind"] == "bbox":
                    box = (np.asarray(geom["xyxy"]).reshape(2, 2)-origin)/wh*224
                    ax.add_patch(Rectangle(box[0], *(box[1]-box[0]), fill=False, edgecolor="#33BBEE", lw=1.3))
                else:
                    for segment in geom["segments"]:
                        xy = (np.asarray(segment)-origin)/wh*224
                        ax.plot(xy[:, 0], xy[:, 1], color="#33BBEE", lw=1.4)
                ax.set(title=title, xlim=(0, 224), ylim=(224, 0), xticks=[], yticks=[])
        truth = "推进" if result[ARMS[0]]["response_y"] else "未推进"
        fig.suptitle(f"#{ui} 整轨迹留出 · 真值{truth} · 固定seed {SEEDS[0]}\n"
                     f"推进分数：尖端冻结{result[ARMS[0]]['probability_advance']:.3f} / "
                     f"头段冻结{result[ARMS[1]]['probability_advance']:.3f}", y=.997, fontsize=11)
        fig.text(.5, .008, "青色：人工粗几何；橙色×：概率峰值。热图log10(p/均匀概率)，显示[-1,2]。\n"
                 "几何仅作离线目标/叠加；不是精确分割、同一物理点追踪或实机结果。",
                 ha="center", fontsize=8)
        fig.subplots_adjust(top=1-1.05/(2.35*len(chosen)), bottom=.065, hspace=.4, wspace=.14)
        stem = f"response_pair_case_{ui:03d}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{stem}.{ext}", dpi=450)
        plt.close(fig)
        links.append(f'<h2>#{ui} · 留出定位与响应</h2><img src="figures/{stem}.png">')
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>头段冻结响应对照</title>'
        '<style>body{max-width:1100px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>头段冻结表示：整轨迹隔离的响应对照</h1>'
        '<p>原45窗、10条轨迹、3seed。反复使用的开发集，不是独立测试或实机收益。'
        '固定阈值0.5，不挑最好seed；所有几何仅用于相应训练折的损失。</p>'
        '<img src="figures/response_pair_summary_zh.png">'
        '<img src="figures/response_pair_hardcases_zh.png">'+''.join(links), encoding="utf-8")
    print(f"Rendered 2 metric sheets and {len(links)} all-frame heldout sheets; user acceptance pending.")


if __name__ == "__main__":
    main()
