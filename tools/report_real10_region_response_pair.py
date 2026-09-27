"""Render saved regional response evidence, without training or model inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from fit_real10_head_region import read_json, read_jsonl, write_json

ARMS = ("global_broadcast", "regional_4x4")
SEEDS = (20261020, 20261120, 20261220)
CASES = (4, 39, 62, 65, 77, 100)
NAMES = ("全局摘要复制", "保留4×4区域")
COLORS = ("#0077BB", "#EE7733", "#009988")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_region_response_pair_v1"))
    args = parser.parse_args()
    report = read_json(args.out / "report.json")
    predictions = read_jsonl(args.out / "oof_predictions.jsonl")
    cases = read_jsonl(args.out / "case_regions.jsonl")
    if report["completed_fits"] != 60 or len(predictions) != 270 or len(cases) != 36:
        raise ValueError("require all fixed fits and paired exports")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "svg.fonttype": "path"})
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 区域时序表示配对：真实开发集OOF结果\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "|---|---|---|---|---|---|\n"
        "| 三seed配对指标 | ../report.json, ../oof_predictions.jsonl | real | 原45窗、10轨迹LOEO | "
        "tools/report_real10_region_response_pair.py | region_pair_summary_zh.png/.svg |\n"
        "| 六个固定难例 | ../oof_predictions.jsonl | real | 六例全部seed | 同上 | region_pair_hardcases_zh.png/.svg |\n"
        "| 六例区域质量 | ../case_regions.jsonl, 原ROI224图 | real | 最小seed、各窗首/中/末帧 | "
        "同上 | region_pair_case_*_zh.png/.svg |\n\n"
        "两组共用各折冻结定位器；响应头8481可训参数、200步、同初始化、相同输入形状。\n"
        "反复使用的45窗开发集，嵌套于10轨迹；三seed不是135个独立样本。\n"
        "固定0.5阈值；分数没有概率校准。人工标签未改，几何从未进入推理输入。\n"
        "区域图显示log10(格内质量/(1/16))，固定范围[-2,log10(16)]，格内不归一化。\n"
        "网格是56×56卷积特征的4×4划分；图像坐标按中心0.5/步长4回映。\n"
        "原图、网格、热图叠加均只作显示；Side/Top同编号格不表示同一物理位置。\n"
        "控制组16槽质量均1/16，只有相同的全局摘要；区域图不是精确分割、接触概率或同点追踪。\n",
        encoding="utf-8")

    def save(fig, stem):
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{stem}.{ext}", dpi=450)
        plt.close(fig)

    lookup = {(r["base_seed"], r["arm"]): r for r in report["seed_reports"]}
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.3))
    for ax, key, title in zip(axes, ("balanced_accuracy", "accuracy", "episode_macro_accuracy"),
                              ("平衡准确率 BA", "窗口准确率", "整轨迹宏准确率")):
        for k, seed in enumerate(SEEDS):
            values = [100*lookup[seed, arm][key] for arm in ARMS]
            ax.plot([0, 1], values, "o-", color=COLORS[k], label=f"seed {seed}", lw=1.3, ms=5)
        mean = [report["means"][arm][key]["mean"]*100 for arm in ARMS]
        ax.set(title=f"{title}\n均值 {mean[0]:.2f}% → {mean[1]:.2f}%", xticks=[0, 1],
               xticklabels=NAMES, ylabel="%", ylim=(0, 100), xlim=(-.25, 1.25))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .91), frameon=False, fontsize=8)
    fig.suptitle("保留区域信息能否改善响应识别？固定预算整轨迹LOEO", y=.99, fontsize=12)
    fig.text(.5, .018, "45窗 / 10轨迹；共用冻结定位器，响应头同参数、同初始化、同200步；不是独立测试。",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.69, bottom=.18, wspace=.34)
    save(fig, "region_pair_summary_zh")

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.8))
    hardcases = []
    for ax, ui in zip(axes.flat, CASES):
        chosen = [r for r in predictions if r["ui_index"] == ui]
        truth = chosen[0]["response_y"]
        for j, arm in enumerate(ARMS):
            rows = [next(r for r in chosen if r["base_seed"] == seed and r["arm"] == arm) for seed in SEEDS]
            values = [r["probability_advance"] for r in rows]
            ax.plot(range(3), values, "o-" if j == 0 else "s--", color=COLORS[j],
                    label=NAMES[j], lw=1.2, ms=4)
            hardcases.append({"ui_index": ui, "truth": truth, "arm": arm, "seeds": list(SEEDS),
                              "probability_advance": values,
                              "correct_seeds": sum(r["prediction"] == truth for r in rows)})
        note = {4: "小幅推进", 65: "有移动但未推进", 100: "小幅推进"}.get(ui, "")
        ax.axhline(.5, color="#888888", ls=":", lw=1)
        ax.set(title=f"#{ui} 真值：{'推进' if truth else '未推进'}"+(f"（{note}）" if note else ""),
               xticks=range(3), xticklabels=["seed 1", "seed 2", "seed 3"],
               ylabel="推进分数", ylim=(-.03, 1.03))
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(.5, .94), frameon=False, fontsize=9)
    fig.suptitle("六个固定难例：展示全部seed，不选择改对案例", y=.99, fontsize=12)
    fig.text(.5, .018, "虚线为固定0.5阈值；分数不是校准置信度。仅移动、小幅推进沿用原人工判断，不生成新标签。",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.82, bottom=.10, hspace=.45, wspace=.32)
    save(fig, "region_pair_hardcases_zh")
    write_json(args.out / "hardcase_readback.json", hardcases)

    links = []
    for ui in CASES:
        case = next(r for r in cases if (r["ui_index"], r["base_seed"], r["arm"]) == (ui, SEEDS[0], ARMS[1]))
        mass = np.asarray(case["region_mass"])
        if mass.shape != (len(case["frames"]), 2, 16) or not np.allclose(mass.sum(-1), 1, atol=1e-5):
            raise ValueError("invalid saved regional mass")
        indices = (0, len(case["frames"])//2, len(case["frames"])-1)
        pair = [next(r for r in predictions if (r["ui_index"], r["base_seed"], r["arm"]) ==
                     (ui, SEEDS[0], arm)) for arm in ARMS]
        fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.5))
        for v, view in enumerate(("side", "top")):
            for j, frame in enumerate(indices):
                ax = axes[v, j]
                path = case["frames"][frame]["images"][view]
                with Image.open(path) as img:
                    ax.imshow(img.convert("RGB"), extent=(0, 224, 224, 0))
                values = np.log10(np.maximum(mass[frame, v]*16, 1e-10)).reshape(4, 4)
                im = ax.imshow(values, extent=(-1.5, 222.5, 222.5, -1.5), cmap="magma",
                               vmin=-2, vmax=np.log10(16), alpha=.48, interpolation="nearest")
                for boundary in (-1.5, 54.5, 110.5, 166.5, 222.5):
                    ax.axvline(boundary, color="#AAAAAA", lw=.4, alpha=.5)
                    ax.axhline(boundary, color="#AAAAAA", lw=.4, alpha=.5)
                ax.set(title=f"{view.capitalize()} · 第{frame+1}/{len(case['frames'])}帧 · {Path(path).stem}",
                       xlim=(0, 224), ylim=(224, 0), xticks=[], yticks=[])
        truth = "推进" if case["response_y"] else "未推进"
        fig.suptitle(f"#{ui} · 真值{truth} · 固定seed {SEEDS[0]} · 留出轨迹\n"
                     f"推进分数：全局摘要{pair[0]['probability_advance']:.3f} / 区域保留{pair[1]['probability_advance']:.3f}",
                     y=.99, fontsize=11)
        # Use an opaque color scale: plotted layers alone have alpha for image visibility.
        sm = plt.cm.ScalarMappable(norm=im.norm, cmap=im.cmap)
        cax = fig.add_axes([.90, .22, .018, .52])
        fig.colorbar(sm, cax=cax, label="log10(格内质量 / 均匀质量)")
        fig.text(.47, .015, "控制组16槽相同（每格质量1/16）；新组保留图示分布，仍非校准导丝概率。\n"
                 "同一定位器可能关注错误位置；网格不是精确头段分割，两视角格子不代表物理对应。",
                 ha="center", fontsize=8)
        fig.subplots_adjust(top=.85, bottom=.115, left=.035, right=.875, hspace=.19, wspace=.13)
        stem = f"region_pair_case_{ui:03d}_zh"
        save(fig, stem)
        links.append(f'<h2>#{ui} · 首/中/末帧区域质量</h2><img src="figures/{stem}.png">')
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>区域时序响应配对</title>'
        '<style>body{max-width:1100px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>全局摘要 vs 区域保留：固定预算响应对照</h1>'
        '<p>原45窗、10条轨迹、3seed；反复使用的开发集，不是独立测试或实机收益。'
        '两组响应头同8481参数、同初始化、同200步，复用对应折的冻结定位器。'
        '人工几何从未作为推理输入；区域质量不是接触或导丝分割概率。</p>'
        '<img src="figures/region_pair_summary_zh.png"><img src="figures/region_pair_hardcases_zh.png">'
        + ''.join(links), encoding="utf-8")
    print("Rendered 2 metric sheets and 6 fixed-case region sheets; user acceptance pending.")


if __name__ == "__main__":
    main()
