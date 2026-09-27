"""Render the saved frozen history/precision audit; no model or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np


def render(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = json.loads((path.parent / "measurements.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["optimizer_steps"] != 0:
        raise ValueError("expected completed frozen-weight audit")
    output = path.parent / "figures"
    output.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    quant = report["conditions"]["true_history"]["quantization"]
    fractions = [100*quant["cast_nonzero_fraction"]["mean"], 100*quant["effective_changed_fraction"]["mean"]]
    names = ("repeat_current", "swap_past")
    residual = {name: [100*r[name]["versus_true"]["requested_residual_relative_l2_change"] for r in rows] for name in names}
    action = {name: [r[name]["versus_true"]["translation_mean_abs_change_mm"] for r in rows] for name in names}
    unchanged = {name: report["conditions"][name]["versus_true"]["active_action_equal_contexts"] for name in names}
    manifest = {"data_kind": "actual frozen inference on 30 training contexts, not held-out performance",
                "source_files": ["../report.json", "../measurements.json"],
                "script": "tools/render_real10_pi05_gated_history_audit.py",
                "cast_nonzero_and_effective_changed_percent": fractions,
                "residual_relative_change_percent": residual, "xyz_output_mean_abs_change_mm": action,
                "active_action_equal_contexts": unchanged,
                "outputs": ["gated_precision_history_zh.png", "gated_precision_history_zh.svg"],
                "visual_status": "not_viewed"}
    (output / "data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    blue, orange, gray = "#4E79A7", "#F28E2B", "#9DA3AD"
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.1, 1.15]})
    ax = axes[0]
    bars = ax.bar([0, 1], fractions, color=[blue, orange], width=.58)
    for bar, value in zip(bars, fractions):
        ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.2f}%", ha="center", va="bottom")
    ax.set_xticks([0, 1], ["残差转BF16后\n仍非零", "加入state后\n实际发生变化"])
    ax.set_ylim(0, 116)
    ax.set_ylabel("坐标占比（%；不是信息保留率）")
    ax.set_title("A  主要数值误差发生在相加时", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    offsets = np.linspace(-.12, .12, len(rows))
    for pos, name, color in zip([0, 1], names, [blue, orange]):
        values = np.asarray(residual[name])
        ax.scatter(pos+offsets, values, s=18, color=color, alpha=.75)
        ax.plot([pos-.18, pos+.18], [np.median(values)]*2, color=color, linewidth=2.5)
    ax.set_yscale("log")
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([0, 1], ["重复当前观测", "交换过去两帧"])
    ax.set_ylabel("量化前残差相对L2变化（%；对数轴）")
    ax.set_title("B  顺序扰动在量化前已很弱", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax = axes[2]
    for pos, name, color in zip([0, 1], names, [blue, orange]):
        values = np.asarray(action[name])
        ax.scatter(pos+offsets, values, s=20, color=color, alpha=.75)
        ax.plot([pos-.18, pos+.18], [values.mean()]*2, color=color, linewidth=2.5)
    ax.axhline(0, color=gray, linewidth=.8)
    ax.set_xlim(-.5, 1.5)
    ax.set_xticks([0, 1], [f"重复当前观测\n{len(rows)-unchanged[names[0]]}/30个xyz输出变化",
                           f"交换过去两帧\n{len(rows)-unchanged[names[1]]}/30个xyz输出变化"])
    ax.set_ylabel("相对真实历史的xyz平均绝对变化（毫米）")
    ax.set_title("C  顺序交换没有改变动作输出", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("冻结时序适配器：区分舍入损失与历史依赖", fontsize=17, weight="bold", y=.97)
    fig.text(.5, .095, "冻结权重、当前双视角/状态/指令与推理噪声；只改变过去内容，保留时间槽和有效掩码。\n"
             "30个训练内上下文；零训练、无精度改动、无实机。变化不等于收益，反事实历史不用于模型选择。\n"
             "B每点为一个上下文、横线为中位数；C横线为均值。原模型与真实历史预测均逐值复现。",
             ha="center", fontsize=9, color="#4B5563", linespacing=1.6)
    fig.subplots_adjust(left=.065, right=.98, top=.81, bottom=.34, wspace=.45)
    for extension in ("png", "svg"):
        fig.savefig(output / f"gated_precision_history_zh.{extension}", dpi=450)
    plt.close(fig)
    print(json.dumps({"figure": str(output / "gated_precision_history_zh.png"), "fractions_percent": fractions}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)
