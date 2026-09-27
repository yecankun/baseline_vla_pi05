"""Chinese figure from completed real10 matched motion-encoding diagnostics."""
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
    rows = json.loads((path.parent / "panel_predictions.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["steps_per_arm"] != 100 or report["world_loss_enabled"]:
        raise ValueError("expected the completed fixed 100-step action-only pair")
    output = path.parent / "figures"
    output.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    names = ("absolute", "motion")
    metrics = report["metrics"]
    mae = [metrics[key]["translation_mae_mm"] for key in ("original", "absolute__true_history", "motion__true_history")]
    differences = report["motion_comparisons"]["absolute__true_history"]["episode_mae_changes_mm"]
    episodes = list(differences)
    tasks = {r["episode"]: r["task"] for r in rows}
    task_ordinals = {"left": 0, "right": 0}
    labels = []
    for ep in episodes:
        task = tasks[ep]
        task_ordinals[task] += 1
        labels.append(f"{'左' if task == 'left' else '右'}{task_ordinals[task]}")
    counts = {name: [len(rows)-report["history_sensitivity"][f"{name}__{c}"]["active_action_equal_contexts"]
                     for c in ("repeat_current", "swap_past")] for name in names}
    manifest = {"data_kind": "actual single-seed paired 100-step experiment; training-only 30-context panel",
                "source_files": ["../report.json", "../panel_predictions.json"],
                "script": "tools/render_real10_pi05_motion_comparison.py",
                "mae_mm_original_absolute_motion": mae, "motion_minus_absolute_episode_mae_mm": differences,
                "episode_display_labels": dict(zip(episodes, labels)), "changed_action_contexts": counts,
                "outputs": ["motion_encoding_comparison_zh.png", "motion_encoding_comparison_zh.svg"],
                "visual_status": "not_viewed"}
    (output / "data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data-manifest.md").write_text(
        "# Figure data manifest\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Matched motion encoding | ../report.json; ../panel_predictions.json | Real | Same initialization/order/noise, 100 steps per arm | "
        "tools/render_real10_pi05_motion_comparison.py | motion_encoding_comparison_zh.png / .svg |\n\n"
        "Original base saw all ten episodes; no held-out/generalization or world-loss claim. "
        "Episode label mapping and plotted numbers are saved in data-manifest.json.\n", encoding="utf-8")
    blue, orange, gray = "#4E79A7", "#F28E2B", "#9DA3AD"
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.4), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})
    ax = axes[0]
    bars = ax.bar([0, 1, 2], mae, color=[gray, blue, orange], width=.6)
    for bar, value in zip(bars, mae):
        ax.text(bar.get_x()+bar.get_width()/2, value+max(mae)*.035, f"{value:.6f}", ha="center")
    ax.set_xticks([0, 1, 2], ["原 PI05", "绝对历史", "显式变化"])
    ax.set_ylim(0, max(mae)*1.3)
    ax.set_ylabel("Elite 平移 MAE（毫米；越低越好）")
    ax.set_title("A  真实历史下的动作误差", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax.text(.5, -.29, "两种时序编码各训练 100 步\n原 PI05 全程冻结", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#4B5563")
    ax = axes[1]
    y = np.asarray([differences[ep] for ep in episodes])
    ax.axhline(0, color=gray, linewidth=1)
    ax.vlines(np.arange(len(episodes)), 0, y, color=orange, linewidth=1.5)
    ax.scatter(np.arange(len(episodes)), y, color=orange, s=28, zorder=3)
    ax.set_xticks(np.arange(len(episodes)), labels, rotation=40)
    ax.set_ylabel("显式变化 − 绝对历史：MAE 差（毫米）")
    ax.set_title("B  各轨迹配对差异", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax.text(.5, -.29, "零线以下：显式变化误差更小\n每个点是同一轨迹的 3 窗均值", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#4B5563")
    ax = axes[2]
    x = np.arange(2)
    for offset, name, color, label in zip([-.18, .18], names, [blue, orange], ["绝对历史", "显式变化"]):
        bars = ax.bar(x+offset, counts[name], width=.32, color=color, label=label)
        for bar, count in zip(bars, counts[name]):
            ax.text(bar.get_x()+bar.get_width()/2, count+.6, str(count), ha="center")
    ax.set_xticks(x, ["重复当前内容", "交换过去两帧"])
    ax.set_ylim(0, len(rows)+8)
    ax.set_ylabel(f"相对真实历史发生变化的 xyz 输出（个 / {len(rows)}）")
    ax.set_title("C  历史干预敏感性", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.text(.5, -.29, "当前观测、时间槽与噪声固定\n更多输出变化不等于更好预测", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#4B5563")
    fig.suptitle("显式运动变化：同参数、同预算的时序编码对照", fontsize=17, weight="bold", y=.97)
    fig.text(.5, .055, "相同数值初始化、样本顺序、优化器与推理噪声；旧编码训练结果逐值复现；不改变残差精度接口。\n"
             "10 条已用于训练的轨迹，30 个固定上下文；单随机种子、非独立验证；无世界模型损失、无实机。",
             ha="center", fontsize=10, color="#4B5563", linespacing=1.7)
    fig.subplots_adjust(left=.062, right=.985, top=.81, bottom=.35, wspace=.44)
    for extension in ("png", "svg"):
        fig.savefig(output / f"motion_encoding_comparison_zh.{extension}", dpi=450)
    plt.close(fig)
    print(json.dumps({"figure": str(output / "motion_encoding_comparison_zh.png"), "mae_mm": mae,
                      "changed_action_contexts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)
