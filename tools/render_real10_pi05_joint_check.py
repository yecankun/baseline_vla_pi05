"""Render saved joint-prefix diagnostics only; no LeRobot or model loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np


def render(report_path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("render only the completed diagnostic report")
    output = report_path.parent / "figures"
    output.mkdir(exist_ok=True)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font_path))
    family = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    colors = ["#4E79A7", "#F28E2B", "#76B7B2", "#A23B72"]
    action = report["shared_action_gradient"]["l2"]
    future = report["shared_future_gradient"]["l2"]
    weight = report["losses_before_joint_step"]["future_weight"]
    ratio = future * weight / action
    rows = report["sample_predictions"]
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.75])
    ax = fig.add_subplot(grid[0, 0])
    values = [action, future, future * weight]
    bars = ax.barh(["动作损失", "未来预测损失", f"未来预测 × {weight:g}"], values,
                   color=[colors[0], colors[2], colors[3]], height=.55)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 3, max(values) * 30)
    for bar, value in zip(bars, values):
        ax.text(value * 1.3, bar.get_y() + bar.get_height()/2, f"{value:.3g}", va="center", fontsize=9)
    ax.set_xlabel("共享时序模块梯度 L2（对数轴）")
    ax.set_title("A  双任务接通，但尺度尚未平衡", loc="left", weight="bold")
    ax.grid(axis="x", alpha=.2)
    notes = fig.add_subplot(grid[1:, 0])
    notes.axis("off")
    text = ("本次检查范围\n\n"
            "• 原真实模型冻结；Piper head 保留\n"
            "• 新增时序适配器和单步潜空间预测\n"
            "• 未来目标不进入当前策略输入\n"
            "• 缺失历史掩码、保存回载检查通过\n"
            "• 关闭适配器，输出逐值恢复原模型\n\n"
            "关键发现\n\n"
            f"加权辅助梯度 / 动作梯度 = {ratio:.2e}\n"
            "该比值仅来自一个选定上下文，\n"
            "不能代表全数据，也不能据此直接调权。\n\n"
            "未来损失单独更新后，固定噪声下的\n"
            "PI0.5 输出发生变化，证明连接有效。\n"
            "变化不等于改善；未做验证集或实机测试。")
    notes.text(0, 1, text, va="top", linespacing=1.7, fontsize=10)
    x = np.arange(len(rows))
    labels = ["左：起点", "左：递丝", "右：递丝", "左：末次转移"]
    for dimension, axis in enumerate("xyz"):
        ax = fig.add_subplot(grid[dimension, 1])
        for key, label, color, marker in (
            ("baseline", "原 PI0.5", colors[0], "o"),
            ("initialized_adapter", "适配器初始", colors[1], "^"),
            ("one_joint_step_adapter", "仅一次联合更新", colors[2], "s"),
        ):
            values = [r[key]["elite_tcp_delta_6d"][dimension] for r in rows]
            ax.plot(x, values, marker=marker, color=color, linewidth=1.2, markersize=4, label=label)
        ax.set_ylabel(f"Δ{axis}（毫米）")
        ax.set_xticks(x, labels)
        ax.grid(alpha=.2)
        if dimension == 0:
            ax.set_title("B  四个上下文的输出连通检查（固定采样噪声）", loc="left", weight="bold")
            ax.legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle("PI0.5–世界模型：共享表征的真实权重连通检查\n单步数值测试，不是训练效果或策略成功率", fontsize=15, weight="bold")
    png, svg = output / "joint_prefix_diagnostic_zh.png", output / "joint_prefix_diagnostic_zh.svg"
    fig.savefig(png, dpi=450)
    fig.savefig(svg)
    plt.close(fig)
    manifest = {"figure": png.name, "data_file": "../report.json", "data_kind": "actual diagnostic measurements",
                "script": "tools/render_real10_pi05_joint_check.py", "outputs": [png.name, svg.name],
                "scope": "4 in-training contexts, no performance evaluation", "weighted_aux_to_action_gradient_ratio": ratio,
                "visual_status": "not_viewed"}
    (output / "data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "svg": str(svg), "weighted_aux_to_action_ratio": ratio}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)
