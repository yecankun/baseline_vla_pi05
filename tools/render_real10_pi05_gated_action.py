"""Chinese figure from saved gated-action measurements; no model dependencies."""
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
    logs = json.loads((path.parent / "training_log.json").read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["world_loss_enabled"]:
        raise ValueError("requires the completed action-only diagnostic")
    out = path.parent / "figures"
    out.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    metrics = report["metrics"]
    values = [metrics["original"]["translation_mae_mm"],
              report["historical_append_only_action_metrics"]["translation_mae_mm"],
              metrics["gated_initial"]["translation_mae_mm"], metrics["gated_final"]["translation_mae_mm"]]
    gate = [0.] + [r["gate_tanh_after_step"] for r in logs]
    episodes = sorted(report["comparison"]["episode_mae_changes_mm"])
    changes = [report["comparison"]["episode_mae_changes_mm"][e] for e in episodes]
    manifest = {"data_kind": "actual train-only action optimization, no world loss",
                "source_files": ["../report.json", "../training_log.json"],
                "script": "tools/render_real10_pi05_gated_action.py", "mae_mm": values,
                "mae_order": ["original", "historical_append_only_action", "gated_initial", "gated_final"],
                "gate_per_step": gate, "episode_order": episodes, "paired_mae_changes_mm": changes,
                "outputs": ["gated_action_diagnostic_zh.png", "gated_action_diagnostic_zh.svg"],
                "visual_status": "not_viewed"}
    (out / "data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    blue, green, orange, gray = "#4E79A7", "#59A14F", "#F28E2B", "#9DA3AD"
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1, 1.2]})
    ax = axes[0]
    bars = ax.bar(np.arange(4), values, color=[blue, gray, "#76B7B2", green], width=.63)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(4), ["原模型", "旧追加式\n100步", "门控\n初始化", "门控\n100步"])
    ax.set_ylim(0, max(values)*1.22)
    ax.set_ylabel("Elite 平移 MAE（毫米；越低越好）")
    ax.set_title("A  同口径动作误差", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax = axes[1]
    ax.plot(np.arange(len(gate)), gate, color=blue, linewidth=1.6)
    ax.axhline(0, color=gray, linewidth=1, linestyle="--")
    ax.set_xlabel("纯动作优化步数")
    ax.set_ylabel("tanh(g)：带符号残差系数")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("B  零初始化门控学习轨迹", loc="left", weight="bold")
    ax.grid(alpha=.2)
    ax = axes[2]
    ax.bar(np.arange(len(episodes)), changes, color=[green if x < 0 else orange for x in changes], width=.66)
    ax.axhline(0, color=gray, linewidth=1)
    ax.set_xticks(np.arange(len(episodes)), [("左" if "_left_" in e else "右") + e.rsplit("_", 1)[1]
                                            for e in episodes], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("门控100步 − 原模型 MAE（毫米）")
    ax.set_title("C  每条训练轨迹：负值更好", loc="left", weight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=.2)
    fig.suptitle("PI0.5 时序残差：先保持原策略，再学习历史信息", fontsize=17, weight="bold", y=.97)
    fig.text(.5, .07, "30/30 上下文：门控初始化动作与损失逐值等于原模型；训练后梯度进入时序模块。\n"
             "仅训练内诊断：单seed / 100步 / 2 hold＋1 feed每轨迹；未训练世界模型、未做实机或独立验证。\n"
             "灰色为上一轮同数据顺序的历史追加式对照，不是本轮并行重训。",
             ha="center", fontsize=9, linespacing=1.65, color="#4B5563")
    fig.subplots_adjust(left=.065, right=.985, top=.80, bottom=.32, wspace=.41)
    for extension in ("png", "svg"):
        fig.savefig(out / f"gated_action_diagnostic_zh.{extension}", dpi=450)
    plt.close(fig)
    print(json.dumps({"figure": str(out / "gated_action_diagnostic_zh.png"), "mae_mm": values}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)
