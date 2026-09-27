"""Plot completed matched diagnostics; reads JSON only, never loads a model."""
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
    if report["status"] != "completed" or report["heldout_evaluation"]:
        raise ValueError("expected the completed train-only matched comparison")
    output = path.parent / "figures"
    output.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "path"})
    calibration = report["calibration"]
    metrics = report["evaluation"]["metrics"]
    names = ["original_pi05", "initialized_adapter", "action_only", "joint"]
    mae = [metrics[k]["translation_mae_mm"] for k in names]
    episodes = sorted(metrics["action_only"]["per_episode_mae_mm"])
    changes = [metrics["joint"]["per_episode_mae_mm"][e] - metrics["action_only"]["per_episode_mae_mm"][e]
               for e in episodes]
    old = [r["old_weighted_ratio"] for r in calibration["contexts"]]
    new = [r["new_weighted_ratio"] for r in calibration["contexts"]]
    persistence = np.asarray(report["evaluation"]["persistence_mse_per_view"])
    future_ratios = {k: np.asarray(metrics[k]["future_mse_per_view"]) / persistence for k in ("action_only", "joint")}
    manifest = {"data_kind": "actual measured training diagnostics, not held-out results",
                "source": "../report.json", "script": "tools/render_real10_pi05_joint_comparison.py",
                "plots": {"gradient_ratio_before": old, "gradient_ratio_after": new,
                          "mae_order": names, "mae_mm": mae, "episodes": episodes,
                          "joint_minus_action_only_mae_mm": changes,
                          "future_to_persistence_per_view": {k: v.tolist() for k, v in future_ratios.items()}},
                "outputs": ["joint_comparison_zh.png", "joint_comparison_zh.svg"],
                "visual_status": "not_viewed"}
    (output / "data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    blue, green, orange, gray = "#4E79A7", "#59A14F", "#F28E2B", "#9DA3AD"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    ax = axes[0, 0]
    x = np.arange(1, len(old) + 1)
    ax.scatter(x, old, color=gray, label="原始未来损失 × 0.1", s=20)
    ax.scatter(x, new, color=green, label="按预定规则校准后", s=24)
    ax.axhline(.1, color=blue, linestyle="--", linewidth=1, label="预定初始中位数 0.1")
    ax.set_yscale("log")
    ax.set_xlabel("固定训练内上下文（每条轨迹 3 个）")
    ax.set_ylabel("辅助 / 动作共享梯度 L2 比")
    ax.set_title("A  辅助梯度尺度校准（训练前）", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(axis="y", alpha=.2)

    ax = axes[0, 1]
    bars = ax.bar(np.arange(4), mae, color=[gray, "#C3C7CE", blue, green], width=.62)
    for bar, value in zip(bars, mae):
        ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(4), ["原 PI0.5", "适配器初始", "仅动作损失", "动作＋未来"])
    ax.set_ylim(0, max(mae)*1.23)
    ax.set_ylabel("Elite 平移 MAE（毫米；越低越好）")
    ax.set_title("B  同一训练诊断面板 / 固定推理噪声", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)

    ax = axes[1, 0]
    x = np.arange(len(episodes))
    ax.bar(x, changes, color=[green if v < 0 else orange for v in changes], width=.65)
    ax.axhline(0, color=gray, linewidth=1)
    labels = [("左" if "_left_" in e else "右") + e.rsplit("_", 1)[1] for e in episodes]
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("联合组 − 动作组 MAE（毫米）")
    ax.set_title("C  每条训练轨迹的配对差（负值更好）", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)

    ax = axes[1, 1]
    x = np.arange(2)
    for offset, key, color, label in ((-.18, "action_only", blue, "动作组（预测器未训练）"),
                                      (.18, "joint", green, "联合组")):
        values = future_ratios[key]
        bars = ax.bar(x+offset, values, width=.33, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.2f}×", ha="center", va="bottom", fontsize=9)
    ax.axhline(1, color=gray, linewidth=1, linestyle="--", label="重复当前特征 = 1")
    ax.set_xticks(x, ["Side", "Top"])
    ax.set_ylabel("未来特征 MSE / persistence MSE")
    ax.set_ylim(0, max(1., *(v.max() for v in future_ratios.values())) * 1.65)
    ax.set_title("D  下一记录预测：是否优于静态参照", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=.2)
    steps = report["protocol"]["steps_per_arm"]
    fig.suptitle("PI0.5 与世界模型共享时序表征：等步数短训诊断", fontsize=17, weight="bold", y=.97)
    fig.text(.5, .025,
             f"两组各 {steps} 步；同初始化 / 数据顺序 / 噪声；原 PI0.5、Piper 冻结。\n"
             "面板：10 条训练轨迹 ×（2 hold＋1 feed）；单随机种子，不是验证集、成功率或实机收益。",
             ha="center", fontsize=10, color="#4B5563", linespacing=1.6)
    fig.subplots_adjust(left=.075, right=.97, top=.89, bottom=.17, wspace=.30, hspace=.54)
    for extension in ("png", "svg"):
        fig.savefig(output / f"joint_comparison_zh.{extension}", dpi=450)
    plt.close(fig)
    print(json.dumps({"figure": str(output / "joint_comparison_zh.png"), "mae_mm": dict(zip(names, mae))}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)
