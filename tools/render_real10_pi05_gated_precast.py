"""Plot actual frozen pre/post-cast diagnostics without loading any model."""
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
    if report["status"] != "completed" or report["optimizer_steps"] != 0:
        raise ValueError("requires completed frozen-weight comparison")
    output = path.parent / "figures"
    output.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font)).get_name(), "font.size": 10,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "path"})
    paths = ("postcast", "precast")
    values = report["conditions"]
    q = [values[f"{p}__true_history"]["quantization"] for p in paths]
    changed = [100*v["effective_changed_fraction"]["mean"] for v in q]
    mae = [report["original_metrics"]["translation_mae_mm"]] + [values[f"{p}__true_history"]["metrics"]["translation_mae_mm"] for p in paths]
    history = ("repeat_current", "swap_past")
    counts = {p: [report["contexts"] - values[f"{p}__{c}"]["versus_true"]["active_action_exactly_equal_contexts"] for c in history] for p in paths}
    manifest = {"data_kind": "actual frozen inference; 30 training contexts, not independent validation",
                "source": "../report.json", "script": "tools/render_real10_pi05_gated_precast.py",
                "paths": list(paths), "updated_coordinate_percent": changed,
                "translation_mae_mm_original_postcast_precast": mae,
                "history_changed_action_contexts": counts,
                "combined_token_rms_error": [v["combined_token_rms_error"]["mean"] for v in q],
                "incremental_relative_l2_error": [v["effective_relative_l2_error"]["mean"] for v in q],
                "outputs": ["gated_precast_comparison_zh.png", "gated_precast_comparison_zh.svg"],
                "visual_status": "not_viewed"}
    (output / "data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data-manifest.md").write_text(
        "# Figure data manifest\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Frozen pre/post-cast comparison | ../report.json | Real | Frozen step100 adapter, 30 train contexts | "
        "tools/render_real10_pi05_gated_precast.py | gated_precast_comparison_zh.png / .svg |\n\n"
        "Per-context capture is in ../measurements.json and ../captured_vectors.npz. "
        "No mock data; no held-out or real-system claim.\n", encoding="utf-8")
    blue, orange, gray = "#4E79A7", "#F28E2B", "#9DA3AD"
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
    ax = axes[0]
    bars = ax.bar([0, 1], changed, color=[blue, orange], width=.58)
    for bar, value in zip(bars, changed):
        ax.text(bar.get_x()+bar.get_width()/2, value+1, f"{value:.2f}%", ha="center")
    ax.set_xticks([0, 1], ["转型后相加\n旧入口", "转型前相加\n新入口"])
    ax.set_ylim(0, max(changed)*1.4+2)
    ax.set_ylabel("最终 state 坐标发生变化的比例（%）")
    ax.set_title("A  实际数值扰动", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax.text(.5, -.29, "坐标占比不是信息保留率\n两种入口最终均为 BF16 token", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#4B5563")
    ax = axes[1]
    bars = ax.bar([0, 1, 2], mae, color=[gray, blue, orange], width=.58)
    for bar, value in zip(bars, mae):
        ax.text(bar.get_x()+bar.get_width()/2, value+.008, f"{value:.6f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1, 2], ["原 PI05", "转型后相加", "转型前相加"])
    ax.set_ylim(0, max(mae)*1.3)
    ax.set_ylabel("Elite 平移 MAE（毫米；越低越好）")
    ax.set_title("B  真实历史下的动作误差", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2)
    ax.text(.5, -.29, "固定的训练内诊断面板\n不用于模型选择或泛化结论", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#4B5563")
    ax = axes[2]
    x = np.arange(2)
    for offset, path_name, color, label in zip([-.18, .18], paths, [blue, orange], ["转型后相加", "转型前相加"]):
        bars = ax.bar(x+offset, counts[path_name], width=.32, color=color, label=label)
        for bar, count in zip(bars, counts[path_name]):
            ax.text(bar.get_x()+bar.get_width()/2, count+.6, str(count), ha="center")
    ax.set_xticks(x, ["重复当前内容", "交换过去两帧"])
    ax.set_ylim(0, report["contexts"]+7)
    ax.set_ylabel(f"相对真实历史发生变化的 xyz 输出（个 / {report['contexts']}）")
    ax.set_title("C  历史干预敏感性", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=.2)
    ax.text(.5, -.29, "只改变过去内容，保留时间槽\n输出变化不等于动作收益", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#4B5563")
    fig.suptitle("转型前相加：数值变化与动作结果分开看", fontsize=17, weight="bold", y=.97)
    fig.text(.5, .055, "同一冻结 step100 适配器、同一门控值、同一输入与噪声；仅改变残差相加位置。\n"
             "10 条训练轨迹 × 3 个上下文；零门控在全部 30 窗保持原输出与 loss 逐值一致；无训练、无实机。",
             ha="center", fontsize=10, color="#4B5563", linespacing=1.7)
    fig.subplots_adjust(left=.065, right=.985, top=.81, bottom=.35, wspace=.44)
    for extension in ("png", "svg"):
        fig.savefig(output / f"gated_precast_comparison_zh.{extension}", dpi=450)
    plt.close(fig)
    print(json.dumps({"figure": str(output / "gated_precast_comparison_zh.png"), "mae_mm": mae}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    render(parser.parse_args().report)
