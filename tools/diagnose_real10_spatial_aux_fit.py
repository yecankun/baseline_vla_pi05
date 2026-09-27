"""Read-only fit/generalization and spatial-distribution diagnosis of 60 saved fits.

infer: remote 4090, frozen cache/checkpoints, no optimizer or changes to the model.
render: local .venv, only exported diagnostics and existing ROI224 pictures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np


ARMS = ("spatial_window", "spatial_window_tip_aux")
SEEDS = (20261020, 20261120, 20261220)
VIEWS = ("side", "top")
PHASES = ("train", "heldout")
TRAIN_VISUAL_FOLD = 0  # first existing fold; no tip-labeled episode is held out here
METRICS = ("expectation_error_px", "argmax_error_px", "uniform_expectation_error_px",
           "expectation_to_argmax_px", "entropy_normalized", "effective_cell_fraction",
           "peak_over_uniform", "mass_within_two_grid_cells", "uniform_mass_same_region",
           "target_mass_same_region", "target_ce_normalized", "target_entropy_normalized",
           "spatial_rms_radius_px")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def balanced_point_summary(records):
    # First average repeated models for ONE labeled point; then points per window;
    # finally windows. Training has nine model exposures per point, heldout only one.
    points = defaultdict(list)
    for r in records:
        points[r["window_id"], r["frame_index"], r["view"]].append(r)
    by_window = defaultdict(list)
    for (window, _, _), values in points.items():
        by_window[window].append({m: float(np.mean([r[m] for r in values])) for m in METRICS})
    windows = {w: {m: float(np.mean([r[m] for r in rs])) for m in METRICS}
               for w, rs in by_window.items()}
    return {"evaluations": len(records), "distinct_points": len(points), "windows": len(windows),
            "per_window": windows,
            "metrics": {m: float(np.mean([r[m] for r in windows.values()])) if windows else None
                        for m in METRICS}}


def infer(args):
    from train_real10_spatial_aux_pair import (
        check_frozen, checkpoint_path, configure, forward, load_tensors, subset, validate_checkpoint)
    from real10_spatial_response import SpatialResponse, point_distribution, spatial_grid
    import torch

    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve an existing diagnosis; use a fresh --out after inspecting a failed run")
    configure()
    inputs, targets, rows, folds = check_frozen(args.source)
    if read_json(args.source / "report.json")["status"] != "complete_fixed_pair":
        raise ValueError("all 60 original fits must have completed")
    audit = {(r["window_id"], r["frame_index"], r["view"]): r for r in
             read_jsonl(args.source / "source/point_audit.jsonl") if r["tip_target_valid"]}
    earliest = {}
    for window, frame_index, view in audit:
        earliest[window, view] = min(frame_index, earliest.get((window, view), frame_index))
    if any(r["source_episode"] == folds[TRAIN_VISUAL_FOLD]["heldout_episode"] for r in audit.values()):
        raise ValueError("first-fold training visualization must contain every existing tip point")
    args.out.mkdir(parents=True)
    shutil.copy2(__file__, args.out / "entrypoint_snapshot.py")
    write_json(args.out / "protocol.json", {
        "source": args.source.as_posix(), "model_count": 60, "seeds": list(SEEDS), "arms": list(ARMS),
        "optimizer_steps": 0, "new_labels": False, "model_or_readout_changes": False,
        "timing_estimate": "less than one minute: 120 inference batches, vs 12000 prior updates in 180s",
        "aggregation": "model exposures per labeled point -> points per window -> windows; separate views/seeds",
        "train_exposures_per_point": 9, "heldout_exposures_per_point": 1,
        "comparisons": ["existing softmax expectation", "diagnostic-only argmax", "constant uniform expectation"],
        "spatial_distribution": "entropy/log(56^2), exp(entropy)/56^2, peak/uniform, target-region mass, normalized target CE",
        "target_region": "radius two layer1 grid cells, fixed at twice the existing Gaussian encoding sigma; not a correctness tolerance",
        "visual_selection": "earliest valid human tip per window/view; all three seeds and both arms; first-fold training vs own heldout fold",
        "train_visual_fold": TRAIN_VISUAL_FOLD,
        "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
    tensors = load_tensors(args.source, inputs, targets)
    roi = read_json(args.source / "source/protocol.json")["roi_xyxy_original"]
    scales = torch.tensor([[roi[v][2]-roi[v][0], roi[v][3]-roi[v][1]] for v in VIEWS],
                          device="cuda", dtype=torch.float32)
    offsets = torch.tensor([[roi[v][0]-.5, roi[v][1]-.5] for v in VIEWS], device="cuda")
    grid = spatial_grid("cuda", torch.float32).reshape(-1, 2)
    cells, log_cells = len(grid), float(np.log(len(grid)))
    point_records, model_records, image_records, maps = [], [], [], {}
    replay_max = 0.
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for seed in SEEDS:
            for fold_id, fold in enumerate(folds):
                batches = {phase: subset(tensors, fold[key]) for phase, key in
                           (("train", "train_indices"), ("heldout", "test_indices"))}
                pair_initial = None
                for arm in ARMS:
                    path = checkpoint_path(args.source, seed, fold_id, arm)
                    before = (path.stat().st_size, path.stat().st_mtime_ns)
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    validate_checkpoint(checkpoint, seed, fold_id, fold, arm)
                    if pair_initial is not None and any(not torch.equal(pair_initial[k], checkpoint["initial_state"][k])
                                                         for k in pair_initial):
                        raise ValueError("original pair initialization mismatch")
                    pair_initial = checkpoint["initial_state"]
                    model = SpatialResponse().cuda().eval().requires_grad_(False)
                    model.load_state_dict(checkpoint["model_state"])
                    for phase in PHASES:
                        batch = batches[phase]
                        indices = fold["train_indices" if phase == "train" else "test_indices"]
                        output = forward(model, batch)
                        if phase == "heldout":
                            errors = [float((output[k]-checkpoint[old].cuda()).abs().max()) for k, old in
                                      (("window_logit", "heldout_window_logit"),
                                       ("location_uv", "heldout_location_uv"))]
                            if not np.isfinite(errors).all() or max(errors) > 1e-6:
                                raise ValueError("inference does not replay the original OOF export")
                            replay_max = max(replay_max, *errors)
                        mask = batch["tip_valid"]
                        positions = mask.nonzero().cpu().tolist()
                        if not positions:
                            continue
                        target = batch["xy_uv"][mask]
                        logp = output["heatmap_logits"][mask].flatten(1).log_softmax(-1)
                        probability = logp.exp()
                        location = output["location_uv"][mask]
                        mode = grid[logp.argmax(-1)]
                        view_ids = torch.tensor([pos[2] for pos in positions], device="cuda")
                        scale = scales[view_ids]
                        offset = offsets[view_ids]
                        q = point_distribution(target, torch.ones(len(target), device="cuda", dtype=torch.bool)).flatten(1)
                        entropy = -(probability*logp).sum(1)
                        in_region = (((grid[None]-target[:, None])*56).square().sum(-1) <= 4)
                        values = {
                            "expectation_error_px": ((location-target)*scale).norm(dim=1),
                            "argmax_error_px": ((mode-target)*scale).norm(dim=1),
                            "uniform_expectation_error_px": ((grid.mean(0)-target)*scale).norm(dim=1),
                            "expectation_to_argmax_px": ((location-mode)*scale).norm(dim=1),
                            "entropy_normalized": entropy/log_cells,
                            "effective_cell_fraction": entropy.exp()/cells,
                            "peak_over_uniform": probability.max(1).values*cells,
                            "mass_within_two_grid_cells": (probability*in_region).sum(1),
                            "uniform_mass_same_region": in_region.float().mean(1),
                            "target_mass_same_region": (q*in_region).sum(1),
                            "target_ce_normalized": -(q*logp).sum(1)/log_cells,
                            "target_entropy_normalized": -(q*q.clamp_min(1e-30).log()).sum(1)/log_cells,
                            "spatial_rms_radius_px": (probability*((grid[None]-location[:, None])*scale[:, None]).square().sum(-1)).sum(1).sqrt()}
                        arrays = {k: v.cpu().tolist() for k, v in values.items()}
                        xy_expectation = (location*scale+offset).cpu().tolist()
                        xy_argmax = (mode*scale+offset).cpu().tolist()
                        p_cpu = probability.cpu().numpy().reshape(-1, 56, 56)
                        for j, (local_index, frame_index, view_id) in enumerate(positions):
                            index = indices[local_index]
                            row, view = rows[index], VIEWS[view_id]
                            label = audit[row["window_id"], frame_index, view]
                            record = {"base_seed": seed, "fold_index": fold_id, "arm": arm, "phase": phase,
                                "sample_index": index, "window_id": row["window_id"], "ui_index": row["ui_index"],
                                "frame_index": frame_index, "view": view, "target_xy_original": label["wire"]["xy"],
                                "expectation_xy_original": xy_expectation[j], "argmax_xy_original": xy_argmax[j],
                                **{k: arrays[k][j] for k in METRICS}}
                            point_records.append(record)
                            if earliest[row["window_id"], view] == frame_index and (
                                    phase == "heldout" or fold_id == TRAIN_VISUAL_FOLD):
                                key = f"map_{len(maps):03d}"
                                maps[key] = p_cpu[j]
                                image_records.append({**record, "array_key": key,
                                    "image_path": inputs[index]["model_input"]["sequence"][frame_index]["images"][view]})
                    if any(not torch.equal(v.cpu(), checkpoint["model_state"][k]) for k, v in model.state_dict().items()):
                        raise ValueError("inference changed a parameter")
                    if before != (path.stat().st_size, path.stat().st_mtime_ns):
                        raise ValueError("checkpoint changed during read-only diagnosis")
                    model_records.append({"base_seed": seed, "fold_index": fold_id, "arm": arm,
                                          "parameter_values_unchanged": True, "checkpoint_stat_unchanged": True})
                    del output, model
                print(f"inferred seed={seed} fold={fold_id} (both saved arms)", flush=True)
                del batches
    if len(point_records) != 1500 or len(image_records) != 108:
        raise ValueError("expected 1350 train/150 heldout evaluations and 108 preselected visualization maps")
    summaries = []
    for seed in SEEDS:
        for arm in ARMS:
            for phase in PHASES:
                for view in VIEWS:
                    values = [r for r in point_records if (r["base_seed"], r["arm"], r["phase"], r["view"])
                              == (seed, arm, phase, view)]
                    summaries.append({"base_seed": seed, "arm": arm, "phase": phase, "view": view,
                                      **balanced_point_summary(values)})
    means = []
    for arm in ARMS:
        for phase in PHASES:
            for view in VIEWS:
                values = [r for r in summaries if (r["arm"], r["phase"], r["view"]) == (arm, phase, view)]
                means.append({"arm": arm, "phase": phase, "view": view,
                              "metrics": {m: float(np.mean([r["metrics"][m] for r in values])) for m in METRICS}})
    np.savez_compressed(args.out / "visual_heatmaps.npz", **maps)
    write_jsonl(args.out / "visual_heatmaps_index.jsonl", image_records)
    write_jsonl(args.out / "point_diagnostics.jsonl", point_records)
    write_jsonl(args.out / "model_readback.jsonl", model_records)
    write_json(args.out / "report.json", {"status": "complete_read_only_fit_diagnosis", "source": args.source.as_posix(),
        "model_count": 60, "distinct_tip_points": 25, "point_windows": 5, "train_point_evaluations": 1350,
        "heldout_point_evaluations": 150, "visual_maps": 108, "seed_summaries": summaries,
        "means_across_seeds": means, "maximum_heldout_replay_difference": replay_max,
        "all_model_values_and_checkpoint_stats_unchanged": True,
        "elapsed_s": time.perf_counter()-started, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
        "optimizer_steps": 0, "new_labels": False, "diagnostic_argmax_is_not_a_new_model": True,
        "policy_input_allowed": False, "deployable": False, "visual_status": "not_viewed"})
    print(f"Complete: 60 unchanged checkpoints, 1500 point evaluations in {time.perf_counter()-started:.2f}s", flush=True)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from PIL import Image

    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path"})
    report = read_json(args.out / "report.json")
    source = Path(report["source"])
    roi = read_json(source / "source/protocol.json")["roi_xyxy_original"]
    index = read_jsonl(args.out / "visual_heatmaps_index.jsonl")
    maps = np.load(args.out / "visual_heatmaps.npz")
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 只读定位诊断：数据来源\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "|---|---|---|---|---|---|\n"
        "| 训练/留出与均值/峰值 | fit_metrics.csv | real | ../report.json，全部60份既有权重 | "
        "tools/diagnose_real10_spatial_aux_fit.py | spatial_fit_diagnosis_zh.png/.svg |\n"
        "| 九个窗/视角热图 | ../visual_heatmaps.npz/index.jsonl | real | 原ROI224及上述checkpoint | "
        "同上 | distribution_case_*_zh.png/.svg |\n\n"
        "聚合：同一点的9个训练模型误差先平均，再逐窗平均；留出为同一点的1个模型。\n"
        "不集成坐标、不按1350次训练观测当独立样本。全部三seed，两相机分别统计。\n"
        "图例训练侧固定fold0，所有点均在该折训练；留出侧各点用自身轨迹留出模型。\n"
        "每窗每视角选择最早有效tip帧，全部3seed/2组，非最优样例。概率色条固定log10(p/均匀概率)，"
        "0为均匀；色条[-1,2]范围外截断仅供显示。原图px不可合成mm。\n"
        "峰值仅为离线诊断，不改原空间期望读出。目标两格邻域来自原Gaussian编码宽度，不是新判定阈值。\n",
        encoding="utf-8")
    with (figures / "fit_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("base_seed", "arm", "phase", "view", *METRICS))
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("base_seed", "arm", "phase", "view")}|r["metrics"]
                         for r in report["seed_summaries"])
    colors = ("#0077BB", "#EE7733")
    lookup = {(r["base_seed"], r["arm"], r["phase"], r["view"]): r["metrics"] for r in report["seed_summaries"]}
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    for v, view in enumerate(VIEWS):
        for a, arm in enumerate(ARMS):
            for i, seed in enumerate(SEEDS):
                x = a + (i-1)*.14
                train, test = (lookup[seed, arm, phase, view]["expectation_error_px"] for phase in PHASES)
                axes[0, v].plot([x-.025, x+.025], [train, test], color="#BBBBBB", lw=1)
                axes[0, v].scatter(x-.025, train, s=35, marker="o", facecolors="none", edgecolors=colors[a])
                axes[0, v].scatter(x+.025, test, s=35, marker="o", color=colors[a])
        axes[0, v].set(xticks=[0, 1], xticklabels=["仅窗口监督", "+尖端辅助"], ylabel="逐窗平均误差（原图px）",
                       title=f"{view.capitalize()}：训练点与留出点",
                       ylim=(0, 1.15*max(lookup[s, a, p, view]["expectation_error_px"]
                                         for s in SEEDS for a in ARMS for p in PHASES)))
        for i, seed in enumerate(SEEDS):
            for p, phase in enumerate(PHASES):
                x = p+(i-1)*.14
                value = lookup[seed, ARMS[1], phase, view]
                axes[1, v].plot([x-.025, x+.025], [value["expectation_error_px"], value["argmax_error_px"]], color="#AAAAAA", lw=1)
                axes[1, v].scatter(x-.025, value["expectation_error_px"], color="#EE7733", marker="o", s=32)
                axes[1, v].scatter(x+.025, value["argmax_error_px"], color="#A23B72", marker="x", s=36)
        axes[1, v].set(xticks=[0, 1], xticklabels=["训练点", "留出点"], ylabel="逐窗平均误差（原图px）",
                       title=f"{view.capitalize()}：辅助组的均值与峰值",
                       ylim=(0, 1.15*max(lookup[s, ARMS[1], p, view][m]
                                         for s in SEEDS for p in PHASES
                                         for m in ("expectation_error_px", "argmax_error_px"))))
    for row in axes:
        for ax in row:
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
    axes[0, 0].legend(handles=[Line2D([], [], marker="o", color="#555555", markerfacecolor="none", lw=0, label="空心：训练"),
                              Line2D([], [], marker="o", color="#555555", lw=0, label="实心：留出")], frameon=False, fontsize=8)
    axes[1, 0].legend(handles=[Line2D([], [], marker="o", color="#EE7733", lw=0, label="现有均值读出"),
                              Line2D([], [], marker="x", color="#A23B72", lw=0, label="诊断峰值（未替换模型）")], frameon=False, fontsize=8)
    fig.suptitle("空间定位分支：已训练权重的只读诊断", fontsize=15, y=.99)
    fig.text(.5, .025, "同一批25个人工tip点 · 全部3seed · 训练侧为9次见过该点的模型误差均值，不是新增样本", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .96), h_pad=2)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"spatial_fit_diagnosis_zh.{ext}", dpi=450)
    plt.close(fig)
    gallery = []
    for ui, view in sorted({(r["ui_index"], r["view"]) for r in index}):
        selected = {(r["base_seed"], r["arm"], r["phase"]): r for r in index
                    if (r["ui_index"], r["view"]) == (ui, view)}
        fig, axes = plt.subplots(4, 3, figsize=(8.8, 12.6))
        row_spec = [(ARMS[0], "train"), (ARMS[0], "heldout"), (ARMS[1], "train"), (ARMS[1], "heldout")]
        x0, y0, x1, y1 = roi[view]
        def pixel(xy):
            return (np.asarray(xy)-[x0, y0]+.5)/[x1-x0, y1-y0]*224-.5
        for r, (arm, phase) in enumerate(row_spec):
            for c, seed in enumerate(SEEDS):
                item = selected[seed, arm, phase]
                with Image.open(item["image_path"]) as image:
                    axes[r, c].imshow(np.asarray(image.convert("RGB")))
                log_ratio = np.log10(np.maximum(maps[item["array_key"]]*3136, 1e-9))
                im = axes[r, c].imshow(log_ratio, extent=(-2, 222, 222, -2), cmap="magma", vmin=-1, vmax=2,
                                       alpha=.62, interpolation="nearest")
                for field, marker, color in (("target_xy_original", "x", "#009988"),
                                             ("expectation_xy_original", "o", "#EE7733"),
                                             ("argmax_xy_original", "+", "#FFFFFF")):
                    xy = pixel(item[field])
                    axes[r, c].scatter(*xy, marker=marker, color=color, linewidths=1.5, s=65,
                                       facecolors="none" if marker == "o" else color)
                axes[r, c].set(xticks=[], yticks=[], xlim=(-.5, 223.5), ylim=(223.5, -.5))
                if r == 0:
                    axes[r, c].set_title(str(seed), fontsize=10)
                if c == 0:
                    name = "仅窗口" if arm == ARMS[0] else "+辅助"
                    axes[r, c].set_ylabel(f"{name} / {'训练fold0' if phase == 'train' else '留出'}", fontsize=9)
                axes[r, c].set_xlabel(f"均值 {item['expectation_error_px']:.0f}px / 峰值 {item['argmax_error_px']:.0f}px\n"
                                      f"归一化熵 {item['entropy_normalized']:.3f}", fontsize=8)
        fig.suptitle(f"#{ui} {view.capitalize()} · 最早有效tip帧{item['frame_index']} · 全部固定种子", fontsize=13, y=.995)
        fig.text(.5, .962, "绿色× 人工尖端　橙色○ 均值读出　白色+ 最高概率位置", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .055, .89, .95), h_pad=1.5)
        cax = fig.add_axes([.91, .22, .016, .56])
        bar = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1, 2])
        bar.set_label("log10(概率 / 均匀概率)：0为均匀；各图同尺度", fontsize=8, labelpad=3)
        fig.text(.5, .015, "训练与留出用不同已保存模型；未重训。颜色仅显示概率分布，不代表可见性或碰壁概率。", ha="center", fontsize=8)
        name = f"distribution_case_{ui}_{view}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=450)
        plt.close(fig)
        gallery.append(f'<h2>#{ui} {view.capitalize()}</h2><img src="figures/{name}.png">')
    (args.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>训练与留出定位诊断</title>'
        '<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1150px;margin:30px auto;color:#223}'
        'img{width:100%;height:auto}p{line-height:1.8}</style><h1>空间定位分支：只读诊断</h1>'
        '<p>使用原60份checkpoint，不重训、不换读出、不改标签。均值误差先对同一点的9个训练模型平均，'
        '再逐窗平均，与该点1个留出模型比较。热图训练侧固定fold0，全部种子、每窗/视角最早有效tip。</p>'
        '<img src="figures/spatial_fit_diagnosis_zh.png">'+''.join(gallery), encoding="utf-8")
    if not (args.out / "review.json").exists():
        write_json(args.out / "review.json", {"visual_status": "not_viewed", "user_acceptance": "pending"})
    print(f"Rendered summary and nine window/view distribution sheets: {args.out / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("infer", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_spatial_aux_fit_diagnostic_v1"))
    args = parser.parse_args()
    {"infer": infer, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
