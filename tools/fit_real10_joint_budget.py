"""Add only the fixed 2000-step joint fit; reuse the completed pure-tip control.

Training-only diagnosis on original fold0. No heldout scoring or policy export.
fit: remote project4090; render: local .venv using exported diagnostics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from diagnose_real10_spatial_aux_fit import (
    METRICS, balanced_point_summary, read_json, read_jsonl, write_json, write_jsonl)


SCHEMA = "real10_joint_budget_training_fit_v1"
SEEDS = (20261020, 20261120, 20261220)
VIEWS = ("side", "top")
STAGES = ("joint_200", "tip_only_200", "joint_2000", "tip_only_2000")
CONFIG = {"seeds": list(SEEDS), "fold_index": 0, "steps_per_seed": 2000,
          "fit_function": "unchanged train_real10_spatial_aux_pair.fit",
          "arm": "spatial_window_tip_aux", "initialization": "same old fold0 initial_state",
          "selection": "final step only, all seeds", "training_windows": 36,
          "localization_weight": .1, "comparison": "equal updates, not equal compute"}
DESIGN = Path("docs/algorithm-real10-joint-budget-protocol-20260922.md")


def fit(args):
    from train_real10_spatial_aux_pair import (
        atomic_torch_save, check_frozen, checkpoint_path, configure, cpu_state,
        fit as original_fit, forward, load_tensors, subset, validate_checkpoint, writer_lock)
    from real10_spatial_response import SpatialResponse, point_distribution, response_losses, spatial_grid
    from fit_real10_tip_only import CONFIG as PURE_CONFIG
    import torch

    started = time.perf_counter()
    configure()
    inputs, targets, _, folds = check_frozen(args.source)
    pure_protocol, pure_report = (read_json(args.pure / name) for name in ("protocol.json", "report.json"))
    if (pure_protocol["config"] != PURE_CONFIG or pure_protocol["source"] != args.source.as_posix()
            or pure_report["status"] != "complete_training_only_fit"):
        raise ValueError("use the completed fixed pure-tip comparison")
    if (args.pure / "entrypoint_snapshot.py").read_bytes() != Path("tools/fit_real10_tip_only.py").read_bytes():
        raise ValueError("pure-tip entrypoint differs from its executed snapshot")
    if args.out.exists():
        if not args.resume:
            raise FileExistsError("preserve this output; --resume skips complete seeds")
        protocol = read_json(args.out / "protocol.json")
        if (protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix()
                or protocol["pure"] != args.pure.as_posix()
                or (args.out / "entrypoint_snapshot.py").read_bytes() != Path(__file__).read_bytes()):
            raise ValueError("do not mix run versions")
    else:
        args.out.mkdir(parents=True)
        (args.out / "checkpoints").mkdir()
        shutil.copy2(__file__, args.out / "entrypoint_snapshot.py")
        shutil.copy2(DESIGN, args.out / "design_protocol.md")
        write_json(args.out / "protocol.json", {
            "schema": SCHEMA, "config": CONFIG, "source": args.source.as_posix(), "pure": args.pure.as_posix(),
            "source_runner_snapshot": (args.source / "runner_snapshot.py").as_posix(),
            "source_model_snapshot": (args.source / "model_snapshot.py").as_posix(),
            "roi_xyxy_original": pure_protocol["roi_xyxy_original"],
            "training_scope": "fold0 train only; no heldout graph, score or model selection",
            "new_optimizer_steps": 6000, "timing_estimate_s": 180,
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "policy_input_allowed": False, "deployable": False, "new_labels": False})
    with writer_lock(args.out):
        if (args.out / "report.json").is_file():
            print("Already complete; no new optimization.", flush=True)
            return
        pure_points = read_jsonl(args.pure / "point_diagnostics.jsonl")
        tensors = load_tensors(args.source, inputs, targets)
        train = subset(tensors, folds[0]["train_indices"])
        del tensors
        positions = train["tip_valid"].nonzero().cpu().tolist()
        if len(train["response_y"]) != 36 or len(positions) != 25:
            raise ValueError("expected the fixed 36 windows and all 25 training points")
        meta_lookup = {(r["sample_index"], r["frame_index"], r["view"]): r for r in pure_points
                       if r["base_seed"] == SEEDS[0] and r["stage"] == "initial"}
        metadata = [meta_lookup[folds[0]["train_indices"][i], frame, VIEWS[v]] for i, frame, v in positions]
        earliest = {}
        for r in metadata:
            key = r["window_id"], r["view"]
            earliest[key] = min(r["frame_index"], earliest.get(key, r["frame_index"]))
        mask = train["tip_valid"]
        target, point_maps = train["xy_uv"][mask], train["maps"][mask]
        q = point_distribution(target, torch.ones(25, device="cuda", dtype=torch.bool)).flatten(1)
        grid = spatial_grid("cuda", torch.float32).reshape(-1, 2)
        roi = pure_protocol["roi_xyxy_original"]
        scale = torch.tensor([[roi[VIEWS[v]][2]-roi[VIEWS[v]][0], roi[VIEWS[v]][3]-roi[VIEWS[v]][1]]
                              for _, _, v in positions], device="cuda")
        offset = torch.tensor([[roi[VIEWS[v]][0]-.5, roi[VIEWS[v]][1]-.5]
                               for _, _, v in positions], device="cuda")
        log_cells = float(np.log(56**2))
        all_train = torch.ones(36, device="cuda", dtype=torch.bool)
        positives = train["response_y"].sum()
        pos_weight = (len(all_train)-positives)/positives

        @torch.no_grad()
        def diagnose(model, seed):
            full = forward(model, train)
            losses = response_losses(full, train["response_y"], train["xy_uv"], mask,
                                     all_train, pos_weight, "spatial_window_tip_aux")
            logits = model.location(model.project(point_maps)).flatten(1)
            logits_diff = float((logits-full["heatmap_logits"][mask].flatten(1)).abs().max())
            logp = logits.log_softmax(-1)
            p = logp.exp()
            xy, mode = p @ grid, grid[logp.argmax(-1)]
            xy_diff = float((xy-full["location_uv"][mask]).abs().max())
            if logits_diff > 1e-5 or xy_diff > 2e-6:
                raise ValueError("point diagnostic changed the original spatial operation")
            entropy = -(p*logp).sum(1)
            region = (((grid[None]-target[:, None])*56).square().sum(-1) <= 4)
            values = {
                "expectation_error_px": ((xy-target)*scale).norm(dim=1),
                "argmax_error_px": ((mode-target)*scale).norm(dim=1),
                "uniform_expectation_error_px": ((grid.mean(0)-target)*scale).norm(dim=1),
                "expectation_to_argmax_px": ((xy-mode)*scale).norm(dim=1),
                "entropy_normalized": entropy/log_cells,
                "effective_cell_fraction": entropy.exp()/len(grid),
                "peak_over_uniform": p.max(1).values*len(grid),
                "mass_within_two_grid_cells": (p*region).sum(1),
                "uniform_mass_same_region": region.float().mean(1),
                "target_mass_same_region": (q*region).sum(1),
                "target_ce_normalized": -(q*logp).sum(1)/log_cells,
                "target_entropy_normalized": -(q*q.clamp_min(1e-30).log()).sum(1)/log_cells,
                "spatial_rms_radius_px": (p*((grid[None]-xy[:, None])*scale[:, None]).square().sum(-1)).sum(1).sqrt()}
            values = {k: v.cpu().tolist() for k, v in values.items()}
            expect_xy, mode_xy = (xy*scale+offset).cpu().tolist(), (mode*scale+offset).cpu().tolist()
            rows, heatmaps = [], {}
            for j, meta in enumerate(metadata):
                key = f"seed{seed}_point{j:02d}" if meta["frame_index"] == earliest[meta["window_id"], meta["view"]] else None
                rows.append({**meta, "base_seed": seed, "stage": "joint_2000", "phase": "training_only",
                    "expectation_xy_original": expect_xy[j], "argmax_xy_original": mode_xy[j],
                    "array_key": key, "heatmap_source": "joint", **{k: values[k][j] for k in METRICS}})
                if key:
                    heatmaps[key] = p[j].cpu().reshape(56, 56)
            checks = {"point_vs_full_logits_max_abs": logits_diff, "point_vs_full_location_uv_max_abs": xy_diff}
            return rows, heatmaps, {k: float(losses[k]) for k in ("total", "response", "localization")}, checks

        torch.cuda.reset_peak_memory_stats()
        results = []
        for seed in SEEDS:
            completed = args.out / f"seed{seed}_complete.pt"
            if completed.is_file():
                saved = torch.load(completed, map_location="cpu", weights_only=True)
                if saved["config"] != CONFIG or saved["seed"] != seed:
                    raise ValueError("completed seed protocol mismatch")
                results.append(saved)
                print(f"seed={seed}: preserved completed joint fit", flush=True)
                continue
            old_path = checkpoint_path(args.source, seed, 0, "spatial_window_tip_aux")
            protected = (old_path, args.pure / "report.json", args.pure / "point_diagnostics.jsonl",
                         args.pure / "checkpoints" / f"seed{seed}_step2000.pt")
            before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]
            old = torch.load(old_path, map_location="cpu", weights_only=True)
            validate_checkpoint(old, seed, 0, folds[0], "spatial_window_tip_aux")
            print(f"seed={seed}: original joint objective, fold0, fixed 2000 updates", flush=True)
            model, initial, timing = original_fit(train, seed, "spatial_window_tip_aux", 2000)
            if any(not torch.equal(v, old["initial_state"][k]) for k, v in initial.items()):
                raise ValueError("initialization differs from old matched fold0")
            model.eval()
            records, heatmaps, losses, checks = diagnose(model, seed)
            path = args.out / "checkpoints" / f"seed{seed}_fold00_joint_step2000.pt"
            atomic_torch_save({"schema": SCHEMA, "config": CONFIG, "base_seed": seed, "fold_index": 0,
                "fit": timing, "initial_state": initial, "model_state": cpu_state(model),
                "training_losses": losses, "policy_input_allowed": False, "deployable": False}, path)
            replay = SpatialResponse().cuda().eval()
            replay.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"])
            with torch.no_grad():
                original, restored = forward(model, train), forward(replay, train)
                replay_errors = {k: float((original[k]-restored[k]).abs().max())
                                 for k in ("window_logit", "heatmap_logits", "location_uv")}
            if max(replay_errors.values()) > 1e-6 or before != [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]:
                raise ValueError("saved replay differs or source artifacts changed")
            checks.update({"initial_state_matches": True, "protected_source_stats_unchanged": True,
                           "saved_replay_max_abs": replay_errors})
            result = {"seed": seed, "config": CONFIG, "records": records, "heatmaps": heatmaps,
                      "training_losses": losses, "fit": timing, "checks": checks}
            atomic_torch_save(result, completed)
            results.append(result)
            print(f"seed={seed}: fit={timing['fit_s']:.2f}s BCE={losses['response']:.6f} tipCE={losses['localization']:.6f}", flush=True)
            del model, replay, original, restored
        points = [{**r, "heatmap_source": "pure"} for r in pure_points if r["stage"] in STAGES]
        maps, checks = {}, []
        for result in results:
            points.extend(result["records"])
            maps.update({k: v.numpy() for k, v in result["heatmaps"].items()})
            checks.append({"base_seed": result["seed"], "fit": result["fit"],
                           "training_losses": result["training_losses"], **result["checks"]})
        summaries = [{"base_seed": seed, "stage": stage, "view": view,
                      **balanced_point_summary([r for r in points if (r["base_seed"], r["stage"], r["view"])
                                                == (seed, stage, view)])}
                     for seed in SEEDS for stage in STAGES for view in VIEWS]
        means = [{"stage": stage, "view": view,
                  "metrics": {m: float(np.mean([r["metrics"][m] for r in summaries
                                               if (r["stage"], r["view"]) == (stage, view)])) for m in METRICS}}
                 for stage in STAGES for view in VIEWS]
        if len(points) != 300 or len(maps) != 27:
            raise ValueError("expected 300 point readings and only 27 new representative maps")
        write_jsonl(args.out / "point_diagnostics.jsonl", points)
        write_jsonl(args.out / "model_readback.jsonl", checks)
        np.savez_compressed(args.out / "heatmaps.npz", **maps)
        write_json(args.out / "report.json", {"status": "complete_joint_budget_training_diagnostic",
            "config": CONFIG, "source": args.source.as_posix(), "pure": args.pure.as_posix(),
            "distinct_points": 25, "point_windows": 5, "new_point_evaluations": 75, "total_comparison_rows": 300,
            "seed_summaries": summaries, "means_across_seeds": means, "new_optimizer_steps": 6000,
            "fit_s_total": sum(r["fit"]["fit_s"] for r in results), "invocation_elapsed_s": time.perf_counter()-started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "generalization_evaluated": False, "response_ba_evaluated": False,
            "policy_input_allowed": False, "deployable": False, "visual_status": "not_viewed"})
        print(f"Complete: 3 new joint fits, no control reruns; {time.perf_counter()-started:.2f}s", flush=True)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "path"})
    report, protocol = (read_json(args.out / f"{name}.json") for name in ("report", "protocol"))
    points = read_jsonl(args.out / "point_diagnostics.jsonl")
    maps = {"joint": np.load(args.out / "heatmaps.npz"), "pure": np.load(args.pure / "heatmaps.npz")}
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 同预算训练拟合对照：真实实验数据\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n"
        "| 四组固定预算比较 | ../report.json | real | 原25tip点，全部3seed，fold0训练侧 | "
        "tools/fit_real10_joint_budget.py --stage render | joint_budget_fit_zh.png/.svg |\n"
        "| 九组固定帧热图 | ../point_diagnostics.jsonl, ../heatmaps.npz, pure/heatmaps.npz | real | "
        "原ROI224，旧联合200/新联合2000/原纯定位2000 | 同上 | joint_case_*_zh.png/.svg |\n\n"
        "全部为训练点，不能证明泛化、响应识别或策略收益。同预算只指优化更新数，不是相同计算量。\n"
        "相机内窗宏平均，保留全部3seed；每窗/视角选最早有效tip，无最佳帧/模型选择。\n"
        "原图px不换算mm。argmax只用于诊断，未替换原空间期望。两格邻域不是正确性阈值。\n"
        "颜色固定log10(p/均匀概率)，统一[-1,2]；超过色阶范围只作显示截断。\n"
        f"pure={args.pure.as_posix()}。旧热图不复制、不重新推理。\n", encoding="utf-8")
    colors = ("#0077BB", "#EE7733", "#009988")
    names = {"joint_200": "联合\n200步", "tip_only_200": "纯定位\n200步",
             "joint_2000": "联合\n2000步", "tip_only_2000": "纯定位\n2000步"}
    lookup = {(r["base_seed"], r["stage"], r["view"]): r["metrics"] for r in report["seed_summaries"]}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.4))
    metrics = (("expectation_error_px", "原空间均值误差（px）", 1),
               ("argmax_error_px", "诊断峰值误差（px）", 1),
               ("mass_within_two_grid_cells", "目标两格概率质量（%）", 100))
    for row, view in enumerate(VIEWS):
        for col, (metric, label, scale) in enumerate(metrics):
            ax = axes[row, col]
            for i, seed in enumerate(SEEDS):
                ax.plot(range(4), [scale*lookup[seed, stage, view][metric] for stage in STAGES],
                        "o-", color=colors[i], ms=4, lw=1.2, label=f"seed {seed}")
            if col == 0:
                ax.axhline(lookup[SEEDS[0], STAGES[0], view]["uniform_expectation_error_px"],
                           color="#888888", ls="--", lw=1, label="均匀分布期望")
            ax.set(xticks=range(4), xticklabels=[names[s] for s in STAGES], ylabel=label,
                   title=f"{view.capitalize()} · {label.split('（')[0]}")
            ax.set_ylim(bottom=0)
            if col == 2:
                ax.set_ylim(0, 100)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("同预算训练拟合对照：原联合目标 vs 纯定位目标", y=.985, fontsize=13)
    fig.text(.5, .018, "同25个人工tip点 / 同初始化 / 全3seed；全部为训练点，无泛化结论。同预算指更新数，不是计算时间。",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.12, left=.075, right=.98, hspace=.54, wspace=.36)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"joint_budget_fit_zh.{ext}", dpi=450)
    plt.close(fig)
    earliest = {}
    for r in points:
        key = r["ui_index"], r["view"]
        earliest[key] = min(r["frame_index"], earliest.get(key, r["frame_index"]))
    shown = ("joint_200", "joint_2000", "tip_only_2000")
    links = []
    for (ui, view), frame in sorted(earliest.items()):
        index = {(r["base_seed"], r["stage"]): r for r in points
                 if r["ui_index"] == ui and r["view"] == view and r["frame_index"] == frame}
        fig, axes = plt.subplots(3, 3, figsize=(9.4, 9.3))
        roi = protocol["roi_xyxy_original"][view]
        wh, origin = np.array(roi[2:])-np.array(roi[:2]), np.array(roi[:2])-.5
        for i, seed in enumerate(SEEDS):
            for j, stage in enumerate(shown):
                r, ax = index[seed, stage], axes[i, j]
                with Image.open(r["image_path"]) as img:
                    ax.imshow(img.convert("RGB"), extent=(0, 224, 224, 0))
                heat = np.log10(np.maximum(maps[r["heatmap_source"]][r["array_key"]]*56**2, 1e-10))
                im = ax.imshow(heat, extent=(-1.5, 222.5, 222.5, -1.5), cmap="magma",
                               vmin=-1, vmax=2, alpha=.48, interpolation="nearest")
                for key, marker, color in (("target_xy_original", "+", "#33BBEE"),
                                            ("expectation_xy_original", "o", "#009988"),
                                            ("argmax_xy_original", "x", "#EE7733")):
                    xy = (np.array(r[key])-origin)/wh*224
                    ax.plot(*xy, marker=marker, color=color, markersize=9, markerfacecolor="none", markeredgewidth=1.4)
                ax.set(xlim=(0, 224), ylim=(224, 0), xticks=[], yticks=[])
                ax.set_title(f"{names[stage].replace(chr(10), ' ')} · seed {seed}\n"
                             f"均值{r['expectation_error_px']:.1f}px / 峰值{r['argmax_error_px']:.1f}px / "
                             f"邻域{100*r['mass_within_two_grid_cells']:.1f}%", fontsize=8)
        fig.suptitle(f"#{ui} {view.capitalize()} · 最早有效尖端帧{frame} · 同预算训练内对照", y=.988, fontsize=12)
        fig.text(.5, .018, "青色＋ 人工尖端；绿色○ 空间期望；橙色× 概率峰值。全部为训练点，不代表留出定位能力。",
                 ha="center", fontsize=8)
        fig.subplots_adjust(top=.92, bottom=.08, left=.03, right=.93, hspace=.34, wspace=.12)
        fig.colorbar(im, cax=fig.add_axes((.95, .2, .012, .56)), label="log10(概率 / 均匀概率)")
        stem = f"joint_case_{ui}_{view}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{stem}.{ext}", dpi=450)
        plt.close(fig)
        links.append(f'<h2>#{ui} {view.capitalize()}</h2><img src="figures/{stem}.png">')
    (args.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>同预算训练拟合对照</title>'
        '<style>body{max-width:1100px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>同预算训练拟合对照</h1><p>全部为训练点；固定3seed，未改架构、损失权重、标注或输入。无泛化/策略收益结论。</p>'
        '<img src="figures/joint_budget_fit_zh.png">'+"".join(links), encoding="utf-8")
    print(f"Rendered summary and {len(links)} fixed case sheets; acceptance pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--pure", type=Path, default=Path("simulation_output/real10_tip_only_fit_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_joint_budget_fit_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (fit if args.stage == "fit" else render)(args)


if __name__ == "__main__":
    main()
