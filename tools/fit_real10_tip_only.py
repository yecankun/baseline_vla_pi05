"""Fixed training-only localization diagnostic; never a deployable tip estimator.

fit runs on project4090; render reads only small exported files on local Windows.
The original spatial model, cache, labels, splits and checkpoints are read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import time

import numpy as np

from diagnose_real10_spatial_aux_fit import (
    METRICS, balanced_point_summary, read_json, read_jsonl, write_json, write_jsonl)


SCHEMA = "real10_tip_only_training_fit_v1"
SEEDS = (20261020, 20261120, 20261220)
VIEWS = ("side", "top")
STAGES = ("initial", "joint_200", "tip_only_200", "tip_only_2000")
CONFIG = {"seeds": list(SEEDS), "initialization": "old fold0 initial_state",
          "checkpoints": [200, 2000], "localization_weight": .1,
          "optimizer": "AdamW", "lr": .001, "weight_decay": .01, "gradient_clip": 1.,
          "target": "unchanged Gaussian sigma1, CE/log(56^2), point-then-window macro",
          "response_head": "frozen, unused", "encoder": "same frozen ResNet18 layer1 cache",
          "precision": "FP32, deterministic, no AMP/TF32", "selection": "no selection"}
DESIGN = Path("docs/algorithm-real10-tip-only-fit-protocol-20260922.md")


def fit(args):
    # Import the existing runner first: it sets CUBLAS determinism before CUDA use.
    from train_real10_spatial_aux_pair import (
        atomic_torch_save, check_frozen, checkpoint_path, configure, cpu_state,
        forward, load_tensors, seed_all, subset, validate_checkpoint, writer_lock)
    from real10_spatial_response import SpatialResponse, point_distribution, response_losses, spatial_grid
    import torch

    started = time.perf_counter()
    configure()
    inputs, targets, rows, folds = check_frozen(args.source)
    audit = {(r["window_id"], r["frame_index"], r["view"]): r
             for r in read_jsonl(args.source / "source/point_audit.jsonl") if r["tip_target_valid"]}
    if len(audit) != 25 or any(r["source_episode"] == folds[0]["heldout_episode"] for r in audit.values()):
        raise ValueError("expected all original 25 tip points in old fold0 training")
    if args.out.exists():
        if not args.resume:
            raise FileExistsError("preserve this output; --resume skips completed seeds")
        protocol = read_json(args.out / "protocol.json")
        if protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix():
            raise ValueError("cannot mix fit protocols")
        if (args.out / "entrypoint_snapshot.py").read_bytes() != Path(__file__).read_bytes():
            raise ValueError("fit entrypoint changed; inspect before resuming")
    else:
        args.out.mkdir(parents=True)
        (args.out / "checkpoints").mkdir()
        shutil.copy2(__file__, args.out / "entrypoint_snapshot.py")
        shutil.copy2("tools/real10_spatial_response.py", args.out / "model_snapshot.py")
        shutil.copy2(DESIGN, args.out / "design_protocol.md")
        write_json(args.out / "protocol.json", {
            "schema": SCHEMA, "source": args.source.as_posix(), "config": CONFIG,
            "scope": "training fit only, all 25 points; not LOEO or generalization",
            "distinct_points": 25, "point_windows": 5, "original_fold": 0,
            "planned_optimizer_steps": 6000, "new_labels": False,
            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False,
            "roi_xyxy_original": read_json(args.source / "source/protocol.json")["roi_xyxy_original"],
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)})
    with writer_lock(args.out):
        if (args.out / "report.json").is_file():
            print("Already complete; no optimization performed. Read report.json.", flush=True)
            return
        tensors = load_tensors(args.source, inputs, targets)
        mask = tensors["tip_valid"]
        positions = mask.nonzero().cpu().tolist()
        maps, target = tensors["maps"][mask], tensors["xy_uv"][mask]
        windows = sorted({pos[0] for pos in positions})
        counts = {w: sum(p[0] == w for p in positions) for w in windows}
        weights = torch.tensor([1 / (len(windows)*counts[p[0]]) for p in positions], device="cuda")
        q = point_distribution(target, torch.ones(25, device="cuda", dtype=torch.bool)).flatten(1)
        log_cells = float(np.log(56**2))
        grid = spatial_grid("cuda", torch.float32).reshape(-1, 2)
        roi = read_json(args.out / "protocol.json")["roi_xyxy_original"]
        scale = torch.tensor([[roi[VIEWS[v]][2]-roi[VIEWS[v]][0], roi[VIEWS[v]][3]-roi[VIEWS[v]][1]]
                              for _, _, v in positions], device="cuda")
        offset = torch.tensor([[roi[VIEWS[v]][0]-.5, roi[VIEWS[v]][1]-.5]
                               for _, _, v in positions], device="cuda")
        metadata = []
        for i, frame, v in positions:
            view = VIEWS[v]
            label = audit[rows[i]["window_id"], frame, view]
            metadata.append({"sample_index": i, "window_id": rows[i]["window_id"],
                "ui_index": rows[i]["ui_index"], "frame_index": frame, "view": view,
                "target_xy_original": label["wire"]["xy"],
                "image_path": inputs[i]["model_input"]["sequence"][frame]["images"][view]})
        if len(windows) != 5 or len(metadata) != 25:
            raise ValueError("fixed point subset changed")

        def logits_for(model):
            return model.location(model.project(maps)).flatten(1)

        def localization(logits):
            return (-(q * logits.log_softmax(-1)).sum(1) / log_cells * weights).sum()

        @torch.no_grad()
        def measure(model, seed, stage):
            logits = logits_for(model)
            logp = logits.log_softmax(-1)
            p = logp.exp()
            xy, mode = p @ grid, grid[logp.argmax(-1)]
            entropy = -(p * logp).sum(1)
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
            arrays = {k: value.cpu().tolist() for k, value in values.items()}
            expect_xy, mode_xy = (xy*scale+offset).cpu().tolist(), (mode*scale+offset).cpu().tolist()
            records = [{**meta, "base_seed": seed, "stage": stage, "phase": "training_only",
                        "expectation_xy_original": expect_xy[j], "argmax_xy_original": mode_xy[j],
                        **{key: arrays[key][j] for key in METRICS}} for j, meta in enumerate(metadata)]
            return records, p.cpu().reshape(-1, 56, 56), float(localization(logits))

        all_results = []
        torch.cuda.reset_peak_memory_stats()
        for seed in SEEDS:
            completed = args.out / f"seed{seed}_complete.pt"
            if completed.is_file():
                result = torch.load(completed, map_location="cpu", weights_only=True)
                if result["config"] != CONFIG or result["seed"] != seed:
                    raise ValueError("completed seed differs from this fit")
                all_results.append(result)
                print(f"seed={seed}: preserved completed 2000-step fit", flush=True)
                continue
            seed_started = time.perf_counter()
            old_path = checkpoint_path(args.source, seed, 0, "spatial_window_tip_aux")
            old_stat = (old_path.stat().st_size, old_path.stat().st_mtime_ns)
            old = torch.load(old_path, map_location="cpu", weights_only=True)
            validate_checkpoint(old, seed, 0, folds[0], "spatial_window_tip_aux")
            seed_all(seed)
            model = SpatialResponse().cuda().eval()
            model.load_state_dict(old["model_state"])
            joint_records, joint_maps, _ = measure(model, seed, "joint_200")
            model.load_state_dict(old["initial_state"])
            # A point-only batch must be the same localization operation and weighting.
            with torch.no_grad():
                batch = subset(tensors, windows)
                full = forward(model, batch)
                reference_loss = response_losses(full, batch["response_y"], batch["xy_uv"], batch["tip_valid"],
                    torch.ones(len(windows), device="cuda", dtype=torch.bool),
                    torch.tensor(1., device="cuda"), "spatial_window_tip_aux")["localization"]
                max_logits = float((full["heatmap_logits"][batch["tip_valid"]].flatten(1)-logits_for(model)).abs().max())
                loss_difference = abs(float(reference_loss)-float(localization(logits_for(model))))
                if max_logits > 1e-5 or loss_difference > 1e-6:
                    raise ValueError("point-batch forward or window weighting changed")
                del batch, full
            model.response.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]
            if sum(p.numel() for p in trainable) != 529:
                raise ValueError("only the original 529 localization parameters may train")
            optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.01)
            records, probabilities, initial_loss = measure(model, seed, "initial")
            stages = {"joint_200": (joint_records, joint_maps), "initial": (records, probabilities)}
            curve = [{"step": 0, "localization_ce": initial_loss}]
            checks = {"point_vs_full_logits_max_abs": max_logits, "point_vs_full_loss_abs": loss_difference,
                      "initial_state_from_old_fold0": True, "trainable_parameters": 529}
            for step in range(1, 2001):
                optimizer.zero_grad(set_to_none=True)
                loss = .1 * localization(logits_for(model))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                if step % 20 == 0:
                    with torch.no_grad():
                        curve.append({"step": step, "localization_ce": float(localization(logits_for(model)))})
                if step in (200, 2000):
                    stage = f"tip_only_{step}"
                    records, probabilities, post_loss = measure(model, seed, stage)
                    stages[stage] = (records, probabilities)
                    if any(not torch.equal(v.cpu(), old["initial_state"][k]) for k, v in model.state_dict().items()
                           if k.startswith("response.")):
                        raise ValueError("unused response parameters changed")
                    path = args.out / "checkpoints" / f"seed{seed}_step{step:04d}.pt"
                    checkpoint = {"schema": SCHEMA, "config": CONFIG, "base_seed": seed,
                                  "optimizer_steps": step, "model_state": cpu_state(model),
                                  "policy_input_allowed": False, "deployable": False,
                                  "training_localization_ce": post_loss}
                    atomic_torch_save(checkpoint, path)
                    replay = SpatialResponse().cuda().eval()
                    replay.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"])
                    with torch.no_grad():
                        replay_diff = float((logits_for(replay)-logits_for(model)).abs().max())
                    if replay_diff > 1e-6:
                        raise ValueError("saved localization replay differs")
                    checks[f"step{step}_saved_logits_max_abs"] = replay_diff
                    del replay
                    print(f"seed={seed} step={step} CE={post_loss:.6f} elapsed={time.perf_counter()-seed_started:.1f}s", flush=True)
            if old_stat != (old_path.stat().st_size, old_path.stat().st_mtime_ns):
                raise ValueError("old checkpoint changed")
            checks.update({"response_parameters_unchanged": True, "old_checkpoint_stat_unchanged": True})
            result = {"seed": seed, "config": CONFIG, "stages": stages, "curve": curve,
                      "checks": checks, "elapsed_s": time.perf_counter()-seed_started}
            atomic_torch_save(result, completed)
            all_results.append(result)
            del optimizer, model
        points, heatmaps, curves, checks = [], {}, [], []
        for result in all_results:
            seed = result["seed"]
            curves.extend({"base_seed": seed, **r} for r in result["curve"])
            checks.append({"base_seed": seed, "elapsed_s": result["elapsed_s"], **result["checks"]})
            for stage in STAGES:
                records, probabilities = result["stages"][stage]
                for j, record in enumerate(records):
                    key = f"map_{len(heatmaps):03d}"
                    points.append({**record, "array_key": key})
                    heatmaps[key] = probabilities[j].numpy()
        summaries = [{"base_seed": seed, "stage": stage, "view": view,
                      **balanced_point_summary([r for r in points if (r["base_seed"], r["stage"], r["view"])
                                                == (seed, stage, view)])}
                     for seed in SEEDS for stage in STAGES for view in VIEWS]
        means = [{"stage": stage, "view": view,
                  "metrics": {m: float(np.mean([r["metrics"][m] for r in summaries
                                               if (r["stage"], r["view"]) == (stage, view)])) for m in METRICS}}
                 for stage in STAGES for view in VIEWS]
        write_jsonl(args.out / "point_diagnostics.jsonl", points)
        write_jsonl(args.out / "loss_curves.jsonl", curves)
        write_jsonl(args.out / "model_readback.jsonl", checks)
        np.savez_compressed(args.out / "heatmaps.npz", **heatmaps)
        write_json(args.out / "report.json", {"status": "complete_training_only_fit", "config": CONFIG,
            "source": args.source.as_posix(), "distinct_points": 25, "point_windows": 5,
            "evaluation_rows": len(points), "seeds": list(SEEDS), "seed_summaries": summaries,
            "means_across_seeds": means, "optimizer_steps": 6000,
            "fit_s_total": sum(r["elapsed_s"] for r in checks), "invocation_elapsed_s": time.perf_counter()-started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "generalization_evaluated": False, "policy_input_allowed": False, "deployable": False,
            "visual_status": "not_viewed"})
        print(f"Complete: fixed 6000 updates, 300 train-point evaluations; {time.perf_counter()-started:.2f}s", flush=True)


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
    report, protocol = (read_json(args.out / f"{name}.json") for name in ("report", "protocol"))
    points, curves = (read_jsonl(args.out / f"{name}.jsonl") for name in ("point_diagnostics", "loss_curves"))
    maps = np.load(args.out / "heatmaps.npz")
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 纯定位训练拟合：真实实验数据\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n"
        "| 定位误差及固定预算曲线 | ../report.json, ../loss_curves.jsonl | real | 25个原人工tip点，3seed | "
        "tools/fit_real10_tip_only.py --stage render | tip_only_fit_zh.png/.svg |\n"
        "| 九组定位概率图 | ../point_diagnostics.jsonl, ../heatmaps.npz | real | 原ROI224，旧fold0与新固定step | "
        "同上 | tip_fit_case_*_zh.png/.svg |\n\n"
        "全部是训练点；不能证明新轨迹泛化、识别或策略收益。200→2000为预先固定的预算诊断。\n"
        "Side/Top分别窗内平均再窗平均，再汇总全部3seed；曲线为原loss的点-窗权重。\n"
        "每窗/视角最早有效tip，全3seed，不选最好帧。期望与argmax均显示；不替换原读出。\n"
        "热图颜色为log10(p/均匀概率)，统一[-1,2]，超过范围仅显示截断；两格邻域不是正确阈值。\n",
        encoding="utf-8")
    colors = ("#0077BB", "#EE7733", "#009988")
    shown = STAGES[1:]
    names = ("原联合\n200步", "纯定位\n200步", "纯定位\n2000步")
    lookup = {(r["base_seed"], r["stage"], r["view"]): r["metrics"] for r in report["seed_summaries"]}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for v, view in enumerate(VIEWS):
        for i, seed in enumerate(SEEDS):
            axes[0, v].plot(range(3), [lookup[seed, st, view]["expectation_error_px"] for st in shown],
                            "o-", color=colors[i], lw=1.2, ms=4, label=f"seed {seed}")
        axes[0, v].axhline(lookup[SEEDS[0], shown[0], view]["uniform_expectation_error_px"],
                           ls="--", color="#888888", lw=1, label="均匀分布期望")
        axes[0, v].set(xticks=range(3), xticklabels=names, ylabel="原图定位误差（px，窗平均）",
                       title=f"{view.capitalize()}：原空间期望读出")
        axes[0, v].set_ylim(bottom=0)
        for i, seed in enumerate(SEEDS):
            axes[1, 1].plot(range(3), [100*lookup[seed, st, view]["mass_within_two_grid_cells"] for st in shown],
                            "o-" if v == 0 else "s--", color=colors[i], lw=1.1, ms=3)
    for i, seed in enumerate(SEEDS):
        selected = [r for r in curves if r["base_seed"] == seed]
        axes[1, 0].plot([r["step"] for r in selected], [r["localization_ce"] for r in selected],
                       color=colors[i], lw=1.3)
    # Loss uses all points with original per-window weighting, not the two-view macro above.
    grouped = defaultdict(list)
    for r in points:
        if r["base_seed"] == SEEDS[0] and r["stage"] == "initial":
            grouped[r["window_id"]].append(r["target_entropy_normalized"])
    floor = float(np.mean([np.mean(rs) for rs in grouped.values()]))
    axes[1, 0].axhline(floor, color="#888888", ls="--", lw=1)
    axes[1, 0].axvline(200, color="#AAAAAA", ls=":", lw=1)
    axes[1, 0].set(xlabel="纯定位更新步数", ylabel="归一化目标CE（未乘0.1）",
                   title="固定预算拟合曲线（虚线为目标自身熵）", ylim=(0, None))
    axes[1, 1].set(xticks=range(3), xticklabels=names, ylabel="目标两格邻域概率质量（%）",
                   title="分布是否集中到目标附近", ylim=(0, 100))
    axes[1, 1].legend(handles=[Line2D([], [], color="#555555", marker="o", label="Side"),
                                Line2D([], [], color="#555555", marker="s", ls="--", label="Top")], fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("纯定位拟合诊断：去掉响应分类目标后，原分支能否学会已有尖端点？", y=.985, fontsize=13)
    fig.text(.5, .018, "25个原人工点 / 5窗 / 全部3seed；全部为训练点。2000步仅作预算诊断，不代表泛化或算法收益。",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.12, left=.09, right=.96, hspace=.57, wspace=.28)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"tip_only_fit_zh.{ext}", dpi=450)
    plt.close(fig)
    earliest = {}
    for r in points:
        key = r["ui_index"], r["view"]
        earliest[key] = min(r["frame_index"], earliest.get(key, r["frame_index"]))
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
                heat = np.log10(np.maximum(maps[r["array_key"]]*56**2, 1e-10))
                im = ax.imshow(heat, extent=(-1.5, 222.5, 222.5, -1.5), cmap="magma",
                               vmin=-1, vmax=2, alpha=.48, interpolation="nearest")
                for key, marker, color in (("target_xy_original", "+", "#33BBEE"),
                                            ("expectation_xy_original", "o", "#009988"),
                                            ("argmax_xy_original", "x", "#EE7733")):
                    xy = (np.array(r[key])-origin)/wh*224
                    ax.plot(*xy, marker=marker, color=color, markersize=9, markerfacecolor="none", markeredgewidth=1.4)
                ax.set(xlim=(0, 224), ylim=(224, 0), xticks=[], yticks=[])
                ax.set_title(f"{names[j].replace(chr(10), ' ')} · seed {seed}\n"
                             f"均值{r['expectation_error_px']:.1f}px / 峰值{r['argmax_error_px']:.1f}px / "
                             f"邻域{100*r['mass_within_two_grid_cells']:.1f}%", fontsize=8)
        fig.suptitle(f"#{ui} {view.capitalize()} · 最早有效尖端帧{frame} · 全部为训练点", y=.988, fontsize=12)
        fig.text(.5, .018, "青色＋ 人工尖端；绿色○ 空间期望；橙色× 概率峰值。相同原图、相同25点监督；不是留出预测。",
                 ha="center", fontsize=8)
        fig.subplots_adjust(top=.92, bottom=.08, left=.03, right=.93, hspace=.34, wspace=.12)
        fig.colorbar(im, cax=fig.add_axes((.95, .2, .012, .56)), label="log10(概率 / 均匀概率)")
        stem = f"tip_fit_case_{ui}_{view}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{stem}.{ext}", dpi=450)
        plt.close(fig)
        links.append(f'<h2>#{ui} {view.capitalize()}</h2><img src="figures/{stem}.png">')
    (args.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>纯定位训练拟合诊断</title>'
        '<style>body{max-width:1100px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>纯定位拟合诊断</h1><p>全部为训练点；无泛化或策略收益结论。固定三seed、200/2000步，无最佳模型选择。</p>'
        '<img src="figures/tip_only_fit_zh.png">'+"".join(links), encoding="utf-8")
    print(f"Rendered summary and {len(links)} fixed case sheets; user acceptance pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_tip_only_fit_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (fit if args.stage == "fit" else render)(args)


if __name__ == "__main__":
    main()
