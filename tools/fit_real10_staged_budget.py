"""Fixed 1000 tip-only + 1000 joint updates, preserving the shared Adam state.

Training-only diagnosis on original fold0; no architecture or dataset changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from diagnose_real10_spatial_aux_fit import (
    METRICS, balanced_point_summary, read_json, read_jsonl, write_json, write_jsonl)


SEEDS = (20261020, 20261120, 20261220)
VIEWS = ("side", "top")
STAGES = ("joint_2000", "warmup_1000", "staged_2000", "tip_only_2000")
SCHEMA = "real10_staged_budget_training_fit_v1"
CONFIG = {"seeds": list(SEEDS), "fold_index": 0, "warmup_steps": 1000, "joint_steps": 1000,
          "optimizer": "single AdamW, shared state retained", "lr": .001, "weight_decay": .01,
          "clip": 1., "localization_weight": .1, "precision": "FP32 deterministic, no AMP/TF32",
          "classification_updates": 1000, "selection": "fixed final step, all seeds"}
DESIGN = Path("docs/algorithm-real10-staged-budget-protocol-20260922.md")


def fit(args):
    from train_real10_spatial_aux_pair import (
        atomic_torch_save, check_frozen, checkpoint_path, configure, cpu_state, forward,
        load_tensors, seed_all, subset, validate_checkpoint, writer_lock)
    from real10_spatial_response import SpatialResponse, point_distribution, response_losses, spatial_grid
    from fit_real10_joint_budget import CONFIG as JOINT_CONFIG
    from fit_real10_tip_only import CONFIG as PURE_CONFIG
    import torch

    started = time.perf_counter()
    configure()
    inputs, targets, _, folds = check_frozen(args.source)
    for directory, expected, filename in ((args.pure, PURE_CONFIG, "fit_real10_tip_only.py"),
                                           (args.joint, JOINT_CONFIG, "fit_real10_joint_budget.py")):
        if read_json(directory / "protocol.json")["config"] != expected or not (directory / "report.json").is_file():
            raise ValueError("completed fixed comparison required")
        if (directory / "entrypoint_snapshot.py").read_bytes() != (Path("tools") / filename).read_bytes():
            raise ValueError("comparison entrypoint differs from executed snapshot")
    if args.out.exists():
        old_protocol = read_json(args.out / "protocol.json")
        if (not args.resume or old_protocol["config"] != CONFIG or
                any(old_protocol[k] != getattr(args, k).as_posix() for k in ("source", "pure", "joint")) or
                (args.out / "entrypoint_snapshot.py").read_bytes() != Path(__file__).read_bytes()):
            raise FileExistsError("preserve output; resume only this same fixed version")
    else:
        args.out.mkdir(parents=True)
        (args.out / "checkpoints").mkdir()
        shutil.copy2(__file__, args.out / "entrypoint_snapshot.py")
        shutil.copy2(DESIGN, args.out / "design_protocol.md")
        write_json(args.out / "protocol.json", {"schema": SCHEMA, "config": CONFIG,
            **{k: getattr(args, k).as_posix() for k in ("source", "pure", "joint")},
            "roi_xyxy_original": read_json(args.pure / "protocol.json")["roi_xyxy_original"],
            "new_optimizer_steps": 6000, "classification_exposure_not_matched": True,
            "timing_estimate_s": 120, "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "policy_input_allowed": False, "deployable": False})
    with writer_lock(args.out):
        if (args.out / "report.json").is_file():
            print("Already complete; no new training.", flush=True)
            return
        old_points = read_jsonl(args.joint / "point_diagnostics.jsonl")
        pure_curve = {(r["base_seed"], r["step"]): r["localization_ce"]
                      for r in read_jsonl(args.pure / "loss_curves.jsonl")}
        tensors = load_tensors(args.source, inputs, targets)
        train = subset(tensors, folds[0]["train_indices"])
        del tensors
        mask = train["tip_valid"]
        positions = mask.nonzero().cpu().tolist()
        if len(train["response_y"]) != 36 or len(positions) != 25:
            raise ValueError("original 36 training windows and 25 tip points required")
        meta_lookup = {(r["sample_index"], r["frame_index"], r["view"]): r for r in old_points
                       if r["base_seed"] == SEEDS[0] and r["stage"] == "joint_2000"}
        meta = [meta_lookup[folds[0]["train_indices"][i], frame, VIEWS[v]] for i, frame, v in positions]
        earliest = {}
        for r in meta:
            key = r["window_id"], r["view"]
            earliest[key] = min(r["frame_index"], earliest.get(key, r["frame_index"]))
        point_maps, target = train["maps"][mask], train["xy_uv"][mask]
        windows = sorted({p[0] for p in positions})
        counts = {w: sum(p[0] == w for p in positions) for w in windows}
        weights = torch.tensor([1/(len(windows)*counts[p[0]]) for p in positions], device="cuda")
        q = point_distribution(target, torch.ones(25, device="cuda", dtype=torch.bool)).flatten(1)
        grid = spatial_grid("cuda", torch.float32).reshape(-1, 2)
        roi = read_json(args.out / "protocol.json")["roi_xyxy_original"]
        scale = torch.tensor([[roi[VIEWS[v]][2]-roi[VIEWS[v]][0], roi[VIEWS[v]][3]-roi[VIEWS[v]][1]]
                              for _, _, v in positions], device="cuda")
        offset = torch.tensor([[roi[VIEWS[v]][0]-.5, roi[VIEWS[v]][1]-.5] for _, _, v in positions], device="cuda")
        log_cells = float(np.log(56**2))
        all_train = torch.ones(36, device="cuda", dtype=torch.bool)
        pos_weight = (36-train["response_y"].sum())/train["response_y"].sum()

        def point_logits(model):
            return model.location(model.project(point_maps)).flatten(1)

        def localization(logits):
            return (-(q*logits.log_softmax(-1)).sum(1)/log_cells*weights).sum()

        def joint_losses(model):
            return response_losses(forward(model, train), train["response_y"], train["xy_uv"], mask,
                                   all_train, pos_weight, "spatial_window_tip_aux")

        @torch.no_grad()
        def measure(model, seed, stage):
            logits = point_logits(model)
            full = forward(model, train)
            logp = logits.log_softmax(-1)
            p = logp.exp()
            xy, mode = p @ grid, grid[logp.argmax(-1)]
            checks = {"point_vs_full_logits_max_abs": float((logits-full["heatmap_logits"][mask].flatten(1)).abs().max()),
                      "point_vs_full_location_uv_max_abs": float((xy-full["location_uv"][mask]).abs().max())}
            if checks["point_vs_full_logits_max_abs"] > 1e-5 or checks["point_vs_full_location_uv_max_abs"] > 2e-6:
                raise ValueError("point diagnostic changed the spatial operation")
            entropy = -(p*logp).sum(1)
            region = (((grid[None]-target[:, None])*56).square().sum(-1) <= 4)
            values = {
                "expectation_error_px": ((xy-target)*scale).norm(dim=1),
                "argmax_error_px": ((mode-target)*scale).norm(dim=1),
                "uniform_expectation_error_px": ((grid.mean(0)-target)*scale).norm(dim=1),
                "expectation_to_argmax_px": ((xy-mode)*scale).norm(dim=1),
                "entropy_normalized": entropy/log_cells, "effective_cell_fraction": entropy.exp()/len(grid),
                "peak_over_uniform": p.max(1).values*len(grid),
                "mass_within_two_grid_cells": (p*region).sum(1), "uniform_mass_same_region": region.float().mean(1),
                "target_mass_same_region": (q*region).sum(1), "target_ce_normalized": -(q*logp).sum(1)/log_cells,
                "target_entropy_normalized": -(q*q.clamp_min(1e-30).log()).sum(1)/log_cells,
                "spatial_rms_radius_px": (p*((grid[None]-xy[:, None])*scale[:, None]).square().sum(-1)).sum(1).sqrt()}
            values = {k: v.cpu().tolist() for k, v in values.items()}
            expected, argmax = (xy*scale+offset).cpu().tolist(), (mode*scale+offset).cpu().tolist()
            records, maps = [], {}
            for j, r in enumerate(meta):
                key = f"seed{seed}_{stage}_point{j:02d}" if r["frame_index"] == earliest[r["window_id"], r["view"]] else None
                records.append({**r, "base_seed": seed, "stage": stage, "array_key": key, "heatmap_source": "staged",
                    "expectation_xy_original": expected[j], "argmax_xy_original": argmax[j],
                    **{k: values[k][j] for k in METRICS}})
                if key:
                    maps[key] = p[j].cpu().reshape(56, 56)
            return records, maps, checks

        results = []
        torch.cuda.reset_peak_memory_stats()
        for seed in SEEDS:
            completed = args.out / f"seed{seed}_complete.pt"
            if completed.is_file():
                result = torch.load(completed, map_location="cpu", weights_only=True)
                if result["config"] != CONFIG or result["seed"] != seed:
                    raise ValueError("completed seed protocol mismatch")
                results.append(result)
                continue
            seed_started = time.perf_counter()
            path = checkpoint_path(args.source, seed, 0, "spatial_window_tip_aux")
            protected = (path, args.joint / "report.json", args.joint / "point_diagnostics.jsonl",
                         args.pure / "report.json", args.pure / "loss_curves.jsonl")
            stats = [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]
            old = torch.load(path, map_location="cpu", weights_only=True)
            validate_checkpoint(old, seed, 0, folds[0], "spatial_window_tip_aux")
            seed_all(seed)
            model = SpatialResponse().cuda().eval()
            model.load_state_dict(old["initial_state"])
            if any(not torch.equal(v.cpu(), old["initial_state"][k]) for k, v in model.state_dict().items()):
                raise ValueError("initialization mismatch")
            model.response.requires_grad_(False)
            # One parameter group for both phases. Frozen parameters have no grad/state/decay.
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            shared = [p for name, p in model.named_parameters() if not name.startswith("response.")]
            head = list(model.response.parameters())
            checks, curve, records, maps = {"initial_state_matches": True}, [], [], {}

            def log_losses(step):
                with torch.no_grad():
                    losses = {k: float(v) for k, v in joint_losses(model).items() if k in ("total", "response", "localization")}
                curve.append({"base_seed": seed, "step": step, **losses,
                              "active_objective": .1*losses["localization"] if step <= 1000 else losses["total"],
                              "classification_optimized": step > 1000})
                return losses

            log_losses(0)
            for step in range(1, 2001):
                if step == 1001:
                    before = {id(p): {k: v.clone() for k, v in optimizer.state[p].items()} for p in shared}
                    model.response.requires_grad_(True)
                    unchanged = all(torch.equal(v, optimizer.state[p][k]) for p in shared for k, v in before[id(p)].items())
                    if not unchanged:
                        raise ValueError("shared Adam state reset at phase switch")
                    checks["shared_adam_state_unchanged_at_switch"] = True
                optimizer.zero_grad(set_to_none=True)
                loss = .1*localization(point_logits(model)) if step <= 1000 else joint_losses(model)["total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if step % 100 == 0 or step == 1001:
                    losses = log_losses(step)
                if step in (1000, 2000):
                    shared_steps = sorted({int(optimizer.state[p]["step"]) for p in shared})
                    head_steps = sorted({int(optimizer.state[p]["step"]) for p in head if p in optimizer.state})
                    if shared_steps != [step] or head_steps != ([] if step == 1000 else [1000]):
                        raise ValueError("incorrect optimizer exposure or state continuity")
                    checks[f"step{step}_adam_counts"] = {"shared": shared_steps, "head": head_steps}
                    if step == 1000:
                        if any(not torch.equal(v.cpu(), old["initial_state"][k]) for k, v in model.state_dict().items()
                               if k.startswith("response.")):
                            raise ValueError("frozen response parameters changed during warmup")
                        delta = abs(float(localization(point_logits(model)).detach())-pure_curve[seed, 1000])
                        if delta > 1e-6:
                            raise ValueError("warmup differs from original pure localization prefix")
                        checks.update({"warmup_response_parameters_unchanged": True, "warmup_vs_pure1000_ce_abs": delta})
                    stage = "warmup_1000" if step == 1000 else "staged_2000"
                    rr, mm, cc = measure(model, seed, stage)
                    records.extend(rr)
                    maps.update(mm)
                    checks[stage] = cc
                    checkpoint = args.out / "checkpoints" / f"seed{seed}_step{step:04d}.pt"
                    atomic_torch_save({"schema": SCHEMA, "config": CONFIG, "base_seed": seed, "step": step,
                        "model_state": cpu_state(model), "optimizer_state": optimizer.state_dict(), "training_losses": losses,
                        "policy_input_allowed": False, "deployable": False}, checkpoint)
                    replay = SpatialResponse().cuda().eval()
                    replay.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"])
                    with torch.no_grad():
                        a, b = forward(model, train), forward(replay, train)
                        replay_diff = {k: float((a[k]-b[k]).abs().max()) for k in ("window_logit", "heatmap_logits", "location_uv")}
                    if max(replay_diff.values()) > 1e-6:
                        raise ValueError("saved replay mismatch")
                    checks[f"step{step}_saved_replay_max_abs"] = replay_diff
                    del replay, a, b
                    print(f"seed={seed} step={step} BCE={losses['response']:.6f} tipCE={losses['localization']:.6f}", flush=True)
            if stats != [(p.stat().st_size, p.stat().st_mtime_ns) for p in protected]:
                raise ValueError("source artifacts changed")
            checks["protected_source_stats_unchanged"] = True
            result = {"seed": seed, "config": CONFIG, "records": records, "maps": maps, "curve": curve,
                      "checks": checks, "elapsed_s": time.perf_counter()-seed_started}
            atomic_torch_save(result, completed)
            results.append(result)
            del model, optimizer
        points = [r for r in old_points if r["stage"] in ("joint_2000", "tip_only_2000")]
        maps, curve, checks = {}, [], []
        for result in results:
            points.extend(result["records"])
            maps.update({k: v.numpy() for k, v in result["maps"].items()})
            curve.extend(result["curve"])
            checks.append({"base_seed": result["seed"], "elapsed_s": result["elapsed_s"], **result["checks"]})
        summaries = [{"base_seed": seed, "stage": stage, "view": view,
                      **balanced_point_summary([r for r in points if (r["base_seed"], r["stage"], r["view"])
                                                == (seed, stage, view)])}
                     for seed in SEEDS for stage in STAGES for view in VIEWS]
        means = [{"stage": stage, "view": view, "metrics": {m: float(np.mean([r["metrics"][m] for r in summaries
                 if (r["stage"], r["view"]) == (stage, view)])) for m in METRICS}} for stage in STAGES for view in VIEWS]
        if len(points) != 300 or len(maps) != 54 or len(curve) != 66:
            raise ValueError("expected 300 comparison rows, 54 new maps and 66 loss rows")
        write_jsonl(args.out / "point_diagnostics.jsonl", points)
        write_jsonl(args.out / "loss_curves.jsonl", curve)
        write_jsonl(args.out / "model_readback.jsonl", checks)
        np.savez_compressed(args.out / "heatmaps.npz", **maps)
        write_json(args.out / "report.json", {"status": "complete_staged_budget_training_diagnostic", "config": CONFIG,
            "distinct_points": 25, "point_windows": 5, "new_point_evaluations": 150, "reused_control_rows": 150,
            "seed_summaries": summaries, "means_across_seeds": means, "new_optimizer_steps": 6000,
            "classification_updates_per_seed": 1000, "fit_s_total": sum(r["elapsed_s"] for r in results),
            "invocation_elapsed_s": time.perf_counter()-started, "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "generalization_evaluated": False, "response_ba_evaluated": False,
            "policy_input_allowed": False, "deployable": False, "visual_status": "not_viewed"})
        print(f"Complete: fixed 1000+1000, three seeds; {time.perf_counter()-started:.2f}s", flush=True)


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
    report, protocol = (read_json(args.out / f"{n}.json") for n in ("report", "protocol"))
    points, curve = (read_jsonl(args.out / f"{n}.jsonl") for n in ("point_diagnostics", "loss_curves"))
    maps = {k: np.load(v / "heatmaps.npz") for k, v in (("staged", args.out), ("joint", args.joint), ("pure", args.pure))}
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 分阶段训练对照：真实实验数据\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n"
        "| 固定2000步与阶段曲线 | ../report.json, ../loss_curves.jsonl | real | 原25tip点，全3seed，fold0训练侧 | "
        "tools/fit_real10_staged_budget.py --stage render | staged_budget_fit_zh.png/.svg |\n"
        "| 九组最早有效帧热图 | ../point_diagnostics.jsonl, 三组heatmaps.npz | real | 原ROI224与固定checkpoint | "
        "同上 | staged_case_*_zh.png/.svg |\n\n"
        "全为训练点，不证明泛化或策略收益；同总步数不等于同分类更新数（联合2000，分阶段1000，纯定位0）。\n"
        "相机内逐窗宏平均，全部3seed；每窗/相机最早有效tip，无最佳选择。阶段切换固定1000，不早停。\n"
        "px为原图单位，不换算mm；argmax仅作诊断。颜色log10(p/均匀概率)，统一[-1,2]显示截断。\n"
        f"旧图复用 {args.joint.as_posix()} 和 {args.pure.as_posix()}，不重新推理。\n", encoding="utf-8")
    colors = ("#0077BB", "#EE7733", "#009988")
    final_stages = ("joint_2000", "staged_2000", "tip_only_2000")
    names = {"joint_2000": "直接联合\n2000步", "staged_2000": "定位→联合\n1000+1000", "tip_only_2000": "纯定位\n2000步",
             "warmup_1000": "定位预热\n1000步"}
    lookup = {(r["base_seed"], r["stage"], r["view"]): r["metrics"] for r in report["seed_summaries"]}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    for v, view in enumerate(VIEWS):
        for i, seed in enumerate(SEEDS):
            axes[0, v].plot(range(3), [lookup[seed, st, view]["expectation_error_px"] for st in final_stages],
                            "o-", color=colors[i], lw=1.2, ms=4, label=f"seed {seed}")
            axes[1, 1].plot(range(3), [100*lookup[seed, st, view]["mass_within_two_grid_cells"] for st in final_stages],
                            "o-" if v == 0 else "s--", color=colors[i], lw=1.2, ms=3)
        axes[0, v].set(xticks=range(3), xticklabels=[names[s] for s in final_stages],
                       title=f"{view.capitalize()}：原空间均值定位误差", ylabel="原图px（窗平均）", ylim=(0, None))
    for i, seed in enumerate(SEEDS):
        rs = [r for r in curve if r["base_seed"] == seed]
        axes[1, 0].plot([r["step"] for r in rs], [r["localization"] for r in rs], color=colors[i], lw=1.3)
    axes[1, 0].axvline(1000, color="#888888", ls="--", lw=1)
    axes[1, 0].set(xlabel="总更新步数（1000步后加入分类）", ylabel="归一化定位CE（未乘0.1）", title="固定阶段切换后的定位变化", ylim=(0, None))
    axes[1, 1].set(xticks=range(3), xticklabels=[names[s] for s in final_stages], ylabel="目标两格邻域概率质量（%）",
                   title="固定2000步：分布集中程度", ylim=(0, 100))
    axes[1, 1].legend(handles=[Line2D([], [], color="#555555", marker="o", label="Side"),
                             Line2D([], [], color="#555555", marker="s", ls="--", label="Top")], fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.suptitle("固定总2000步：定位预热＋延后分类，能否保住定位拟合？", y=.985, fontsize=13)
    fig.text(.5, .018, "同25个训练点 / 全3seed；无泛化结论。分类头更新数：直接联合2000，分阶段1000，纯定位0。", ha="center", fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.12, left=.09, right=.96, hspace=.57, wspace=.28)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"staged_budget_fit_zh.{ext}", dpi=450)
    plt.close(fig)
    earliest = {}
    for r in points:
        key = r["ui_index"], r["view"]
        earliest[key] = min(r["frame_index"], earliest.get(key, r["frame_index"]))
    links = []
    for (ui, view), frame in sorted(earliest.items()):
        index = {(r["base_seed"], r["stage"]): r for r in points
                 if r["ui_index"] == ui and r["view"] == view and r["frame_index"] == frame}
        fig, axes = plt.subplots(3, 4, figsize=(12.2, 9.3))
        roi = protocol["roi_xyxy_original"][view]
        wh, origin = np.array(roi[2:])-np.array(roi[:2]), np.array(roi[:2])-.5
        for i, seed in enumerate(SEEDS):
            for j, stage in enumerate(STAGES):
                r, ax = index[seed, stage], axes[i, j]
                with Image.open(r["image_path"]) as img:
                    ax.imshow(img.convert("RGB"), extent=(0, 224, 224, 0))
                heat = np.log10(np.maximum(maps[r["heatmap_source"]][r["array_key"]]*56**2, 1e-10))
                im = ax.imshow(heat, extent=(-1.5, 222.5, 222.5, -1.5), cmap="magma", vmin=-1, vmax=2, alpha=.48, interpolation="nearest")
                for key, marker, color in (("target_xy_original", "+", "#33BBEE"), ("expectation_xy_original", "o", "#009988"),
                                            ("argmax_xy_original", "x", "#EE7733")):
                    ax.plot(*((np.array(r[key])-origin)/wh*224), marker=marker, color=color, markersize=9, markerfacecolor="none", markeredgewidth=1.4)
                ax.set(xlim=(0, 224), ylim=(224, 0), xticks=[], yticks=[])
                ax.set_title(f"{names[stage].replace(chr(10), ' ')} · {seed}\n均值{r['expectation_error_px']:.1f}px / 峰值{r['argmax_error_px']:.1f}px\n邻域质量{100*r['mass_within_two_grid_cells']:.1f}%", fontsize=8)
        fig.suptitle(f"#{ui} {view.capitalize()} · 最早有效tip帧{frame} · 预热与联合后对照（全部训练点）", y=.99, fontsize=12)
        fig.text(.5, .018, "青色＋ 人工尖端；绿色○ 空间期望；橙色× 概率峰值。预热1000步为阶段读数，不作为最终模型选择。", ha="center", fontsize=9)
        fig.subplots_adjust(top=.90, bottom=.08, left=.025, right=.945, hspace=.43, wspace=.13)
        fig.colorbar(im, cax=fig.add_axes((.96, .2, .009, .55)), label="log10(概率 / 均匀概率)")
        stem = f"staged_case_{ui}_{view}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{stem}.{ext}", dpi=450)
        plt.close(fig)
        links.append(f'<h2>#{ui} {view.capitalize()}</h2><img src="figures/{stem}.png">')
    (args.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>分阶段训练内拟合对照</title>'
        '<style>body{max-width:1300px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>固定1000定位＋1000联合</h1><p>全部为训练点；分类更新量不相同，不是纯顺序因果实验。无泛化或策略收益结论。</p>'
        '<img src="figures/staged_budget_fit_zh.png">'+"".join(links), encoding="utf-8")
    print(f"Rendered summary and {len(links)} fixed case sheets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--pure", type=Path, default=Path("simulation_output/real10_tip_only_fit_v1"))
    parser.add_argument("--joint", type=Path, default=Path("simulation_output/real10_joint_budget_fit_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_staged_budget_fit_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (fit if args.stage == "fit" else render)(args)


if __name__ == "__main__":
    main()
