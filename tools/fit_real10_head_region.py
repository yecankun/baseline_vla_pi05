"""Matched-frame, training-only head-region localization feasibility study.

fit runs on the 4090 with the existing frozen cache. render is local and torch-free.
Human geometry is target-only; old models, labels and episode folds stay read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np

SCHEMA = "real10_head_region_training_fit_v1"
SEEDS = (20261020, 20261120, 20261220)
VIEWS = ("side", "top")
STAGES = ("uniform", "spatial_prior", "initial", "tip_only_2000", "head_200", "head_2000")
METRICS = ("support_mass", "peak_distance_px", "peak_in_support",
           "mean_distance_px", "entropy_normalized", "support_area_fraction")
DESIGN = Path("docs/algorithm-real10-head-region-fit-protocol-20260923.md")
CONFIG = {"seeds": list(SEEDS), "steps": 2000, "checkpoints": [200, 2000],
          "lr": .001, "weight_decay": .01, "gradient_clip": 1., "loss_scale": .1,
          "loss": "window-macro negative log support probability / log(56^2)",
          "line_radius_cells": 1., "bbox": "grid centers inside original bbox",
          "selection": "none; final step, all seeds", "split": "unchanged old fold0",
          "encoder": "same frozen ResNet18 layer1 cache",
          "parameters": "original 529 project/location; unused response frozen"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def distance_to_geometry(points, geometry, scale=(1., 1.), offset=(0., 0.)):
    """Exact distance to a box or visible segments in the requested coordinate frame."""
    points = np.asarray(points)
    scale, offset = np.asarray(scale), np.asarray(offset)
    if geometry["kind"] == "bbox":
        corners = (np.asarray(geometry["xyxy"]).reshape(2, 2)-offset)/scale
        return np.linalg.norm(np.maximum(np.maximum(corners[0]-points, points-corners[1]), 0), axis=-1)
    distances = []
    for segment in geometry["segments"]:
        p = (np.asarray(segment)-offset)/scale
        for a, b in zip(p[:-1], p[1:]):
            v = b-a
            t = np.clip(((points-a)*v).sum(-1)/max(float(v @ v), 1e-20), 0, 1)
            distances.append(np.linalg.norm(points-(a+t[..., None]*v), axis=-1))
    if not distances:
        raise ValueError("a visible polyline needs at least two distinct vertices")
    return np.minimum.reduce(distances)


def make_records(args, inputs, rows, folds):
    audit = read_json(args.labels / "report.json")
    manifest = read_json(args.labels / "source_manifest.json")
    cases = {case["id"]: case for case in manifest["cases"]}
    lookup = {row["ui_index"]: i for i, row in enumerate(rows)}
    roi = read_json(args.source / "source/protocol.json")["roi_xyxy_original"]
    coordinate = (np.arange(56)*4+.5)/224
    xx, yy = np.meshgrid(coordinate, coordinate)
    grid = np.stack((xx, yy), -1).reshape(-1, 2)
    records, supports, excluded = [], [], []
    for frame in audit["frames"]:
        i = lookup[frame["ui_index"]]
        case, view, f = cases[frame["case_id"]], frame["view"], frame["frame_index"]
        if rows[i]["source_episode"] != case["source_episode"]:
            raise ValueError("head annotation and feature episode differ")
        if i not in folds[0]["train_indices"] or frame["head"]["geometry"] is None:
            excluded.append({"case_id": case["id"], "frame_index": f, "view": view,
                             "status": frame["head"]["status"],
                             "reason": "old_fold0_heldout" if i not in folds[0]["train_indices"] else "no_geometry"})
            continue
        geometry = frame["head"]["geometry"]
        wh = np.subtract(roi[view][2:], roi[view][:2])
        origin = np.asarray(roi[view][:2])-.5
        distance = distance_to_geometry(grid*56, geometry, wh/56, origin)
        support = distance <= (CONFIG["line_radius_cells"] if geometry["kind"] == "polyline" else 1e-8)
        if not support.any():
            raise ValueError("empty coarse support; inspect mapping instead of changing tolerance")
        image = inputs[i]["model_input"]["sequence"][f]["images"][view]
        if Path(image).name != Path(case["frames"][view][f]["path"]).name:
            raise ValueError("head annotation and cached image frame differ")
        records.append({"sample_index": i, "window_id": rows[i]["window_id"],
                        "ui_index": frame["ui_index"], "source_episode": case["source_episode"],
                        "frame_index": f, "view": view, "image_path": image,
                        "geometry": geometry, "coverage": frame["head"]["coverage"],
                        "status": frame["head"]["status"], "head_revision": frame["head_revision"]})
        supports.append(support)
    old_points = {(r["window_id"], r["frame_index"], r["view"]) for r in
                  read_jsonl(args.source / "source/point_audit.jsonl") if r["tip_target_valid"]}
    if {(r["window_id"], r["frame_index"], r["view"]) for r in records} != old_points or len(records) != 25:
        raise ValueError("expected exactly the same 25 training frames as the old tip-only fit")
    return records, np.stack(supports), excluded, roi, grid


def aggregate(records):
    by_window = defaultdict(list)
    for row in records:
        by_window[row["window_id"]].append(row)
    return {key: float(np.mean([np.mean([r[key] for r in group]) for group in by_window.values()]))
            for key in METRICS}


def fit(args):
    from train_real10_spatial_aux_pair import (
        atomic_torch_save, check_frozen, checkpoint_path, configure, cpu_state,
        seed_all, validate_checkpoint, writer_lock)
    from real10_spatial_response import SpatialResponse
    import torch

    started = time.perf_counter()
    configure()
    inputs, _, rows, folds = check_frozen(args.source)
    records, support, excluded, roi, grid = make_records(args, inputs, rows, folds)
    if args.out.exists():
        if not args.resume:
            raise FileExistsError("preserve output; --resume skips completed seeds")
        protocol = read_json(args.out / "protocol.json")
        if protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix():
            raise ValueError("different fit protocol")
        if (args.out / "entrypoint_snapshot.py").read_bytes() != Path(__file__).read_bytes():
            raise ValueError("fit entrypoint changed; inspect before resuming")
        for name in ("report.json", "source_manifest.json", "source_head_annotations.jsonl"):
            if (args.out / "source" / name).read_bytes() != (args.labels / name).read_bytes():
                raise ValueError("human annotation snapshot changed")
    else:
        args.out.mkdir(parents=True)
        (args.out / "checkpoints").mkdir()
        (args.out / "source").mkdir()
        shutil.copy2(__file__, args.out / "entrypoint_snapshot.py")
        shutil.copy2("tools/real10_spatial_response.py", args.out / "model_snapshot.py")
        shutil.copy2(DESIGN, args.out / "design_protocol.md")
        for name in ("report.json", "source_manifest.json", "source_head_annotations.jsonl"):
            shutil.copy2(args.labels / name, args.out / "source" / name)
        write_json(args.out / "protocol.json", {
            "schema": SCHEMA, "config": CONFIG, "source": args.source.as_posix(),
            "labels": args.labels.as_posix(), "old_tip_fit": args.pure.as_posix(),
            "training_frames": 25, "training_windows": 5, "training_source_episodes": 5,
            "heldout_episode_unchanged": folds[0]["heldout_episode"], "heldout_evaluated": False,
            "roi_xyxy_original": roi, "grid_size": 56, "excluded": excluded,
            "input": "same fixed-ROI image features only; no geometry, visibility or task input",
            "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0),
            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
        write_jsonl(args.out / "training_geometry.jsonl", records)

    with writer_lock(args.out):
        if (args.out / "report.json").is_file():
            print("Already complete; no training performed.", flush=True)
            return
        cache_path = args.source / "feature_cache.pt"
        protected = [cache_path] + [
            path for seed in SEEDS for path in (
                checkpoint_path(args.source, seed, 0, "spatial_window_tip_aux"),
                args.pure / "checkpoints" / f"seed{seed}_step2000.pt")]
        before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in protected}
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        lookup = {p: i for i, p in enumerate(cache["image_paths"])}
        maps = cache["maps"][[lookup[r["image_path"]] for r in records]].cuda()
        del cache
        counts = {w: sum(r["window_id"] == w for r in records) for w in {r["window_id"] for r in records}}
        weights = torch.tensor([1/(len(counts)*counts[r["window_id"]]) for r in records], device="cuda")
        mask = torch.from_numpy(support).cuda()
        log_cells = float(np.log(56**2))
        uniform = np.full(support.shape, 1/(56**2), dtype=np.float32)
        prior = uniform.copy()
        for view in VIEWS:
            by_window = defaultdict(list)
            for j, record in enumerate(records):
                if record["view"] == view:
                    by_window[record["window_id"]].append(support[j]/support[j].sum())
            distribution = np.mean([np.mean(v, axis=0) for v in by_window.values()], axis=0)
            prior[[j for j, r in enumerate(records) if r["view"] == view]] = distribution

        def logits_for(model):
            return model.location(model.project(maps)).flatten(1)

        def loss_for(logits):
            logmass = logits.masked_fill(~mask, -torch.inf).logsumexp(-1)-logits.logsumexp(-1)
            return (-logmass / log_cells * weights).sum()

        def measure(probability, seed, stage):
            result = []
            for j, meta in enumerate(records):
                p = probability[j]
                wh = np.subtract(roi[meta["view"]][2:], roi[meta["view"]][:2])
                origin = np.asarray(roi[meta["view"]][:2])-.5
                peak_idx = int(p.argmax())
                peak, mean = grid[peak_idx]*wh+origin, (p @ grid)*wh+origin
                result.append({**meta, "base_seed": seed, "stage": stage, "phase": "training_only",
                    "support_mass": float(p[support[j]].sum()),
                    "peak_distance_px": float(distance_to_geometry(peak, meta["geometry"])),
                    "mean_distance_px": float(distance_to_geometry(mean, meta["geometry"])),
                    "peak_in_support": float(support[j, peak_idx]),
                    "entropy_normalized": float(-(p*np.log(np.maximum(p, 1e-30))).sum()/log_cells),
                    "support_area_fraction": float(support[j].mean()),
                    "peak_xy_original": peak.tolist(), "mean_xy_original": mean.tolist(),
                    "peak_tie_note": "argmax tie resolves to first grid cell; uniform peak is not meaningful" if stage == "uniform" else None})
            return result

        torch.cuda.reset_peak_memory_stats()
        completed_results = []
        for seed in SEEDS:
            path = args.out / f"seed{seed}_complete.pt"
            if path.is_file():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["config"] != CONFIG:
                    raise ValueError("completed seed uses a different config")
                completed_results.append(saved)
                print(f"seed={seed}: reused completed fit", flush=True)
                continue
            tick = time.perf_counter()
            old = torch.load(checkpoint_path(args.source, seed, 0, "spatial_window_tip_aux"),
                             map_location="cpu", weights_only=True)
            validate_checkpoint(old, seed, 0, folds[0], "spatial_window_tip_aux")
            seed_all(seed)
            model = SpatialResponse().cuda().eval()
            model.load_state_dict(old["initial_state"])
            model.response.requires_grad_(False)
            trainable = [p for p in model.parameters() if p.requires_grad]
            assert sum(p.numel() for p in trainable) == 529
            optimizer = torch.optim.AdamW(trainable, lr=.001, weight_decay=.01)
            arrays = {"uniform": torch.from_numpy(uniform), "spatial_prior": torch.from_numpy(prior)}
            with torch.no_grad():
                arrays["initial"] = logits_for(model).softmax(-1).cpu()
                initial_loss = float(loss_for(logits_for(model)))
                model.load_state_dict(torch.load(args.pure / "checkpoints" / f"seed{seed}_step2000.pt",
                                                map_location="cpu", weights_only=True)["model_state"])
                arrays["tip_only_2000"] = logits_for(model).softmax(-1).cpu()
                model.load_state_dict(old["initial_state"])
            curves = [{"step": 0, "support_nll_normalized": initial_loss}]
            replay_differences = {}
            for step in range(1, 2001):
                optimizer.zero_grad(set_to_none=True)
                loss = .1 * loss_for(logits_for(model))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
                optimizer.step()
                if step % 100 == 0:
                    with torch.no_grad():
                        curves.append({"step": step, "support_nll_normalized": float(loss_for(logits_for(model)))})
                if step in (200, 2000):
                    with torch.no_grad():
                        arrays[f"head_{step}"] = logits_for(model).softmax(-1).cpu()
                    checkpoint = args.out / "checkpoints" / f"seed{seed}_step{step:04d}.pt"
                    atomic_torch_save({"schema": SCHEMA, "base_seed": seed, "optimizer_steps": step,
                        "config": CONFIG, "model_state": cpu_state(model),
                        "policy_input_allowed": False, "deployable": False}, checkpoint)
                    replay = SpatialResponse().cuda().eval()
                    replay.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"])
                    with torch.no_grad():
                        difference = float((logits_for(model)-logits_for(replay)).abs().max())
                    if difference > 1e-6:
                        raise ValueError("checkpoint replay changed")
                    replay_differences[str(step)] = difference
                    del replay
                    print(f"seed={seed} step={step} loss={curves[-1]['support_nll_normalized']:.6f} elapsed={time.perf_counter()-tick:.1f}s", flush=True)
            if any(not torch.equal(v.cpu(), old["initial_state"][k]) for k, v in model.state_dict().items()
                   if k.startswith("response.")):
                raise ValueError("unused response parameters changed")
            saved = {"seed": seed, "config": CONFIG, "probabilities": arrays, "curves": curves,
                     "checks": {"initialization_from_old_fold0": True, "trainable_parameters": 529,
                                "response_parameters_unchanged": True, "replay_max_abs": replay_differences},
                     "elapsed_s": time.perf_counter()-tick}
            atomic_torch_save(saved, path)
            completed_results.append(saved)
            del optimizer, model
        if any(before[str(p)] != (p.stat().st_size, p.stat().st_mtime_ns) for p in protected):
            raise ValueError("protected old cache/checkpoint changed")
        diagnostics, heatmaps, curves, checks = [], {}, [], []
        for saved in completed_results:
            seed = saved["seed"]
            curves.extend({"base_seed": seed, **r} for r in saved["curves"])
            checks.append({"base_seed": seed, "elapsed_s": saved["elapsed_s"], **saved["checks"]})
            for stage in STAGES:
                p = saved["probabilities"][stage].numpy()
                for j, row in enumerate(measure(p, seed, stage)):
                    key = f"map_{len(heatmaps):03d}"
                    diagnostics.append({**row, "array_key": key})
                    heatmaps[key] = p[j].reshape(56, 56)
        summaries = []
        for seed in SEEDS:
            for stage in STAGES:
                for view in VIEWS:
                    selected = [r for r in diagnostics if (r["base_seed"], r["stage"], r["view"]) == (seed, stage, view)]
                    summaries.append({"base_seed": seed, "stage": stage, "view": view,
                                      "frames": len(selected), "metrics": aggregate(selected)})
        means = [{"stage": stage, "view": view,
                  "metrics": {m: float(np.mean([r["metrics"][m] for r in summaries if
                     (r["stage"], r["view"]) == (stage, view)])) for m in METRICS},
                  "std": {m: float(np.std([r["metrics"][m] for r in summaries if
                     (r["stage"], r["view"]) == (stage, view)])) for m in METRICS}}
                 for stage in STAGES for view in VIEWS]
        strata = []
        for field, values in (("coverage", ("complete", "partial")), ("kind", ("polyline", "bbox"))):
            for value in values:
                for seed in SEEDS:
                    for stage in STAGES:
                        group = [r for r in diagnostics if r["base_seed"] == seed and r["stage"] == stage
                                 and (r["geometry"]["kind"] if field == "kind" else r[field]) == value]
                        strata.append({"field": field, "value": value, "base_seed": seed,
                                       "stage": stage, "frames": len(group), "metrics": aggregate(group)})
        write_jsonl(args.out / "frame_diagnostics.jsonl", diagnostics)
        write_jsonl(args.out / "loss_curves.jsonl", curves)
        write_jsonl(args.out / "model_readback.jsonl", checks)
        np.savez_compressed(args.out / "heatmaps.npz", **heatmaps)
        write_json(args.out / "report.json", {"status": "complete_training_only_fit", "config": CONFIG,
            "scope": "same 25 TRAIN frames; no heldout prediction, response metric, segmentation or policy claim",
            "training_windows": 5, "training_frames": 25, "optimizer_steps_total": 6000,
            "old_source_stats_unchanged": True, "seed_summaries": summaries, "means": means,
            "strata": strata, "elapsed_s": time.perf_counter()-started,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
        print(json.dumps({"status": "complete", "elapsed_s": time.perf_counter()-started,
                          "means": [r for r in means if r["stage"] == "head_2000"]}), flush=True)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    from PIL import Image

    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight", "svg.fonttype": "path"})
    report, protocol = (read_json(args.out / f"{name}.json") for name in ("report", "protocol"))
    rows = read_jsonl(args.out / "frame_diagnostics.jsonl")
    curves = read_jsonl(args.out / "loss_curves.jsonl")
    maps = np.load(args.out / "heatmaps.npz")
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    shutil.copy2(__file__, args.out / "render_snapshot.py")
    (figures / "data-manifest.md").write_text(
        "# 头段弱定位：真实训练拟合数据\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n"
        "|---|---|---|---|---|---|\n"
        "| 支持域概率及距离 | ../report.json, ../loss_curves.jsonl | real | 同25训练帧、全部3seed | "
        "tools/fit_real10_head_region.py --stage render | head_region_summary_zh.png/.svg |\n"
        "| 全部训练帧叠加 | ../frame_diagnostics.jsonl, ../heatmaps.npz | real | 原ROI224、固定最小seed | "
        "同上 | head_region_case_*_zh.png/.svg |\n\n"
        "几何仅是人工粗范围，线带宽1格是离散化选择；不是像素分割真值。曲线是训练损失。\n"
        "Side/Top先帧到窗再跨seed平均；全帧/全seed数字保存，图版固定20261020不挑最好seed。\n"
        "不看图像的位置先验由训练支持域构造；#4仍留出，未计算预测或选择模型。\n", encoding="utf-8")
    stages = ("spatial_prior", "initial", "tip_only_2000", "head_2000")
    names = ("固定位置\n先验", "相同\n初始化", "旧尖端\n2000步", "头段弱定位\n2000步")
    lookup = {(r["base_seed"], r["stage"], r["view"]): r["metrics"] for r in report["seed_summaries"]}
    colors = ("#0077BB", "#EE7733", "#009988")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for v, view in enumerate(VIEWS):
        for k, seed in enumerate(SEEDS):
            axes[0, v].plot(range(4), [lookup[seed, s, view]["support_mass"]*100 for s in stages],
                            "o-", color=colors[k], lw=1.3, ms=4, label=f"seed {seed}")
            axes[1, v].plot(range(4), [lookup[seed, s, view]["peak_distance_px"] for s in stages],
                            "o-", color=colors[k], lw=1.3, ms=4)
        axes[0, v].axhline(lookup[SEEDS[0], "uniform", view]["support_mass"]*100,
                           color="#888888", ls="--", lw=1, label="均匀概率质量")
        axes[0, v].set(title=f"{view.capitalize()}：人工支持域概率", ylabel="概率质量（%，窗平均）",
                       xticks=range(4), xticklabels=names, ylim=(0, 105))
        axes[1, v].set(title=f"{view.capitalize()}：峰值到粗几何的距离", ylabel="原图像素（px，窗平均）",
                       xticks=range(4), xticklabels=names, ylim=(0, None))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94),
               ncol=4, frameon=False, fontsize=8)
    fig.suptitle("头段弱定位最小实验：同25个训练帧，不是泛化或推进识别结果", y=.99, fontsize=12)
    fig.text(.5, .014, "支持域是人工短线邻域或粗框，不是精确分割；旧尖端和新头段训练目标不同。", ha="center", fontsize=8)
    fig.subplots_adjust(top=.84, bottom=.11, hspace=.6, wspace=.26)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"head_region_summary_zh.{ext}", dpi=450)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3))
    for k, seed in enumerate(SEEDS):
        selected = [r for r in curves if r["base_seed"] == seed]
        ax.plot([r["step"] for r in selected], [r["support_nll_normalized"] for r in selected],
                color=colors[k], label=f"seed {seed}")
    ax.set(title="固定预算的训练支持域负对数概率", xlabel="优化步数", ylabel="归一化损失（未乘0.1）",
           ylim=(0, None))
    ax.legend(frameon=False, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"head_region_loss_zh.{ext}", dpi=450)
    plt.close(fig)
    links = []
    for ui in sorted({r["ui_index"] for r in rows}):
        shown = sorted([r for r in rows if r["ui_index"] == ui and r["base_seed"] == SEEDS[0]
                        and r["stage"] == "head_2000"], key=lambda r: (r["view"], r["frame_index"]))
        fig, axes = plt.subplots(len(shown), 3, figsize=(9, 2.35*len(shown)), squeeze=False)
        for i, row in enumerate(shown):
            match = {(r["stage"]): r for r in rows if (r["ui_index"], r["frame_index"], r["view"], r["base_seed"]) ==
                     (ui, row["frame_index"], row["view"], SEEDS[0])}
            roi = protocol["roi_xyxy_original"][row["view"]]
            wh, origin = np.subtract(roi[2:], roi[:2]), np.asarray(roi[:2])-.5
            for col, stage in enumerate(("raw", "tip_only_2000", "head_2000")):
                ax = axes[i, col]
                with Image.open(row["image_path"]) as image:
                    ax.imshow(image.convert("RGB"), extent=(0, 224, 224, 0))
                title = f"{row['view'].capitalize()} 帧{row['frame_index']} · {row['coverage']}"
                if stage != "raw":
                    current = match[stage]
                    im = ax.imshow(np.log10(np.maximum(maps[current["array_key"]]*3136, 1e-10)),
                                   extent=(-1.5, 222.5, 222.5, -1.5), cmap="magma",
                                   vmin=-1, vmax=2, alpha=.48, interpolation="nearest")
                    xy = (np.array(current["peak_xy_original"])-origin)/wh*224
                    ax.plot(*xy, "x", color="#EE7733", ms=8, mew=1.5)
                    title = ("旧尖端2000" if stage == "tip_only_2000" else "新头段2000") + (
                        f" · 域内{100*current['support_mass']:.1f}%\n峰值距离{current['peak_distance_px']:.1f}px")
                geom = row["geometry"]
                if geom["kind"] == "bbox":
                    box = (np.array(geom["xyxy"]).reshape(2, 2)-origin)/wh*224
                    ax.add_patch(Rectangle(box[0], *(box[1]-box[0]), fill=False, edgecolor="#33BBEE", lw=1.3))
                else:
                    for segment in geom["segments"]:
                        xy = (np.array(segment)-origin)/wh*224
                        ax.plot(xy[:, 0], xy[:, 1], color="#33BBEE", lw=1.4)
                ax.set(title=title, xlim=(0, 224), ylim=(224, 0), xticks=[], yticks=[])
        fig.suptitle(f"#{ui} · 全部有监督训练帧 · 固定seed {SEEDS[0]}", y=.997, fontsize=12)
        fig.text(.5, .008, "青色：人工粗几何；橙色×：概率峰值。热图log10(概率/均匀概率)，显示范围[-1,2]。\n"
                 "只有训练拟合证据，不是推进判断或分割结果。",
                 ha="center", fontsize=8)
        fig.subplots_adjust(top=1-.85/(2.35*len(shown)), bottom=.065, hspace=.4, wspace=.14)
        stem = f"head_region_case_{ui:03d}_zh"
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{stem}.{ext}", dpi=450)
        plt.close(fig)
        links.append(f'<h2>#{ui} · 同帧定位对照</h2><img src="figures/{stem}.png">')
    (args.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>头段弱定位训练拟合</title>'
        '<style>body{max-width:1100px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>头段弱定位最小实验</h1><p>同25个训练帧，#4仍留出。无泛化、响应或策略收益结论。'
        '不看图像的位置先验为简单参照，不是最优基线；图版固定最小seed，不挑最好结果。</p>'
        '<img src="figures/head_region_summary_zh.png"><img src="figures/head_region_loss_zh.png">'
        + "".join(links), encoding="utf-8")
    print(f"Rendered summary, curve and {len(links)} all-frame case sheets; visual acceptance pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_spatial_aux_pair_v1"))
    parser.add_argument("--labels", type=Path, default=Path("simulation_output/real10_head_segment_annotation_audit_v1"))
    parser.add_argument("--pure", type=Path, default=Path("simulation_output/real10_tip_only_fit_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_head_region_fit_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    (fit if args.stage == "fit" else render)(args)


if __name__ == "__main__":
    main()
