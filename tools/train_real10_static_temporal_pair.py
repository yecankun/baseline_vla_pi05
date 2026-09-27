"""Train a first-frame-only reference matched to the frozen regional experiment.

train: remote CUDA, 60 fixed static fits. render: local saved results, no Torch.
The original labels describe observed windows, not synthetic static trajectories.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

from fit_real10_head_region import read_json, read_jsonl, write_json, write_jsonl
from train_real10_head_response_pair import metrics

SCHEMA = "real10_static_temporal_pair_v1"
ARMS = ("global_broadcast", "regional_4x4")
MODES = ("static_first", "temporal_sequence")
SEEDS = (20261020, 20261120, 20261220)
CASES = (4, 39, 62, 65, 77, 100)
METRICS = ("balanced_accuracy", "accuracy", "episode_macro_accuracy", "recall_stationary", "recall_advance")
DESIGN = Path("docs/algorithm-real10-static-temporal-protocol-20260923.md")
CONFIG = {"arms": list(ARMS), "modes": list(MODES), "seeds": list(SEEDS),
          "steps_per_fit": 200, "new_static_fits": 60, "reused_temporal_fits": 60,
          "threshold": .5, "trainable_parameters": 8481, "frozen_parameters": 529,
          "initialization": "source response_initial, not fitted response",
          "normalization": "each condition's training valid transitions only; std floor .001",
          "preserve": ["task", "observed_dt", "sequence_length", "valid_mask", "original_labels"],
          "static_transform": "repeat first synchronized Side/Top maps in TRAIN and TEST",
          "effective_input_rank_matched": False, "primary_arm": "regional_4x4",
          "selection": "all fixed folds/seeds, final step; no search"}


def static_batch(batch):
    return {**batch, "maps": batch["maps"][:, :1].expand_as(batch["maps"])}


def initialize(args, source):
    source_protocol = read_json(args.source / "protocol.json")
    source_report = read_json(args.source / "report.json")
    if source_report["completed_fits"] != 60 or source_report["config"] != source.CONFIG:
        raise ValueError("require the original completed 60-fit regional experiment")
    inputs, targets, rows, folds, audits = source.initialize(SimpleNamespace(
        out=args.source, source=Path(source_protocol["source"]), localizers=Path(source_protocol["localizers"])))
    protected = dict(source_protocol["protected_files"])
    for seed in SEEDS:
        for fold_id in range(10):
            for arm in ARMS:
                path = source.checkpoint_path(args.source, seed, fold_id, arm)
                protected[str(path)] = source.file_stat(path)
    snapshots = {"runner_snapshot.py": Path(__file__), "design_protocol.md": DESIGN,
                 "source_protocol.json": args.source / "protocol.json",
                 "source_report.json": args.source / "report.json",
                 "source_oof_predictions.jsonl": args.source / "oof_predictions.jsonl",
                 "source_runner_snapshot.py": Path("tools/train_real10_region_response_pair.py"),
                 "region_model_snapshot.py": Path("tools/real10_region_response.py")}
    if (args.out / "protocol.json").exists():
        protocol = read_json(args.out / "protocol.json")
        if (protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix() or
                protocol["protected_files"] != protected or protocol["folds"] != folds):
            raise ValueError("fixed protocol or protected source changed")
        if any((args.out / name).read_bytes() != path.read_bytes() for name, path in snapshots.items()):
            raise ValueError("fixed run snapshot changed; do not mix results")
    else:
        (args.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        for name, path in snapshots.items():
            shutil.copy2(path, args.out / name)
        text = DESIGN.read_text(encoding="utf-8")
        documents = {
            "plan/experiment-protocol.md": text,
            "plan/review/method-experiment-traceability.md": text.split("## 主张—实验对应\n", 1)[1].split("## 表格", 1)[0],
            "tables/table-schema.md": text.split("## 表格与图版数据合同\n", 1)[1].split("## 执行", 1)[0],
            "figures/data-manifest.md": "# 真实开发集输出，不是mock\n\n"
                "| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n"
                "| 全seed配对 | ../report.json, ../predictions.jsonl | real | 45窗/10轨迹LOEO | "
                "tools/train_real10_static_temporal_pair.py --stage render | static_temporal_summary_zh.png/.svg |\n"
                "| 六难例区域主比较 | ../predictions.jsonl | real | 固定六例/全部seed | 同上 | static_temporal_hardcases_zh.png/.svg |\n"
                "| 固定#77输入 | ../case_inputs.jsonl, 原ROI图 | actual information ablation | 同窗首帧重复/真实序列 | "
                "同上 | static_temporal_inputs_077_zh.png/.svg |\n\n"
                "训练及测试一致的首帧视觉参照；task/dt仍保留，不是纯图像或动作前预测器。"
                "原标签仅指原窗口，不是重复帧的物理真值；三seed不是独立新增样本。\n"}
        for name, content in documents.items():
            path = args.out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        write_json(args.out / "protocol.json", {"schema": SCHEMA, "config": CONFIG,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": args.source.as_posix(),
            "folds": folds, "protected_files": protected, "whole_job_estimate_s": 120,
            "conservative_upper_estimate_s": 180, "estimate_basis": "previous 60-fit job 12.1s plus replay, checks and rendering",
            "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})
        write_jsonl(args.out / "case_inputs.jsonl", [{"ui_index": rows[i]["ui_index"],
            "sample_index": i, "source_episode": rows[i]["source_episode"], "response_y": targets[i]["response_y"],
            "frames": inp["model_input"]["sequence"]} for i, inp in enumerate(inputs) if rows[i]["ui_index"] in CASES])
    return source_protocol, inputs, targets, rows, folds, audits, protected


def train(args):
    import torch
    import train_real10_region_response_pair as source
    import train_real10_spatial_aux_pair as cache
    from real10_region_response import RegionResponse

    cache.configure()
    protocol, inputs, targets, rows, folds, audits, protected = initialize(args, source)
    if (args.out / "report.json").exists():
        if args.resume:
            print("Already complete; no optimization, inference or overwrite.", flush=True)
            return
        raise FileExistsError("preserve the completed experiment")
    if any((args.out / "checkpoints").glob("*.pt")) and not args.resume:
        raise FileExistsError("use --resume for the fixed existing run")
    with cache.writer_lock(args.out):
        tick = time.perf_counter()
        tensors = cache.load_tensors(Path(protocol["source"]), inputs, targets)
        torch.cuda.reset_peak_memory_stats()
        new, skipped = 0, 0
        for seed in SEEDS:
            for fold_id, fold in enumerate(folds):
                train_batch = static_batch(cache.subset(tensors, fold["train_indices"]))
                test_real = cache.subset(tensors, fold["test_indices"])
                test_static = static_batch(test_real)
                for arm in ARMS:
                    path = source.checkpoint_path(args.out, seed, fold_id, arm)
                    expected = {"schema": SCHEMA, "config": CONFIG, "base_seed": seed, "fold_index": fold_id,
                                "arm": arm, "heldout_episode": fold["heldout_episode"],
                                "train_indices": fold["train_indices"], "test_indices": fold["test_indices"]}
                    if path.exists():
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if any(saved[k] != v for k, v in expected.items()) or saved["fit"]["optimizer_steps"] != 200:
                            raise ValueError("invalid completed static fit")
                        skipped += 1
                        continue
                    source_path = source.checkpoint_path(args.source, seed, fold_id, arm)
                    original = torch.load(source_path, map_location="cpu", weights_only=True)
                    localizer_path = str(source.previous.checkpoint_path(Path(protocol["localizers"]), seed, fold_id, "head_frozen"))
                    source.validate_checkpoint(original, seed, fold_id, fold, arm, localizer_path)
                    if original["train_indices"] != fold["train_indices"] or original["geometry_audit"] != audits[fold_id]:
                        raise ValueError("source response or geometry fold changed")
                    cache.seed_all(seed + fold_id)
                    model = RegionResponse().cuda().eval()
                    model.load_localizer(original["model_state"])
                    initial = original["response_initial"]
                    localizer = {k: v for k, v in original["model_state"].items() if k.startswith(("project.", "location."))}
                    _, features, extract_s = source.extract(model, train_batch)
                    differences = features[arm][train_batch["transition_valid"]][:, 352:1056]
                    delta_error = float(differences.abs().max())
                    if delta_error != 0:
                        raise ValueError("static training inputs retained temporal visual information")
                    info = source.fit(model, initial, localizer, features[arm], train_batch, arm, 200)
                    if (model.feature_mean[352:1056].count_nonzero() or
                            not torch.equal(model.feature_scale[352:1056], torch.full_like(model.feature_scale[352:1056], .001))):
                        raise ValueError("static normalizer did not reflect zero training differences")
                    known = features[arm][train_batch["transition_valid"]]
                    mean_error = float((model.feature_mean - known.mean(0)).abs().max())
                    scale_error = float((model.feature_scale - known.std(0, unbiased=False).clamp_min(.001)).abs().max())
                    with torch.no_grad():
                        output = source.full_forward(model, test_static, arm)
                    test_delta = float((output["region_tokens"][:, 1:] - output["region_tokens"][:, :1]).abs().max())
                    if test_delta != 0 or not torch.isfinite(output["window_logit"]).all():
                        raise ValueError("invalid static heldout forward")
                    predictions = []
                    for j, i in enumerate(fold["test_indices"]):
                        length = len(inputs[i]["model_input"]["sequence"])
                        logit = float(output["window_logit"][j])
                        predictions.append({"base_seed": seed, "fold_index": fold_id, "arm": arm, "mode": MODES[0],
                            "sample_index": i, "window_id": rows[i]["window_id"], "ui_index": rows[i]["ui_index"],
                            "source_episode": rows[i]["source_episode"], "task": rows[i]["task"],
                            "response_y": targets[i]["response_y"], "logit": logit, "prediction": int(logit >= 0),
                            "probability_advance": float(torch.sigmoid(output["window_logit"][j])),
                            "local_logits": output["local_logits"][j, :length-1].cpu().tolist()})
                    saved = {**expected, "fit_seed": seed + fold_id, "mode": MODES[0],
                        "source_checkpoint": str(source_path), "response_initial": initial,
                        "model_state": cache.cpu_state(model), "geometry_audit": audits[fold_id],
                        "fit": {**info, "extract_s": extract_s}, "predictions": predictions,
                        "static_train_difference_max_abs": delta_error, "static_test_difference_max_abs": test_delta,
                        "normalizer_recompute_max_abs": max(mean_error, scale_error),
                        "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False}
                    partial = path.with_suffix(".pt.partial")
                    torch.save(saved, partial)
                    restored = torch.load(partial, map_location="cpu", weights_only=True)
                    replay = RegionResponse().cuda().eval()
                    replay.load_state_dict(restored["model_state"])
                    with torch.no_grad():
                        repeat = source.full_forward(replay, test_static, arm)
                        replay_error = max(float((output[k] - repeat[k]).abs().max()) for k in ("window_logit", "local_logits"))
                        replay.load_state_dict(original["model_state"])
                        real = source.full_forward(replay, test_real, arm)
                    temporal_error = 0.
                    for j, record in enumerate(original["predictions"]):
                        if record["sample_index"] != fold["test_indices"][j]:
                            raise ValueError("source OOF ordering differs")
                        temporal_error = max(temporal_error, abs(float(real["window_logit"][j]) - record["logit"]),
                            float(np.max(np.abs(np.asarray(record["local_logits"]) - real["local_logits"][j, :len(record["local_logits"])].cpu().numpy()))))
                    if max(mean_error, scale_error, replay_error, temporal_error) > 1e-6:
                        raise ValueError("normalization, saved replay or original temporal replay differs")
                    if any(not torch.equal(restored["response_initial"][k], v) for k, v in initial.items()):
                        raise ValueError("static initialization differs from its temporal match")
                    saved.update({"saved_replay_max_abs": replay_error, "source_temporal_replay_max_abs": temporal_error,
                                  "initialization_matches_source": True, "frozen_localizer_matches_source": True})
                    cache.atomic_torch_save(saved, path)
                    new += 1
                    del model, replay, restored, output, repeat, real, features, known
                    print(f"{new+skipped}/60 seed={seed} fold={fold_id} {arm}: 200 static steps; temporal reused", flush=True)
                del train_batch, test_real, test_static
        if any(source.file_stat(Path(name)) != value for name, value in protected.items()):
            raise ValueError("protected source artifact changed")
        write_json(args.out / "timing.json", {"new_fits": new, "skipped_complete_fits": skipped,
            "elapsed_s": time.perf_counter()-tick, "protected_source_artifacts_unchanged": True,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/1024**2})
        predictions, fits = [], []
        for seed in SEEDS:
            for fold_id in range(10):
                pair_initial = None
                for arm in ARMS:
                    saved = torch.load(source.checkpoint_path(args.out, seed, fold_id, arm), map_location="cpu", weights_only=True)
                    if pair_initial is not None and any(not torch.equal(saved["response_initial"][k], v) for k, v in pair_initial.items()):
                        raise ValueError("global/regional initial tensors differ")
                    pair_initial = saved["response_initial"]
                    predictions.extend(saved["predictions"])
                    fits.append({k: v for k, v in saved.items() if k not in ("model_state", "response_initial", "predictions", "config")})
        predictions.extend({**r, "mode": MODES[1]} for r in read_jsonl(args.out / "source_oof_predictions.jsonl"))
        write_jsonl(args.out / "predictions.jsonl", predictions)
        write_jsonl(args.out / "fit_records.jsonl", fits)
        summarize(args.out)
    print("Complete: 60 static fits, 12000 new updates; 60 temporal fits reused, 540 OOF rows.", flush=True)


def summarize(out):
    rows = read_jsonl(out / "predictions.jsonl")
    seed_reports, lookup = [], {}
    for seed in SEEDS:
        for arm in ARMS:
            for mode in MODES:
                group = [r for r in rows if (r["base_seed"], r["arm"], r["mode"]) == (seed, arm, mode)]
                if sorted(r["sample_index"] for r in group) != list(range(45)):
                    raise ValueError("need all 45 heldout windows exactly once per condition")
                entry = {"base_seed": seed, "arm": arm, "mode": mode, **metrics(group)}
                lookup[seed, arm, mode] = entry
                seed_reports.append(entry)
    means = {a: {m: {k: {"mean": float(np.mean([lookup[s, a, m][k] for s in SEEDS])),
                         "sample_std_ddof1": float(np.std([lookup[s, a, m][k] for s in SEEDS], ddof=1))}
                     for k in METRICS} for m in MODES} for a in ARMS}
    temporal_deltas = [{"base_seed": s, "arm": a, **{k: lookup[s, a, MODES[1]][k] - lookup[s, a, MODES[0]][k]
                        for k in METRICS}} for a in ARMS for s in SEEDS]
    spatial_deltas = [{"base_seed": s, "mode": m, **{k: lookup[s, ARMS[1], m][k] - lookup[s, ARMS[0], m][k]
                       for k in METRICS}} for m in MODES for s in SEEDS]
    primary = [r for r in temporal_deltas if r["arm"] == ARMS[1]]
    consistent = (all(r["balanced_accuracy"] > 0 for r in primary) and
                  all(np.mean([r[k] for r in primary]) >= 0 for k in ("episode_macro_accuracy", "recall_stationary")))
    checks = read_jsonl(out / "fit_records.jsonl")
    write_json(out / "report.json", {"schema": SCHEMA, "config": CONFIG, "status": "complete_developmental_loeo",
        "new_static_fits": 60, "reused_temporal_fits": 60, "new_optimizer_updates": 12000,
        "new_localization_updates": 0, "oof_rows": len(rows), "distinct_windows": 45, "distinct_source_episodes": 10,
        "seed_reports": seed_reports, "means": means, "temporal_minus_static": temporal_deltas,
        "region_minus_global": spatial_deltas, "primary_descriptive_consistency_met": bool(consistent),
        "consistency_is_significance_or_deployment_gate": False, "confusion_order": ["stationary", "advance"],
        "checks": {k: max(r[k] for r in checks) for k in ("static_train_difference_max_abs", "static_test_difference_max_abs",
            "normalizer_recompute_max_abs", "saved_replay_max_abs", "source_temporal_replay_max_abs")},
        "same_initial_tensors_and_fold_localizers": all(r["initialization_matches_source"] and r["frozen_localizer_matches_source"] for r in checks),
        "same_nominal_parameter_budget_not_effective_rank": True, "independent_test": False,
        "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False})


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    report = read_json(args.out / "report.json")
    rows = read_jsonl(args.out / "predictions.jsonl")
    if len(rows) != 540 or report["new_static_fits"] != 60:
        raise ValueError("complete matched run required")
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 9, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "svg.fonttype": "path"})
    names, colors = ("首帧训练/评估", "真实时序"), ("#0077BB", "#EE7733", "#009988")
    figures = args.out / "figures"
    if not (figures / "data-manifest.md").exists():
        raise ValueError("figure data contract missing")

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=450)
        plt.close(fig)

    lut = {(r["base_seed"], r["arm"], r["mode"]): r for r in report["seed_reports"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    for ax, arm, title in zip(axes, ARMS, ("全局摘要参照", "保留4×4区域（主比较）")):
        for j, seed in enumerate(SEEDS):
            ax.plot([0, 1], [100*lut[seed, arm, m]["balanced_accuracy"] for m in MODES], "o-",
                    color=colors[j], lw=1.2, ms=4, label=f"seed {seed}")
        mean = [100*report["means"][arm][m]["balanced_accuracy"]["mean"] for m in MODES]
        ax.set(title=f"{title}\n平均BA {mean[0]:.2f}% → {mean[1]:.2f}%", xticks=[0, 1],
               xticklabels=names, ylabel="平衡准确率（%）", ylim=(0, 100), xlim=(-.22, 1.22))
    fig.suptitle("训练匹配的静态参照：跨帧视觉信息是否带来增益？", y=.99, fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .92), ncol=3, frameon=False)
    fig.text(.5, .015, "同45窗 / 10轨迹、定位器、初始张量、200步；task/dt均保留。\n"
             "时序权重只读复用；相同名义参数量不等于相同有效输入秩。反复使用的开发集，非独立测试。", ha="center", fontsize=8)
    fig.subplots_adjust(top=.71, bottom=.19, wspace=.26)
    save(fig, "static_temporal_summary_zh")

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.8))
    hardcases = []
    for ax, ui in zip(axes.flat, CASES):
        group = [r for r in rows if r["ui_index"] == ui and r["arm"] == ARMS[1]]
        truth = group[0]["response_y"]
        for j, mode in enumerate(MODES):
            selected = [next(r for r in group if r["base_seed"] == s and r["mode"] == mode) for s in SEEDS]
            values = [r["probability_advance"] for r in selected]
            ax.plot(range(3), values, "o-" if j == 0 else "s--", color=colors[j], label=names[j], lw=1.2, ms=4)
            hardcases.append({"ui_index": ui, "truth": truth, "arm": ARMS[1], "mode": mode,
                "seeds": list(SEEDS), "probability_advance": values, "correct_seeds": sum(r["prediction"] == truth for r in selected)})
        note = {4: "小幅推进", 65: "有移动但未推进", 100: "小幅推进"}.get(ui, "")
        ax.axhline(.5, color="#888888", ls=":", lw=1)
        ax.set(title=f"#{ui} {'推进' if truth else '未推进'}" + (f"（{note}）" if note else ""),
               xticks=range(3), xticklabels=["seed 1", "seed 2", "seed 3"], ylabel="推进分数", ylim=(-.03, 1.03))
    fig.suptitle("区域主比较：六个固定难例，全部种子", y=.99, fontsize=12)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.text(.5, .015, "固定0.5阈值，未校准分数。保留原窗口标签，不将重复首帧重新标为静止。", ha="center", fontsize=8)
    fig.subplots_adjust(top=.83, bottom=.10, hspace=.43, wspace=.31)
    save(fig, "static_temporal_hardcases_zh")
    write_json(args.out / "hardcase_readback.json", hardcases)

    case = next(r for r in read_jsonl(args.out / "case_inputs.jsonl") if r["ui_index"] == 77)
    frames = case["frames"]
    positions = (0, len(frames)//2, len(frames)-1)
    fig, axes = plt.subplots(4, 3, figsize=(9.5, 10.2))
    for row, (mode, view) in enumerate((m, v) for m in MODES for v in ("side", "top")):
        for col, position in enumerate(positions):
            index = 0 if mode == MODES[0] else position
            with Image.open(frames[index]["images"][view]) as img:
                axes[row, col].imshow(img.convert("RGB"))
            axes[row, col].set_title(f"{'首帧参照' if mode == MODES[0] else '真实时序'} · {view} · 输入位置{position} / 图像帧{index}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("固定示例 #77：静态训练/评估实际只看首帧双视角", y=.995, fontsize=12)
    fig.text(.5, .012, "每列Side/Top来自同一时刻；task、dt、长度和mask保留。原标签为推进。\n"
             "上两行仅为输入信息消融，不是新采集的静止序列；不产生接触或运动监督。", ha="center", fontsize=8)
    fig.subplots_adjust(top=.955, bottom=.07, hspace=.20, wspace=.08)
    save(fig, "static_temporal_inputs_077_zh")
    (args.out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>静态/时序公平对照</title>'
        '<style>body{max-width:1100px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>首帧训练参照 vs 已有时序模型</h1><p>45窗/10轨迹/3seed开发集；同定位器、初始化、200步。'
        '保留task/dt，不是纯图像、动作前效应预测、接触识别或实机验证。</p>'
        + ''.join(f'<img src="figures/{name}.png">' for name in
                  ("static_temporal_summary_zh", "static_temporal_hardcases_zh", "static_temporal_inputs_077_zh")), encoding="utf-8")
    print("Rendered three Chinese PNG/SVG sheets from saved outputs; user acceptance pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "render"), required=True)
    parser.add_argument("--source", type=Path, default=Path("simulation_output/real10_region_response_pair_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_static_temporal_pair_v1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"train": train, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
