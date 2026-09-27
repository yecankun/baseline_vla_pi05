"""Frozen regional/global response sensitivity to repeated endpoint frames.

probe: remote, no optimization. render: local, saved JSON/images only, no Torch.
Labels refer to the ORIGINAL observed window, not to synthetic static sequences.
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

SCHEMA = "real10_region_temporal_dependence_v1"
MODES = ("real_sequence", "repeat_first", "repeat_last")
ARMS = ("global_broadcast", "regional_4x4")
SEEDS = (20261020, 20261120, 20261220)
CASES = (4, 39, 62, 65, 77, 100)
DESIGN = Path("docs/algorithm-real10-region-temporal-protocol-20260923.md")
METRICS = ("balanced_accuracy", "accuracy", "episode_macro_accuracy", "recall_stationary", "recall_advance")
CONFIG = {"modes": list(MODES), "arms": list(ARMS), "seeds": list(SEEDS),
          "optimizer_updates": 0, "threshold": .5,
          "normalizer": "unchanged training-fold buffers from each source checkpoint",
          "preserve": ["task", "observed_dt", "sequence_length", "valid_transition_mask"],
          "target_semantics": "original observed window label; NOT counterfactual physical truth",
          "selection": "all source checkpoints; no tuning or extra conditions"}


def summarize(out):
    from train_real10_head_response_pair import metrics
    units = [read_json(p) for p in sorted((out / "units").glob("*.json"))]
    if len(units) != 60:
        raise ValueError("need all 60 frozen checkpoint units")
    predictions = [r for unit in units for r in unit["predictions"]]
    seed_reports, impacts, arm_deltas = [], [], []
    lookup = {}
    for seed in SEEDS:
        for arm in ARMS:
            for mode in MODES:
                group = [r for r in predictions if (r["base_seed"], r["arm"], r["mode"]) == (seed, arm, mode)]
                if sorted(r["sample_index"] for r in group) != list(range(45)):
                    raise ValueError("require one prediction per original heldout window")
                entry = {"base_seed": seed, "arm": arm, "mode": mode, **metrics(group)}
                seed_reports.append(entry)
                lookup[seed, arm, mode] = entry
                if mode != MODES[0]:
                    impacts.append({"base_seed": seed, "arm": arm, "mode": mode,
                        "metric_delta_repeat_minus_real": {k:entry[k]-lookup[seed,arm,MODES[0]][k] for k in METRICS},
                        "flipped_windows": sum(r["flipped_vs_real"] for r in group),
                        "flip_rate": float(np.mean([r["flipped_vs_real"] for r in group])),
                        "correct_to_wrong": sum(r["real_prediction"] == r["response_y"] and r["prediction"] != r["response_y"] for r in group),
                        "wrong_to_correct": sum(r["real_prediction"] != r["response_y"] and r["prediction"] == r["response_y"] for r in group),
                        "flip_rate_by_original_class": {name:float(np.mean([r["flipped_vs_real"] for r in group if r["response_y"]==y])) for y,name in enumerate(("stationary","advance"))},
                        "mean_abs_score_change": float(np.mean([abs(r["score_delta_vs_real"]) for r in group])),
                        "mean_abs_logit_change": float(np.mean([abs(r["logit_delta_vs_real"]) for r in group]))})
        for mode in MODES:
            arm_deltas.append({"base_seed":seed,"mode":mode,
                **{k:lookup[seed,ARMS[1],mode][k]-lookup[seed,ARMS[0],mode][k] for k in METRICS}})
    means = {arm:{mode:{k:{"mean":float(np.mean([lookup[s,arm,mode][k] for s in SEEDS])),
                               "sample_std_ddof1":float(np.std([lookup[s,arm,mode][k] for s in SEEDS],ddof=1))}
                         for k in METRICS} for mode in MODES} for arm in ARMS}
    write_jsonl(out / "predictions.jsonl", predictions)
    write_jsonl(out / "unit_checks.jsonl", [{k:v for k,v in u.items() if k != "predictions"} for u in units])
    write_json(out / "report.json", {"schema":SCHEMA,"status":"complete_fixed_weight_sensitivity_probe",
        "config":CONFIG,"source_checkpoints":60,"prediction_rows":len(predictions),"original_windows":45,
        "source_episodes":10,"optimizer_updates":0,"real_replay_rows":270,"static_diagnostic_rows":540,
        "seed_reports":seed_reports,"means":means,"repeat_minus_real_impacts":impacts,
        "region_minus_global_by_condition":arm_deltas,"confusion_order":["stationary","advance"],
        "real_replay_max_abs":max(u["real_replay_max_abs"] for u in units),
        "repeat_raw_temporal_difference_max_abs":max(u["repeat_raw_temporal_difference_max_abs"] for u in units),
        "all_parameters_and_buffers_unchanged":all(u["all_parameters_and_buffers_unchanged"] for u in units),
        "synthetic_static_labels_are_physical_truth":False,"independent_test":False,
        "policy_input_allowed":False,"formal_data_allowed":False,"deployable":False})


def probe(args):
    import torch
    import train_real10_region_response_pair as source
    import train_real10_spatial_aux_pair as cache
    from real10_region_response import RegionResponse

    cache.configure()
    tick = time.perf_counter()
    source_protocol = read_json(args.source / "protocol.json")
    source_report = read_json(args.source / "report.json")
    if source_report["completed_fits"] != 60 or source_report["config"] != source.CONFIG:
        raise ValueError("require the completed fixed regional/global pair")
    # Existing completed source only: initialize validates its snapshots without creating any files.
    inputs, targets, rows, folds, _ = source.initialize(SimpleNamespace(
        out=args.source, source=Path(source_protocol["source"]), localizers=Path(source_protocol["localizers"])))
    paths = [source.checkpoint_path(args.source,s,f,a) for s in SEEDS for f in range(10) for a in ARMS]
    paths.append(Path(source_protocol["source"]) / "feature_cache.pt")
    protected = {str(p):source.file_stat(p) for p in paths}
    snapshots = {"runner_snapshot.py":Path(__file__), "design_protocol.md":DESIGN,
                 "source_protocol.json":args.source/"protocol.json", "source_report.json":args.source/"report.json",
                 "source_oof_predictions.jsonl":args.source/"oof_predictions.jsonl",
                 "region_model_snapshot.py":Path("tools/real10_region_response.py")}
    if args.out.exists():
        protocol = read_json(args.out / "protocol.json")
        if protocol["config"] != CONFIG or protocol["source"] != args.source.as_posix() or protocol["protected_files"] != protected:
            raise ValueError("diagnostic protocol or source artifacts changed")
        if any((args.out/name).read_bytes() != path.read_bytes() for name,path in snapshots.items()):
            raise ValueError("diagnostic snapshot differs; do not mix results")
        if (args.out / "report.json").exists():
            if args.resume:
                print("Already complete; no inference, optimization or overwrite.", flush=True)
                return
            raise FileExistsError("preserve this completed diagnostic")
        if not args.resume:
            raise FileExistsError("use --resume for this fixed diagnostic")
    else:
        (args.out / "units").mkdir(parents=True)
        for name,path in snapshots.items():
            shutil.copy2(path,args.out/name)
        write_json(args.out/"protocol.json", {"schema":SCHEMA,"config":CONFIG,
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"source":args.source.as_posix(),
            "folds":folds,"protected_files":protected,"whole_job_estimate_s":90,"conservative_upper_estimate_s":180,
            "estimate_basis":"previous full 60-fit job 12.1s; here 180 heldout forward batches, no optimizer, plus verification/render",
            "policy_input_allowed":False,"formal_data_allowed":False,"deployable":False})
        write_jsonl(args.out/"case_inputs.jsonl", [{"ui_index":rows[i]["ui_index"],"sample_index":i,
            "source_episode":rows[i]["source_episode"],"response_y":targets[i]["response_y"],
            "frames":inp["model_input"]["sequence"]} for i,inp in enumerate(inputs) if rows[i]["ui_index"] in CASES])
    with cache.writer_lock(args.out):
        tensors = cache.load_tensors(Path(source_protocol["source"]),inputs,targets)
        torch.cuda.reset_peak_memory_stats()
        new, skipped = 0, 0
        with torch.inference_mode():
            for seed in SEEDS:
                for fold_id,fold in enumerate(folds):
                    batch = cache.subset(tensors,fold["test_indices"])
                    lengths = batch["transition_valid"].sum(1)+1
                    for arm in ARMS:
                        unit_path = args.out/"units"/f"seed{seed}_fold{fold_id:02d}_{arm}.json"
                        expected = {"base_seed":seed,"fold_index":fold_id,"arm":arm,"heldout_episode":fold["heldout_episode"]}
                        if unit_path.exists():
                            unit = read_json(unit_path)
                            if (any(unit[k]!=v for k,v in expected.items()) or
                                    len(unit["predictions"]) != 3*len(fold["test_indices"]) or not unit["all_parameters_and_buffers_unchanged"]):
                                raise ValueError("invalid complete inference unit")
                            skipped += 1
                            continue
                        path = source.checkpoint_path(args.source,seed,fold_id,arm)
                        saved = torch.load(path,map_location="cpu",weights_only=True)
                        source.validate_checkpoint(saved,seed,fold_id,fold,arm,
                            str(Path(source_protocol["localizers"])/"checkpoints"/f"seed{seed}_fold{fold_id:02d}_head_frozen.pt"))
                        if saved["train_indices"] != fold["train_indices"]:
                            raise ValueError("source train split changed")
                        model = RegionResponse().cuda().eval().requires_grad_(False)
                        model.load_state_dict(saved["model_state"])
                        predictions, real, replay_error, repeat_difference = [], {}, 0., 0.
                        saved_lookup = {r["sample_index"]:r for r in saved["predictions"]}
                        for mode in MODES:
                            maps = batch["maps"]
                            if mode == "repeat_first":
                                maps = maps[:,:1].expand_as(maps)
                            elif mode == "repeat_last":
                                maps = maps[torch.arange(len(lengths),device=maps.device),lengths-1][:,None].expand_as(maps)
                            output = model(maps,batch["dt_s"],batch["task_right"],batch["transition_valid"],arm)
                            if not torch.isfinite(output["window_logit"]).all() or not torch.isfinite(output["local_logits"]).all():
                                raise ValueError("nonfinite frozen output")
                            if mode != MODES[0]:
                                tokens = output["region_tokens"]
                                repeat_difference = max(repeat_difference,float((tokens[:,1:]-tokens[:,:1]).abs().max()))
                                if repeat_difference > 1e-6:
                                    raise ValueError("repeated images retained a raw temporal difference")
                            for j,i in enumerate(fold["test_indices"]):
                                length = len(inputs[i]["model_input"]["sequence"])
                                if int(lengths[j]) != length:
                                    raise ValueError("last valid frame differs from the observed sequence")
                                value = float(output["window_logit"][j])
                                score = float(torch.sigmoid(output["window_logit"][j]))
                                pred = int(value >= 0)
                                local = output["local_logits"][j,:length-1].cpu().tolist()
                                if mode == MODES[0]:
                                    original = saved_lookup[i]
                                    replay_error = max(replay_error,abs(value-original["logit"]),
                                        float(np.max(np.abs(np.asarray(local)-original["local_logits"]))))
                                    if replay_error > 1e-6 or pred != original["prediction"]:
                                        raise ValueError("real-sequence replay differs from source OOF")
                                    real[i] = (value,score,pred)
                                reference = real[i]
                                predictions.append({**expected,"mode":mode,"sample_index":i,
                                    "window_id":rows[i]["window_id"],"ui_index":rows[i]["ui_index"],
                                    "source_episode":rows[i]["source_episode"],"task":rows[i]["task"],
                                    "response_y":targets[i]["response_y"],"label_is_original_window_reference":True,
                                    "observed_frames":length,"repeated_frame_index":0 if mode=="repeat_first" else length-1 if mode=="repeat_last" else None,
                                    "logit":value,"probability_advance":score,"prediction":pred,"local_logits":local,
                                    "real_prediction":reference[2],"flipped_vs_real":pred!=reference[2],
                                    "score_delta_vs_real":score-reference[1],"logit_delta_vs_real":value-reference[0]})
                        unchanged = all(torch.equal(v.cpu(),saved["model_state"][k]) for k,v in model.state_dict().items())
                        if not unchanged:
                            raise ValueError("frozen model/normalizer changed")
                        unit = {**expected,"source_checkpoint":str(path),"predictions":predictions,
                            "real_replay_max_abs":replay_error,"repeat_raw_temporal_difference_max_abs":repeat_difference,
                            "all_parameters_and_buffers_unchanged":unchanged,"optimizer_updates":0}
                        partial = unit_path.with_suffix(".json.partial")
                        write_json(partial,unit)
                        partial.replace(unit_path)
                        new += 1
                        del model, saved, output
                    print(f"seed={seed} fold={fold_id}: both arms, all three modes complete",flush=True)
                    del batch
        torch.cuda.synchronize()
        if any(source.file_stat(Path(name)) != stat for name,stat in protected.items()):
            raise ValueError("protected source artifact changed")
        write_json(args.out/"timing.json", {"elapsed_s":time.perf_counter()-tick,"new_units":new,"skipped_units":skipped,
            "optimizer_updates":0,"protected_source_artifacts_unchanged":True,
            "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/1024**2})
        summarize(args.out)
    print("Complete: 60 frozen checkpoints, 810 diagnostic predictions, zero optimizer updates.",flush=True)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from PIL import Image

    report = read_json(args.out/"report.json")
    rows = read_jsonl(args.out/"predictions.jsonl")
    cases = read_jsonl(args.out/"case_inputs.jsonl")
    if report["prediction_rows"] != 810 or len(rows) != 810 or len(cases) != 6:
        raise ValueError("complete fixed diagnostic results required")
    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family":"Microsoft YaHei","font.size":9,"axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight","svg.fonttype":"path"})
    names = ("真实序列","首帧重复","末帧重复")
    arm_names = ("全局摘要","保留4×4区域")
    colors = ("#0077BB","#EE7733","#009988")
    figures = args.out/"figures"
    figures.mkdir(exist_ok=True)
    (figures/"data-manifest.md").write_text(
        "# 冻结权重时序依赖诊断：真实模型输出\n\n"
        "| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n"
        "| 全seed/条件BA | ../report.json, ../predictions.jsonl | real model output | 60冻结checkpoint，45原窗 | "
        "tools/probe_real10_region_temporal_dependence.py --stage render | temporal_summary_zh.png/.svg |\n"
        "| 六难例分数 | ../predictions.jsonl | real model output | 全部seed/arm/condition | 同上 | temporal_hardcases_zh.png/.svg |\n"
        "| 固定#65/#77输入示例 | ../case_inputs.jsonl, 原ROI224图 | actual diagnostic intervention | 原图或同窗端点重复 | "
        "同上 | temporal_inputs_065_zh.png/.svg, temporal_inputs_077_zh.png/.svg |\n\n"
        "不是mock性能。重复图像是显式诊断干预，不是实际静止采集；标签仍参照原窗口，不是合成序列物理真值。\n"
        "同一权重与归一化，任务/dt/长度保留；不重新拟合或校准。重复帧可能OOD，性能变化不是正确运动理解的证明。\n"
        "45窗嵌套于10轨迹，810条输出与3seed不是独立新增样本。原标签未改变。\n",encoding="utf-8")

    def save(fig,name):
        for ext in ("png","svg"):
            fig.savefig(figures/f"{name}.{ext}",dpi=450)
        plt.close(fig)

    lut = {(r["base_seed"],r["arm"],r["mode"]):r for r in report["seed_reports"]}
    fig,axes = plt.subplots(1,2,figsize=(10.2,4.7))
    for ax,arm,title in zip(axes,ARMS,arm_names):
        for j,seed in enumerate(SEEDS):
            values = [100*lut[seed,arm,mode]["balanced_accuracy"] for mode in MODES]
            ax.plot(range(3),values,"o-",color=colors[j],label=f"seed {seed}",lw=1.25,ms=4)
        means = [100*report["means"][arm][mode]["balanced_accuracy"]["mean"] for mode in MODES]
        ax.plot(range(3),means,"D--",color="#555555",lw=1.5,ms=4,label="三seed均值")
        ax.set(title=title+"\n均值 "+" / ".join(f"{x:.2f}%" for x in means),xticks=range(3),xticklabels=names,
               ylabel="相对原窗口标签的平衡准确率（%）",ylim=(0,100),xlim=(-.18,2.18))
    fig.suptitle("冻结权重：移除帧间视觉变化后，原标签判定保留多少？",y=.99,fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.92),ncol=4,frameon=False,fontsize=8)
    fig.text(.5,.015,"重复帧仅为诊断输入；原标签不代表合成序列的物理真值。任务、dt、长度和训练归一化全部保持不变。\n"
             "反复使用的45窗 / 10轨迹；零训练。分数变化不能单独证明正确的导丝运动理解。",ha="center",fontsize=8)
    fig.subplots_adjust(top=.70,bottom=.18,wspace=.30)
    save(fig,"temporal_summary_zh")

    fig,axes = plt.subplots(2,3,figsize=(11,7))
    hardcases = []
    for ax,ui in zip(axes.flat,CASES):
        chosen = [r for r in rows if r["ui_index"]==ui]
        truth = chosen[0]["response_y"]
        for j,arm in enumerate(ARMS):
            for k,seed in enumerate(SEEDS):
                values = [next(r["probability_advance"] for r in chosen if (r["arm"],r["base_seed"],r["mode"])==(arm,seed,mode)) for mode in MODES]
                ax.plot(range(3),values,marker=("o","s","^")[k],ls="-" if j==0 else "--",color=colors[j],alpha=.75,
                        label=f"{arm_names[j]} · seed {k+1}",lw=1,ms=4)
                hardcases.append({"ui_index":ui,"original_response_y":truth,"arm":arm,"base_seed":seed,
                                  "modes":list(MODES),"probability_advance":values})
        ax.axhline(.5,color="#888888",ls=":",lw=1)
        ax.set(title=f"#{ui} 原标签：{'推进' if truth else '未推进'}",xticks=range(3),xticklabels=names,
               ylabel="推进分数",ylim=(-.03,1.03))
    fig.suptitle("六个固定难例：全seed / 全条件，不选择最好结果",y=.99,fontsize=12)
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,.94),ncol=3,frameon=False,fontsize=8)
    fig.text(.5,.015,"固定阈值0.5；原标签仅作参照，输出不是校准置信度。seed 1/2/3：20261020 / 20261120 / 20261220。",ha="center",fontsize=8)
    fig.subplots_adjust(top=.80,bottom=.11,hspace=.38,wspace=.30)
    save(fig,"temporal_hardcases_zh")
    write_json(args.out/"hardcase_readback.json",hardcases)

    links=[]
    for ui in (65,77):
        case = next(r for r in cases if r["ui_index"]==ui)
        frames = case["frames"]
        selected = (0,len(frames)//2,len(frames)-1)
        fig,axes = plt.subplots(6,3,figsize=(9.2,14))
        for m,mode in enumerate(MODES):
            for v,view in enumerate(("side","top")):
                for j,t in enumerate(selected):
                    idx = t if mode=="real_sequence" else 0 if mode=="repeat_first" else len(frames)-1
                    ax = axes[2*m+v,j]
                    path = frames[idx]["images"][view]
                    with Image.open(path) as img:
                        ax.imshow(img.convert("RGB"))
                    ax.set(title=f"位置{t+1}/{len(frames)} ← 图像{Path(path).stem}",xticks=[],yticks=[])
                    if j==0:
                        ax.set_ylabel(names[m]+"\n"+view.capitalize(),fontsize=10)
        fig.suptitle(f"#{ui} 固定输入示例 · 原标签{'推进' if case['response_y'] else '未推进'}\n"
                     "首/中/末显示位置；每个位置的Side/Top始终来自同一帧",y=.997,fontsize=12)
        fig.text(.5,.008,"重复行是人为诊断输入，不是新的静止采集，也不改变原人工标签；图像只作输入展示，不运行模型。",ha="center",fontsize=8)
        fig.subplots_adjust(top=.935,bottom=.04,left=.09,right=.98,hspace=.31,wspace=.09)
        name=f"temporal_inputs_{ui:03d}_zh"
        save(fig,name)
        links.append(f'<h2>#{ui} · 诊断输入示例</h2><img src="figures/{name}.png">')
    (args.out/"index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>冻结权重时序依赖诊断</title>'
        '<style>body{max-width:1150px;margin:30px auto;font-family:Microsoft YaHei,sans-serif}img{width:100%}</style>'
        '<h1>真实序列 / 首帧重复 / 末帧重复</h1><p>60份冻结权重、原45窗、零训练。'
        '重复帧可能偏离训练分布；原标签仅作参照，不是合成序列物理真值，也不是导丝运动理解或接触证明。</p>'
        '<img src="figures/temporal_summary_zh.png"><img src="figures/temporal_hardcases_zh.png">'+''.join(links),encoding="utf-8")
    print("Rendered 4 Chinese PNG/SVG sheets. User visual acceptance pending.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("probe","render"),required=True)
    parser.add_argument("--source",type=Path,default=Path("simulation_output/real10_region_response_pair_v1"))
    parser.add_argument("--out",type=Path,default=Path("simulation_output/real10_region_temporal_dependence_v1"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    {"probe":probe,"render":render}[args.stage](args)


if __name__=="__main__":
    main()
