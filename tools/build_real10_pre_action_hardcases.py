"""Read-only human review of pre-action response disagreements; never relabels.

Build metadata/HTML remotely; serve locally against existing original captures.
No image copying, model loading, training or annotation writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_real10_event_windows import read_jsonl, write_json


LABEL_ZH = {"advance": "沿管推进", "stationary": "无明显推进", "uncertain": "无法判断", "retract": "回退"}
ARMS = ("observation_only", "observation_plus_request")


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Real10 · 动作前预测难例复核（只读）</title>
<style>
*{box-sizing:border-box}body{font:16px "Microsoft YaHei",sans-serif;max-width:1480px;margin:24px auto;padding:0 22px;color:#20314a;background:#f3f6fa}h1{font-size:28px;margin-bottom:10px}h2{font-size:21px}.notice,section{background:white;border-radius:12px;padding:18px;margin:16px 0}.notice{border-left:5px solid #4e79a7;line-height:1.75}.muted{font-size:14px;color:#607089;line-height:1.7}.bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap}button,select{font:inherit;border:1px solid #c7d2e1;border-radius:6px;padding:8px 12px;background:white;color:#20314a;cursor:pointer}button.primary{background:#315d88;color:white;border-color:#315d88}.views{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}.views figure{margin:0}.views img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#101820;border-radius:5px}.views figcaption{padding:8px 0;font-size:14px}input[type=range]{width:100%;margin:15px 0}details{margin-top:15px}summary{cursor:pointer;font-weight:bold;padding:10px 0}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.8 "Microsoft YaHei",sans-serif}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #dce3ed}a{color:#315d88}#phase{padding:5px 10px;border-radius:5px;background:#eef4fb}#frame-error{color:#b42318}@media(max-width:850px){.views{grid-template-columns:1fr}body{padding:0 10px}}
</style>
<h1>动作前预测难例复核 · 只读</h1>
<div class="notice"><b>模型预测错，不等于人工标注错。</b>建议先看双视角回放独立判断，再展开已有标签与模型结果。<br>
运动响应只看<b>锚点及之后</b>是否沿管推进；动作前画面只作上下文。模糊但可可靠辨认仍可判断；单路遮挡时可依据另一路；两路证据都不足时才是“无法判断”。<br>
不要根据 feed/hold、Elite日志或模型分数猜标签，也不要把历史帧运动算成后续推进。此页不保存或修改任何源标注。</div>
<section><div class="bar"><label>范围 <select id="filter"><option value="priority">优先8窗</option><option value="hard">全部难例22窗</option><option value="flipped">新增错例2窗</option><option value="all">全部45窗（含一致窗口）</option></select></label><label>窗口 <select id="case"></select></label><button id="prev">上一窗</button><button id="next">下一窗</button><span id="position" class="muted"></span></div>
<p class="muted">20窗两组都错，2窗仅加入请求后错。优先8窗按两组平均分数与0.5阈值的距离排序、每种人工标签各取4窗；这只是复核顺序，不是标错概率。原始图像来自本机 collected_data，未裁剪/增强；点击图像可看原尺寸。</p></section>
<section><h2 id="title"></h2><div class="bar"><button id="anchor">回到锚点</button><button class="primary" id="play">播放响应</button><button id="back">前一帧</button><button id="forward">后一帧</button><label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option></select></label><label><input id="loop" type="checkbox" checked>循环</label><span id="phase"></span></div>
<div class="views"><figure><a id="side-link" target="_blank" rel="noopener"><img id="side" alt="Side 原始图像"></a><figcaption id="side-caption"></figcaption></figure><figure><a id="top-link" target="_blank" rel="noopener"><img id="top" alt="Top 原始图像"></a><figcaption id="top-caption"></figcaption></figure></div>
<input type="range" id="slider" aria-label="窗口帧" min="0" value="0"><div id="time" class="muted"></div><div id="frame-error"></div>
<p class="muted">拖动可看整段历史和响应；“播放响应”从锚点开始。两路按记录帧配对，保留采样间隔和相机时间偏差，不是假定恒定帧率或严格硬件同步。</p>
<details id="reveal"><summary>看完后：展开实验标注快照、预测结果和动作审计</summary><pre id="evidence"></pre></details>
<p class="muted">若认为需修订，可回复：“#编号：改为沿管推进/无明显推进/无法判断/回退；依据Side/Top/两路；简述可见证据”。若原标签正确，回复“#编号：保留”即可。此页没有自动写回或训练功能。</p></section>
<section><details><summary>全部难例清单（展开会显示原标签）</summary><table><thead><tr><th>编号</th><th>任务</th><th>现有人工标签</th><th>仅观测分数</th><th>加请求分数</th><th>类别</th></tr></thead><tbody id="all-hard"></tbody></table></details><p class="muted"><a href="summary.md">文字版复核清单</a> · <a href="cases.json">完整复核数据</a> · <a href="../figures/pre_action_response_zh.png">本次对照结果图</a></p></section>
<script>
const DATA=__DATA__;
const $=id=>document.getElementById(id), zh=__LABELS__;
let subset=[], current=null, frame=0, playing=false, generation=0, timer=null;
function stop(){playing=false;clearTimeout(timer);generation++;$('play').textContent='播放响应'}
function rawImage(path){return new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(new Error(path));i.src=path})}
async function draw(index){
  frame=Math.max(0,Math.min(index,current.frames.length-1));const token=++generation,f=current.frames[frame];
  $('slider').value=frame;$('phase').textContent=frame===current.anchor_index?'锚点':frame<current.anchor_index?'动作前 · 仅上下文':'响应区间 · 用于判断';
  $('time').textContent=`step ${f.step} · 相对锚点 ${f.relative_time_s.toFixed(3)} s · ${frame+1}/${current.frames.length}`;$('frame-error').textContent='';
  try{await Promise.all(['side','top'].map(v=>rawImage(f.images[v].original)));if(token!==generation)return;
    for(const v of ['side','top']){$(v).src=f.images[v].original;$(v+'-link').href=f.images[v].original;$(v+'-caption').textContent=`${v==='side'?'Side 侧视':'Top 俯视'} · 原始图像 · ${f.images[v].relative_time_s.toFixed(3)} s`;}
  }catch(err){if(token===generation){stop();$('frame-error').textContent='原图未加载：'+err.message+'。请使用本机项目根目录的只读服务打开。'}}
}
function revealText(c){
  const p=c.prediction,a=c.annotation;
  return `本次实验人工标签：${zh[a.joint_motion_response]}；判断依据：${a.motion_evidence}\n锚点可见性：Side=${a.anchor_visibility.side}，Top=${a.anchor_visibility.top}\n人工备注：${a.notes||'无'}\n\n仅动作前信息：${zh[p.methods.observation_only.prediction]}，score=${p.methods.observation_only.score.toFixed(6)}\n加入Piper请求：${zh[p.methods.observation_plus_request.prediction]}，score=${p.methods.observation_plus_request.score.toFixed(6)}\n固定阈值0.5；score不是置信度或校准概率。\n\n锚点Piper请求：${p.piper_request}（不是实际位移）\n${c.audit_text}\n这些记录只帮助理解混合动作窗口，不是运动/碰壁标签证据。\n\n完整窗口：${p.window_id}\n标注快照时间：${a.saved_at_utc}`;
}
async function choose(ui){stop();current=DATA.cases.find(c=>c.prediction.ui_index===Number(ui));$('case').value=String(ui);$('title').textContent=`#${ui} · ${current.short_episode} · 锚点 step ${current.anchor_step}`;$('reveal').open=false;$('evidence').textContent=revealText(current);$('slider').max=current.frames.length-1;$('position').textContent=`${subset.indexOf(current)+1}/${subset.length}`;history.replaceState(null,'','#'+ui);await draw(current.anchor_index)}
function filter(keep){stop();const f=$('filter').value;subset=DATA.cases.filter(c=>f==='all'||f==='priority'&&c.priority||f==='hard'&&c.hard||f==='flipped'&&c.flipped);if(f==='priority')subset.sort((a,b)=>a.priority_rank-b.priority_rank);$('case').replaceChildren();for(const c of subset){const o=document.createElement('option');o.value=c.prediction.ui_index;o.textContent=`#${c.prediction.ui_index} · ${c.short_episode}`;$('case').append(o)}choose(subset.find(c=>c.prediction.ui_index===keep)?.prediction.ui_index||subset[0].prediction.ui_index)}
async function playStep(){if(!playing)return;const before=generation;await draw(frame);if(!playing||generation!==before+1)return;const next=frame+1;if(next>=current.frames.length){if(!$('loop').checked){stop();return}timer=setTimeout(()=>{frame=current.anchor_index;playStep()},700);return}const delay=Math.max(50,1000*(current.frames[next].relative_time_s-current.frames[frame].relative_time_s)/Number($('speed').value));timer=setTimeout(()=>{frame=next;playStep()},delay)}
$('play').onclick=()=>{if(playing){stop();return}if(frame<current.anchor_index||frame>=current.frames.length-1)frame=current.anchor_index;playing=true;$('play').textContent='暂停';playStep()};
$('filter').onchange=()=>filter(current?.prediction.ui_index);$('case').onchange=()=>choose($('case').value);$('anchor').onclick=()=>{stop();draw(current.anchor_index)};$('slider').oninput=()=>{stop();draw(Number($('slider').value))};$('back').onclick=()=>{stop();draw(frame-1)};$('forward').onclick=()=>{stop();draw(frame+1)};
for(const [id,d] of [['prev',-1],['next',1]])$(id).onclick=()=>{const i=(subset.indexOf(current)+d+subset.length)%subset.length;choose(subset[i].prediction.ui_index)};
for(const c of DATA.cases.filter(c=>c.hard)){const p=c.prediction,tr=document.createElement('tr');for(const text of ['#'+p.ui_index,c.short_episode,zh[p.label],p.methods.observation_only.score.toFixed(4),p.methods.observation_plus_request.score.toFixed(4),c.flipped?'加入请求后错':'两组都错']){const td=document.createElement('td');td.textContent=text;tr.append(td)}tr.firstChild.style.cursor='pointer';tr.firstChild.onclick=()=>{$('filter').value='hard';filter(p.ui_index);window.scrollTo({top:0,behavior:'smooth'})};$('all-hard').append(tr)}
const requested=Number(location.hash.slice(1));if(requested&&DATA.cases.some(c=>c.prediction.ui_index===requested)){$('filter').value='all';filter(requested)}else filter();
</script></html>'''


def build(args):
    if args.out.exists():
        raise FileExistsError("preserve review output; choose a fresh --out")
    predictions = read_jsonl(args.experiment / "oof_predictions.jsonl")
    snapshot = {r["window_id"]: r for r in read_jsonl(args.experiment / "annotation_snapshot.jsonl")}
    windows = {r["window_id"]: r for r in read_jsonl(args.pack / "windows.jsonl")}
    targets = {r["window_id"]: r for r in read_jsonl(args.pack / "window_targets.jsonl")}
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    audits = {r["window_id"]: r for r in read_jsonl(args.interface / "timing_audit.jsonl")}
    both_wrong = [p for p in predictions if all(p["methods"][a]["prediction"] != p["label"] for a in ARMS)]
    priority = []
    for label in ("stationary", "advance"):
        pool = sorted((p for p in both_wrong if p["label"] == label), key=lambda p: (
            -abs(mean_scores(p)-.5), p["ui_index"]))
        priority.extend(p["ui_index"] for p in pool[:4])
    cases = []
    for p in predictions:
        wid, ui = p["window_id"], p["ui_index"]
        w, a = windows[wid], audits[wid]
        ids = w["history_frame_ids"] + targets[wid]["future_frame_ids"]
        frames = []
        for fid in ids:
            obs = observations[fid]
            frames.append({"frame_id": fid, "step": obs["step"],
                "relative_time_s": obs["observation_available_at_s"]-w["anchor_timestamp_s"],
                "images": {v: {"original": "../../../collected_data/"+obs["images"][v]["path"],
                               "relative_time_s": obs["images"][v]["recorded_timestamp_s"]-w["anchor_timestamp_s"]}
                           for v in ("side", "top")}})
        log_lines = []
        for key, title in (("later_piper_requests_before_last_image", "后续Piper请求"),
                           ("later_elite_submissions_before_last_image", "后续Elite提交")):
            for row in a[key]:
                ts = row["request_time_bracket_s"]
                kind = row.get("elite_command", {0:"retract",1:"hold",2:"feed"}.get(row.get("piper_intent_id")))
                log_lines.append(f"{title} {kind}：+{ts[0]-w['anchor_timestamp_s']:.3f}～+{ts[1]-w['anchor_timestamp_s']:.3f}秒")
        for row in a["elite_sent_timestamps_inside_response"]:
            log_lines.append(f"Elite发送后时间记录：+{row['recorded_after_send_at_s']-w['anchor_timestamp_s']:.3f}秒（非精确运动开始）")
        hard = any(p["methods"][arm]["prediction"] != p["label"] for arm in ARMS)
        flipped = p["methods"][ARMS[0]]["prediction"] == p["label"] and p["methods"][ARMS[1]]["prediction"] != p["label"]
        cases.append({"prediction": p, "annotation": snapshot[wid]["human_annotation"],
            "short_episode": ("左" if p["task"] == "left" else "右")+p["episode"].rsplit("_",1)[1],
            "anchor_step": observations[w["anchor_frame_id"]]["step"], "frames": frames,
            "anchor_index": len(w["history_frame_ids"])-1,
            "audit_text": "\n".join(log_lines) if log_lines else "未记录响应图像终点前的后续请求/发送时间；不等于没有先前运动延续。",
            "hard": hard, "flipped": flipped, "priority": ui in priority,
            "priority_rank": priority.index(ui) if ui in priority else None})
    summary = {"schema": "real10_pre_action_hardcase_review_v1", "samples": len(cases),
               "both_wrong": len(both_wrong), "any_wrong": sum(c["hard"] for c in cases),
               "newly_wrong": sum(c["flipped"] for c in cases), "priority_ui_indices": priority,
               "selection": "top four per human label among both-wrong, by mean score distance from0.5; not confidence or label-error probability",
               "display": "original local images; prediction/annotation hidden until reveal",
               "source_labels_modified": False, "read_only": True, "cases": cases}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "cases.json", summary)
    data = json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c")
    page = PAGE.replace("__DATA__", data).replace("__LABELS__", json.dumps(LABEL_ZH, ensure_ascii=False))
    (args.out / "index.html").write_text(page, encoding="utf-8")
    lines = ["# 动作前预测难例：只读人工复核", "", "模型分歧不构成标注错误证据。先独立看原始双视角回放，再展开标签/模型结果。",
             "运动响应仅看锚点及后续；历史运动不算。模糊不自动等于不可见；不要依赖feed/hold或日志猜标签。", "",
             f"共{len(cases)}窗：两组都错{len(both_wrong)}窗，新增错{sum(c['flipped'] for c in cases)}窗。",
             "优先8窗："+"、".join(f"#{ui}" for ui in priority)+"；另查看新增错例#62、#77。", "",
             "| 编号 | 任务 | 标注快照 | 仅观测分数 | 加请求分数 | 类型 |", "| --- | --- | --- | ---: | ---: | --- |"]
    for c in cases:
        if c["hard"]:
            p = c["prediction"]
            lines.append(f"| #{p['ui_index']} | {c['short_episode']} | {LABEL_ZH[p['label']]} | {p['methods'][ARMS[0]]['score']:.4f} | {p['methods'][ARMS[1]]['score']:.4f} | {'加入请求后错' if c['flipped'] else '两组都错'} |")
    lines.extend(["", "分数不是置信度。全部45窗含一致窗口均可回放；错误筛选清单不能估计总体标注错误率。",
                  "若修订，请提供编号、新标签、证据视角与可见依据；只根据图像证据修订，不为提高模型分数改标签。",
                  "本页不保存任何源标注；旧实验保留旧快照，后续修订应另版本记录，不能覆盖或包装为新独立验证。"])
    (args.out / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "cases"}, ensure_ascii=False))


def mean_scores(p):
    return sum(p["methods"][arm]["score"] for arm in ARMS)/len(ARMS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("simulation_output/real10_pre_action_response_v1"))
    parser.add_argument("--interface", type=Path, default=Path("simulation_output/real10_action_effect_interface_v1"))
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pre_action_response_v1/hardcase_review"))
    build(parser.parse_args())
