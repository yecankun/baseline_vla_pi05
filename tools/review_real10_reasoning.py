"""Offline evidence packets and scoring for an actual agent's Real10 reviews.

No hardware/model/API imports. This does not generate or impersonate LLM reviews.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "real10_once_demo_20260924_112930",
    "real10_bounded_demo_20260924_114052",
    "real10_bounded_demo_20260924_115218",
    "real10_operator_steps_20260924_120458",
)
OBS_KEYS = ("task", "elite_tcp_pose_6d", "previous_controller_state", "piper_history_validity",
            "camera_host_timestamps", "pose_query_started", "pose_query_finished",
            "distance_from_recorded_home_mm")
PLAN_KEYS = ("raw_elite_tcp_delta_6d", "demo_scaled_elite_tcp_delta_6d", "translation_gain",
             "guarded_elite_tcp_delta_6d", "translation_clipped", "translation_limit_mm",
             "elite_target_tcp_pose_6d", "piper_intent_id", "piper_step_command", "units",
             "elite_action_source")
DECISIONS = ("continue_candidate", "pause_review", "observe_again")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rows():
    index = 0
    for run in RUNS:
        report = read(ROOT / "simulation_output" / run / "report.json")
        steps = report.get("steps", [report])
        for local, step in enumerate(steps, 1):
            index += 1
            yield f"R{index:02d}", run, local, report, step


def reference_alignment(pose, delta, points):
    candidates = []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        v = [y - x for x, y in zip(a, b)]
        vv = sum(x*x for x in v)
        if vv == 0:
            continue
        t = max(0.0, min(1.0, sum((p-x)*d for p,x,d in zip(pose, a, v))/vv))
        distance = sum((p - x - t*d)**2 for p,x,d in zip(pose, a, v))**0.5
        candidates.append((distance, i, v, vv))
    distance, i, v, vv = min(candidates)
    progress = sum(a*b for a,b in zip(delta, v))/math.sqrt(vv)
    norm = math.sqrt(sum(x*x for x in delta))
    return {"segment": i, "distance_to_path_mm": distance,
            "projected_step_mm": progress, "cosine": progress/norm if norm > 0 else None,
            "meaning": "recorded arm-path alignment, not wire progress"}


def rule_review(packet):
    plan = packet["plan"]
    delta = plan["guarded_elite_tcp_delta_6d"]
    reasons = []
    if len(delta) != 6 or not all(math.isfinite(x) for x in delta):
        reasons.append("invalid_delta")
    elif math.sqrt(sum(x*x for x in delta[:3])) > plan["translation_limit_mm"] + 1e-6:
        reasons.append("step_limit_exceeded")
    if plan["piper_intent_id"] not in (1, 2):
        reasons.append("unsupported_retract_or_intent")
    if packet["reference_alignment"]["projected_step_mm"] < -1e-9:
        reasons.append("opposes_recorded_arm_path")
    obs = packet["observation"]
    if plan["piper_intent_id"] == 2 and not obs.get("piper_history_validity", bool(obs.get("previous_controller_state"))):
        reasons.append("feed_history_unverified")
    return {"decision": "pause_review" if reasons else "continue_candidate", "reasons": reasons}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    points = [[float(x) for x in line.split()] for line in (ROOT / "path/path1_pose.txt").read_text().splitlines() if line.strip()]
    packets, history, inventory = [], {}, []
    for cid, run, local, report, step in rows():
        folder = ROOT / "simulation_output" / run
        before = folder / ("before.png" if "steps" not in report else f"before_{local:02d}.png")
        if not before.is_file():
            raise FileNotFoundError(before)
        obs = {k: step["observation"][k] for k in OBS_KEYS if k in step["observation"]}
        plan = {k: step["plan"][k] for k in PLAN_KEYS if k in step["plan"]}
        packet = {"id": cid, "run": run, "step": local, "task": report["task"],
                  "before_image": str(before.relative_to(ROOT)), "observation": obs, "plan": plan,
                  "prior_case_ids_in_run": list(history.get(run, [])),
                  "reference_alignment": reference_alignment(obs["elite_tcp_pose_6d"][:3], plan["guarded_elite_tcp_delta_6d"][:3], points)}
        packet["rule_baseline"] = rule_review(packet)
        packets.append(packet)
        history.setdefault(run, []).append(cid)
        for source in (before, folder / "report.json"):
            st = source.stat()
            inventory.append({"path": str(source.relative_to(ROOT)), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    (out / "packets.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False, allow_nan=False)+"\n" for p in packets), encoding="utf-8")
    write(out / "manifest.json", {"schema": "real10_reasoning_review_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "cases": len(packets), "runs": list(RUNS), "reviewer": "current Codex session; historical conclusions already seen",
          "source_inventory": inventory, "reference": "path/path1_pose.txt", "blinded": False,
          "hardware_connected": False, "policy_inference_executed": False, "optimizer_updates": 0})
    print(json.dumps({"prepared": len(packets), "out": str(out)}, ensure_ascii=False))


def score(out):
    if (out / "report.json").exists():
        raise FileExistsError("completed report exists; preserve this evaluation")
    packets = jsonl(out / "packets.jsonl")
    reviews = jsonl(out / "reviews.jsonl")
    expected = [p["id"] for p in packets]
    if [r["id"] for r in reviews] != expected:
        raise ValueError("reviews must cover every case once, in packet order")
    for r in reviews:
        if r["decision"] not in DECISIONS or not r["evidence"] or not r["visual_observation"] or not r["limitations"]:
            raise ValueError(f"incomplete review: {r['id']}")
    for item in read(out / "manifest.json")["source_inventory"]:
        st = (ROOT / item["path"]).stat()
        if (st.st_size, st.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise ValueError(f"source changed: {item['path']}")
    outcomes = {}
    for cid, run, local, report, step in rows():
        outcomes[cid] = {"move_attempted": bool(step.get("elite_move_attempted")),
                         "target_reached": bool(step.get("elite_target_reached")),
                         "packets_sent": step.get("piper_packets_sent", 0),
                         "physical_feed_confirmed": step.get("piper_physical_execution_confirmed"),
                         "target_error_mm": step.get("elite_target_error_mm"),
                         "observation_age_at_dispatch_s": step.get("observation_age_at_dispatch_s"),
                         "step_status": step.get("status"), "run_stop_reason": report.get("termination_reason"),
                         "task_correctness_label": None, "task_success_label": None}
    count = Counter(r["decision"] for r in reviews)
    rule_equal = sum(p["rule_baseline"]["decision"] == r["decision"] for p,r in zip(packets,reviews))
    report = {"schema": "real10_reasoning_review_v1", "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
              "cases": len(packets), "runs": len(RUNS), "agent_decisions": dict(count),
              "rule_decisions": dict(Counter(p["rule_baseline"]["decision"] for p in packets)),
              "agreement_with_rule_count": rule_equal,
              "historical_move_attempted": sum(o["move_attempted"] for o in outcomes.values()),
              "historical_target_reached": sum(o["target_reached"] for o in outcomes.values()),
              "historical_feed_packets": sum(o["packets_sent"] for o in outcomes.values()),
              "historically_dispatched_but_agent_would_withhold": [r["id"] for r in reviews if r["decision"] != "continue_candidate" and outcomes[r["id"]]["move_attempted"]],
              "agent_continue_but_historical_controller_withheld": [r["id"] for r in reviews if r["decision"] == "continue_candidate" and not outcomes[r["id"]]["move_attempted"]],
              "task_labeled_cases": 0, "false_block_rate": None, "missed_error_rate": None,
              "task_success_improvement": None, "online_latency_validated": False,
              "blinded": False, "counterfactual_rollout_executed": False,
              "visual_status": "not_viewed", "outcomes": outcomes}
    write(out / "report.json", report)
    labels = {"continue_candidate":"继续候选（仍受原控制器检查）", "pause_review":"暂停核查", "observe_again":"补充观察"}
    blocks = []
    for p,r in zip(packets,reviews):
        outcome = outcomes[p["id"]]
        image_path = "../../" + p["before_image"]
        blocks.append(f'<section><h2>{p["id"]} · {html.escape(p["run"])} · 第 {p["step"]} 步</h2>'
                      f'<p><b>{labels[r["decision"]]}</b> ｜ 简单规则：{labels[p["rule_baseline"]["decision"]]}</p>'
                      f'<img src="{html.escape(image_path, quote=True)}" alt="动作前侧视与俯视">'
                      f'<p>视觉观察：{html.escape(r["visual_observation"])}</p><p>审核依据：{html.escape(r["evidence"])}</p>'
                      f'<p>局限：{html.escape(r["limitations"])}</p>'
                      f'<p>事后对照：下发={outcome["move_attempted"]}，到位={outcome["target_reached"]}，递丝包={outcome["packets_sent"]}；任务正确性未知。</p></section>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Real10 推理审核第一版</title><style>body{font:17px/1.65 sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;background:#f5f7fa;color:#172033}section{background:white;padding:20px;margin:24px 0;border:1px solid #ccd4df;border-radius:10px}img{max-width:100%;height:auto}h2{font-size:20px;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style><h1>Real10 推理审核第一版：离线回溯</h1><p>16 个候选、4 次相关现场运行。审核者已知历史总结；不是盲测。没有连接机器人或重新运行模型。</p><p>动作到位不等于任务正确；缺少逐步任务标签，误拦率、漏检率与成功率收益均未建立。后续日志仅作逐步 shadow 审核，不是干预后的反事实轨迹。</p>' + f'<pre>{html.escape(json.dumps({k:v for k,v in report.items() if k!="outcomes"},ensure_ascii=False,indent=2))}</pre>' + ''.join(blocks) + '</html>'
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="outcomes"}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/real10_reasoning_review_v1")
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else score)(args.out)


if __name__ == "__main__":
    main()
