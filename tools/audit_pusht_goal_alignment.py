"""Post-hoc goal/ranking audit of saved RGBs and frozen scorer decisions.

No checkpoint, model, environment or training import. All future images and
coverage values are retrospective diagnostics, never policy observations.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pusht_object_goal as vision

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_fresh_scorer_pair20_v1"
OUT = ROOT / "simulation_output/pusht_goal_alignment_audit_v1"
GOAL_CACHE = Path("/media/zsw/SSD1T/project_2026_weights_v1/features/pusht_object_dynamics_v1")
EPS, CEPS = 1e-7, 1e-6
PLAN = {
    "schema": "pusht_goal_alignment_audit_v1",
    "source": str(SOURCE), "population": "same57_previously_eligible_contexts_20_seeds",
    "goal": "unchanged_training_episode1_frame117_global278_gray_object_RGB",
    "selection": "no_context_exclusion_or_candidate_change",
    "decomposition": "Cmax-Cmodel=(Cmax-CobjectOracle)+(CobjectOracle-Cmodel)",
    "signed_term": "objectOracle_minus_model_coverage_can_be_negative_not_causal_error_fraction",
    "proxy_ties": "deterministic_exact_argmin_plus_full_1e-7_near_optimal_set",
    "resolution_sensitivity": "same_RGB_mask_and_goal_96binary_Dice_vs_existing24pooled_Dice_only",
    "resolution_sensitivity_is_posthoc": True,
    "cases": [[300001, 160], [300007, 80], [300010, 160], [300005, 160]],
    "case_selection": "three_previously_reported_proxy_harms_and_previous_worst_DS0_harm",
    "model_forward_calls": 0, "optimizer_steps": 0, "environment_steps": 0,
    "hardware_actions": 0, "checkpoint_loads": 0,
    "no_new_training_target_goal_gate_or_threshold": True,
    "not_independent_test_or_policy_comparison": True,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def dice(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.square(a - b).sum() / (np.square(a).sum() + np.square(b).sum()))


def aggregate(rows, fields):
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {str(s): {k: float(np.mean([r[k] for r in rows if r["seed"] == s])) for k in fields} for s in seeds}
    return {"context_mean": {k: float(np.mean([r[k] for r in rows])) for k in fields},
            "seed_macro": {k: float(np.mean([v[k] for v in per_seed.values()])) for k in fields}, "per_seed": per_seed}


def proxy_audit(costs, coverage):
    chosen = int(np.argmin(costs))
    near = costs <= costs.min() + EPS
    best = float(coverage.max())
    return {"choice": chosen, "near_optimal_ids": np.flatnonzero(near).tolist(),
            "coverage_gain": float(coverage[chosen] - coverage[0]),
            "coverage_regret": best - float(coverage[chosen]),
            "tie_robust_regret_lower_bound": best - float(coverage[near].max()),
            "near_optimal_best_gain": float(coverage[near].max() - coverage[0]),
            "near_optimal_worst_gain": float(coverage[near].min() - coverage[0]),
            "harm": bool(coverage[chosen] < coverage[0] - CEPS),
            "all_near_optimal_harm": bool(coverage[near].max() < coverage[0] - CEPS)}


def pair_counts(costs, coverage):
    counts = {"coverage_tie": 0, "agree": 0, "opposite": 0, "proxy_tie": 0}
    for i in range(5):
        for j in range(i + 1, 5):
            dc, dd = coverage[i] - coverage[j], costs[i] - costs[j]
            label = ("coverage_tie" if abs(dc) <= CEPS else "proxy_tie" if abs(dd) <= EPS
                     else "agree" if dc * dd < 0 else "opposite")
            counts[label] += 1
    return counts


def render(rows, pred, goals):
    title, font, small = (ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n) for n in (25, 19, 16))
    im = Image.new("RGB", (1540, 1410), "white")
    d = ImageDraw.Draw(im)
    d.text((15, 12), "目标对齐审查：视觉示范目标不等于环境覆盖率；全部未来图像均为事后观测", font=title, fill="black")
    im.paste(Image.fromarray(goals).resize((132, 132)), (15, 52))
    d.text((165, 60), "原目标：训练 episode 1 / frame 117 的灰色 T，不是环境精确目标几何。", font=font, fill="black")
    d.text((165, 96), "案例：上轮已报告的三处视觉目标伤害 + DS0 最大伤害；未按本次审查结果换例。", font=font, fill="black")
    d.text((165, 132), "模型/动作/目标/阈值未改。下列绿色区域为画面目标，灰色为物体；未来真值不进入策略。", font=font, fill="black")
    keyed = {(r["seed"], r["anchor_step"]): r for r in rows}
    for n, key in enumerate(PLAN["cases"]):
        key = tuple(key)
        row, p = keyed[key], pred[key]
        top = 205 + n * 284
        d.text((15, top), f"seed {key[0]} / 前缀 {key[1]}：原目标最佳 {row['proxy24']['choice']}；覆盖率最佳 {row['coverage_choice']}；96像素目标最佳 {row['proxy96']['choice']}", font=font, fill="black")
        roles = [("当前观测", None), ("ACT", 0), ("真实视觉目标最优", row["proxy24"]["choice"]),
                 ("真实覆盖率最优", row["coverage_choice"]), ("DS0 选择", p["selected"]["DS0"]),
                 ("残差选择", p["selected"]["residual"])]
        for col, (label, k) in enumerate(roles):
            left = 15 + col * 254
            d.text((left, top + 33), label if k is None else f"{label} / {k}", font=small, fill="black")
            path = p["current_rgb"] if k is None else row["terminal_rgb"][k]
            im.paste(Image.fromarray(rgb(SOURCE / path)).resize((174, 174)), (left, top + 61))
            if k is not None:
                d.text((left, top + 239), f"覆盖率 {row['coverage'][k]:.6f}", font=small, fill="black")
                d.text((left, top + 260), f"视觉代价 {row['cost24'][k]:.6f}", font=small, fill="black")
    d.text((15, 1370), "仅事后错误分解，不是新成功率或新策略收益；无新增模型推理、训练或环境步。待用户视觉验收。", font=font, fill="black")
    im.save(OUT / "goal_alignment_cases_zh.png")


def main():
    started = time.monotonic()
    source = read(SOURCE / "report.json")
    assert source["status"] == "completed_fresh_seed_candidate_comparison"
    goal_manifest = read(GOAL_CACHE / "manifest.json")["goal"]
    assert [goal_manifest[k] for k in ("episode_index", "frame_index", "global_index", "split")] == [1, 117, 278, "train"]
    goal_rgb = rgb(GOAL_CACHE / "goal_rgb.png")
    goal = vision.observe_rgb(goal_rgb)
    assert goal.valid
    pred = {(r["seed"], r["anchor_step"]): r for r in lines(SOURCE / "predictions.jsonl")}
    outcome = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in lines(SOURCE / "outcomes.jsonl")}
    rows, max_cost_error, max_decomposition_error = [], 0.0, 0.0
    for saved in source["analysis"]["coverage"]["per_context"]:
        key = (saved["seed"], saved["anchor_step"])
        p = pred[key]
        targets = [outcome[(*key, k)] for k in range(5)]
        assert p["current_valid"] and all(p["valid"]) and all(t["full_horizon"] and t["terminal_object_valid"] for t in targets)
        seen = [vision.observe_rgb(rgb(SOURCE / t["terminal_rgb"])) for t in targets]
        assert all(s.valid for s in seen)
        c = np.array([t["terminal_coverage"] for t in targets])
        q = np.array([t["terminal_object_cost"] for t in targets])
        q96 = np.array([dice(s.mask96, goal.mask96) for s in seen])
        error = max(abs(dice(s.features.values[0], goal.features.values[0]) - cost) for s, cost in zip(seen, q))
        max_cost_error = max(max_cost_error, error)
        assert error < 1e-12
        row = {"seed": key[0], "anchor_step": key[1], "coverage": c.tolist(),
               "cost24": q.tolist(), "cost96": q96.tolist(), "proxy24": proxy_audit(q, c),
               "proxy96": proxy_audit(q96, c), "coverage_choice": int(c.argmax()),
               "terminal_rgb": [t["terminal_rgb"] for t in targets], "models": {},
               "pair24": pair_counts(q, c), "pair96": pair_counts(q96, c)}
        oracle = row["proxy24"]["choice"]
        for name in ("DS0", "residual"):
            chosen = p["selected"][name]
            proxy_term = float(c.max() - c[oracle])
            ranking_term = float(c[oracle] - c[chosen])
            total = float(c.max() - c[chosen])
            max_decomposition_error = max(max_decomposition_error, abs(total - proxy_term - ranking_term))
            assert abs(total - saved["strategies"][name]["regret"]) < 1e-12
            assert abs(float(c[chosen] - c[0]) - saved["strategies"][name]["gain"]) < 1e-12
            m = {"choice": chosen, "coverage_gain": float(c[chosen] - c[0]), "coverage_regret": total,
                 "proxy_term": proxy_term, "signed_ranking_term": ranking_term,
                 "true_proxy_regret": float(q[chosen] - q.min()),
                 "harm": bool(c[chosen] < c[0] - CEPS),
                 "chosen_proxy_optimal": bool(q[chosen] <= q.min() + EPS)}
            row["models"][name] = m
        rows.append(row)
    assert len(rows) == 57 and len({r["seed"] for r in rows}) == 20
    summary = {}
    for name in ("proxy24", "proxy96"):
        flattened = [{"seed": r["seed"], **r[name]} for r in rows]
        summary[name] = {**aggregate(flattened, ("coverage_gain", "coverage_regret", "tie_robust_regret_lower_bound")),
                         "harm_cases": [[r["seed"], r["anchor_step"]] for r in rows if r[name]["harm"]],
                         "tie_robust_harm_cases": [[r["seed"], r["anchor_step"]] for r in rows if r[name]["all_near_optimal_harm"]],
                         "regret_cases": [[r["seed"], r["anchor_step"]] for r in rows if r[name]["coverage_regret"] > CEPS]}
    for name in ("DS0", "residual"):
        flattened = [{"seed": r["seed"], **r["models"][name]} for r in rows]
        harmed = [r for r in rows if r["models"][name]["harm"]]
        summary[name] = {**aggregate(flattened, ("coverage_gain", "coverage_regret", "proxy_term", "signed_ranking_term", "true_proxy_regret")),
                        "harm_contexts": len(harmed),
                        "model_proxy_suboptimal_contexts": sum(not r["models"][name]["chosen_proxy_optimal"] for r in rows),
                        "ranking_term_positive_zero_negative": [sum(r["models"][name]["signed_ranking_term"] > CEPS for r in rows),
                            sum(abs(r["models"][name]["signed_ranking_term"]) <= CEPS for r in rows),
                            sum(r["models"][name]["signed_ranking_term"] < -CEPS for r in rows)],
                        "harm_cases": [{"seed": r["seed"], "anchor_step": r["anchor_step"],
                            "model_proxy_optimal": r["models"][name]["chosen_proxy_optimal"],
                            "proxy_oracle_also_harmful": r["proxy24"]["harm"], **r["models"][name]} for r in harmed]}
    summary["resolution_choice_changes"] = sum(r["proxy24"]["choice"] != r["proxy96"]["choice"] for r in rows)
    summary["pairs"] = {name: {k: sum(r[name][k] for r in rows) for k in ("coverage_tie", "agree", "opposite", "proxy_tie")} for name in ("pair24", "pair96")}
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "protocol.json", PLAN)
    with (OUT / "per_context.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    render(rows, pred, goal_rgb)
    report = {"status": "completed_saved_goal_alignment_audit", "plan": PLAN, "goal_source": goal_manifest,
              "contexts": len(rows), "seeds": 20, "actual_future_RGBs_read": 285,
              "max_recomputed_original_cost_error": max_cost_error,
              "max_decomposition_identity_error": max_decomposition_error,
              "original_frozen_gain_and_regret_reproduced": True,
              "summary": summary, "visual_status": "not_viewed", "runtime_seconds": time.monotonic() - started,
              "decision": "inspect_proxy_and_ranking_separately_no_model_or_target_adoption"}
    write(OUT / "report.json", report)
    print(json.dumps({"status": report["status"], "runtime_seconds": report["runtime_seconds"],
                      "proxy24_harms": summary["proxy24"]["harm_cases"], "proxy96_harms": summary["proxy96"]["harm_cases"]}), flush=True)


if __name__ == "__main__":
    main()
