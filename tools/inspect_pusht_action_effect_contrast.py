"""Summarize saved paired validation predictions; no model or environment calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("pointwise", "pointwise_plus_effect_difference")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def inspect(readback, data, out):
    report = read(readback / "report.json")
    if report["status"] != "completed_fixed_paired_training_and_validation":
        raise ValueError("completed fixed paired training is required")
    seeds = report["contract"]["protocol"]["paired_training_seeds"]
    contexts = rows(data / "validation/pack/contexts.jsonl")
    keys = np.array([(r["source_seed"], r["anchor_step"]) for r in contexts])
    with np.load(data / "validation/pack/targets.npz", allow_pickle=False) as saved:
        target = saved["terminal_coverage"].copy()
    selected, predicted, gains = {}, {}, {}
    metrics = {arm: [] for arm in ARMS}
    source_delta = []
    for seed in seeds:
        validation = report["validation"][str(seed)]
        with np.load(readback / str(seed) / "validation_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(keys, saved["context_keys"]):
                raise ValueError("saved prediction rows do not match the original validation sources")
            for arm in ARMS:
                score, choice = saved[arm + "_scores"], saved[arm + "_selected"]
                if not np.isfinite(score).all() or not np.array_equal(choice, score.argmax(1)):
                    raise ValueError("saved argmax selection changed")
                predicted[seed, arm], selected[seed, arm] = score.copy(), choice.copy()
                gains[seed, arm] = target[np.arange(len(keys)), choice] - target[:, 0]
                method = validation["metrics"]["methods"][arm]
                measured = method["context_mean"]
                if not (np.isclose(gains[seed, arm].mean(), measured["gain_vs_ACT"], atol=1e-14, rtol=0)
                        and np.isclose(np.abs(score - target).mean(), measured["coverage_mae"], atol=1e-14, rtol=0)):
                    raise ValueError("saved choices/errors do not reproduce the training report")
                metrics[arm].append({**method["source_seed_macro"],
                    "informative_top1_context": method["informative_top1"]["context_mean"],
                    "informative_top1_source": method["informative_top1"]["source_seed_macro"],
                    "pair_accuracy_pooled": validation["metrics"]["pair_agreement"][arm]["pooled_pairs"]["accuracy"],
                    "pair_accuracy_source": validation["metrics"]["pair_agreement"][arm]["source_seed_macro_accuracy"],
                    "training_seconds": report["training"][str(seed)]["training_seconds_by_arm"][arm],
                    "warm_forward_ms": validation["warm_scoring_ms_one_context_five_candidates"][arm]})
        source_delta.append(validation["metrics"]["paired_minus_pointwise"]["per_source_seed"])
    mean = {arm: {k: float(np.mean([r[k] for r in values])) for k in values[0]}
            for arm, values in metrics.items()}
    delta = {k: mean[ARMS[1]][k] - mean[ARMS[0]][k] for k in mean[ARMS[0]]}
    by_source = {s: float(np.mean([r[s] for r in source_delta])) for s in source_delta[0]}
    positive = sorted((s for s in by_source if by_source[s] > 0), key=by_source.get, reverse=True)
    leading = positive[:2]
    remainder = [v for s, v in by_source.items() if s not in leading]
    # Post-hoc diagnostic: show ALL five constants, never choose a deployment arm
    # or fit a constant on validation and then call it a predeclared baseline.
    fixed = {str(k): {"coverage": float(target[:, k].mean()),
                     "gain_vs_ACT": float((target[:, k] - target[:, 0]).mean()),
                     "harm_rate": float(np.mean(target[:, k] < target[:, 0] - 1e-6)),
                     "regret": float((target.max(1) - target[:, k]).mean())} for k in range(5)}
    # Diagnostic cases are post-hoc, from the FIRST training seed, never a best seed.
    first = seeds[0]
    context_delta = gains[first, ARMS[1]] - gains[first, ARMS[0]]
    improvement = np.flatnonzero(context_delta > 1e-6)
    improvement = improvement[np.argsort(-context_delta[improvement])][:2].tolist()
    cases = improvement + [int(context_delta.argmin())]
    out.mkdir(parents=True, exist_ok=False)
    draw_cases(out, data, contexts, keys, target, selected, predicted, first, cases, mean, fixed)
    result = {
        "status": "completed_saved_validation_readback", "source_report": str(readback / "report.json"),
        "training_seeds": seeds, "validation_sources": len(np.unique(keys[:, 0])), "validation_contexts": len(keys),
        "mean_across_training_seeds": mean, "paired_minus_pointwise": delta,
        "relative_change_vs_pointwise": {k: delta[k] / mean[ARMS[0]][k] if mean[ARMS[0]][k] != 0 else None
                                          for k in ("coverage", "coverage_mae", "regret", "harm_rate")},
        "per_source_mean_paired_delta": by_source,
        "posthoc_concentration": {"top_two_positive_sources": leading,
            "share_of_positive_source_deltas": sum(by_source[s] for s in leading) / sum(by_source[s] for s in positive),
            "mean_delta_without_those_two": float(np.mean(remainder)),
            "sensitivity_only_not_a_new_exclusion_rule": True},
        "posthoc_fixed_candidate_controls": {"all_five_constants": fixed,
            "candidate1_is_native_x_plus_ramp_not_a_new_policy": True,
            "effect_difference_mean_minus_fixed_candidate1": mean[ARMS[1]]["coverage"] - fixed["1"]["coverage"],
            "diagnostic_only_no_validation_selected_policy_promotion": True},
        "selected_candidate_histograms": {str(seed): {arm: np.bincount(selected[seed, arm], minlength=5).tolist()
                                                       for arm in ARMS} for seed in seeds},
        "saved_choices_and_mae_match_report": True,
        "visual_cases": {"training_seed": first, "context_keys": keys[cases].tolist(),
            "selection": "posthoc_first_training_seed_two_largest_improvements_and_worst_regression",
            "visual_status": "not_viewed"},
        "new_optimizer_steps": 0, "new_model_inferences": 0, "new_environment_steps": 0, "hardware_actions": 0,
        "fresh_test": False,
        "decision": "promising_validation_selection_signal_freeze_all_six_no_retraining_no_promotion",
    }
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def draw_cases(out, data, contexts, keys, target, selected, predicted, seed, cases, mean, fixed):
    font_path = next(path for path in (Path("C:/Windows/Fonts/msyh.ttc"),
                     Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")) if path.is_file())
    title, font, small = (ImageFont.truetype(str(font_path), n) for n in (27, 20, 17))
    canvas = Image.new("RGB", (1550, 277 + 323 * len(cases)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "同起点效应对比：固定验证集的动作选择回读", font=title, fill="black")
    draw.text((18, 53), "7 个验证来源 / 21 个上下文；三组训练重复，不是 63 个独立测试场景。", font=font, fill="#333333")
    for j, (arm, label, color) in enumerate(((ARMS[0], "普通评分器", "#2455bb"), (ARMS[1], "效应对比评分器", "#bb3333"))):
        m = mean[arm]
        draw.text((18, 87 + j * 30), f"{label}：覆盖率 {m['coverage']:.6f} | 相对 ACT 增益 {m['gain_vs_ACT']:+.6f} | 有害选择 {m['harm_rate']:.2%}", font=font, fill=color)
    draw.text((18, 151), f"重要限制：事后固定选择候选 1 的覆盖率也有 {fixed['1']['coverage']:.6f}；尚未证明评分收益超越固定方向偏好。", font=font, fill="#883300")
    draw.text((18, 185), f"以下用首个训练 seed={seed}；事后选择两处最大改善与一处最差退步，用于诊断，不代表随机样本。", font=small, fill="#555555")
    for row, index in enumerate(cases):
        top = 222 + 323 * row
        context = contexts[index]
        folder = data / context["source_attempt"]
        outcomes = {(r["anchor_step"], r["candidate"]): r for r in rows(folder / "outcomes.jsonl")}
        choices = [int(selected[seed, arm][index]) for arm in ARMS]
        d = target[index, choices[1]] - target[index, choices[0]]
        draw.text((18, top), f"来源 {keys[index, 0]} / 起点 {keys[index, 1]}：普通选 {choices[0]}，对比选 {choices[1]}；实际覆盖率差 {d:+.6f}", font=font, fill="black")
        for col in range(6):
            left, k = 18 + col * 255, col - 1
            path = context["current_rgb"] if col == 0 else outcomes[keys[index, 1], k]["terminal_rgb"]
            draw.text((left, top + 32), "当前观测" if col == 0 else f"候选 {k}" + (" / ACT" if k == 0 else ""), font=small, fill="black")
            with Image.open(folder / path) as image:
                canvas.paste(image.convert("RGB").resize((160, 160)), (left, top + 61))
            if col:
                for pick, color, pad in ((choices[0], "#2455bb", 2), (choices[1], "#bb3333", 5)):
                    if pick == k:
                        draw.rectangle((left - pad, top + 61 - pad, left + 159 + pad, top + 220 + pad), outline=color, width=2)
                draw.text((left, top + 235), f"实际 coverage {target[index, k]:.6f}", font=small, fill="black")
                for j, (arm, label, color) in enumerate(((ARMS[0], "普通预测", "#2455bb"), (ARMS[1], "对比预测", "#bb3333"))):
                    draw.text((left, top + 258 + 23 * j), f"{label} {predicted[seed, arm][index, k]:+.6f}", font=small, fill=color)
    draw.text((18, canvas.height - 34), "图像均为已保存的真实仿真执行后果；不是模型生成图像，不是闭环或实机成功率。测试组尚未执行。", font=small, fill="#333333")
    canvas.save(out / "validation_selection_readback_zh.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readback", type=Path, default=ROOT / "simulation_output/pusht_action_effect_contrast_training_readback_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "simulation_output/pusht_action_effect_pairs_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_action_effect_contrast_inspection_v1")
    args = parser.parse_args()
    inspect(args.readback, args.data, args.out)
