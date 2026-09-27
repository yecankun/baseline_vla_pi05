"""One locked, unused-seed DS0/residual candidate experiment, not a policy rollout.

Prepare inventories prior seed metadata without environment execution. Run uses
one ACT prefix and one shared set of native candidate outcomes for both scorers.
No training, tuning, future-conditioned anchor selection or automatic extension.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

if __package__:
    from . import check_pusht_wm_action_ranking as replay
    from . import pusht_direct_action_scorer as ds
    from . import pusht_object_residual_dynamics as residual
    from . import pusht_object_goal as vision
    from .compare_pusht_direct_scorer_ranking import dice
else:
    import check_pusht_wm_action_ranking as replay
    import pusht_direct_action_scorer as ds
    import pusht_object_residual_dynamics as residual
    import pusht_object_goal as vision
    from compare_pusht_direct_scorer_ranking import dice

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation_output/pusht_fresh_scorer_pair20_v1"
common, baseline, inference = replay.common, replay.baseline, replay.inference
PLAN = {
    "schema": "pusht_fresh_scorer_pair20_v1", "seeds": list(range(300000, 300020)),
    "anchors": [0, 80, 160], "nominal_cap": 168, "horizon": 8, "candidates": 5, "offset_xy": 8.0,
    "maximum_contexts": 60, "maximum_candidate_continuations": 300,
    "maximum_environment_steps": {"nominal": 3360, "replay": 24000, "candidate": 2400, "total": 29760},
    "checkpoint_selection": "frozen_ACT_final100000_DS0_and_residual_final10000",
    "goal": "unchanged_training_episode1_frame117_global278",
    "context_selection": "ACT_only_prefix_fixed_anchors_no_outcome_or_scorer_conditioning",
    "branching": "native_reset_then_recorded_prefix_no_hidden_state_injection",
    "score_persistence": "both_predictions_saved_before_nominal_or_candidate_future_at_each_anchor",
    "primary": "paired_seed_macro_coverage_gain_DS0_minus_residual_at8_native_steps",
    "secondary": "context_gain_regret_harm_worst_loss_headroom_top1_pairs_object_goal_metrics",
    "eligibility": "current_grid_valid_all5_native_chunks_valid_all5_complete8_steps",
    "object_population": "coverage_eligible_and_all5_terminal_objects_visible",
    "coverage_population_not_filtered_by_future_object_visibility": True,
    "invalid_or_early_done": "report_exclusions_no_replacement_no_clipping_no_padding_after_done",
    "cost_tie_epsilon": 1e-7, "coverage_tie_epsilon": 1e-6,
    "uniform": "analytical_expectation", "oracles": "post_prediction_diagnostic_only",
    "statistics": "context_means_seed_means_sample_SD_and_per_seed_values_no_independent_candidate_claim",
    "visual_selection": "first_seed_all_planned_anchors_and_candidates_not_outcome_selected",
    "seed_novelty_scope": "recorded_project_PushT_evaluation_metadata_not_unknown_dataset_generation_seeds",
    "new_training_steps": 0, "hardware_actions": 0, "selector_closed_loop": False,
    "new_weights_gate_or_tuning": False, "automatic_extension": False,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(out):
    """Conservatively include training RNG seeds too; inspect metadata, not outcomes for selection."""
    records, all_seeds, resets = [], set(), set()

    def walk(value, found):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "seed" or key.endswith("_seed"):
                    if isinstance(child, int):
                        found.add(child)
                if "seeds" in key and isinstance(child, list):
                    found.update(x for x in child if isinstance(x, int))
                if key == "per_seed" and isinstance(child, dict):
                    found.update(int(x) for x in child if x.isdigit())
                if key == "reset_observation_sha256" and isinstance(child, str):
                    resets.add(child)
                walk(child, found)
        elif isinstance(value, list):
            for child in value:
                walk(child, found)

    for folder in sorted((ROOT / "simulation_output").iterdir()):
        if not folder.is_dir() or "pusht" not in folder.name.lower() or folder.resolve() == out.resolve():
            continue
        for path in sorted(folder.rglob("*.json")):
            if path.name not in ("report.json", "started.json", "protocol.json", "manifest.json"):
                continue
            found = set()
            walk(read(path), found)
            all_seeds.update(found)
            records.append({"path": str(path.relative_to(ROOT)), "seed_ids": sorted(found)})
    all_seeds.update(baseline.PROTOCOL["benchmark_seeds"] + baseline.PROTOCOL["smoke_seeds"] + replay.PLAN["development_seeds"])
    overlap = set(PLAN["seeds"]) & all_seeds
    if overlap:
        raise ValueError(f"proposed fixed seeds already appear in prior metadata: {sorted(overlap)}")
    return {"metadata_files": records, "prior_seed_ids": sorted(all_seeds), "overlap": [],
        "recorded_prior_reset_ids": sorted(resets), "scope": PLAN["seed_novelty_scope"]}


def checkpoint_identity():
    paths = {"DS0": ds.TRAIN_ROOT / "final.pt", "residual": residual.TRAIN_ROOT / "final.pt"}
    reports = {name: read(path.parent / "report.json") for name, path in paths.items()}
    if reports["DS0"]["cache_identity"] != reports["residual"]["cache_identity"]:
        raise ValueError("DS0/residual training data, split, goal or normalization differs")
    result = {}
    for name, path in paths.items():
        if reports[name]["optimizer_steps"] != 10000:
            raise ValueError("requires completed fixed-final10000 training")
        stat = path.stat()
        result[name] = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                        "schema": reports[name]["schema"]}
    return result


def prepare(out):
    used, identities = inventory(out), checkpoint_identity()
    out.mkdir(parents=True, exist_ok=False)
    common._write(out / "protocol.json", {"plan": PLAN, "checkpoint_identity": identities})
    common._write(out / "seed_inventory.json", used)
    common._write(out / "status.json", {"status": "prepared_no_environment_or_model_execution", "environment_steps": 0})
    print(json.dumps({"status": "prepared", "metadata_files": len(used["metadata_files"]),
        "prior_seed_ids": used["prior_seed_ids"], "fixed_new_seeds": PLAN["seeds"],
        "maximum_environment_steps": PLAN["maximum_environment_steps"], "optimizer_steps": 0}), flush=True)


@torch.inference_mode()
def nominal_prefix(env, runtime, act, prior, models, goal, seed, out, counter, reset_ids):
    act.reset()
    raw = replay.reset(env, seed, runtime)
    reset_id = baseline.observation_hash(raw)  # One relevant reset identity per seed; no per-step hashing.
    if reset_id in reset_ids:
        raise ValueError("new reset duplicates a recorded prior or current-study reset")
    reset_ids.add(reset_id)
    common._append(out / "resets.jsonl", {"seed": seed, "reset_observation_sha256": reset_id})
    raws, actions, metrics, contexts = [replay.clone_observation(raw)], [], [], []
    for t in range(PLAN["nominal_cap"]):
        obs = inference.observation_from_native(raw, runtime["preprocess_observation"])
        if t in PLAN["anchors"]:
            if act.action_calls != t or len(act.policy._action_queue):
                raise ValueError("anchor must be an ACT queue boundary")
            candidates = prior.candidates(obs, offset_xy=PLAN["offset_xy"])
            seen = vision.observe_rgb(raw["pixels"])
            flags = np.array([seen.valid], dtype=bool)
            direct = ds.predict_and_score(models["DS0"], seen.features, flags, obs["observation.state"], candidates)
            predicted = residual.predict_and_score(models["residual"], seen.features, flags,
                obs["observation.state"], candidates, vision.ObjectGridFeatures(goal[None]))
            finite = lambda x: [float(v) if np.isfinite(v) else None for v in x]
            current_cost = float(dice(seen.features.values[0].astype(float), goal.astype(float))) if seen.valid else None
            row = {"seed": seed, "anchor_step": t, "current_valid": seen.valid,
                "valid": candidates.valid[0].tolist(), "actions": candidates.actions[0].cpu().numpy(),
                "agent_xy": raw["agent_pos"].tolist(), "current_goal_cost": current_cost,
                "DS0_effect": finite(direct.scores[0]), "residual_costs": finite(predicted.costs[0]),
                "selected": {"DS0": int(direct.selected_index[0]), "residual": int(predicted.selected_index[0])},
                "current_rgb": replay.save_rgb(out, f"{seed}_{t}_current", raw)}
            contexts.append(row)
            common._append(out / "predictions.jsonl", {**row, "actions": row["actions"].tolist()})
        action = act.select_action(obs).cpu().numpy().copy()
        if contexts and t - contexts[-1]["anchor_step"] < 8:
            if not np.array_equal(action[0], contexts[-1]["actions"][0, t - contexts[-1]["anchor_step"]]):
                raise ValueError("reference candidate differs from original queued ACT action")
        raw, measured = replay.step(env, action, counter, "nominal")
        actions.append(action)
        metrics.append(measured)
        raws.append(replay.clone_observation(raw))
        common._append(out / "nominal_prefix.jsonl", {"seed": seed, "step": t + 1, "action": action[0].tolist(), **measured})
        if measured["terminated"] or measured["truncated"]:
            break
    skipped = [{"seed": seed, "anchor_step": a, "reason": "nominal_ended_before_anchor", "nominal_steps": len(actions)}
        for a in PLAN["anchors"] if a not in {c["anchor_step"] for c in contexts}]
    return {"raws": raws, "actions": actions, "metrics": metrics, "contexts": contexts, "skipped": skipped}


def branch(env, runtime, seed, context, nominal, candidate, goal, out, counter):
    anchor = context["anchor_step"]
    raw = replay.reset(env, seed, runtime)
    replay.same_observation(raw, nominal["raws"][0], "reset")
    for t, action in enumerate(nominal["actions"][:anchor]):
        raw, measured = replay.step(env, action, counter, "replay")
        replay.same_observation(raw, nominal["raws"][t + 1], "shared prefix")
        if measured["terminated"] or measured["truncated"]:
            raise ValueError("replayed prefix ended before anchor")
    for h, action in enumerate(context["actions"][candidate]):
        raw, measured = replay.step(env, action[None], counter, "candidate")
        if candidate == 0:
            replay.same_observation(raw, nominal["raws"][anchor + h + 1], "reference continuation")
            expected = nominal["metrics"][anchor + h]
            if any(measured[k] != expected[k] for k in ("terminated", "truncated")) or abs(measured["coverage"] - expected["coverage"]) > 1e-12:
                raise ValueError("reference outcome differs from nominal ACT")
        common._append(out / "candidate_steps.jsonl", {"seed": seed, "anchor_step": anchor, "candidate": candidate,
            "horizon_step": h + 1, "action": action.tolist(), **measured})
        if measured["terminated"] or measured["truncated"]:
            break
    seen = vision.observe_rgb(raw["pixels"])
    result = {"seed": seed, "anchor_step": anchor, "candidate": candidate, "steps": h + 1,
        "full_horizon": h + 1 == 8, "terminal_coverage": measured["coverage"],
        "terminated": measured["terminated"], "truncated": measured["truncated"],
        "terminal_object_valid": seen.valid,
        "terminal_object_cost": float(dice(seen.features.values[0].astype(float), goal.astype(float))) if seen.valid else None,
        "terminal_rgb": replay.save_rgb(out, f"{seed}_{anchor}_candidate{candidate}", raw)}
    common._append(out / "outcomes.jsonl", result)
    return result


def analyze(contexts, outcomes):
    by_key = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in outcomes}
    joined, exclusions = [], []
    for c in contexts:
        targets = [by_key.get((c["seed"], c["anchor_step"], k)) for k in range(5)]
        reasons = []
        if not c["current_valid"]:
            reasons.append("invalid_current_object")
        if not all(c["valid"]):
            reasons.append("not_all_five_candidates_native_valid")
        if any(t is None or not t["full_horizon"] for t in targets):
            reasons.append("not_all_five_complete8_step_futures")
        if reasons:
            exclusions.append({"seed": c["seed"], "anchor_step": c["anchor_step"], "reasons": reasons})
            continue
        joined.append({"context": c, "targets": targets})
    result = {"planned_contexts": 60, "available_contexts": len(contexts), "coverage_exclusions": exclusions,
        "coverage_eligible_contexts": len(joined), "new_seed_generalization_not_closed_loop_success": True}
    for metric, epsilon in (("coverage", 1e-6), ("object", 1e-7)):
        group = joined if metric == "coverage" else [r for r in joined if all(t["terminal_object_valid"] for t in r["targets"])]
        rows, pair_counts = [], {name: dict.fromkeys(("comparable", "correct", "wrong", "estimated_tie", "target_tie"), 0) for name in ("DS0", "residual")}
        for r in group:
            c, targets = r["context"], r["targets"]
            utility = np.array([t["terminal_coverage"] if metric == "coverage" else -t["terminal_object_cost"] for t in targets])
            weights = {"ACT": np.eye(5)[0], "uniform": np.full(5, .2),
                **{name: np.eye(5)[c["selected"][name]] for name in ("DS0", "residual")},
                "oracle": np.eye(5)[utility.argmax()]}
            informative = bool(np.ptp(utility) > epsilon)
            row = {"seed": c["seed"], "anchor_step": c["anchor_step"], "informative": informative,
                "headroom": float(utility.max() - utility[0]), "strategies": {}}
            for name, w in weights.items():
                gain = utility - utility[0]
                row["strategies"][name] = {"gain": float(w @ utility - utility[0]), "regret": float(utility.max() - w @ utility),
                    "harm": float(w @ (gain < -epsilon)), "better": float(w @ (gain > epsilon)), "tie": float(w @ (np.abs(gain) <= epsilon)),
                    "worst_supported_gain": float(gain[w > 0].min()), "top1": float(w @ (utility >= utility.max() - epsilon))}
            for name, estimate in (("DS0", -np.array(c["DS0_effect"])), ("residual", np.array(c["residual_costs"]))):
                pairs = replay.pair_agreement(estimate, -utility, 1e-7, epsilon)
                for k in pair_counts[name]:
                    pair_counts[name][k] += pairs[k]
            rows.append(row)
        methods = {}
        for name in ("ACT", "uniform", "DS0", "residual", "oracle"):
            per_seed = {str(s): {k: float(np.mean([r["strategies"][name][k] for r in rows if r["seed"] == s])) for k in ("gain", "regret", "harm")}
                for s in PLAN["seeds"] if any(r["seed"] == s for r in rows)}
            info = [r for r in rows if r["informative"]]
            methods[name] = {"context_mean": {k: float(np.mean([r["strategies"][name][k] for r in rows])) if rows else None for k in ("gain", "regret", "harm")},
                "seed_macro": {k: float(np.mean([v[k] for v in per_seed.values()])) if per_seed else None for k in ("gain", "regret", "harm")},
                "per_seed": per_seed, "worst_supported_gain": min((r["strategies"][name]["worst_supported_gain"] for r in rows), default=None),
                "better_tie_harm_count_or_uniform_expectation": {k: float(sum(r["strategies"][name][k] for r in rows)) for k in ("better", "tie", "harm")},
                "top1_informative_only": float(np.mean([r["strategies"][name]["top1"] for r in info])) if info else None}
        seeds = list(methods["DS0"]["per_seed"])
        deltas = np.array([methods["DS0"]["per_seed"][s]["gain"] - methods["residual"]["per_seed"][s]["gain"] for s in seeds])
        result[metric] = {"eligible_contexts": len(rows), "eligible_seeds": len(seeds),
            "missing_seeds": [s for s in PLAN["seeds"] if str(s) not in seeds],
            "informative_contexts": sum(r["informative"] for r in rows),
            "informative_seeds": len({r["seed"] for r in rows if r["informative"]}),
            "methods": methods, "per_context": rows, "pair_agreement": {name: {**p, "accuracy": p["correct"] / p["comparable"] if p["comparable"] else None} for name, p in pair_counts.items()},
            "paired_seed_delta": {"mean": float(deltas.mean()) if len(deltas) else None, "sample_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
                "better_tie_worse": [int((deltas > epsilon).sum()), int((np.abs(deltas) <= epsilon).sum()), int((deltas < -epsilon).sum())],
                "values": dict(zip(seeds, deltas.tolist()))}}
    result["object_only_exclusions"] = [{"seed": r["context"]["seed"], "anchor_step": r["context"]["anchor_step"], "reason": "one_or_more_terminal_objects_not_visible"}
        for r in joined if not all(t["terminal_object_valid"] for t in r["targets"])]
    return result


def render(out, contexts, outcomes):
    title, font = (ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", n) for n in (24, 17))
    sheet = Image.new("RGB", (1280, 1110), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 12), "新 seed 共用候选对照：固定首个 seed=300000，全部预定决策点", font=title, fill="black")
    draw.text((14, 50), "均为实际 RGB；同一 ACT 前缀与相同五候选。DS0 / 残差只评分，不驱动后续前缀。", font=font, fill="black")
    by_key = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in outcomes}
    for n, anchor in enumerate(PLAN["anchors"]):
        top = 95 + 323 * n
        c = next((r for r in contexts if r["seed"] == 300000 and r["anchor_step"] == anchor), None)
        if c is None:
            draw.text((14, top), f"前缀 {anchor}：原 ACT 已结束，没有补选状态。", font=title, fill="black")
            continue
        draw.text((14, top), f"前缀 {anchor}；DS0 选 {c['selected']['DS0']}，残差选 {c['selected']['residual']}；观测有效={c['current_valid']}", font=title, fill="black")
        for col in range(6):
            left = 14 + 210 * col
            target = by_key.get((300000, anchor, col - 1)) if col else None
            path = c["current_rgb"] if col == 0 else target["terminal_rgb"] if target else None
            draw.text((left, top + 39), "当前输入" if col == 0 else f"候选 {col-1}" + (" / ACT" if col == 1 else ""), font=font, fill="black")
            if path:
                with Image.open(out / path) as im:
                    sheet.paste(im.convert("RGB").resize((176, 176)), (left, top + 68))
            if target:
                draw.text((left, top + 253), f"覆盖率 {target['terminal_coverage']:.5f}", font=font, fill="black")
                draw.text((left, top + 279), f"已执行 {target['steps']}/8 步", font=font, fill="black")
    draw.text((14, 1071), "仅短时候选效应验证，不是成功率测试；未来后果仅用于离线评价；尚未通过用户视觉验收。", font=font, fill="black")
    sheet.save(out / "first_seed_shared_candidates_zh.png")


def run(out):
    locked = read(out / "protocol.json")
    if locked != {"plan": PLAN, "checkpoint_identity": checkpoint_identity()}:
        raise ValueError("prepared plan/checkpoint identity changed")
    used = inventory(out)
    if (out / "report.json").exists() or (out / "predictions.jsonl").exists():
        raise ValueError("do not overwrite or automatically rerun an attempted study")
    (out / "observations").mkdir()
    counter = {"nominal": 0, "replay": 0, "candidate": 0}
    report = {"status": "running", "plan": PLAN, "checkpoint_identity": locked["checkpoint_identity"],
        "environment_steps_by_kind": counter, "optimizer_steps": 0, "hardware_actions": 0,
        "seeds_completed": [], "skipped_anchors": [], "seed_inventory_rechecked_before_execution": True}
    common._write(out / "started.json", report)
    started, vector, contexts, outcomes = time.monotonic(), None, [], []
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        runtime = baseline.imports()
        sources, reference = baseline.source_binding(runtime)
        act, binding = inference.load_final("act")
        prior = replay.FrozenACTPrior(act, binding)
        models = {"DS0": ds.load_final(ds.TRAIN_ROOT / "final.pt"), "residual": residual.load_final(residual.TRAIN_ROOT / "final.pt")}
        goal = models["DS0"].goal.cpu().numpy()
        config, vector = runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
        env = vector.envs[0]
        common.validate_env_kwargs(config.gym_kwargs)
        parity = {"environment_gym_kwargs": config.gym_kwargs,
            "environment_step_source_sha256": common.canonical_hash(inspect.getsource(type(env.unwrapped).step)),
            "package_versions": runtime["_package_versions"]()}
        common.verify_environment_parity(reference, parity)
        pre, post = runtime["make_env_pre_post_processors"](config, act.policy.config)
        if pre.steps or post.steps:
            raise ValueError("native environment processors must be identity")
        report.update({"ACT_binding": binding, "source_binding": sources, "environment_parity": parity})
        reset_ids = set(used["recorded_prior_reset_ids"])
        for seed in PLAN["seeds"]:
            nominal = nominal_prefix(env, runtime, act, prior, models, goal, seed, out, counter, reset_ids)
            contexts.extend(nominal["contexts"])
            report["skipped_anchors"].extend(nominal["skipped"])
            for context in nominal["contexts"]:
                for k, valid in enumerate(context["valid"]):
                    if valid:
                        outcomes.append(branch(env, runtime, seed, context, nominal, k, goal, out, counter))
            report["seeds_completed"].append(seed)
            common._write(out / "status.json", {"status": "running", "seeds_completed": report["seeds_completed"], "environment_steps_by_kind": counter})
            print(json.dumps({"seed": seed, "contexts": len(nominal["contexts"]), "total_environment_steps": sum(counter.values())}), flush=True)
        if len(contexts) + len(report["skipped_anchors"]) != 60 or sum(counter.values()) > 29760:
            raise ValueError("frozen context or environment budget exceeded")
        report["analysis"] = analyze(contexts, outcomes)
        report.update({"status": "completed_fresh_seed_candidate_comparison", "predictions_saved_before_futures": True,
            "all_replayed_observations_exact": True, "reference_continuations_exact": True,
            "candidate_continuations": len(outcomes), "rejected_candidate_chunks": sum(not v for c in contexts for v in c["valid"]),
            "no_posthoc_replacements": True, "new_weights_or_threshold_written": False,
            "policy_success_rate_measured": False, "model_promotion": False,
            "visual_status": "not_viewed", "decision": "inspect_frozen_new_seed_evidence_no_automatic_tuning_or_policy_switch"})
        render(out, contexts, outcomes)
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "partial_outputs_preserved": True})
    finally:
        if vector is not None:
            vector.close()
        report["environment_steps"] = sum(counter.values())
        report["runtime_seconds"] = time.monotonic() - started
        common._write(out / "report.json", report)
        common._write(out / "status.json", {"status": report["status"], "seeds_completed": report["seeds_completed"], "environment_steps_by_kind": counter})
    print(json.dumps({k: report[k] for k in ("status", "environment_steps", "runtime_seconds", "error") if k in report}), flush=True)
    if report["status"] == "failed":
        print(report["traceback"], flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(OUT)
        return 0
    return run(OUT)


if __name__ == "__main__":
    raise SystemExit(main())
