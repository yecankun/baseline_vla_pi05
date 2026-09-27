"""Frozen DP best-of-five diagnostic: prepare, user-run branches, summarize.

One intervention at a nominal DP anchor, then the SAME DP until done/step300.
Oracle outcomes are offline diagnostics, never an executed selector or input.
No training, ACT/real10 changes, old test reuse, or automatic seed extension.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
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
    from . import diffusion_pusht_candidates as adapter
    from . import run_diffusion_pusht_baseline as common
else:
    import diffusion_pusht_candidates as adapter
    import run_diffusion_pusht_baseline as common

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation_output/diffusion_pusht_candidate_headroom_v1"
PLAN = {
    "schema": "diffusion_pusht_candidate_headroom_v1", "data_role": "diagnostic_only",
    "preparation_seed": 499999, "source_seeds": list(range(500000, 500008)),
    "anchors": [80, 160], "candidates": 5, "candidate_action_steps": 8,
    "episode_limit": 300, "n_obs_steps": 2, "diffusion_horizon": 16,
    "ddpm_steps": 100, "reference": "unchanged_official_DP_native_queue_and_RNG",
    "proposal": "reference_plus_four_full_DDPM_samples_same_observation_history",
    "alternative_seed_rule": "10000000 + source_seed*10000 + anchor*10 + candidate_index",
    "continuation": "same_frozen_DP_with_common_post_reference_torch_RNG_until_done_or_global_step300",
    "branching": "native_reset_and_exact_nominal_prefix_replay_no_hidden_state_injection",
    "primary": "complete5_source_macro_oracle_success_minus_reference_success_for_one_intervention",
    "secondary": ["reference_fail_any_candidate_success", "all_valid_candidates_fail",
                  "coverage_at8_or_early_done_headroom", "final_coverage_headroom",
                  "all_fixed_indices", "uniform_expectation", "proposal_diversity_and_validity"],
    "exclusions": "unreached_anchors_not_replaced_invalid_chunks_not_clipped_or_executed",
    "aggregation": "anchors_within_source_then_sources_candidates_not_independent_episodes",
    "statistics": "descriptive_pilot_no_significance_or_power_claim_one_continuation_stream",
    "training_allowed": False, "policy_training_ready": False,
    "oracle_is_deployable": False, "selector_closed_loop": False,
    "optimizer_steps": 0, "hardware_actions": 0,
    "maximum_contexts": 16, "maximum_branches": 80,
    "maximum_run_environment_steps": 26400,
    "maximum_run_model_action_steps": 16800,
    "maximum_additional_proposal_chunks": 64,
    "used_ACT_test_outcomes_accessed": False,
}
# Geometry intersection area can vary by a few float64 ULPs across native resets.
# This is verification tolerance only: keep raw metrics and the experiment fixed.
COVERAGE_REPLAY_ATOL = 1e-12


def check_replay_metrics(actual, expected, *, where):
    exact_actual = {k: v for k, v in actual.items() if k != "coverage"}
    exact_expected = {k: v for k, v in expected.items() if k != "coverage"}
    delta = abs(actual["coverage"] - expected["coverage"])
    if (exact_actual != exact_expected
            or not np.isclose(actual["coverage"], expected["coverage"],
                              rtol=0.0, atol=COVERAGE_REPLAY_ATOL)):
        raise ValueError(f"{where}: replay outcomes differ; expected={expected}, actual={actual}, "
                         f"coverage_absolute_delta={delta}, coverage_atol={COVERAGE_REPLAY_ATOL}")
    return delta


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    common._write(path, value)


def clone(raw):
    return {key: np.asarray(value).copy() for key, value in raw.items()}


def same_raw(actual, expected):
    if actual.keys() != expected.keys() or any(not np.array_equal(actual[k], expected[k]) for k in actual):
        raise ValueError("native reset/prefix/reference observation replay differs")


def reset_id(raw):
    # One reset identity per source is needed for split/duplicate checks, not per-file hashing.
    return hashlib.sha256(raw["pixels"].tobytes() + raw["agent_pos"].tobytes()).hexdigest()


def inventory(out):
    """Read only prior protocol/seed-identity metadata, not predictions/outcomes/test reports."""
    seeds, ids, paths = set(common.PROTOCOL["benchmark_seeds"] + common.PROTOCOL["smoke_seeds"]), set(), []

    def walk(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "future_source_seed_ranges":
                    for lo, hi in v.values():
                        seeds.update(range(lo, hi + 1))
                elif k == "reset_observation_sha256" and isinstance(v, str):
                    ids.add(v)
                elif k == "recorded_prior_reset_ids" and isinstance(v, list):
                    ids.update(v)
                walk(v, k)
        elif isinstance(value, list):
            if "seed" in key:
                seeds.update(v for v in value if isinstance(v, int))
            for v in value:
                walk(v)
        elif isinstance(value, int) and "seed" in key:
            seeds.add(value)

    for folder in sorted((ROOT / "simulation_output").glob("*pusht*")):
        if not folder.is_dir() or folder.resolve() == out.resolve():
            continue
        for name in ("protocol.json", "seed_inventory.json"):
            path = folder / name
            if path.is_file():
                walk(read(path))
                paths.append(str(path.relative_to(ROOT)))
    proposed = {PLAN["preparation_seed"], *PLAN["source_seeds"]}
    if proposed & seeds:
        raise ValueError(f"reserved seeds already recorded: {sorted(proposed & seeds)}")
    return {"metadata_files": paths, "prior_seed_ids": sorted(seeds),
            "recorded_prior_reset_ids": sorted(ids), "overlap": [],
            "scope": "known_protocol_and_seed_inventory_metadata_not_unknown_demonstration_initial_states"}


class Runtime:
    def __init__(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.runtime = common._imports()
        self.policy, self.pre, self.post, self.binding = self.runtime["load_verified"](device="cuda")
        common._validate_policy(self.policy)
        self.config, self.vector = self.runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
        self.env = self.vector.envs[0]
        common.validate_env_kwargs(self.config.gym_kwargs)
        self.env_pre, self.env_post = self.runtime["make_env_pre_post_processors"](self.config, self.policy.config)
        if self.env_pre.steps or self.env_post.steps:
            raise ValueError("expected unchanged identity native environment processors")
        self.counts = {"nominal": 0, "replay": 0, "candidate": 0, "continuation": 0}

    def reset(self, seed):
        common._seed(seed, self.runtime)
        self.policy.reset()
        for pipe in (self.pre, self.post, self.env_pre, self.env_post):
            pipe.reset()
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        raw, _ = self.env.reset(seed=seed)
        return clone(raw)

    def batch(self, raw):
        batch = self.runtime["preprocess_observation"](raw)
        batch = self.runtime["add_envs_task"](self.vector, batch)
        return self.pre(self.env_pre(batch))

    def native(self, normalized):
        return self.env_post({"action": self.post(normalized)})["action"].detach().cpu().numpy().copy()

    def action(self, raw):
        return self.native(self.policy.select_action(self.batch(raw)))

    def step(self, action, kind):
        checked, outside = common.validate_native_action(action, self.env.action_space.low, self.env.action_space.high)
        if outside:
            raise ValueError("out-of-bounds action: no clipping, execution or replacement")
        raw, _, term, trunc, info = self.env.step(checked[0])
        self.counts[kind] += 1
        metrics = common.info_metrics(info, terminal=bool(term or trunc))
        return clone(raw), {**metrics, "terminated": bool(term), "truncated": bool(trunc)}

    def close(self):
        self.vector.close()


def proposal_sheet(raw, native, valid, path, title):
    fonts = [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("C:/Windows/Fonts/msyh.ttc")]
    font = ImageFont.truetype(str(next(p for p in fonts if p.is_file())), 18)
    canvas = Image.new("RGB", (792, 684), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font, fill="black")
    draw.text((12, 35), "同一当前观测｜线条仅为指令设定点，不是实际轨迹或预测后果", font=font, fill="black")
    frame = Image.fromarray(raw["pixels"]).resize((256, 256))
    colors = ("black", "red", "blue", "orange", "purple")
    for panel in range(6):
        left, top = 4 + (panel % 3) * 264, 74 + (panel // 3) * 302
        label = "当前观测" if panel == 0 else f"候选 {panel - 1}：" + ("原 DP" if panel == 1 else "重新采样")
        if panel and not valid[panel - 1]:
            label += "（越界）"
        draw.text((left, top), label, font=font, fill="black")
        canvas.paste(frame, (left, top + 29))
        if panel == 0:
            continue
        k, color = panel - 1, colors[panel - 1]
        points = [(left + float(x) * 256 / 512, top + 29 + float(y) * 256 / 512) for x, y in native[k]]
        draw.line(points, fill=color, width=2)
    canvas.save(path)


@torch.inference_mode()
def nominal(rt, seed, out, *, anchors, cap, used_ids):
    raw = rt.reset(seed)
    identity = reset_id(raw)
    if identity in used_ids:
        raise ValueError("duplicate recorded reset identity; no replacement seed")
    raws, actions, metrics, contexts = [clone(raw)], [], [], []
    for step in range(cap):
        if step in anchors:
            alt_seeds = [10000000 + seed * 10000 + step * 10 + k for k in range(1, 5)]
            proposal = adapter.propose(rt.policy, rt.batch(raw), rt.native, alternative_seeds=alt_seeds)
            if not proposal["valid"][0]:
                raise ValueError("normal DP reference chunk is invalid; no replacement")
            context = {"anchor": step, "proposal": proposal}
            contexts.append(context)
            np.savez_compressed(out / f"context_{step}.npz", pixels=raw["pixels"], agent_xy=raw["agent_pos"],
                                actions=proposal["native"], valid=proposal["valid"],
                                previous_pixels=raws[max(0, step - 1)]["pixels"],
                                previous_agent_xy=raws[max(0, step - 1)]["agent_pos"])
            common._append(out / "contexts.jsonl", {"source_seed": seed, "anchor": step,
                "alternative_seeds": alt_seeds, "valid": proposal["valid"].tolist(),
                "actions": proposal["native"].tolist(), "policy_training_ready": False})
            action = proposal["native"][0, 0][None]
        else:
            action = rt.action(raw)
        if contexts and step - contexts[-1]["anchor"] < 8:
            expected = contexts[-1]["proposal"]["native"][0, step - contexts[-1]["anchor"]]
            if not np.array_equal(action[0], expected):
                raise ValueError("native candidate0 differs from unmodified DP queue")
        raw, measured = rt.step(action, "nominal")
        actions.append(action.copy())
        metrics.append(measured)
        raws.append(clone(raw))
        common._append(out / "nominal.jsonl", {"step": step + 1, "action": action[0].tolist(), **measured})
        if measured["terminated"] or measured["truncated"]:
            break
    return {"raws": raws, "actions": actions, "metrics": metrics, "contexts": contexts,
            "reset_id": identity, "skipped": [a for a in anchors if a not in {c["anchor"] for c in contexts}]}


@torch.inference_mode()
def branch(rt, seed, context, original, candidate, out, *, cap=300):
    anchor, proposal = context["anchor"], context["proposal"]
    raw = rt.reset(seed)
    same_raw(raw, original["raws"][0])
    prefix_coverage_delta, reference_coverage_delta = 0.0, 0.0
    for t, action in enumerate(original["actions"][:anchor]):
        raw, measured = rt.step(action, "replay")
        same_raw(raw, original["raws"][t + 1])
        delta = check_replay_metrics(measured, original["metrics"][t],
                                    where=f"source={seed} anchor={anchor} candidate={candidate} prefix_step={t + 1}")
        prefix_coverage_delta = max(prefix_coverage_delta, delta)
        if measured["terminated"] or measured["truncated"]:
            raise ValueError("prefix ended before the decision point")
    adapter.restore_queues(rt.policy, proposal["reference_queues"])
    rt.policy._queues["action"].clear()
    rt.policy._queues["action"].extend(proposal["normalized"][candidate, 1:, None].clone())
    adapter.restore_rng(proposal["reference_rng"])
    trace, begin = [], time.monotonic()
    for step in range(anchor, cap):
        h = step - anchor
        action = proposal["native"][candidate, 0][None] if h == 0 else rt.action(raw)
        if h < 8 and not np.array_equal(action[0], proposal["native"][candidate, h]):
            raise ValueError("candidate queue is not the saved proposal")
        raw, measured = rt.step(action, "candidate" if h < 8 else "continuation")
        if candidate == 0:
            if step >= len(original["actions"]) or not np.array_equal(action, original["actions"][step]):
                raise ValueError("reference continuation action differs from original DP")
            same_raw(raw, original["raws"][step + 1])
            delta = check_replay_metrics(measured, original["metrics"][step],
                                        where=f"source={seed} anchor={anchor} reference_step={step + 1}")
            reference_coverage_delta = max(reference_coverage_delta, delta)
        trace.append({"step": step + 1, "action": action[0].tolist(), **measured})
        common._append(out / f"branch_{anchor}_{candidate}.jsonl", trace[-1])
        if measured["terminated"] or measured["truncated"]:
            break
    Image.fromarray(raw["pixels"]).save(out / f"terminal_{anchor}_{candidate}.png")
    result = {"source_seed": seed, "anchor": anchor, "candidate": candidate,
              "steps": len(trace), "success": any(x["is_success"] for x in trace),
              "coverage_at8_or_done": trace[min(7, len(trace) - 1)]["coverage"],
              "short_steps": min(8, len(trace)), "final_coverage": trace[-1]["coverage"],
              "post_anchor_max_coverage": max(x["coverage"] for x in trace),
              "terminated": trace[-1]["terminated"], "truncated": trace[-1]["truncated"],
              "coverage_replay_atol": COVERAGE_REPLAY_ATOL,
              "max_prefix_coverage_absolute_delta": prefix_coverage_delta,
              "max_reference_coverage_absolute_delta": reference_coverage_delta if candidate == 0 else None,
              "seconds": time.monotonic() - begin}
    common._append(out / "diagnostic_outcomes.jsonl", result)
    return result


def diversity(native, valid):
    pairs = [float(np.sqrt(np.square(native[i] - native[j]).mean()))
             for i in range(5) for j in range(i + 1, 5) if valid[i] and valid[j]]
    return {"valid_candidates": int(valid.sum()), "pairwise_action_rmse": pairs,
            "unique_valid_chunks": len({native[k].tobytes() for k in range(5) if valid[k]})}


def prepare(out):
    if out.exists():
        raise ValueError("preparation output exists; preserve it, do not overwrite")
    used = inventory(out)
    out.mkdir(parents=True)
    write(out / "protocol.json", PLAN)
    write(out / "seed_inventory.json", used)
    check = out / "preparation"
    check.mkdir()
    rt, start = None, time.monotonic()
    try:
        rt = Runtime()
        original = nominal(rt, PLAN["preparation_seed"], check, anchors=[0], cap=16,
                           used_ids=set(used["recorded_prior_reset_ids"]))
        if len(original["actions"]) != 16:
            raise ValueError("fixed preparation source ended before queue/continuation check")
        context = original["contexts"][0]
        result = branch(rt, PLAN["preparation_seed"], context, original, 0, check, cap=16)
        # Independently replay ordinary DP with no candidate adapter at all.
        raw = rt.reset(PLAN["preparation_seed"])
        for t in range(16):
            action = rt.action(raw)
            if not np.array_equal(action, original["actions"][t]):
                raise ValueError("adding candidate generation changed ordinary DP action")
            raw, measured = rt.step(action, "replay")
            same_raw(raw, original["raws"][t + 1])
            check_replay_metrics(measured, original["metrics"][t], where=f"prepare_plain_step={t + 1}")
        proposal = context["proposal"]
        values = diversity(proposal["native"], proposal["valid"])
        proposal_sheet(original["raws"][0], proposal["native"], proposal["valid"],
                       out / "preparation_candidates_zh.png", "DP 多候选接口准备：未执行四个替代候选")
        report = {"status": "prepared", "checkpoint_binding": rt.binding,
                  "environment_kwargs": rt.config.gym_kwargs,
                  "package_versions": rt.runtime["_package_versions"](),
                  "preparation_reset_id": original["reset_id"], "diversity": values,
                  "reference_equals_plain_DP_first16": True,
                  "reference_actions_observations_discrete_first16_exact": True,
                  "coverage_replay_atol": COVERAGE_REPLAY_ATOL,
                  "extra_sampling_preserves_reference_RNG_and_queues": True,
                  "observation_history_shape": {k: list(v.shape) for k, v in proposal["history"].items()},
                  "native_action_shape": list(proposal["native"].shape),
                  "full_run_started": False, "optimizer_steps": 0, "hardware_actions": 0,
                  "environment_steps_by_kind": rt.counts, "environment_steps": sum(rt.counts.values()),
                  "seconds": time.monotonic() - start, "visual_status": "not_viewed",
                  "headroom_result_available": False, "reference_check_steps": result["steps"],
                  "planned_full_run_minutes": [35, 55]}
        write(out / "preparation_report.json", report)
        write(out / "status.json", {"status": "prepared_user_run_next", "full_sources_completed": 0})
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except BaseException:
        write(out / "status.json", {"status": "preparation_failed", "traceback": traceback.format_exc(),
                                   "environment_steps_by_kind": rt.counts if rt else {}})
        raise
    finally:
        if rt:
            rt.close()


@torch.inference_mode()
def verify_reference(out):
    """Bounded regression on the failed source only; no alternate execution."""
    if read(out / "protocol.json") != PLAN:
        raise ValueError("failed-source regression requires the unchanged experiment protocol")
    seed = 500001
    parent = out / "reference_verification"
    parent.mkdir(exist_ok=True)
    folder = parent / f"attempt_{len(list(parent.glob('attempt_*'))) + 1:03d}"
    folder.mkdir()
    # Keep real divergences fatal, including success/termination near a threshold.
    metric = {"coverage": 0.2, "is_success": False, "source": "info",
              "terminated": False, "truncated": False}
    check_replay_metrics({**metric, "coverage": 0.2 + 1e-13}, metric, where="roundoff_regression")
    rejected = []
    for key, value in (("coverage", 0.2 + 1e-9), ("coverage", float("nan")),
                       ("is_success", True), ("terminated", True), ("truncated", True),
                       ("source", "final_info")):
        try:
            check_replay_metrics({**metric, key: value}, metric, where="mismatch_regression")
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError(f"real mismatch accepted: {key}")
    rt, start = None, time.monotonic()
    try:
        rt = Runtime()
        prepared = read(out / "preparation_report.json")
        if rt.binding != prepared["checkpoint_binding"] or rt.runtime["_package_versions"]() != prepared["package_versions"]:
            raise ValueError("checkpoint/runtime changed since preparation")
        original = nominal(rt, seed, folder, anchors=PLAN["anchors"], cap=300, used_ids=set())
        saved = [json.loads(line) for line in
                 (out / f"sources/{seed}/attempt_001/nominal.jsonl").read_text().splitlines()]
        if len(saved) != len(original["actions"]):
            raise ValueError("nominal length differs from the saved failed attempt")
        for step, row in enumerate(saved):
            if not np.array_equal(original["actions"][step][0], np.asarray(row["action"], dtype=np.float32)):
                raise ValueError(f"nominal action differs from the saved failed attempt at step {step + 1}")
            check_replay_metrics(original["metrics"][step], {k: row[k] for k in original["metrics"][step]},
                                 where=f"saved_failed_attempt_step={step + 1}")
        references = [branch(rt, seed, context, original, 0, folder) for context in original["contexts"]]
        if [r["anchor"] for r in references] != PLAN["anchors"]:
            raise ValueError("failed-source regression did not reach both original anchors")
        report = {"status": "passed", "source_seed": seed, "attempt": str(folder.relative_to(out)),
                  "anchors": PLAN["anchors"], "nominal_steps": len(original["actions"]),
                  "nominal_matches_saved_failed_attempt": True,
                  "reference_branches": references, "actions_observations_discrete_exact": True,
                  "coverage_replay_atol": COVERAGE_REPLAY_ATOL, "rejected_mismatch_cases": rejected,
                  "alternative_candidates_executed": 0, "optimizer_steps": 0, "hardware_actions": 0,
                  "environment_steps": sum(rt.counts.values()), "environment_steps_by_kind": rt.counts,
                  "seconds": time.monotonic() - start, "full_cohort_resumed": False}
        write(folder / "report.json", report)
        print(json.dumps(report), flush=True)
        return 0
    finally:
        if rt:
            rt.close()


def source_roots(out):
    return [out / "sources" / str(seed) for seed in PLAN["source_seeds"]]


def summarize(out):
    per_source, context_rows, whole_episode = [], [], []
    for root in source_roots(out):
        if not (root / "done.json").exists():
            continue
        record = read(root / "done.json")
        folder = out / record["attempt"]
        contexts = [json.loads(x) for x in (folder / "contexts.jsonl").read_text().splitlines()] if (folder / "contexts.jsonl").exists() else []
        outcomes = [json.loads(x) for x in (folder / "diagnostic_outcomes.jsonl").read_text().splitlines()] if (folder / "diagnostic_outcomes.jsonl").exists() else []
        whole_episode.append({"source_seed": record["source_seed"], **record["nominal"]})
        rows = []
        for context in contexts:
            values = [x for x in outcomes if x["anchor"] == context["anchor"]]
            valid_ids = [k for k, valid in enumerate(context["valid"]) if valid]
            if sorted(x["candidate"] for x in values) != valid_ids:
                raise ValueError("completed source lacks a valid candidate outcome")
            reference = next(x for x in values if x["candidate"] == 0)
            row = {"source_seed": record["source_seed"], "anchor": context["anchor"],
                   **diversity(np.asarray(context["actions"], dtype=np.float32), np.asarray(context["valid"])),
                   "reference_success": float(reference["success"]),
                   "oracle_success": float(any(x["success"] for x in values)),
                   "uniform_success": float(np.mean([x["success"] for x in values])),
                   "all_valid_candidates_fail": float(not any(x["success"] for x in values)),
                   "short_coverage_headroom": max(x["coverage_at8_or_done"] for x in values) - reference["coverage_at8_or_done"],
                   "final_coverage_headroom": max(x["final_coverage"] for x in values) - reference["final_coverage"],
                   "candidate_outcomes": values}
            row["rescue_available"] = row["oracle_success"] - row["reference_success"]
            rows.append(row)
            context_rows.append(row)
        metrics = ("reference_success", "oracle_success", "uniform_success", "all_valid_candidates_fail",
                   "short_coverage_headroom", "final_coverage_headroom", "rescue_available")
        complete = [r for r in rows if r["valid_candidates"] == 5]
        per_source.append({"source_seed": record["source_seed"], "contexts": len(rows),
                           "complete5_contexts": len(complete),
                           "skipped_anchors": record["skipped_anchors"],
                           "macro": {k: float(np.mean([r[k] for r in complete])) for k in metrics} if complete else None,
                           "all_available_candidates_macro": {k: float(np.mean([r[k] for r in rows])) for k in metrics} if rows else None})
    eligible = [r["macro"] for r in per_source if r["macro"] is not None]
    aggregate = {k: float(np.mean([r[k] for r in eligible])) for k in eligible[0]} if eligible else None
    controls = {}
    for k in range(5):
        source_values = []
        for source in per_source:
            # Common complete-five population for direct fixed-index comparisons.
            rows = [r for r in context_rows if r["source_seed"] == source["source_seed"] and r["valid_candidates"] == 5]
            if rows:
                source_values.append(np.mean([next(v["success"] for v in r["candidate_outcomes"] if v["candidate"] == k) for r in rows]))
        controls[str(k)] = float(np.mean(source_values)) if source_values else None
    attempts = [read(p) for root in source_roots(out) for p in root.glob("attempt_*/report.json")]
    rescue_sources = [r["source_seed"] for r in per_source if r["macro"] and r["macro"]["rescue_available"] > 0]
    report = {"schema": PLAN["schema"], "status": "completed" if len(per_source) == 8 else "partial",
              "sources_completed": len(per_source), "contexts": len(context_rows),
              "eligible_sources": len(eligible), "source_macro": aggregate, "per_source": per_source,
              "contexts_detail": context_rows, "all_fixed_index_success_complete5_only": controls,
              "complete5_contexts": sum(r["valid_candidates"] == 5 for r in context_rows),
              "nominal_whole_episodes": whole_episode,
              "rescue_source_ids": rescue_sources,
              "nominal_success_rate": float(np.mean([r["success"] for r in whole_episode])) if whole_episode else None,
              "all_recorded_attempt_environment_steps": sum(r["environment_steps"] for r in attempts),
              "oracle_is_deployable": False, "selector_closed_loop": False,
              "training_allowed": False, "optimizer_steps": 0, "hardware_actions": 0,
              "interpretation": "one-intervention finite-candidate hindsight ceiling conditional on one DP continuation RNG, not achieved policy success",
              "decision": ("await_fixed_cohort_completion" if len(per_source) != 8 else
                           "no_complete5_contexts_report_validity_blocker" if not eligible else
                           "no_success_headroom_observed_do_not_train_selector_yet" if not rescue_sources else
                           "possible_headroom_single_source_only_not_stable" if len(rescue_sources) == 1 else
                           "headroom_on_multiple_sources_not_a_learned_selector_gain")}
    write(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("contexts_detail", "per_source", "nominal_whole_episodes")}), flush=True)
    return 0


@torch.inference_mode()
def run(out, *, resume, stop_after_sources):
    if read(out / "protocol.json") != PLAN or read(out / "preparation_report.json")["status"] != "prepared":
        raise ValueError("successful preparation under this exact protocol is required")
    if any(root.exists() for root in source_roots(out)) and not resume:
        raise ValueError("existing run: use --resume; preserve completed sources/partial attempts")
    if all((root / "done.json").exists() for root in source_roots(out)):
        return summarize(out)
    used = inventory(out)
    used_ids = set(used["recorded_prior_reset_ids"])
    used_ids.add(read(out / "preparation_report.json")["preparation_reset_id"])
    for root in source_roots(out):
        if (root / "done.json").exists():
            identity = read(root / "done.json")["reset_id"]
            if identity in used_ids:
                raise ValueError("completed source duplicates prior reset")
            used_ids.add(identity)
    rt, active, begin, completed_now = None, None, time.monotonic(), 0
    old_handlers = {}
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; preserve current source attempt")
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, stop)
    try:
        rt = Runtime()
        prepared = read(out / "preparation_report.json")
        if rt.binding != prepared["checkpoint_binding"] or rt.runtime["_package_versions"]() != prepared["package_versions"]:
            raise ValueError("checkpoint/runtime changed since preparation")
        for seed, root in zip(PLAN["source_seeds"], source_roots(out), strict=True):
            if (root / "done.json").exists():
                continue
            root.mkdir(parents=True, exist_ok=True)
            attempt = root / f"attempt_{len(list(root.glob('attempt_*'))) + 1:03d}"
            attempt.mkdir()
            rt.counts = {k: 0 for k in rt.counts}
            active = {"source_seed": seed, "attempt": str(attempt.relative_to(out)), "status": "running"}
            write(out / "status.json", active)
            start = time.monotonic()
            try:
                original = nominal(rt, seed, attempt, anchors=PLAN["anchors"], cap=300, used_ids=used_ids)
                reference_deltas = []
                for context in original["contexts"]:
                    proposal = context["proposal"]
                    for k, valid in enumerate(proposal["valid"]):
                        if valid:
                            outcome = branch(rt, seed, context, original, k, attempt)
                            if k == 0:
                                reference_deltas.append(outcome["max_reference_coverage_absolute_delta"])
                            print(json.dumps({"progress": "branch_complete", "source_seed": seed,
                                              "anchor": context["anchor"], "candidate": k,
                                              "steps": outcome["steps"]}), flush=True)
                    if seed == PLAN["source_seeds"][0]:
                        proposal_sheet(original["raws"][context["anchor"]], proposal["native"], proposal["valid"],
                                       attempt / f"candidates_{context['anchor']}_zh.png", f"固定首源 {seed} / 起点 {context['anchor']}")
                active.update({"status": "completed_source", "reset_id": original["reset_id"],
                               "skipped_anchors": original["skipped"],
                               "nominal": {"steps": len(original["actions"]),
                                           "success": any(m["is_success"] for m in original["metrics"]),
                                           "max_coverage": max(m["coverage"] for m in original["metrics"])},
                               "all_reference_actions_observations_discrete_exact": True,
                               "coverage_replay_atol": COVERAGE_REPLAY_ATOL,
                               "max_reference_coverage_absolute_delta": max(reference_deltas, default=0.0)})
                used_ids.add(original["reset_id"])
            except BaseException:
                active.update({"status": "incomplete_preserved", "traceback": traceback.format_exc()})
                raise
            finally:
                active.update({"environment_steps": sum(rt.counts.values()), "environment_steps_by_kind": rt.counts.copy(),
                               "seconds": time.monotonic() - start})
                write(attempt / "report.json", active)
            write(root / "done.json", active)
            completed_now += 1
            print(json.dumps(active), flush=True)
            if stop_after_sources and completed_now >= stop_after_sources:
                break
        summarize(out)
        write(out / "status.json", {"status": "completed" if all((p / 'done.json').exists() for p in source_roots(out)) else "paused_at_source_boundary",
                                   "seconds_this_invocation": time.monotonic() - begin})
        return 0
    except BaseException:
        write(out / "status.json", {"status": "interrupted_or_failed", "active_source": active,
                                   "traceback": traceback.format_exc(), "resume": "--stage run --resume"})
        raise
    finally:
        if rt:
            rt.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize", "render-preparation", "verify-reference"), required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int)
    args = parser.parse_args()
    if args.stop_after_sources is not None and args.stop_after_sources < 1:
        parser.error("--stop-after-sources must be positive")
    if args.stage == "prepare":
        return prepare(args.out)
    if args.stage == "summarize":
        return summarize(args.out)
    if args.stage == "verify-reference":
        return verify_reference(args.out)
    if args.stage == "render-preparation":
        with np.load(args.out / "preparation/context_0.npz", allow_pickle=False) as saved:
            proposal_sheet({"pixels": saved["pixels"]}, saved["actions"], saved["valid"],
                           args.out / "preparation_candidates_zh.png", "DP 多候选接口准备：未执行四个替代候选")
        print("Redrawn saved proposals only: zero model forwards/environment steps.")
        return 0
    return run(args.out, resume=args.resume, stop_after_sources=args.stop_after_sources)


if __name__ == "__main__":
    raise SystemExit(main())
