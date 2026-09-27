"""Fixed20-seed native ACT vs ACT+WM closed-loop diagnostic; no training.

The original100-seed benchmark is read-only. Both conditions use the same ACT,
reset seeds, native absolute XY actions and execute8 schedule. Only ACT+WM uses
the previously fixed training goal and five-candidate visual-MSE selector.
"""
from __future__ import annotations

import argparse
from collections import deque
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
import torch

if __package__:
    from . import run_pusht_bc_act_baseline as baseline
    from . import pusht_bc_act_inference as inference
    from . import pusht_bc_act_models as models
    from . import pusht_visual_dynamics as wm
    from .pusht_world_model_adapter import FrozenACTPrior, VisualFeatures, score_visual_goal
else:
    import run_pusht_bc_act_baseline as baseline
    import pusht_bc_act_inference as inference
    import pusht_bc_act_models as models
    import pusht_visual_dynamics as wm
    from pusht_world_model_adapter import FrozenACTPrior, VisualFeatures, score_visual_goal

common = baseline.common
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "schema": "pusht_act_wm_paired20_v1", "seeds": list(range(100000, 100020)),
    "seed_selection": "first20_of_existing100_seed_schedule_not_selected_by_outcome",
    "conditions": ["act", "act_wm"], "order": "alternate_condition_order_by_pair_index",
    "max_steps": 300, "execute_steps": 8, "act_chunk_size": 16,
    "world_model_checkpoint": "fixed_final10000", "act_checkpoint": "fixed_final100000",
    "candidates": 5, "offset_xy": 8, "candidate_rule": "reference_plus_linear_ramp_positive_negative_XY",
    "score": "terminal_raw_visual_feature_MSE_uniform_patches", "goal": "train_episode1_frame117_index278",
    "action_units": "absolute_native_XY_0..512", "action_clipping": False,
    "observation_keys": [models.IMAGE, models.STATE], "precision": "CUDA_float32_no_AMP_no_TF32",
    "outcome_input_to_policy": False, "oracle_input_to_policy": False,
    "new_training_steps": 0, "retry": "none_automatic_preserve_failed_attempt",
    "scope": "small_public_task_diagnostic_not_new_independent100_seed_benchmark",
    "visual_selection": "first_seed_only_at_0_100_200_and_terminal_or_cap",
}


class ChunkSelector:
    """Own execute8 queue; never consume the official ACT queue in WM mode."""

    def __init__(self, prior, *, mode, world_model=None, goal=None):
        if mode not in ("reference_check", "wm"):
            raise ValueError("invalid selector mode")
        if mode == "wm" and (world_model is None or goal is None):
            raise ValueError("WM selection requires trained dynamics and explicit goal")
        self.prior, self.mode, self.world_model, self.goal = prior, mode, world_model, goal
        self.reset()

    def reset(self):
        self.prior.policy.reset()
        self.queue, self.reference_queue = deque(), deque()
        self.action_calls, self.decisions = 0, 0
        self.last_trace = self.last_decision = None
        self.selected_index = 0

    @torch.inference_mode()
    def select_action(self, observation):
        models.validate_observation(observation)
        before = len(self.queue)
        if before != (-self.action_calls) % 8:
            raise ValueError("selector queue/reset schedule drift")
        self.last_decision = None
        if not self.queue:
            if self.mode == "wm":
                request = self.prior.prepare(observation, offset_xy=PROTOCOL["offset_xy"])
                candidates = request["candidates"]
                predicted = self.world_model.predict_candidates(request["current_visual"], request["agent_pos"], candidates)
                scored = score_visual_goal(candidates, predicted, self.goal)
                self.selected_index = int(scored.selected_index.item())
                chosen = scored.selected_chunk
                costs = [float(x) if np.isfinite(x) else None for x in scored.costs[0].tolist()]
            else:
                candidates = self.prior.candidates(observation, offset_xy=PROTOCOL["offset_xy"])
                self.selected_index, chosen, costs = 0, candidates.actions[:, 0], None
            reference = candidates.actions[:, 0]
            self.queue.extend(chosen[:, i].clone() for i in range(8))
            self.reference_queue.extend(reference[:, i].clone() for i in range(8))
            self.decisions += 1
            self.last_decision = {"decision": self.decisions, "selected_index": self.selected_index,
                "valid": candidates.valid[0].tolist(), "costs": costs,
                "reference_chunk": reference[0].tolist(), "selected_chunk": chosen[0].tolist()}
        action, reference_action = self.queue.popleft(), self.reference_queue.popleft()
        # Shared ACT is used as a stateless chunk prior; its official queue stays empty.
        if self.prior.policy.action_calls != 0 or len(self.prior.policy.policy._action_queue) != 0:
            raise ValueError("WM selector unexpectedly consumed the official ACT queue")
        self.last_trace = {"before": before, "after": len(self.queue), "chunk_generated": before == 0,
            "selected_index": self.selected_index, "reference_action": reference_action[0].tolist(),
            "max_abs_reference_offset": float((action - reference_action).abs().max())}
        self.action_calls += 1
        return action


@torch.inference_mode()
def zero_step_check(act, selector, observation):
    """17 action calls cover two queue boundaries, with no environment step."""
    act.reset()
    expected = [act.select_action(observation).clone() for _ in range(17)]
    ref = ChunkSelector(selector.prior, mode="reference_check")
    observed = [ref.select_action(observation).clone() for _ in range(17)]
    if not all(torch.equal(a, b) for a, b in zip(expected, observed)):
        raise ValueError("reference-only selector is not exact ACT execution")
    if ref.decisions != 3 or len(ref.queue) != 7:
        raise ValueError("reference check failed execute8 schedule")
    ref.reset()
    selector.reset()
    sample = selector.select_action(observation)
    if not bool(torch.isfinite(sample).all()):
        raise ValueError("nonfinite actual world-model selected action")
    selector.reset()
    act.reset()
    return {"reference_first17_exact": True, "reference_decisions": 3,
            "partial_queue_reset": True, "trained_wm_selection_finite": True,
            "environment_steps": 0, "optimizer_steps": 0}


def episode(name, policy, env, runtime, seed, out, reset_reference, capture):
    common._seed(seed, runtime)
    policy.reset()
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    raw, _ = env.reset(seed=seed)
    reset_hash = baseline.observation_hash(raw)
    if reset_hash != reset_reference:
        raise ValueError("reset RGB/agent state differs from the frozen benchmark seed")
    snapshots = [(0, common._render(env, np))] if capture else []
    started = time.monotonic()
    coverages, latencies, decision_latencies = [], [], []
    selected_counts, predicted_candidates = [0] * 5, 0
    changed_steps, max_offset, success = 0, 0.0, False
    terminated = truncated = False
    with (out / "steps.jsonl").open("a", encoding="utf-8") as log, (out / "decisions.jsonl").open("a", encoding="utf-8") as decisions:
        for step in range(1, PROTOCOL["max_steps"] + 1):
            observation = inference.observation_from_native(raw, runtime["preprocess_observation"])
            torch.cuda.synchronize()
            measured = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                native = policy.select_action(observation)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - measured
            action, outside = common.validate_native_action(native, env.action_space.low, env.action_space.high)
            if outside:
                raise ValueError("out-of-bounds native action; no clipping or execution")
            latencies.append(elapsed)
            if name == "act":
                trace = dict(policy.last_queue_trace)
                if trace["chunk_generated"]:
                    selected_counts[0] += 1
                    decision_latencies.append(elapsed)
            else:
                trace = dict(policy.last_trace)
                max_offset = max(max_offset, trace["max_abs_reference_offset"])
                changed_steps += int(trace["max_abs_reference_offset"] > 0)
                if trace["chunk_generated"]:
                    decision = policy.last_decision
                    selected_counts[decision["selected_index"]] += 1
                    predicted_candidates += sum(decision["valid"])
                    decision_latencies.append(elapsed)
                    decisions.write(json.dumps({"condition": name, "seed": seed, "step": step, **decision}, allow_nan=False) + "\n")
                if max_offset > PROTOCOL["offset_xy"] + 1e-5:
                    raise ValueError("selected action exceeded the fixed offset budget")
            raw, _reward, term, trunc, info = env.step(action[0])
            terminated, truncated = common._boolean(term, "terminated"), common._boolean(trunc, "truncated")
            metrics = common.info_metrics(info, terminal=terminated or truncated)
            coverages.append(metrics["coverage"])
            success = success or metrics["is_success"]
            log.write(json.dumps({"condition": name, "seed": seed, "step": step,
                "native_action": action[0].tolist(), "queue": trace, "terminated": terminated,
                "truncated": truncated, "coverage": metrics["coverage"], "is_success": metrics["is_success"]}, allow_nan=False) + "\n")
            if capture and (step in (100, 200, 300) or terminated or truncated):
                snapshots.append((step, common._render(env, np)))
            if terminated or truncated:
                break
    result = {"condition": name, "seed": seed, "steps": len(coverages), "is_success": success,
        "max_coverage": max(coverages), "final_coverage": coverages[-1], "terminated": terminated,
        "truncated": truncated, "reset_observation_sha256": reset_hash,
        "seconds": time.monotonic() - started, "policy_seconds": sum(latencies),
        "mean_policy_call_ms": 1000 * float(np.mean(latencies)), "p95_policy_call_ms": 1000 * float(np.percentile(latencies, 95)),
        "mean_decision_ms": 1000 * float(np.mean(decision_latencies)), "decisions": sum(selected_counts),
        "selected_counts": selected_counts, "predicted_candidate_chunks": predicted_candidates,
        "changed_action_steps": changed_steps, "max_abs_reference_offset": max_offset,
        "out_of_bounds_executed_actions": 0}
    return result, snapshots


def summary(episodes):
    count, successes = len(episodes), sum(e["is_success"] for e in episodes)
    decisions = sum(e["decisions"] for e in episodes)
    return {"episodes": count, "successes": successes, "success_rate": successes / count,
        "wilson95": common.wilson95(successes, count),
        "mean_max_coverage": float(np.mean([e["max_coverage"] for e in episodes])),
        "mean_final_coverage": float(np.mean([e["final_coverage"] for e in episodes])),
        "environment_steps": sum(e["steps"] for e in episodes), "episode_seconds": sum(e["seconds"] for e in episodes),
        "policy_seconds": sum(e["policy_seconds"] for e in episodes), "decisions": decisions,
        "mean_policy_call_ms": 1000 * sum(e["policy_seconds"] for e in episodes) / sum(e["steps"] for e in episodes),
        "mean_decision_ms": sum(e["mean_decision_ms"] * e["decisions"] for e in episodes) / decisions,
        "selected_counts": np.sum([e["selected_counts"] for e in episodes], axis=0).tolist(),
        "predicted_candidate_chunks": sum(e["predicted_candidate_chunks"] for e in episodes),
        "changed_action_steps": sum(e["changed_action_steps"] for e in episodes),
        "out_of_bounds_executed_actions": sum(e["out_of_bounds_executed_actions"] for e in episodes)}


def sheet(snapshots, episodes, path):
    from PIL import Image, ImageDraw, ImageFont
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font = (ImageFont.truetype(font_path, size) for size in (24, 21))
    canvas = Image.new("RGB", (1536, 990), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "ACT / ACT＋世界模型：固定首个配对场景 seed=100000 的实际闭环", fill="black", font=title)
    draw.text((12, 45), "同一初始状态，每次执行8步；非预测画面，非挑选的成功案例；本轮无训练", fill="black", font=font)
    for row, name in enumerate(PROTOCOL["conditions"]):
        record = next(e for e in episodes if e["seed"] == PROTOCOL["seeds"][0] and e["condition"] == name)
        top = 87 + row * 447
        label = "ACT" if name == "act" else "ACT＋世界模型"
        draw.text((12, top), f'{label}｜{"成功" if record["is_success"] else "未成功"}｜最大覆盖率 {record["max_coverage"]:.3f}', fill="black", font=font)
        frames = snapshots[name]
        for column in range(4):
            step, frame = frames[min(column, len(frames) - 1)]
            draw.text((column * 384 + 12, top + 31), f"步骤 {step}" + ("（已结束）" if column >= len(frames) else ""), fill="black", font=font)
            canvas.paste(Image.fromarray(frame), (column * 384, top + 59))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "simulation_output/pusht_act_wm_paired20_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started, vector = time.monotonic(), None
    report = {"protocol": PROTOCOL, "status": "running", "episodes": [], "optimizer_steps": 0,
              "hardware_actions": 0, "original_benchmark_modified": False}
    common._write(args.out / "started.json", report)
    try:
        runtime = baseline.imports()
        sources, dp_reference = baseline.source_binding(runtime)
        act, binding = inference.load_final("act")
        prior = FrozenACTPrior(act, binding)
        world_model = wm.load_final(wm.TRAIN_ROOT / "final.pt")
        train_report = json.loads((wm.TRAIN_ROOT / "report.json").read_text())
        manifest = json.loads((wm.CACHE_ROOT / "manifest.json").read_text())
        if train_report["status"] != "completed_fixed_visual_dynamics_training" or train_report["plan"] != wm.PLAN:
            raise ValueError("world-model training is not the completed fixed run")
        if manifest["goal"] != train_report["cache_identity"]["goal"] or prior.space_id != world_model.space_id:
            raise ValueError("goal provenance or frozen feature space differs")
        from PIL import Image
        with Image.open(wm.CACHE_ROOT / "goal_rgb.png") as image:
            pixels = np.array(image.convert("RGB"))
        # Match data.batch: cast/divide on CPU before transferring float32 RGB.
        goal_rgb = torch.from_numpy(pixels).permute(2, 0, 1)[None].contiguous().float().div_(255).cuda()
        single_encoding = prior.encode_image(goal_rgb)
        # Read just the fixed goal row, not the450MiB cache into memory.
        cached_goal = np.load(wm.CACHE_ROOT / "features.npy", mmap_mode="r")[manifest["goal"]["global_index"]].copy()
        goal = VisualFeatures(torch.from_numpy(cached_goal)[None].cuda(), prior.space_id)
        # The investigated source batch256 reproduced the cache exactly. CPU/GPU
        # division plus batch layout changed isolated float32 features by <3e-6.
        # This check allows that rounding only; the score uses the cache exactly.
        torch.testing.assert_close(single_encoding.values, goal.values, rtol=1e-4, atol=1e-5)
        report["goal_encoding_check"] = {
            "score_goal_source": "exact_training_goal_feature_cache",
            "single_vs_cache_max_abs": float((single_encoding.values - goal.values).abs().max()),
            "readback_rtol": 0.0001, "readback_atol": 0.00001,
            "readback_tolerance_does_not_modify_goal_or_score": True,
        }
        selector = ChunkSelector(prior, mode="wm", world_model=world_model, goal=goal)
        historical = json.loads((ROOT / "simulation_output/pusht_bc_act_benchmark_v1/act/report.json").read_text())
        if historical["checkpoint_binding"] != binding or not historical["strict_protocol_pass"]:
            raise ValueError("ACT identity differs from its completed benchmark")
        config, vector = runtime["_make_environment"]("pusht", argparse.Namespace(episode_length=300))
        env = vector.envs[0]
        common.validate_env_kwargs(config.gym_kwargs)
        common.verify_environment_parity(historical, {"environment_gym_kwargs": config.gym_kwargs,
            "environment_step_source_sha256": common.canonical_hash(inspect.getsource(type(env.unwrapped).step)),
            "package_versions": runtime["_package_versions"]()})
        # source_binding above verifies the unchanged environment/preprocessing implementations.
        env_pre, env_post = runtime["make_env_pre_post_processors"](config, act.policy.config)
        if env_pre.steps or env_post.steps:
            raise ValueError("native environment processors must remain identity")
        raw, _ = env.reset(seed=PROTOCOL["seeds"][0])
        probe = inference.observation_from_native(raw, runtime["preprocess_observation"])
        report["zero_step_check"] = zero_step_check(act, selector, probe)
        report.update({"act_checkpoint": binding["checkpoint_path"], "act_model_identity": binding["model_state_sha256"],
            "wm_checkpoint": str(wm.TRAIN_ROOT / "final.pt"), "wm_training_steps": wm.PLAN["steps"],
            "goal": manifest["goal"], "feature_space_id": prior.space_id,
            "environment_gym_kwargs": config.gym_kwargs, "source_binding": sources,
            "package_versions": runtime["_package_versions"](), "goal_rgb_matches_fixed_cache": True})
        common._write(args.out / "started.json", report)
        resets = {e["seed"]: e["reset_observation_sha256"] for e in dp_reference["episodes"]}
        visuals = {}
        for pair_index, seed in enumerate(PROTOCOL["seeds"]):
            names = PROTOCOL["conditions"] if pair_index % 2 == 0 else list(reversed(PROTOCOL["conditions"]))
            for name in names:
                value, snapshots = episode(name, act if name == "act" else selector, env, runtime, seed,
                                           args.out, resets[seed], seed == PROTOCOL["seeds"][0])
                report["episodes"].append(value)
                common._append(args.out / "episodes.jsonl", value)
                if snapshots:
                    visuals[name] = snapshots
                common._write(args.out / "status.json", {"status": "running", "episodes_completed": len(report["episodes"])})
                print(json.dumps({"condition": name, "seed": seed, "steps": value["steps"],
                    "success": value["is_success"], "max_coverage": value["max_coverage"]}), flush=True)
        conditions = {name: [e for e in report["episodes"] if e["condition"] == name] for name in PROTOCOL["conditions"]}
        report["summary"] = {name: summary(rows) for name, rows in conditions.items()}
        pairs = [(conditions["act"][i], conditions["act_wm"][i]) for i in range(20)]
        if any(a["seed"] != b["seed"] or a["reset_observation_sha256"] != b["reset_observation_sha256"] for a, b in pairs):
            raise ValueError("paired seed/reset mismatch")
        a, b = report["summary"]["act"], report["summary"]["act_wm"]
        report["paired"] = {"act_only_success": sum(x["is_success"] and not y["is_success"] for x, y in pairs),
            "wm_only_success": sum(y["is_success"] and not x["is_success"] for x, y in pairs),
            "both_success": sum(x["is_success"] and y["is_success"] for x, y in pairs),
            "both_fail": sum(not x["is_success"] and not y["is_success"] for x, y in pairs),
            "success_rate_change_pp": 100 * (b["success_rate"] - a["success_rate"]),
            "mean_max_coverage_change": b["mean_max_coverage"] - a["mean_max_coverage"],
            "policy_call_latency_ratio": b["mean_policy_call_ms"] / a["mean_policy_call_ms"],
            "decision_latency_ratio": b["mean_decision_ms"] / a["mean_decision_ms"]}
        old = {e["seed"]: e for e in historical["episodes"]}
        report["historical_act_subset_differences"] = [{"seed": e["seed"], "field": key, "fresh": e[key], "historical": old[e["seed"]][key]}
            for e in conditions["act"] for key in ("steps", "is_success", "max_coverage") if e[key] != old[e["seed"]][key]]
        if any(p.requires_grad or p.grad is not None for module in (act.policy, world_model) for p in module.parameters()):
            raise ValueError("inference unexpectedly enabled gradients")
        sheet(visuals, report["episodes"], args.out / "first_seed_pair_zh.png")
        report.update({"status": "completed_paired20_diagnostic", "environment_steps": a["environment_steps"] + b["environment_steps"],
            "visual_status": "not_viewed", "visual_artifact": "first_seed_pair_zh.png",
            "decision": "read_paired_outcomes_no_automatic_promotion_or_extra_training",
            "independent_new_benchmark": False, "multi_training_seed_stability_tested": False})
    except BaseException as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(),
                       "partial_outputs_preserved": True})
    finally:
        if vector is not None:
            vector.close()
        report["runtime_seconds"] = time.monotonic() - started
        common._write(args.out / "report.json", report)
        common._write(args.out / "status.json", {"status": report["status"], "episodes_completed": len(report["episodes"])})
    print(json.dumps({key: report.get(key) for key in ("status", "summary", "paired", "runtime_seconds", "error")}, indent=2), flush=True)
    return 0 if report["status"] == "completed_paired20_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
