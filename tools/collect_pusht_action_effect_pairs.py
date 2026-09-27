"""Bounded native Push-T action-effect pairs; no training or robot control.

Only the locked train/validation sources are supported. Each candidate starts
from native reset + exactly replayed ACT prefix. Resume skips complete sources
and preserves incomplete attempts in separate directories. Test sources stay
untouched until a later fixed-checkpoint evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
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
    from . import pusht_action_effect_contrast as effect
    from . import prepare_pusht_action_effect_contrast as prep
    from . import run_pusht_fresh_scorer_pair as previous
else:
    import pusht_action_effect_contrast as effect
    import prepare_pusht_action_effect_contrast as prep
    import run_pusht_fresh_scorer_pair as previous

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation_output/pusht_action_effect_pairs_v1"
SCHEMA = "pusht_action_effect_execution_v1"
SPLITS = ("train", "validation")
MAX_STEPS_PER_SOURCE = 1488  # 168 nominal + 5*(0+80+160) replay + 15*8 candidate.
replay, vision = previous.replay, previous.vision
baseline, inference, common = previous.baseline, previous.inference, previous.common
read, rows = prep.read, prep.rows


def write(path, value):
    """Atomic completion marker/status; source journals are append-only."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def source_list(protocol):
    return [(split, seed) for split in SPLITS
            for seed in range(protocol["future_source_seed_ranges"][split][0],
                              protocol["future_source_seed_ranges"][split][1] + 1)]


def reset_ids_from_metadata(inventory):
    found = set()

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "reset_observation_sha256" and isinstance(child, str):
                    found.add(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for name in inventory["metadata_files"]:
        path = ROOT / name
        walk(rows(path) if path.suffix == ".jsonl" else read(path))
    return found


def cache_identity():
    m = read(prep.base.CACHE_ROOT / "manifest.json")
    if m["schema"] != prep.base.SCHEMA or m["status"] != "prepared" or m["goal"]["global_index"] != 278:
        raise ValueError("original audited train-only goal/cache required")
    return {key: m[key] for key in ("schema", "source_revision", "normalization", "goal", "split")}


def prepare(out):
    protocol = read(prep.PROTOCOL)
    inventory = prep.reserved_seed_inventory(protocol, out)
    identity = cache_identity()
    out.mkdir(parents=True, exist_ok=False)
    spec = {"schema": SCHEMA, "protocol": protocol, "cache_identity": identity,
            "source_splits": [{"split": split, "seed": seed} for split, seed in source_list(protocol)],
            "maximum_environment_steps_per_source": MAX_STEPS_PER_SOURCE,
            "maximum_environment_steps_single_pass": MAX_STEPS_PER_SOURCE * len(source_list(protocol)),
            "prior_reset_observation_ids": sorted(reset_ids_from_metadata(inventory)),
            "prior_seed_inventory": inventory, "test_sources_allowed": False,
            "checkpoint": "unchanged_ACT_fixed_final100000", "optimizer_steps": 0, "hardware_actions": 0}
    write(out / "execution_contract.json", spec)
    write(out / "status.json", {"status": "prepared_no_environment_or_model_execution",
                                "completed_sources": 0, "environment_steps": 0})
    print(json.dumps({"status": "prepared", "sources": len(spec["source_splits"]),
                      "maximum_environment_steps": spec["maximum_environment_steps_single_pass"],
                      "environment_steps": 0, "optimizer_steps": 0, "test_sources": 0}), flush=True)


def contract(out):
    spec = read(out / "execution_contract.json")
    if (spec["schema"] != SCHEMA or spec["protocol"] != read(prep.PROTOCOL)
            or spec["cache_identity"] != cache_identity()):
        raise ValueError("locked protocol or original normalization/goal changed")
    return spec


@torch.inference_mode()
def nominal_prefix(env, runtime, act, prior, seed, attempt, counter, protocol, used_resets, *, on_context=None):
    act.reset()
    raw = replay.reset(env, seed, runtime)
    reset_id = baseline.observation_hash(raw)  # One necessary initial-state identity, not per-file/step hashes.
    if reset_id in used_resets:
        raise ValueError("source reset duplicates an earlier project/completed-study observation")
    write(attempt / "reset.json", {"seed": seed, "reset_observation_sha256": reset_id})
    raws, actions, metrics, contexts = [replay.clone_observation(raw)], [], [], []
    for step in range(protocol["nominal_cap"]):
        obs = inference.observation_from_native(raw, runtime["preprocess_observation"])
        if step in protocol["anchors"]:
            if act.action_calls != step or len(act.policy._action_queue):
                raise ValueError("anchor must be an unmodified ACT action-queue boundary")
            candidates = prior.candidates(obs, offset_xy=8.0)
            seen = vision.observe_rgb(raw["pixels"])
            context = {"seed": seed, "anchor_step": step, "current_valid": seen.valid,
                       "valid": candidates.valid[0].tolist(), "actions": candidates.actions[0].cpu().numpy(),
                       "agent_xy": raw["agent_pos"].tolist(),
                       "current_rgb": replay.save_rgb(attempt, f"{seed}_{step}_current", raw)}
            contexts.append(context)
            # Proposal identity and current observation are durable before any future target.
            common._append(attempt / "contexts.jsonl", {**context, "actions": context["actions"].tolist()})
            if on_context is not None:
                on_context(context, seen, candidates)  # Current-only hook, before this anchor's future.
        action = act.select_action(obs).cpu().numpy().copy()
        if contexts and step - contexts[-1]["anchor_step"] < 8:
            if not np.array_equal(action[0], contexts[-1]["actions"][0, step - contexts[-1]["anchor_step"]]):
                raise ValueError("candidate0 differs from the actual unchanged ACT action")
        raw, measured = replay.step(env, action, counter, "nominal")
        actions.append(action)
        metrics.append(measured)
        raws.append(replay.clone_observation(raw))
        common._append(attempt / "nominal_prefix.jsonl", {"seed": seed, "step": step + 1,
                                                         "action": action[0].tolist(), **measured})
        if measured["terminated"] or measured["truncated"]:
            break
    skipped = [{"seed": seed, "anchor_step": anchor, "reason": "nominal_ended_before_anchor"}
               for anchor in protocol["anchors"] if anchor not in {c["anchor_step"] for c in contexts}]
    return {"raws": raws, "actions": actions, "metrics": metrics, "contexts": contexts,
            "skipped": skipped, "reset_id": reset_id}


def completed(out, protocol):
    return [(split, seed, read(out / split / str(seed) / "done.json"))
            for split, seed in source_list(protocol) if (out / split / str(seed) / "done.json").is_file()]


def total_attempt_steps(out):
    return sum(read(path).get("environment_steps", 0)
               for split in SPLITS for path in (out / split).glob("*/attempt_*/report.json"))


def collect(out, *, resume, stop_after_sources=None):
    spec = contract(out)
    protocol = spec["protocol"]
    done = completed(out, protocol)
    if len(done) == len(source_list(protocol)):
        print("All fixed sources completed; packing only, no model/environment rerun.", flush=True)
        return pack(out)
    attempted = any((out / split).exists() for split in SPLITS)
    if attempted and not resume:
        raise ValueError("prior attempt exists; use --resume to preserve complete sources and retry only incomplete ones")
    inventory = prep.reserved_seed_inventory(protocol, out)
    used_resets = reset_ids_from_metadata(inventory)
    for split, seed, item in done:
        if item["status"] != "completed_source" or item["split"] != split or item["seed"] != seed:
            raise ValueError("source completion identity mismatch")
        if item["reset_observation_sha256"] in used_resets:
            raise ValueError("completed source overlaps another source/project reset")
        used_resets.add(item["reset_observation_sha256"])
    start, vector, invocation_steps, new_sources = time.monotonic(), None, 0, 0
    state = {"status": "running", "optimizer_steps": 0, "hardware_actions": 0, "test_sources": 0}
    old_handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}: preserve partial source and resume at its start")

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, interrupt)
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
        runtime_binding = {"ACT_binding": binding, "source_binding": sources}
        binding_path = out / "runtime_binding.json"
        if binding_path.exists() and read(binding_path) != runtime_binding:
            raise ValueError("ACT/source identity changed across resume")
        if not binding_path.exists():
            write(binding_path, runtime_binding)
        goal = np.load(prep.base.CACHE_ROOT / "grid_counts.npy", mmap_mode="r", allow_pickle=False)[278].astype(np.float32) / 16
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
        write(out / "environment_parity.json", parity)
        for split, seed in source_list(protocol):
            root = out / split / str(seed)
            if (root / "done.json").exists():
                continue
            root.mkdir(parents=True, exist_ok=True)
            attempts = sorted(root.glob("attempt_*"))
            attempt = root / f"attempt_{len(attempts) + 1:03d}"
            attempt.mkdir()
            (attempt / "observations").mkdir()
            counter = {"nominal": 0, "replay": 0, "candidate": 0}
            entry = {"status": "running", "seed": seed, "split": split,
                     "attempt": str(attempt.relative_to(out)), "environment_steps_by_kind": counter,
                     "started_utc": datetime.now(timezone.utc).isoformat()}
            begin = time.monotonic()
            write(attempt / "report.json", entry)
            try:
                nominal = nominal_prefix(env, runtime, act, prior, seed, attempt, counter, protocol, used_resets)
                for context in nominal["contexts"]:
                    for candidate, valid in enumerate(context["valid"]):
                        if valid:
                            previous.branch(env, runtime, seed, context, nominal, candidate, goal, attempt, counter)
                if len(nominal["contexts"]) + len(nominal["skipped"]) != 3 or sum(counter.values()) > MAX_STEPS_PER_SOURCE:
                    raise ValueError("fixed source context/step budget exceeded")
                entry.update({"status": "completed_source", "contexts": len(nominal["contexts"]),
                              "skipped_anchors": nominal["skipped"],
                              "reset_observation_sha256": nominal["reset_id"],
                              "all_replayed_observations_exact": True, "ACT_reference_continuations_exact": True})
            except BaseException as exc:
                entry.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              "error": str(exc), "traceback": traceback.format_exc()})
                raise
            finally:
                entry.update({"environment_steps": sum(counter.values()), "runtime_seconds": time.monotonic() - begin})
                invocation_steps += sum(counter.values())
                write(attempt / "report.json", entry)
            write(root / "done.json", entry)
            used_resets.add(entry["reset_observation_sha256"])
            new_sources += 1
            state.update({"completed_sources": len(completed(out, protocol)), "last_seed": seed,
                          "invocation_environment_steps": invocation_steps})
            write(out / "status.json", state)
            print(json.dumps({"split": split, "seed": seed, "completed_sources": state["completed_sources"],
                              "environment_steps": invocation_steps}), flush=True)
            if stop_after_sources is not None and new_sources >= stop_after_sources:
                break
        state["status"] = "sources_completed" if len(completed(out, protocol)) == len(source_list(protocol)) else "paused_at_source_boundary"
    except BaseException as exc:
        state.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        if vector is not None:
            vector.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        state.update({"completed_sources": len(completed(out, protocol)), "runtime_seconds": time.monotonic() - start,
                      "invocation_environment_steps": invocation_steps,
                      "all_recorded_attempt_environment_steps": total_attempt_steps(out)})
        common._append(out / "invocations.jsonl", state)
        write(out / "status.json", state)
    if state["status"] == "sources_completed":
        return pack(out)
    print(json.dumps(state), flush=True)
    return 1 if state["status"] in ("failed", "interrupted") else 0


def render_first_sources(out, entries, protocol):
    """Fixed first train/val source, all planned anchors, including exclusions."""
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    title, font, small = (ImageFont.truetype(font_path, n) for n in (25, 18, 15))
    image = Image.new("RGB", (1440, 1810), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), "同起点动作效应数据：固定首个训练源与验证源", font=title, fill="black")
    draw.text((16, 48), "当前图像 + 五个候选的实际 8 步后果；不是预测图像，不是模型收益。候选 0 为原 ACT。", font=font, fill="black")
    row_index = 0
    for split in SPLITS:
        seed = protocol["future_source_seed_ranges"][split][0]
        item = next(e for s, n, e in entries if s == split and n == seed)
        folder = out / item["attempt"]
        contexts = {r["anchor_step"]: r for r in rows(folder / "contexts.jsonl")}
        outcomes = {(r["anchor_step"], r["candidate"]): r for r in rows(folder / "outcomes.jsonl")}
        for anchor in protocol["anchors"]:
            y = 91 + row_index * 280
            row_index += 1
            context = contexts.get(anchor)
            name = "训练" if split == "train" else "验证"
            draw.text((16, y), f"{name}源 {seed} / 起点 {anchor}；当前观测有效={context['current_valid'] if context else '未到达'}", font=font, fill="black")
            if context is None:
                draw.text((16, y + 40), "原 ACT 提前结束；未补选起点。", font=font, fill="black")
                continue
            for column in range(6):
                x = 16 + column * 238
                target = outcomes.get((anchor, column - 1)) if column else None
                label = "当前输入" if column == 0 else f"候选 {column - 1}" + (" / ACT" if column == 1 else "")
                draw.text((x, y + 32), label, font=small, fill="black")
                path = context["current_rgb"] if column == 0 else target["terminal_rgb"] if target else None
                if path:
                    with Image.open(folder / path) as tile:
                        image.paste(tile.convert("RGB").resize((170, 170)), (x, y + 57))
                if target:
                    draw.text((x, y + 233), f"coverage={target['terminal_coverage']:.5f} / {target['steps']}步", font=small, fill="black")
                elif column:
                    draw.text((x, y + 233), "未执行：候选越界", font=small, fill="black")
    draw.text((16, 1780), "候选对非独立轨迹；测试源未运行。初始状态不重复不代表物理变化覆盖充分。", font=small, fill="black")
    image.save(out / "first_sources_pairs_zh.png")


def pack(out):
    spec = contract(out)
    protocol = spec["protocol"]
    entries = completed(out, protocol)
    if len(entries) != len(source_list(protocol)):
        raise ValueError("complete every planned train/validation source before packing; no partial-cohort training")
    reset_ids = [e["reset_observation_sha256"] for _, _, e in entries]
    if len(reset_ids) != len(set(reset_ids)):
        raise ValueError("duplicate source initial observations")
    packs, summaries = {}, {}
    for split in SPLITS:
        obs = {key: [] for key in effect.INPUT_FIELDS}
        targets, metadata, exclusions, skipped, actual_candidates = [], [], [], [], 0
        for source_split, seed, item in entries:
            if source_split != split:
                continue
            if item["status"] != "completed_source" or not item["all_replayed_observations_exact"] or not item["ACT_reference_continuations_exact"]:
                raise ValueError("unverified source replay")
            folder = out / item["attempt"]
            skipped.extend(item["skipped_anchors"])
            results = rows(folder / "outcomes.jsonl")
            actual_candidates += len(results)
            keyed = {(r["anchor_step"], r["candidate"]): r for r in results}
            if len(keyed) != len(results) or any(r["seed"] != seed for r in results):
                raise ValueError("duplicate/mismatched future target")
            for context in rows(folder / "contexts.jsonl"):
                actual = [keyed.get((context["anchor_step"], k)) for k in range(5)]
                reasons = []
                if not context["current_valid"]:
                    reasons.append("invalid_current_object")
                if not all(context["valid"]):
                    reasons.append("not_all_five_candidates_valid")
                if any(t is None or not t["full_horizon"] or t["steps"] != 8 for t in actual):
                    reasons.append("not_all_five_complete8step_futures")
                if reasons:
                    exclusions.append({"seed": seed, "anchor_step": context["anchor_step"], "reasons": reasons})
                    continue
                with Image.open(folder / context["current_rgb"]) as image:
                    current = vision.observe_rgb(np.asarray(image.convert("RGB")))
                if not current.valid or context["seed"] != seed:
                    raise ValueError("current input extraction/source identity changed")
                obs["current"].append(current.features.values[0])
                obs["agent_xy"].append(context["agent_xy"])
                obs["actions"].append(context["actions"])
                targets.append([t["terminal_coverage"] for t in actual])
                metadata.append({"source_seed": seed, "anchor_step": context["anchor_step"], "split": split,
                                 "source_attempt": item["attempt"], "current_rgb": context["current_rgb"],
                                 "reset_observation_sha256": item["reset_observation_sha256"]})
        if not metadata:
            raise ValueError(f"no eligible groups in fixed {split}; do not invent labels or replace seeds")
        root = out / split / "pack"
        root.mkdir(exist_ok=True)
        np.savez_compressed(root / "observations.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in obs.items()})
        target = np.asarray(targets, dtype=np.float64)
        np.savez_compressed(root / "targets.npz", terminal_coverage=target)
        (root / "contexts.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metadata), encoding="utf-8")
        manifest = {"schema": effect.SCHEMA, "status": "completed", "data_role": split,
                    "training_allowed": split == "train", "scope": "public_PushT_scorer_only_not_guidewire_formal_data",
                    "contexts": len(metadata), "source_seeds": sorted({m["source_seed"] for m in metadata}),
                    "input_fields": list(effect.INPUT_FIELDS), "target_fields": ["terminal_coverage"],
                    "future_targets_are_observations": False, "same_start_provenance": "exact_native_reset_and_ACT_prefix_replay",
                    "execution_contract": "../../execution_contract.json", "fresh_final_test": False,
                    "test_set": "reserved_not_executed", "no_future_visibility_filter": True}
        write(root / "manifest.json", manifest)
        packs[split] = effect.PairPack(root, for_training=(split == "train"))
        i, j = np.triu_indices(5, 1)
        informative = np.ptp(target, axis=1) > 1e-6
        summaries[split] = {"planned_sources": sum(s == split for s, _, _ in entries),
            "eligible_sources": len(manifest["source_seeds"]), "eligible_contexts": len(metadata),
            "eligible_candidates": int(target.size), "actually_executed_candidates": actual_candidates,
            "informative_contexts": int(informative.sum()), "informative_pairs": int((np.abs(target[:, i] - target[:, j]) > 1e-6).sum()),
            "all_pairs_including_ties": int(len(target) * 10), "exclusions": exclusions, "unreached_anchors": skipped,
            "informative_source_seeds": sorted({m["source_seed"] for m, ok in zip(metadata, informative) if ok})}
    effect.require_disjoint_sources(*packs.values())
    stats = effect.training_target_stats(packs["train"].targets)
    write(out / "normalization.json", {"input": spec["cache_identity"]["normalization"], "target": stats,
                                       "goal": spec["cache_identity"]["goal"], "validation_used": False})
    render_first_sources(out, entries, protocol)
    report = {"schema": SCHEMA, "status": "completed_train_validation_candidate_pairs", "splits": summaries,
        "source_seeds": [seed for _, seed, _ in entries], "source_count": len(entries),
        "distinct_reset_observations": len(set(reset_ids)), "split_source_overlap": [],
        "exact_replay_and_ACT_reference_verified": True, "target_normalization_training_only": stats,
        "environment_steps_completed_sources": sum(e["environment_steps"] for _, _, e in entries),
        "environment_steps_all_recorded_attempts": total_attempt_steps(out),
        "optimizer_steps": 0, "hardware_actions": 0, "test_sources_executed": 0,
        "data_ready_for_fixed_scorer_experiment": True, "training_entrypoint_ready": False,
        "real_system_validated": False, "policy_benefit_evaluated": False,
        "visual_status": "not_viewed", "visual_selection": "fixed_first_train_and_validation_sources_all_anchors",
        "reset_novelty_limit": "no_exact_observation_duplicates_against_recorded_project_ids_not_unknown_original_dataset_states",
        "behavior_diversity_limit": "seed_and_reset_uniqueness_does_not_prove_broad_behavior_or_geometry_coverage"}
    write(out / "report.json", report)
    write(out / "status.json", {"status": report["status"], "completed_sources": len(entries),
                                "environment_steps": report["environment_steps_all_recorded_attempts"]})
    print(json.dumps(report), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "collect", "pack"), default="prepare")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int, help="planned pause after N newly completed sources; never changes cohort")
    args = parser.parse_args()
    if args.stop_after_sources is not None and args.stop_after_sources < 1:
        parser.error("--stop-after-sources must be positive")
    if args.stage == "prepare":
        prepare(args.out)
        return 0
    if args.stage == "pack":
        return pack(args.out)
    return collect(args.out, resume=args.resume, stop_after_sources=args.stop_after_sources)


if __name__ == "__main__":
    raise SystemExit(main())
