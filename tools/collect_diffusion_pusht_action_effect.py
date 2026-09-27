"""Independent DP train/validation candidate pairs; never train or run test seeds.

Reuses the verified DP headroom runtime, proposal, prefix replay and continuation.
The inspected eight-source diagnostic is used for identity metadata only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

if __package__:
    from . import probe_diffusion_pusht_candidate_headroom as dp
    from . import pusht_action_effect_contrast as pair
    from . import pusht_object_goal as vision
    from .pusht_object_dynamics import CACHE_ROOT
    from .train_pusht_action_effect_contrast import write, save_arrays
else:
    import probe_diffusion_pusht_candidate_headroom as dp
    import pusht_action_effect_contrast as pair
    import pusht_object_goal as vision
    from pusht_object_dynamics import CACHE_ROOT
    from train_pusht_action_effect_contrast import write, save_arrays

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "simulation_output/diffusion_pusht_action_effect_pairs_v1"
PROTOCOL = ROOT / "docs/diffusion-pusht-action-effect-protocol-v1.json"
SCHEMA = "diffusion_pusht_action_effect_pairs_v1"
SPLITS = ("train", "validation")
read = dp.read


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sources(protocol):
    return [(split, seed) for split in SPLITS
            for seed in range(protocol["future_source_seed_ranges"][split][0],
                              protocol["future_source_seed_ranges"][split][1] + 1)]


def inventory(out, protocol):
    """Only protocol/seed/reset metadata; no old branch labels or test reports."""
    seeds = set(dp.common.PROTOCOL["benchmark_seeds"] + dp.common.PROTOCOL["smoke_seeds"])
    identities, paths = set(), []

    def walk(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                if name == "future_source_seed_ranges":
                    for low, high in child.values():
                        seeds.update(range(low, high + 1))
                if name in ("reset_id", "reset_observation_sha256", "preparation_reset_id") and isinstance(child, str):
                    identities.add(child)
                if name in ("recorded_prior_reset_ids", "prior_reset_observation_ids") and isinstance(child, list):
                    identities.update(child)
                walk(child, name)
        elif isinstance(value, list):
            if "seed" in key:
                seeds.update(v for v in value if isinstance(v, int))
            for child in value:
                walk(child, key)
        elif isinstance(value, int) and "seed" in key:
            seeds.add(value)

    for path in (ROOT / "docs").glob("*pusht*protocol*.json"):
        if path != PROTOCOL:
            walk(read(path))
            paths.append(str(path.relative_to(ROOT)))
    for folder in sorted((ROOT / "simulation_output").glob("*pusht*")):
        if not folder.is_dir() or folder.resolve() == out.resolve():
            continue
        for name in ("protocol.json", "seed_inventory.json", "execution_contract.json"):
            path = folder / name
            if path.is_file():
                walk(read(path))
                paths.append(str(path.relative_to(ROOT)))
        # Select identity fields only; do not use nominal outcome summaries.
        for pattern in ("sources/*/done.json", "train/*/done.json", "validation/*/done.json"):
            for path in folder.glob(pattern):
                record = read(path)
                walk({k: record[k] for k in ("source_seed", "seed", "reset_id", "reset_observation_sha256") if k in record})
                paths.append(str(path.relative_to(ROOT)))
    reserved = [seed for low, high in protocol["future_source_seed_ranges"].values() for seed in range(low, high + 1)]
    if len(set(reserved)) != len(reserved) or seeds.intersection(reserved):
        raise ValueError("fixed train/validation/test reservation overlaps existing source metadata")
    return {"metadata_files": paths, "prior_seed_ids": sorted(seeds),
            "recorded_prior_reset_ids": sorted(identities), "overlap": [],
            "scope": "known_metadata_only_not_unknown_demonstration_initial_states"}


def cache_identity():
    manifest = read(CACHE_ROOT / "manifest.json")
    goal = manifest["goal"]
    if (manifest["status"] != "prepared" or
            (goal["episode_index"], goal["frame_index"], goal["global_index"]) != (1, 117, 278)
            or 1 not in manifest["split"]["train_episodes"]):
        raise ValueError("requires the unchanged original training-only goal/cache")
    return {key: manifest[key] for key in ("schema", "source_revision", "normalization", "goal", "split")}


def goal_grid():
    return np.load(CACHE_ROOT / "grid_counts.npy", mmap_mode="r", allow_pickle=False)[278].astype(np.float32) / 16


def write_pack(root, obs, target, success, metadata, *, split, synthetic=False):
    root.mkdir(parents=True, exist_ok=True)
    save_arrays(root / "observations.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in obs.items()})
    save_arrays(root / "targets.npz", terminal_coverage=np.asarray(target, dtype=np.float64))
    save_arrays(root / "evaluation_targets.npz", success=np.asarray(success, dtype=bool))
    (root / "contexts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    write(root / "manifest.json", {
        "schema": pair.SCHEMA, "proposal_family": SCHEMA, "status": "completed",
        "data_role": split, "training_allowed": split == "train" and not synthetic,
        "synthetic_unit_fixture": synthetic, "contexts": len(metadata),
        "source_seeds": sorted({r["source_seed"] for r in metadata}),
        "input_fields": list(pair.INPUT_FIELDS), "target_fields": ["terminal_coverage"],
        "evaluation_target_fields": ["success"], "future_targets_are_observations": False,
        "target_semantics": "synthetic_unit_fixture_not_measured" if synthetic else "candidate_then_DP_until_first_done_or_global300",
        "scope": "public_PushT_scorer_only_not_guidewire_formal_data",
        "fresh_final_test": False})
    return pair.PairPack(root, for_training=(split == "train"))


def contract(out):
    spec = read(out / "execution_contract.json")
    if spec["protocol"] != read(PROTOCOL) or spec["cache_identity"] != cache_identity():
        raise ValueError("fixed DP protocol or original training-only input normalization/goal changed")
    return spec


def prepare(out):
    if out.exists():
        raise FileExistsError("preserve prepared output; use collect --resume after an interrupted collection")
    protocol, identity = read(PROTOCOL), cache_identity()
    used = inventory(out, protocol)
    prepared = read(dp.OUT / "preparation_report.json")
    if prepared["status"] != "prepared":
        raise ValueError("requires the verified DP interface/checkpoint preparation")
    out.mkdir(parents=True)
    spec = {"schema": SCHEMA, "protocol": protocol, "cache_identity": identity,
            "checkpoint_binding": prepared["checkpoint_binding"], "package_versions": prepared["package_versions"],
            "source_splits": [{"split": split, "source_seed": seed} for split, seed in sources(protocol)],
            "test_execution_allowed": False}
    write(out / "protocol.json", protocol)
    write(out / "execution_contract.json", spec)
    write(out / "seed_inventory.json", used)
    # Separate preparation source499999 only; no inspected pilot500000..500007 labels.
    with np.load(dp.OUT / "preparation/context_0.npz", allow_pickle=False) as saved:
        seen = vision.observe_rgb(saved["pixels"])
        if not seen.valid:
            raise ValueError("preparation current RGB feature invalid")
        obs = {"current": seen.features.values[0][None], "agent_xy": saved["agent_xy"][None],
               "actions": saved["actions"][None]}
    # Explicit mechanical fixture; cannot train, cannot be mistaken for measured outcomes.
    synthetic_targets = np.asarray([[0., .25, .5, .75, 1.]])
    probe = write_pack(out / "synthetic_schema_probe", obs, synthetic_targets, synthetic_targets > .95,
                       [{"source_seed": 499999, "anchor_step": 0, "split": "schema_probe"}],
                       split="schema_probe", synthetic=True)
    try:
        pair.PairPack(probe.root, for_training=True)
    except ValueError:
        pass
    else:
        raise AssertionError("synthetic probe accepted for training")
    torch.set_num_threads(1)
    torch.manual_seed(protocol["paired_training_seeds"][0])
    first = pair.DirectActionScorer(identity["normalization"], {"mean": 0., "std": 1.}, goal_grid())
    second = deepcopy(first)
    a, b = first(*probe.inputs(slice(None))), second(*probe.inputs(slice(None)))
    if not torch.equal(a, b) or sum(p.numel() for p in first.parameters()) != 117633:
        raise AssertionError("matched scorer initialization/architecture changed")
    y = torch.tensor(synthetic_targets, dtype=torch.float32)
    la = pair.effect_difference_loss(a, y, pair_weight=0.)
    lb = pair.effect_difference_loss(b, y, pair_weight=1.)
    la["loss"].backward()
    lb["loss"].backward()
    gradients = [torch.cat([p.grad.flatten() for p in model.parameters()]) for model in (first, second)]
    if not all(bool(torch.isfinite(g).all()) for g in gradients) or torch.equal(*gradients):
        raise AssertionError("two loss paths require finite distinct gradients")
    if __package__:
        from .train_diffusion_pusht_action_effect import DPData, success_metrics
        from .train_pusht_action_effect_contrast import selection_metrics
    else:
        from train_diffusion_pusht_action_effect import DPData, success_metrics
        from train_pusht_action_effect_contrast import selection_metrics
    synthetic_scores = {pair.ARMS[0]: synthetic_targets, pair.ARMS[1]: -synthetic_targets}
    keys = np.asarray([499999])
    mechanical = success_metrics(synthetic_targets > .95, synthetic_scores, keys)
    if (mechanical["paired_minus_pointwise"]["source_macro"] != -1.
            or mechanical["methods"]["DP_reference"]["source_macro"]["success"] != 0.
            or mechanical["methods"]["oracle_offline"]["source_macro"]["success"] != 1.
            or mechanical["methods"]["uniform_expectation"]["source_macro"]["success"] != .2):
        raise AssertionError("synthetic success metric fixture changed")
    old_metrics = selection_metrics(synthetic_targets, synthetic_scores, keys,
                                   {k: protocol[k] for k in ("coverage_tie_epsilon", "prediction_tie_epsilon")})
    new_metrics = selection_metrics(synthetic_targets, synthetic_scores, keys, protocol)
    # Exclude the newly requested fixed-index controls for the legacy-default parity check.
    for k in range(5):
        new_metrics["methods"].pop(f"fixed_candidate_{k}", None)
    if json.dumps(old_metrics, sort_keys=True).replace("ACT", "DP") != json.dumps(new_metrics, sort_keys=True):
        raise AssertionError("reference naming changed coverage metric calculations")
    try:
        DPData(out)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete collection accepted by DP training loader")
    report = {"status": "prepared_user_run_collection_next", "schema": SCHEMA,
              "train_sources": 32, "validation_sources": 8, "test_sources_executed": 0,
              "input_shapes": {k: list(v.shape) for k, v in probe.observations.items()},
              "parameter_count_each": 117633, "same_initial_outputs": True,
              "finite_distinct_loss_gradients": True, "synthetic_probe_training_rejected": True,
              "success_metrics_synthetic_fixture_passed": True, "legacy_metric_default_parity": True,
              "incomplete_collection_rejected_by_training_loader": True,
              "data_ready_for_fixed_scorer_experiment": False,
              "normalization_check": "original_inputs_identity_target_scale_only_for_synthetic_fixture",
              "old_pilot_labels_read": False, "environment_steps": 0, "optimizer_steps": 0,
              "hardware_actions": 0, "planned_collection_minutes": [120, 180]}
    write(out / "preparation_report.json", report)
    write(out / "status.json", report)
    print(json.dumps(report), flush=True)
    return 0


def completed(out, protocol):
    return [read(out / "sources" / str(seed) / "done.json") for _, seed in sources(protocol)
            if (out / "sources" / str(seed) / "done.json").exists()]


@torch.inference_mode()
def collect(out, *, resume=False, stop_after_sources=None):
    spec = contract(out)
    protocol = spec["protocol"]
    if len(completed(out, protocol)) == len(sources(protocol)):
        return pack(out)
    if (out / "sources").exists() and not resume:
        raise ValueError("existing attempt; use --resume and preserve incomplete source attempts")
    used = inventory(out, protocol)
    used_ids = set(used["recorded_prior_reset_ids"])
    for record in completed(out, protocol):
        if record["reset_id"] in used_ids:
            raise ValueError("completed source duplicates known reset observation")
        used_ids.add(record["reset_id"])
    rt, active, n = None, None, 0
    handlers = {}
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}: preserve attempt; resume restarts only incomplete source")
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, interrupt)
    try:
        rt = dp.Runtime()
        if rt.binding != spec["checkpoint_binding"] or rt.runtime["_package_versions"]() != spec["package_versions"]:
            raise ValueError("DP checkpoint or runtime changed since preparation")
        for split, seed in sources(protocol):
            root = out / "sources" / str(seed)
            if (root / "done.json").exists():
                continue
            root.mkdir(parents=True, exist_ok=True)
            attempt = root / f"attempt_{len(list(root.glob('attempt_*'))) + 1:03d}"
            attempt.mkdir()
            rt.counts = dict.fromkeys(rt.counts, 0)
            active = {"source_seed": seed, "split": split, "attempt": str(attempt.relative_to(out)), "status": "running"}
            write(out / "status.json", active)
            start = time.monotonic()
            try:
                original = dp.nominal(rt, seed, attempt, anchors=protocol["anchors"], cap=300, used_ids=used_ids)
                reference_deltas = []
                for context in original["contexts"]:
                    for k, valid in enumerate(context["proposal"]["valid"]):
                        if not valid:
                            continue
                        outcome = dp.branch(rt, seed, context, original, k, attempt)
                        if k == 0:
                            reference_deltas.append(outcome["max_reference_coverage_absolute_delta"])
                        print(json.dumps({"source_seed": seed, "split": split, "anchor": context["anchor"],
                                          "candidate": k, "steps": outcome["steps"]}), flush=True)
                    if seed == protocol["future_source_seed_ranges"][split][0]:
                        dp.proposal_sheet(original["raws"][context["anchor"]], context["proposal"]["native"],
                                          context["proposal"]["valid"], attempt / f"candidates_{context['anchor']}_zh.png",
                                          f"{'训练' if split == 'train' else '验证'}首源 {seed} / 起点 {context['anchor']}")
                active.update({"status": "completed_source", "reset_id": original["reset_id"],
                               "skipped_anchors": original["skipped"], "contexts": len(original["contexts"]),
                               "all_reference_actions_observations_discrete_exact": True,
                               "coverage_replay_atol": dp.COVERAGE_REPLAY_ATOL,
                               "max_reference_coverage_absolute_delta": max(reference_deltas, default=0.)})
                used_ids.add(original["reset_id"])
            except BaseException:
                active.update({"status": "incomplete_preserved", "traceback": traceback.format_exc()})
                raise
            finally:
                active.update({"environment_steps": sum(rt.counts.values()), "environment_steps_by_kind": rt.counts.copy(),
                               "seconds": time.monotonic() - start})
                write(attempt / "report.json", active)
            write(root / "done.json", active)
            n += 1
            print(json.dumps({"completed_sources": len(completed(out, protocol)), **active}), flush=True)
            if stop_after_sources and n >= stop_after_sources:
                break
        if len(completed(out, protocol)) == len(sources(protocol)):
            return pack(out)
        write(out / "status.json", {"status": "paused_at_source_boundary", "completed_sources": len(completed(out, protocol))})
        return 0
    except BaseException:
        write(out / "status.json", {"status": "interrupted_or_failed", "active_source": active,
                                   "traceback": traceback.format_exc(), "resume": "--stage collect --resume"})
        raise
    finally:
        if rt:
            rt.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def pack(out):
    spec = contract(out)
    protocol = spec["protocol"]
    records = completed(out, protocol)
    if len(records) != len(sources(protocol)):
        raise ValueError("all40 fixed sources must complete before packing/training")
    ids = [r["reset_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate initial observation across source groups")
    packs, summaries = {}, {}
    for split in SPLITS:
        obs = {key: [] for key in pair.INPUT_FIELDS}
        targets, success, metadata, exclusions, skipped = [], [], [], [], []
        for record in records:
            if record["split"] != split:
                continue
            if record["status"] != "completed_source" or not record["all_reference_actions_observations_discrete_exact"]:
                raise ValueError("unverified source replay")
            folder = out / record["attempt"]
            skipped.extend({"source_seed": record["source_seed"], "anchor_step": a} for a in record["skipped_anchors"])
            results = rows(folder / "diagnostic_outcomes.jsonl") if (folder / "diagnostic_outcomes.jsonl").exists() else []
            contexts = rows(folder / "contexts.jsonl") if (folder / "contexts.jsonl").exists() else []
            keyed = {(r["anchor"], r["candidate"]): r for r in results}
            if len(keyed) != len(results) or any(r["source_seed"] != record["source_seed"] for r in results):
                raise ValueError("duplicate/misaligned measured branch outcome")
            for context in contexts:
                anchor = context["anchor"]
                with np.load(folder / f"context_{anchor}.npz", allow_pickle=False) as saved:
                    current = vision.observe_rgb(saved["pixels"])
                    agent_xy, actions = saved["agent_xy"].copy(), saved["actions"].copy()
                    if not np.array_equal(actions, np.asarray(context["actions"], dtype=np.float32)):
                        raise ValueError("saved proposal metadata differs from observable arrays")
                actual = [keyed.get((anchor, k)) for k in range(5)]
                if not current.valid or not all(context["valid"]):
                    exclusions.append({"source_seed": record["source_seed"], "anchor_step": anchor,
                                       "reason": "current_RGB_invalid_or_not_all5_native_candidates_valid"})
                    continue
                if any(r is None or not (r["terminated"] or r["truncated"]) or r["steps"] < 1
                       or anchor + r["steps"] > 300 for r in actual):
                    raise ValueError("missing/incomplete five-candidate future; no fabricated targets")
                obs["current"].append(current.features.values[0])
                obs["agent_xy"].append(agent_xy)
                obs["actions"].append(actions)
                targets.append([r["final_coverage"] for r in actual])
                success.append([r["success"] for r in actual])
                metadata.append({"source_seed": record["source_seed"], "anchor_step": anchor, "split": split,
                                 "source_attempt": record["attempt"], "reset_id": record["reset_id"]})
        if not metadata:
            raise ValueError(f"no eligible {split} contexts; do not replace seeds or fabricate labels")
        packs[split] = write_pack(out / split / "pack", obs, targets, success, metadata, split=split)
        summaries[split] = {"planned_sources": sum(r["split"] == split for r in records),
                            "eligible_sources": len(packs[split].manifest["source_seeds"]),
                            "eligible_contexts": len(metadata), "exclusions": exclusions, "unreached_anchors": skipped,
                            "informative_coverage_contexts": int((np.ptp(targets, axis=1) > 1e-6).sum()),
                            "mixed_success_contexts": int((np.ptp(np.asarray(success, dtype=int), axis=1) > 0).sum())}
    pair.require_disjoint_sources(*packs.values())
    write(out / "normalization.json", {"input": spec["cache_identity"]["normalization"],
                                       "goal": spec["cache_identity"]["goal"],
                                       "target": pair.training_target_stats(packs["train"].targets), "validation_used": False})
    attempts = [read(p) for p in (out / "sources").glob("*/attempt_*/report.json")]
    report = {"schema": SCHEMA, "status": "completed_train_validation_candidate_pairs", "splits": summaries,
              "source_count": len(records), "distinct_reset_observations": len(set(ids)), "split_source_overlap": [],
              "environment_steps_completed_sources": sum(r["environment_steps"] for r in records),
              "environment_steps_all_recorded_attempts": sum(r["environment_steps"] for r in attempts),
              "data_ready_for_fixed_scorer_experiment": True, "test_sources_executed": 0,
              "optimizer_steps": 0, "hardware_actions": 0, "real_system_validated": False,
              "visual_status": "not_viewed", "old_pilot_used_for_training": False}
    write(out / "report.json", report)
    write(out / "status.json", {"status": report["status"], "completed_sources": len(records)})
    print(json.dumps(report), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "collect", "pack"), required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-sources", type=int)
    args = parser.parse_args()
    if args.stop_after_sources is not None and args.stop_after_sources < 1:
        parser.error("stop-after-sources must be positive")
    if args.stage == "prepare":
        return prepare(args.out)
    if args.stage == "pack":
        return pack(args.out)
    return collect(args.out, resume=args.resume, stop_after_sources=args.stop_after_sources)


if __name__ == "__main__":
    raise SystemExit(main())
