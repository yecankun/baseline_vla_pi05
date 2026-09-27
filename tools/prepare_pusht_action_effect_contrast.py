"""Prepare a development-only paired interface check; never train or step an env.

The inspected dev20 outcomes are only schema/gradient evidence. They are not
silently reassigned to training or promoted to an independent test set.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

if __package__:
    from . import pusht_action_effect_contrast as effect
    from . import pusht_object_dynamics as base
    from . import pusht_object_goal as vision
    from .pusht_world_model_adapter import make_candidates
else:
    import pusht_action_effect_contrast as effect
    import pusht_object_dynamics as base
    import pusht_object_goal as vision
    from pusht_world_model_adapter import make_candidates

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulation_output/pusht_fresh_scorer_pair20_v1"
OUT = ROOT / "simulation_output/pusht_action_effect_contrast_prepare_v1"
PROTOCOL = ROOT / "docs/pusht-action-effect-contrast-protocol-v1.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def reserved_seed_inventory(protocol, out):
    """Check recorded project seed IDs, not unknown original dataset init states."""
    used = set()

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if (key == "seed" or key.endswith("_seed")) and isinstance(child, int):
                    used.add(child)
                if "seeds" in key and isinstance(child, list):
                    used.update(v for v in child if isinstance(v, int))
                if key == "per_seed" and isinstance(child, dict):
                    used.update(int(v) for v in child if v.isdigit())
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    paths = []
    for folder in sorted((ROOT / "simulation_output").glob("*pusht*")):
        if not folder.is_dir() or folder.resolve() == out.resolve():
            continue
        for name in ("report.json", "started.json", "protocol.json", "manifest.json", "resets.jsonl"):
            path = folder / name
            if path.is_file():
                walk(rows(path) if path.suffix == ".jsonl" else read(path))
                paths.append(str(path.relative_to(ROOT)))
    reserved = []
    for bounds in protocol["future_source_seed_ranges"].values():
        reserved.extend(range(bounds[0], bounds[1] + 1))
    if len(reserved) != len(set(reserved)) or used.intersection(reserved):
        raise ValueError("proposed grouped seed ranges overlap each other or existing project runs")
    return {"files_checked": len(paths), "metadata_files": paths, "prior_seed_ids": sorted(used),
            "overlap": [], "freshness_scope": "recorded_project_metadata_only_not_unknown_dataset_generation",
            "collection_must_recheck_initial_observation_identity": True}


def run(out):
    start = time.monotonic()
    protocol = read(PROTOCOL)
    if protocol["schema"] != effect.SCHEMA:
        raise ValueError("protocol/schema mismatch")
    if out.exists():
        raise FileExistsError("preserve earlier evidence; choose a new --out for an explicit retry")
    inventory = reserved_seed_inventory(protocol, out)
    source_report = read(SOURCE / "report.json")
    if (source_report["status"] != "completed_fresh_seed_candidate_comparison"
            or not source_report["all_replayed_observations_exact"]
            or not source_report["reference_continuations_exact"]):
        raise ValueError("requires recorded same-start execution provenance")

    # First build observable inputs without loading candidate outcomes or old scores.
    observations = {name: [] for name in effect.INPUT_FIELDS}
    contexts, exclusions = [], []
    for row in rows(SOURCE / "predictions.jsonl"):
        with Image.open(SOURCE / row["current_rgb"]) as image:
            seen = vision.observe_rgb(np.asarray(image.convert("RGB")))
        if seen.valid != row["current_valid"]:
            raise ValueError("current RGB extraction changed")
        actions = np.asarray(row["actions"], dtype=np.float32)
        rebuilt = make_candidates(torch.from_numpy(actions[0][None]), offset_xy=8.0)
        if not np.array_equal(actions, rebuilt.actions[0].numpy()) or row["valid"] != rebuilt.valid[0].tolist():
            raise ValueError("recorded candidate actions differ from unchanged ACT+ramp rule")
        if not seen.valid or not all(row["valid"]):
            exclusions.append({"seed": row["seed"], "anchor_step": row["anchor_step"],
                               "reason": "current_observation_or_candidate_invalid"})
            continue
        observations["current"].append(seen.features.values[0])
        observations["agent_xy"].append(row["agent_xy"])
        observations["actions"].append(actions)
        contexts.append({"source_seed": row["seed"], "anchor_step": row["anchor_step"],
                         "current_rgb": row["current_rgb"], "role": "development_schema_probe"})
    observations = {k: np.asarray(v, dtype=np.float32) for k, v in observations.items()}
    out.mkdir(parents=True)
    np.savez_compressed(out / "observations.npz", **observations)
    (out / "contexts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in contexts), encoding="utf-8")

    # Immutable original train-only state/action normalization and goal. Do not
    # estimate any normalization from the inspected candidate outcomes.
    cache = read(base.CACHE_ROOT / "manifest.json")
    if cache["schema"] != base.SCHEMA or cache["status"] != "prepared":
        raise ValueError("original train-only feature cache unavailable")
    goal_index = cache["goal"]["global_index"]
    if goal_index != 278 or 1 not in cache["split"]["train_episodes"]:
        raise ValueError("fixed training goal changed")
    goal = np.load(base.CACHE_ROOT / "grid_counts.npy", mmap_mode="r", allow_pickle=False)[goal_index].astype(np.float32) / 16
    torch.set_num_threads(1)
    torch.manual_seed(protocol["paired_training_seeds"][0])
    # Identity target scale for mechanical verification only; not a fitted or
    # proposed training normalization, and no checkpoint is written.
    point = effect.DirectActionScorer(cache["normalization"], {"mean": 0.0, "std": 1.0}, goal)
    pair = copy.deepcopy(point)
    inputs = tuple(torch.from_numpy(observations[k]) for k in effect.INPUT_FIELDS)
    p, q = point(*inputs), pair(*inputs)
    if not torch.equal(p, q):
        raise ValueError("identical initialized arms must produce identical scores")
    np.savez_compressed(out / "random_forward.npz", pointwise=p.detach().numpy(), paired=q.detach().numpy())

    # Only now join actual independently executed candidate futures, as labels.
    outcomes = rows(SOURCE / "outcomes.jsonl")
    keyed = {(r["seed"], r["anchor_step"], r["candidate"]): r for r in outcomes}
    if len(keyed) != len(outcomes):
        raise ValueError("duplicate actual outcome")
    targets = []
    for context in contexts:
        actual = [keyed.get((context["source_seed"], context["anchor_step"], k)) for k in range(5)]
        if any(r is None or not r["full_horizon"] or r["steps"] != 8 for r in actual):
            raise ValueError("missing/short executed candidate; no invented completion or negative")
        targets.append([r["terminal_coverage"] for r in actual])
    targets = np.asarray(targets, dtype=np.float64)
    np.savez_compressed(out / "targets.npz", terminal_coverage=targets)
    manifest = {"schema": effect.SCHEMA, "data_role": "development_schema_probe", "training_allowed": False,
                "source": str(SOURCE), "contexts": len(contexts), "input_fields": list(effect.INPUT_FIELDS),
                "target_fields": ["terminal_coverage"], "future_targets_are_observations": False,
                "same_start_provenance": "source_report_exact_reset_prefix_and_ACT_reference_replay",
                "fresh_test": False, "new_executions": 0}
    write(out / "manifest.json", manifest)
    loaded = effect.PairPack(out)
    try:
        effect.PairPack(out, for_training=True)
    except ValueError:
        training_rejected = True
    else:
        raise AssertionError("development probe must be rejected for training")
    y = torch.as_tensor(loaded.targets, dtype=torch.float32)
    losses = [effect.effect_difference_loss(p, y, pair_weight=0.0),
              effect.effect_difference_loss(q, y, pair_weight=1.0)]
    losses[0]["loss"].backward()
    losses[1]["loss"].backward()
    gradients = [torch.cat([v.grad.flatten() for v in m.parameters()]) for m in (point, pair)]
    if not all(bool(torch.isfinite(g).all()) for g in gradients) or torch.equal(*gradients):
        raise ValueError("expected finite, different loss gradients; no optimizer is used")
    # Algebra checks document precisely what the pair term can and cannot do.
    error = q.detach() - y
    equivalent = error.var(dim=1, unbiased=False).mean() * 5 / 4
    pair_term = losses[1]["effect_difference"].detach()
    permutation = [3, 0, 4, 1, 2]
    permuted = effect.effect_difference_loss(q.detach()[:, permutation], y[:, permutation], pair_weight=1.0)
    shifted = effect.effect_difference_loss(q.detach() + .125, y, pair_weight=1.0)
    if not (torch.allclose(pair_term, equivalent, atol=1e-7)
            and torch.allclose(pair_term, shifted["effect_difference"], atol=1e-7)
            and torch.allclose(losses[1]["loss"], permuted["loss"], atol=1e-7)):
        raise AssertionError("pair weighting or candidate-order invariance changed")
    i, j = np.triu_indices(5, 1)
    gap = targets[:, i] - targets[:, j]
    report = {"status": "prepared_interface_only_missing_independent_training_pairs", "schema": effect.SCHEMA,
        "source_contexts": len(rows(SOURCE / "predictions.jsonl")), "eligible_contexts": len(contexts),
        "source_seeds": len({r["source_seed"] for r in contexts}), "actual_candidates": int(targets.size),
        "all_pairs_including_ties": int(gap.size), "informative_pairs": int((np.abs(gap) > 1e-6).sum()),
        "informative_contexts": int((np.ptp(targets, axis=1) > 1e-6).sum()), "exclusions": exclusions,
        "arm_parameter_counts": {name: sum(p.numel() for p in point.parameters()) for name in effect.ARMS},
        "identical_initialization_and_outputs": True, "finite_distinct_loss_gradients": True,
        "pair_term_equals_reweighted_within_context_error_variance": True,
        "pair_term_invariant_to_context_constant_and_candidate_permutation": True,
        "development_pack_rejected_for_training": training_rejected,
        "target_normalization_for_check": "identity_only_not_fitted_not_a_training_statistic",
        "observations_and_random_forward_written_before_target_join": True,
        "forbidden_model_inputs": ["coverage", "future_RGB", "old_scores", "seed", "anchor_step", "role", "object_pose"],
        "reserved_seed_inventory": inventory, "protocol": protocol,
        "training_ready": False, "optimizer_steps": 0, "environment_steps": 0, "hardware_actions": 0,
        "trained_weights_written": False, "policy_benefit_evaluated": False,
        "missing": ["new independent same-start training candidate executions", "new source-grouped validation candidate executions",
                    "matched training runner and final-only fresh-test evaluator"],
        "runtime_seconds": time.monotonic() - start}
    write(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("protocol", "reserved_seed_inventory")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
