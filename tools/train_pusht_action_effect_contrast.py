"""Fixed-budget, matched Push-T scorers; no ACT, environment or hardware run.

Both arms consume the same full five-candidate batches in lockstep. Only the
loss weight differs. Validation runs after all six fixed-final checkpoints;
it never selects a checkpoint, seed, loss weight or deployment policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

if __package__:
    from . import pusht_action_effect_contrast as pair
    from .pusht_object_dynamics import CACHE_ROOT
else:
    import pusht_action_effect_contrast as pair
    from pusht_object_dynamics import CACHE_ROOT

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "simulation_output/pusht_action_effect_pairs_v1"
PROTOCOL = ROOT / "docs/pusht-action-effect-contrast-protocol-v1.json"
TRAIN_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_action_effect_contrast_v1")
SCHEMA = "pusht_action_effect_training_v1"
CHECKPOINT_EVERY = 100
PARAMETERS = 117633


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_arrays(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


class MatchedData:
    """Read only the completed train/validation packs and original training goal."""

    def __init__(self, root, cache_root):
        protocol = read(PROTOCOL)
        execution = read(root / "execution_contract.json")
        report = read(root / "report.json")
        normalization = read(root / "normalization.json")
        if (execution["protocol"] != protocol
                or report["status"] != "completed_train_validation_candidate_pairs"
                or not report["data_ready_for_fixed_scorer_experiment"]
                or report["test_sources_executed"] != 0):
            raise ValueError("requires the completed, unchanged new paired-data protocol")
        if (protocol["paired_training_seeds"] != [20260918, 20260919, 20260920]
                or protocol["optimizer"] != {"name": "AdamW", "steps_per_arm_per_seed": 2000,
                    "contexts_per_batch": 16, "lr": 0.0003, "weight_decay": 0.0001, "grad_clip": 1.0}
                or protocol["pair_weight"] != 1.0
                or protocol["arms"] != ["ACT_reference", *pair.ARMS]):
            raise ValueError("this entrypoint implements only the frozen three-pair/final2000 protocol")
        self.train = pair.PairPack(root / "train/pack", for_training=True)
        self.validation = pair.PairPack(root / "validation/pack")
        pair.require_disjoint_sources(self.train, self.validation)
        for split, pack in (("train", self.train), ("validation", self.validation)):
            low, high = protocol["future_source_seed_ranges"][split]
            sources = sorted({r["source_seed"] for r in pack.contexts})
            if (pack.manifest["status"] != "completed" or pack.manifest["data_role"] != split
                    or pack.manifest["training_allowed"] != (split == "train")
                    or pack.manifest["source_seeds"] != sources
                    or len(pack.contexts) != report["splits"][split]["eligible_contexts"]
                    or any(not low <= r["source_seed"] <= high or r["split"] != split
                           or r["anchor_step"] not in protocol["anchors"] for r in pack.contexts)):
                raise ValueError(f"{split} cohort does not match the frozen source split")
        identity = execution["cache_identity"]
        manifest = read(cache_root / "manifest.json")
        if ({k: manifest[k] for k in identity} != identity
                or normalization["input"] != identity["normalization"]
                or normalization["goal"] != identity["goal"]
                or normalization["target"] != pair.training_target_stats(self.train.targets)
                or normalization["validation_used"]):
            raise ValueError("input/goal identity or new training-only target statistics changed")
        goal = identity["goal"]
        if ((goal["episode_index"], goal["frame_index"], goal["global_index"]) != (1, 117, 278)
                or goal["episode_index"] not in identity["split"]["train_episodes"]):
            raise ValueError("goal must remain the fixed original training image")
        counts = np.load(cache_root / "grid_counts.npy", mmap_mode="r", allow_pickle=False)
        self.goal = counts[goal["global_index"]].astype(np.float32) / 16
        if self.goal.shape != (24, 24) or not np.isfinite(self.goal).all() or self.goal.max() > 1:
            raise ValueError("invalid original training-goal grid")
        self.contract = {
            "schema": SCHEMA, "protocol": protocol, "normalization": normalization,
            "cache_identity": identity,
            "packs": {"train": self.train.manifest, "validation": self.validation.manifest},
            "collection_environment_steps": report["environment_steps_all_recorded_attempts"],
            "selection": "argmax_raw_coverage_first_index_on_exact_ties_no_clipping",
            "checkpointing": "atomic_joint_pair_every100_updates_and_on_graceful_stop",
            "validation": "all_six_fixed_finals_only_no_selection_no_ensemble",
            "pair_accuracy": "correct_over_all_target_nonties_prediction_tie_counts_not_correct",
            "warm_latency": "one_context_five_candidates_10_warmups_50_timed_forward_calls",
            "environment_steps": 0, "hardware_actions": 0, "test_sources_executed": 0,
        }

    def snapshot(self):
        # Small exact array comparisons on resume, not repeated file hashes.
        arrays = {"goal": self.goal}
        for split, pack in (("train", self.train), ("validation", self.validation)):
            arrays.update({f"{split}_{key}": value for key, value in pack.observations.items()})
            arrays[f"{split}_targets"] = pack.targets
            arrays[f"{split}_keys"] = np.array([(r["source_seed"], r["anchor_step"])
                                                 for r in pack.contexts], dtype=np.int64)
        return arrays


def load_final(path, *, device="cuda", expected_schema=SCHEMA):
    """Inference loader for this experiment only; scores are raw terminal coverage."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["schema"] != expected_schema or payload["kind"] != "final"
            or payload["arm"] not in pair.ARMS or payload["step"] != 2000
            or payload["training_seed"] not in [20260918, 20260919, 20260920]):
        raise ValueError("requires this experiment's fixed-final2000 checkpoint")
    norm = payload["contract"]["normalization"]
    model = pair.DirectActionScorer(norm["input"], norm["target"], payload["model"]["goal"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), payload


def optimizer_steps_match(optimizer, step):
    states = list(optimizer.state.values())
    if step == 0:
        return not states
    return (len(states) == sum(len(g["params"]) for g in optimizer.param_groups)
            and all(int(s["step"]) == step for s in states))


def train_seed(data, root, seed, stop):
    """One checkpoint holds both optimizers at the same completed paired update."""
    out = root / str(seed)
    out.mkdir(exist_ok=True)
    contract = data.contract
    schema = contract.get("training_schema", SCHEMA)
    budget = contract["protocol"]["optimizer"]
    steps, batch = budget["steps_per_arm_per_seed"], budget["contexts_per_batch"]
    if (out / "training_report.json").exists():
        done = read(out / "training_report.json")
        if done["contract"] != contract or done["optimizer_steps_per_arm"] != steps:
            raise ValueError("completed training pair belongs to a different contract")
        return done

    torch.manual_seed(seed)
    norm = contract["normalization"]
    initial = pair.DirectActionScorer(norm["input"], norm["target"], data.goal)
    if sum(p.numel() for p in initial.parameters()) != PARAMETERS:
        raise ValueError("unchanged scorer parameter count required")
    initial_path = out / "initial_state.pt"
    if initial_path.exists():
        saved = torch.load(initial_path, map_location="cpu", weights_only=True)
        if not all(torch.equal(value, saved[key]) for key, value in initial.state_dict().items()):
            raise ValueError("initialization changed on resume")
    else:
        save_torch(initial_path, initial.state_dict())
    models = {arm: deepcopy(initial).cuda() for arm in pair.ARMS}
    del initial
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=budget["lr"],
                                       weight_decay=budget["weight_decay"])
                  for arm, model in models.items()}
    schedule = np.random.default_rng(seed).integers(len(data.train.contexts), size=(steps, batch), dtype=np.int32)
    schedule_path = out / "sampled_rows.npz"
    if schedule_path.exists():
        with np.load(schedule_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["rows"], schedule):
                raise ValueError("ordered batch schedule changed on resume")
    else:
        save_arrays(schedule_path, rows=schedule)
    schedule = torch.as_tensor(schedule, device="cuda", dtype=torch.long)
    inputs = data.train.inputs(slice(None), device="cuda")
    targets = torch.as_tensor(data.train.targets, device="cuda", dtype=torch.float32)
    targets = (targets - models[pair.ARMS[0]].target_mean) / models[pair.ARMS[0]].target_std
    first, seconds = 0, dict.fromkeys(pair.ARMS, 0.)
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        if saved["contract"] != contract or saved["training_seed"] != seed or saved["schema"] != schema:
            raise ValueError("cannot resume a different paired experiment")
        first, seconds = saved["step"], saved["training_seconds_by_arm"]
        if not 0 <= first <= steps:
            raise ValueError("resume step outside frozen budget")
        for arm in pair.ARMS:
            models[arm].load_state_dict(saved["models"][arm], strict=True)
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
            if not optimizer_steps_match(optimizers[arm], first):
                raise ValueError("resume optimizers must have equal completed step counts")
        del saved

    def save(step):
        save_torch(out / "last.pt", {
            "schema": schema, "kind": "paired_resume", "training_seed": seed,
            "step": step, "next_schedule_row": step, "contract": contract,
            "models": {arm: model.state_dict() for arm, model in models.items()},
            "optimizers": {arm: optimizer.state_dict() for arm, optimizer in optimizers.items()},
            "training_seconds_by_arm": seconds,
        })

    if first == 0:
        save(0)
    if stop["requested"]:
        save(first)
        return None
    for model in models.values():
        model.train()
    interval = {arm: dict.fromkeys(("loss", "pointwise", "effect_difference"), 0.) for arm in pair.ARMS}
    count = 0
    for step in range(first + 1, steps + 1):
        rows = schedule[step - 1]
        shared_inputs = tuple(value[rows] for value in inputs)
        shared_targets = targets[rows]
        for arm, weight in zip(pair.ARMS, (0., 1.)):
            model, optimizer = models[arm], optimizers[arm]
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = pair.effect_difference_loss(model(*shared_inputs), shared_targets, pair_weight=weight)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), budget["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            seconds[arm] += time.perf_counter() - started
            for name, value in losses.items():
                interval[arm][name] += float(value.detach())
        # No checkpoint/stop between arms: a hard failure resumes the last joint checkpoint.
        count += 1
        if step % CHECKPOINT_EVERY == 0 or stop["requested"]:
            save(step)
            row = {"training_seed": seed, "paired_step": step, "resume_from_step": first,
                   "loss_means": {arm: {k: v / count for k, v in values.items()}
                                  for arm, values in interval.items()},
                   "training_seconds_by_arm": dict(seconds)}
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            write(root / "status.json", {"status": "interrupted" if stop["requested"] else "training",
                                          "training_seed": seed, "paired_step": step})
            print(json.dumps(row), flush=True)
            count = 0
            for values in interval.values():
                values.update(dict.fromkeys(values, 0.))
        if stop["requested"]:
            return None
    for arm in pair.ARMS:
        if not optimizer_steps_match(optimizers[arm], steps):
            raise ValueError("final optimizer did not reach exactly 2000 updates")
        (out / arm).mkdir(exist_ok=True)
        save_torch(out / arm / "final.pt", {
            "schema": schema, "kind": "final", "arm": arm, "training_seed": seed,
            "step": steps, "contract": contract,
            "model": {k: v.detach().cpu() for k, v in models[arm].state_dict().items()},
            "prediction_units": "raw_terminal_coverage_not_delta_or_probability",
        })
    result = {"status": "completed_fixed_training_pair", "contract": contract, "training_seed": seed,
              "optimizer_steps_per_arm": steps, "parameter_count_each": PARAMETERS,
              "same_initial_state_dict": True, "same_shared_batch_tensor_each_step": True,
              "final_optimizer_steps_verified": True, "resume_from_step": first,
              "training_seconds_by_arm": seconds, "validation_used_in_training": False,
              "environment_steps": 0, "hardware_actions": 0}
    write(out / "training_report.json", result)
    return result


def selection_metrics(target, scores, sources, protocol, *, fixed_candidates=False):
    """Same coverage definitions as the previous comparison, without environment imports."""
    eps, pred_eps = protocol["coverage_tie_epsilon"], protocol["prediction_tie_epsilon"]
    reference_name = protocol.get("reference_name", "ACT")
    source_ids = np.unique(sources)
    informative = np.ptp(target, axis=1) > eps
    truth_pairs = target[:, np.triu_indices(5, 1)[0]] - target[:, np.triu_indices(5, 1)[1]]
    selected = {arm: values.argmax(1) for arm, values in scores.items()}
    weights = {f"{reference_name}_reference": np.tile(np.eye(5)[0], (len(target), 1)),
               "uniform_expectation": np.full_like(target, .2),
               "oracle_offline": np.eye(5)[target.argmax(1)],
               **{arm: np.eye(5)[ids] for arm, ids in selected.items()}}
    if fixed_candidates:
        weights.update({f"fixed_candidate_{k}": np.tile(np.eye(5)[k], (len(target), 1)) for k in range(5)})
    gain = target - target[:, :1]
    methods = {}
    for name, weight in weights.items():
        coverage = (weight * target).sum(1)
        values = {"coverage": coverage, f"gain_vs_{reference_name}": coverage - target[:, 0],
                  "regret": target.max(1) - coverage, "harm_rate": (weight * (gain < -eps)).sum(1),
                  "better_rate": (weight * (gain > eps)).sum(1),
                  "tie_rate": (weight * (np.abs(gain) <= eps)).sum(1),
                  "headroom": target.max(1) - target[:, 0]}
        if name in scores:
            values["coverage_mae"] = np.abs(scores[name] - target).mean(1)
        per_source = {str(int(s)): {k: float(v[sources == s].mean()) for k, v in values.items()}
                      for s in source_ids}
        top1 = (weight * (target >= target.max(1)[:, None] - eps)).sum(1)
        info_by_source = {str(int(s)): float(top1[informative & (sources == s)].mean())
                          for s in np.unique(sources[informative])}
        methods[name] = {
            "context_mean": {k: float(v.mean()) for k, v in values.items()},
            "source_seed_macro": {k: float(np.mean([row[k] for row in per_source.values()])) for k in values},
            "per_source_seed": per_source,
            f"worst_supported_gain_vs_{reference_name}": float(gain[weight > 0].min()),
            "better_tie_harm_counts_or_expectation": {k: float(values[k + "_rate"].sum())
                                                       for k in ("better", "tie", "harm")},
            "informative_top1": {"context_mean": float(top1[informative].mean()) if informative.any() else None,
                                 "source_seed_macro": float(np.mean(list(info_by_source.values()))) if info_by_source else None,
                                 "per_source_seed": info_by_source},
        }
    agreement = {}
    i, j = np.triu_indices(5, 1)
    comparable = np.abs(truth_pairs) > eps
    for name, prediction in scores.items():
        gap = prediction[:, i] - prediction[:, j]
        correct = comparable & (np.abs(gap) > pred_eps) & (gap * truth_pairs > 0)
        estimated_tie = comparable & (np.abs(gap) <= pred_eps)

        def counts(mask):
            n, c, ties = int(comparable[mask].sum()), int(correct[mask].sum()), int(estimated_tie[mask].sum())
            return {"comparable": n, "correct": c, "wrong": n - c - ties, "estimated_tie": ties,
                    "target_tie": int((~comparable[mask]).sum()), "accuracy": c / n if n else None}

        per_source = {str(int(s)): counts(sources == s) for s in source_ids}
        accuracies = [v["accuracy"] for v in per_source.values() if v["accuracy"] is not None]
        agreement[name] = {"pooled_pairs": counts(np.ones(len(target), dtype=bool)),
                           "source_seed_macro_accuracy": float(np.mean(accuracies)) if accuracies else None,
                           "per_source_seed": per_source}
    delta = target[np.arange(len(target)), selected[pair.ARMS[1]]] - target[np.arange(len(target)), selected[pair.ARMS[0]]]
    per_source_delta = {str(int(s)): float(delta[sources == s].mean()) for s in source_ids}
    return {"contexts": len(target), "sources": len(source_ids), "informative_contexts": int(informative.sum()),
            "informative_sources": len(np.unique(sources[informative])), "methods": methods,
            "pair_agreement": agreement,
            "paired_minus_pointwise": {"context_mean": float(delta.mean()),
                "source_seed_macro": float(np.mean(list(per_source_delta.values()))),
                "per_source_seed": per_source_delta,
                "better_tie_worse_contexts": [int((delta > eps).sum()), int((np.abs(delta) <= eps).sum()), int((delta < -eps).sum())]}}


@torch.inference_mode()
def evaluate_final_seed(data, root, seed, stop):
    out = root / str(seed)
    pack, scores, latency = data.validation, {}, {}
    inputs = pack.inputs(slice(None), device="cuda")
    one = tuple(value[:1] for value in inputs)
    for arm in pair.ARMS:
        if stop["requested"]:
            return None
        model, payload = load_final(out / arm / "final.pt", expected_schema=data.contract.get("training_schema", SCHEMA))
        if payload["contract"] != data.contract or payload["training_seed"] != seed or payload["arm"] != arm:
            raise ValueError("final checkpoint does not match the frozen pair")
        # The legacy method name predict_effect returns raw terminal coverage here.
        scores[arm] = model.predict_effect(*inputs).cpu().numpy().astype(np.float64)
        if not np.isfinite(scores[arm]).all():
            raise ValueError("nonfinite final validation scores")
        for _ in range(10):
            model.predict_effect(*one)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(50):
            model.predict_effect(*one)
        torch.cuda.synchronize()
        latency[arm] = 1000 * (time.perf_counter() - started) / 50
        del model, payload
    keys = np.array([(r["source_seed"], r["anchor_step"]) for r in pack.contexts], dtype=np.int64)
    # Persist both predictions without targets before joining future labels for metrics.
    save_arrays(out / "validation_predictions.npz", context_keys=keys,
                **{arm + "_scores": values for arm, values in scores.items()},
                **{arm + "_selected": values.argmax(1) for arm, values in scores.items()})
    result = {"status": "completed_fixed_final_validation", "training_seed": seed,
              "metrics": selection_metrics(pack.targets, scores, keys[:, 0], data.contract["protocol"],
                                           fixed_candidates=data.contract["protocol"].get("report_fixed_candidates", False)),
              "warm_scoring_ms_one_context_five_candidates": latency,
              "validation_for_checkpoint_selection": False, "fresh_test": False,
              "closed_loop_policy_benefit_evaluated": False, "environment_steps": 0, "hardware_actions": 0}
    write(out / "validation_report.json", result)
    return result


def run(args, stop):
    if not torch.cuda.is_available():
        raise RuntimeError("use project2026-pi on the remote 4090; this entrypoint requires CUDA")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    data = MatchedData(args.data_root, args.cache_root)
    contract = data.contract
    if args.resume:
        if read(args.out / "run.json")["contract"] != contract:
            raise ValueError("cannot resume with changed data manifests or protocol")
        with np.load(args.out / "data_snapshot.npz", allow_pickle=False) as saved:
            expected = data.snapshot()
            if set(saved.files) != set(expected) or not all(np.array_equal(saved[k], v) for k, v in expected.items()):
                raise ValueError("observations, targets, row order or training goal changed on resume")
        if (args.out / "report.json").exists():
            print(f"Already completed; no additional training: {args.out / 'report.json'}", flush=True)
            return 0
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        save_arrays(args.out / "data_snapshot.npz", **data.snapshot())
        write(args.out / "run.json", {"schema": SCHEMA, "contract": contract,
              "invocation": "explicit_--train", "data_root": str(args.data_root.resolve()),
              "cache_root": str(args.cache_root.resolve()), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "device": "cuda", "precision": "float32_no_TF32_no_AMP",
              "deterministic_algorithms": True, "source_entrypoint": str(Path(__file__).resolve())})
    training, validation = {}, {}
    seeds = contract["protocol"]["paired_training_seeds"]
    for seed in seeds:
        if stop["requested"]:
            break
        training[str(seed)] = train_seed(data, args.out, seed, stop)
    if not stop["requested"]:
        # All three pairs are final before the first validation inference.
        write(args.out / "status.json", {"status": "all_six_finals_saved_validation_pending"})
        for seed in seeds:
            validation[str(seed)] = evaluate_final_seed(data, args.out, seed, stop)
            if stop["requested"]:
                break
    if stop["requested"]:
        write(args.out / "status.json", {"status": "interrupted", "resume_with": "--train --resume"})
        return 2
    deltas = np.array([validation[str(s)]["metrics"]["paired_minus_pointwise"]["source_seed_macro"] for s in seeds])
    write(args.out / "report.json", {
        "schema": SCHEMA, "status": "completed_fixed_paired_training_and_validation", "contract": contract,
        "training": training, "validation": validation,
        "paired_validation_delta_across_training_seeds": {"values": dict(zip(map(str, seeds), deltas.tolist())),
            "mean": float(deltas.mean()), "training_seed_sample_std": float(deltas.std(ddof=1)),
            "min": float(deltas.min()), "max": float(deltas.max()),
            "independent_test_trajectories": False, "test_confidence_interval": None},
        "optimizer_steps_each_arm_each_seed": 2000, "total_optimizer_steps": 12000,
        "parameter_count_each": PARAMETERS, "environment_steps": 0, "hardware_actions": 0,
        "test_sources_executed": 0, "validation_used_for_selection": False,
        "decision": "descriptive_validation_only_keep_ACT_real10_no_promotion_pending_fixed_fresh_test",
    })
    write(args.out / "status.json", {"status": "completed", "total_optimizer_steps": 12000})
    print(json.dumps({"status": "completed_fixed_paired_training_and_validation", "out": str(args.out),
                      "paired_validation_deltas": deltas.tolist()}, indent=2), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", required=True,
                        help="explicitly execute all three matched pairs at the frozen budget")
    parser.add_argument("--resume", action="store_true", help="resume joint checkpoints; skip finished seeds")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out", type=Path, default=TRAIN_ROOT)
    args = parser.parse_args()
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True
        print("Stop requested; finish both arms of this update and save one joint checkpoint.", flush=True)

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    return run(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
