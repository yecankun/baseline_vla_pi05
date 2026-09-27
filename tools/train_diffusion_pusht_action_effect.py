"""DP-matched scorer pair using the existing shared training core.

Explicit --train only, all six fixed final checkpoints before validation.
No DP inference, new environment step, old pilot label, or hardware execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

import numpy as np
import torch

if __package__:
    from . import train_pusht_action_effect_contrast as shared
    from . import collect_diffusion_pusht_action_effect as data_source
else:
    import train_pusht_action_effect_contrast as shared
    import collect_diffusion_pusht_action_effect as data_source

pair = shared.pair
read, write, save_arrays = shared.read, shared.write, shared.save_arrays
SCHEMA = "diffusion_pusht_action_effect_training_v1"
OUT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/diffusion_pusht_action_effect_v1")


class DPData:
    def __init__(self, root):
        root = Path(root)
        if not (root / "report.json").is_file():
            raise ValueError("finish all40 fixed DP train/validation sources and pack first; no training started")
        spec = data_source.contract(root)
        protocol, report = spec["protocol"], read(root / "report.json")
        norm = read(root / "normalization.json")
        if (report["schema"] != data_source.SCHEMA
                or report["status"] != "completed_train_validation_candidate_pairs"
                or not report["data_ready_for_fixed_scorer_experiment"] or report["source_count"] != 40
                or report["test_sources_executed"] or protocol["reference_name"] != "DP"
                or protocol["paired_training_seeds"] != [20260918, 20260919, 20260920]
                or protocol["optimizer"] != {"name": "AdamW", "steps_per_arm_per_seed": 2000,
                    "contexts_per_batch": 16, "lr": .0003, "weight_decay": .0001, "grad_clip": 1.}
                or protocol["pair_weight"] != 1.):
            raise ValueError("requires the completed fixed DP experiment, not ACT or diagnostic data")
        self.train = pair.PairPack(root / "train/pack", for_training=True)
        self.validation = pair.PairPack(root / "validation/pack")
        pair.require_disjoint_sources(self.train, self.validation)
        self.success = {}
        for split, pack in (("train", self.train), ("validation", self.validation)):
            low, high = protocol["future_source_seed_ranges"][split]
            source_ids = sorted({r["source_seed"] for r in pack.contexts})
            if (pack.manifest["proposal_family"] != data_source.SCHEMA
                    or pack.manifest["synthetic_unit_fixture"] or pack.manifest["data_role"] != split
                    or pack.manifest["training_allowed"] != (split == "train")
                    or pack.manifest["target_semantics"] != "candidate_then_DP_until_first_done_or_global300"
                    or pack.manifest["source_seeds"] != source_ids
                    or report["splits"][split]["eligible_contexts"] != len(pack.contexts)
                    or any(not low <= r["source_seed"] <= high or r["split"] != split
                           or r["anchor_step"] not in protocol["anchors"] for r in pack.contexts)):
                raise ValueError("pack split, candidate family or target semantics mismatch")
            with np.load(pack.root / "evaluation_targets.npz", allow_pickle=False) as saved:
                if saved.files != ["success"] or saved["success"].dtype != bool or saved["success"].shape != pack.targets.shape:
                    raise ValueError("actual success sidecar must align with the same candidates")
                self.success[split] = saved["success"].copy()
        if (norm["input"] != spec["cache_identity"]["normalization"]
                or norm["goal"] != spec["cache_identity"]["goal"] or norm["validation_used"]
                or norm["target"] != pair.training_target_stats(self.train.targets)):
            raise ValueError("train-only normalization changed or used validation")
        self.goal = data_source.goal_grid()
        self.contract = {"training_schema": SCHEMA, "protocol": protocol, "normalization": norm,
                         "cache_identity": spec["cache_identity"], "DP_checkpoint_binding": spec["checkpoint_binding"],
                         "packs": {"train": self.train.manifest, "validation": self.validation.manifest},
                         "collection_environment_steps": report["environment_steps_all_recorded_attempts"],
                         "validation": "all_six_fixed_finals_no_selection_no_ensemble",
                         "selection": "argmax_raw_terminal_coverage_first_index_on_exact_ties",
                         "success_sidecar_usage": "offline_evaluation_only_not_forward_or_loss",
                         "environment_steps": 0, "hardware_actions": 0, "test_sources_executed": 0}

    def snapshot(self):
        arrays = shared.MatchedData.snapshot(self)
        arrays.update({f"{split}_success": value for split, value in self.success.items()})
        return arrays


def load_final(path, *, device="cuda"):
    model, payload = shared.load_final(path, device=device, expected_schema=SCHEMA)
    if payload["contract"]["protocol"] != read(data_source.PROTOCOL):
        raise ValueError("checkpoint does not match the frozen DP scorer protocol")
    return model, payload


def success_metrics(success, scores, sources):
    """Descriptive same-candidate selection, source macro; no independent test claim."""
    y = np.asarray(success, dtype=np.float64)
    eye = np.eye(5)
    methods = {"DP_reference": np.tile(eye[0], (len(y), 1)),
               "uniform_expectation": np.full_like(y, .2), "oracle_offline": eye[y.argmax(1)],
               **{f"fixed_candidate_{k}": np.tile(eye[k], (len(y), 1)) for k in range(5)},
               **{arm: eye[value.argmax(1)] for arm, value in scores.items()}}
    result = {}
    for method, weight in methods.items():
        selected = (weight * y).sum(1)
        values = {"success": selected, "gain_vs_DP": selected - y[:, 0],
                  "rescue": (1 - y[:, 0]) * selected, "harm": y[:, 0] * (1 - selected),
                  "regret_to_oracle": y.max(1) - selected}
        per_source = {str(int(seed)): {name: float(value[sources == seed].mean()) for name, value in values.items()}
                      for seed in np.unique(sources)}
        result[method] = {"source_macro": {name: float(np.mean([r[name] for r in per_source.values()])) for name in values},
                          "context_mean": {name: float(value.mean()) for name, value in values.items()},
                          "per_source": per_source}
    delta = y[np.arange(len(y)), scores[pair.ARMS[1]].argmax(1)] - y[np.arange(len(y)), scores[pair.ARMS[0]].argmax(1)]
    per_source = {str(int(s)): float(delta[sources == s].mean()) for s in np.unique(sources)}
    return {"contexts": len(y), "sources": len(per_source), "methods": result,
            "paired_minus_pointwise": {"source_macro": float(np.mean(list(per_source.values()))),
                                        "per_source": per_source},
            "fresh_test": False, "repeated_closed_loop_selector": False}


def run(args, stop):
    data = DPData(args.data_root)
    if args.check_data:
        print(json.dumps({"status": "DP_training_data_ready", "contexts": {
            "train": len(data.train.contexts), "validation": len(data.validation.contexts)},
            "optimizer_steps": 0, "environment_steps": 0}), flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("run training in project2026-pi on4090")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.resume:
        if read(args.out / "run.json")["contract"] != data.contract:
            raise ValueError("cannot resume changed DP experiment")
        with np.load(args.out / "data_snapshot.npz", allow_pickle=False) as saved:
            expected = data.snapshot()
            if set(saved.files) != set(expected) or any(not np.array_equal(saved[k], v) for k, v in expected.items()):
                raise ValueError("pack arrays/order or fixed goal changed on resume")
        if (args.out / "report.json").exists():
            print("Already completed; no additional training.", flush=True)
            return 0
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        save_arrays(args.out / "data_snapshot.npz", **data.snapshot())
        write(args.out / "run.json", {"schema": SCHEMA, "contract": data.contract,
                                      "invocation": "explicit_--train", "data_root": str(args.data_root.resolve()),
                                      "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                                      "precision": "float32_no_TF32_no_AMP"})
    seeds = data.contract["protocol"]["paired_training_seeds"]
    training, validation = {}, {}
    for seed in seeds:
        if stop["requested"]:
            break
        training[str(seed)] = shared.train_seed(data, args.out, seed, stop)
    if not stop["requested"]:
        for seed in seeds:
            result = shared.evaluate_final_seed(data, args.out, seed, stop)
            if stop["requested"]:
                break
            with np.load(args.out / str(seed) / "validation_predictions.npz", allow_pickle=False) as saved:
                keys = saved["context_keys"]
                scores = {arm: saved[arm + "_scores"].copy() for arm in pair.ARMS}
            result["success_metrics"] = success_metrics(data.success["validation"], scores, keys[:, 0])
            write(args.out / str(seed) / "validation_report.json", result)
            validation[str(seed)] = result
    if stop["requested"]:
        write(args.out / "status.json", {"status": "interrupted", "resume": "--train --resume"})
        return 2
    deltas = [validation[str(s)]["success_metrics"]["paired_minus_pointwise"]["source_macro"] for s in seeds]
    report = {"schema": SCHEMA, "status": "completed_fixed_DP_paired_training_and_validation",
              "contract": data.contract, "training": training, "validation": validation,
              "primary_success_delta_across_training_seeds": {"values": dict(zip(map(str, seeds), deltas)),
                 "mean": float(np.mean(deltas)), "training_seed_sample_std": float(np.std(deltas, ddof=1)),
                 "independent_test_trajectories": False, "test_confidence_interval": None},
              "total_optimizer_steps": 12000, "optimizer_steps_each_arm_each_seed": 2000,
              "parameter_count_each": 117633, "environment_steps": 0, "hardware_actions": 0,
              "test_sources_executed": 0, "validation_used_for_selection": False,
              "decision": "development_validation_only_no_promotion_no_test_or_pilot_reuse"}
    write(args.out / "report.json", report)
    write(args.out / "status.json", {"status": "completed", "total_optimizer_steps": 12000})
    print(json.dumps({"status": report["status"], "primary_validation_deltas": deltas}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--check-data", action="store_true")
    parser.add_argument("--data-root", type=Path, default=data_source.OUT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
        print("Stop requested; finish both arms and save the joint checkpoint.", flush=True)
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    return run(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
