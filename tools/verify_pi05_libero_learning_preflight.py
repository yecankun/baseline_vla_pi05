"""Verify the repaired evaluation path against all fixed references, without training."""
import argparse
from pathlib import Path
import time

from pi05_libero_world_model_adapter import LiberoFeaturePack, LiberoWindowDataset, fit_train_normalization, read_json
from smoke_pi05_libero_world_model import BATCH_SIZE, COUNTS, SEED, contained, parameter_sha, sha256_file, validate_artifacts, write_json
from eval_pi05_libero_world_model_persistence import PROTOCOL_SHA, assert_matching_batch, load_protocol, validate_reference
from run_pi05_libero_world_model_learning_diagnostic import evaluate, numeric_tree_close, ordinary_tensors, runtime_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-pack", "random-reference", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preflight output must not exist; never overwrite a failed learning attempt")
    protocol = load_protocol(args.protocol, PROTOCOL_SHA)
    _, verified = validate_artifacts(args.feature_pack, protocol["data"]["feature_report_sha256"])
    reference, traces, reference_hashes = validate_reference(args.random_reference, protocol, verified)
    own = {name: sha256_file(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, "run_pi05_libero_world_model_learning_diagnostic.py",
        "test_pi05_libero_world_model_learning_diagnostic.py")}
    import torch
    from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig, collate_window_inputs
    from pi05_libero_world_model_objectives import collate_window_targets

    runtime_gate(protocol)
    started = time.monotonic()
    with LiberoFeaturePack(args.feature_pack) as pack:
        split = pack.load_split(args.feature_pack / "split.json")
        stats = fit_train_normalization(pack, split)
        if stats != read_json(args.feature_pack / "preparation" / "normalization.json"):
            raise ValueError("normalization mismatch")
        batches = {}
        for part, count in COUNTS.items():
            dataset = LiberoWindowDataset(pack, split, partition=part, normalization=stats)
            if len(dataset) != count:
                raise ValueError("window count mismatch")
            batches[part] = []
            for start in range(0, count, BATCH_SIZE):
                rows = [dataset[i] for i in range(start, min(start + BATCH_SIZE, count))]
                inputs = collate_window_inputs([r["inputs"] for r in rows], device="cpu")
                targets = collate_window_targets([r["targets"] for r in rows], device="cpu")
                ordinary_tensors(inputs)
                ordinary_tensors(targets)
                assert_matching_batch(inputs, targets, targets, rows, traces[(part, start)])
                batches[part].append((start, inputs, targets))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = LiberoWorldModel(LiberoWorldModelConfig(**protocol["model"]["config"]), pack.manifest["task_registry"])
        initial_sha = parameter_sha(model)
        if initial_sha != protocol["baselines"]["random_reference_initial_parameter_sha256"]:
            raise ValueError("initial parameter SHA mismatch")
        args.out.mkdir(parents=True, exist_ok=False)
        # Deliberately enter with training=True and requires_grad=True, as in the
        # failed production attempt; evaluate must temporarily freeze and restore.
        result = evaluate(model, batches, stats, args.out, 0, reference_traces=traces)
        for part in COUNTS:
            numeric_tree_close(result[part]["metrics"], reference["partitions"][part]["diagnostic_untrained_metrics"])
            numeric_tree_close(result[part]["objective"], reference["partitions"][part]["objective"])
        if parameter_sha(model) != initial_sha or any(not p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("evaluation failed to restore gradient flags or preserve parameters")
        if any(not m.training for m in model.modules()):
            raise ValueError("evaluation failed to restore training modes")
    for root, hashes in ((args.feature_pack, verified), (args.random_reference, reference_hashes)):
        if any(sha256_file(contained(root, name)) != digest for name, digest in hashes.items()):
            raise ValueError("source changed during preflight")
    if any(sha256_file(Path(__file__).with_name(name)) != digest for name, digest in own.items()):
        raise ValueError("implementation changed during preflight")
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA initialization")
    report = {"schema": "pi05_libero_learning_evaluation_repair_preflight_v1", "status": "passed",
              "protocol_sha256": PROTOCOL_SHA, "implementation_sha256": own,
              "verified_feature_artifacts": verified, "verified_random_reference_artifacts": reference_hashes,
              "batch_count": sum(len(b) for b in batches.values()), "windows": COUNTS,
              "all_prediction_hashes_exact": True, "all_metrics_match": True,
              "metric_rtol": 1e-12, "metric_atol": 1e-10,
              "initial_parameter_sha256": initial_sha, "model_parameters_unchanged": True,
              "training_mode_and_requires_grad_restored": True, "ordinary_input_tensors": True,
              "optimizer_steps": 0, "backward_calls": 0, "pi05_loaded": False,
              "automatic_learning_retry_allowed": False, "runtime_seconds": time.monotonic() - started,
              "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "output_sha256": {p.name: sha256_file(p) for p in args.out.iterdir() if p.is_file()}}
    write_json(args.out / "report.json", report)
    print("REPAIRED_STEP0_PREFLIGHT_PASSED_NO_TRAINING", sha256_file(args.out / "report.json"), flush=True)


if __name__ == "__main__":
    main()
