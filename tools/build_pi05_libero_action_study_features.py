"""Fixed12 complete public trajectories -> frozen native PI0.5 cache, no training.

Preflight is read-only unless a fresh --preflight-out is provided. Run requires
the explicit feature flag and a fresh fixed output; no resume or replacement IDs.
Old two-episode tools and all model/loss definitions remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

from build_pi05_libero_feature_pack import TraceEncoder, TASK_REGISTRY, close_source, write_json
from pi05_libero_world_model_adapter import ARRAY_NAMES, LAYOUT, SCHEMA, SPLIT_SCHEMA, sha256_file
from prepare_pi05_libero_world_model import prepare
from probe_pi05_libero_public_features import Tee, load_policy, read_json
from pi05_libero_action_study_feature_source import validate_source, source_unchanged

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = "docs/libero-action-study-feature-plan-v1.json"
PLAN_SHA256 = "ee85b9efbabf980f1aae550652a4db92c4f68b3b47b0bd34d25f3658cdefdf7a"
BUILD_SCHEMA = "pi05_libero_action_study_feature_build_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(files):
    for name, digest in files.items():
        require(sha256_file(Path(name)) == digest, f"bound input changed: {name}")


def load_plan(repo=ROOT):
    path = repo / PLAN_PATH
    require(sha256_file(path) == PLAN_SHA256, "feature plan changed")
    plan = read_json(path)
    require(plan["feature_extraction_allowed"] is True and plan["optimization_allowed"] is False,
            "feature-only authorization required")
    return plan


def runtime_snapshot(plan, repo=ROOT):
    """Pin checkpoint/config/installed code without constructing a policy."""
    pins = plan["checkpoint"]
    checkpoint = Path(pins["path"]).resolve()
    protocol = repo / pins["protocol_path"]
    files = {str(repo / PLAN_PATH): PLAN_SHA256,
             str(checkpoint / "model.safetensors"): pins["model_sha256"],
             str(checkpoint / "config.json"): pins["config_sha256"],
             str(protocol): pins["protocol_sha256"]}
    for name, digest in plan["reuse_code_sha256"].items():
        files[str(repo / "tools" / name)] = digest
    for name in (Path(__file__).name, "pi05_libero_action_study_feature_source.py"):
        path = repo / "tools" / name
        files[str(path)] = sha256_file(path)
    spec = importlib.util.find_spec("lerobot.policies.pi05.modeling_pi05")
    require(spec is not None and spec.origin is not None, "installed PI05 implementation missing")
    files[str(Path(spec.origin).resolve())] = pins["installed_modeling_sha256"]
    verify_files(files)
    require((checkpoint / "model.safetensors").stat().st_size == pins["model_size_bytes"], "checkpoint size mismatch")
    versions = {name: importlib.metadata.version(name) for name in pins["versions"]}
    require(versions == pins["versions"], "installed dependency version changed")
    return {"checkpoint": str(checkpoint), "protocol": str(protocol), "files_sha256": files, "versions": versions}


def check_loaded_checkpoint(checkpoint, plan):
    pins = plan["checkpoint"]
    for name in ("model_sha256", "config_sha256", "protocol_sha256"):
        require(checkpoint[name] == pins[name], f"loaded checkpoint {name} drift")
    require(checkpoint["versions"] == pins["versions"] and checkpoint["device"] == "cuda", "loaded runtime drift")
    require(checkpoint["compile_model"] is False and checkpoint["gradient_checkpointing"] is False, "compilation/checkpointing enabled")
    loaded = checkpoint["load"]
    require(loaded["status"] == "loaded" and loaded["loaded_parameter_fraction"] == 1.0
            and not loaded["load_state_dict_missing_keys"] and not loaded["load_state_dict_unexpected_keys"], "strict checkpoint coverage failed")
    implementations = checkpoint["installed_implementation_sources"]
    require(set(implementations) == {"policy_class", "image_preprocess", "image_embedding"}, "installed implementation roles differ")
    for entry in implementations.values():
        require(entry["sha256"] == pins["installed_modeling_sha256"], "installed implementation drift")
        require(sha256_file(Path(entry["path"])) == entry["sha256"], "installed implementation changed after load")


def preflight(repo=ROOT):
    started = time.monotonic()
    plan = load_plan(repo)
    readback_path = repo / "simulation_output/libero_action_study_sources_readback_v1.json"
    require(sha256_file(readback_path) == plan["source_readback_sha256"], "completed source readback missing/changed")
    readback = read_json(readback_path)
    require(readback["status"] == "passed_source_hash_rows_rgb_and_pair_readback_only"
            and readback["report_sha256"] == plan["source_report_sha256"], "source readback did not pass")
    source = validate_source(repo / "simulation_output" / plan["source_directory_name"], plan)
    try:
        runtime = runtime_snapshot(plan, repo)
        runtime["files_sha256"][str(readback_path)] = plan["source_readback_sha256"]
        import torch
        require(torch.cuda.is_available(), "CUDA unavailable; no fallback")
        free, total = torch.cuda.mem_get_info()
        require(free >= 16_000_000_000, "at least16GB free CUDA memory required; do not displace other jobs")
        require(shutil.disk_usage(repo / "simulation_output").free >= 1_000_000_000, "at least1GB free output disk required")
        report = {"schema": "pi05_libero_action_study_feature_preflight_v1", "status": "passed_preflight_not_extracted",
                  "plan_sha256": PLAN_SHA256, "source_report_sha256": plan["source_report_sha256"],
                  "source_readback_sha256": plan["source_readback_sha256"], "runtime_snapshot": runtime,
                  "selected_split": plan["selected_split"], "expected": plan["expected"],
                  "gpu": {"name": torch.cuda.get_device_name(), "free_bytes": free, "total_bytes": total},
                  "output_already_exists": (repo / "simulation_output" / plan["output_directory_name"]).exists(),
                  "source_episodes_verified": len(source["episodes"]), "source_rows_verified": 1658,
                  "features_extracted": 0, "policy_loaded": False, "normalization_fitted": False,
                  "optimizer_steps": 0, "training_ready": False, "runtime_seconds": time.monotonic() - started}
        return plan, source, runtime, report
    except BaseException:
        close_source(source)
        raise


def build_feature_pack(source, policy, checkpoint, plan, runtime, out, progress=None):
    check_loaded_checkpoint(checkpoint, plan)
    expected = plan["expected"]
    declarations = [episode["declaration"] for episode in source["episodes"]]
    require([(d["episode_index"], d["record_count"]) for d in declarations]
            == [(e["episode_index"], e["record_count"]) for e in plan["episodes"]], "fixed complete source choices differ")
    rows = [row for episode in source["episodes"] for row in episode["adapted"]]
    require(len(rows) == expected["rows"], "no partial episodes allowed")
    require(out.is_dir() and not any(out.glob("*.npy")) and not (out / "manifest.json").exists()
            and not (out / "extraction_trace.jsonl").exists(), "feature output is not fresh")
    arrays = {"episode_index": np.asarray([r["episode_index"] for r in rows], dtype=np.int64),
              "frame_index": np.asarray([r["frame_index"] for r in rows], dtype=np.int64),
              "task_id": np.asarray([r["task_id"] for r in rows], dtype=np.int64),
              "state": np.stack([r["state"] for r in rows]).astype(np.float32),
              "state_valid": np.stack([r["state_valid"] for r in rows]),
              "action": np.stack([r["action"] for r in rows]).astype(np.float32),
              "visual_valid": np.ones((len(rows), 2), dtype=bool),
              "visual_latent": np.empty(expected["latent_shape"], dtype=np.float32)}
    require(set(arrays) == ARRAY_NAMES, "native array contract drift")
    extraction_started = time.monotonic()
    with TraceEncoder(policy, out) as encoder:
        offset = 0
        for episode in source["episodes"]:
            eid = episode["declaration"]["episode_index"]
            count = episode["declaration"]["record_count"]
            for frame in range(count):
                arrays["visual_latent"][offset] = encoder.row(episode["images"][frame], episode_index=eid,
                                                            frame_index=frame, preview=frame in {0, count-1})
                offset += 1
                if offset % 50 == 0 or frame == count-1:
                    print(f"FROZEN_STUDY_FEATURE_PROGRESS rows={offset}/{len(rows)} episode={eid} frame={frame}", flush=True)
                    if progress:
                        progress(offset, eid, frame)
        evidence = encoder.evidence(offset)
    require(offset == len(rows) and np.isfinite(arrays["visual_latent"]).all(), "missing/nonfinite latent output")
    extraction_seconds = time.monotonic() - extraction_started
    provenance = {"feature_plan_sha256": PLAN_SHA256, "source_report_sha256": source["report_sha256"],
                  "source_readback_sha256": plan["source_readback_sha256"], "checkpoint": checkpoint,
                  "runtime_snapshot": runtime, "orientation": "stored_dataset_rgb_no_extra_flip"}
    write_json(out / "extractor_provenance.json", provenance)
    manifest = {"schema": SCHEMA, "benchmark": "libero_spatial",
                "source": {**source["report"]["source"], "metadata_sha256": source["report_sha256"]},
                "extractor": {"kind": "frozen_pi05_native_multiview_v1", "checkpoint_sha256": checkpoint["model_sha256"],
                              "preprocessing_sha256": plan["checkpoint"]["installed_modeling_sha256"],
                              "code_sha256": sha256_file(out / "extractor_provenance.json")},
                "layout": deepcopy(LAYOUT), "fps": 10.0, "task_registry": deepcopy(TASK_REGISTRY),
                "episodes": [{k: d[k] for k in ("episode_index", "task_id", "source_trajectory_id", "leakage_group_id", "record_count", "complete_episode")} for d in declarations], "arrays": {}}
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        manifest["arrays"][name] = {"path": path.name, "sha256": sha256_file(path)}
    write_json(out / "manifest.json", manifest)
    split = {"schema": SPLIT_SCHEMA, "feature_pack_manifest_sha256": sha256_file(out / "manifest.json"), **deepcopy(plan["selected_split"])}
    write_json(out / "split.json", split)
    preparation_started = time.monotonic()
    prepared = prepare(out, out / "split.json", out / "preparation", context_len=4, horizon=3)
    windows = {key: value["windows"] for key, value in prepared["partitions"].items()}
    require(windows == {"train": expected["train_windows"], "validation": expected["validation_windows"]}, "complete window counts differ")
    stats = read_json(out / "preparation/normalization.json")
    require(stats["state_record_count"] == expected["train_state_rows"] and stats["action_record_count"] == expected["train_action_rows"]
            and stats["train_episode_indices"] == sorted(plan["selected_split"]["train_episode_indices"])
            and stats["final_row_actions_excluded"] is True, "train-only normalization/terminal action guard failed")
    source_unchanged(source)
    verify_files(runtime["files_sha256"])
    for entry in checkpoint["installed_implementation_sources"].values():
        require(sha256_file(Path(entry["path"])) == entry["sha256"], "installed code changed during extraction")
    variation = []
    offset = 0
    for declaration in declarations:
        count = declaration["record_count"]
        features = arrays["visual_latent"][offset:offset+count]
        variation.append({"episode_index": declaration["episode_index"], "rows": count,
                          "noninitial_rows_different_from_first": int(np.any(features[1:] != features[:1], axis=(1, 2)).sum()),
                          "max_abs_difference_from_first": float(np.abs(features-features[:1]).max())})
        offset += count
    return {"schema": BUILD_SCHEMA, "status": "passed_frozen_feature_cache_only", "feature_plan_sha256": PLAN_SHA256,
            "source_report_sha256": source["report_sha256"], "source_readback_sha256": plan["source_readback_sha256"],
            "selected_split": deepcopy(plan["selected_split"]), "source_groups": source["report"]["duplicate_audit"]["same_partition_groups"],
            "complete_episode_counts": {d["episode_index"]: d["record_count"] for d in declarations},
            "manifest_sha256": sha256_file(out / "manifest.json"), "split_sha256": sha256_file(out / "split.json"),
            "checkpoint": checkpoint, "extraction_evidence": evidence, "latent_shape": list(arrays["visual_latent"].shape),
            "latent_dtype": "float32", "state_shape": list(arrays["state"].shape), "action_shape": list(arrays["action"].shape),
            "windows": windows, "normalization_state_rows": stats["state_record_count"], "normalization_action_rows": stats["action_record_count"],
            "normalization_sha256": sha256_file(out / "preparation/normalization.json"),
            "extraction_seconds": extraction_seconds, "preparation_and_posthash_seconds": time.monotonic() - preparation_started,
            "source_and_checkpoint_and_code_unchanged": True, "features_extracted": len(rows), "feature_pack_complete": True,
            "normalization_fitted": True, "normalization_scope": "world_model_train_only_not_PI05_processors",
            "grouped_sampler_fitted": False, "optimizer_steps": 0, "training_started": False,
            "actions_generated": 0, "simulation_steps": 0, "training_ready": False, "formal_data_allowed": False,
            "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
            "per_episode_latent_variation_descriptive_only": variation,
            "world_model_quality_evaluated": False, "policy_performance_evaluated": False,
            "real_system_validated": False, "visual_status": "not_viewed"}


def execute(plan, source, runtime, preflight_report, repo=ROOT):
    out = repo / "simulation_output" / plan["output_directory_name"]
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def status(kind, **extra):
        write_json(out / "status.json", {"status": kind, "pid": os.getpid(), "runtime_seconds": time.monotonic()-started,
                                          "optimizer_steps": 0, "training_started": False, **extra})
    status("running", phase="checkpoint_load", features_extracted=0)
    write_json(out / "preflight.json", preflight_report)
    with (out / "run.log").open("x", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            policy, checkpoint = load_policy(Path(runtime["checkpoint"]), Path(runtime["protocol"]), "cuda")
            check_loaded_checkpoint(checkpoint, plan)
            print("PINNED_FROZEN_CHECKPOINT_VERIFIED_NO_OPTIMIZER", flush=True)
            report = build_feature_pack(source, policy, checkpoint, plan, runtime, out,
                progress=lambda count, eid, frame: status("running", phase="extraction", features_extracted=count, episode_index=eid, frame_index=frame))
            report["runtime_seconds_excluding_preflight"] = time.monotonic()-started
            report["output_sha256"] = {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob("*")
                                       if p.is_file() and p.relative_to(out).as_posix() not in {"status.json", "report.json", "run.log"}}
            write_json(out / "report.json", report)
            status("completed", phase="completed", features_extracted=report["features_extracted"], report_sha256=sha256_file(out / "report.json"))
            print(json.dumps({"status": report["status"], "rows": report["features_extracted"], "windows": report["windows"], "report_sha256": sha256_file(out / "report.json"), "optimizer_steps": 0}), flush=True)
            return report
        except BaseException as exc:
            traceback.print_exc()
            status("failed", error_type=type(exc).__name__, error=str(exc), resume_allowed=False)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("preflight", "run"))
    parser.add_argument("--execute-frozen-feature-cache", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    args = parser.parse_args()
    require(args.execute_frozen_feature_cache == (args.stage == "run"), "run requires explicit feature execution flag; preflight must omit it")
    if args.preflight_out:
        require(args.stage == "preflight" and not args.preflight_out.exists()
                and args.preflight_out.resolve().parent == ROOT / "simulation_output", "preflight output must be a fresh simulation_output file")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    plan, source, runtime, checked = preflight()
    try:
        if args.stage == "preflight":
            if args.preflight_out:
                with args.preflight_out.open("x", encoding="utf-8") as stream:
                    json.dump(checked, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            print(json.dumps(checked, allow_nan=False))
        else:
            execute(plan, source, runtime, checked)
    finally:
        close_source(source)


if __name__ == "__main__":
    main()
