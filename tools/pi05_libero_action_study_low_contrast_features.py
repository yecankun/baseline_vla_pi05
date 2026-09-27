"""Validation-only clean replay/low-contrast frozen features; no training.

The orchestration entrypoint owns old feature preflight and policy loading. This
helper never constructs another policy or a full feature pack, fits statistics,
changes targets, or encodes training images. The two encoders run serially so
their temporary hooks never nest on the same policy.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from build_pi05_libero_action_study_features import (
    PLAN_SHA256, TraceEncoder, check_loaded_checkpoint, runtime_snapshot, verify_files,
)
from pi05_libero_action_study_feature_source import FIXED_EPISODES, source_unchanged
from pi05_libero_world_model_adapter import IMAGE_KEYS, LiberoFeaturePack, sha256_file
from probe_pi05_libero_public_features import local_file, read_json
from vla_benchmark_contract import VisualCorruption, apply_visual_corruption


SCHEMA = "pi05_libero_action_study_low_contrast_features_v1"
VALIDATION_EPISODES = ((1530, 126), (1476, 125), (1458, 130), (1566, 143))
CLEAN_REPORT_SHA256 = "9e0537b5d520f5e242a02be1f94c454febfec62854a8c43a517f7acdce05ac56"
CLEAN_MANIFEST_SHA256 = "133894fb46d9d2920dc5393affa2d5fb07e5c0baf3fb4c12d7a6af25238ee56b"
CORRUPTION_SEED = 20260911


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _runtime_guard(plan, runtime):
    current = runtime_snapshot(plan)
    for key in ("checkpoint", "protocol", "versions"):
        _require(current[key] == runtime[key], f"loaded feature runtime changed: {key}")
    for path, digest in current["files_sha256"].items():
        _require(runtime["files_sha256"].get(path) == digest, "feature runtime pin set changed")
    # Old preflight additionally pins its source-readback file. Core files were
    # just rehashed by runtime_snapshot; only these additional entries remain.
    verify_files({p: h for p, h in runtime["files_sha256"].items() if p not in current["files_sha256"]})
    return deepcopy(runtime)


def _clean_files(root):
    _require(sha256_file(local_file(root, "report.json")) == CLEAN_REPORT_SHA256, "old clean feature report bytes changed")
    _require(sha256_file(local_file(root, "manifest.json")) == CLEAN_MANIFEST_SHA256, "old clean feature manifest bytes changed")
    report = read_json(root / "report.json")
    _require(report.get("status") == "passed_frozen_feature_cache_only", "old clean cache did not pass")
    outputs = report.get("output_sha256")
    _require(type(outputs) is dict and outputs and outputs.get("manifest.json") == CLEAN_MANIFEST_SHA256,
             "complete old clean output inventory required")
    files = {str(root / "report.json"): CLEAN_REPORT_SHA256}
    for name, digest in outputs.items():
        _require(name not in {"report.json", "status.json", "run.log"}, "mutable/self-referential clean inventory")
        files[str(local_file(root, name))] = digest
    verify_files(files)
    return files, report


def _latent(value):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.shape == (2, 2048)
             and np.isfinite(value).all(), "encoder must return finite native float32 [2,2048]")
    return value


def _save_array(path, value):
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def extract_validation_features(source, policy, checkpoint, old_feature_plan, runtime,
                                out, clean_pack_dir, progress=None) -> dict:
    """Extract only fixed held-out rows and retain original-cache row indices.

    ``progress(completed_rows, episode_index, frame_index, condition)`` receives
    condition-local counts1..524 for ``clean`` or ``low_contrast``. It is called
    every50 rows and at each episode end. Checkpoint/preflight are reused rather
    than repeated policy loading. Output must be a fresh caller-created folder;
    failures preserve partial evidence and do not resume or overwrite.
    """
    started = time.monotonic()
    out, clean_root = Path(out).resolve(), Path(clean_pack_dir).resolve()
    _require(out.is_dir() and not out.is_relative_to(clean_root)
             and not out.is_relative_to(Path(source["root"]).resolve()), "output must be a separate fresh existing directory")
    output_names = ("clean", "low_contrast", "degraded_visual_latent.npy", "row_indices.npy", "corruption_trace.jsonl")
    _require(not any((out / name).exists() for name in output_names), "low-contrast output artifacts already exist; no overwrite/resume")
    _require(progress is None or callable(progress), "progress must be callable or None")
    _require(old_feature_plan.get("episodes") == list(FIXED_EPISODES), "fixed complete12 source plan changed")
    declarations = [e["declaration"] for e in source["episodes"]]
    _require([{k: d.get(k) for k in ("episode_index", "partition", "record_count")} for d in declarations]
             == list(FIXED_EPISODES), "source fixed12 declaration order/count/split changed")
    selected = [e for e in source["episodes"] if e["declaration"]["partition"] == "validation"]
    _require([(e["declaration"]["episode_index"], e["declaration"]["record_count"]) for e in selected]
             == list(VALIDATION_EPISODES), "exactly the complete four fixed validation episodes required")
    _require(old_feature_plan["selected_split"]["validation_episode_indices"] == [e for e, _ in VALIDATION_EPISODES],
             "validation split changed")
    _require(source["report_sha256"] == old_feature_plan["source_report_sha256"], "source report provenance changed")
    _require(type(source.get("verified_files")) is dict and source["verified_files"], "source verified-file pins required")
    source_before = {"root": str(Path(source["root"]).resolve()), "report_sha256": source["report_sha256"],
                     "files_sha256": deepcopy(source["verified_files"])}
    _require(source_unchanged(source) is True, "source immutable rehash did not pass")
    runtime_before = _runtime_guard(old_feature_plan, runtime)
    check_loaded_checkpoint(checkpoint, old_feature_plan)
    clean_files, clean_report = _clean_files(clean_root)
    _require(clean_report.get("source_report_sha256") == source["report_sha256"], "clean cache source differs")
    parameters = list(policy.parameters())
    versions = [p._version for p in parameters]
    _require(not any(m.training for m in policy.modules())
             and not any(p.requires_grad or p.grad is not None for p in parameters), "policy must remain frozen/eval/gradient-free")
    count = sum(n for _, n in VALIDATION_EPISODES)
    metadata, row_indices, evidence, condition_seconds = [], [], {}, {}
    degraded_latent = np.empty((count, 2, 2048), dtype=np.float32)
    corruption = VisualCorruption(name="low_contrast", severity=2)
    with LiberoFeaturePack(clean_root) as pack:
        _require(pack.manifest_sha256 == CLEAN_MANIFEST_SHA256 and pack.visual_dim == 2048,
                 "unchanged native clean manifest/latent dimension required")
        _require(pack.manifest["source"]["metadata_sha256"] == source["report_sha256"]
                 and pack.manifest["extractor"]["checkpoint_sha256"] == checkpoint["model_sha256"], "clean source/checkpoint identity differs")
        _require(len(pack.arrays["episode_index"]) == old_feature_plan["expected"]["rows"], "full clean cache row count differs")
        _require([(e["episode_index"], e["record_count"]) for e in pack.manifest["episodes"]]
                 == [(d["episode_index"], d["record_count"]) for d in declarations], "clean cache complete source registry differs")
        for episode in selected:
            d = episode["declaration"]
            eid, n = d["episode_index"], d["record_count"]
            _require(d.get("complete_episode") is True and episode["images"].shape == (n, 2, 256, 256, 3)
                     and episode["images"].dtype == np.uint8, "complete native uint8 validation RGB required")
            indices = pack.indices[eid]
            _require(len(indices) == n and np.array_equal(pack.arrays["frame_index"][indices], np.arange(n)), "complete cache episode/frame mapping differs")
            _require(bool(pack.arrays["visual_valid"][indices].all()), "both clean target views must remain valid")
            row_indices.extend(int(index) for index in indices)
            metadata.extend((episode, frame, int(index), frame in {0, n-1}) for frame, index in enumerate(indices))
        _require(len(metadata) == count and len(set(row_indices)) == count, "duplicate or incomplete validation cache rows")
        clean_hashes, raw_hashes = [], []
        # One complete context per condition. Never nest TraceEncoder hooks.
        for condition in ("clean", "low_contrast"):
            folder = out / condition
            folder.mkdir(exist_ok=False)
            begin = time.monotonic()
            trace = (out / "corruption_trace.jsonl").open("x", encoding="utf-8") if condition == "low_contrast" else None
            try:
                with TraceEncoder(policy, folder) as encoder:
                    for position, (episode, frame, global_index, preview) in enumerate(metadata):
                        eid = episode["declaration"]["episode_index"]
                        raw = np.array(episode["images"][frame], copy=True)
                        before = [_array_sha(view) for view in raw]
                        if condition == "clean":
                            value = _latent(encoder.row(raw, episode_index=eid, frame_index=frame, preview=preview))
                            cached = pack.arrays["visual_latent"][global_index]
                            _require(value.tobytes() == np.ascontiguousarray(cached).tobytes(),
                                     f"clean replay must byte-match existing cache without tolerance: episode={eid}, frame={frame}")
                            clean_hashes.append(_array_sha(value))
                            raw_hashes.append(before)
                        else:
                            _require(before == raw_hashes[position], "raw source RGB changed between conditions")
                            sample_key = f"libero_action_study/episode-{eid}/frame-{frame:06d}"
                            degraded = np.stack([apply_visual_corruption(raw[v], sample_key=sample_key,
                                view_key=key, corruption=corruption, seed=CORRUPTION_SEED)
                                for v, key in enumerate(IMAGE_KEYS)])
                            _require(degraded.shape == raw.shape and degraded.dtype == np.uint8, "corruption changed native RGB shape/dtype")
                            _require([_array_sha(view) for view in raw] == before, "corruption mutated clean caller RGB")
                            value = _latent(encoder.row(degraded, episode_index=eid, frame_index=frame, preview=preview))
                            degraded_latent[position] = value
                            after = [_array_sha(view) for view in degraded]
                            view_details = [{"view_key": key, "source_rgb_sha256": before[v],
                                "degraded_rgb_sha256": after[v], "changed": before[v] != after[v],
                                "mean_absolute_pixel_change": float(np.abs(degraded[v].astype(np.float32)-raw[v]).mean()),
                                "max_absolute_pixel_change": float(np.abs(degraded[v].astype(np.int16)-raw[v]).max())}
                                for v, key in enumerate(IMAGE_KEYS)]
                            trace.write(json.dumps({"validation_row_index": position, "original_cache_row_index": global_index,
                                "episode_index": eid, "frame_index": frame, "sample_key": sample_key,
                                "corruption": {"name": "low_contrast", "severity": 2, "seed": CORRUPTION_SEED, "alpha": .45},
                                "views": view_details, "clean_latent_sha256": clean_hashes[position],
                                "degraded_latent_sha256": _array_sha(value),
                                "latent_changed": _array_sha(value) != clean_hashes[position]}, allow_nan=False) + "\n")
                            trace.flush()
                        if progress and ((position+1) % 50 == 0 or frame == episode["declaration"]["record_count"]-1):
                            progress(position+1, eid, frame, condition)
                    evidence[condition] = encoder.evidence(count)
            finally:
                if trace is not None:
                    trace.close()
            condition_seconds[condition] = time.monotonic()-begin
    _require(not any(m.training for m in policy.modules())
             and not any(p.requires_grad or p.grad is not None for p in parameters)
             and [p._version for p in parameters] == versions, "policy mode/gradient/versions changed across conditions")
    _require(source_unchanged(source) is True, "source immutable post-rehash did not pass")
    source_after = {"root": str(Path(source["root"]).resolve()), "report_sha256": source["report_sha256"],
                    "files_sha256": deepcopy(source["verified_files"])}
    _require(source_after == source_before, "source pin inventory changed during paired extraction")
    runtime_after = _runtime_guard(old_feature_plan, runtime)
    check_loaded_checkpoint(checkpoint, old_feature_plan)
    verify_files(clean_files)
    _require(runtime_before == runtime_after, "feature runtime changed during paired extraction")
    _save_array(out / "degraded_visual_latent.npy", degraded_latent)
    _save_array(out / "row_indices.npy", np.asarray(row_indices, dtype=np.int64))
    records = [json.loads(line) for line in (out / "corruption_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    _require(len(records) == count, "corruption trace row count differs")
    return {"schema": SCHEMA, "status": "passed_validation_only_frozen_clean_replay_and_low_contrast",
        "validation_episode_counts": [{"episode_index": eid, "record_count": n} for eid, n in VALIDATION_EPISODES],
        "rows_per_condition": count, "real_view_images_per_condition": 2*count,
        "total_encoded_rows": 2*count, "total_real_view_embeddings": 4*count,
        "training_rows_encoded": 0, "clean_replay_matches_cached_latent_bytes": True,
        "clean_replay_tolerance": "none; exact float32 bytes including signed-zero",
        "degraded_latent_shape": [count, 2, 2048], "row_indices_shape": [count],
        "row_mapping": "validation source order -> original clean1658 cache row index; targets remain clean",
        "corruption": {"name": "low_contrast", "severity": 2, "seed": CORRUPTION_SEED, "alpha": .45,
            "placement": "raw stored uint8 RGB before unchanged native PI05 preprocessing",
            "formula": "per-image per-channel spatial mean + alpha*(pixel-mean) in float32/255; unchanged wrapper clip/rint uint8",
            "geometry_or_extra_flip": False},
        "rgb_viewframes_changed": sum(v["changed"] for r in records for v in r["views"]),
        "latent_rows_changed": sum(r["latent_changed"] for r in records),
        "change_counts_are_descriptive_not_required_nonzero": True,
        "source_report_sha256": source["report_sha256"], "old_feature_plan_sha256": PLAN_SHA256,
        "source_before": source_before, "source_after": source_after, "checkpoint": deepcopy(checkpoint),
        "clean_report_sha256": CLEAN_REPORT_SHA256,
        "clean_manifest_sha256": CLEAN_MANIFEST_SHA256, "clean_files_sha256": clean_files,
        "runtime_before": runtime_before, "runtime_after": runtime_after,
        "source_and_clean_cache_and_runtime_unchanged": True, "encoder_evidence": evidence,
        "extraction_seconds_by_condition": condition_seconds, "runtime_seconds": time.monotonic()-started,
        "output_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for name in output_names
                           for p in ([out/name] if (out/name).is_file() else sorted((out/name).rglob("*"))) if p.is_file()},
        "target_tensors_modified": False, "normalization_fitted": False, "feature_pack_created": False,
        "optimizer_steps": 0, "training_started": False, "policy_actions": 0, "simulation_steps": 0,
        "visual_status": "not_viewed", "training_ready": False, "formal_data_allowed": False,
        "real_system_validated": False, "world_model_quality_evaluated_by_this_helper": False}
