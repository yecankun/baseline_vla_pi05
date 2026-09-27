"""Read back fixed12 frozen features; no model, extraction, optimizer or writes to inputs.

The source root may be the metadata-only review subset: original source image
arrays, MP4s and Parquet files are deliberately not opened. RGB evidence is a
link to previously hash-audited decoded-frame records, not a new video decode.
An optional fresh --out must be outside both input roots; otherwise print only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pi05_libero_world_model_adapter import (
    ARRAY_NAMES, IMAGE_KEYS, LiberoFeaturePack, LiberoWindowDataset, sha256_file,
)
from probe_pi05_libero_public_features import checked_hash, local_file, read_json


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "docs/libero-action-study-feature-plan-v1.json"
PLAN_SHA256 = "ee85b9efbabf980f1aae550652a4db92c4f68b3b47b0bd34d25f3658cdefdf7a"
SOURCE_REPORT_SHA256 = "20d34377c636469e2d06a61cc7519db6fb0ee6c33379d9631b058870d101ef5c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lines(path):
    """Strict JSONL via the existing duplicate-key/nonfinite JSON reader."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSONL key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSONL constant: {value}")

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            require(line.strip(), "blank JSONL line")
            yield json.loads(line, object_pairs_hook=pairs, parse_constant=reject)


def independent_normalization(pack, split, stats):
    """Recompute raw train statistics independently, without fitting/writing a pack."""
    train = sorted(split["train_episode_indices"])
    state_indices = np.concatenate([pack.indices[e] for e in train])
    action_indices = np.concatenate([pack.indices[e][:-1] for e in train])
    state = np.asarray(pack.arrays["state"][state_indices], dtype=np.float64)
    valid = pack.arrays["state_valid"][state_indices]
    count = valid.sum(axis=0)
    # Per-coordinate observed values provide an independent implementation of
    # the production masked sums; population standard deviations are intended.
    means, scales = [], []
    for coordinate in range(state.shape[1]):
        values = state[valid[:, coordinate], coordinate]
        means.append(float(values.mean()) if len(values) else 0.0)
        std = float(values.std(ddof=0)) if len(values) else 0.0
        scales.append(std if len(values) and std >= 1e-6 else 1.0)
    action = np.asarray(pack.arrays["action"][action_indices], dtype=np.float64)
    action_std = action.std(axis=0, ddof=0)
    computed = {"state_mean": means, "state_std": scales,
                "action_mean": action.mean(axis=0),
                "action_std": np.where(action_std >= 1e-6, action_std, 1.0)}
    for key, expected in computed.items():
        actual = np.asarray(stats[key], dtype=np.float64)
        require(actual.shape == np.asarray(expected).shape
                and np.allclose(actual, expected, atol=1e-12, rtol=1e-12),
                f"independent train normalization differs: {key}")
    require(stats["train_episode_indices"] == train, "normalization train IDs differ")
    require(stats["state_count"] == count.tolist(), "normalization support counts differ")
    require(stats["state_record_count"] == len(state_indices)
            and stats["action_record_count"] == len(action_indices)
            and stats["final_row_actions_excluded"] is True,
            "normalization counts or terminal-action exclusion differs")
    require(stats["feature_pack_manifest_sha256"] == pack.manifest_sha256
            and stats["split_sha256"] == split["split_sha256"], "normalization provenance differs")
    return {"state_rows": len(state_indices), "action_rows": len(action_indices),
            "terminal_actions_excluded": len(train), "train_ids": train}


def independent_window_checks(pack, split, stats, partition, saved_windows=None):
    """Compare every unchanged adapter window against independently indexed arrays."""
    dataset = LiberoWindowDataset(pack, split, partition=partition, normalization=stats,
                                 context_len=4, horizon=3)
    expected = []
    for eid in sorted(split[f"{partition}_episode_indices"]):
        indices = pack.indices[eid]
        for current in range(3, len(indices) - 3):
            expected.append((eid, indices[current-3:current+1],
                             indices[current:current+3], indices[current+1:current+4]))
    require(len(dataset) == len(expected), "window count differs from complete-episode enumeration")
    if saved_windows is not None:
        require(len(saved_windows) == len(dataset), "saved window count differs")
    a = pack.arrays
    mean, std = np.asarray(stats["state_mean"]), np.asarray(stats["state_std"])
    action_mean, action_std = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])
    supported = np.asarray(stats["state_count"]) > 0
    state_targets = visual_targets = 0
    for index, (eid, h, actions, targets) in enumerate(expected):
        w, item = dataset.windows[index], dataset[index]
        require(w.episode_index == eid and list(w.history) == h.tolist()
                and list(w.actions) == actions.tolist() and list(w.targets) == targets.tolist(),
                "adapter window indices differ")
        if saved_windows is not None:
            canonical = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(w).items()}
            require(saved_windows[index] == canonical, "saved complete window index differs")
        history_mask = a["state_valid"][h] & supported
        target_mask = a["state_valid"][targets] & a["state_valid"][h[-1]] & supported
        history = ((np.where(history_mask, a["state"][h], mean) - mean) / std).astype(np.float32)
        future = np.where(target_mask, a["state"][targets], 0).astype(np.float64)
        anchor = np.where(target_mask, a["state"][h[-1]], 0).astype(np.float64)
        expected_inputs = {
            "history_visual_latent": np.where(a["visual_valid"][h, :, None], a["visual_latent"][h], 0),
            "history_visual_valid": a["visual_valid"][h],
            "history_state": history, "history_state_valid": history_mask,
            "candidate_actions": ((a["action"][actions] - action_mean) / action_std).astype(np.float32)[None],
        }
        expected_targets = {
            "future_visual_latent": np.where(a["visual_valid"][targets, :, None], a["visual_latent"][targets], 0),
            "future_visual_valid": a["visual_valid"][targets],
            "state_delta": ((future-anchor) / std).astype(np.float32), "state_target_valid": target_mask,
        }
        require(set(item["inputs"]) == {*expected_inputs, "task_instruction"}
                and set(item["targets"]) == set(expected_targets), "window model-facing key boundary changed")
        for section, values in (("inputs", expected_inputs), ("targets", expected_targets)):
            for key, value in values.items():
                require(np.array_equal(item[section][key], value), f"window value differs: {section}/{key}")
        task = pack.tasks[pack.episodes[eid]["task_id"]]
        require(item["inputs"]["task_instruction"] == task["task_instruction"], "native task text differs")
        require(item["metadata"] == {"episode_index": eid, "task_id": task["task_id"],
                "source_task_index": task["source_task_index"], "history_indices": h.tolist(),
                "action_indices": actions.tolist(), "target_indices": targets.tolist()}, "window metadata differs")
        require(not any(r == pack.indices[eid][-1] for r in actions), "terminal row used as candidate action")
        state_targets += int(target_mask.sum())
        visual_targets += int(a["visual_valid"][targets].sum())
    return {"windows": len(dataset), "valid_state_target_coordinates": state_targets,
            "valid_visual_target_views": visual_targets}


def check_tensor_evidence(record, shape, dtype, low=None, high=None):
    require(record["shape"] == shape and record["dtype"] == dtype and record["finite"] is True,
            "recorded tensor shape/dtype/finite boundary differs")
    checked_hash(record["sha256"])
    require(np.isfinite(record["min"]) and np.isfinite(record["max"])
            and record["min"] <= record["max"], "recorded tensor bounds are invalid")
    if low is not None:
        require(record["min"] >= low and record["max"] <= high, "recorded tensor range differs")


def audit(feature_pack, source_root, report_sha256):
    started = time.monotonic()
    feature_pack, source_root = Path(feature_pack).resolve(), Path(source_root).resolve()
    require(sha256_file(PLAN_PATH) == PLAN_SHA256, "fixed feature plan changed")
    plan = read_json(PLAN_PATH)
    report_path = local_file(feature_pack, "report.json")
    report_sha256 = checked_hash(report_sha256)
    require(sha256_file(report_path) == report_sha256, "externally supplied feature report hash differs")
    report = read_json(report_path)
    require(report["schema"] == "pi05_libero_action_study_feature_build_v1"
            and report["status"] == "passed_frozen_feature_cache_only", "completed fixed feature build required")
    require(report["feature_plan_sha256"] == PLAN_SHA256
            and report["source_report_sha256"] == SOURCE_REPORT_SHA256
            and report["source_readback_sha256"] == plan["source_readback_sha256"], "feature/source provenance differs")
    require(report["selected_split"] == plan["selected_split"]
            and report["features_extracted"] == 1658 and report["feature_pack_complete"] is True,
            "fixed feature coverage differs")
    require(report["latent_shape"] == [1658, 2, 2048] and report["latent_dtype"] == "float32"
            and report["state_shape"] == [1658, 8] and report["action_shape"] == [1658, 7],
            "reported fixed tensor shapes differ")
    require(report["normalization_fitted"] is True and report["normalization_state_rows"] == 1134
            and report["normalization_action_rows"] == 1126
            and report["source_and_checkpoint_and_code_unchanged"] is True,
            "reported normalization or mutation evidence differs")
    for key in ("optimizer_steps", "actions_generated", "simulation_steps"):
        require(type(report[key]) is int and report[key] == 0, f"scope changed: {key}")
    for key in ("training_started", "training_ready", "formal_data_allowed", "grouped_sampler_fitted",
                "family_independence_verified", "world_model_quality_evaluated",
                "policy_performance_evaluated", "real_system_validated"):
        require(report[key] is False, f"scope flag changed: {key}")
    require(report["checkpoint_training_overlap_unknown"] is True, "checkpoint overlap boundary changed")
    status = read_json(local_file(feature_pack, "status.json"))
    require(status["status"] == status["phase"] == "completed" and status["report_sha256"] == report_sha256
            and status["features_extracted"] == 1658 and status["optimizer_steps"] == 0, "feature job not completed")
    outputs = report["output_sha256"]
    require(isinstance(outputs, dict) and outputs, "complete output hash inventory required")
    hashes = {"report.json": report_sha256, "status.json": sha256_file(feature_pack / "status.json")}
    for name, digest in outputs.items():
        require(name not in {"report.json", "status.json", "run.log"}, "mutable/self-referential inventory")
        require(sha256_file(local_file(feature_pack, name)) == checked_hash(digest), f"feature output changed: {name}")
        hashes[name] = digest
    for name, digest in (("manifest.json", report["manifest_sha256"]), ("split.json", report["split_sha256"]),
                         ("preparation/normalization.json", report["normalization_sha256"])):
        require(outputs.get(name) == digest, f"output inventory misses pinned {name}")
    require("preparation/report.json" in outputs or report.get("preparation_report_sha256")
            == sha256_file(local_file(feature_pack, "preparation/report.json")), "preparation report lacks hash binding")
    if "preparation/report.json" not in outputs:
        hashes["preparation/report.json"] = report["preparation_report_sha256"]
    present = {p.relative_to(feature_pack).as_posix() for p in feature_pack.rglob("*") if p.is_file()}
    require(present == set(hashes) | {"run.log"}, "unlisted or missing feature artifact")
    source_path = local_file(source_root, "report.json")
    require(sha256_file(source_path) == SOURCE_REPORT_SHA256, "metadata-only source report hash differs")
    source_report = read_json(source_path)
    source_hashes = {"report.json": SOURCE_REPORT_SHA256}
    source_outputs = source_report["output_sha256"]
    source_eps = {e["episode_index"]: e for e in source_report["episodes"]}
    require(report["source_groups"] == source_report["duplicate_audit"]["same_partition_groups"], "source groups changed")
    groups = {eid: g["group_id"] for g in report["source_groups"] for eid in g["episode_indices"]}
    trace = list(lines(local_file(feature_pack, "extraction_trace.jsonl")))
    require(len(trace) == 1658 and outputs["extraction_trace.jsonl"] == report["extraction_evidence"]["trace_sha256"], "trace coverage/hash differs")
    checkpoint = report["checkpoint"]
    for key in ("model_sha256", "config_sha256", "protocol_sha256", "versions"):
        require(checkpoint[key] == plan["checkpoint"][key], f"recorded checkpoint pin differs: {key}")
    require(checkpoint["compile_model"] is False and checkpoint["gradient_checkpointing"] is False
            and checkpoint["device"] == "cuda", "recorded extraction runtime differs")
    require(set(checkpoint["installed_implementation_sources"])
            == {"policy_class", "image_preprocess", "image_embedding"}, "recorded installed source roles differ")
    for item in checkpoint["installed_implementation_sources"].values():
        require(item["sha256"] == plan["checkpoint"]["installed_modeling_sha256"], "recorded installed code differs")
    load = checkpoint["load"]
    require(load["status"] == "loaded" and load["loaded_parameter_fraction"] == 1.0
            and not load["load_state_dict_missing_keys"] and not load["load_state_dict_unexpected_keys"], "recorded strict checkpoint load failed")
    e = report["extraction_evidence"]
    require(e["complete_rows_extracted"] == e["preprocessing_call_count"] == 1658
            and e["real_view_embedding_call_count"] == 3316 and e["empty_camera_embedded"] is False, "extraction counts differ")
    for key in ("all_modules_eval", "all_parameters_frozen", "all_parameter_gradients_absent",
                "parameter_versions_unchanged", "all_calls_inference_mode"):
        require(e[key] is True, f"recorded frozen boundary differs: {key}")
    preview_source_count = 0
    with LiberoFeaturePack(feature_pack) as pack:
        require(pack.visual_dim == 2048 and pack.arrays["visual_latent"].shape == (1658, 2, 2048)
                and pack.arrays["state"].shape == (1658, 8) and pack.arrays["action"].shape == (1658, 7), "fixed tensor layout differs")
        require(set(pack.manifest["arrays"]) == ARRAY_NAMES
                and pack.arrays["state_valid"].all() and pack.arrays["visual_valid"].all(), "missing observation imputation introduced")
        require(pack.manifest["source"] == {**source_report["source"], "metadata_sha256": SOURCE_REPORT_SHA256}, "manifest source identity differs")
        require(list(pack.episodes) == [p["episode_index"] for p in plan["episodes"]], "manifest episode order differs")
        split = pack.load_split(feature_pack / "split.json")
        require({k: split[k] for k in plan["selected_split"]} == plan["selected_split"], "fixed split changed")
        offset = 0
        variations = []
        for choice in plan["episodes"]:
            eid, count = choice["episode_index"], choice["record_count"]
            declaration, original = pack.episodes[eid], source_eps[eid]
            require(declaration["record_count"] == count and declaration["task_id"] == 9
                    and declaration["complete_episode"] is True
                    and declaration["source_trajectory_id"] == original["source_trajectory_id"]
                    and declaration["leakage_group_id"] == groups[eid], "episode identity/group/completeness differs")
            for key in ("records_path", "decoded_frames_path"):
                name = original[key]
                require(sha256_file(local_file(source_root, name)) == source_outputs[name], "source metadata payload hash differs")
                source_hashes[name] = source_outputs[name]
            rows = read_json(local_file(source_root, original["records_path"]))
            frames = read_json(local_file(source_root, original["decoded_frames_path"]))
            require(len(rows) == count and len(frames) == 2, "source complete row/two-view count differs")
            indices = np.arange(offset, offset+count)
            require(np.array_equal(pack.indices[eid], indices), "feature row order differs from fixed concatenation")
            a = pack.arrays
            require(np.array_equal(a["state"][indices], np.asarray([r["observation.state"] for r in rows], dtype=np.float32))
                    and np.array_equal(a["action"][indices], np.asarray([r["action"] for r in rows], dtype=np.float32)), "raw state/action differs from source JSON")
            for frame, row in enumerate(rows):
                require(row["episode_index"] == eid and row["frame_index"] == frame and row["task_index"] == 39
                        and row["task"] == pack.tasks[9]["task_instruction"], "source row/task differs")
                item = trace[offset+frame]
                require(item["episode_index"] == eid and item["frame_index"] == frame
                        and item["preview"] == (frame in {0, count-1}), "trace row identity/preview differs")
                rgb_hashes = [v["frames"][frame]["decoded_rgb_sha256"] for v in frames]
                require(item["source_pixel_sha256"] == rgb_hashes, "trace/source decoded RGB hash differs")
                latent = np.ascontiguousarray(a["visual_latent"][offset+frame])
                check_tensor_evidence(item["latent"], [1, 2, 2048], "torch.float32")
                require(item["latent"]["sha256"] == hashlib.sha256(latent.tobytes()).hexdigest()
                        and item["latent"]["min"] == float(latent.min()) and item["latent"]["max"] == float(latent.max()), "saved feature bytes differ from traced latent")
                prep = item["preprocessing"]
                require(set(prep["inputs"]) == set(IMAGE_KEYS) and prep["masks"] == [[True], [True], [False]]
                        and prep["grad_enabled"] is False and prep["inference_mode"] is True, "image-only preprocessing boundary differs")
                for key in IMAGE_KEYS:
                    check_tensor_evidence(prep["inputs"][key], [1, 3, 256, 256], "torch.float32", 0, 1)
                require(len(prep["images"]) == 3 and len(item["embeddings"]) == 2, "preprocess/embedding count differs")
                for p in prep["images"]:
                    check_tensor_evidence(p, [1, 3, 224, 224], "torch.float32", -1, 1)
                require(prep["images"][2]["min"] == prep["images"][2]["max"] == -1, "empty camera pixels differ")
                for view, embedding in enumerate(item["embeddings"]):
                    require(embedding == {"view": view, "tokens_shape": [1, 256, 2048], "tokens_dtype": "torch.bfloat16",
                            "finite": True, "grad_enabled": False, "inference_mode": True}, "recorded frozen image embedding differs")
                    if item["preview"]:
                        from PIL import Image
                        name = f"episode{eid}_frame{frame:04d}_view{view}_source.png"
                        with Image.open(local_file(feature_pack, name)) as image:
                            pixels = np.asarray(image)
                        require(pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8
                                and hashlib.sha256(pixels.tobytes()).hexdigest() == rgb_hashes[view], "source preview PNG differs from decoded source hash")
                        preview_source_count += 1
            features = a["visual_latent"][indices]
            variations.append({"episode_index": eid, "rows": count,
                "noninitial_rows_different_from_first": int(np.any(features[1:] != features[:1], axis=(1, 2)).sum()),
                "max_abs_difference_from_first": float(np.abs(features-features[:1]).max())})
            offset += count
        require(offset == 1658 and preview_source_count == 48, "complete selected row/preview accounting differs")
        require(report["complete_episode_counts"] == {str(p["episode_index"]): p["record_count"] for p in plan["episodes"]}, "outer episode counts differ")
        require(report["per_episode_latent_variation_descriptive_only"] == variations, "descriptive variation report differs")
        stats = read_json(feature_pack / "preparation/normalization.json")
        normalization = independent_normalization(pack, split, stats)
        require(normalization["state_rows"] == 1134 and normalization["action_rows"] == 1126, "fixed training support differs")
        preparation = read_json(feature_pack / "preparation/report.json")
        windows = {part: independent_window_checks(pack, split, stats, part,
                    list(lines(feature_pack / f"preparation/{part}_windows.jsonl"))) for part in ("train", "validation")}
        require({k: v["windows"] for k, v in windows.items()} == report["windows"] == {"train": 1086, "validation": 500}, "fixed window counts differ")
        for part, results in windows.items():
            require(all(preparation["partitions"][part][k] == v for k, v in results.items()), "preparation count evidence differs")
        for name, digest in preparation["outputs_sha256"].items():
            require(outputs.get("preparation/" + name) == digest, "preparation output hashes differ")
    for name, digest in hashes.items():
        require(sha256_file(local_file(feature_pack, name)) == digest, "feature input changed during readback")
    for name, digest in source_hashes.items():
        require(sha256_file(local_file(source_root, name)) == digest, "source metadata changed during readback")
    return {"schema": "pi05_libero_action_study_feature_readback_v1",
            "status": "passed_feature_bytes_trace_source_metadata_normalization_windows_only",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "report_sha256": report_sha256,
            "feature_plan_sha256": PLAN_SHA256, "source_report_sha256": SOURCE_REPORT_SHA256,
            "audit_tool_sha256": sha256_file(Path(__file__)), "verified_feature_outputs": len(outputs),
            "verified_source_metadata_files": len(source_hashes), "rows": 1658, "trace_rows": 1658,
            "decoded_rgb_hash_links": 3316, "source_preview_pngs_pixel_checked": preview_source_count,
            "normalization": normalization, "partitions": windows, "source_readback_scope": "metadata_only_previous_decoded_rgb_hash_links",
            "raw_source_image_arrays_opened": False, "videos_decoded": 0, "source_parquet_opened": False,
            "live_checkpoint_rehashed_by_this_audit": False, "installed_code_rehashed_by_this_audit": False,
            "features_extracted_by_this_audit": 0, "normalization_artifacts_written": 0,
            "policy_loaded": False, "optimizer_steps": 0, "training_ready": False,
            "formal_data_allowed": False, "family_independence_verified": False,
            "checkpoint_training_overlap_unknown": True, "world_model_quality_evaluated": False,
            "real_system_validated": False, "visual_status": "not_viewed",
            "runtime_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-pack", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        output = args.out.resolve()
        require(not output.exists() and not output.is_relative_to(args.feature_pack.resolve())
                and not output.is_relative_to(args.source_root.resolve()), "audit output must be fresh and outside input roots")
        require(output.parent.is_dir(), "audit output parent must already exist")
    report = audit(args.feature_pack, args.source_root, args.report_sha256)
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
