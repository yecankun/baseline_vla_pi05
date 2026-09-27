"""Two complete native LIBERO episodes -> frozen feature pack, no training.

This deliberately bounded, fixed-split single-task diagnostic is not a claim
of independent-family validation or unseen-checkpoint-training data. The
checkpoint, native image extraction and schema/window preparation are reused
unchanged. Existing output directories are never resumed or overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from pi05_libero_world_model_adapter import (
    ARRAY_NAMES, DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, LAYOUT, SCHEMA,
    SPLIT_SCHEMA, LiberoFeaturePack, adapt_native_record, sha256_file,
    validate_task_registry,
)
from prepare_pi05_libero_world_model import prepare
from probe_pi05_libero_public_features import (
    DEFAULT_CHECKPOINT, MODEL_SHA256, ORIENTATION, ROW_KEYS, Tee,
    checked_hash, checked_int, load_policy, local_file, read_json, tensor_record,
)


BUILD_SCHEMA = "pi05_libero_feature_pair_build_v1"
SOURCE_SCHEMA = "pi05_libero_source_pair_v1"
PLAN_SHA256 = "bad36b6a93769d4210792c60a42892a325d30694d8a087316deeafab7de48a53"
EPISODE_COUNTS = {1400: 140, 1402: 173}
FIXED_SPLIT = {"train_episode_indices": [1400], "validation_episode_indices": [1402]}
TASK_REGISTRY = [{"task_id": 9, "source_task_index": 39,
                  "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]
PRIOR_REPORT_SHA256 = "0d431859588a2f14758fbf1c27b17e1ec6114f161c14f5cc246ae2e55317c4e5"
SOURCE_ROW_KEYS = ROW_KEYS | {"timestamp", "index"}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def close_source(source: dict) -> None:
    for episode in source.get("episodes", []):
        images = episode.get("images")
        if isinstance(images, np.memmap):
            images._mmap.close()


def validate_source(root: Path, report_sha256: str) -> dict:
    """Validate all complete records/images and split before importing torch."""
    root = root.resolve()
    report_path = local_file(root, "report.json")
    if sha256_file(report_path) != checked_hash(report_sha256):
        raise ValueError("source report differs from externally supplied SHA256")
    report = read_json(report_path)
    if report.get("schema") != SOURCE_SCHEMA or report.get("status") != "passed":
        raise ValueError("passed independent offline source-pair audit required")
    if report.get("source") != {"kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION}:
        raise ValueError("only pinned public demonstration source allowed; no synthetic/rollout source")
    if report.get("task_registry") != TASK_REGISTRY:
        raise ValueError("source requires exactly the fixed Spatial9/source39/native instruction registry")
    validate_task_registry(report["task_registry"])
    if (report.get("fps") != 10.0 or report.get("split") != FIXED_SPLIT
            or report.get("orientation") != ORIENTATION or report.get("row_count") != 313):
        raise ValueError("fixed fps, image orientation, complete row count or episode split differs")
    if (report.get("family_independence_verified") is not False
            or report.get("checkpoint_training_overlap_unknown") is not True
            or report.get("training_ready") is not False):
        raise ValueError("diagnostic independence/overlap/training boundaries required")
    if report.get("duplicate_suffix_audit", {}).get("status") != "passed":
        raise ValueError("cross-episode duplicate/suffix audit must pass before feature extraction")
    if report.get("prior_source_report_sha256") != PRIOR_REPORT_SHA256:
        raise ValueError("source audit must bind the previously verified public source")
    if report.get("plan_path") != "plan.json" or report.get("plan_sha256") != PLAN_SHA256:
        raise ValueError("source pair must bind the frozen pre-extraction plan")
    plan_path = local_file(root, report["plan_path"])
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("frozen plan bytes changed")
    plan = read_json(plan_path)
    if (plan.get("schema") != "pi05_libero_feature_pair_plan_v2"
            or plan.get("context_len") != 4 or plan.get("horizon") != 3
            or plan.get("orientation") != ORIENTATION):
        raise ValueError("unsupported frozen pair plan")

    verified = {"report.json": report_sha256}
    outputs = report.get("output_sha256")
    if not isinstance(outputs, dict) or not outputs or outputs.get("plan.json") != PLAN_SHA256:
        raise ValueError("complete hash-bound source derived output inventory required")
    for name, digest in outputs.items():
        if name in {"status.json", "report.json"}:
            raise ValueError("self-referential or mutable source output inventory")
        if sha256_file(local_file(root, name)) != checked_hash(digest):
            raise ValueError(f"source derived output hash mismatch: {name}")
        verified[name] = digest
    if ("duplicate_suffix_audit.json" not in outputs
            or read_json(local_file(root, "duplicate_suffix_audit.json")) != report["duplicate_suffix_audit"]):
        raise ValueError("duplicate/suffix audit sidecar is missing or inconsistent")

    declarations = report.get("episodes")
    if not isinstance(declarations, list) or len(declarations) != 2:
        raise ValueError("exactly two complete episode declarations required")
    if [row.get("episode_index") for row in declarations] != list(EPISODE_COUNTS):
        raise ValueError("source episode order/IDs differ from fixed plan")
    if (len({row.get("source_trajectory_id") for row in declarations}) != 2
            or len({row.get("leakage_group_id") for row in declarations}) != 2):
        raise ValueError("source trajectory/leakage group crosses fixed split")
    source = {"root": root, "report": report, "report_sha256": report_sha256, "plan": plan,
              "verified_files": verified, "episodes": []}
    try:
        for declaration in declarations:
            episode_id = checked_int(declaration["episode_index"], "episode_index")
            count = EPISODE_COUNTS[episode_id]
            if (checked_int(declaration.get("record_count"), "record_count") != count
                    or declaration.get("task_id") != 9 or declaration.get("source_task_index") != 39
                    or declaration.get("complete_episode") is not True):
                raise ValueError("source episode count/task/completeness differs from fixed plan")
            if (declaration.get("source_trajectory_id") != f"{DEMO_REPO}@{DEMO_REVISION}/episode{episode_id}"
                    or declaration.get("leakage_group_id") != f"source_episode_{episode_id}"):
                raise ValueError("source trajectory/leakage identity is not native episode identity")
            for stem, filename in (("images", "images.npy"), ("records", "records.json")):
                name = declaration.get(f"{stem}_path")
                digest = checked_hash(declaration.get(f"{stem}_sha256"))
                if name != f"episode_{episode_id}/{filename}" or outputs.get(name) != digest:
                    raise ValueError(f"episode {stem} must match hash-bound source inventory")
            images = np.load(local_file(root, declaration["images_path"]), mmap_mode="r", allow_pickle=False)
            item = {"declaration": deepcopy(declaration), "images": images}
            source["episodes"].append(item)
            if images.dtype != np.uint8 or images.shape != (count, 2, 256, 256, 3):
                raise ValueError("complete source images require uint8 [record_count,2,256,256,3]")
            records = read_json(local_file(root, declaration["records_path"]))
            if not isinstance(records, list) or len(records) != count:
                raise ValueError("source records truncated or extended relative to complete episode")
            metadata = declaration.get("metadata", {})
            if (checked_int(metadata.get("episode_index"), "metadata episode_index") != episode_id
                    or checked_int(metadata.get("length"), "metadata length") != count):
                raise ValueError("source episode metadata differs from declared complete record count")
            start = checked_int(metadata.get("dataset_from_index"), "metadata dataset_from_index")
            if checked_int(metadata.get("dataset_to_index"), "metadata dataset_to_index") != start + count:
                raise ValueError("metadata global row interval differs from episode length")
            adapted = []
            for frame, record in enumerate(records):
                if not isinstance(record, dict) or set(record) != SOURCE_ROW_KEYS:
                    raise ValueError("source record unknown/missing fields; policy accepts only native state/action/task")
                selected = {key: record[key] for key in ROW_KEYS}
                row = adapt_native_record(selected, TASK_REGISTRY)
                if (row["episode_index"] != episode_id or row["frame_index"] != frame or row["task_id"] != 9
                        or checked_int(record["index"], "record global index") != start + frame):
                    raise ValueError("source row ID gap/reset/task switch or complete-episode order mismatch")
                stamp = record["timestamp"]
                if (type(stamp) not in {float, int} or not np.isfinite(stamp)
                        or not np.isclose(stamp, frame / 10.0, atol=2e-5, rtol=0)):
                    raise ValueError("stored timestamp is not consecutive native 10Hz")
                if not row["state_valid"].all():
                    raise ValueError("no source state imputation allowed")
                adapted.append(row)
            item["records"], item["adapted"] = records, adapted
        return source
    except BaseException:
        close_source(source)
        raise


def _restore(owner, key, previous, present):
    if present:
        setattr(owner, key, previous)
    else:
        delattr(owner, key)


class TraceEncoder:
    """Record actual native preprocessing and image-only embedding boundaries."""

    def __init__(self, policy, out: Path):
        import torch
        self.policy, self.out = policy, out
        self.parameters = list(policy.parameters())
        if (any(module.training for module in policy.modules())
                or any(p.requires_grad or p.grad is not None for p in self.parameters)):
            raise ValueError("all modules must be eval, parameters frozen and gradients absent")
        self.versions = [parameter._version for parameter in self.parameters]
        self.device = self.parameters[0].device if self.parameters else torch.device("cpu")
        self.vision = policy.model.paligemma_with_expert
        self.preprocess = policy._preprocess_images
        self.embed = self.vision.embed_image
        self.hooks = []
        self.preprocessing_count = self.embedding_count = 0
        self.record = None

    def __enter__(self):
        self.stream = (self.out / "extraction_trace.jsonl").open("x", encoding="utf-8")
        for owner, key, method in ((self.policy, "_preprocess_images", self._preprocess),
                                   (self.vision, "embed_image", self._embed)):
            self.hooks.append((owner, key, owner.__dict__.get(key), key in owner.__dict__))
            setattr(owner, key, method)
        return self

    def __exit__(self, *_):
        for arguments in reversed(self.hooks):
            _restore(*arguments)
        self.stream.close()

    def _preprocess(self, batch):
        import torch
        images, masks = self.preprocess(batch)
        self.preprocessing_count += 1
        if self.record is None or "preprocessing" in self.record:
            raise ValueError("preprocess must execute exactly once for each current row")
        if len(images) != 3 or len(masks) != 3:
            raise ValueError("pinned preprocessing must return two real and one declared empty view")
        self.record["preprocessing"] = {
            "inputs": {key: tensor_record(value) for key, value in batch.items()},
            "images": [tensor_record(value) for value in images],
            "masks": [mask.detach().cpu().tolist() for mask in masks],
            "grad_enabled": torch.is_grad_enabled(), "inference_mode": torch.is_inference_mode_enabled(),
        }
        if self.record["preprocessing"]["masks"] != [[True], [True], [False]]:
            raise ValueError("native valid cameras and fully masked empty camera required")
        if any(tuple(image.shape) != (1, 3, 224, 224) for image in images):
            raise ValueError("pinned preprocessing shape must be [1,3,224,224]")
        if self.record["preview"]:
            from PIL import Image
            for index in range(2):
                pixels = ((images[index][0].detach().float().cpu().permute(1, 2, 0) + 1) * 127.5).round()
                if bool((pixels < 0).any()) or bool((pixels > 255).any()):
                    raise ValueError("processed preview outside native [-1,1]")
                Image.fromarray(pixels.to(torch.uint8).numpy()).save(
                    self.out / f"episode{self.record['episode_index']}_frame{self.record['frame_index']:04d}_view{index}_processed.png")
        return images, masks

    def _embed(self, image):
        import torch
        if self.record is None or "preprocessing" not in self.record:
            raise ValueError("embedding occurred outside the native preprocessing boundary")
        view = len(self.record["embeddings"])
        if view >= 2 or tensor_record(image) != self.record["preprocessing"]["images"][view]:
            raise ValueError("only the exact two native preprocessed real views may be embedded")
        tokens = self.embed(image)
        self.embedding_count += 1
        self.record["embeddings"].append({
            "view": view, "tokens_shape": list(tokens.shape), "tokens_dtype": str(tokens.dtype),
            "finite": bool(torch.isfinite(tokens).all()), "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
        })
        return tokens

    def row(self, image_array: np.ndarray, *, episode_index: int, frame_index: int, preview: bool) -> np.ndarray:
        import torch
        from pi05_libero_feature_extraction import extract_native_visual_features
        self.record = {"episode_index": episode_index, "frame_index": frame_index, "preview": preview,
                       "embeddings": [], "source_pixel_sha256": [hashlib.sha256(view.tobytes()).hexdigest()
                                                                   for view in image_array]}
        batch = {key: torch.from_numpy(np.array(image_array[index], copy=True)).permute(2, 0, 1)
                 .unsqueeze(0).to(device=self.device, dtype=torch.float32).div(255)
                 for index, key in enumerate(IMAGE_KEYS)}
        expected_inputs = {key: tensor_record(value) for key, value in batch.items()}
        if preview:
            from PIL import Image
            for index in range(2):
                Image.fromarray(np.asarray(image_array[index])).save(
                    self.out / f"episode{episode_index}_frame{frame_index:04d}_view{index}_source.png")
        latent = extract_native_visual_features(self.policy, batch)
        if tuple(latent.shape) != (1, 2, 2048) or latent.requires_grad or not torch.isfinite(latent).all():
            raise ValueError("frozen native feature row must be finite [1,2,2048]")
        if self.record.get("preprocessing", {}).get("inputs") != expected_inputs:
            raise ValueError("native float32 /255 BCHW input pixel mapping changed")
        if any(tensor_record(batch[key]) != expected_inputs[key] for key in IMAGE_KEYS):
            raise ValueError("caller images were mutated")
        if len(self.record["embeddings"]) != 2:
            raise ValueError("each row must embed exactly two real views")
        for event in [self.record["preprocessing"], *self.record["embeddings"]]:
            if event["grad_enabled"] or not event["inference_mode"] or event.get("finite") is False:
                raise ValueError("frozen finite inference-only execution boundary violated")
        self.record["latent"] = tensor_record(latent)
        self.stream.write(json.dumps(self.record, allow_nan=False) + "\n")
        self.stream.flush()
        self.record = None
        return latent[0].detach().float().cpu().numpy().copy()

    def evidence(self, rows: int) -> dict:
        if self.preprocessing_count != rows or self.embedding_count != 2 * rows:
            raise ValueError("complete source rows did not execute exactly one preprocess/two embeds each")
        if (any(module.training for module in self.policy.modules())
                or any(p.requires_grad or p.grad is not None for p in self.parameters)
                or [p._version for p in self.parameters] != self.versions):
            raise ValueError("policy mode/gradients/parameter versions changed")
        return {"complete_rows_extracted": rows, "preprocessing_call_count": self.preprocessing_count,
                "real_view_embedding_call_count": self.embedding_count, "empty_camera_embedded": False,
                "all_modules_eval": True, "all_parameters_frozen": True,
                "all_parameter_gradients_absent": True, "parameter_versions_unchanged": True,
                "all_calls_inference_mode": True, "input_mapping": "stored RGB -> float32 BCHW /255; no flip",
                "batch_size": 1, "trace_sha256": sha256_file(self.out / "extraction_trace.jsonl")}


def build_feature_pack(source: dict, policy, checkpoint: dict, out: Path) -> dict:
    """Called only after source audit checks and strict offline checkpoint load."""
    if (checkpoint.get("model_sha256") != MODEL_SHA256 or checkpoint.get("compile_model") is not False
            or checkpoint.get("load", {}).get("status") != "loaded"
            or checkpoint["load"].get("loaded_parameter_fraction") != 1.0):
        raise ValueError("strict pinned frozen checkpoint evidence required")
    if (out / "manifest.json").exists() or (out / "extraction_trace.jsonl").exists():
        raise FileExistsError("feature artifacts already exist; use a fresh attempt")
    rows = [row for episode in source["episodes"] for row in episode["adapted"]]
    if len(rows) != 313:
        raise ValueError("fixed complete two-episode pack requires 313 rows")
    arrays = {
        "episode_index": np.asarray([row["episode_index"] for row in rows], dtype=np.int64),
        "frame_index": np.asarray([row["frame_index"] for row in rows], dtype=np.int64),
        "task_id": np.asarray([row["task_id"] for row in rows], dtype=np.int64),
        "state": np.stack([row["state"] for row in rows]).astype(np.float32),
        "state_valid": np.stack([row["state_valid"] for row in rows]),
        "action": np.stack([row["action"] for row in rows]).astype(np.float32),
        "visual_valid": np.ones((len(rows), 2), dtype=bool),
        "visual_latent": np.empty((len(rows), 2, 2048), dtype=np.float32),
    }
    with TraceEncoder(policy, out) as encoder:
        offset = 0
        for episode in source["episodes"]:
            declaration = episode["declaration"]
            count, episode_id = declaration["record_count"], declaration["episode_index"]
            for frame in range(count):
                arrays["visual_latent"][offset] = encoder.row(episode["images"][frame], episode_index=episode_id,
                                                             frame_index=frame, preview=frame in {0, count - 1})
                offset += 1
                if offset % 20 == 0 or frame == count - 1:
                    print(f"FROZEN_FEATURE_PROGRESS rows={offset}/313 episode={episode_id} frame={frame}/{count - 1}", flush=True)
        evidence = encoder.evidence(offset)
    if set(arrays) != ARRAY_NAMES or not np.isfinite(arrays["visual_latent"]).all():
        raise ValueError("feature output array contract violated")

    implementations = {name: sha256_file(Path(__file__).with_name(name)) for name in (
        Path(__file__).name, "pi05_libero_feature_extraction.py", "pi05_action_effect_world_model.py",
        "probe_pi05_libero_public_features.py", "pi05_pretrained_loader.py",
        "pi05_libero_world_model_adapter.py", "prepare_pi05_libero_world_model.py",
    )}
    provenance = {"repo_sources": implementations, "checkpoint": checkpoint,
                  "source_report_sha256": source["report_sha256"], "plan_sha256": PLAN_SHA256,
                  "orientation": ORIENTATION}
    write_json(out / "extractor_provenance.json", provenance)
    preprocessing_sha256 = checked_hash(checkpoint["installed_implementation_sources"]["image_preprocess"]["sha256"])
    manifest = {
        "schema": SCHEMA, "benchmark": "libero_spatial",
        "source": {**source["report"]["source"], "metadata_sha256": source["report_sha256"]},
        "extractor": {"kind": "frozen_pi05_native_multiview_v1", "checkpoint_sha256": MODEL_SHA256,
                      "preprocessing_sha256": preprocessing_sha256,
                      "code_sha256": sha256_file(out / "extractor_provenance.json")},
        "layout": deepcopy(LAYOUT), "fps": 10.0, "task_registry": deepcopy(TASK_REGISTRY),
        "episodes": [{key: episode["declaration"][key] for key in (
            "episode_index", "task_id", "source_trajectory_id", "leakage_group_id", "record_count", "complete_episode")}
                     for episode in source["episodes"]], "arrays": {},
    }
    for name, array in arrays.items():
        path = out / f"{name}.npy"
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        manifest["arrays"][name] = {"path": path.name, "sha256": sha256_file(path)}
    write_json(out / "manifest.json", manifest)
    split = {"schema": SPLIT_SCHEMA, "feature_pack_manifest_sha256": sha256_file(out / "manifest.json"),
             **deepcopy(FIXED_SPLIT)}
    write_json(out / "split.json", split)
    # Existing prepare intentionally reports schema-only evidence. Its generic
    # unverified-source wording remains unchanged; separate source/build reports
    # supply the narrower actual acquisition and extraction facts.
    preparation = prepare(out, out / "split.json", out / "preparation", context_len=4, horizon=3)
    if {key: value["windows"] for key, value in preparation["partitions"].items()} != {"train": 134, "validation": 167}:
        raise ValueError("fixed complete episode window counts differ")
    normalization = read_json(out / "preparation" / "normalization.json")
    if (normalization["state_record_count"] != 140 or normalization["action_record_count"] != 139
            or normalization["train_episode_indices"] != [1400]
            or normalization["final_row_actions_excluded"] is not True):
        raise ValueError("train-only normalization/final-action exclusion failed")
    for name, digest in source["verified_files"].items():
        if sha256_file(local_file(source["root"], name)) != digest:
            raise ValueError(f"source artifact mutated during extraction: {name}")
    return {"schema": BUILD_SCHEMA, "status": "passed", "source_report_sha256": source["report_sha256"],
            "source_root": str(source["root"]), "source": deepcopy(source["report"]["source"]),
            "plan_sha256": PLAN_SHA256, "manifest_sha256": sha256_file(out / "manifest.json"),
            "split_sha256": sha256_file(out / "split.json"), "split": deepcopy(FIXED_SPLIT),
            "checkpoint": checkpoint, "extraction_evidence": evidence,
            "latent_shape": list(arrays["visual_latent"].shape), "latent_dtype": "float32",
            "state_shape": [313, 8], "action_shape": [313, 7], "complete_episode_counts": EPISODE_COUNTS,
            "preparation_report_sha256": sha256_file(out / "preparation" / "report.json"),
            "normalization_state_rows": 140, "normalization_action_rows": 139,
            "windows": {"train": 134, "validation": 167}, "window_context_len": 4, "window_horizon": 3,
            "source_files_unchanged": True, "output_sha256": {
                path.relative_to(out).as_posix(): sha256_file(path) for path in out.rglob("*")
                if path.is_file() and path.name not in {"report.json", "status.json", "run.log"}},
            "feature_pack_complete_for_fixed_two_episodes": True, "full_suite_coverage": False,
            "family_independence_verified": False, "checkpoint_training_overlap_unknown": True,
            "policy_generalization_evaluated": False, "world_model_quality_evaluated": False,
            "training_ready": False, "training_started": False, "optimizer_steps": 0,
            "actions_generated": 0, "simulation_steps": 0, "learned_world_model_present": False,
            "real_system_validated": False, "visual_status": "not_viewed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-report-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=Path("docs/libero-spatial-score-protocol-v1.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started, source = time.monotonic(), None
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    write_json(args.out / "status.json", {"schema": BUILD_SCHEMA, "status": "running", "pid": os.getpid(),
                                          "optimizer_steps": 0, "training_started": False})
    with (args.out / "run.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                source = validate_source(args.source_root, args.source_report_sha256)
                print("COMPLETE_SOURCE_ROWS_IMAGES_AND_FIXED_SPLIT_VERIFIED rows=313", flush=True)
                policy, checkpoint = load_policy(args.checkpoint, args.protocol, args.device)
                print("PINNED_FROZEN_CHECKPOINT_VERIFIED", flush=True)
                report = build_feature_pack(source, policy, checkpoint, args.out)
                report["runtime_seconds"] = time.monotonic() - started
                write_json(args.out / "report.json", report)
                write_json(args.out / "status.json", {"schema": BUILD_SCHEMA, "status": "completed",
                           "report_sha256": sha256_file(args.out / "report.json"), "optimizer_steps": 0})
                print("COMPLETE_FEATURE_PAIR_PREPARED", json.dumps({"rows": 313, "windows": report["windows"],
                      "runtime_seconds": report["runtime_seconds"], "training_ready": False}), flush=True)
            except BaseException as exc:
                traceback.print_exc()
                write_json(args.out / "status.json", {"schema": BUILD_SCHEMA, "status": "failed",
                           "error": str(exc), "error_type": type(exc).__name__, "optimizer_steps": 0,
                           "training_started": False})
                raise
            finally:
                if source is not None:
                    close_source(source)


if __name__ == "__main__":
    main()
