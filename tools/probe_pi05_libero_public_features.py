"""Seven-row public-demo frozen PI0.5 image probe; never a training pack.

Consumes an independently audited, hash-pinned public source subset. No source
download, simulator, action generation, optimizer, imputation, or guidewire
adapter is reachable from this entrypoint. A fresh output directory is required.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    from pi05_libero_world_model_adapter import (
        DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, adapt_native_record,
        sha256_file, validate_task_registry,
    )
else:
    from .pi05_libero_world_model_adapter import (
        DEMO_REPO, DEMO_REVISION, IMAGE_KEYS, adapt_native_record,
        sha256_file, validate_task_registry,
    )


SCHEMA = "pi05_libero_public_feature_probe_v1"
SOURCE_SCHEMA = "pi05_libero_public_source_audit_v1"
MODEL_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
MODEL_SHA256 = "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae"
DEFAULT_CHECKPOINT = Path("/home/zsw/models/project_2026/pi05_libero_finetuned_8e174154")
ORIENTATION = "stored_dataset_rgb_no_extra_flip"
REPEAT_ATOL = 1e-3
REPEAT_RTOL = 1e-3
ROW_KEYS = {"episode_index", "frame_index", "task_index", "task", "observation.state", "action"}
SOURCE_TASK_INDICES = [34, 37, 38, 35, 31, 32, 30, 33, 36, 39]
EMPTY_IMAGE_KEY = "observation.images.empty_camera_0"


def read_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def checked_hash(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256")
    return value


def checked_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def local_file(root: Path, name: Any) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("source path must be a nonempty relative POSIX path")
    relative = Path(name)
    if relative.is_absolute() or ":" in name or ".." in relative.parts:
        raise ValueError("source path must stay within source root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"source file is missing or escapes source root: {name}")
    return path


def validate_source(root: Path, report_sha256: str) -> dict[str, Any]:
    """All source and native-record checks precede any heavyweight model import."""
    root = root.resolve()
    report_path = local_file(root, "report.json")
    if sha256_file(report_path) != checked_hash(report_sha256):
        raise ValueError("source report SHA256 does not match pinned input")
    report = read_json(report_path)
    if report.get("schema") != SOURCE_SCHEMA or report.get("status") != "passed":
        raise ValueError("passed public source acquisition audit required")
    source = report.get("source", {})
    if any(source.get(key) != value for key, value in {
        "kind": "public_demonstrations", "repo_id": DEMO_REPO, "revision": DEMO_REVISION,
    }.items()):
        raise ValueError("source must be the pinned public demonstrations, not rollouts or synthetic data")
    registry = report.get("task_registry")
    tasks = validate_task_registry(registry)
    if set(tasks) != set(range(10)):
        raise ValueError("complete ten-task native Spatial mapping is required")
    if [tasks[index]["source_task_index"] for index in range(10)] != SOURCE_TASK_INDICES:
        raise ValueError("native source-to-Spatial mapping differs from audited pinned revision")
    selected = report.get("selected_episode", {})
    episode_index = checked_int(selected.get("episode_index"), "episode_index")
    task_id = checked_int(selected.get("task_id"), "task_id")
    source_task = checked_int(selected.get("source_task_index"), "source_task_index")
    checked_int(selected.get("record_count"), "complete episode record_count", 7)
    if selected.get("complete_episode") is not True or task_id not in tasks:
        raise ValueError("audited complete source episode is required")
    if tasks[task_id]["source_task_index"] != source_task:
        raise ValueError("selected episode differs from native registry")

    files = report.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("hash-bound source metadata and payload files required")
    file_hashes = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("invalid source file inventory entry")
        name = entry.get("local_path")
        path = local_file(root, name)
        if name in file_hashes:
            raise ValueError("duplicate source inventory path")
        digest = checked_hash(entry.get("sha256"))
        if path.stat().st_size != checked_int(entry.get("size"), "size") or sha256_file(path) != digest:
            raise ValueError(f"source inventory hash/size mismatch: {name}")
        file_hashes[name] = digest

    probe = report.get("probe", {})
    if (probe.get("frame_indices") != list(range(7))
            or any(type(index) is not int for index in probe["frame_indices"])
            or type(probe.get("row_count")) is not int
            or probe["row_count"] != 7 or probe.get("orientation") != ORIENTATION):
        raise ValueError("probe requires exactly native frames 0..6 with stored RGB orientation")
    for stem, expected in (("images", "probe_images.npy"), ("records", "probe_records.json")):
        name = probe.get(f"{stem}_path")
        digest = checked_hash(probe.get(f"{stem}_sha256"))
        if name != expected or sha256_file(local_file(root, name)) != digest:
            raise ValueError(f"probe {stem} must be hash-bound in source report")
        file_hashes[name] = digest
    output_hashes = report.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("source derived output hash inventory required")
    for name, digest in output_hashes.items():
        if sha256_file(local_file(root, name)) != checked_hash(digest):
            raise ValueError(f"source derived output hash mismatch: {name}")
        file_hashes[name] = digest
    images = np.load(local_file(root, probe["images_path"]), allow_pickle=False)
    if images.dtype != np.uint8 or images.shape != (7, 2, 256, 256, 3):
        raise ValueError("images must be uint8 [7,2,256,256,3]; no fallback or layout inference")
    records = read_json(local_file(root, probe["records_path"]))
    if not isinstance(records, list) or len(records) != 7:
        raise ValueError("exactly seven selected native records required")
    for index, row in enumerate(records):
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError("probe record requires only native IDs/task/state/action without missing masks or truth")
        adapted = adapt_native_record(row, registry)
        if (adapted["episode_index"] != episode_index or adapted["frame_index"] != index
                or adapted["task_id"] != task_id or row["task_index"] != source_task):
            raise ValueError("probe rows must be consecutive within the selected native episode/task")
        if not adapted["state_valid"].all():
            raise ValueError("probe does not permit state imputation")
    # Retain the entire independently audited file inventory in provenance;
    # checking this subset does not reconstruct the acquisition audit itself.
    return {"report": report, "report_sha256": report_sha256, "images": images,
            "records": records, "verified_files": file_hashes}


def tensor_record(tensor) -> dict:
    import torch
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "min": float(value.float().min()), "max": float(value.float().max()),
            "finite": bool(torch.isfinite(value).all())}


def _restore(obj, key, previous, present):
    if present:
        setattr(obj, key, previous)
    else:
        delattr(obj, key)


def extract_probe(policy, images: np.ndarray, out: Path | None = None) -> tuple[np.ndarray, dict]:
    """Small real/fake-policy testable runner; does not load or mutate weights."""
    import torch
    from PIL import Image
    if __package__ in {None, ""}:
        from pi05_libero_feature_extraction import extract_native_visual_features
    else:
        from .pi05_libero_feature_extraction import extract_native_visual_features

    if images.dtype != np.uint8 or images.shape != (7, 2, 256, 256, 3):
        raise ValueError("exact seven-row stored RGB probe required")
    if any(module.training for module in policy.modules()) or any(p.requires_grad for p in policy.parameters()):
        raise ValueError("all policy modules must be eval and every parameter frozen")
    parameters = list(policy.parameters())
    if any(p.grad is not None for p in parameters):
        raise ValueError("frozen probe must start without parameter gradients")
    device = parameters[0].device if parameters else torch.device("cpu")
    vision = policy.model.paligemma_with_expert
    original_preprocess = policy._preprocess_images
    original_embed = vision.embed_image
    preprocess_present = "_preprocess_images" in policy.__dict__
    embed_present = "embed_image" in vision.__dict__
    preprocess_previous = policy.__dict__.get("_preprocess_images")
    embed_previous = vision.__dict__.get("embed_image")
    calls, embeds = [], []

    def traced_preprocess(batch):
        processed, masks = original_preprocess(batch)
        item = {"call_index": len(calls), "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "input_keys": list(batch), "inputs": {key: tensor_record(value) for key, value in batch.items()},
                "processed": [tensor_record(value) for value in processed],
                "masks": [mask.detach().cpu().tolist() for mask in masks]}
        if out is not None and not calls:
            for view in range(2):
                Image.fromarray(images[0, view]).save(out / f"source_frame000_view{view}_stored_rgb.png")
                preview = ((processed[view][0].detach().float().cpu().permute(1, 2, 0) + 1) * 127.5).round()
                if bool((preview < 0).any()) or bool((preview > 255).any()):
                    raise ValueError("processed preview is outside [-1,1]")
                Image.fromarray(preview.to(torch.uint8).numpy()).save(out / f"processed_frame000_view{view}_policy_rgb.png")
        calls.append(item)
        return processed, masks

    def traced_embed(value):
        result = original_embed(value)
        embeds.append({"call_index": len(embeds), "grad_enabled": torch.is_grad_enabled(),
                       "inference_mode": torch.is_inference_mode_enabled(),
                       "image": tensor_record(value), "tokens": tensor_record(result)})
        return result

    policy._preprocess_images = traced_preprocess
    vision.embed_image = traced_embed
    latent_rows = []
    input_hashes = []
    try:
        for index in [*range(7), 0]:
            batch = {key: torch.from_numpy(images[index, view].copy()).permute(2, 0, 1)
                     .unsqueeze(0).to(device=device, dtype=torch.float32).div(255)
                     for view, key in enumerate(IMAGE_KEYS)}
            input_hashes.append({key: tensor_record(value)["sha256"] for key, value in batch.items()})
            before = len(calls), len(embeds)
            latent = extract_native_visual_features(policy, batch)
            if len(calls) != before[0] + 1 or len(embeds) != before[1] + 2:
                raise ValueError("extractor must preprocess once and embed only the two real cameras")
            if tuple(latent.shape) != (1, 2, 2048) or not torch.isfinite(latent).all() or latent.requires_grad:
                raise ValueError("real PI0.5 pooled feature must be finite frozen [1,2,2048]")
            if any(tensor_record(batch[key])["sha256"] != input_hashes[-1][key] for key in IMAGE_KEYS):
                raise ValueError("image input changed during extraction")
            latent_rows.append(latent.detach().float().cpu().numpy()[0].copy())
    finally:
        _restore(policy, "_preprocess_images", preprocess_previous, preprocess_present)
        _restore(vision, "embed_image", embed_previous, embed_present)

    latents = np.stack(latent_rows[:7])
    difference = np.abs(latent_rows[0] - latent_rows[7])
    repeat = {"row": 0, "atol": REPEAT_ATOL, "rtol": REPEAT_RTOL,
              "max_abs_difference": float(difference.max()), "mean_abs_difference": float(difference.mean()),
              "bitwise_equal": bool(np.array_equal(latent_rows[0], latent_rows[7])),
              "allclose": bool(np.allclose(latent_rows[0], latent_rows[7], atol=REPEAT_ATOL, rtol=REPEAT_RTOL)),
              "input_hashes_equal": input_hashes[0] == input_hashes[7]}
    if not repeat["allclose"] or not repeat["input_hashes_equal"]:
        raise ValueError(f"repeat-first feature probe failed prespecified numerical tolerance: {repeat}")
    if (any(module.training for module in policy.modules()) or any(p.requires_grad or p.grad is not None for p in parameters)
            or any(item["grad_enabled"] or not item["inference_mode"] for item in calls + embeds)):
        raise ValueError("eval/no-grad/inference-only execution proof failed")
    return latents, {"preprocessing_calls": calls, "embed_calls": embeds, "repeat_first": repeat,
                     "preprocessing_call_count": len(calls), "embed_call_count": len(embeds),
                     "batch_size": 1, "all_modules_eval": True, "all_parameters_frozen": True,
                     "all_parameter_gradients_absent": True, "parameter_tensor_count": len(parameters),
                     "all_calls_inference_mode": True, "inputs_unchanged": True,
                     "latent_shape": list(latents.shape), "latent_dtype": str(latents.dtype),
                     "latent_finite": bool(np.isfinite(latents).all()),
                     "row_variation_max_abs_from_first": float(np.max(np.abs(latents - latents[:1])))}


def validate_checkpoint_config(config: Any) -> dict:
    """Validate this pinned checkpoint's two real views and declared empty view.

    The stored checkpoint already declares empty_camera_0 before constructing
    PI05Policy. Do not invent a two-view pre-construction configuration or add
    a replacement view. The empty image is generated/masked by native policy
    preprocessing and is never an extractor input or embedded visual target.
    """
    expected_images = {
        IMAGE_KEYS[0]: [3, 256, 256], IMAGE_KEYS[1]: [3, 256, 256],
        EMPTY_IMAGE_KEY: [3, 224, 224],
    }
    if list(config.image_features) != list(expected_images):
        raise ValueError("pinned checkpoint requires ordered two real cameras plus declared empty_camera_0")
    if set(config.input_features) != {*expected_images, "observation.state"}:
        raise ValueError("pinned checkpoint input feature keys differ")
    for key, shape in expected_images.items():
        if (list(config.image_features[key].shape) != shape
                or list(config.input_features[key].shape) != shape):
            raise ValueError(f"pinned checkpoint image shape differs: {key}")
    if (list(config.image_resolution) != [224, 224] or type(config.empty_cameras) is not int
            or config.empty_cameras != 1 or config.dtype != "bfloat16"
            or list(config.input_features["observation.state"].shape) != [8]
            or set(config.output_features) != {"action"}
            or list(config.output_features["action"].shape) != [7]):
        raise ValueError("pinned checkpoint native resolution/state/action/dtype/empty-camera layout differs")
    return {"image_features": expected_images, "image_resolution": [224, 224],
            "empty_cameras": 1, "state_dimension": 8, "action_dimension": 7,
            "dtype": "bfloat16", "real_extractor_input_keys": list(IMAGE_KEYS),
            "empty_camera_embedded": False}


def load_policy(checkpoint: Path, protocol_path: Path, device: str):
    """Offline strict checkpoint and image contract verification, no processors/actions."""
    protocol = read_json(protocol_path)
    route = protocol["checkpoint_routes"]["pi05_primary"]
    if (route["repo_id"] != "lerobot/pi05_libero_finetuned" or route["revision"] != MODEL_REVISION
            or route["model_sha256"] != MODEL_SHA256 or route["model_file"] != "model.safetensors"):
        raise ValueError("checkpoint route differs from pinned LIBERO fine-tuned identity")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError("offline local checkpoint directory required; no download fallback")
    model_path = checkpoint / "model.safetensors"
    model_hash = sha256_file(model_path)
    if model_hash != MODEL_SHA256 or model_path.stat().st_size != route["model_size_bytes"]:
        raise ValueError("checkpoint size/hash differs from pinned model")
    versions = {key: importlib.metadata.version(key) for key in ("lerobot", "transformers", "tokenizers")}
    if any(version != protocol["software_contract"][key] for key, version in versions.items()):
        raise ValueError(f"pinned runtime dependency drift: {versions}")
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    if __package__ in {None, ""}:
        from pi05_pretrained_loader import load_verified_pi05_weights
    else:
        from .pi05_pretrained_loader import load_verified_pi05_weights

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA unavailable; no CPU/fake-policy fallback")
    torch.manual_seed(1000)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    config.compile_model = False
    config.gradient_checkpointing = False
    checkpoint_layout = validate_checkpoint_config(config)
    policy = PI05Policy(config).to(device)
    load_report = load_verified_pi05_weights(
        policy, checkpoint, revision=None, local_files_only=True,
        min_loaded_parameter_fraction=1.0, max_missing_keys=0, max_unexpected_keys=0,
        compute_file_sha256=False,
    )
    if (load_report["status"] != "loaded" or load_report["loaded_parameter_fraction"] != 1.0
            or load_report["load_state_dict_missing_keys"] or load_report["load_state_dict_unexpected_keys"]):
        raise ValueError("strict verified checkpoint loading did not complete")
    policy.eval()
    policy.requires_grad_(False)
    parameter_dtypes = {}
    for parameter in policy.parameters():
        name = str(parameter.dtype)
        parameter_dtypes.setdefault(name, {"parameter_tensors": 0, "numel": 0})
        parameter_dtypes[name]["parameter_tensors"] += 1
        parameter_dtypes[name]["numel"] += parameter.numel()
    installed_sources = {}
    for owner, method in (("policy_class", type(policy)), ("image_preprocess", policy._preprocess_images),
                          ("image_embedding", policy.model.paligemma_with_expert.embed_image)):
        path = Path(inspect.getsourcefile(method))
        installed_sources[owner] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return policy, {"model_sha256": model_hash, "config_sha256": sha256_file(checkpoint / "config.json"),
                    "protocol_sha256": sha256_file(protocol_path), "checkpoint_path": str(checkpoint),
                    "device": device, "versions": versions, "compile_model": False,
                    "gradient_checkpointing": False, "validated_checkpoint_layout": checkpoint_layout,
                    "parameter_dtypes": parameter_dtypes,
                    "installed_implementation_sources": installed_sources, "load": load_report}


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=Path("docs/libero-spatial-score-protocol-v1.json"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    write_json(args.out / "status.json", {"schema": SCHEMA, "status": "running", "optimizer_steps": 0})
    with (args.out / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                source = validate_source(args.source_root, args.source_report_sha256)
                print("SOURCE_PROVENANCE_AND_SEVEN_NATIVE_ROWS_VERIFIED", flush=True)
                policy, checkpoint = load_policy(args.checkpoint, args.protocol, args.device)
                print("PINNED_CHECKPOINT_STRICTLY_LOADED", flush=True)
                latents, evidence = extract_probe(policy, source["images"], args.out)
                np.save(args.out / "visual_latent.npy", latents, allow_pickle=False)
                source_files = [Path(__file__), Path(__file__).with_name("pi05_libero_feature_extraction.py"),
                                Path(__file__).with_name("pi05_action_effect_world_model.py"),
                                Path(__file__).with_name("pi05_libero_world_model_adapter.py"),
                                Path(__file__).with_name("pi05_pretrained_loader.py")]
                report = {"schema": SCHEMA, "status": "passed", "source_report_sha256": source["report_sha256"],
                          "source": source["report"]["source"], "source_root": str(args.source_root.resolve()),
                          "task_registry": source["report"]["task_registry"],
                          "selected_episode": source["report"]["selected_episode"],
                          "source_files_verified": source["verified_files"],
                          "probe_frame_indices": list(range(7)), "probe_row_count": 7,
                          "probe_is_complete_episode": False, "orientation": ORIENTATION,
                          "checkpoint": checkpoint, "evidence": evidence,
                          "implementation_hashes": {path.name: sha256_file(path) for path in source_files},
                          "artifacts": {path.name: sha256_file(path) for path in args.out.iterdir()
                                        if path.suffix in {".npy", ".png"}},
                          "runtime_seconds": time.monotonic() - started,
                          "visual_status": "not_viewed", "optimizer_steps": 0, "actions_generated": 0,
                          "training_started": False, "training_ready": False, "feature_pack_complete": False,
                          "split_created": False, "world_model_quality_evaluated": False,
                          "policy_performance_evaluated": False, "real_system_validated": False}
                write_json(args.out / "report.json", report)
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "completed",
                           "report_sha256": sha256_file(args.out / "report.json"), "optimizer_steps": 0})
                print("FROZEN_PUBLIC_FEATURE_PROBE_COMPLETE", json.dumps({"latent_shape": list(latents.shape),
                      "runtime_seconds": report["runtime_seconds"], "training_ready": False}), flush=True)
            except Exception as exc:
                traceback.print_exc()
                write_json(args.out / "status.json", {"schema": SCHEMA, "status": "failed", "error": str(exc),
                           "error_type": type(exc).__name__, "optimizer_steps": 0, "training_started": False})
                raise


if __name__ == "__main__":
    main()
