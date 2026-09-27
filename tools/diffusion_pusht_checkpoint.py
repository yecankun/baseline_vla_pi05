"""Pinned legacy DP / Push-T migration and strict offline loading; no training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

REVISION = "84a7c23178445c6bbf7e1a884ff497017910f653"
SOURCE = Path("/home/zsw/models/project_2026/diffusion_pusht_84a7c231")
MIGRATED = SOURCE.with_name(SOURCE.name + "_lerobot044_v1")
SCHEMA = "diffusion_pusht_legacy_migration_v1"
PINS = {
    "model.safetensors": (1050862408, "995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8"),
    "config.json": (1509, "d391a7bf488accd1c26b2043482f0060b0855b1ec236f3d7358486918472c0a5"),
    "train_config.json": (5939, "500ea79ba1bef13810219697ce60eca580a0f259f5c9bf4f847f8f843b06b14a"),
    "README.md": (2939, "122b7d5067abfaf7e2ac44e6ab184bee8fb5c58d777e82d5c92bd3849dd308c3"),
}
STAT_KEYS = {
    "normalize_inputs.buffer_observation_image.mean": ("observation.image", "mean"),
    "normalize_inputs.buffer_observation_image.std": ("observation.image", "std"),
    "normalize_inputs.buffer_observation_state.min": ("observation.state", "min"),
    "normalize_inputs.buffer_observation_state.max": ("observation.state", "max"),
    "normalize_targets.buffer_action.min": ("action", "min"),
    "normalize_targets.buffer_action.max": ("action", "max"),
    "unnormalize_outputs.buffer_action.min": ("action", "min"),
    "unnormalize_outputs.buffer_action.max": ("action", "max"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_files(directory, pins):
    result = {}
    for name, (size, digest) in pins.items():
        require(Path(name).name == name, "Only flat pinned filenames are allowed")
        p = Path(directory) / name
        require(p.is_file() and p.stat().st_size == size, f"Size/missing file: {p}")
        actual = sha256(p)
        require(actual == digest, f"SHA256 mismatch: {p}")
        result[name] = {"bytes": size, "sha256": actual}
    return result


def tensor_signature(tensor):
    import torch
    require(tensor.dtype == torch.float32, "Expected source float32 tensor")
    t = tensor.detach().cpu().contiguous()
    require(bool(torch.isfinite(t).all()), "Nonfinite checkpoint tensor")
    return {"shape": list(t.shape), "dtype": str(t.dtype),
            "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()}


def split_legacy(state):
    import torch
    other = set(state) - {k for k in state if k.startswith("diffusion.")}
    require(other == set(STAT_KEYS), "Unexpected or missing legacy normalization keys")
    model = {k: v for k, v in state.items() if k not in STAT_KEYS}
    require(len(model) == 213 and len(state) == 221, "Unexpected tensor counts")
    stats = {}
    for key, (feature, kind) in STAT_KEYS.items():
        value = state[key]
        tensor_signature(value)
        expected_shape = (3, 1, 1) if feature == "observation.image" else (2,)
        require(tuple(value.shape) == expected_shape, f"Bad stats shape: {key}")
        group = stats.setdefault(feature, {})
        if kind in group:
            require(tensor_signature(group[kind]) == tensor_signature(value), "Action target/output stats disagree")
        group[kind] = value.clone()
    require(bool((stats["observation.image"]["std"] > 0).all()), "Invalid image std")
    for key in ("action", "observation.state"):
        require(bool((stats[key]["max"] > stats[key]["min"]).all()), f"Invalid min/max: {key}")
    require(list(model["diffusion.rgb_encoder.pool.pos_grid"].shape) == [9, 2], "Bad spatial grid")
    return model, stats


def make_config(raw, device="cpu"):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    values = dict(raw)
    require(values.pop("type") == "diffusion", "Wrong policy type")
    values["device"] = device
    require(values["use_amp"] is False, "AMP must remain disabled")
    for field in ("input_features", "output_features"):
        values[field] = {k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                         for k, v in values[field].items()}
    values["normalization_mapping"] = {k: NormalizationMode(v) for k, v in values["normalization_mapping"].items()}
    config = DiffusionConfig(**values)
    require(config.resize_shape is None and tuple(config.crop_shape) == (84, 84), "Visual config drift")
    require((config.n_obs_steps, config.horizon, config.n_action_steps) == (2, 16, 8), "Queue config drift")
    require(config.num_train_timesteps == 100 and config.num_inference_steps is None, "DDPM config drift")
    return config


def check_saved_config(original, saved):
    for key, value in original.items():
        if key != "device":
            require(saved.get(key) == value, f"Saved config changed original field: {key}")
    require(saved["device"] == "cpu", "Migrated on-disk device must be CPU")


def check_processors(pre, post, stats, device="cpu"):
    """Check saved statistics and actual transforms, including no input mutation."""
    import torch
    require([type(s).__name__ for s in pre.steps] == ["RenameObservationsProcessorStep",
            "AddBatchDimensionProcessorStep", "DeviceProcessorStep", "NormalizerProcessorStep"],
            "Unexpected preprocessor topology")
    require([type(s).__name__ for s in post.steps] == ["UnnormalizerProcessorStep", "DeviceProcessorStep"],
            "Unexpected postprocessor topology")
    require(pre.steps[0].rename_map == {}, "Unexpected observation renaming")
    require(str(pre.steps[2].device) == device and str(post.steps[1].device) == "cpu", "Processor device drift")
    expected = {f"{feature}.{kind}": value for feature, group in stats.items() for kind, value in group.items()}
    # Official pipelines each serialize the full six-item statistics dictionary.
    processor_stats = []
    for pipe, name in ((pre, "normalizer_processor"), (post, "unnormalizer_processor")):
        matches = [step for step in pipe.steps if getattr(step, "registry_name", None) == name]
        if not matches:
            # Public state_dict API; registry_name is not exposed in every build.
            matches = [step for step in pipe.steps if type(step).__name__ ==
                       ("NormalizerProcessorStep" if name == "normalizer_processor" else "UnnormalizerProcessorStep")]
        require(len(matches) == 1, "Expected exactly one normalization step")
        step = matches[0]
        require(step.eps == 1e-8 and step.normalize_observation_keys is None, "Normalization semantics drift")
        modes = {getattr(k, "value", k): getattr(v, "value", v) for k, v in step.norm_map.items()}
        require(modes == {"ACTION": "MIN_MAX", "STATE": "MIN_MAX", "VISUAL": "MEAN_STD"}, "Normalization mode drift")
        features = {k: (getattr(v.type, "value", v.type), tuple(v.shape)) for k, v in step.features.items()}
        expected_features = {"action": ("ACTION", (2,))}
        if pipe is pre:
            expected_features.update({"observation.image": ("VISUAL", (3, 96, 96)), "observation.state": ("STATE", (2,))})
        require(features == expected_features, "Normalization features drift")
        actual = matches[0].state_dict()
        require(set(actual) == set(expected), f"Processor statistics keys differ: {set(actual)}")
        for key, value in expected.items():
            require(tensor_signature(actual[key]) == tensor_signature(value), f"Processor statistics changed: {key}")
        processor_stats.append({k: tensor_signature(v) for k, v in actual.items()})
    fractions = torch.tensor([[-0.25], [0.0], [0.5], [1.0], [1.25]], dtype=torch.float32)
    state = stats["observation.state"]["min"] + fractions * (stats["observation.state"]["max"] - stats["observation.state"]["min"])
    act = stats["action"]["min"] + fractions * (stats["action"]["max"] - stats["action"]["min"])
    image = torch.linspace(0, 1, 5 * 3 * 96 * 96).reshape(5, 3, 96, 96)
    original = {"observation.image": image.clone(), "observation.state": state.clone(), "action": act.clone()}
    before = {k: v.clone() for k, v in original.items()}
    batch = pre(original)
    errors = {}
    for key, raw in before.items():
        group = stats[key]
        expected_value = ((raw - group["mean"]) / (group["std"] + 1e-8) if key == "observation.image"
                          else 2 * (raw - group["min"]) / (group["max"] - group["min"]) - 1)
        actual = batch[key].detach().cpu()
        require(actual.shape == expected_value.shape, f"Processor changed shape: {key}")
        require(torch.allclose(actual, expected_value, rtol=0, atol=1e-6), f"Transform mismatch: {key}")
        require(torch.equal(original[key], before[key]), f"Input mutated: {key}")
        errors[key] = float((actual - expected_value).abs().max())
    normalized = (2 * fractions - 1).expand(-1, 2).to(device)
    restored = post(normalized).detach().cpu()
    require(torch.allclose(restored, act, rtol=0, atol=1e-5), "Action unnormalization mismatch")
    errors["action_inverse"] = float((restored - act).abs().max())
    return {"saved_statistics": processor_stats, "transform_max_abs_errors": errors,
            "probe_fractions": [-0.25, 0, 0.5, 1, 1.25], "normalization_clipping": False,
            "historical_bitwise_execution_equivalence_claimed": False}


def _offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    require(importlib.metadata.version("lerobot") == "0.4.4", "This migration requires LeRobot 0.4.4")


def runtime_binding():
    distribution = importlib.metadata.distribution("lerobot")
    names = ("policies/diffusion/modeling_diffusion.py", "policies/diffusion/configuration_diffusion.py",
             "policies/diffusion/processor_diffusion.py", "processor/normalize_processor.py",
             "processor/pipeline.py", "policies/factory.py")
    return {"versions": {name: importlib.metadata.version(name) for name in
                         ("lerobot", "torch", "torchvision", "diffusers", "safetensors")},
            "source_sha256": {name: sha256(distribution.locate_file("lerobot/" + name)) for name in names}}


def _load_artifact(source, destination, device):
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    raw = json.loads((source / "config.json").read_text())
    saved = json.loads((destination / "config.json").read_text())
    check_saved_config(raw, saved)
    raw_state = load_file(str(source / "model.safetensors"))
    expected, stats = split_legacy(raw_state)
    actual = load_file(str(destination / "model.safetensors"))
    require(set(actual) == set(expected), "Migrated model keys changed")
    signatures = {}
    for key in expected:
        signature = tensor_signature(expected[key])
        require(tensor_signature(actual[key]) == signature, f"Migrated model tensor changed: {key}")
        signatures[key] = signature
    cfg = make_config(raw, "cpu")
    policy = DiffusionPolicy(cfg)
    require(set(policy.state_dict()) == set(actual), "Runtime model keys differ")
    loaded = policy.load_state_dict(actual, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, "Strict loading failed")
    for key, value in policy.state_dict().items():
        require(tensor_signature(value) == signatures[key], f"Loaded tensor changed: {key}")
    del raw_state, expected, actual
    gc.collect()
    policy.to(device).eval().requires_grad_(False)
    policy.config.device = device
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(destination),
                                        preprocessor_overrides={"device_processor": {"device": device}})
    probes = check_processors(pre, post, stats, device)
    policy.reset()
    pre.reset()
    post.reset()
    return policy, pre, post, {"policy_tensor_count": len(signatures), "strict_load": True,
                              "policy_tensor_signatures": signatures, "processor_checks": probes}


def prepare(source=SOURCE, destination=MIGRATED):
    _offline()
    import torch
    from safetensors.torch import load_file
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors
    started = time.monotonic()
    source, destination = Path(source), Path(destination)
    require(source.resolve().parent == destination.resolve().parent and source.resolve() != destination.resolve(),
            "Destination must be a separate sibling on the same model volume")
    pins = verify_files(source, PINS)
    destination.mkdir(exist_ok=False)
    write_json(destination / "started.json", {"pid": os.getpid(), "source": str(source.resolve()), "schema": SCHEMA})
    try:
        raw = json.loads((source / "config.json").read_text())
        state = load_file(str(source / "model.safetensors"))
        model_state, stats = split_legacy(state)
        policy = DiffusionPolicy(make_config(raw))
        policy.load_state_dict(model_state, strict=True)
        pre, post = make_pre_post_processors(policy.config, dataset_stats=stats)
        before_save = check_processors(pre, post, stats)
        pre.save_pretrained(destination)
        post.save_pretrained(destination)
        policy.save_pretrained(destination)
        # Preserve training and model-card provenance without regenerating it.
        for name in ("README.md", "train_config.json"):
            with (destination / name).open("xb") as f:
                f.write((source / name).read_bytes())
        del state, model_state, policy, pre, post
        gc.collect()
        _, _, _, checks = _load_artifact(source, destination, "cpu")
        require(checks["processor_checks"] == before_save, "Processor save/reload changed evidence")
        require(verify_files(source, PINS) == pins, "Source changed during migration")
        files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(destination.iterdir()) if p.is_file()}
        report = {"schema": SCHEMA, "status": "completed", "revision": REVISION,
                  "source_files": pins, "output_files": files, "checks": checks,
                  "source": str(source.resolve()), "destination": str(destination.resolve()),
                  "runtime_binding": runtime_binding(),
                  "script_sha256": sha256(__file__), "runtime_seconds": time.monotonic() - started,
                  "policy_forward_calls": 0, "optimizer_steps": 0,
                  "benchmark_score_claim_allowed": False, "visual_status": "not_viewed"}
        write_json(destination / "migration_report.json", report)
        return report
    except Exception:
        write_json(destination / "failure.json", {"status": "failed", "traceback": traceback.format_exc(),
                                                  "optimizer_steps": 0})
        raise


def load_verified(device="cuda"):
    _offline()
    require(not (MIGRATED / "failure.json").exists(), "Migration contains a failure marker")
    report_path = MIGRATED / "migration_report.json"
    report = json.loads(report_path.read_text())
    require(report["schema"] == SCHEMA and report["status"] == "completed" and report["revision"] == REVISION,
            "Missing or incomplete migration")
    require(report["script_sha256"] == sha256(__file__), "Migration helper changed; audit required")
    require(report["runtime_binding"] == runtime_binding(), "Migration runtime changed; audit required")
    verify_files(SOURCE, PINS)
    verify_files(MIGRATED, {k: (v["bytes"], v["sha256"]) for k, v in report["output_files"].items()})
    policy, pre, post, checks = _load_artifact(SOURCE, MIGRATED, device)
    require(checks["policy_tensor_signatures"] == report["checks"]["policy_tensor_signatures"], "Model binding changed")
    return policy, pre, post, {"revision": REVISION, "migration_report_sha256": sha256(report_path),
                              "source": str(SOURCE), "migrated": str(MIGRATED),
                              "model_sha256": sha256(MIGRATED / "model.safetensors"),
                              "strict_load": True, "processor_checks": checks["processor_checks"],
                              "policy_tensor_count": checks["policy_tensor_count"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("verify-source", "prepare"), required=True)
    args = parser.parse_args()
    if args.stage == "verify-source":
        result = {"status": "passed", "files": verify_files(SOURCE, PINS)}
    else:
        import torch
        torch.set_num_threads(1)
        result = prepare()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
