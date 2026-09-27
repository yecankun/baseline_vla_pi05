from __future__ import annotations

import hashlib
import json
import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch


DEFAULT_PI05_REPO = "lerobot/pi05_base"
DEFAULT_PI05_REVISION = "7de663972b7817d2c4cf2d84c821153dfea772e9"
VERIFIED_TIED_WEIGHT_ALIASES = (
    (
        "model.paligemma_with_expert.paligemma.lm_head.weight",
        "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
    ),
)


class PI05PretrainedLoadRejected(ValueError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def add_pretrained_arguments(parser: ArgumentParser, *, allow_random_init: bool = True) -> None:
    parser.add_argument("--pretrained-name-or-path")
    parser.add_argument("--pretrained-revision", default=DEFAULT_PI05_REVISION)
    parser.add_argument("--pretrained-cache-dir", type=Path)
    parser.add_argument("--pretrained-local-files-only", action="store_true")
    parser.add_argument("--pretrained-min-loaded-parameter-fraction", type=float, default=0.99)
    parser.add_argument("--pretrained-max-missing-keys", type=int, default=0)
    parser.add_argument("--pretrained-max-unexpected-keys", type=int, default=0)
    parser.add_argument("--pretrained-compute-file-sha256", action="store_true")
    if allow_random_init:
        parser.add_argument(
            "--allow-random-init",
            action="store_true",
            help="Explicitly allow a randomly initialized PI05 policy for architecture-only diagnostics.",
        )


def initialize_pi05_policy(
    policy_class: type[torch.nn.Module],
    config: Any,
    args: Namespace,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    source = getattr(args, "pretrained_name_or_path", None)
    allow_random_init = bool(getattr(args, "allow_random_init", False))
    if not source and not allow_random_init:
        raise ValueError(
            "PI05 policy initialization is fail-closed: provide --pretrained-name-or-path "
            "or explicitly opt into --allow-random-init for architecture-only diagnostics"
        )
    policy = policy_class(config).to(args.device)
    if source:
        report = load_verified_pi05_weights(
            policy,
            source,
            revision=getattr(args, "pretrained_revision", None),
            cache_dir=getattr(args, "pretrained_cache_dir", None),
            local_files_only=bool(getattr(args, "pretrained_local_files_only", False)),
            min_loaded_parameter_fraction=float(
                getattr(args, "pretrained_min_loaded_parameter_fraction", 0.99)
            ),
            max_missing_keys=int(getattr(args, "pretrained_max_missing_keys", 0)),
            max_unexpected_keys=int(getattr(args, "pretrained_max_unexpected_keys", 0)),
            compute_file_sha256=bool(getattr(args, "pretrained_compute_file_sha256", False)),
        )
        return policy, report
    if allow_random_init:
        return policy, {
            "status": "random_init_explicitly_allowed",
            "warning": "architecture-only diagnostic; not a pretrained PI05 baseline",
            "project_config": _project_config(policy),
        }
    raise AssertionError("unreachable PI05 initialization state")


def inspect_huggingface_repo(repo_id: str, revision: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision) and info.sha != revision:
        raise ValueError(
            f"resolved Hugging Face revision {info.sha!r} does not match pinned revision {revision!r}"
        )
    files = []
    for sibling in info.siblings or []:
        files.append(
            {
                "name": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
            }
        )
    return {
        "repo_id": info.id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "private": bool(info.private),
        "gated": info.gated,
        "last_modified": str(info.last_modified) if info.last_modified is not None else None,
        "files": files,
    }


def _snapshot_revision(path: Path) -> str | None:
    parts = path.absolute().parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def resolve_pretrained_files(
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> dict[str, Any]:
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.is_dir():
            model_path = source_path / "model.safetensors"
            config_path = source_path / "config.json"
        else:
            model_path = source_path
            config_path = source_path.with_name("config.json")
        if not model_path.is_file():
            raise FileNotFoundError(f"model.safetensors not found under {source_path}")
        return {
            "source_kind": "local",
            "source": str(source_path.resolve()),
            "requested_revision": revision,
            "resolved_revision": _snapshot_revision(model_path),
            "model_path": str(model_path.resolve()),
            "config_path": str(config_path.resolve()) if config_path.is_file() else None,
            "model_size_bytes": model_path.stat().st_size,
        }

    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": str(source),
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    model_path = Path(hf_hub_download(filename="model.safetensors", **kwargs)).absolute()
    config_path = Path(hf_hub_download(filename="config.json", **kwargs)).absolute()
    resolved_revision = _snapshot_revision(model_path)
    if revision and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        if resolved_revision != revision:
            raise ValueError(
                f"resolved Hugging Face revision {resolved_revision!r} does not match pinned revision {revision!r}"
            )
    return {
        "source_kind": "huggingface",
        "source": str(source),
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "model_size_bytes": model_path.stat().st_size,
    }


def sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor, *, sample_values: int = 1024) -> str:
    flat = tensor.detach().reshape(-1)
    sample = flat[: min(flat.numel(), sample_values)].float().cpu().contiguous()
    return hashlib.sha256(sample.numpy().tobytes()).hexdigest()


def _same_tensor_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and left.storage_offset() == right.storage_offset()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _storage_identity(key: str, tensor: torch.Tensor) -> tuple[Any, ...]:
    if tensor.numel() == 0:
        return ("empty", key)
    return (
        str(tensor.device),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _resolve_verified_tied_aliases(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    directly_matched_keys: set[str],
) -> tuple[dict[str, torch.Tensor], set[str], list[dict[str, Any]]]:
    effective_state = dict(checkpoint_state)
    effective_matched_keys = set(directly_matched_keys)
    aliases = []
    for source_key, target_key in VERIFIED_TIED_WEIGHT_ALIASES:
        if source_key not in directly_matched_keys or target_key in directly_matched_keys:
            continue
        if target_key not in model_state:
            continue
        source_model_tensor = model_state[source_key]
        target_model_tensor = model_state[target_key]
        source_checkpoint_tensor = checkpoint_state[source_key]
        if tuple(source_checkpoint_tensor.shape) != tuple(target_model_tensor.shape):
            continue
        if not _same_tensor_storage(source_model_tensor, target_model_tensor):
            continue
        effective_state[target_key] = source_checkpoint_tensor
        effective_matched_keys.add(target_key)
        aliases.append(
            {
                "checkpoint_key": source_key,
                "model_alias_key": target_key,
                "shape": list(source_checkpoint_tensor.shape),
                "reason": "verified shared storage in the instantiated PI05 model",
            }
        )
    return effective_state, effective_matched_keys, aliases


def _unique_parameter_coverage(
    model_state: dict[str, torch.Tensor],
    loaded_keys: set[str],
) -> tuple[int, int, float]:
    unique_tensors: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, tensor in model_state.items():
        identity = _storage_identity(key, tensor)
        record = unique_tensors.setdefault(
            identity,
            {"numel": int(tensor.numel()), "loaded": False},
        )
        if key in loaded_keys:
            record["loaded"] = True
    total_numel = sum(record["numel"] for record in unique_tensors.values())
    loaded_numel = sum(
        record["numel"] for record in unique_tensors.values() if record["loaded"]
    )
    return loaded_numel, total_numel, loaded_numel / max(total_numel, 1)


def _project_config(policy: torch.nn.Module) -> dict[str, Any]:
    config = policy.config
    keys = (
        "paligemma_variant",
        "action_expert_variant",
        "device",
        "dtype",
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "image_resolution",
        "tokenizer_max_length",
        "freeze_vision_encoder",
        "train_expert_only",
        "gradient_checkpointing",
    )
    report: dict[str, Any] = {}
    for key in keys:
        value = getattr(config, key, None)
        report[key] = list(value) if isinstance(value, tuple) else value
    return report


def _source_config(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    config_path = Path(path)
    if not config_path.is_file():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def _architecture_compatibility(
    project_config: dict[str, Any],
    source_config: dict[str, Any] | None,
) -> dict[str, Any]:
    if source_config is None:
        return {
            "checked": False,
            "mismatches": ["source config.json unavailable"],
            "non_parameter_differences": [],
        }
    checks = {
        "paligemma_variant": source_config.get("paligemma_variant"),
        "action_expert_variant": source_config.get("action_expert_variant"),
        "max_state_dim": source_config.get("max_state_dim"),
        "max_action_dim": source_config.get("max_action_dim"),
        "image_resolution": source_config.get("image_resolution"),
    }
    mismatches = []
    for key, source_value in checks.items():
        project_value = project_config.get(key)
        if project_value != source_value:
            mismatches.append(
                {
                    "field": key,
                    "project": project_value,
                    "source": source_value,
                }
            )
    non_parameter_differences = []
    for key in ("device", "dtype", "chunk_size", "n_action_steps", "gradient_checkpointing"):
        project_value = project_config.get(key)
        source_value = source_config.get(key)
        if project_value != source_value:
            non_parameter_differences.append(
                {
                    "field": key,
                    "project": project_value,
                    "source": source_value,
                }
            )
    return {
        "checked": True,
        "mismatches": mismatches,
        "non_parameter_differences": non_parameter_differences,
    }


def load_verified_pi05_weights(
    policy: torch.nn.Module,
    source: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    min_loaded_parameter_fraction: float = 0.99,
    max_missing_keys: int = 0,
    max_unexpected_keys: int = 0,
    fingerprint_keys: int = 8,
    compute_file_sha256: bool = False,
) -> dict[str, Any]:
    if not 0.0 < min_loaded_parameter_fraction <= 1.0:
        raise ValueError("min_loaded_parameter_fraction must be in (0, 1]")

    resolved = resolve_pretrained_files(
        source,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model_path = Path(resolved["model_path"])
    source_config = _source_config(resolved.get("config_path"))
    project_config = _project_config(policy)
    compatibility = _architecture_compatibility(project_config, source_config)
    report: dict[str, Any] = {
        "status": "preflight",
        "resolved": resolved,
        "source_config": source_config,
        "project_config": project_config,
        "architecture_compatibility": compatibility,
    }
    if compatibility["mismatches"]:
        report["status"] = "rejected"
        report["errors"] = [f"architecture mismatches={compatibility['mismatches']}"]
        raise PI05PretrainedLoadRejected("pretrained architecture mismatch", report)

    from safetensors.torch import load_file

    original_state = load_file(str(model_path), device="cpu")
    fixed_state = policy._fix_pytorch_state_dict_keys(original_state, policy.config)
    remapped_state = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in fixed_state.items()
    }
    model_state = policy.state_dict()

    matched_keys = []
    mismatched_shapes = []
    unexpected_keys = []
    for key, value in remapped_state.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            mismatched_shapes.append(
                {
                    "key": key,
                    "checkpoint": list(value.shape),
                    "model": list(model_state[key].shape),
                }
            )
            continue
        matched_keys.append(key)

    matched_set = set(matched_keys)
    raw_missing_keys = sorted(key for key in model_state if key not in matched_set)
    effective_state, effective_matched_set, tied_aliases = _resolve_verified_tied_aliases(
        remapped_state,
        model_state,
        matched_set,
    )
    missing_keys = sorted(key for key in model_state if key not in effective_matched_set)
    unexpected_keys = sorted(unexpected_keys)
    matched_keys = sorted(matched_keys)
    raw_total_state_numel = sum(value.numel() for value in model_state.values())
    raw_loaded_state_numel = sum(model_state[key].numel() for key in matched_keys)
    raw_loaded_state_fraction = raw_loaded_state_numel / max(raw_total_state_numel, 1)
    loaded_parameter_numel, total_parameter_numel, loaded_fraction = _unique_parameter_coverage(
        model_state,
        effective_matched_set,
    )

    errors = []
    if mismatched_shapes:
        errors.append(f"shape mismatches={len(mismatched_shapes)}")
    if len(missing_keys) > max_missing_keys:
        errors.append(f"missing keys={len(missing_keys)} > {max_missing_keys}")
    if len(unexpected_keys) > max_unexpected_keys:
        errors.append(f"unexpected keys={len(unexpected_keys)} > {max_unexpected_keys}")
    if loaded_fraction < min_loaded_parameter_fraction:
        errors.append(
            f"loaded parameter fraction={loaded_fraction:.6f} < {min_loaded_parameter_fraction:.6f}"
        )
    report.update(
        {
            "checkpoint_key_count": len(remapped_state),
            "model_key_count": len(model_state),
            "matched_key_count": len(matched_keys),
            "raw_missing_key_count": len(raw_missing_keys),
            "missing_key_count": len(missing_keys),
            "unexpected_key_count": len(unexpected_keys),
            "mismatched_shape_count": len(mismatched_shapes),
            "raw_missing_keys": raw_missing_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_shapes": mismatched_shapes,
            "tied_weight_alias_count": len(tied_aliases),
            "tied_weight_aliases": tied_aliases,
            "raw_loaded_state_numel": int(raw_loaded_state_numel),
            "raw_total_state_numel": int(raw_total_state_numel),
            "raw_loaded_state_fraction": float(raw_loaded_state_fraction),
            "loaded_parameter_numel": int(loaded_parameter_numel),
            "total_parameter_numel": int(total_parameter_numel),
            "loaded_parameter_fraction": float(loaded_fraction),
        }
    )
    if errors:
        report["status"] = "rejected"
        report["errors"] = errors
        raise PI05PretrainedLoadRejected("verified PI05 load rejected: " + "; ".join(errors), report)

    candidate_fingerprints = []
    for key in matched_keys:
        before = _tensor_fingerprint(model_state[key])
        checkpoint = _tensor_fingerprint(remapped_state[key])
        if before != checkpoint:
            candidate_fingerprints.append((key, before, checkpoint))
        if len(candidate_fingerprints) >= max(int(fingerprint_keys), 0):
            break
    if fingerprint_keys > 0 and not candidate_fingerprints:
        report["status"] = "rejected"
        report["errors"] = ["no sampled checkpoint tensor differs from random initialization"]
        raise PI05PretrainedLoadRejected(report["errors"][0], report)
    sample_keys = [item[0] for item in candidate_fingerprints]
    before_fingerprints = {item[0]: item[1] for item in candidate_fingerprints}
    checkpoint_fingerprints = {item[0]: item[2] for item in candidate_fingerprints}
    incompatible = policy.load_state_dict(effective_state, strict=False)
    after_state = policy.state_dict()
    after_fingerprints = {key: _tensor_fingerprint(after_state[key]) for key in sample_keys}
    fingerprint_report = {
        "keys": sample_keys,
        "random_to_loaded_changed": [
            key for key in sample_keys if before_fingerprints[key] != after_fingerprints[key]
        ],
        "loaded_matches_checkpoint": [
            key for key in sample_keys if checkpoint_fingerprints[key] == after_fingerprints[key]
        ],
    }
    if len(fingerprint_report["loaded_matches_checkpoint"]) != len(sample_keys):
        report["status"] = "rejected"
        report["errors"] = ["post-load fingerprints do not match checkpoint tensors"]
        report["fingerprints"] = fingerprint_report
        raise PI05PretrainedLoadRejected(report["errors"][0], report)

    alias_fingerprints = []
    for alias in tied_aliases:
        source_key = alias["checkpoint_key"]
        target_key = alias["model_alias_key"]
        checkpoint_fingerprint = _tensor_fingerprint(remapped_state[source_key])
        source_fingerprint = _tensor_fingerprint(after_state[source_key])
        target_fingerprint = _tensor_fingerprint(after_state[target_key])
        alias_fingerprints.append(
            {
                **alias,
                "source_matches_checkpoint": source_fingerprint == checkpoint_fingerprint,
                "alias_matches_checkpoint": target_fingerprint == checkpoint_fingerprint,
                "shared_storage_after_load": _same_tensor_storage(
                    after_state[source_key], after_state[target_key]
                ),
            }
        )
    if any(
        not item["source_matches_checkpoint"]
        or not item["alias_matches_checkpoint"]
        or not item["shared_storage_after_load"]
        for item in alias_fingerprints
    ):
        report["status"] = "rejected"
        report["errors"] = ["verified tied-weight alias did not match checkpoint after loading"]
        report["tied_weight_alias_verification"] = alias_fingerprints
        raise PI05PretrainedLoadRejected(report["errors"][0], report)

    report.update(
        {
            "status": "loaded",
            "load_state_dict_missing_keys": list(incompatible.missing_keys),
            "load_state_dict_unexpected_keys": list(incompatible.unexpected_keys),
            "fingerprints": fingerprint_report,
            "tied_weight_alias_verification": alias_fingerprints,
        }
    )
    if compute_file_sha256:
        report["resolved"]["model_sha256"] = sha256_file(model_path)

    del original_state
    del fixed_state
    del remapped_state
    del effective_state
    return report
