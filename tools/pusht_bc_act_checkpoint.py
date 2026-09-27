"""Explicit, hash-verified checkpoints for local user-owned BC/ACT runs.

Never pass downloaded or otherwise untrusted pickle files to this loader.
The colocated manifest detects accidental corruption, not malicious replacement.
The caller must construct model and optimizer on the requested device before
loading, and must restore any private sampler generator from payload['extra'].
No model forward, backward, optimizer step, or automatic checkpoint selection.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import torch


SCHEMA = "pusht_bc_act_checkpoint_v1"
VERSION = 1
STEP_PATTERN = re.compile(r"step_([0-9]{6})")


def _step(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 999999:
        raise ValueError("step must be an integer in [0,999999], not a bool")
    return value


def _binding(value: Any) -> dict[str, Any]:
    def visit(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                visit(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                visit(child)
            return
        raise ValueError("binding must contain finite plain JSON values and string keys")

    if type(value) is not dict or not value:
        raise ValueError("an explicit nonempty binding dictionary is required")
    visit(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _same_binding(left: Any, right: Any) -> bool:
    # Plain dict equality would silently treat False == 0 and 1 == 1.0.
    return json.dumps(_binding(left), sort_keys=True, separators=(",", ":")) == json.dumps(
        _binding(right), sort_keys=True, separators=(",", ":"))


def _safe_path(value: Any) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("parent traversal is forbidden")
    absolute = path.absolute()
    for part in (absolute, *absolute.parents):
        if part.is_symlink():
            raise ValueError(f"symlink checkpoint paths are forbidden: {part}")
    return absolute


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("state_dict hashing requires dense, nonquantized tensor entries")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def state_dict_hash(model: torch.nn.Module) -> str:
    """Hash every parameter and persistent buffer, including scalar/int tensors."""
    return _state_hash(model.state_dict())


def capture_rng_state() -> dict[str, Any]:
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "cuda_device_count": count,
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()] if count else [],
    }


def _validate_rng(state: Any) -> None:
    if type(state) is not dict or set(state) != {"python", "numpy", "torch_cpu", "cuda_device_count", "torch_cuda"}:
        raise ValueError("invalid RNG state schema")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if type(state["cuda_device_count"]) is not int or state["cuda_device_count"] != count:
        raise ValueError("CUDA device count differs from the checkpoint")
    if type(state["torch_cuda"]) is not list or len(state["torch_cuda"]) != count:
        raise ValueError("incomplete CUDA RNG state")
    tensors = [state["torch_cpu"], *state["torch_cuda"]]
    if any(not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype != torch.uint8
           or x.ndim != 1 or x.numel() == 0 for x in tensors):
        raise ValueError("RNG tensors must be nonempty CPU uint8 vectors")
    # Validate without consuming or changing the process-wide generators.
    random.Random().setstate(state["python"])
    np.random.RandomState(0).set_state(state["numpy"])
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    for index, saved in enumerate(state["torch_cuda"]):
        if saved.shape != torch.cuda.get_rng_state(index).shape:
            raise ValueError("CUDA RNG state shape differs from the current runtime")


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore global RNGs; reject changed CUDA device count instead of dropping it."""
    _validate_rng(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["cuda_device_count"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_name(optimizer: torch.optim.Optimizer) -> str:
    cls = type(optimizer)
    return f"{cls.__module__}.{cls.__qualname__}"


def _optimizer_matches_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if not optimizer_ids or len(set(optimizer_ids)) != len(optimizer_ids) or not set(optimizer_ids) <= model_ids:
        raise ValueError("optimizer parameters must be unique parameters of this model")


def _optimizer_parameter_names(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]


def _commit_directory(source: Path, destination: Path) -> None:
    """Atomic no-replace rename on the two supported project operating systems."""
    if sys.platform == "win32":
        # Windows os.rename fails if destination already exists.
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace directory commit requires renameat2")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            code = ctypes.get_errno()
            if code == errno.EEXIST:
                raise FileExistsError(destination)
            raise OSError(code, os.strerror(code), str(destination))
        return
    raise RuntimeError("atomic checkpoint commits are supported on Windows/Linux only")


def save_checkpoint(run_dir, step, model, optimizer, binding, extra=None) -> Path:
    """Commit checkpoint.pt + manifest.json; never replace final/partial outputs."""
    step = _step(step)
    binding = _binding(binding)
    if extra is not None and type(extra) is not dict:
        raise ValueError("extra must be a dictionary or None")
    _optimizer_matches_model(model, optimizer)
    base = _safe_path(run_dir) / "checkpoints"
    base = _safe_path(base)
    destination = base / f"step_{step:06d}"
    partial = base / f"step_{step:06d}.partial"
    if destination.exists() or destination.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError("checkpoint or partial already exists; preserve and inspect it")
    base.mkdir(parents=True, exist_ok=True)
    partial.mkdir(exist_ok=False)
    rng = capture_rng_state()
    state = model.state_dict()
    digest = _state_hash(state)
    payload = {"schema": SCHEMA, "version": VERSION, "step": step, "binding": binding,
               "model": state, "optimizer": optimizer.state_dict(), "rng": rng,
               "extra": dict(extra) if extra is not None else {}}
    data_path = partial / "checkpoint.pt"
    with data_path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema": SCHEMA, "version": VERSION, "step": step, "binding": binding,
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "optimizer_class": _optimizer_name(optimizer), "state_dict_sha256": digest,
        "optimizer_parameter_names": _optimizer_parameter_names(model, optimizer),
        "owner_uid": os.getuid() if hasattr(os, "getuid") else None,
        "files": {"checkpoint.pt": {"size": data_path.stat().st_size, "sha256": _file_hash(data_path)}},
    }
    with (partial / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    _commit_directory(partial, destination)
    return destination


def _manifest(path: Path, binding: dict[str, Any]) -> dict[str, Any]:
    match = STEP_PATTERN.fullmatch(path.name)
    if not match or path.parent.name != "checkpoints" or not path.is_dir():
        raise ValueError("explicit committed checkpoints/step_NNNNNN directory required")
    if {entry.name for entry in path.iterdir()} != {"manifest.json", "checkpoint.pt"}:
        raise ValueError("incomplete or unexpected checkpoint files")
    for name in ("manifest.json", "checkpoint.pt"):
        candidate = _safe_path(path / name)
        if not candidate.is_file():
            raise ValueError("checkpoint entries must be regular files")
        if hasattr(os, "getuid") and candidate.stat().st_uid != os.getuid():
            raise ValueError("only current-user-owned generated checkpoints may be loaded")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or type(manifest.get("version")) is not int or manifest["version"] != VERSION:
        raise ValueError("unsupported checkpoint version")
    if _step(manifest.get("step")) != int(match.group(1)) or not _same_binding(manifest.get("binding"), binding):
        raise ValueError("checkpoint step or exact binding differs")
    if manifest.get("torch_version") != str(torch.__version__) or manifest.get("numpy_version") != str(np.__version__):
        raise ValueError("checkpoint torch/numpy runtime changed")
    if hasattr(os, "getuid") and (manifest.get("owner_uid") != os.getuid() or path.stat().st_uid != os.getuid()):
        raise ValueError("checkpoint owner differs")
    files = manifest.get("files")
    if type(files) is not dict or set(files) != {"checkpoint.pt"}:
        raise ValueError("unexpected checkpoint manifest paths")
    spec = files["checkpoint.pt"]
    if (type(spec) is not dict or set(spec) != {"size", "sha256"} or type(spec["size"]) is not int
            or spec["size"] <= 0 or not isinstance(spec["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
        raise ValueError("invalid checkpoint file pin")
    data = path / "checkpoint.pt"
    if data.stat().st_size != spec["size"] or _file_hash(data) != spec["sha256"]:
        raise ValueError("checkpoint size/SHA256 mismatch; refusing deserialization")
    return manifest


def _move_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    def move(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(child, device) for key, child in value.items()}
        if isinstance(value, list):
            return [move(child, device) for child in value]
        if isinstance(value, tuple):
            return tuple(move(child, device) for child in value)
        return value

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if state is None:
                continue
            for key in list(state):
                # Preserve PyTorch Adam/AdamW scalar step placement: noncapturable,
                # nonfused step lives on CPU; parameter moments follow the parameter.
                device = ("cpu" if key == "step" and not group.get("capturable", False)
                          and not group.get("fused", False) else parameter.device)
                state[key] = move(state[key], device)


def load_checkpoint(path, model, optimizer, binding, device) -> dict[str, Any]:
    """Verify all file pins before loading a trusted local generated pickle.

    Any exception is a failed resume, not permission to fall back to another
    checkpoint. The caller owns model/optimizer construction and sampler state.
    """
    path = _safe_path(path)
    binding = _binding(binding)
    manifest = _manifest(path, binding)
    _optimizer_matches_model(model, optimizer)
    if manifest.get("optimizer_class") != _optimizer_name(optimizer):
        raise ValueError("optimizer class changed")
    if manifest.get("optimizer_parameter_names") != _optimizer_parameter_names(model, optimizer):
        raise ValueError("optimizer parameter order/groups changed")
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if any(value.device != target for value in (*model.parameters(), *model.buffers())):
        raise ValueError("construct model and optimizer on the requested device before loading")
    # This flag is intentional: only this module's local, user-owned generated
    # checkpoint format is accepted; optimizer + NumPy/Python RNG require pickle.
    payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=False)
    if (type(payload) is not dict or set(payload) != {"schema", "version", "step", "binding", "model", "optimizer", "rng", "extra"}
            or payload["schema"] != SCHEMA or type(payload["version"]) is not int or payload["version"] != VERSION
            or _step(payload["step"]) != manifest["step"] or not _same_binding(payload["binding"], binding)
            or type(payload["extra"]) is not dict):
        raise ValueError("checkpoint payload differs from manifest")
    expected, saved = model.state_dict(), payload["model"]
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("checkpoint model keys changed")
    for key, tensor in expected.items():
        if not isinstance(saved[key], torch.Tensor) or saved[key].shape != tensor.shape or saved[key].dtype != tensor.dtype:
            raise ValueError(f"checkpoint tensor shape/dtype differs: {key}")
    if _state_hash(saved) != manifest.get("state_dict_sha256"):
        raise ValueError("checkpoint model-state hash differs")
    saved_opt = payload["optimizer"]
    if (type(saved_opt) is not dict or set(saved_opt) != {"state", "param_groups"}
            or len(saved_opt["param_groups"]) != len(optimizer.param_groups)
            or any(len(a["params"]) != len(b["params"]) for a, b in zip(saved_opt["param_groups"], optimizer.param_groups))):
        raise ValueError("checkpoint optimizer parameter groups changed")
    _validate_rng(payload["rng"])
    model.load_state_dict(saved, strict=True)
    if state_dict_hash(model) != manifest["state_dict_sha256"]:
        raise ValueError("loaded model tensors changed")
    optimizer.load_state_dict(saved_opt)
    _move_optimizer_state(optimizer)
    restore_rng_state(payload["rng"])
    return payload
