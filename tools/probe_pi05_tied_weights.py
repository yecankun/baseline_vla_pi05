from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from pi05_pretrained_loader import DEFAULT_PI05_REPO, DEFAULT_PI05_REVISION, resolve_pretrained_files
from probe_pi05_lerobot_adapter import build_config, import_pi05


def tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "data_ptr": int(tensor.data_ptr()),
        "storage_data_ptr": int(tensor.untyped_storage().data_ptr()),
        "storage_offset": int(tensor.storage_offset()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect PI05 embedding/output-head tied-weight serialization without loading checkpoint tensors."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pretrained-name-or-path", default=DEFAULT_PI05_REPO)
    parser.add_argument("--pretrained-revision", default=DEFAULT_PI05_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    resolved = resolve_pretrained_files(
        args.pretrained_name_or_path,
        revision=args.pretrained_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    from safetensors import safe_open

    with safe_open(resolved["model_path"], framework="pt", device="cpu") as checkpoint:
        checkpoint_keys = [
            {
                "name": key,
                "shape": list(checkpoint.get_slice(key).get_shape()),
            }
            for key in checkpoint.keys()
            if "embed_tokens" in key or "lm_head" in key
        ]

    pi05 = import_pi05()
    config_args = Namespace(
        device=args.device,
        use_amp=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype=args.dtype,
        image_size=224,
        max_state_dim=32,
        max_action_dim=32,
        tokenizer_max_length=200,
        freeze_vision_encoder=True,
        train_expert_only=True,
        gradient_checkpointing=True,
    )
    config = build_config(config_args, pi05, state_dim=22, action_dim=32)
    policy = pi05["PI05Policy"](config).to(args.device)
    state = policy.state_dict(keep_vars=True)
    model_tensors = {
        key: value
        for key, value in state.items()
        if "embed_tokens" in key or "lm_head" in key
    }
    model_records = [tensor_record(key, value) for key, value in model_tensors.items()]
    shared_storage_pairs = []
    names = sorted(model_tensors)
    for index, left_name in enumerate(names):
        left = model_tensors[left_name]
        for right_name in names[index + 1 :]:
            right = model_tensors[right_name]
            if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
                shared_storage_pairs.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "same_shape": tuple(left.shape) == tuple(right.shape),
                        "same_data_ptr": left.data_ptr() == right.data_ptr(),
                    }
                )

    report = {
        "scope": "PI05 tied-weight serialization diagnostic only",
        "resolved": resolved,
        "checkpoint_keys": checkpoint_keys,
        "model_tensors": model_records,
        "shared_storage_pairs": shared_storage_pairs,
        "config_tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        "status": "inspected",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
