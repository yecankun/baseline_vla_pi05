from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pi05_pretrained_loader import (
    DEFAULT_PI05_REPO,
    DEFAULT_PI05_REVISION,
    PI05PretrainedLoadRejected,
    inspect_huggingface_repo,
    load_verified_pi05_weights,
    resolve_pretrained_files,
    sha256_file,
)
from probe_pi05_lerobot_adapter import (
    ACTIVE_ACTION_DIMS,
    PI05ProbeDataset,
    build_config,
    collate_probe_batch,
    compute_dataset_stats,
    import_lerobot_dataset,
    import_pi05,
    set_seed,
)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed PI05 pretrained checkpoint probe. It can inspect repository metadata, download the "
            "pinned checkpoint, verify key/shape/parameter coverage, and run one project forward/backward pass."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pretrained-name-or-path", default=DEFAULT_PI05_REPO)
    parser.add_argument("--pretrained-revision", default=DEFAULT_PI05_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--compute-file-sha256", action="store_true")
    parser.add_argument("--min-loaded-parameter-fraction", type=float, default=0.99)
    parser.add_argument("--max-missing-keys", type=int, default=0)
    parser.add_argument("--max-unexpected-keys", type=int, default=0)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-stats-records", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-state-dim", type=int, default=32)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--tokenizer-max-length", type=int, default=200)
    parser.add_argument("--paligemma-variant", default="gemma_2b", choices=["gemma_300m", "gemma_2b"])
    parser.add_argument("--action-expert-variant", default="gemma_300m", choices=["gemma_300m", "gemma_2b"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--freeze-vision-encoder", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision-encoder", dest="freeze_vision_encoder", action="store_false")
    parser.add_argument("--train-expert-only", action="store_true", default=True)
    parser.add_argument("--no-train-expert-only", dest="train_expert_only", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    base_report: dict[str, Any] = {
        "scope": "PI05 pretrained loading feasibility only; not real-system validation",
        "pretrained_name_or_path": str(args.pretrained_name_or_path),
        "pretrained_revision": args.pretrained_revision,
        "mode": "metadata" if args.metadata_only else "download" if args.download_only else "load",
    }
    try:
        if args.metadata_only:
            base_report["repository"] = inspect_huggingface_repo(
                str(args.pretrained_name_or_path),
                args.pretrained_revision,
            )
            base_report["status"] = "metadata_verified"
            write_report(args.out, base_report)
            print(json.dumps(base_report, indent=2, ensure_ascii=False))
            return

        if args.download_only:
            base_report["resolved"] = resolve_pretrained_files(
                args.pretrained_name_or_path,
                revision=args.pretrained_revision,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            )
            if args.compute_file_sha256:
                base_report["resolved"]["model_sha256"] = sha256_file(
                    Path(base_report["resolved"]["model_path"])
                )
            base_report["status"] = "download_resolved"
            write_report(args.out, base_report)
            print(json.dumps(base_report, indent=2, ensure_ascii=False))
            return

        if args.root is None or not args.repo_id:
            raise ValueError("--root and --repo-id are required for full load/forward probing")

        set_seed(args.seed)
        LeRobotDataset = import_lerobot_dataset()
        dataset = LeRobotDataset(args.repo_id, root=args.root, download_videos=False)
        if len(dataset) == 0:
            raise ValueError(f"empty LeRobotDataset: root={args.root} repo_id={args.repo_id}")
        sample = dataset[0]
        state_dim = int(sample["observation.state"].numel())
        action_dim = int(sample["action"].numel())
        stats_indices = list(range(min(len(dataset), max(int(args.max_stats_records), 1))))
        dataset_stats = compute_dataset_stats(dataset, stats_indices)
        loader = DataLoader(
            PI05ProbeDataset(dataset, list(range(min(len(dataset), args.batch_size)))),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_probe_batch,
        )
        raw_batch = next(iter(loader))

        pi05 = import_pi05()
        config = build_config(args, pi05, state_dim, action_dim)
        preprocessor, _postprocessor = pi05["make_pi05_pre_post_processors"](config, dataset_stats)
        processed = preprocessor(raw_batch)
        policy = pi05["PI05Policy"](config).to(args.device)
        base_report["load"] = load_verified_pi05_weights(
            policy,
            args.pretrained_name_or_path,
            revision=args.pretrained_revision,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
            min_loaded_parameter_fraction=args.min_loaded_parameter_fraction,
            max_missing_keys=args.max_missing_keys,
            max_unexpected_keys=args.max_unexpected_keys,
            compute_file_sha256=args.compute_file_sha256,
        )
        policy.train()
        loss, loss_dict = policy(processed)
        base_report["forward"] = {
            "loss": float(loss.detach().cpu().item()),
            "loss_per_dim": list(loss_dict.get("loss_per_dim", [])[:ACTIVE_ACTION_DIMS]),
        }
        if args.backward:
            loss.backward()
            gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
            base_report["backward"] = {
                "ok": True,
                "gradient_tensor_count": len(gradients),
                "gradient_l2": float(
                    sum(gradient.detach().float().pow(2).sum().cpu().item() for gradient in gradients)
                    ** 0.5
                ),
            }
        base_report["status"] = "load_and_forward_verified"
    except PI05PretrainedLoadRejected as exc:
        base_report["status"] = "failed"
        base_report["error"] = repr(exc)
        base_report["load"] = exc.report
        write_report(args.out, base_report)
        raise
    except Exception as exc:
        base_report["status"] = "failed"
        base_report["error"] = repr(exc)
        write_report(args.out, base_report)
        raise

    write_report(args.out, base_report)
    print(
        f"status={base_report['status']} loaded_fraction="
        f"{base_report['load']['loaded_parameter_fraction']:.6f} "
        f"loss={base_report['forward']['loss']:.6f} wrote={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
