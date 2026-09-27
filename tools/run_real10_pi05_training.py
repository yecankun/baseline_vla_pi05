"""Prepare the real10 training pair; optimizer execution requires --execute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import torch

from export_openpi_compat_to_lerobot import read_json, write_json
from pi05_pretrained_loader import DEFAULT_PI05_REVISION, resolve_pretrained_files
from prepare_real_pi05_pack import ADAPTER_VERSION, EPISODES
from probe_pi05_lerobot_adapter import compute_dataset_stats, import_lerobot_dataset, stats_to_json
from train_pi05_lerobot_adapter import split_by_episode


def training_commands(args):
    common = ["--root", str(args.root), "--repo-id", "project2026/real10-prototype-v1",
              "--split-manifest", str(args.root / "project2026_lerobot_export_manifest.json"),
              "--train-only", "--save-checkpoint", "--seed", "123", "--log-every", "50",
              "--num-workers", "0"]
    scripts = Path(__file__).resolve().parent
    elite = [sys.executable, "-B", "-u", str(scripts / "train_pi05_lerobot_adapter.py"), *common,
             "--out", str(args.out / "elite"), "--max-steps", str(args.elite_steps), "--batch-size", "1",
             "--elite-translation-only-loss", "--state-conditioning", "--lr", "2.5e-5",
             "--pretrained-name-or-path", "lerobot/pi05_base",
             "--pretrained-revision", DEFAULT_PI05_REVISION,
             "--pretrained-cache-dir", str(args.cache_dir), "--pretrained-local-files-only"]
    piper = [sys.executable, "-B", "-u", str(scripts / "train_pi05_mixed_head_adapter.py"), *common,
             "--out", str(args.out / "piper"), "--max-steps", str(args.piper_steps), "--batch-size", "32",
             "--init-policy-checkpoint", str(args.out / "elite/final_policy.pt"),
             "--freeze-pi05", "--head-mode", "state_only", "--class-weighting", "balanced",
             "--head-feature-dim", "128", "--head-hidden-dim", "256", "--head-lr", "1e-3"]
    return {"elite": elite, "piper": piper}


def preflight(args):
    """Read converted data and weight-file availability; never construct a model."""
    manifest_path = args.root / "project2026_lerobot_export_manifest.json"
    manifest = read_json(manifest_path)
    source = read_json(args.pack / "manifest.json")
    if manifest.get("adapter_version") != ADAPTER_VERSION or source.get("source_episodes") != list(EPISODES):
        raise ValueError("expected the exact real10 observable-history adapter and source allowlist")
    dataset = import_lerobot_dataset()("project2026/real10-prototype-v1", root=args.root, download_videos=False)
    # Tabular fields are enough for split/stats checks; do not decode thousands
    # of image pairs again to obtain episode IDs or label counts.
    tabular = dataset.hf_dataset.select_columns(
        ["episode_index", "observation.state", "action", "elite_tcp_delta_6d", "piper_intent_id"]
    ).with_format("torch")
    train, val, mode = split_by_episode(tabular, val_fraction=.15, seed=123, max_records=None,
                                      split_manifest=manifest_path, train_only=True)
    if len(train) != 2719 or val or manifest["episodes"] != 10:
        raise ValueError("real10 export must contain all 2719 transitions in 10 training episodes")
    with np.load(args.pack / "openpi_arrays.npz") as arrays:
        for key, array_key in (("observation.state", "state_32"), ("action", "action_32"),
                               ("elite_tcp_delta_6d", "elite_tcp_delta_6d")):
            values = torch.stack([tabular[i][key] for i in train]).numpy()
            np.testing.assert_array_equal(values, arrays[array_key])
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {key}")
        labels = np.array([int(tabular[i]["piper_intent_id"]) for i in train])
        np.testing.assert_array_equal(labels, arrays["piper_intent_id"])
    counts = np.bincount(labels, minlength=3).tolist()
    if counts != [0, 2649, 70]:
        raise ValueError(f"unexpected real10 intent counts: {counts}")
    offset = 0
    for episode in manifest["episode_exports"]:
        row = dataset[offset]
        for key in ("observation.images.side", "observation.images.top"):
            image = row[key]
            if tuple(image.shape) != (3, 224, 224) or not torch.isfinite(image).all():
                raise ValueError(f"invalid image at episode {episode['source_episode']}: {key}")
        offset += episode["frames"]
    stats = stats_to_json(compute_dataset_stats(tabular, train))
    weights = resolve_pretrained_files("lerobot/pi05_base", revision=DEFAULT_PI05_REVISION,
                                       cache_dir=args.cache_dir, local_files_only=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "project2026_real10_training_preflight_v1", "adapter_version": ADAPTER_VERSION,
        "status": "passed", "split_mode": mode, "episodes": 10, "train_samples": len(train),
        "val_samples": 0, "piper_class_counts": counts,
        "all_converted_state_and_action_values_equal_source_pack": True,
        "representative_image_rows_read": 10, "image_shape_chw": [3, 224, 224],
        "stats_samples": len(train), "normalization_stats": stats,
        "pretrained_files": weights, "weight_tensors_loaded": False,
        "output_free_bytes": shutil.disk_usage(args.out).free,
        "optimizer_steps_executed": 0, "hardware_executed": False,
        "training_commands": training_commands(args),
        "checkpoint_policy": "fixed final step only; no validation or best checkpoint",
    }
    write_json(args.out / "preflight.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("normalization_stats", "training_commands")}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("simulation_output/real10_pi05_lerobot_v1"))
    parser.add_argument("--pack", type=Path, default=Path("simulation_output/real10_pi05_compat_v1"))
    parser.add_argument("--out", type=Path, default=Path("simulation_output/real10_pi05_train_v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--elite-steps", type=int, default=3000)
    parser.add_argument("--piper-steps", type=int, default=1000)
    parser.add_argument("--component", choices=("both", "elite", "piper"), default="both")
    parser.add_argument("--execute", action="store_true", help="Actually start the user-run training job")
    args = parser.parse_args()
    if args.elite_steps < 1 or args.piper_steps < 1:
        raise ValueError("step counts must be positive")
    # Check requested outputs before preflight, without deleting/replacing an old run.
    selected = ("elite", "piper") if args.component == "both" else (args.component,)
    if args.execute:
        for component in selected:
            if (args.out / component).exists():
                raise FileExistsError(f"preserve existing run: {args.out / component}")
        if args.component == "piper" and not (args.out / "elite/final_policy.pt").is_file():
            raise FileNotFoundError("Piper-only requires the completed real10 Elite checkpoint")
    report = preflight(args)
    for name, command in report["training_commands"].items():
        print(f"{name}: {shlex.join(command)}", flush=True)
    if not args.execute:
        print("Preparation complete. No model loaded or training started; --execute is user-run.", flush=True)
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    for component in selected:
        subprocess.run(report["training_commands"][component], check=True, env=env)


if __name__ == "__main__":
    main()
