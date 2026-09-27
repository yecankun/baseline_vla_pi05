from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi05_action_effect_dataset import (
    ActionEffectFeaturePack,
    ActionEffectWindowDataset,
    FEATURE_PACK_SCHEMA,
    SPLIT_SCHEMA,
    STATE_32_LAYOUT,
    build_temporal_windows,
    compute_train_normalization,
    make_episode_stratified_real_folds,
    normalize_active_action_candidates,
    normalize_state_32_tensor,
    source_episode_balanced_weights,
)
from pi05_action_effect_world_model import (
    ActionEffectWorldModel,
    ActionEffectWorldModelConfig,
    OxidationCoverageAugmenter,
    action_effect_world_model_loss,
    action32_candidates_to_active,
    decode_active_action,
    encode_active_action,
    extract_pi05_multiview_visual_latent,
    paired_degradation_consistency_loss,
    score_action_candidates,
    validate_policy_batch_no_privileged_fields,
)


class _FakePaliGemma(torch.nn.Module):
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(image, (2, 2))
        return pooled.flatten(2).transpose(1, 2)


class _FakePI05Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.paligemma_with_expert = _FakePaliGemma()


class _FakePI05Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakePI05Model()


def _write_synthetic_pack(root: Path, visual_dim: int) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    records_per_episode = 9
    episodes = 6
    count = records_per_episode * episodes
    rng = np.random.default_rng(123)
    episode_index = np.repeat(np.arange(episodes, dtype=np.int64), records_per_episode)
    frame_index = np.tile(np.arange(records_per_episode, dtype=np.int64), episodes)
    task_id = np.repeat(np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64), records_per_episode)
    domain_id = np.repeat(np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64), records_per_episode)
    elite = rng.normal(0.0, 0.05, size=(count, 6)).astype(np.float32)
    elite[:, 3:6] = 0.0
    piper = np.where(frame_index % 3 == 0, 2, 1).astype(np.int64)
    state = rng.normal(0.0, 1.0, size=(count, 32)).astype(np.float32)
    visual = rng.normal(0.0, 1.0, size=(count, 2, visual_dim)).astype(np.float32)
    degraded_visual = (visual + 0.05 * rng.normal(size=visual.shape)).astype(np.float32)
    visual_valid = np.ones((count, 2), dtype=np.bool_)
    degradation_pair_valid = domain_id == 0
    state_valid = np.ones((count, 32), dtype=np.bool_)
    state_valid[0, 10] = False
    guidance = np.where(task_id == 0, 1, 2).astype(np.int64)
    branch = task_id.copy()
    invalid_feed = np.zeros(count, dtype=np.int64)
    arrays_path = root / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        visual_latent=visual,
        degraded_visual_latent=degraded_visual,
        degradation_pair_valid=degradation_pair_valid,
        visual_valid=visual_valid,
        state_valid_32=state_valid,
        state_32=state,
        elite_tcp_delta_6d=elite,
        piper_intent_id=piper,
        task_id=task_id,
        episode_index=episode_index,
        frame_index=frame_index,
        domain_id=domain_id,
        guidance_effect_id=guidance,
        branch_outcome_id=branch,
        invalid_feed_flag=invalid_feed,
    )
    arrays_sha256 = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    manifest = {
        "schema": FEATURE_PACK_SCHEMA,
        "counts": {"records": count, "episodes": episodes},
        "outputs": {"arrays": arrays_path.name, "arrays_sha256": arrays_sha256},
        "contract": {
            "action_interface": "elite_tcp_delta_6d + piper_intent_id",
            "exact_truth_in_policy_input": False,
            "senior_historical_real_data_allowed": False,
            "domain_id": {"sim": 0, "real_current": 1},
            "state_32_layout": STATE_32_LAYOUT,
            "visual_latent_source": {
                "extractor": "pi05_multiview_mean_patch_v1",
                "backbone_frozen": True,
                "pretrained_name_or_path": "synthetic/pi05-smoke",
                "pretrained_revision": "synthetic-smoke-pinned-revision",
            },
        },
        "episode_provenance": [
            {
                "episode_index": episode,
                "episode_instance_id": f"synthetic_episode_{episode}",
                "source_trajectory_id": f"synthetic_source_{episode}",
                "scenario_family_id": f"synthetic_family_{episode}",
                "task": "left" if episode % 2 == 0 else "right",
                "domain": "sim" if episode < 4 else "real_current",
            }
            for episode in range(episodes)
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    split = {
        "schema": SPLIT_SCHEMA,
        "feature_pack_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_episode_indices": [0, 1, 4, 5],
        "validation_episode_indices": [2, 3],
    }
    split_path = root / "episode_split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return root, split_path


def _assert_close(left: float, right: float, tolerance: float = 1e-8) -> None:
    if abs(left - right) > tolerance:
        raise AssertionError(f"{left} != {right}")


def run_smoke(work_dir: Path) -> dict[str, Any]:
    torch.manual_seed(123)
    visual_dim = 32
    pack_root, split_path = _write_synthetic_pack(work_dir / "synthetic_pack", visual_dim)
    pack = ActionEffectFeaturePack(pack_root)
    split_value = json.loads(split_path.read_text(encoding="utf-8"))
    pack.validate_episode_split(split_value)
    original_family = pack.episode_provenance[2]["scenario_family_id"]
    pack.episode_provenance[2]["scenario_family_id"] = pack.episode_provenance[0][
        "scenario_family_id"
    ]
    family_leakage_rejected = False
    try:
        pack.validate_episode_split(split_value)
    except ValueError as exc:
        family_leakage_rejected = "leakage" in str(exc)
    finally:
        pack.episode_provenance[2]["scenario_family_id"] = original_family
    if not family_leakage_rejected:
        raise AssertionError("scenario-family leakage did not fail closed")
    windows = build_temporal_windows(
        pack.arrays["episode_index"],
        pack.arrays["frame_index"],
        pack.arrays["task_id"],
        pack.arrays["domain_id"],
        context_len=4,
        horizon=3,
    )
    if len(windows) != 18:
        raise AssertionError(f"unexpected temporal-window count: {len(windows)}")
    for window in windows:
        all_indices = window.history_indices + window.action_indices + window.target_indices
        episodes = {int(pack.arrays["episode_index"][index]) for index in all_indices}
        if episodes != {window.episode_index}:
            raise AssertionError("temporal window crossed an episode boundary")

    train_episode_set = {int(value) for value in split_value["train_episode_indices"]}
    normalization = compute_train_normalization(pack, train_episode_set)
    if int(normalization["record_count"][0]) != 36:
        raise AssertionError("normalization did not use training episodes only")
    normalized_dataset = ActionEffectWindowDataset(pack, windows, normalization)
    normalized_sample = normalized_dataset[0]
    if normalized_sample["history_state_valid"][0, 10]:
        raise AssertionError("state validity mask was not preserved")
    if float(normalized_sample["history_state"][0, 10]) != 0.0:
        raise AssertionError("missing normalized state did not use neutral zero imputation")

    dataset = ActionEffectWindowDataset(pack, windows)
    sample = dataset[0]
    if not torch.allclose(
        normalize_active_action_candidates(sample["candidate_actions"], normalization),
        normalized_sample["candidate_actions"],
    ):
        raise AssertionError("inference candidate normalization differs from dataset normalization")
    if not torch.allclose(
        normalize_state_32_tensor(
            sample["history_state"], sample["history_state_valid"], normalization
        ),
        normalized_sample["history_state"],
    ):
        raise AssertionError("inference state normalization differs from dataset normalization")
    decoded = decode_active_action(sample["candidate_actions"])
    if not torch.allclose(
        decoded["elite_tcp_delta_6d"],
        torch.from_numpy(pack.arrays["elite_tcp_delta_6d"][[3, 4, 5]]).float().unsqueeze(0),
    ):
        raise AssertionError("explicit Elite action did not survive active-action round trip")
    if decoded["piper_intent_id"].shape != (1, 3):
        raise AssertionError("Piper intent round-trip shape changed")
    action_32 = torch.zeros(1, 2, 3, 32)
    action_32[..., :6] = torch.randn(1, 2, 3, 6)
    action_32[:, 0, :, 7] = 1.0
    action_32[:, 1, :, 8] = 1.0
    active_from_compat = action32_candidates_to_active(action_32)
    if tuple(active_from_compat.shape) != (1, 2, 3, 9):
        raise AssertionError("PI05 action_32 candidate conversion changed the active interface")
    fake_images = [torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)]
    extracted = extract_pi05_multiview_visual_latent(_FakePI05Policy(), fake_images)
    if tuple(extracted.shape) != (2, 2, 3):
        raise AssertionError("PI05 multiview latent extraction interface changed")

    config = ActionEffectWorldModelConfig(
        visual_dim=visual_dim,
        hidden_dim=64,
        context_len=4,
        horizon=3,
        dropout=0.0,
    )
    model = ActionEffectWorldModel(config)
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    outputs = model(
        history_visual_latent=batch["history_visual_latent"],
        history_state=batch["history_state"],
        history_visual_valid=batch["history_visual_valid"],
        history_state_valid=batch["history_state_valid"],
        task_id=batch["task_id"],
        candidate_actions=batch["candidate_actions"],
    )
    degraded_outputs = model(
        history_visual_latent=batch["history_degraded_visual_latent"],
        history_state=batch["history_state"],
        history_visual_valid=batch["history_visual_valid"],
        history_state_valid=batch["history_state_valid"],
        task_id=batch["task_id"],
        candidate_actions=batch["candidate_actions"],
    )
    degradation_consistency = paired_degradation_consistency_loss(
        outputs,
        degraded_outputs,
        batch["degradation_pair_valid"],
    )
    if not torch.isfinite(degradation_consistency) or degradation_consistency <= 0:
        raise AssertionError("paired degradation consistency loss is not finite and positive")
    expected_shapes = {
        "pred_next_visual_latent": (1, 1, 3, visual_dim),
        "pred_state_delta": (1, 1, 3, 32),
        "guidance_logits": (1, 1, 3, 3),
        "branch_logits": (1, 1, 3, 2),
        "invalid_feed_logits": (1, 1, 3),
        "effect_log_variance": (1, 1, 3),
    }
    for key, expected in expected_shapes.items():
        if tuple(outputs[key].shape) != expected:
            raise AssertionError(f"{key} shape {tuple(outputs[key].shape)} != {expected}")
    loss, components = action_effect_world_model_loss(outputs, batch)
    if not torch.isfinite(loss):
        raise AssertionError("world-model loss is not finite")
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.norm().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
        raise AssertionError("world-model backward did not produce finite nonzero gradients")

    elite = torch.zeros(1, 3, 3, 6)
    elite[:, 1, :, 0] = 0.1
    elite[:, 2, :, 1] = 0.1
    piper = torch.ones(1, 3, 3, dtype=torch.long)
    candidates = encode_active_action(elite, piper)
    crafted = {
        "branch_logits": torch.tensor(
            [[[[4.0, 0.0]] * 3, [[5.0, 0.0]] * 3, [[0.0, 5.0]] * 3]]
        ),
        "guidance_logits": torch.tensor(
            [[[[5.0, 0.0, 0.0]] * 3, [[0.0, 5.0, 0.0]] * 3, [[0.0, 0.0, 5.0]] * 3]]
        ),
        "effect_log_variance": torch.zeros(1, 3, 3),
        "invalid_feed_logits": torch.full((1, 3, 3), -5.0),
    }
    scored = score_action_candidates(
        crafted,
        task_id=torch.tensor([0]),
        candidate_actions=candidates,
        candidate_policy_eligible=torch.tensor([[False, True, True]]),
    )
    if int(scored["selected_index"].item()) != 1:
        raise AssertionError("left task did not select the eligible left-guidance candidate")
    if not torch.isneginf(scored["score"][0, 0]):
        raise AssertionError("diagnostic no-guidance candidate was policy-eligible")

    weights = source_episode_balanced_weights(
        windows,
        real_draw_fraction=0.25,
        episode_source=pack.episode_source_map(),
    )
    sim_mass = float(
        weights[
            torch.tensor([window.domain_id == 0 for window in windows], dtype=torch.bool)
        ].sum()
    )
    real_mass = float(
        weights[
            torch.tensor([window.domain_id == 1 for window in windows], dtype=torch.bool)
        ].sum()
    )
    _assert_close(sim_mass, 0.75)
    _assert_close(real_mass, 0.25)

    fold_rows = [
        {"episode_index": index, "task": "left" if index < 5 else "right", "domain": "real_current"}
        for index in range(10)
    ]
    folds = make_episode_stratified_real_folds(fold_rows, folds=5)
    if any(
        len(fold["validation_left"]) != 1 or len(fold["validation_right"]) != 1
        for fold in folds
    ):
        raise AssertionError("five-fold real split is not one-left/one-right per fold")

    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(1, 3, 32, 32)
    augmenter = OxidationCoverageAugmenter()
    degraded_a, degradation_metadata = augmenter(
        image, generator=torch.Generator().manual_seed(7)
    )
    degraded_b, _ = augmenter(image, generator=torch.Generator().manual_seed(7))
    if not torch.equal(degraded_a, degraded_b):
        raise AssertionError("coverage degradation is not deterministic under a fixed seed")
    if torch.equal(image, degraded_a) or degraded_a.shape != image.shape:
        raise AssertionError("coverage degradation did not change pixels or changed geometry")

    safe_batch = {
        "observation": {"state_32": torch.zeros(32)},
        "candidate": {"elite_tcp_delta_6d": torch.zeros(6), "piper_intent_id": 1},
    }
    validate_policy_batch_no_privileged_fields(safe_batch)
    forbidden_rejected = False
    try:
        validate_policy_batch_no_privileged_fields(
            {"observation": {"piper_event_id": "offline-id"}}
        )
    except ValueError:
        forbidden_rejected = True
    if not forbidden_rejected:
        raise AssertionError("raw event id did not fail closed at policy boundary")

    dry_run_out = work_dir / "trainer_dry_run"
    trainer = Path(__file__).with_name("train_pi05_action_effect_world_model.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--feature-pack",
            str(pack_root),
            "--split-manifest",
            str(split_path),
            "--out",
            str(dry_run_out),
            "--context-len",
            "4",
            "--horizon",
            "3",
            "--hidden-dim",
            "64",
            "--batch-size",
            "2",
            "--max-steps",
            "3",
            "--device",
            "cpu",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dry_summary = json.loads((dry_run_out / "summary.json").read_text(encoding="utf-8"))
    if dry_summary["training_started"] is not False:
        raise AssertionError("dry-run unexpectedly started training")
    if "degradation_consistency" not in dry_summary["initial_batch_components"]:
        raise AssertionError("trainer dry-run did not expose degradation consistency")

    return {
        "status": "passed",
        "model_version": model.metadata()["version"],
        "feature_pack_schema": FEATURE_PACK_SCHEMA,
        "synthetic_records": int(pack.arrays["state_32"].shape[0]),
        "synthetic_episodes": int(len(np.unique(pack.arrays["episode_index"]))),
        "temporal_windows": len(windows),
        "episode_boundary_check": "passed",
        "source_family_split_leakage_rejected": family_leakage_rejected,
        "active_action_round_trip": "passed",
        "pi05_action32_candidate_adapter": "passed",
        "pi05_multiview_latent_extractor": "passed",
        "output_shapes": {key: list(value) for key, value in expected_shapes.items()},
        "initial_loss": float(loss.item()),
        "loss_components": {key: float(value.item()) for key, value in components.items()},
        "gradient_norm_sum": gradient_norm,
        "candidate_reranking": {
            "selected_index": int(scored["selected_index"].item()),
            "no_guidance_diagnostic_candidate_ineligible": True,
        },
        "sampler_mass": {"sim": sim_mass, "real_current": real_mass},
        "train_only_normalization": {
            "status": "passed",
            "records": int(normalization["record_count"][0]),
            "missing_state_imputation": 0.0,
        },
        "five_fold_real_split": folds,
        "oxidation_coverage_transform": degradation_metadata,
        "paired_degradation_consistency": {
            "status": "passed",
            "loss": float(degradation_consistency.item()),
            "trainer_component_present": True,
        },
        "raw_event_id_rejected": forbidden_rejected,
        "exact_truth_policy_input_allowed": False,
        "senior_historical_real_data_allowed": False,
        "trainer_dry_run": {
            "status": "passed",
            "stdout": completed.stdout.strip(),
            "training_started": dry_summary["training_started"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the PI05 action-effect world-model stack.")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.work_dir is None:
        with tempfile.TemporaryDirectory(prefix="project2026_action_effect_smoke_") as value:
            report = run_smoke(Path(value))
    else:
        report = run_smoke(args.work_dir)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        f"status=passed windows={report['temporal_windows']} "
        f"selected_candidate={report['candidate_reranking']['selected_index']} "
        f"trainer_dry_run={report['trainer_dry_run']['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
