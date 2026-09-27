from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from eval_pi05_mixed_head_open_loop import load_mixed_head_checkpoint, verify_policy_provenance
from train_pi05_mixed_head_adapter import PiperIntentHead


def main() -> None:
    torch.manual_seed(123)
    batch_size = 4
    prefix_dim = 12
    state_dim = 5
    feature_dim = 7
    prefix = torch.randn(batch_size, 6, prefix_dim)
    alternate_prefix = torch.randn(batch_size, 6, prefix_dim)
    mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
            [True, True, False, False, False, False],
        ]
    )
    state = torch.randn(batch_size, state_dim)

    pretrained = {
        "status": "loaded",
        "resolved": {"source": "lerobot/pi05_base", "resolved_revision": "pinned"},
        "loaded_parameter_fraction": 1.0,
    }
    derived = {
        "status": "finetuned_policy_checkpoint_loaded",
        "path": "fine_tuned.pt",
        "base_pretrained": pretrained,
        "checkpoint_selection": "final_step",
        "checkpoint_step": 1000,
        "active_action_dims": 6,
        "loss_scope": "Elite TCP delta dims 0:6 only; Piper compatibility dims excluded",
    }
    mixed_checkpoint = {"policy_initialization": pretrained}
    provenance = verify_policy_provenance(derived, mixed_checkpoint)
    assert provenance["matched"]
    assert provenance["resolved_revision"] == "pinned"
    mismatched = {**derived, "base_pretrained": {**pretrained, "resolved": {**pretrained["resolved"], "resolved_revision": "wrong"}}}
    try:
        verify_policy_provenance(mismatched, mixed_checkpoint)
    except ValueError as exc:
        assert "resolved_revision mismatch" in str(exc)
    else:
        raise AssertionError("mismatched fine-tuned provenance was not rejected")

    state_only = PiperIntentHead(
        prefix_dim,
        state_dim,
        hidden_dim=9,
        dropout=0.0,
        mode="state_only",
        feature_dim=feature_dim,
    )
    state_logits = state_only(None, None, state)
    assert tuple(state_logits.shape) == (batch_size, 3)
    assert state_only.feature_contract()["classifier_input_dim"] == feature_dim
    assert not state_only.requires_prefix

    prefix_state = PiperIntentHead(
        prefix_dim,
        state_dim,
        hidden_dim=9,
        dropout=0.0,
        mode="prefix_state",
        feature_dim=feature_dim,
    )
    prefix_state.eval()
    logits = prefix_state(prefix, mask, state)
    alternate_logits = prefix_state(alternate_prefix, mask, state)
    assert tuple(logits.shape) == (batch_size, 3)
    assert prefix_state.feature_contract()["classifier_input_dim"] == 2 * feature_dim
    assert prefix_state.feature_contract()["dimension_balanced"]
    assert not torch.equal(logits, alternate_logits)

    loss = logits.square().mean()
    loss.backward()
    assert prefix_state.prefix_encoder[1].weight.grad is not None
    assert prefix_state.state_encoder[1].weight.grad is not None

    with tempfile.TemporaryDirectory(prefix="pi05-head-mode-smoke-") as temp_dir:
        checkpoint_path = Path(temp_dir) / "mixed_head_policy.pt"
        torch.save(
            {
                "piper_head_state": prefix_state.state_dict(),
                "args": {
                    "head_mode": "prefix_state",
                    "head_feature_dim": feature_dim,
                    "head_hidden_dim": 9,
                    "head_dropout": 0.0,
                },
                "head_mode": "prefix_state",
                "head_feature_dim": feature_dim,
                "prefix_dim": prefix_dim,
                "state_dim": state_dim,
            },
            checkpoint_path,
        )
        restored, _checkpoint = load_mixed_head_checkpoint(
            checkpoint_path,
            prefix_dim=prefix_dim,
            state_dim=state_dim,
            device="cpu",
        )
        restored.eval()
        assert restored.feature_contract() == prefix_state.feature_contract()
        assert torch.equal(restored(prefix, mask, state), logits)

        legacy = PiperIntentHead(
            prefix_dim,
            state_dim,
            hidden_dim=9,
            dropout=0.0,
            mode="legacy_concat",
        )
        legacy_path = Path(temp_dir) / "legacy_mixed_head_policy.pt"
        torch.save(
            {
                "piper_head_state": legacy.state_dict(),
                "args": {"head_hidden_dim": 9, "head_dropout": 0.0},
                "prefix_dim": prefix_dim,
                "state_dim": state_dim,
            },
            legacy_path,
        )
        restored_legacy, _legacy_checkpoint = load_mixed_head_checkpoint(
            legacy_path,
            prefix_dim=prefix_dim,
            state_dim=state_dim,
            device="cpu",
        )
        assert restored_legacy.mode == "legacy_concat"
        assert restored_legacy.load_state_dict(legacy.state_dict(), strict=True) is not None

    print("pi05_mixed_head_modes_smoke_ok")


if __name__ == "__main__":
    main()
