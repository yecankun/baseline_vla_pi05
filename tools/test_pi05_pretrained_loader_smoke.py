from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file

from pi05_pretrained_loader import (
    PI05PretrainedLoadRejected,
    initialize_pi05_policy,
    load_verified_pi05_weights,
)


class FakeConfig:
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    dtype = "float32"
    chunk_size = 1
    n_action_steps = 1
    max_state_dim = 32
    max_action_dim = 32
    image_resolution = (224, 224)
    tokenizer_max_length = 200
    freeze_vision_encoder = True
    train_expert_only = True
    gradient_checkpointing = False


class FakePolicy(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FakeConfig()
        self.model = torch.nn.Linear(4, 3)

    def _fix_pytorch_state_dict_keys(self, state, config):
        return state


class FakeTiedPolicy(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FakeConfig()
        self.model = torch.nn.Module()
        self.model.paligemma_with_expert = torch.nn.Module()
        self.model.paligemma_with_expert.paligemma = torch.nn.Module()
        paligemma = self.model.paligemma_with_expert.paligemma
        paligemma.model = torch.nn.Module()
        paligemma.model.language_model = torch.nn.Module()
        paligemma.model.language_model.embed_tokens = torch.nn.Embedding(5, 3)
        paligemma.lm_head = torch.nn.Linear(3, 5, bias=False)
        paligemma.lm_head.weight = paligemma.model.language_model.embed_tokens.weight

    def _fix_pytorch_state_dict_keys(self, state, config):
        return state


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pi05-loader-smoke-") as temp_dir:
        root = Path(temp_dir)
        checkpoint = {
            "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "bias": torch.tensor([1.0, 2.0, 3.0]),
        }
        save_file(checkpoint, root / "model.safetensors")
        (root / "config.json").write_text(
            json.dumps(
                {
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                    "max_state_dim": 32,
                    "max_action_dim": 32,
                    "image_resolution": [224, 224],
                }
            ),
            encoding="utf-8",
        )
        policy = FakePolicy()
        report = load_verified_pi05_weights(
            policy,
            root,
            revision=None,
            min_loaded_parameter_fraction=1.0,
            fingerprint_keys=2,
        )

        assert report["status"] == "loaded"
        assert report["loaded_parameter_fraction"] == 1.0
        assert report["missing_key_count"] == 0
        assert report["unexpected_key_count"] == 0
        assert len(report["fingerprints"]["random_to_loaded_changed"]) == 2
        assert len(report["fingerprints"]["loaded_matches_checkpoint"]) == 2
        assert torch.equal(policy.model.weight, checkpoint["weight"])
        assert torch.equal(policy.model.bias, checkpoint["bias"])

        tied_root = root / "tied"
        tied_root.mkdir()
        tied_weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        save_file(
            {"paligemma_with_expert.paligemma.lm_head.weight": tied_weight},
            tied_root / "model.safetensors",
        )
        (tied_root / "config.json").write_text(
            (root / "config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        tied_policy = FakeTiedPolicy()
        tied_report = load_verified_pi05_weights(
            tied_policy,
            tied_root,
            revision=None,
            min_loaded_parameter_fraction=1.0,
            fingerprint_keys=1,
        )
        tied_state = tied_policy.state_dict()
        assert tied_report["status"] == "loaded"
        assert tied_report["raw_missing_key_count"] == 1
        assert tied_report["missing_key_count"] == 0
        assert tied_report["tied_weight_alias_count"] == 1
        assert tied_report["raw_loaded_state_fraction"] == 0.5
        assert tied_report["loaded_parameter_fraction"] == 1.0
        assert tied_report["load_state_dict_missing_keys"] == []
        assert tied_report["tied_weight_alias_verification"][0]["alias_matches_checkpoint"]
        assert torch.equal(
            tied_state["model.paligemma_with_expert.paligemma.lm_head.weight"], tied_weight
        )
        assert torch.equal(
            tied_state[
                "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            ],
            tied_weight,
        )

        fail_closed_args = Namespace(
            device="cpu",
            pretrained_name_or_path=None,
            allow_random_init=False,
        )
        try:
            initialize_pi05_policy(FakePolicy, FakeConfig(), fail_closed_args)
        except ValueError as exc:
            assert "fail-closed" in str(exc)
        else:
            raise AssertionError("missing pretrained source did not fail closed")

        random_args = Namespace(
            device="cpu",
            pretrained_name_or_path=None,
            allow_random_init=True,
        )
        _random_policy, random_report = initialize_pi05_policy(
            FakePolicy,
            FakeConfig(),
            random_args,
        )
        assert random_report["status"] == "random_init_explicitly_allowed"

        rejected_root = root / "rejected"
        rejected_root.mkdir()
        save_file({"weight": checkpoint["weight"]}, rejected_root / "model.safetensors")
        (rejected_root / "config.json").write_text(
            (root / "config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        try:
            load_verified_pi05_weights(
                FakePolicy(),
                rejected_root,
                revision=None,
                min_loaded_parameter_fraction=1.0,
            )
        except PI05PretrainedLoadRejected as exc:
            assert exc.report["status"] == "rejected"
            assert exc.report["missing_key_count"] == 1
            assert exc.report["missing_keys"] == ["model.bias"]
        else:
            raise AssertionError("incomplete checkpoint was not rejected")
    print("pi05_pretrained_loader_smoke_ok")


if __name__ == "__main__":
    main()
