"""Bounded synthetic CPU gradient/AdamW/checkpoint tests; no real feature pack."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig
from pi05_libero_world_model_training import (
    HASH_KEYS, OPTIMIZER_CONFIG, build_optimizer, train_step,
    save_final_checkpoint, load_final_checkpoint,
)
from smoke_pi05_libero_world_model import parameter_sha


REGISTRY = [{"task_id": 9, "source_task_index": 39,
             "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]


class UnsupportedCheckpointObject:
    """Benign custom class that restricted checkpoint deserialization must reject."""
    pass


def fixture():
    # Isolated initialization RNG; no public data and no protocol initialization claim.
    with torch.random.fork_rng():
        torch.manual_seed(371)
        config = LiberoWorldModelConfig(visual_dim=3, hidden_dim=5, context_len=2, horizon=2)
        model = LiberoWorldModel(config, REGISTRY)
    generator = torch.Generator().manual_seed(372)
    inputs = {
        "history_visual_latent": torch.randn(2, 2, 2, 3, generator=generator),
        "history_visual_valid": torch.ones(2, 2, 2, dtype=torch.bool),
        "history_state": torch.randn(2, 2, 8, generator=generator),
        "history_state_valid": torch.ones(2, 2, 8, dtype=torch.bool),
        "task_instruction": [REGISTRY[0]["task_instruction"]] * 2,
        "candidate_actions": torch.randn(2, 1, 2, 7, generator=generator),
    }
    targets = {
        "future_visual_latent": torch.randn(2, 2, 2, 3, generator=generator),
        "future_visual_valid": torch.ones(2, 2, 2, dtype=torch.bool),
        "state_delta": torch.randn(2, 2, 8, generator=generator),
        "state_target_valid": torch.ones(2, 2, 8, dtype=torch.bool),
    }
    return model, inputs, targets


def checkpoint_fixture():
    model, inputs, targets = fixture()
    initial_sha = parameter_sha(model)
    optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
    train_step(model, optimizer, inputs, targets, expected_step=1)
    # Synthetic serialization fixture only: exercise final-counter guards without
    # claiming or performing 200 training updates on a real or synthetic dataset.
    for state in optimizer.state.values():
        state["step"].fill_(200)
    optimizer.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    metadata = {key: "a" * 64 for key in HASH_KEYS}
    metadata.update(model_config=asdict(model.config), task_registry=deepcopy(REGISTRY),
                    step=200, checkpoint_kind="final_diagnostic_no_resume",
                    initial_parameter_sha256=initial_sha, final_parameter_sha256=parameter_sha(model))
    return model, optimizer, metadata, inputs


class TrainingPrimitiveTests(unittest.TestCase):
    def test_fixed_optimizer_nine_fields_and_fresh_state(self):
        model, _, _ = fixture()
        config = deepcopy(OPTIMIZER_CONFIG)
        optimizer = build_optimizer(model, config)
        self.assertIs(type(optimizer), torch.optim.AdamW)
        self.assertFalse(optimizer.state)
        self.assertEqual(config, OPTIMIZER_CONFIG)
        self.assertIs(optimizer.param_groups[0]["foreach"], False)
        self.assertIs(optimizer.param_groups[0]["fused"], False)

    def test_optimizer_config_missing_extra_and_changed_rejected(self):
        for change in ({"lr": 0.0}, {"fused": True}, {"weight_decay": 0.01}, {"name": "SGD"}, {"new": 1}, {"amsgrad": 0}):
            model, _, _ = fixture()
            config = deepcopy(OPTIMIZER_CONFIG)
            config.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_optimizer(model, config)
        config = deepcopy(OPTIMIZER_CONFIG)
        del config["eps"]
        with self.assertRaises(ValueError):
            build_optimizer(fixture()[0], config)

    def test_finite_backward_update_counts_and_clipping(self):
        model, inputs, targets = fixture()
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        before = parameter_sha(model)
        result = train_step(model, optimizer, inputs, targets, expected_step=1)
        self.assertNotEqual(before, parameter_sha(model))
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["visual_count"], 24)
        self.assertEqual(result["state_count"], 32)
        self.assertGreater(result["global_grad_norm_before_clip"], 0)
        self.assertGreater(result["global_grad_norm_after_clip"], 0)
        self.assertLessEqual(result["global_grad_norm_after_clip"], 1.000001)
        self.assertEqual(result["parameters_with_grad"], len(list(model.parameters())))
        self.assertTrue(result["parameter_changed"])
        self.assertTrue(all(value.grad is None for value in inputs.values() if isinstance(value, torch.Tensor)))
        self.assertTrue(all(value.grad is None for value in targets.values()))
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer.state.values()))
        train_step(model, optimizer, inputs, targets, expected_step=2)
        self.assertTrue(all(state["step"].item() == 2 for state in optimizer.state.values()))

    def test_zero_grad_clears_prior_gradients_not_accumulation(self):
        model, inputs, targets = fixture()
        other = fixture()[0]
        other.load_state_dict(model.state_dict(), strict=True)
        for value in model.parameters():
            value.grad = torch.full_like(value, 12345)
        first = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        second = build_optimizer(other, deepcopy(OPTIMIZER_CONFIG))
        train_step(model, first, inputs, targets, expected_step=1)
        train_step(other, second, inputs, targets, expected_step=1)
        self.assertEqual(parameter_sha(model), parameter_sha(other))

    def test_nonfinite_loss_rejected_before_optimizer(self):
        model, inputs, targets = fixture()
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        targets["state_delta"][0, 0, 0] = float("nan")
        before = parameter_sha(model)
        with self.assertRaises(ValueError):
            train_step(model, optimizer, inputs, targets, expected_step=1)
        self.assertEqual(before, parameter_sha(model))
        self.assertFalse(optimizer.state)

    def test_missing_gradient_rejected(self):
        model, inputs, targets = fixture()
        model.register_parameter("unused_parameter", torch.nn.Parameter(torch.ones(2)))
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        with self.assertRaisesRegex(ValueError, "missing or nonfinite gradient"):
            train_step(model, optimizer, inputs, targets, expected_step=1)
        self.assertFalse(optimizer.state)

    def test_nonfinite_gradient_rejected(self):
        model, inputs, targets = fixture()
        next(model.parameters()).register_hook(lambda grad: torch.full_like(grad, float("nan")))
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        with self.assertRaisesRegex(ValueError, "gradient"):
            train_step(model, optimizer, inputs, targets, expected_step=1)
        self.assertFalse(optimizer.state)

    def test_zero_global_gradient_rejected(self):
        model, inputs, targets = fixture()
        for value in model.parameters():
            value.register_hook(torch.zeros_like)
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        with self.assertRaisesRegex(ValueError, "zero global gradient"):
            train_step(model, optimizer, inputs, targets, expected_step=1)
        self.assertFalse(optimizer.state)

    def test_optimizer_counter_skip_or_noop_rejected(self):
        model, inputs, targets = fixture()
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        with self.assertRaises(ValueError):
            train_step(model, optimizer, inputs, targets, expected_step=2)
        with patch.object(optimizer, "step", return_value=None), self.assertRaises(ValueError):
            train_step(model, optimizer, inputs, targets, expected_step=1)
        train_step(model, optimizer, inputs, targets, expected_step=1)
        next(iter(optimizer.state.values()))["step"].fill_(3)
        with self.assertRaisesRegex(ValueError, "counter"):
            train_step(model, optimizer, inputs, targets, expected_step=2)

    def test_missing_targets_and_target_model_leakage_rejected(self):
        model, inputs, targets = fixture()
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        bad_targets = deepcopy(targets)
        del bad_targets["state_delta"]
        with self.assertRaises(ValueError):
            train_step(model, optimizer, inputs, bad_targets, expected_step=1)
        bad_inputs = deepcopy(inputs)
        bad_inputs["targets"] = targets
        with self.assertRaises(ValueError):
            train_step(model, optimizer, bad_inputs, targets, expected_step=1)
        captured = []
        hook = model.register_forward_pre_hook(lambda module, args, kwargs: captured.append(set(kwargs)), with_kwargs=True)
        train_step(model, optimizer, inputs, targets, expected_step=1)
        hook.remove()
        self.assertEqual(captured, [set(inputs)])

    def test_masked_poison_equal_updates_and_no_input_gradients(self):
        model, inputs, targets = fixture()
        other = fixture()[0]
        other.load_state_dict(model.state_dict(), strict=True)
        inputs["history_visual_valid"][0, 1, 1] = False
        inputs["history_state_valid"][1, 0, 3] = False
        targets["future_visual_valid"][1, 0, 0] = False
        targets["state_target_valid"][0, 1, 2] = False
        clean_inputs, clean_targets = deepcopy(inputs), deepcopy(targets)
        for batch, key, index in ((inputs, "history_visual_latent", (0, 1, 1)),
                                   (inputs, "history_state", (1, 0, 3)),
                                   (targets, "future_visual_latent", (1, 0, 0)),
                                   (targets, "state_delta", (0, 1, 2))):
            batch[key][index] = float("nan")
        first = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        second = build_optimizer(other, deepcopy(OPTIMIZER_CONFIG))
        train_step(model, first, inputs, targets, expected_step=1)
        train_step(other, second, clean_inputs, clean_targets, expected_step=1)
        self.assertEqual(parameter_sha(model), parameter_sha(other))
        self.assertTrue(all(value.grad is None for value in inputs.values() if isinstance(value, torch.Tensor)))

    def test_input_target_gradients_and_inference_tensors_rejected(self):
        for owner, key, use_inference in (("inputs", "history_state", False), ("targets", "state_delta", False),
                                          ("inputs", "candidate_actions", True), ("targets", "future_visual_valid", True)):
            model, inputs, targets = fixture()
            optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
            batch = inputs if owner == "inputs" else targets
            if use_inference:
                with torch.inference_mode():
                    batch[key] = batch[key].clone()
            else:
                batch[key].requires_grad_(True)
            with self.subTest(owner=owner, key=key), self.assertRaisesRegex(ValueError, "ordinary detached"):
                train_step(model, optimizer, inputs, targets, expected_step=1)

    def test_mode_freeze_no_grad_budget_and_clip_guards(self):
        for case in ("eval", "freeze", "no_grad", "budget", "clip"):
            model, inputs, targets = fixture()
            optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
            if case == "eval":
                model.eval()
            elif case == "freeze":
                next(model.parameters()).requires_grad_(False)
            with self.subTest(case=case), self.assertRaises(ValueError):
                if case == "no_grad":
                    with torch.no_grad():
                        train_step(model, optimizer, inputs, targets, expected_step=1)
                else:
                    train_step(model, optimizer, inputs, targets, expected_step=201 if case == "budget" else 1,
                               clip_norm=2 if case == "clip" else 1)

    def test_wrong_optimizer_ownership_and_changed_lr_rejected(self):
        for change in ("lr", "ownership"):
            model, inputs, targets = fixture()
            optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
            if change == "lr":
                optimizer.param_groups[0]["lr"] = 0
            else:
                optimizer.param_groups[0]["params"] = list(reversed(optimizer.param_groups[0]["params"]))
            with self.subTest(change=change), self.assertRaises(ValueError):
                train_step(model, optimizer, inputs, targets, expected_step=1)

    def test_post_update_nonfinite_parameter_or_optimizer_rejected(self):
        for poison in ("parameter", "optimizer"):
            model, inputs, targets = fixture()
            optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
            original = optimizer.step
            def bad_step():
                original()
                with torch.no_grad():
                    if poison == "parameter":
                        next(model.parameters()).fill_(float("inf"))
                    else:
                        next(iter(optimizer.state.values()))["exp_avg"].fill_(float("inf"))
            with self.subTest(poison=poison), patch.object(optimizer, "step", side_effect=bad_step), self.assertRaises(ValueError):
                train_step(model, optimizer, inputs, targets, expected_step=1)

    def test_counter_increments_without_parameter_change_rejected(self):
        model, inputs, targets = fixture()
        optimizer = build_optimizer(model, deepcopy(OPTIMIZER_CONFIG))
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        original = optimizer.step
        def revert_values():
            original()
            model.load_state_dict(before, strict=True)
        with patch.object(optimizer, "step", side_effect=revert_values), self.assertRaisesRegex(ValueError, "summaries unchanged"):
            train_step(model, optimizer, inputs, targets, expected_step=1)


class FinalCheckpointTests(unittest.TestCase):
    def test_save_reload_exact_predictions_and_counter200(self):
        model, optimizer, metadata, inputs = checkpoint_fixture()
        fresh = fixture()[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            digest = save_final_checkpoint(path, model, optimizer, metadata)
            payload = load_final_checkpoint(path, fresh, metadata, digest)
            self.assertEqual(parameter_sha(fresh), metadata["final_parameter_sha256"])
            self.assertTrue(all(state["step"].item() == 200 for state in payload["optimizer_state_dict"]["state"].values()))
            fresh.eval().requires_grad_(False)
            with torch.inference_mode():
                first, second = model(**inputs), fresh(**inputs)
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_existing_checkpoint_not_overwritten(self):
        model, optimizer, metadata, _ = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            digest = save_final_checkpoint(path, model, optimizer, metadata)
            with self.assertRaises(FileExistsError):
                save_final_checkpoint(path, model, optimizer, metadata)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_only_final200_complete_metadata_and_final_sha(self):
        for change in ({"step": 199}, {"checkpoint_kind": "resume"}, {"extra": 1},
                       {"final_parameter_sha256": "b" * 64}, {"protocol_sha256": "not a hash"}):
            model, optimizer, metadata, _ = checkpoint_fixture()
            metadata.update(change)
            with tempfile.TemporaryDirectory() as directory, self.subTest(change=change), self.assertRaises(ValueError):
                save_final_checkpoint(Path(directory) / "final.pt", model, optimizer, metadata)

    def test_incomplete_or_nonfinite_optimizer_checkpoint_rejected(self):
        for change in ("counter", "moment", "missing"):
            model, optimizer, metadata, _ = checkpoint_fixture()
            state = next(iter(optimizer.state.values()))
            if change == "counter":
                state["step"].fill_(199)
            elif change == "moment":
                state["exp_avg_sq"].fill_(float("nan"))
            else:
                state.pop("exp_avg")
            with tempfile.TemporaryDirectory() as directory, self.subTest(change=change), self.assertRaises(ValueError):
                save_final_checkpoint(Path(directory) / "final.pt", model, optimizer, metadata)

    def test_file_hash_tamper_rejected_before_torch_load(self):
        model, optimizer, metadata, _ = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            digest = save_final_checkpoint(path, model, optimizer, metadata)
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with patch("torch.load", side_effect=AssertionError("must not deserialize")), self.assertRaisesRegex(ValueError, "SHA256"):
                load_final_checkpoint(path, fixture()[0], metadata, digest)

    def test_wrong_expected_metadata_rejected_without_parameter_change(self):
        model, optimizer, metadata, _ = checkpoint_fixture()
        fresh = fixture()[0]
        before = parameter_sha(fresh)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            digest = save_final_checkpoint(path, model, optimizer, metadata)
            changed = deepcopy(metadata)
            changed["authorization_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_final_checkpoint(path, fresh, changed, digest)
        self.assertEqual(parameter_sha(fresh), before)

    def test_rehashed_bad_parameter_and_counter_payload_rejected(self):
        for change in ("parameter", "counter", "schema"):
            model, optimizer, metadata, _ = checkpoint_fixture()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "final.pt"
                save_final_checkpoint(path, model, optimizer, metadata)
                payload = torch.load(path, weights_only=True)
                if change == "parameter":
                    next(iter(payload["model_state_dict"].values())).fill_(float("nan"))
                elif change == "counter":
                    next(iter(payload["optimizer_state_dict"]["state"].values()))["step"].fill_(201)
                else:
                    payload["unknown"] = True
                torch.save(payload, path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_final_checkpoint(path, fixture()[0], metadata, digest)

    def test_weights_only_deserialization_is_mandatory(self):
        model, optimizer, metadata, _ = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final.pt"
            digest = save_final_checkpoint(path, model, optimizer, metadata)
            original = torch.load
            with patch("torch.load", wraps=original) as observed:
                load_final_checkpoint(path, fixture()[0], metadata, digest)
            self.assertIs(observed.call_args.kwargs["weights_only"], True)
            self.assertEqual(observed.call_args.kwargs["map_location"], "cpu")

    def test_custom_python_pickle_object_not_deserialized(self):
        model, _, metadata, _ = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsupported.pt"
            torch.save({"object": UnsupportedCheckpointObject()}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            import pickle
            with self.assertRaises(pickle.UnpicklingError):
                load_final_checkpoint(path, model, metadata, digest)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
