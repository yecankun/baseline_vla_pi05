"""Small synthetic forward/backward/update and serialization guards only.

Final200 checkpoint fixtures perform ONE synthetic update then deliberately set
optimizer counters to200 solely to exercise serialization. They are never
evidence of200 learned updates or any result on the public feature cache.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import pi05_libero_action_study_training as training
from pi05_libero_action_ablation import action_ablation_inputs
from pi05_libero_world_model import LiberoWorldModel, LiberoWorldModelConfig
from smoke_pi05_libero_world_model import parameter_sha


REGISTRY = [{"task_id": 9, "source_task_index": 39,
             "task_instruction": "pick up the black bowl on the wooden cabinet and place it on the plate"}]


def fixture(arm="observed_action"):
    # Default fork_rng snapshots every CUDA device, initializing CUDA even for
    # this CPU-only fixture and contaminating later runtime-guard tests.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(841)
        model = LiberoWorldModel(LiberoWorldModelConfig(hidden_dim=5), REGISTRY)
    generator = torch.Generator(device="cpu").manual_seed(842)
    inputs = {"history_visual_latent": torch.randn(2, 4, 2, 2048, generator=generator),
              "history_visual_valid": torch.ones(2, 4, 2, dtype=torch.bool),
              "history_state": torch.randn(2, 4, 8, generator=generator),
              "history_state_valid": torch.ones(2, 4, 8, dtype=torch.bool),
              "candidate_actions": torch.randn(2, 1, 3, 7, generator=generator),
              "task_instruction": [REGISTRY[0]["task_instruction"]] * 2}
    targets = {"future_visual_latent": torch.randn(2, 3, 2, 2048, generator=generator),
               "future_visual_valid": torch.ones(2, 3, 2, dtype=torch.bool),
               "state_delta": torch.randn(2, 3, 8, generator=generator),
               "state_target_valid": torch.ones(2, 3, 8, dtype=torch.bool)}
    return model, action_ablation_inputs(inputs, arm), targets


def metadata_and_binding(model, initial, arm="observed_action"):
    final = parameter_sha(model)
    binding = training.make_checkpoint_binding(seed=20260912, arm=arm, study_plan_sha256="b" * 64,
        training_plan_sha256="c" * 64, training_authorization_sha256="d" * 64, sampling_plan_sha256="e" * 64,
        initial_parameter_sha256=initial, final_parameter_sha256=final)
    metadata = {key: "a" * 64 for key in training.native.HASH_KEYS}
    metadata.update(model_config=asdict(model.config), task_registry=deepcopy(REGISTRY), step=200,
        checkpoint_kind="final_diagnostic_no_resume", authorization_sha256=binding["training_authorization_sha256"],
        protocol_sha256=binding["study_plan_sha256"],
        sampling_plan_sha256=binding["sampling_plan_sha256"], initial_parameter_sha256=initial,
        final_parameter_sha256=final)
    return metadata, binding


def serialization_fixture(arm="observed_action"):
    model, inputs, targets = fixture(arm)
    initial = parameter_sha(model)
    optimizer = training.build_optimizer(model, deepcopy(training.OPTIMIZER_CONFIG))
    training.train_step(model, optimizer, inputs, targets, arm=arm, expected_step=1)
    # SYNTHETIC COUNTER FIXTURE ONLY. Not200 actual updates, never a run result.
    for state in optimizer.state.values():
        state["step"].fill_(200)
    optimizer.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    metadata, binding = metadata_and_binding(model, initial, arm)
    return model, optimizer, metadata, binding


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_both_arm_gradient_probe_has_no_optimizer_or_parameter_change(self):
        for arm in training.ACTION_ABLATION_MODES:
            model, inputs, targets = fixture(arm)
            before = parameter_sha(model)
            versions = [p._version for p in model.parameters()]
            input_sha, target_sha = training._batch_signature(inputs), training._batch_signature(targets)
            with self.subTest(arm=arm), patch.object(torch.optim, "AdamW", side_effect=AssertionError("must not construct optimizer")):
                result = training.gradient_probe(model, inputs, targets, arm)
            self.assertEqual(parameter_sha(model), before)
            self.assertEqual([p._version for p in model.parameters()], versions)
            self.assertEqual(result["parameter_sha256_before"], result["parameter_sha256_after"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertFalse(result["optimizer_constructed"])
            self.assertEqual(result["backward_calls"], 1)
            self.assertGreater(result["global_grad_norm"], 0)
            self.assertTrue(all(p.grad is None and p.requires_grad for p in model.parameters()))
            self.assertTrue(all(m.training for m in model.modules()))
            self.assertEqual(training._batch_signature(inputs), input_sha)
            self.assertEqual(training._batch_signature(targets), target_sha)
            self.assertEqual(result["parameters_with_grad"], len(list(model.parameters())))
            self.assertEqual(result["action_projection_weight_exact_zero_grad"], arm == "normalized_zero_action")

    def test_zero_arm_requires_pretransformed_inputs(self):
        model, inputs, targets = fixture()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with self.assertRaisesRegex(ValueError, "already transformed exact-zero"):
            training.gradient_probe(model, inputs, targets, "normalized_zero_action")
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_observed_arm_may_contain_actual_all_zero_actions(self):
        model, inputs, targets = fixture("normalized_zero_action")
        result = training.gradient_probe(model, inputs, targets, "observed_action")
        self.assertTrue(result["action_projection_weight_exact_zero_grad"])
        self.assertGreater(result["global_grad_norm"], 0)

    def test_missing_nonfinite_and_global_zero_gradient_fail_then_clear(self):
        for defect in ("missing", "nonfinite", "all_zero"):
            model, inputs, targets = fixture()
            handles = []
            if defect == "missing":
                model.register_parameter("unused_parameter", torch.nn.Parameter(torch.ones(1)))
            elif defect == "nonfinite":
                handles.append(next(model.parameters()).register_hook(lambda g: torch.full_like(g, float("nan"))))
            else:
                handles.extend(p.register_hook(torch.zeros_like) for p in model.parameters())
            before = parameter_sha(model)
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, "gradient"):
                training.gradient_probe(model, inputs, targets, "observed_action")
            self.assertEqual(parameter_sha(model), before)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            for handle in handles:
                handle.remove()

    def test_wrong_mode_or_frozen_parameter_clears_existing_gradients(self):
        for mode in ("eval", "frozen"):
            model, inputs, targets = fixture()
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            if mode == "eval":
                model.eval()
            else:
                next(model.parameters()).requires_grad_(False)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                training.gradient_probe(model, inputs, targets, "observed_action")
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_probe_rejects_inference_or_no_grad(self):
        model, inputs, targets = fixture()
        for context in (torch.inference_mode, torch.no_grad):
            with context(), self.assertRaisesRegex(ValueError, "enabled ordinary autograd"):
                training.gradient_probe(model, inputs, targets, "observed_action")

    def test_probe_detects_version_change_even_when_parameter_bytes_restored(self):
        model, inputs, targets = fixture()
        def mutate(owner, _):
            with torch.no_grad():
                owner.action_projection.weight.add_(0)
        handle = model.register_forward_pre_hook(mutate)
        before = parameter_sha(model)
        with self.assertRaisesRegex(ValueError, "parameter bytes or versions"):
            training.gradient_probe(model, inputs, targets, "observed_action")
        self.assertEqual(parameter_sha(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        handle.remove()

    def test_probe_target_and_metadata_guards(self):
        for defect in ("extra_input", "extra_target", "target_grad", "target_nan", "unknown_arm"):
            model, inputs, targets = fixture()
            arm = "observed_action"
            if defect == "extra_input":
                inputs["group_id"] = "forbidden"
            elif defect == "extra_target":
                targets["exact_contact"] = torch.zeros(2)
            elif defect == "target_grad":
                targets["state_delta"].requires_grad_()
            elif defect == "target_nan":
                targets["state_delta"][0, 0, 0] = float("nan")
            else:
                arm = "new_arm"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                training.gradient_probe(model, inputs, targets, arm)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_step_reuses_fixed_native_update_for_each_arm(self):
        for arm in training.ACTION_ABLATION_MODES:
            model, inputs, targets = fixture(arm)
            optimizer = training.build_optimizer(model, deepcopy(training.OPTIMIZER_CONFIG))
            initial = parameter_sha(model)
            weight = model.action_projection.weight.detach().clone()
            with self.subTest(arm=arm):
                result = training.train_step(model, optimizer, inputs, targets, arm=arm, expected_step=1)
            self.assertEqual(result["step"], 1)
            self.assertNotEqual(parameter_sha(model), initial)
            self.assertEqual(result["action_projection_weight_exact_zero_grad"], arm == "normalized_zero_action")
            self.assertGreater(result["global_grad_norm_before_clip"], 0)
            self.assertLessEqual(result["global_grad_norm_after_clip"], 1.000001)
            self.assertTrue(all(s["step"].item() == 1 for s in optimizer.state.values()))
            if arm == "normalized_zero_action":
                self.assertTrue(torch.equal(weight, model.action_projection.weight))

    def test_zero_action_gradient_corruption_rejected_before_optimizer_step(self):
        model, inputs, targets = fixture("normalized_zero_action")
        handle = model.action_projection.weight.register_hook(lambda grad: grad + 1)
        optimizer = training.build_optimizer(model, deepcopy(training.OPTIMIZER_CONFIG))
        before = parameter_sha(model)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "exactly zero before optimizer"):
                training.train_step(model, optimizer, inputs, targets, arm="normalized_zero_action", expected_step=1)
            step.assert_not_called()
        self.assertFalse(optimizer.state)
        self.assertEqual(parameter_sha(model), before)
        handle.remove()
        model.zero_grad(set_to_none=True)

    def test_step_counter_and_budget_still_owned_by_old_guard(self):
        for step in (0, 2, 201, True):
            model, inputs, targets = fixture()
            optimizer = training.build_optimizer(model, deepcopy(training.OPTIMIZER_CONFIG))
            with self.subTest(step=step), self.assertRaises(ValueError):
                training.train_step(model, optimizer, inputs, targets, arm="observed_action", expected_step=step)
            self.assertFalse(optimizer.state)

    def test_checkpoint_envelope_restricted_roundtrip_and_explicit_binding_semantics(self):
        model, optimizer, metadata, binding = serialization_fixture()
        original_metadata = deepcopy(metadata)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_counter_fixture_only.pt"
            envelope = training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            self.assertEqual(metadata, original_metadata)
            self.assertEqual(envelope["training_authorization_sha256"], metadata["authorization_sha256"])
            self.assertEqual(envelope["legacy_metadata"]["authorization_sha256"], training.canonical_sha256(binding))
            self.assertNotEqual(envelope["legacy_metadata"]["authorization_sha256"], metadata["authorization_sha256"])
            other = fixture()[0].eval().requires_grad_(False)
            with patch.object(training.native.torch, "load", wraps=torch.load) as load:
                payload = training.load_final_checkpoint(path, other, envelope, binding, expected_metadata=metadata)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertEqual(parameter_sha(other), parameter_sha(model))
            self.assertEqual(payload["metadata"], envelope["legacy_metadata"])

    def test_checkpoint_rejects_cross_arm_seed_draw_and_plan_before_load(self):
        model, optimizer, metadata, binding = serialization_fixture()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_counter_fixture_only.pt"
            envelope = training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            for key, value in (("arm", "normalized_zero_action"), ("seed", 20260913),
                               ("sampling_plan_sha256", "f" * 64), ("training_plan_sha256", "f" * 64),
                               ("study_plan_sha256", "f" * 64), ("training_authorization_sha256", "f" * 64),
                               ("initial_parameter_sha256", "f" * 64), ("final_parameter_sha256", "f" * 64)):
                changed = {**deepcopy(binding), key: value}
                other = fixture()[0]
                before = parameter_sha(other)
                with self.subTest(key=key), patch.object(training.native, "load_final_checkpoint", side_effect=AssertionError("must reject first")):
                    with self.assertRaisesRegex(ValueError, "binding mismatch"):
                        training.load_final_checkpoint(path, other, envelope, changed, expected_metadata=metadata)
                self.assertEqual(parameter_sha(other), before)

    def test_zero_arm_checkpoint_roundtrip_uses_its_own_binding(self):
        model, optimizer, metadata, binding = serialization_fixture("normalized_zero_action")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_zero_counter_fixture_only.pt"
            envelope = training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            other = fixture("normalized_zero_action")[0].eval().requires_grad_(False)
            training.load_final_checkpoint(path, other, envelope, binding, expected_metadata=metadata)
            self.assertEqual(parameter_sha(other), binding["final_parameter_sha256"])
            self.assertEqual(envelope["binding"]["arm"], "normalized_zero_action")

    def test_zero_arm_hook_removed_when_native_counter_guard_rejects(self):
        model, inputs, targets = fixture("normalized_zero_action")
        optimizer = training.build_optimizer(model, deepcopy(training.OPTIMIZER_CONFIG))
        before_hooks = len(model.action_projection.weight._backward_hooks or {})
        with self.assertRaises(ValueError):
            training.train_step(model, optimizer, inputs, targets, arm="normalized_zero_action", expected_step=2)
        self.assertEqual(len(model.action_projection.weight._backward_hooks or {}), before_hooks)
        self.assertFalse(optimizer.state)

    def test_checkpoint_file_hash_nooverwrite_and_only_final200(self):
        model, optimizer, metadata, binding = serialization_fixture()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_counter_fixture_only.pt"
            envelope = training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            contents = path.read_bytes()
            with self.assertRaises(FileExistsError):
                training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            self.assertEqual(path.read_bytes(), contents)
            with path.open("ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaisesRegex(ValueError, "file SHA256"):
                training.load_final_checkpoint(path, fixture()[0], envelope, binding, expected_metadata=metadata)
            changed = {**binding, "step": 199}
            with self.assertRaisesRegex(ValueError, "only final200"):
                training.save_final_checkpoint(Path(temp) / "never_created.pt", model, optimizer, metadata, changed)
            self.assertFalse((Path(temp) / "never_created.pt").exists())

    def test_checkpoint_metadata_authorization_sampler_and_parameter_hash_guards(self):
        model, optimizer, metadata, binding = serialization_fixture()
        with TemporaryDirectory() as temp:
            for key in ("authorization_sha256", "sampling_plan_sha256", "initial_parameter_sha256", "final_parameter_sha256"):
                changed = {**metadata, key: "f" * 64}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    training.save_final_checkpoint(Path(temp) / "never_created.pt", model, optimizer, changed, binding)
            self.assertFalse(list(Path(temp).iterdir()))

    def test_envelope_rejects_mutation_unknown_fields_and_bool_step(self):
        model, optimizer, metadata, binding = serialization_fixture()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_counter_fixture_only.pt"
            envelope = training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            for change in (lambda e: e.update(extra="bad"), lambda e: e.update(binding_sha256="f" * 64),
                           lambda e: e["legacy_metadata"].update(authorization_sha256="f" * 64),
                           lambda e: e.update(training_authorization_sha256="f" * 64),
                           lambda e: e.update(checkpoint_is_final_only_not_resumable=1),
                           lambda e: e.update(authorization_field_semantics="ordinary authorization file SHA")):
                changed = deepcopy(envelope)
                change(changed)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    training.load_final_checkpoint(path, fixture()[0], changed, binding, expected_metadata=metadata)
            for key, value in (("seed", True), ("step", True), ("extra", 1), ("training_plan_sha256", "invalid")):
                changed = {**binding, key: value}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    training.load_final_checkpoint(path, fixture()[0], envelope, changed, expected_metadata=metadata)

    def test_load_requires_all_caller_verified_source_and_normalization_metadata(self):
        model, optimizer, metadata, binding = serialization_fixture()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_counter_fixture_only.pt"
            envelope = training.save_final_checkpoint(path, model, optimizer, metadata, binding)
            for key in ("source_report_sha256", "feature_report_sha256", "manifest_sha256",
                        "split_sha256", "normalization_sha256", "protocol_sha256"):
                expected = {**metadata, key: "f" * 64}
                with self.subTest(key=key), patch.object(training.native, "load_final_checkpoint", side_effect=AssertionError("reject before loading")):
                    with self.assertRaisesRegex(ValueError, "caller-verified|protocol SHA"):
                        training.load_final_checkpoint(path, fixture()[0], envelope, binding, expected_metadata=expected)
            with self.assertRaises(TypeError):
                training.load_final_checkpoint(path, fixture()[0], envelope, binding)


if __name__ == "__main__":
    unittest.main()
