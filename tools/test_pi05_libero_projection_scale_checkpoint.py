"""Synthetic final-checkpoint contracts; no dataset training or updates.

Fixture states are intentionally fabricated and explicitly not training proof.
They exercise the real fixed AdamW counter/shape/parameter contract without any
optimizer.step(), backward, public-data read, checkpoint promotion, or resume.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import pi05_libero_projection_scale_checkpoint as ckpt
from test_pi05_libero_action_study_step0 import REGISTRY, TinyDataset
from test_pi05_libero_visual_normalization import synthetic_pack


class ProjectionScaleCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        pack, split = synthetic_pack()
        self.stats = ckpt.gate.previous.fit_visual_statistics(pack, split)
        self.config = ckpt.LiberoWorldModelConfig(hidden_dim=4)
        self.inputs = ckpt.gate.previous.collate_window_inputs([TinyDataset()[i]["inputs"] for i in range(2)])

    def fixture(self, variant=None):
        variant = variant or ckpt.gate.VARIANTS[1]
        seed = ckpt.base.SEEDS[0]
        model = ckpt.gate.build_model(seed, REGISTRY, variant, self.stats, self.config)
        initial = ckpt.named_parameter_sha(model)
        # Synthetic serialization fixture: deliberately edit one scalar. This
        # is not an optimization update and never contributes experiment data.
        with torch.no_grad():
            next(model.parameters()).reshape(-1)[0].add_(0.125)
        model.requires_grad_(True)
        optimizer = ckpt.native.build_optimizer(model, ckpt.native.OPTIMIZER_CONFIG)
        for parameter in model.parameters():
            optimizer.state[parameter] = dict(step=torch.tensor(200., dtype=torch.float32),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        model.eval().requires_grad_(False)
        stats_bytes = (json.dumps(self.stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        binding = dict(seed=seed, arm=ckpt.base.ARMS[0], variant=variant, step=200,
            plan_sha256="a" * 64, smoke_report_sha256="b" * 64,
            normalization_sha256=hashlib.sha256(stats_bytes).hexdigest(), normalization_applied=True,
            projection_scale_control_applied=variant == ckpt.gate.VARIANTS[1], draw_sha256="c" * 64,
            initial_named_parameter_sha256=initial, final_named_parameter_sha256=ckpt.named_parameter_sha(model),
            final_state_sha256=ckpt.base.parameter_hash(model))
        return model, optimizer, binding

    def save(self, fixture=None, name="final.pt"):
        model, optimizer, binding = fixture or self.fixture()
        fresh, record = ckpt.save_reload(self.root / name, model, optimizer, registry=REGISTRY,
            visual_stats=self.stats, binding=binding)
        return model, fresh, record

    def payload(self, name="final.pt"):
        return torch.load(self.root / name, map_location="cpu", weights_only=True)

    def mutated_load(self, payload, record, *, expected_binding=None, stats=None, registry=None):
        path = self.root / "mutated.pt"
        with path.open("wb") as stream:
            torch.save(payload, stream)
        return ckpt.load_checkpoint(path, expected_sha256=ckpt.sha256_file(path),
            expected_binding=expected_binding or record["binding"], registry=REGISTRY if registry is None else registry,
            visual_stats=self.stats if stats is None else stats)

    def test_both_variants_roundtrip_predictions_parameters_buffers_and_metadata(self):
        for variant in ckpt.gate.VARIANTS:
            model, optimizer, binding = self.fixture(variant)
            before = ckpt.base.parameter_hash(model)
            rng = torch.get_rng_state().clone()
            model, fresh, record = self.save((model, optimizer, binding), name=variant + ".pt")
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(ckpt.base.parameter_hash(model), before)
            self.assertEqual(ckpt.base.parameter_hash(fresh), before)
            self.assertEqual(ckpt.named_parameter_sha(fresh), binding["final_named_parameter_sha256"])
            self.assertEqual(fresh.metadata(), model.metadata())
            self.assertEqual(record["schema"], ckpt.SCHEMA)
            self.assertFalse(record["optimizer_state_saved"] or record["resumable"])
            self.assertFalse(any(p.requires_grad or p.grad is not None for p in fresh.parameters()))
            with torch.inference_mode():
                self.assertEqual(ckpt.base.fingerprints(model(**self.inputs)), ckpt.base.fingerprints(fresh(**self.inputs)))
            for name, value in model.state_dict().items():
                self.assertNotEqual(value.data_ptr(), fresh.state_dict()[name].data_ptr())
            self.assertEqual(set(self.payload(variant + ".pt")), ckpt.PAYLOAD_KEYS)
            scaled = variant == ckpt.gate.VARIANTS[1]
            metadata = record["projection_scale_control"]
            self.assertEqual(metadata["formula"] if scaled else metadata, ckpt.control.FORMULA if scaled else None)
            if scaled:
                self.assertEqual(metadata["compute_dtype"], ckpt.control.COMPUTE_DTYPE)
                self.assertEqual(metadata["version"], ckpt.control.VERSION)

    def test_public_loader_requires_correct_external_file_hash(self):
        _, _, record = self.save()
        with patch.object(ckpt.torch, "load", side_effect=AssertionError("must reject before deserializing")):
            for digest in ("0" * 64, "A" * 64, "bad", None):
                with self.subTest(digest=digest), self.assertRaises(ValueError):
                    ckpt.load_checkpoint(self.root / "final.pt", expected_sha256=digest,
                        expected_binding=record["binding"], registry=REGISTRY, visual_stats=self.stats)

    def test_save_rejects_binding_schema_types_flags_and_provenance(self):
        model, optimizer, original = self.fixture()
        changes = (("seed", True), ("seed", 42), ("arm", "unknown"), ("variant", "baseline"),
                   ("step", 201), ("step", 200.), ("normalization_applied", 1),
                   ("normalization_applied", False), ("projection_scale_control_applied", 1),
                   ("projection_scale_control_applied", False), ("plan_sha256", "A" * 64),
                   ("draw_sha256", "invalid"), ("final_state_sha256", "0" * 64),
                   ("normalization_sha256", "0" * 64), ("initial_named_parameter_sha256", "0" * 64))
        for key, value in changes:
            binding = deepcopy(original); binding[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.save((model, optimizer, binding), name="bad.pt")
            self.assertFalse((self.root / "bad.pt").exists())
        for binding in (dict(original, extra=True), {k: v for k, v in original.items() if k != "draw_sha256"}):
            with self.assertRaises(ValueError): self.save((model, optimizer, binding))

    def test_final_checkpoint_cannot_relabel_initial_weights(self):
        model, optimizer, binding = self.fixture()
        binding["initial_named_parameter_sha256"] = binding["final_named_parameter_sha256"]
        with self.assertRaisesRegex(ValueError, "differ from initial"):
            self.save((model, optimizer, binding))

    def test_model_variant_marker_and_eval_contracts_are_strict(self):
        for mutation in ("variant", "version", "train", "grad", "requires_grad"):
            model, optimizer, binding = self.fixture()
            if mutation == "variant":
                binding["variant"] = ckpt.gate.VARIANTS[0]; binding["projection_scale_control_applied"] = False
            elif mutation == "version":
                model.projection_scale_control_version.fill_(2)
            elif mutation == "train": model.train()
            elif mutation == "grad": next(model.parameters()).grad = torch.zeros_like(next(model.parameters()))
            else: model.requires_grad_(True)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.save((model, optimizer, binding))

    def test_real_optimizer_contract_rejects_wrong_step_settings_and_missing_state(self):
        for mutation in ("step", "missing", "lr", "nonfinite", "owner"):
            model, optimizer, binding = self.fixture()
            first = next(model.parameters())
            if mutation == "step": optimizer.state[first]["step"].fill_(199)
            elif mutation == "missing": del optimizer.state[first]
            elif mutation == "lr": optimizer.param_groups[0]["lr"] = 0.002
            elif mutation == "nonfinite": optimizer.state[first]["exp_avg"].fill_(float("nan"))
            else: optimizer.param_groups[0]["params"] = list(reversed(optimizer.param_groups[0]["params"]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.save((model, optimizer, binding))

    def test_overwrite_forbidden_preserves_original_bytes(self):
        fixture = self.fixture()
        self.save(fixture)
        before = (self.root / "final.pt").read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.save(fixture)
        self.assertEqual(before, (self.root / "final.pt").read_bytes())

    def test_rejects_old_envelopes_extra_keys_resume_and_formula_tampering(self):
        _, _, record = self.save()
        source = self.payload()
        cases = []
        for value in ("libero_visual_normalization_final_checkpoint_v1", "pi05_libero_native_final_diagnostic_checkpoint_v1"):
            payload = deepcopy(source); payload["schema"] = value; cases.append(payload)
        payload = deepcopy(source); payload["optimizer_state"] = {}; cases.append(payload)
        for field in ("optimizer_state_saved", "resumable"):
            payload = deepcopy(source); payload[field] = True; cases.append(payload)
        for field, value in (("formula", "identity"), ("version", True), ("compute_dtype", "float32")):
            payload = deepcopy(source); payload["projection_scale_control"][field] = value; cases.append(payload)
        payload = deepcopy(source); payload["model_metadata"]["schema"] = "old"; cases.append(payload)
        for payload in cases:
            with self.subTest(payload=payload["schema"]), self.assertRaises(ValueError):
                self.mutated_load(payload, record)

    def test_loader_provenance_registry_normalization_and_config_tampering(self):
        _, _, record = self.save()
        source = self.payload()
        for field, value in (("seed", ckpt.base.SEEDS[1]), ("arm", ckpt.base.ARMS[1]), ("draw_sha256", "e" * 64)):
            payload = deepcopy(source); payload["binding"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding"):
                self.mutated_load(payload, record)
        payload = deepcopy(source); payload["registry"][0]["task_instruction"] += " changed"
        with self.assertRaisesRegex(ValueError, "registry"): self.mutated_load(payload, record)
        payload = deepcopy(source); payload["normalization"]["mean"][0][0] += 1.
        with self.assertRaisesRegex(ValueError, "normalization"): self.mutated_load(payload, record)
        payload = deepcopy(source); payload["config"]["hidden_dim"] = 5
        with self.assertRaisesRegex(ValueError, "initial"): self.mutated_load(payload, record)
        for field, value in (("context_len", 3), ("horizon", 2), ("dropout", 0.1), ("dropout", False)):
            payload = deepcopy(source); payload["config"][field] = value
            # A relabeled metadata block must not bypass fields whose change
            # would preserve the initialized parameter bytes.
            payload["model_metadata"]["config"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.mutated_load(payload, record)

    def test_normalization_buffers_must_match_supplied_statistics_even_if_final_hash_rebound(self):
        _, _, record = self.save()
        payload = self.payload(); payload["model_state"]["visual_mean"][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, "normalization buffers"):
            self.mutated_load(payload, record)

    def test_loader_rejects_state_keys_dtype_shape_nonfinite_marker_and_final_hash_change(self):
        _, _, record = self.save()
        source = self.payload()
        parameter = "view_projections.0.weight"
        for mutation in ("missing", "dtype", "shape", "nonfinite", "marker", "value", "requires_grad"):
            payload = deepcopy(source)
            if mutation == "missing": del payload["model_state"][parameter]
            elif mutation == "dtype": payload["model_state"][parameter] = payload["model_state"][parameter].double()
            elif mutation == "shape": payload["model_state"][parameter] = payload["model_state"][parameter][1:]
            elif mutation == "nonfinite": payload["model_state"][parameter].fill_(float("inf"))
            elif mutation == "marker": payload["model_state"][ckpt.control.VERSION_BUFFER].fill_(2)
            elif mutation == "value": payload["model_state"][parameter][0, 0] += 1.
            else: payload["model_state"][parameter].requires_grad_(True)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.mutated_load(payload, record)

    def test_reference_cannot_be_silently_promoted_to_scaled_checkpoint(self):
        _, _, record = self.save(self.fixture(ckpt.gate.VARIANTS[0]))
        payload = self.payload()
        payload["projection_scale_control"] = {"version": 1}
        with self.assertRaisesRegex(ValueError, "scale formula"):
            self.mutated_load(payload, record)

    def test_save_and_load_do_not_call_optimizer_step_backward_or_forward(self):
        fixture = self.fixture()
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("no step")), \
             patch.object(torch.Tensor, "backward", side_effect=AssertionError("no backward")), \
             patch.object(ckpt.control.ProjectionScaledLiberoWorldModel, "forward", side_effect=AssertionError("no forward")):
            self.save(fixture)


if __name__ == "__main__":
    unittest.main()
