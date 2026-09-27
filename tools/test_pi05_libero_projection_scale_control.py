"""Synthetic projection-scale interface/numerics tests, never training evidence."""
from copy import deepcopy
import io
import unittest

import numpy as np
import torch
from torch.nn import functional as F

import pi05_libero_projection_scale_control as control
from pi05_libero_visual_normalization import NormalizedLiberoWorldModel, fit_visual_statistics
from pi05_libero_world_model import LiberoWorldModelConfig, collate_window_inputs
from test_pi05_libero_visual_normalization import synthetic_pack
from test_pi05_libero_action_study_step0 import REGISTRY, TinyDataset


class ProjectionScaleControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def assert_bytes(self, first, second):
        self.assertEqual(first.dtype, second.dtype)
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(torch.equal(first.detach().contiguous().reshape(-1).view(torch.uint8),
                                    second.detach().contiguous().reshape(-1).view(torch.uint8)))

    def model_pair(self):
        pack, split = synthetic_pack()
        stats = fit_visual_statistics(pack, split)
        config = LiberoWorldModelConfig(hidden_dim=4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(83)
            baseline = NormalizedLiberoWorldModel(config, REGISTRY, stats).eval()
            after_baseline = torch.get_rng_state().clone()
            torch.manual_seed(83)
            scaled = control.ProjectionScaledLiberoWorldModel(config, REGISTRY, stats).eval()
            after_scaled = torch.get_rng_state().clone()
        return baseline, scaled, after_baseline, after_scaled

    def inputs(self, *, candidates=1):
        dataset = TinyDataset()
        rng = np.random.default_rng(17)
        for item in dataset.items[:2]:
            item["inputs"]["history_visual_latent"][:] = rng.normal(1, .25, (4, 2, 2048)).astype(np.float32)
            item["inputs"]["history_state"][:] = rng.normal(0, .2, (4, 8)).astype(np.float32)
            item["inputs"]["candidate_actions"] = rng.normal(0, .25, (candidates, 3, 7)).astype(np.float32)
        return collate_window_inputs([row["inputs"] for row in dataset.items[:2]])

    def test_exact_formula_for_vector_and_native_multidimensional_values(self):
        for shape in ((4,), (2, 4), (2, 3, 4)):
            z = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) / 3 - 1
            work = z.double()
            expected = (work / torch.sqrt(1 + torch.mean(work.square(), dim=-1, keepdim=True))).float()
            self.assert_bytes(control.smooth_projection_scale(z), expected)

    def test_zero_is_exact_and_zero_gradient_is_identity(self):
        z = torch.zeros((2, 3, 4), dtype=torch.float32, requires_grad=True)
        result = control.smooth_projection_scale(z)
        self.assert_bytes(result, z)
        result.sum().backward()
        self.assert_bytes(z.grad, torch.ones_like(z))

    def test_direction_preserved_amplitude_reduced_and_rms_bounded(self):
        z = torch.tensor([[1., -2., 3., -4.], [0., 2., 0., -3.]])
        result = control.smooth_projection_scale(z)
        torch.testing.assert_close(torch.nn.functional.cosine_similarity(z, result, dim=-1), torch.ones(2))
        self.assertTrue(bool((result.abs() <= z.abs()).all()))
        self.assertTrue(bool((result.square().mean(-1) < 1).all()))
        # RMS control does not imply that each coordinate is below one.
        concentrated = control.smooth_projection_scale(torch.tensor([[10., 0., 0., 0.]]))
        self.assertGreater(concentrated[0, 0].item(), 1.)

    def test_last_dimension_only_no_batch_frame_or_candidate_statistics(self):
        z = torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4) / 8 - 2
        result = control.smooth_projection_scale(z)
        for index in np.ndindex(tuple(z.shape[:-1])):
            self.assert_bytes(result[index], control.smooth_projection_scale(z[index]))
        changed = z.clone()
        changed[1, 1, 2] *= 99
        later = control.smooth_projection_scale(changed)
        self.assert_bytes(result[0], later[0])
        self.assert_bytes(result[1, 0], later[1, 0])

    def test_owned_output_and_noncontiguous_input_remain_unchanged(self):
        z = torch.arange(48, dtype=torch.float32).reshape(3, 16)[:, ::2]
        original = z.clone()
        result = control.smooth_projection_scale(z)
        self.assert_bytes(z, original)
        self.assertNotEqual(result.untyped_storage().data_ptr(), z.untyped_storage().data_ptr())
        result.zero_()
        self.assert_bytes(z, original)

    def test_nonfinite_wrong_dtype_empty_scalar_and_sparse_rejected(self):
        invalid = [torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.ones(3, dtype=torch.float64),
                   torch.ones(3, dtype=torch.int64), torch.empty(0), torch.empty(2, 0), torch.tensor(1.),
                   torch.eye(2).to_sparse(), [1., 2.]]
        for value in invalid:
            with self.subTest(value_type=type(value), shape=getattr(value, "shape", None)), self.assertRaises(ValueError):
                control.smooth_projection_scale(value)

    def test_gradients_finite_nonzero_and_radial_derivative_positive(self):
        direction = torch.tensor([1., -2., 3., 4.])
        alpha = torch.tensor(2., requires_grad=True)
        result = control.smooth_projection_scale(alpha * direction)
        derivative, = torch.autograd.grad((result * direction).sum(), alpha)
        expected = direction.square().sum() / (1 + alpha.detach().square() * direction.square().mean()).pow(1.5)
        self.assertGreater(derivative.item(), 0.)
        torch.testing.assert_close(derivative, expected, rtol=1e-5, atol=1e-7)
        z = torch.tensor([[.5, -1., 2., 3.]], requires_grad=True)
        control.smooth_projection_scale(z).square().sum().backward()
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
        self.assertGreater(torch.count_nonzero(z.grad).item(), 0)

    def test_large_finite_f32_values_use_f64_intermediates_without_overflow(self):
        maximum = torch.finfo(torch.float32).max
        z = torch.tensor([[1000., -2000., 3000., -4000.], [1e20, -1e20, 1e20, -1e20],
                          [maximum, -maximum, maximum, -maximum]])
        result = control.smooth_projection_scale(z)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool((result.square().mean(-1) <= 1 + 2e-7).all()))
        # Moderate-scale radial differentiation still retains a positive signal.
        alpha = torch.tensor(20., requires_grad=True)
        direction = torch.tensor([1., -2., 3., -4.])
        derivative, = torch.autograd.grad((control.smooth_projection_scale(alpha * direction) * direction).sum(), alpha)
        self.assertTrue(bool(torch.isfinite(derivative)))
        self.assertGreater(derivative.item(), 0.)

    def test_same_23_parameter_names_bytes_and_rng_no_new_trainable_parameters(self):
        baseline, scaled, initial_rng, scaled_rng = self.model_pair()
        left, right = dict(baseline.named_parameters()), dict(scaled.named_parameters())
        self.assertEqual(len(left), 23)
        self.assertEqual(set(left), set(right))
        for name in left: self.assert_bytes(left[name], right[name])
        self.assert_bytes(initial_rng, scaled_rng)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(p.numel() for p in scaled.parameters()))
        for i in range(2): self.assertIsInstance(scaled.view_projections[i], torch.nn.Linear)

    def test_standalone_wrapper_owns_parameters_preserves_flags_and_consumes_no_rng(self):
        with torch.random.fork_rng(devices=[]):
            original = torch.nn.Linear(3, 4, bias=False).eval().requires_grad_(False)
            before_rng = torch.get_rng_state().clone()
            wrapped = control.ScaledVisualLinear(original)
            self.assert_bytes(torch.get_rng_state(), before_rng)
        self.assertFalse(wrapped.training)
        self.assertFalse(wrapped.weight.requires_grad)
        self.assertIsNone(wrapped.bias)
        self.assert_bytes(wrapped.weight, original.weight)
        self.assertNotEqual(wrapped.weight.data_ptr(), original.weight.data_ptr())
        before = wrapped.weight.clone()
        with torch.no_grad(): original.weight.add_(1)
        self.assert_bytes(wrapped.weight, before)

    def test_metadata_and_fixed_persistent_version_marker(self):
        _, model, _, _ = self.model_pair()
        metadata = model.metadata()
        self.assertEqual(metadata["schema"], control.SCHEMA)
        details = metadata["projection_scale_control"]
        self.assertEqual(details["formula"], control.FORMULA)
        self.assertEqual(details["version"], control.VERSION)
        self.assertEqual(details["version_buffer"], control.VERSION_BUFFER)
        self.assertEqual(details["compute_dtype"], "float64_intermediate_float32_output")
        self.assertEqual(details["fixed_additive_constant"], 1.)
        self.assertEqual(details["normalization_axis"], "last_hidden_dimension")
        for key in ("trainable_scale", "centering", "batch_statistics", "rms_bound_implies_coordinate_nonsaturation"):
            self.assertFalse(details[key])
        marker = model.state_dict()[control.VERSION_BUFFER]
        self.assertEqual(marker.dtype, torch.int64)
        self.assertEqual(marker.shape, torch.Size([]))
        self.assertEqual(marker.item(), 1)
        self.assertFalse(marker.requires_grad)

    def test_actual_projection_control_then_unchanged_tanh_and_mask_contract(self):
        _, model, _, _ = self.model_pair()
        inputs = self.inputs()
        inputs["history_visual_valid"][0, 0, 0] = False
        inputs["history_visual_latent"][0, 0, 0] = float("nan")
        saved, handles = {}, []
        def projection_input(view):
            def hook(module, args):
                # Compute from the actual strided view in the actual no_grad
                # context. Recomputing later from an owned contiguous clone
                # can select a different Linear kernel on torch 2.10.
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(args[0].is_contiguous())
                raw = F.linear(args[0], module.weight, module.bias)
                saved[f"input{view}"] = args[0].detach().clone()
                saved[f"raw{view}"] = raw.detach().clone()
                saved[f"expected{view}"] = control.smooth_projection_scale(raw).detach().clone()
                return None
            return hook
        def projection_output(view):
            def hook(module, args, result):
                self.assert_bytes(result, saved[f"expected{view}"])
                saved[f"output{view}"] = result.detach().clone()
                saved[f"post{view}"] = torch.tanh(result).detach().clone()
                return None
            return hook
        def history_input(module, args):
            saved["history"] = args[0].detach().clone()
            return None
        for i, layer in enumerate(model.view_projections):
            handles.append(layer.register_forward_pre_hook(projection_input(i)))
            handles.append(layer.register_forward_hook(projection_output(i)))
        handles.append(model.history_projection.register_forward_pre_hook(history_input))
        try:
            with torch.no_grad(): model(**inputs)
        finally:
            for handle in handles: handle.remove()
        self.assertEqual(torch.count_nonzero(saved["input0"][0, 0]).item(), 0)
        for view in range(2):
            self.assert_bytes(saved[f"output{view}"], saved[f"expected{view}"])
            self.assert_bytes(saved["history"][..., view * 4:(view + 1) * 4], saved[f"post{view}"])
        self.assert_bytes(saved["history"][..., -16:-8], inputs["history_state"])
        self.assert_bytes(saved["history"][..., -8:], inputs["history_state_valid"].float())

    def test_raw_skip_and_all_caller_inputs_preserved(self):
        _, model, _, _ = self.model_pair()
        inputs = self.inputs()
        inputs["history_visual_valid"][0, -1, 1] = False
        inputs["history_visual_latent"][0, -1, 1] = float("nan")
        before = deepcopy(inputs)
        with torch.no_grad():
            model.visual_residual_head.weight.zero_()
            model.visual_residual_head.bias.zero_()
            prediction = model(**inputs)
        skip = torch.where(inputs["history_visual_valid"][..., None], inputs["history_visual_latent"], 0.)[:, -1, None, None]
        self.assert_bytes(prediction["pred_future_visual_latent"], skip.expand_as(prediction["pred_future_visual_latent"]))
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor): self.assert_bytes(value, before[name])
            else: self.assertEqual(value, before[name])

    def test_model_backward_finite_nonzero_without_updates_or_buffer_gradients(self):
        _, model, _, _ = self.model_pair()
        before = deepcopy(model.state_dict())
        inputs = self.inputs()
        outputs = model(**inputs)
        (outputs["pred_future_visual_latent"].square().mean() + outputs["pred_state_delta"].square().mean()).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        self.assertGreater(torch.count_nonzero(model.view_projections[0].weight.grad).item(), 0)
        for name, value in model.state_dict().items(): self.assert_bytes(value, before[name])
        self.assertTrue(all(value.grad is None for value in model.buffers()))

    def test_candidate_independence_and_horizon_causality_remain_native(self):
        _, model, _, _ = self.model_pair()
        inputs = self.inputs(candidates=2)
        with torch.no_grad(): first = model(**inputs)
        other_candidate = deepcopy(inputs)
        other_candidate["candidate_actions"][:, 1] += 4
        late_action = deepcopy(inputs)
        late_action["candidate_actions"][:, 0, -1] -= 4
        with torch.no_grad(): second, third = model(**other_candidate), model(**late_action)
        for name in first:
            self.assert_bytes(first[name][:, 0], second[name][:, 0])
            self.assert_bytes(first[name][:, 0, :2], third[name][:, 0, :2])
            self.assert_bytes(first[name][:, 1], third[name][:, 1])

    def test_strict_checkpoint_roundtrip_preserves_predictions(self):
        _, model, _, _ = self.model_pair()
        _, restored, _, _ = self.model_pair()
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        inputs = self.inputs()
        with torch.no_grad(): first, second = model(**inputs), restored(**inputs)
        for name in first: self.assert_bytes(first[name], second[name])

    def test_legacy_missing_tampered_marker_non_strict_assign_rejected_before_weights(self):
        baseline, model, _, _ = self.model_pair()
        states = []
        states.append((baseline.state_dict(), {}))
        for mode in ("missing", "value", "dtype", "shape", "strict", "assign"):
            state, kwargs = deepcopy(model.state_dict()), {}
            state[next(iter(dict(model.named_parameters())))].add_(5)
            if mode == "missing": del state[control.VERSION_BUFFER]
            elif mode == "value": state[control.VERSION_BUFFER].add_(1)
            elif mode == "dtype": state[control.VERSION_BUFFER] = state[control.VERSION_BUFFER].float()
            elif mode == "shape": state[control.VERSION_BUFFER] = state[control.VERSION_BUFFER].reshape(1)
            elif mode == "strict": kwargs["strict"] = False
            else: kwargs["assign"] = True
            states.append((state, kwargs))
        for i, (state, kwargs) in enumerate(states):
            before = deepcopy(model.state_dict())
            with self.subTest(case=i), self.assertRaises((ValueError, RuntimeError)):
                model.load_state_dict(state, **kwargs)
            for name, value in before.items(): self.assert_bytes(model.state_dict()[name], value)

    def test_live_version_and_inherited_normalization_guards_are_retained(self):
        _, model, _, _ = self.model_pair()
        pristine = deepcopy(model.state_dict())
        with torch.no_grad(): getattr(model, control.VERSION_BUFFER).add_(1)
        with self.assertRaisesRegex(ValueError, "version marker"):
            model(**self.inputs())
        model.load_state_dict(pristine)
        wrong_stats = deepcopy(pristine)
        wrong_stats["visual_mean"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "constructor-bound statistics"):
            model.load_state_dict(wrong_stats)
        for name, value in pristine.items(): self.assert_bytes(model.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
