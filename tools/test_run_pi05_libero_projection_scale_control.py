"""Synthetic paired scale-control runner tests; no public data or optimization."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

import run_pi05_libero_projection_scale_control as runner
from test_pi05_libero_action_study_step0 import TinyDataset, REGISTRY
from test_pi05_libero_visual_normalization import synthetic_pack


class ScaleControlRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        pack, split = synthetic_pack()
        self.stats = runner.previous.fit_visual_statistics(pack, split)
        dataset = TinyDataset()
        for i, item in enumerate(dataset.items):
            item['inputs']['history_visual_latent'].fill(i + 1.)
        self.raw = runner.previous.collate_window_inputs([dataset[i]['inputs'] for i in (0, 1)])
        self.targets = runner.previous.collate_window_targets([dataset[i]['targets'] for i in (0, 1)])
        self.seed, self.arm = runner.base.SEEDS[0], runner.base.ARMS[0]
        self.config = runner.previous.LiberoWorldModelConfig(hidden_dim=4)
        self.models = {v: runner.build_model(self.seed, REGISTRY, v, self.stats, self.config) for v in runner.VARIANTS}
        self.make_reference()

    def make_reference(self):
        model = self.models[runner.VARIANTS[0]]
        inputs = runner.previous.action_ablation_inputs(self.raw, self.arm)
        captured, views = runner.previous.capture_views(model, inputs)
        totals = runner.previous.ViewTotals(); totals.update(views)
        model.train().requires_grad_(True)
        try: grad = runner.previous.previous.training.gradient_probe(model, inputs, self.targets, self.arm)
        finally: model.eval().requires_grad_(False)
        self.reference = dict(input_sha256=runner.base.fingerprints(inputs), target_sha256=runner.base.fingerprints(self.targets),
            window_indices=[0, 1], initial_named_parameter_sha256=runner.previous.named_parameter_sha(model),
            prediction_sha256=runner.base.fingerprints(captured['predictions']), activation=totals.summary(), gradient=grad)

    def execute(self):
        return runner.check_pair(self.models, self.raw, self.targets, self.seed, self.arm, [0, 1], self.reference)

    def test_pair_exact_reference_replay_four_forwards_two_backwards_no_optimizer(self):
        calls, handles = [], []
        for model in self.models.values(): handles.append(model.register_forward_hook(lambda *a: calls.append(1)))
        before = {v:runner.base.parameter_hash(m) for v,m in self.models.items()}
        try:
            with ExitStack() as stack:
                for optimizer in ('Adam','AdamW','SGD'):
                    stack.enter_context(patch('torch.optim.'+optimizer, side_effect=AssertionError('optimizer forbidden')))
                result = self.execute()
        finally:
            for h in handles: h.remove()
        self.assertEqual(len(calls), 4)
        self.assertEqual(result['model_forwards'], 4)
        self.assertEqual(result['backward_calls'], 2)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertTrue(result['old_normalized_reference_replayed'])
        self.assertEqual({v:runner.base.parameter_hash(m) for v,m in self.models.items()}, before)
        for model in self.models.values():
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
            self.assertFalse(model.training)

    def test_normalized_zero_action_contract_retained(self):
        self.arm = runner.base.ARMS[1]
        self.make_reference()
        result = self.execute()
        self.assertTrue(all(row['gradient']['action_projection_weight_exact_zero_grad'] for row in result['models'].values()))

    def test_actual_scale_rms_raw_input_and_skip_preserved(self):
        inputs = runner.previous.action_ablation_inputs(self.raw, self.arm)
        before = runner.base.fingerprints(inputs)
        capture, views = runner.capture_views(self.models[runner.VARIANTS[1]], inputs)
        for tensors in views.values():
            self.assertTrue(torch.equal(tensors['pre'], runner.smooth_projection_scale(tensors['raw_affine'])))
            self.assertLessEqual(float(tensors['pre'].double().square().mean(-1).max()), 1.+1e-6)
            self.assertTrue(torch.equal(tensors['post'], torch.tanh(tensors['pre'])))
        self.assertTrue(torch.equal(capture['predictions']['pred_future_visual_latent'], inputs['history_visual_latent'][:,-1,None,None] + capture['residual']))
        self.assertEqual(runner.base.fingerprints(inputs), before)

    def test_capture_restores_hook_registries_on_exception(self):
        model = self.models[runner.VARIANTS[1]]
        before = runner.previous._hook_state(model)
        inputs = runner.previous.action_ablation_inputs(self.raw, self.arm)
        with patch.object(model.history_projection, 'forward', side_effect=ValueError('synthetic failure')):
            with self.assertRaises(ValueError): runner.capture_views(model, inputs)
        self.assertEqual(runner.previous._hook_state(model), before)

    def test_reference_input_target_prediction_gradient_tamper_rejected(self):
        original = deepcopy(self.reference)
        for field in ('input_sha256','target_sha256','prediction_sha256'):
            self.reference = deepcopy(original)
            self.reference[field][next(iter(self.reference[field]))] = '0'*64
            with self.subTest(field=field), self.assertRaises(ValueError): self.execute()
        self.reference = deepcopy(original)
        self.reference['gradient']['loss'] += 1.
        with self.assertRaisesRegex(ValueError, 'gradients did not replay'): self.execute()

    def test_parameter_difference_rejected_before_model_forward(self):
        model = self.models[runner.VARIANTS[1]]
        with torch.no_grad(): next(model.parameters()).add_(.01)
        with patch.object(model, 'forward', side_effect=AssertionError('must not forward')):
            with self.assertRaisesRegex(ValueError, 'fresh parameters differ'): self.execute()

    def test_false_mask_reference_is_rejected_before_capture(self):
        self.targets['state_target_valid'][0,0,0] = False
        self.reference['target_sha256'] = runner.base.fingerprints(self.targets)
        with patch.object(runner, 'capture_views', side_effect=AssertionError('must not capture')):
            with self.assertRaisesRegex(ValueError, 'complete mask support'): self.execute()

    def test_unknown_variant_and_execution_flags_rejected(self):
        with self.assertRaises(ValueError): runner.build_model(self.seed, REGISTRY, 'other', self.stats)
        with patch.object(runner, 'validate', side_effect=AssertionError('must not preflight')):
            with self.assertRaises(ValueError): runner.smoke(Path('.'), 'a'*64, False)
            with self.assertRaises(ValueError): runner.main(['--stage','preflight','--plan-sha256','a'*64,'--execute'])


if __name__ == '__main__': unittest.main()
