"""Synthetic read-only diagnosis tests; no public data or optimizer calls."""
import io
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_pi05_libero_normalization_diagnosis as runner
from test_pi05_libero_action_study_step0 import TinyDataset, REGISTRY
from test_pi05_libero_visual_normalization import synthetic_pack


class ErrorBucketTests(unittest.TestCase):
    def tensors(self, values):
        return torch.tensor(values, dtype=torch.float32).reshape(1, 3, 2, -1)

    def test_shared_bins_counts_and_exact_edge(self):
        baseline = self.tensors([0., .02, .06, .2, .5, 1.])
        target = torch.zeros_like(baseline)
        buckets = runner.ErrorBuckets()
        buckets.update(baseline, torch.zeros_like(baseline), target)
        result = buckets.summary()
        self.assertEqual([x['count'] for x in result['bins']], [1]*6)
        self.assertEqual([x['improved_count'] for x in result['bins']], [0,1,1,1,1,1])
        self.assertEqual(result['count'], 6)
        self.assertEqual(result['bins'][4]['lower'], .5)
        self.assertEqual(result['bins'][5]['lower'], 1.)
        self.assertIsNone(result['bins'][5]['upper'])
        self.assertAlmostEqual(sum(x['mae_delta_contribution_to_full_support'] for x in result['bins']),
                               -baseline.double().mean().item(), places=14)

    def test_empty_bins_null_not_zero_and_baseline_conditioning(self):
        baseline = self.tensors([0.]*6)
        buckets = runner.ErrorBuckets()
        buckets.update(baseline, torch.ones_like(baseline), baseline)
        result = buckets.summary()
        self.assertEqual(result['bins'][0]['count'], 6)
        self.assertEqual(result['bins'][0]['worsened_count'], 6)
        self.assertEqual(result['bins'][0]['normalized_mae'], 1.)
        for row in result['bins'][1:]:
            self.assertEqual(row['count'], 0)
            self.assertIsNone(row['normalized_mae'])
        json.dumps(result, allow_nan=False)

    def test_mae_can_increase_while_smooth_l1_decreases(self):
        baseline = self.tensors([0.,0.,0.,0.,0.,1.])
        normalized = self.tensors([.1,.1,.1,.1,.1,.7])
        buckets = runner.ErrorBuckets()
        buckets.update(baseline, normalized, torch.zeros_like(baseline))
        sums = buckets.summary()['totals']
        self.assertGreater(sums['normalized_abs_sum'], sums['baseline_abs_sum'])
        self.assertLess(sums['normalized_smooth_l1_sum'], sums['baseline_smooth_l1_sum'])

    def test_equal_episode_macro_not_micro(self):
        summaries = {}
        for name, n, change in [('a',1,1.),('b',3,0.)]:
            b = runner.ErrorBuckets()
            target = torch.zeros((n,3,2,1), dtype=torch.float32)
            b.update(target, target+change, target)
            summaries[name] = b.summary()
        macro = runner.bucket_macro(summaries)
        self.assertEqual(sum(x['mae_delta_contribution_to_full_support'] for x in macro), .5)
        micro = sum(x['totals']['normalized_abs_sum'] for x in summaries.values())/sum(x['count'] for x in summaries.values())
        self.assertEqual(micro, .25)

    def test_invalid_and_empty_inputs_fail_closed(self):
        with self.assertRaises(ValueError): runner.ErrorBuckets().summary()
        with self.assertRaises(ValueError): runner.bucket_macro({})
        good = self.tensors([0.]*6)
        for bad in (good.double(), good.reshape(1,6), good.clone().fill_(float('nan')), good.clone().requires_grad_(True)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                runner.ErrorBuckets().update(bad, good, good)


class FrozenRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        pack, split = synthetic_pack()
        self.norm = runner.previous.fit_visual_statistics(pack, split)
        self.seed, self.arm = runner.base.SEEDS[0], runner.base.ARMS[0]
        self.dataset = TinyDataset()
        self.models = {v: runner.previous.build_model(self.seed, REGISTRY, v, self.norm,
                        runner.LiberoWorldModelConfig(hidden_dim=4)) for v in runner.previous.VARIANTS}
        self.refs = {}
        grouped = {}
        for i, window in enumerate(self.dataset.windows): grouped.setdefault(window.episode_index, []).append(i)
        for ids in grouped.values():
            items = [self.dataset[i] for i in ids]
            inputs = runner.action_ablation_inputs(runner.collate_window_inputs([x['inputs'] for x in items]), self.arm)
            targets = runner.collate_window_targets([x['targets'] for x in items])
            for variant, model in self.models.items():
                with torch.inference_mode(): pred = model(**inputs)
                self.refs[(self.seed, self.arm, variant, tuple(ids))] = dict(input_sha256=runner.base.fingerprints(inputs),
                    target_sha256=runner.base.fingerprints(targets), prediction_sha256=runner.base.fingerprints(pred))

    def execute(self):
        self.trace = io.StringIO()
        return runner.diagnose_errors(self.models, self.dataset, {'state_std':[1.]*8}, self.seed, self.arm, self.refs, self.trace)

    def test_exact_frozen_replay_forward_budget_and_unchanged_state(self):
        counts, hooks = [], []
        before = {v:runner.base.parameter_hash(m) for v,m in self.models.items()}
        for model in self.models.values(): hooks.append(model.register_forward_hook(lambda *args: counts.append(1)))
        try: result = self.execute()
        finally:
            for h in hooks: h.remove()
        self.assertEqual(len(counts), 4)
        self.assertEqual(result['model_forwards'], 4)
        self.assertEqual(result['micro']['windows'], 4)
        self.assertEqual(result['micro']['count'], 4*3*2*2048)
        self.assertEqual(set(result['per_episode']), {'11','22'})
        self.assertEqual({v:runner.base.parameter_hash(m) for v,m in self.models.items()}, before)
        self.assertEqual(len(self.trace.getvalue().splitlines()), 2)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in self.models.values() for p in m.parameters()))

    def test_rejects_prediction_input_target_reference_changes(self):
        for field in ('prediction_sha256','input_sha256','target_sha256'):
            self.setUp()
            ref = next(iter(self.refs.values()))[field]
            ref[next(iter(ref))] = '0'*64
            with self.subTest(field=field), self.assertRaises(ValueError): self.execute()

    def test_masked_targets_rejected_before_bucket_analysis(self):
        self.dataset.items[0]['targets']['future_visual_valid'].fill(False)
        with self.assertRaisesRegex(ValueError, 'all visual targets valid'): self.execute()

    def test_other_masks_rejected_before_frozen_forward(self):
        for part,key in [('inputs','history_visual_valid'),('inputs','history_state_valid'),('targets','state_target_valid')]:
            self.setUp()
            self.dataset.items[0][part][key].fill(False)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'history/state support'): self.execute()

    def test_execution_flag_rejected_before_preflight(self):
        with patch.object(runner, 'validate', side_effect=AssertionError('must not preflight')):
            with self.assertRaises(ValueError): runner.run(Path('.'), 'a'*64, False)
            with self.assertRaises(ValueError): runner.main(['--stage','preflight','--plan-sha256','a'*64,'--execute'])

    def checkpoint_fixture(self, root, variant, corrupt=None):
        model = runner.previous.build_model(self.seed, REGISTRY, variant, self.norm)
        binding = dict(seed=self.seed, arm=self.arm, variant=variant, step=200,
            plan_sha256='a'*64, normalization_sha256='b'*64, normalization_applied=variant!='baseline',
            initial_named_parameter_sha256=runner.previous.named_parameter_sha(model),
            final_named_parameter_sha256=runner.previous.named_parameter_sha(model),
            final_state_sha256=runner.base.parameter_hash(model))
        payload = dict(schema='libero_visual_normalization_final_checkpoint_v1', binding=binding,
            config=asdict(model.config), registry=REGISTRY, model_state=model.state_dict(),
            normalization=self.norm if variant!='baseline' else None, optimizer_state_saved=False, resumable=False)
        if corrupt: corrupt(payload)
        path = root/runner.previous.OUT/'synthetic.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        record = dict(binding=binding, path='synthetic.pt', sha256=runner.sha256_file(path))
        report = dict(plan_sha256='a'*64, normalization_sha256='b'*64,
                      runs={str(self.seed):{self.arm:{variant:{'checkpoint':record}}}})
        return model, report

    def test_checkpoint_exact_roundtrip_and_initial_projection_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            for variant in runner.previous.VARIANTS:
                model, report = self.checkpoint_fixture(Path(tmp), variant)
                loaded, initial = runner.load_frozen(Path(tmp), report, self.seed, self.arm, variant, self.norm, REGISTRY)
                self.assertEqual(runner.base.parameter_hash(model), runner.base.parameter_hash(loaded))
                self.assertEqual(set(initial), {'view_projections.0.weight','view_projections.0.bias',
                                               'view_projections.1.weight','view_projections.1.bias'})
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in loaded.parameters()))

    def test_checkpoint_rejects_schema_normalization_and_bad_tensors(self):
        corruptions = [lambda p:p.update(resumable=True), lambda p:p.update(normalization=None),
                      lambda p:p['model_state']['view_projections.0.weight'].fill_(float('nan'))]
        with tempfile.TemporaryDirectory() as tmp:
            for corrupt in corruptions:
                _, report = self.checkpoint_fixture(Path(tmp), 'train_visual_normalized', corrupt)
                with self.assertRaises(ValueError):
                    runner.load_frozen(Path(tmp), report, self.seed, self.arm, 'train_visual_normalized', self.norm, REGISTRY)


if __name__ == '__main__':
    unittest.main()
