"""Synthetic file-only checks; no model, dataset, optimizer or environment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import audit_pusht_bc_act_training as audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = {'steps': 2, 'checkpoint_every': 2, 'validation_every': 2,
                     'val_frames': 4, 'val_episodes': 2,
                     'optimizer': {'bc': {'grad_clip_norm': 10}, 'act': {'grad_clip_norm': 10}}}
        self.val = {'frames': 4, 'episodes': 2, 'mae_xy': [1, 3], 'frame_mean_mae': 2,
                    'episode_macro_mae': 2.5, 'rmse': 3, 'out_of_bounds_predictions': 0,
                    'runtime_seconds': 0.01, 'action_clipping': False, 'checkpoint_selection_allowed': False}
        self.paths, self.rows = {}, {}
        for name in ('bc', 'act'):
            self.paths[name] = self.root / name / 'sessions' / 'test' / 'metrics.jsonl'
            self.paths[name].parent.mkdir(parents=True)
            parts = {'mse': 0.2} if name == 'bc' else {'l1_loss': 0.1, 'kld_loss': 0.01}
            self.rows[name] = [{'step': s, 'sample_indices_sha256': str(s) * 64, 'loss': 0.2,
                               'components': parts.copy(), 'gradient_norm_before_clip': 1,
                               'update_seconds': 0.001} for s in (1, 2)]
            self.rows[name][-1].update({'validation': copy.deepcopy(self.val),
                'checkpoint': str(self.root / name / 'checkpoints' / 'step_000002')})

    def scan(self):
        for name in self.paths:
            self.paths[name].write_text(''.join(json.dumps(row) + '\n' for row in self.rows[name]))
        return audit.scan_metrics(self.paths, self.plan, iter(['1' * 64, '2' * 64]))

    def test_complete_pair(self):
        result, digest = self.scan()
        self.assertEqual(result['act']['optimizer_updates_logged'], 2)
        self.assertEqual(result['bc']['final_validation']['frame_mean_mae'], 2)
        self.assertEqual(len(digest), 64)

    def test_missing_update_rejected(self):
        self.rows['bc'].pop()
        with self.assertRaisesRegex(ValueError, 'extra/missing'):
            self.scan()

    def test_duplicate_step_rejected(self):
        self.rows['bc'][-1]['step'] = 1
        with self.assertRaisesRegex(ValueError, 'nonconsecutive'):
            self.scan()

    def test_sampler_drift_rejected(self):
        self.rows['act'][0]['sample_indices_sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'sampler anchor'):
            self.scan()

    def test_nan_loss_rejected(self):
        self.rows['bc'][0]['loss'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid loss'):
            self.scan()

    def test_loss_semantic_drift_rejected(self):
        self.rows['act'][0]['components']['kld_loss'] = 0.02
        with self.assertRaisesRegex(ValueError, 'reduction mismatch'):
            self.scan()

    def test_partial_validation_rejected(self):
        self.rows['act'][-1]['validation']['frames'] = 2
        with self.assertRaisesRegex(ValueError, 'full split'):
            self.scan()

    def test_best_checkpoint_selection_rejected(self):
        self.rows['act'][-1]['validation']['checkpoint_selection_allowed'] = True
        with self.assertRaisesRegex(ValueError, 'selection contract'):
            self.scan()

    def test_wrong_checkpoint_path_rejected(self):
        self.rows['bc'][-1]['checkpoint'] += '_other'
        with self.assertRaisesRegex(ValueError, 'checkpoint path'):
            self.scan()

    def test_validation_schedule_drift_rejected(self):
        self.rows['bc'][0]['validation'] = copy.deepcopy(self.val)
        with self.assertRaisesRegex(ValueError, 'validation schedule'):
            self.scan()

    def test_inconsistent_coordinate_metric_rejected(self):
        self.val['frame_mean_mae'] = 1
        with self.assertRaisesRegex(ValueError, 'coordinate/frame'):
            audit.validation(self.val, self.plan)

    def checkpoint_fixture(self):
        binding = {'protocol': self.plan, 'torch_runtime': 'test', 'package_versions': {'numpy': 'test'}}
        directory = self.root / 'bc' / 'checkpoints' / 'step_000002'
        directory.mkdir(parents=True)
        payload = directory / 'checkpoint.pt'
        payload.write_bytes(b'synthetic non-pickle payload')
        manifest = {'schema': 'pusht_bc_act_checkpoint_v1', 'version': 1, 'step': 2, 'binding': binding,
                    'torch_version': 'test', 'numpy_version': 'test', 'optimizer_class': 'torch.optim.adamw.AdamW',
                    'state_dict_sha256': '1' * 64,
                    'files': {'checkpoint.pt': {'size': payload.stat().st_size, 'sha256': audit.sha256(payload)}}}
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        return directory, binding

    def test_payload_hash_passes_without_unpickling(self):
        _, binding = self.checkpoint_fixture()
        checked = audit.verify_checkpoints(self.root / 'bc', binding, '0' * 64)
        self.assertEqual(len(checked), 1)

    def test_corrupt_same_size_payload_rejected(self):
        directory, binding = self.checkpoint_fixture()
        payload = directory / 'checkpoint.pt'
        payload.write_bytes(b'x' * payload.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'payload SHA256'):
            audit.verify_checkpoints(self.root / 'bc', binding, '0' * 64)

    def test_partial_checkpoint_rejected(self):
        directory, binding = self.checkpoint_fixture()
        (directory.parent / 'step_000003.partial').mkdir()
        with self.assertRaisesRegex(ValueError, 'partial checkpoint'):
            audit.verify_checkpoints(self.root / 'bc', binding, '0' * 64)

    def test_initial_state_not_trained(self):
        _, binding = self.checkpoint_fixture()
        with self.assertRaisesRegex(ValueError, 'initial weights'):
            audit.verify_checkpoints(self.root / 'bc', binding, '1' * 64)


if __name__ == '__main__':
    unittest.main()
