"""Read-only audit of the frozen, completed single-session BC/ACT training pair.

Reads public numeric indices, logs and checkpoint bytes; never unpickles weights,
constructs a policy, executes an optimizer or steps an environment. A passing
audit is training/provenance evidence, not a closed-loop benchmark score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_bc_act_v1')
PREFLIGHT = REPO / 'simulation_output/pusht_bc_act_training_preflight_v1_retry2'
PROTOCOL_SHA = '22e26bb73f40cd408b2df4a90f6bc113aaa811abbeb80b90711844f624ba86ea'
PREFLIGHT_SHA = {
    'bc': '28c2b98e064ba06eacf2fa8d13231abbb26446b3fe165a1e5135cb578d8b3bb4',
    'act': 'c45bc3551681a4cb5861203bee051465f12b21a04e3162968295e2e9ff75da8c',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), f'invalid {name}')
    return value


def digest_string(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256')
    return value


def validation(value, plan):
    require(value['frames'] == plan['val_frames'] and value['episodes'] == plan['val_episodes'],
            'validation does not cover the frozen full split')
    require(value['action_clipping'] is False and value['checkpoint_selection_allowed'] is False,
            'validation clipping/selection contract changed')
    require(len(value['mae_xy']) == 2, 'MAE must have two native coordinates')
    for item in value['mae_xy']:
        finite(item, 'coordinate MAE')
    for key in ('frame_mean_mae', 'episode_macro_mae', 'rmse', 'runtime_seconds'):
        finite(value[key], key)
    require(math.isclose(statistics.mean(value['mae_xy']), value['frame_mean_mae'], abs_tol=1e-9),
            'coordinate/frame MAE disagree')
    require(value['rmse'] + 1e-9 >= value['frame_mean_mae'], 'RMSE is below MAE')
    bounds = value['out_of_bounds_predictions']
    require(type(bounds) is int and 0 <= bounds <= plan['val_frames'], 'invalid bounds count')


def scan_metrics(paths, plan, expected_anchors):
    """Check every update, including independent sampler replay, not only tails."""
    result = {name: {'losses': [], 'updates': [], 'validations': [], 'gradient_clip_exceedances': 0}
              for name in ('bc', 'act')}
    sequence_digest = hashlib.sha256()
    with Path(paths['bc']).open() as bc, Path(paths['act']).open() as act:
        count = 0
        for step, pair in enumerate(itertools.zip_longest(bc, act), start=1):
            require(step <= plan['steps'] and all(pair), 'extra/missing model update')
            expected = next(expected_anchors, None)
            require(expected is not None, 'missing expected sampler anchor')
            digest_string(expected)
            sequence_digest.update(bytes.fromhex(expected))
            for name, line in zip(('bc', 'act'), pair):
                row = json.loads(line)
                require(type(row['step']) is int and row['step'] == step, 'nonconsecutive update')
                require(row['sample_indices_sha256'] == expected, f'{name} sampler anchor mismatch at {step}')
                finite(row['loss'], 'loss')
                finite(row['gradient_norm_before_clip'], 'gradient norm', positive=True)
                finite(row['update_seconds'], 'update time', positive=True)
                parts = row['components']
                require(set(parts) == ({'mse'} if name == 'bc' else {'l1_loss', 'kld_loss'}),
                        'loss components changed')
                for key, value in parts.items():
                    finite(value, key)
                expected_loss = parts['mse'] if name == 'bc' else parts['l1_loss'] + 10 * parts['kld_loss']
                require(math.isclose(row['loss'], expected_loss, rel_tol=2e-6, abs_tol=1e-7),
                        'loss/component reduction mismatch')
                scheduled = step % plan['validation_every'] == 0 or step == plan['steps']
                require(('validation' in row) == scheduled, 'validation schedule drift')
                checkpoint_due = step % plan['checkpoint_every'] == 0 or step == plan['steps']
                require(('checkpoint' in row) == checkpoint_due, 'checkpoint schedule drift')
                if checkpoint_due:
                    checkpoint = paths[name].parents[2] / 'checkpoints' / f'step_{step:06d}'
                    require(row['checkpoint'] == str(checkpoint), 'logged checkpoint path differs')
                if scheduled:
                    validation(row['validation'], plan)
                    result[name]['validations'].append({'step': step, **row['validation']})
                data = result[name]
                data['gradient_clip_exceedances'] += row['gradient_norm_before_clip'] > plan['optimizer'][name]['grad_clip_norm']
                data['losses'].append(row['loss'])
                data['updates'].append(row['update_seconds'])
                data['final_components'] = parts
            count = step
    require(count == plan['steps'] and next(expected_anchors, None) is None, 'incomplete update budget')
    for name, data in result.items():
        losses, updates = data.pop('losses'), data.pop('updates')
        data.update({'optimizer_updates_logged': count, 'first_loss': losses[0], 'final_loss': losses[-1],
                     'first_1000_loss_mean': statistics.mean(losses[:1000]),
                     'last_1000_loss_mean': statistics.mean(losses[-1000:]),
                     'mean_update_seconds': statistics.mean(updates),
                     'sum_update_seconds': sum(updates),
                     'final_validation': data['validations'][-1],
                     'metrics_sha256': sha256(paths[name])})
    return result, sequence_digest.hexdigest()


def verify_checkpoints(run_dir, binding, initial_hash):
    plan = binding['protocol']
    steps = list(range(plan['checkpoint_every'], plan['steps'] + 1, plan['checkpoint_every']))
    base = run_dir / 'checkpoints'
    require({p.name for p in base.iterdir()} == {f'step_{s:06d}' for s in steps},
            'missing/extra/partial checkpoint directory')
    checked = []
    for step in steps:
        path = base / f'step_{step:06d}'
        require(not path.is_symlink() and path.is_dir(), 'checkpoint path is not a regular directory')
        require({p.name for p in path.iterdir()} == {'manifest.json', 'checkpoint.pt'}, 'checkpoint entries differ')
        manifest = read_json(path / 'manifest.json')
        require(manifest['schema'] == 'pusht_bc_act_checkpoint_v1' and manifest['version'] == 1,
                'checkpoint schema changed')
        require(manifest['step'] == step and manifest['binding'] == binding, 'checkpoint binding changed')
        require(manifest['torch_version'] == binding['torch_runtime']
                and manifest['numpy_version'] == binding['package_versions']['numpy'], 'checkpoint runtime drift')
        require(manifest['optimizer_class'] == 'torch.optim.adamw.AdamW', 'optimizer class changed')
        require(set(manifest['files']) == {'checkpoint.pt'}, 'unexpected payload path')
        payload = path / 'checkpoint.pt'
        require(not payload.is_symlink() and payload.is_file(), 'payload is not a regular file')
        spec = manifest['files']['checkpoint.pt']
        require(payload.stat().st_size == spec['size'], 'checkpoint size mismatch')
        digest = sha256(payload)
        require(digest == spec['sha256'], 'checkpoint payload SHA256 mismatch')
        state_hash = digest_string(manifest['state_dict_sha256'])
        require(state_hash != initial_hash, 'checkpoint retained initial weights')
        checked.append({'step': step, 'path': str(path), 'payload_bytes': spec['size'],
                        'payload_sha256': digest, 'state_dict_sha256': state_hash,
                        'manifest_sha256': sha256(path / 'manifest.json')})
    require(len({row['state_dict_sha256'] for row in checked}) == len(checked), 'unchanged saved model states')
    return checked


def sampler_hashes(binding, plan):
    """CPU-only, independent reconstruction from two public numeric columns."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    require(str(torch.__version__) == binding['torch_runtime'], 'torch sampler runtime differs')
    data = binding['data']
    split = read_json(Path(data['gate_dir']) / 'split.json')
    rows = pq.read_table(Path(data['source_root']) / 'data/chunk-000/file-000.parquet',
                         columns=['index', 'episode_index']).to_pydict()
    indices = np.asarray(rows['index'], dtype=np.int64)
    episodes = np.asarray(rows['episode_index'], dtype=np.int64)
    require(np.array_equal(indices, np.arange(data['record_count'])), 'public numeric indices changed')
    train, val = set(split['train_episodes']), set(split['val_episodes'])
    require(not train.intersection(val) and train | val == set(episodes.tolist()), 'split coverage/leakage')
    anchors = indices[np.isin(episodes, list(train))]
    require(len(anchors) == plan['train_frames'], 'training frame count changed')
    for step in range(plan['steps']):
        generator = torch.Generator(device='cpu').manual_seed(plan['sampler_seed'] + step)
        positions = torch.randint(len(anchors), (plan['batch_size'],), generator=generator).numpy()
        yield hashlib.sha256(anchors[positions].tobytes()).hexdigest()


def verify_current_inputs(binding):
    sources = {'trainer': 'tools/train_pusht_bc_act.py', 'models': 'tools/pusht_bc_act_models.py',
               'data_adapter': 'tools/pusht_bc_act_training_data.py', 'checkpoint': 'tools/pusht_bc_act_checkpoint.py'}
    for key, relative in sources.items():
        require(sha256(REPO / relative) == binding['sources'][key], f'current source changed: {key}')
    for relative, digest in binding['data']['source_code_sha256'].items():
        require(sha256(REPO / relative) == digest, f'data source code changed: {relative}')
    for name, version in binding['package_versions'].items():
        require(importlib.metadata.version(name) == version, f'package version changed: {name}')
    if binding['model'] == 'act':
        package = Path(next(iter(importlib.util.find_spec('lerobot').submodule_search_locations)))
        for name, relative in {'installed_act': 'policies/act/modeling_act.py',
                               'installed_normalizer': 'processor/normalize_processor.py'}.items():
            require(sha256(package / relative) == binding['sources'][name], f'installed source changed: {name}')
    data = binding['data']
    for name, digest in data['gate_file_sha256'].items():
        require(sha256(Path(data['gate_dir']) / name) == digest, f'data gate changed: {name}')
    for name, spec in data['source_files'].items():
        path = Path(data['source_root']) / name
        require(path.stat().st_size == spec['size'] and sha256(path) == spec['sha256'], f'dataset file changed: {name}')


def audit(root):
    plan_path = REPO / 'docs/pusht-bc-act-training-protocol-v1.json'
    require(sha256(plan_path) == PROTOCOL_SHA, 'frozen protocol changed')
    plan = read_json(plan_path)
    runs, evidence, paths = {}, {}, {}
    for name in ('bc', 'act'):
        run_dir = root / name
        run, status = read_json(run_dir / 'run.json'), read_json(run_dir / 'status.json')
        preflight_path = PREFLIGHT / name / 'report.json'
        require(sha256(preflight_path) == PREFLIGHT_SHA[name] == run['preflight_report_sha256'], 'preflight pin changed')
        preflight = read_json(preflight_path)
        binding = run['binding']
        require(binding == preflight['binding'] and binding['model'] == name, 'run/preflight binding differs')
        require(binding['protocol'] == plan and binding['protocol_file_sha256'] == PROTOCOL_SHA, 'run protocol differs')
        require(run['checkpoint_selection'] == plan['selection'], 'checkpoint selection changed')
        session = Path(status['session'])
        require(session.parent == run_dir / 'sessions' and list(session.parent.iterdir()) == [session],
                'audit v1 requires exactly one uninterrupted session per model')
        report, started = read_json(session / 'report.json'), read_json(session / 'started.json')
        require(report == status and status['status'] == 'completed', 'final report/status incomplete or disagree')
        require(status['step'] == plan['steps'] == status['total_steps'] and status['resume_from'] is None,
                'incomplete or resumed run')
        require(started['step'] == 0 and started['resume_from'] is None and started['model'] == name,
                'session did not start at zero')
        require(status['environment_steps'] == 0 and status['benchmark_score_claim_allowed'] is False,
                'training/benchmark boundary changed')
        final_path = str(run_dir / 'checkpoints' / f"step_{plan['steps']:06d}")
        require(status['latest_checkpoint'] == status['selected_final_checkpoint'] == final_path,
                'non-final checkpoint selected')
        finite(status['session_seconds'], 'session time', positive=True)
        verify_current_inputs(binding)
        print(f'{name}: binding verified; hashing all checkpoint payloads', flush=True)
        checkpoints = verify_checkpoints(run_dir, binding, preflight['model_state_sha256'])
        paths[name] = session / 'metrics.jsonl'
        runs[name] = binding
        evidence[name] = {'run_root': str(run_dir), 'session': str(session), 'status': status,
                          'parameter_count': preflight['parameter_count'], 'checkpoints': checkpoints,
                          'binding': binding, 'preflight_report_sha256': PREFLIGHT_SHA[name],
                          'run_sha256': sha256(run_dir / 'run.json'), 'status_sha256': sha256(run_dir / 'status.json'),
                          'started_sha256': sha256(session / 'started.json'), 'report_sha256': sha256(session / 'report.json')}
    require(runs['bc']['data'] == runs['act']['data'], 'BC/ACT data bindings differ')
    require(runs['bc']['package_versions'] == runs['act']['package_versions'], 'BC/ACT runtime differs')
    print('replaying all 100000 CPU sampler batches and auditing both metric streams', flush=True)
    metrics, anchor_hash = scan_metrics(paths, plan, sampler_hashes(runs['bc'], plan))
    for name in evidence:
        evidence[name]['metrics'] = metrics[name]
        require(metrics[name]['sum_update_seconds'] <= evidence[name]['status']['session_seconds'], 'session time below update sum')
    changes = {}
    for key in ('frame_mean_mae', 'episode_macro_mae', 'rmse'):
        bc, act = (metrics[name]['final_validation'][key] for name in ('bc', 'act'))
        changes[key] = {'bc': bc, 'act': act, 'act_minus_bc': act - bc,
                        'relative_percent_vs_bc': (act - bc) / bc * 100 if bc else None}
    return {'models': evidence, 'final_validation_changes': changes, 'protocol_sha256': PROTOCOL_SHA,
            'all_100000_anchor_batches_exactly_replayed': True, 'anchor_sequence_sha256': anchor_hash,
            'all_checkpoint_payloads_hash_verified': True,
            'final_checkpoint_loading_and_prediction_verified': False,
            'training_resume_continuation_verified': False,
            'closed_loop_benchmark_available': False,
            'observations': 'Public image/state training only. Source bytes checked; no new image decoding or human visual review.',
            'limits': ['single-seed pilot', 'final first-action open-loop metrics, not task success',
                       'BC MSE and ACT L1+10KL are not comparable loss values',
                       'equal updates do not equal FLOPs, losses or future-target counts',
                       'official pretrained DP is an external reference, not controlled same-data training',
                       'no robustness, guidewire, formal-data or real-system claims']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=REPO / 'simulation_output/pusht_bc_act_training_audit_v1/report.json')
    args = parser.parse_args(argv)
    require(not args.out.exists(), 'refusing to overwrite an audit report')
    start = time.monotonic()
    report = {'schema': 'pusht_bc_act_completed_training_audit_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'audit_source_sha256': sha256(__file__), 'audit_optimizer_steps': 0, 'audit_environment_steps': 0,
              'checkpoint_payloads_unpickled': False, 'benchmark_score_claim_allowed': False}
    try:
        report.update(audit(args.root))
        report['status'] = 'passed'
    except Exception as exc:
        report.update({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()})
    report['audit_runtime_seconds'] = time.monotonic() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'out': str(args.out), 'error': report.get('error'),
                      'final_validation_changes': report.get('final_validation_changes')}, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
