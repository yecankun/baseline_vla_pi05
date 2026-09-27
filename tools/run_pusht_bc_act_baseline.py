"""Clean native Push-T final-checkpoint smoke/benchmark; never train.

Default preflight checks imports/provenance only. Smoke executes full validation
replay plus a fixed 20-step episode and reset replay for each selected model.
The 100-episode benchmark is a separate explicit user-run command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

if __package__:
    from . import run_diffusion_pusht_baseline as common
    from . import pusht_bc_act_inference as inference
else:
    import run_diffusion_pusht_baseline as common
    import pusht_bc_act_inference as inference

ROOT = Path(__file__).resolve().parents[1]
require, sha256 = inference.require, inference.sha256
DP_REPORT = ROOT / 'simulation_output/diffusion_pusht_benchmark_v1/report.json'
DP_REPORT_SHA = 'c1d4c4e582af1116ec41d554c8ae8be76480cee73488acff4471097bafe314b9'
PROTOCOL = {
    'schema': 'pusht_bc_act_final_clean_evaluation_v1',
    'training_audit_sha256': inference.AUDIT_SHA,
    'checkpoint_selection': 'audited final step100000 only',
    'environment': 'gym_pusht/PushT-v0', 'clean_only': True,
    'smoke_seeds': [20260913], 'smoke_max_steps': 20, 'smoke_reset_replays': 1,
    'benchmark_seeds': list(range(100000, 100100)), 'benchmark_max_steps': 300,
    'env_episode_length': 300, 'device': 'cuda', 'precision': 'float32',
    'amp': False, 'tf32': False, 'deterministic_algorithms': True,
    'episode_rng': 'reset python/numpy/torch/cuda to episode seed',
    'raw_observation': ['pixels uint8 HWC96 RGB', 'agent_pos 2D'],
    'policy_observation': ['observation.image', 'observation.state'],
    'normalization': 'same saved train-only state/action statistics and fixed ImageNet image statistics',
    'task': 'constant public Push-T task; no added language encoder',
    'native_action_dim': 2, 'native_action_bounds': [0.0, 512.0], 'action_clipping': False,
    'bc_execution': 'one observation -> one action',
    'act_execution': 'official chunk16 -> execute first8; reset queue only per episode',
    'success_source': 'info.is_success; never reward', 'coverage_source': 'info.coverage; never reward',
    'autoreset': 'step vector.envs[0] directly, stop at first terminated/truncated',
    'dp_comparison': 'same eval task/seeds; official DP is external pretrained reference, not same-data training',
    'retry': 'no auto retry/resume; inspected failure uses explicit fresh attempt',
}


def output_directory(stage, name, attempt=None, root=ROOT):
    require(stage in ('smoke', 'benchmark') and name in ('bc', 'act'), 'invalid stage/model')
    if attempt is not None:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}', attempt) is not None, 'invalid attempt suffix')
    suffix = '_' + attempt if attempt else ''
    return root / 'simulation_output' / f'pusht_bc_act_{stage}_v1{suffix}' / name


def imports():
    import numpy as np
    import torch
    from lerobot.envs.utils import preprocess_observation
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.configs import PushtEnv
    if __package__:
        from .smoke_lerobot_public_envs import _make_environment, _package_versions
    else:
        from smoke_lerobot_public_envs import _make_environment, _package_versions
    return locals()


def source_binding(runtime):
    require(sha256(DP_REPORT) == DP_REPORT_SHA, 'frozen DP environment reference changed')
    reference = json.loads(DP_REPORT.read_text())
    require(reference['strict_protocol_pass'] is True, 'DP reference is not complete')
    paths = {'runner': Path(__file__), 'inference': Path(inference.__file__),
             'common_metrics': Path(common.__file__),
             **{key: Path(inspect.getfile(runtime[key])) for key in
                ('_make_environment', 'preprocess_observation', 'make_env_pre_post_processors')}}
    sources = {key: sha256(path) for key, path in paths.items()}
    require(sources['common_metrics'] == reference['source_files']['runner']['sha256'], 'frozen DP helper changed')
    for key in ('_make_environment', 'preprocess_observation', 'make_env_pre_post_processors'):
        require(sources[key] == reference['source_files'][key]['sha256'], f'environment source drift: {key}')
    require(PROTOCOL['benchmark_seeds'] == common.seed_plan('benchmark')
            and PROTOCOL['smoke_seeds'] == common.seed_plan('smoke'), 'shared seed schedule drift')
    return sources, reference


def verify_smoke(path, binding, sources):
    report = json.loads(path.read_text())
    require(report.get('stage') == 'smoke' and report.get('status') == 'completed'
            and report.get('strict_protocol_pass') is True, 'successful strict smoke required')
    require(report.get('protocol_sha256') == common.canonical_hash(PROTOCOL)
            and report.get('checkpoint_binding') == binding and report.get('source_files') == sources,
            'smoke source/protocol/checkpoint binding drift')
    require(report.get('validation_replay', {}).get('passed') is True
            and report.get('queue_check', {}).get('passed') is True
            and report.get('reset_replay_exact') is True
            and report.get('model_state_unchanged') is True, 'smoke load/queue/reset/immutability check incomplete')
    require(common.aggregate(report.get('episodes', []), 'smoke')['strict_protocol_pass'], 'smoke schedule incomplete')
    return report


def observation_hash(raw):
    return hashlib.sha256(raw['pixels'].tobytes() + raw['agent_pos'].tobytes()).hexdigest()


def episode(policy, env, runtime, *, seed, max_steps, output, index, phase,
            snapshots=None, reference_reset=None, execution_counter=None):
    np, torch = runtime['np'], runtime['torch']
    started = time.monotonic()
    common._seed(seed, runtime)
    policy.reset()
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    raw, _reset_info = env.reset(seed=seed)
    # Validate the entire raw allowlist before logging or acting.
    inference.observation_from_native(raw, runtime['preprocess_observation'], policy.device)
    reset_hash = observation_hash(raw)
    if reference_reset is not None:
        require(reset_hash == reference_reset, 'reset observation differs from frozen DP benchmark seed')
    trace = hashlib.sha256()
    trace.update(bytes.fromhex(reset_hash))
    common._append(output / 'per_step.jsonl', {'kind': 'reset', 'phase': phase, 'episode': index,
                    'seed': seed, 'observation_sha256': reset_hash})
    if snapshots is not None:
        snapshots.append({'step': 0, 'frame': common._render(env, np)})
    success, coverages, outside_steps, outside_elements = False, [], 0, 0
    terminated = truncated = False
    for step in range(1, max_steps + 1):
        observation = inference.observation_from_native(raw, runtime['preprocess_observation'], policy.device)
        before_hash = observation_hash(raw)
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            native = policy.select_action(observation)
        action, outside = common.validate_native_action(native, env.action_space.low, env.action_space.high)
        outside_steps += int(outside > 0)
        outside_elements += outside
        action_row = {'kind': 'action', 'phase': phase, 'episode': index, 'seed': seed, 'step': step,
                      'observation_sha256': before_hash, 'native_action': action[0].tolist(),
                      'native_action_out_of_bounds_elements': outside, 'queue': policy.last_queue_trace}
        common._append(output / 'per_step.jsonl', action_row)
        if execution_counter is not None:
            execution_counter['environment_step_calls'] += 1
        raw, reward, term, trunc, info = env.step(action[0])
        if execution_counter is not None:
            execution_counter['environment_steps'] += 1
        terminated, truncated = common._boolean(term, 'terminated'), common._boolean(trunc, 'truncated')
        metrics = common.info_metrics(info, terminal=terminated or truncated)
        numeric_reward = float(common._scalar(reward, 'reward'))
        require(np.isfinite(numeric_reward), 'nonfinite reward')
        coverages.append(metrics['coverage'])
        success = success or metrics['is_success']
        row = {'kind': 'step', 'episode': index, 'seed': seed, 'step': step,
               'observation_sha256': before_hash, 'next_observation_sha256': observation_hash(raw),
               'native_action': action[0].tolist(), 'native_action_out_of_bounds_elements': outside,
               'queue': policy.last_queue_trace, 'reward': numeric_reward,
               'terminated': terminated, 'truncated': truncated, **metrics}
        trace.update(bytes.fromhex(common.canonical_hash(row)))
        common._append(output / 'per_step.jsonl', {**row, 'phase': phase})
        if snapshots is not None and (step in {1, 10, max_steps} or terminated or truncated):
            snapshots.append({'step': step, 'frame': common._render(env, np)})
        if terminated or truncated:
            break
    return {'episode': index, 'seed': seed, 'steps': len(coverages), 'is_success': success,
            'max_coverage': max(coverages), 'terminated': terminated, 'truncated': truncated,
            'stopped_at_stage_cap': not (terminated or truncated), 'native_action_out_of_bounds_steps': outside_steps,
            'native_action_out_of_bounds_elements': outside_elements, 'seconds': time.monotonic() - started,
            'reset_observation_sha256': reset_hash, 'trace_sha256': trace.hexdigest()}


def sheet(name, snapshots, path):
    from PIL import Image, ImageDraw, ImageFont
    candidates = [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'), Path('C:/Windows/Fonts/msyh.ttc')]
    font_path = next((p for p in candidates if p.is_file()), None)
    require(font_path is not None, 'Chinese font missing')
    title_font, font = (ImageFont.truetype(str(font_path), size) for size in (24, 20))
    canvas = Image.new('RGB', (768, 90 + ((len(snapshots) + 1) // 2) * 420), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'{name.upper()} / Push-T 最终训练权重', font=title_font, fill='black')
    draw.text((12, 47), '20 步闭环接入检查｜不是完整基准成绩｜本次无训练', font=font, fill='black')
    for i, item in enumerate(snapshots):
        left, top = i % 2 * 384, 90 + i // 2 * 420
        draw.text((left + 10, top + 2), f'步骤 {item["step"]}', font=font, fill='black')
        canvas.paste(Image.fromarray(item['frame']), (left, top + 36))
    canvas.save(path)
    return {'path': str(path), 'sha256': sha256(path), 'visual_status': 'not_viewed', 'font': str(font_path)}


def run(stage, name, output, runtime, audit, smoke_root=None, validation_data=None):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'schema': PROTOCOL['schema'], 'stage': stage, 'model': name, 'protocol': PROTOCOL,
              'protocol_sha256': common.canonical_hash(PROTOCOL), 'status': 'running', 'pid': os.getpid(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'episodes': [],
              'training_started': False, 'optimizer_steps': 0, 'environment_steps': 0, 'environment_step_calls': 0,
              'guidewire_or_real_system_claim_allowed': False, 'benchmark_score_claim_allowed': False}
    common._write(output / 'started.json', report)
    common._write(output / 'status.json', {'status': 'running', 'pid': os.getpid()})
    vector = None
    try:
        sources, dp_reference = source_binding(runtime)
        report['source_files'] = sources
        report['dp_reference_sha256'] = DP_REPORT_SHA
        policy, binding = inference.load_final(name, audit)
        report['checkpoint_binding'] = binding
        if stage == 'smoke':
            require(validation_data is not None, 'smoke requires complete pinned validation data')
            report['validation_replay'] = inference.verify_validation(policy, validation_data, audit)
            probe = validation_data.batch(validation_data.val_indices[:1], policy.device)['observation']
            report['queue_check'] = inference.verify_queue_and_reset(policy, probe)
            print(f'model={name} full_validation_replay=passed queue_check=passed', flush=True)
        else:
            smoke_path = (smoke_root / name / 'report.json') if smoke_root else output_directory('smoke', name) / 'report.json'
            smoke = verify_smoke(smoke_path, binding, sources)
            report['smoke_prerequisite'] = {'path': str(smoke_path), 'sha256': sha256(smoke_path)}
        config, vector = runtime['_make_environment']('pusht', argparse.Namespace(episode_length=300))
        require(len(vector.envs) == 1, 'exactly one synchronous native environment required')
        env = vector.envs[0]
        report.update({'environment_gym_kwargs': config.gym_kwargs,
                       'environment_step_source_sha256': common.canonical_hash(inspect.getsource(type(env.unwrapped).step)),
                       'package_versions': runtime['_package_versions']()})
        common.validate_env_kwargs(config.gym_kwargs)
        common.verify_environment_parity(dp_reference, report)
        if stage == 'benchmark':
            common.verify_environment_parity(smoke, report)
        env_pre, env_post = runtime['make_env_pre_post_processors'](config, getattr(policy.policy, 'config', None))
        require(not env_pre.steps and not env_post.steps, 'native Push-T environment processors are not identity')
        report['processor_order'] = ['strict native pixels+agent_pos', 'preprocess_observation',
                                    'saved policy normalization', 'episode-scoped policy.select_action',
                                    'saved inverse action normalization', 'raw single env.step; no clipping']
        report['environment_processors_verified_identity'] = True
        common._write(output / 'started.json', report)
        snapshots = []
        resets = {row['seed']: row['reset_observation_sha256'] for row in dp_reference['episodes']}
        for index, seed in enumerate(common.seed_plan(stage)):
            value = episode(policy, env, runtime, seed=seed, max_steps=PROTOCOL[f'{stage}_max_steps'],
                            output=output, index=index, phase='primary',
                            snapshots=snapshots if stage == 'smoke' else None,
                            reference_reset=resets[seed] if stage == 'benchmark' else None, execution_counter=report)
            report['episodes'].append(value)
            common._append(output / 'per_episode.jsonl', value)
            common._write(output / 'status.json', {'status': 'running', 'pid': os.getpid(),
                           'model': name, 'episodes_completed': len(report['episodes']), 'seed': seed})
            print(f'model={name} stage={stage} episode={index + 1}/{len(common.seed_plan(stage))} '
                  f'seed={seed} steps={value["steps"]} success={value["is_success"]} seconds={value["seconds"]:.3f}', flush=True)
            if stage == 'smoke':
                replay = episode(policy, env, runtime, seed=seed, max_steps=PROTOCOL['smoke_max_steps'],
                                 output=output, index=index, phase='reset_replay', reference_reset=value['reset_observation_sha256'],
                                 execution_counter=report)
                require({k: v for k, v in replay.items() if k != 'seconds'} ==
                        {k: v for k, v in value.items() if k != 'seconds'}, 'closed-loop reset replay differs')
                report['reset_replay_exact'] = True
                report['reset_replay_episode'] = replay
        report.update(common.aggregate(report['episodes'], stage))
        require(inference.checkpoint.state_dict_hash(policy.policy) == binding['model_state_sha256'], 'inference mutated model state')
        require(all(p.grad is None and not p.requires_grad for p in policy.policy.parameters()), 'inference gradients appeared')
        report['model_state_unchanged'] = True
        if stage == 'smoke':
            report['visual_artifact'] = sheet(name, snapshots, output / 'smoke_sheet_zh.png')
            measured = sum(e['seconds'] for e in report['episodes'])
            report['full_horizon_100_episode_seconds_estimate'] = measured / sum(e['steps'] for e in report['episodes']) * 30000
            report['estimate_caveat'] = '20-step primary throughput includes initial warmup/render; excludes model load and early termination savings'
        report['status'] = 'completed'
    except BaseException as exc:
        report.update({'status': 'failed', 'strict_protocol_pass': False, 'benchmark_score_claim_allowed': False,
                       'error': str(exc), 'error_type': type(exc).__name__, 'traceback': traceback.format_exc(),
                       'partial_trace_preserved': True})
    finally:
        if vector is not None:
            try:
                vector.close()
            except Exception as exc:
                report.update({'status': 'failed', 'strict_protocol_pass': False,
                               'benchmark_score_claim_allowed': False, 'close_error': str(exc)})
        report['runtime_seconds'] = time.monotonic() - started
        common._write(output / 'report.json', report)
        common._write(output / 'status.json', {'status': report['status'], 'pid': os.getpid(),
                       'model': name, 'episodes_completed': len(report['episodes']),
                       'strict_protocol_pass': report.get('strict_protocol_pass', False)})
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'smoke', 'benchmark'), default='preflight')
    parser.add_argument('--model', choices=('bc', 'act', 'both'), default='both')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--attempt', help='Fresh suffix after inspecting a failure; never automatic')
    parser.add_argument('--smoke-root', type=Path, help='Successful same-binding smoke root for benchmark only')
    args = parser.parse_args(argv)
    if args.stage == 'preflight' and (args.execute or args.attempt or args.smoke_root):
        parser.error('preflight does not execute policies or create outputs')
    if args.stage != 'preflight' and not args.execute:
        parser.error('smoke/benchmark require --execute')
    if args.smoke_root and args.stage != 'benchmark':
        parser.error('--smoke-root is benchmark-only')
    if args.attempt:
        output_directory(args.stage, 'bc', args.attempt)
    return args


def main(argv=None):
    args = parse_args(argv)
    names = ('bc', 'act') if args.model == 'both' else (args.model,)
    runtime = imports()
    audit = inference.training_audit()
    sources, _ = source_binding(runtime)
    common.validate_env_kwargs(runtime['PushtEnv'](obs_type='pixels_agent_pos', render_mode='rgb_array', episode_length=300).gym_kwargs)
    if args.stage == 'preflight':
        print(json.dumps({'status': 'passed', 'stage': 'preflight', 'source_files': sources,
                          'protocol': PROTOCOL, 'checkpoint_loaded': False, 'environment_constructed': False,
                          'optimizer_steps': 0}, indent=2))
        return 0
    for name in names:
        require(not output_directory(args.stage, name, args.attempt).exists(), 'output already exists; inspect before any rerun')
    data = None
    if args.stage == 'smoke':
        if __package__:
            from .pusht_bc_act_training_data import load_training_data
        else:
            from pusht_bc_act_training_data import load_training_data
        data = load_training_data()
    strict = True
    for name in names:
        report = run(args.stage, name, output_directory(args.stage, name, args.attempt), runtime, audit,
                     args.smoke_root, data)
        print(json.dumps({'model': name, 'status': report['status'], 'strict_protocol_pass': report.get('strict_protocol_pass'),
                          'error': report.get('error'), 'report': str(output_directory(args.stage, name, args.attempt) / 'report.json')}))
        if report['status'] != 'completed':
            return 1
        strict = strict and report['strict_protocol_pass']
    return 0 if strict else 2


if __name__ == '__main__':
    raise SystemExit(main())
