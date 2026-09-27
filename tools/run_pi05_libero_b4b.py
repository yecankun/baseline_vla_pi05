"""User-run B4b entrypoint with explicit episode initialization and durable logs.

The installed LeRobot terminal step resets its environment internally. Select
the frozen init state again before each official rollout, retaining the policy,
processors, action semantics, physics, and global policy RNG behavior.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import traceback


ROOT = Path('/home/zsw/project_2026')
PROTOCOL = ROOT / 'docs/libero-spatial-score-protocol-v1.json'
CHECKPOINT = Path('/home/zsw/models/project_2026/pi05_libero_finetuned_8e174154')
OUT = ROOT / 'simulation_output/pi05_libero_spatial_b4b_v1'
PREFLIGHT = ROOT / 'simulation_output/pi05_libero_spatial_b4b_preflight_v1.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def select_init_state(env, seeds):
    """Called before official rollout/reset; counters never define the schedule."""
    if env.num_envs != 1 or len(seeds or []) != 1:
        raise ValueError('B4b requires one synchronous environment and one explicit seed')
    base = env.envs[0].unwrapped
    seed = int(seeds[0])
    index = seed - 1000
    if getattr(env, '_b4b_suite', None) != 'libero_spatial' or base.task_id not in range(10):
        raise ValueError('Unexpected B4b suite/task')
    if index not in range(10) or not base.init_states or len(base._init_states) < 10:
        raise ValueError('Seed or init state outside frozen B4b schedule')
    if base._reset_stride != 1 or base._max_episode_steps != 280:
        raise ValueError('Unexpected reset stride/horizon')
    base.init_state_id = index
    return {'task_id': int(base.task_id), 'seed': seed, 'init_state_index': index}


def dependencies():
    os.chdir(ROOT)
    os.environ['LIBERO_CONFIG_PATH'] = str(ROOT / 'simulation_output/libero_runtime_config_v1')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from lerobot.scripts import lerobot_eval
    from lerobot.envs import libero
    return lerobot_eval, libero


def resource_checks():
    p = json.loads(PROTOCOL.read_text())
    route = p['checkpoint_routes']['pi05_primary']
    stage = p['evaluation_stages']['b4b_full_spatial_score']
    expected = {'task_ids': list(range(10)), 'total_episodes': 100,
                'episodes_per_task': 10, 'init_state_indices_per_task': list(range(10)),
                'episode_seeds_per_task': list(range(1000, 1010)),
                'batch_size': 1, 'max_parallel_tasks': 1, 'max_episode_steps': 280}
    for key, value in expected.items():
        if stage[key] != value:
            raise ValueError(f'Frozen protocol differs: {key}')
    versions = {k: importlib.metadata.version(k) for k in
                ('lerobot', 'hf-libero', 'transformers', 'tokenizers')}
    if any(value != p['software_contract'][key] for key, value in versions.items()):
        raise ValueError(f'Runtime version drift: {versions}')
    files = ['model.safetensors', 'config.json', 'policy_preprocessor.json',
             'policy_postprocessor.json',
             'policy_preprocessor_step_2_normalizer_processor.safetensors',
             'policy_postprocessor_step_0_unnormalizer_processor.safetensors']
    hashes = {name: sha256(CHECKPOINT / name) for name in files}
    if hashes['model.safetensors'] != route['model_sha256']:
        raise ValueError('Checkpoint differs from protocol')
    if route['n_action_steps_override'] != 10:
        raise ValueError('Action execution horizon changed')
    return {'versions': versions, 'protocol_sha256': sha256(PROTOCOL),
            'checkpoint_resolved_path': str(CHECKPOINT.resolve()),
            'checkpoint_hashes': hashes, 'wrapper_sha256': sha256(__file__)}


def preflight():
    """Real environment reset checks only; no policy construction or rollout."""
    module, _ = dependencies()
    metadata = resource_checks()
    from lerobot.envs.configs import LiberoEnv
    cfg = LiberoEnv(task='libero_spatial', task_ids=[0], episode_length=280,
                    init_states=True, max_parallel_tasks=1, control_mode='relative')
    envs = module.make_env(cfg, n_envs=1, use_async_envs=False)
    env = envs['libero_spatial'][0]
    env._b4b_suite = 'libero_spatial'
    base = env.envs[0].unwrapped
    import numpy as np
    calls = []
    original = base._env.set_init_state

    def record_set(state):
        matching = [i for i in range(10) if np.array_equal(np.asarray(state),
                                                       np.asarray(base._init_states[i]))]
        calls.append(matching)
        return original(state)

    base._env.set_init_state = record_set
    try:
        for seed in (1000, 1001, 1001):
            select_init_state(env, [seed])
            before = len(calls)
            observation, _ = env.reset(seed=[seed])
            if calls[before:] != [[seed - 1000]]:
                raise AssertionError(f'Wrong state actually applied: {calls[before:]}')
            if sorted(observation['pixels']) != ['image', 'image2']:
                raise AssertionError('Two native camera views missing')
            # Reproduce the extra unseeded reset inside upstream terminal step.
            base.reset()
        for bad_seed in (999, 1010):
            try:
                select_init_state(env, [bad_seed])
            except ValueError:
                pass
            else:
                raise AssertionError('Out-of-protocol seed accepted')
    finally:
        module.close_envs(envs)
    metadata.update(status='passed', actual_init_state_indices=calls,
                    explicit_episode_indices=[0, 1, 1],
                    automatic_intervening_indices=[1, 2, 2],
                    policy_loaded=False, policy_inference=False,
                    optimizer_steps=0, b4b_started=False)
    write_json(PREFLIGHT, metadata)
    print(f'preflight=passed actual_set_init_state_calls={calls} policy_loaded=false b4b_started=false')


def run():
    # mkdir is an atomic no-overwrite guard. Failed attempts remain inspectable.
    OUT.mkdir(parents=False, exist_ok=False)
    sys.stdout.flush()
    sys.stderr.flush()
    log = (OUT / 'run.log').open('ab', buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    write_json(OUT / 'status.json', {'status': 'running', 'pid': os.getpid()})
    events = []
    try:
        module, libero = dependencies()
        metadata = resource_checks()
        metadata['runtime_sources'] = {str(Path(m.__file__)): sha256(m.__file__)
                                       for m in (module, libero)}
        args = [f'--policy.path={CHECKPOINT}', '--policy.n_action_steps=10',
                '--env.type=libero', '--env.task=libero_spatial',
                '--env.task_ids=[0,1,2,3,4,5,6,7,8,9]', '--env.episode_length=280',
                '--env.control_mode=relative', '--env.init_states=true',
                '--env.max_parallel_tasks=1', '--eval.batch_size=1',
                '--eval.use_async_envs=false', '--eval.n_episodes=10',
                '--seed=1000', f'--output_dir={OUT}']
        metadata['official_eval_argv'] = args
        metadata['policy_rng'] = 'unchanged upstream: global seed 1000, not reseeded per episode'
        metadata['init_state_selection'] = 'explicit seed minus 1000 before every official rollout'
        write_json(OUT / 'run_manifest.json', metadata)

        original_make_env = module.make_env
        original_make_policy = module.make_policy
        original_rollout = module.rollout
        original_reset = libero.LiberoEnv.reset

        def emit(event):
            events.append(event)
            with (OUT / 'episode_trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event) + '\n')

        def reset_traced(self, seed=None, **kwargs):
            self._b4b_reset_seed = None if seed is None else int(seed)
            self._b4b_applied_index = None
            result = original_reset(self, seed=seed, **kwargs)
            index = getattr(self, '_b4b_applied_index', None)
            # Constructor resets can precede installation of the payload hook.
            if index is None:
                if seed is not None:
                    raise AssertionError('Explicit reset has no applied-state payload evidence')
                return result
            emit({'kind': 'reset', 'task_id': int(self.task_id),
                  'seed': None if seed is None else int(seed),
                  'applied_init_state_index': index,
                  'reset_kind': 'automatic' if seed is None else 'episode_start'})
            return result

        def make_env_traced(cfg, *pos, **kw):
            write_json(OUT / 'effective_env_config.json', asdict(cfg))
            envs = original_make_env(cfg, *pos, **kw)
            if set(envs) != {'libero_spatial'} or set(envs['libero_spatial']) != set(range(10)):
                raise ValueError('Effective environment suite differs from B4b')
            import numpy as np
            for task_id, env in envs['libero_spatial'].items():
                env._b4b_suite = 'libero_spatial'
                base = env.envs[0].unwrapped
                original_set = base._env.set_init_state
                index_by_hash = {hashlib.sha256(np.asarray(s).tobytes()).hexdigest(): i
                                 for i, s in enumerate(base._init_states)}
                if len(index_by_hash) != len(base._init_states):
                    raise ValueError('Duplicate init-state payloads cannot establish unique IDs')

                def applied(state, base=base, original_set=original_set,
                            index_by_hash=index_by_hash):
                    digest = hashlib.sha256(np.asarray(state).tobytes()).hexdigest()
                    if digest not in index_by_hash:
                        raise ValueError('Undeclared init-state payload')
                    result = original_set(state)
                    base._b4b_applied_index = index_by_hash[digest]
                    emit({'kind': 'state_applied', 'task_id': int(base.task_id),
                          'seed': getattr(base, '_b4b_reset_seed', None),
                          'init_state_index': index_by_hash[digest], 'state_sha256': digest})
                    return result

                base._env.set_init_state = applied
            return envs

        def make_policy_traced(*pos, **kw):
            policy = original_make_policy(*pos, **kw)
            write_json(OUT / 'effective_policy_config.json', asdict(policy.config))
            return policy

        def rollout_traced(*pos, **kw):
            env = kw.get('env', pos[0] if pos else None)
            if pos or 'seeds' not in kw:
                raise ValueError('Upstream rollout calling convention changed')
            selected = select_init_state(env, kw['seeds'])
            before = len(events)
            result = original_rollout(*pos, **kw)
            starts = [e for e in events[before:] if e['kind'] == 'reset'
                      and e['reset_kind'] == 'episode_start']
            if len(starts) != 1 or starts[0]['seed'] != selected['seed'] or starts[0][
                    'applied_init_state_index'] != selected['init_state_index']:
                raise AssertionError('Executed reset does not match selected episode')
            action = result['action']
            import torch
            if not torch.isfinite(action).all().item():
                raise ValueError('Nonfinite action in rollout')
            emit({'kind': 'episode_complete', **selected,
                  'success': bool(result['success'].any().item()),
                  'steps': int(result['done'].shape[1]),
                  'action_min': float(action.min().item()),
                  'action_max': float(action.max().item())})
            return result

        libero.LiberoEnv.reset = reset_traced
        module.make_env = make_env_traced
        module.make_policy = make_policy_traced
        module.rollout = rollout_traced
        sys.argv = [sys.argv[0], *args]
        module.main()
        info = json.loads((OUT / 'eval_info.json').read_text())
        completed = [e for e in events if e['kind'] == 'episode_complete']
        if len(completed) != 100 or len(info['per_task']) != 10:
            raise AssertionError('Incomplete ten-task evaluation')
        for task_id in range(10):
            rows = [e for e in completed if e['task_id'] == task_id]
            tasks = [x for x in info['per_task'] if x['task_id'] == task_id]
            if len(tasks) != 1 or [(r['seed'], r['init_state_index']) for r in rows] != [
                    (1000 + i, i) for i in range(10)]:
                raise AssertionError(f'Incomplete or duplicated task schedule: {task_id}')
            if [r['success'] for r in rows] != tasks[0]['metrics']['successes']:
                raise AssertionError('Trace and aggregate success flags disagree')
        if info['overall']['n_episodes'] != 100:
            raise AssertionError('Aggregate episode count differs')
        pc = sum(r['success'] for r in completed)  # 100 episodes -> same percent
        if abs(pc - info['overall']['pc_success']) > 1e-6:
            raise AssertionError('Aggregate success arithmetic differs')
        write_json(OUT / 'status.json', {'status': 'completed', 'exit_code': 0,
                   'episodes': 100, 'success_rate_percent': pc, 'schedule_verified': True,
                   'eval_info_sha256': sha256(OUT / 'eval_info.json')})
        print(f'B4B_COMPLETE episodes=100 success_rate_percent={pc}', flush=True)
    except BaseException:
        traceback.print_exc()
        write_json(OUT / 'status.json', {'status': 'failed', 'exit_code': 1,
                   'completed_episode_count': sum(e['kind'] == 'episode_complete' for e in events),
                   'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--attempt', default='v1', help='New output suffix for a full rerun; never resumes episodes')
    options = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', options.attempt):
        parser.error('--attempt must contain only letters, digits, underscores, or hyphens')
    OUT = ROOT / f'simulation_output/pi05_libero_spatial_b4b_{options.attempt}'
    preflight() if options.check else run()
