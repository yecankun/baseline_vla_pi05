"""Pinned final BC/ACT inference, with episode-scoped ACT action queues.

Only the audited, user-owned step100000 checkpoints may be loaded. The frozen
training loader restores model/optimizer/RNG once; the optimizer is then dropped.
This module exposes no training or optimizer-update entrypoint.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

if __package__:
    from . import audit_pusht_bc_act_training as audit_tools
    from . import pusht_bc_act_checkpoint as checkpoint
    from . import pusht_bc_act_models as models
else:
    import audit_pusht_bc_act_training as audit_tools
    import pusht_bc_act_checkpoint as checkpoint
    import pusht_bc_act_models as models

REPO = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO / 'simulation_output/pusht_bc_act_training_audit_v1/report.json'
AUDIT_SHA = 'e5f005742df350758ffca19f053f8a5df6c47718e8a96ecd674189196393276a'
AUDITOR_SHA = 'd053ba02543c84c14ab30991cc3b4aa11e1d54b9302eebfaedc5a7ef89379a43'
require = audit_tools.require
sha256 = audit_tools.sha256


def training_audit():
    require(sha256(AUDIT_PATH) == AUDIT_SHA, 'training audit SHA256 differs')
    require(sha256(audit_tools.__file__) == AUDITOR_SHA, 'audit implementation changed')
    value = audit_tools.read_json(AUDIT_PATH)
    require(value['status'] == 'passed' and value['all_checkpoint_payloads_hash_verified'] is True
            and value['all_100000_anchor_batches_exactly_replayed'] is True, 'training audit is incomplete')
    return value


class InferencePolicy:
    """select_action preserves queues; predict is explicitly fresh-frame validation."""
    def __init__(self, runtime):
        self.name, self.device = runtime.name, runtime.device
        self.policy, self.stats = runtime.policy, runtime.stats
        self.pre, self.post = runtime.pre, runtime.post
        self.policy.eval().requires_grad_(False)
        self.reset()

    def reset(self):
        self.policy.reset()
        for processor in (self.pre, self.post):
            if processor is not None:
                processor.reset()
        self.action_calls = 0
        self.last_queue_trace = None

    @torch.inference_mode()
    def select_action(self, observation):
        models.validate_observation(observation)
        require(all(value.dtype == torch.float32 for value in observation.values()), 'float32 inputs required')
        require(not any(module.training for module in self.policy.modules()), 'policy must remain in eval mode')
        if self.name == 'bc':
            state_stats, action_stats = self.stats[models.STATE], self.stats['action']
            obs = {models.IMAGE: observation[models.IMAGE],
                   models.STATE: (observation[models.STATE] - state_stats['mean']) / (state_stats['std'] + 1e-8)}
            result = self.policy.select_action(obs) * action_stats['std'] + action_stats['mean']
            self.last_queue_trace = {'before': 0, 'after': 0, 'chunk_generated': True}
        elif self.name == 'act':
            require(self.policy.config.chunk_size == 16 and self.policy.config.n_action_steps == 8
                    and self.policy.config.temporal_ensemble_coeff is None, 'ACT execution config drift')
            before = len(self.policy._action_queue)
            require(before == (-self.action_calls) % 8, 'ACT queue was reset or consumed externally')
            # No reset here. The official policy owns its unmodified chunk16/execute8 queue.
            result = self.post(self.policy.select_action(self.pre(models.act_batch(observation)))).to(self.device)
            after = len(self.policy._action_queue)
            require(after == (7 if before == 0 else before - 1), 'ACT queue consumption differs')
            self.last_queue_trace = {'before': before, 'after': after, 'chunk_generated': before == 0}
        else:
            raise ValueError('unknown model')
        require(result.shape == (observation[models.STATE].shape[0], 2)
                and result.dtype == torch.float32 and bool(torch.isfinite(result).all()), 'invalid native action')
        self.action_calls += 1
        return result  # Native, unbounded: never clamp/clip.

    def predict(self, batch):
        """Training-metric compatibility ONLY; never call this in a rollout loop."""
        self.reset()
        return self.select_action(batch['observation'])


def observation_from_native(raw, preprocess, device='cuda'):
    require(isinstance(raw, dict) and set(raw) == {'pixels', 'agent_pos'}, 'unexpected native observation fields')
    require(isinstance(raw['pixels'], np.ndarray) and raw['pixels'].shape == (96, 96, 3)
            and raw['pixels'].dtype == np.uint8, 'native RGB must be uint8 HWC96')
    require(isinstance(raw['agent_pos'], np.ndarray) and raw['agent_pos'].shape == (2,)
            and np.isfinite(raw['agent_pos']).all(), 'native agent position must be finite 2D')
    observation = preprocess(raw)
    require(set(observation) == {models.IMAGE, models.STATE}, 'only image and state may reach policy')
    return {key: value.to(device) for key, value in observation.items()}


def load_final(name, audit=None):
    if __package__:
        from . import train_pusht_bc_act as trainer
    else:
        import train_pusht_bc_act as trainer
    audit = training_audit() if audit is None else audit
    require(name in ('bc', 'act'), 'model must be bc/act')
    require(torch.cuda.is_available(), 'CUDA required; no fallback')
    record = audit['models'][name]
    binding = record['binding']
    audit_tools.verify_current_inputs(binding)
    require(sha256(trainer.PROTOCOL_PATH) == audit['protocol_sha256'], 'frozen protocol changed')
    plan = trainer.protocol()
    require(trainer.binding_for(SimpleNamespace(binding=binding['data']), name, plan) == binding,
            'current runtime/data/source binding differs from training')
    root = Path(record['run_root'])
    require(sha256(root / 'run.json') == record['run_sha256']
            and sha256(root / 'status.json') == record['status_sha256'], 'completed run metadata drift')
    final = record['checkpoints'][-1]
    path = Path(final['path'])
    require(final['step'] == 100000 and path == root / 'checkpoints' / 'step_100000', 'only final step may be selected')
    require(sha256(path / 'manifest.json') == final['manifest_sha256'], 'final manifest differs from audit')
    manifest = audit_tools.read_json(path / 'manifest.json')
    require(manifest['files']['checkpoint.pt'] == {'size': final['payload_bytes'], 'sha256': final['payload_sha256']},
            'payload pin differs from completed audit')
    stats = audit_tools.read_json(Path(binding['data']['gate_dir']) / 'normalization.json')
    trainer.seed_all(plan['seed'])
    runtime = trainer.ModelRuntime(name, stats, plan)
    # Strict local checkpoint helper verifies ownership, paths, bytes, shapes,
    # optimizer parameter order and all RNG before loading; no optimizer.step.
    payload = checkpoint.load_checkpoint(path, runtime.policy, runtime.optimizer, binding, 'cuda')
    require(payload['step'] == 100000 and payload['extra']['kind'] == 'training'
            and payload['extra']['optimizer_steps'] == 100000, 'payload is not completed training')
    saved_steps = [int(state['step'].item()) for state in payload['optimizer']['state'].values()]
    require(saved_steps and set(saved_steps) == {100000}, 'saved optimizer steps are inconsistent')
    state_hash = checkpoint.state_dict_hash(runtime.policy)
    require(state_hash == final['state_dict_sha256'], 'loaded final model hash differs')
    # Do not retain training optimizer moments on the GPU during inference.
    runtime.optimizer.state.clear()
    del runtime.optimizer, payload
    policy = InferencePolicy(runtime)
    require(all(p.device.type == 'cuda' and p.dtype == torch.float32 and not p.requires_grad
                and p.grad is None for p in policy.policy.parameters()), 'inference parameters are not frozen CUDA float32')
    info = {'model': name, 'training_audit_sha256': AUDIT_SHA, 'checkpoint_path': str(path),
            'checkpoint_payload_sha256': final['payload_sha256'], 'checkpoint_manifest_sha256': final['manifest_sha256'],
            'model_state_sha256': state_hash, 'normalization_file_sha256': binding['protocol']['normalization_file_sha256'],
            'training_binding': binding, 'optimizer_retained': False, 'optimizer_steps_executed': 0,
            'saved_optimizer_step': 100000, 'saved_optimizer_states': len(saved_steps),
            'policy_input_keys': [models.IMAGE, models.STATE], 'task_instruction': 'constant Push-T task; no language encoder',
            'action_execution': 'single action' if name == 'bc' else 'official ACT chunk16/execute8; reset only per episode'}
    return policy, info


def verify_validation(policy, data, audit):
    if __package__:
        from . import train_pusht_bc_act as trainer
    else:
        import train_pusht_bc_act as trainer
    record = audit['models'][policy.name]
    require(data.binding == record['binding']['data'], 'validation data binding changed')
    measured = trainer.evaluation(policy, data, record['binding']['protocol'])
    expected = record['metrics']['final_validation']
    differences = {}
    for key in ('frame_mean_mae', 'episode_macro_mae', 'rmse', 'mae_xy'):
        diff = float(np.max(np.abs(np.asarray(measured[key]) - np.asarray(expected[key]))))
        require(diff <= 1e-6, f'loaded final validation differs: {key}, max_abs={diff}')
        differences[key] = diff
    for key in ('frames', 'episodes', 'out_of_bounds_predictions', 'action_clipping', 'checkpoint_selection_allowed'):
        require(measured[key] == expected[key], f'validation count/contract differs: {key}')
    policy.reset()
    return {'passed': True, 'metrics': measured, 'max_abs_differences': differences, 'tolerance': 1e-6,
            'checkpoint_selection_performed': False}


@torch.inference_mode()
def verify_queue_and_reset(policy, observation):
    """Direct chunk comparison and an actual underlying-model forward-call count."""
    if policy.name == 'bc':
        policy.reset()
        first = policy.select_action(observation).clone()
        policy.reset()
        replay = policy.select_action(observation)
        require(torch.equal(first, replay), 'BC reset replay differs')
        policy.reset()
        return {'passed': True, 'reset_replay_exact': True, 'action_queue': 'none; single-action BC'}
    policy.reset()
    chunk = policy.policy.predict_action_chunk(policy.pre(models.act_batch(observation)))
    require(chunk.shape == (1, 16, 2), 'ACT generated chunk shape changed')
    expected = [policy.post(chunk[:, i]).to(policy.device).clone() for i in range(8)]
    calls = []
    handle = policy.policy.model.register_forward_pre_hook(lambda module, args: calls.append(len(args)))
    traces, actions = [], []
    try:
        policy.reset()
        for index in range(9):
            action = policy.select_action(observation).clone()
            actions.append(action)
            traces.append(dict(policy.last_queue_trace))
            require(len(calls) == index // 8 + 1, 'ACT recomputed a chunk at an unexpected step')
            if index < 8:
                require(torch.equal(action, expected[index]), 'ACT queue did not emit the first 8 chunk actions')
        policy.reset()  # Also proves a partially consumed queue is cleared.
        replay = [policy.select_action(observation).clone() for _ in range(9)]
        require(all(torch.equal(a, b) for a, b in zip(actions, replay)), 'ACT partial-queue reset replay differs')
        require(len(calls) == 4, 'ACT replay chunk count differs')
    finally:
        handle.remove()
        policy.reset()
    return {'passed': True, 'reset_replay_exact': True, 'direct_chunk_first8_exact': True,
            'action_calls_per_replay': 9, 'model_forwards_per_replay': 2, 'queue_trace': traces}
