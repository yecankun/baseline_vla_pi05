"""Native DP proposals from the same queued observation, without changing the prior.

No environment, targets, training, clipping or learned selection lives here.
Reference sampling consumes its normal RNG; additional DDPM samples use isolated
streams (including every denoising-step noise draw), not just new initial noise.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


def copy_queues(policy):
    return {key: deque((value.clone() for value in queue), maxlen=queue.maxlen)
            for key, queue in policy._queues.items()}


def restore_queues(policy, queues):
    policy._queues = {key: deque((value.clone() for value in queue), maxlen=queue.maxlen)
                      for key, queue in queues.items()}


def rng_state():
    return (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone())


def restore_rng(state):
    torch.random.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def same_rng(left, right):
    return all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def same_queues(left, right):
    return (left.keys() == right.keys()
            and all(len(left[k]) == len(right[k])
                    and all(torch.equal(a, b) for a, b in zip(left[k], right[k], strict=True))
                    for k in left))


@torch.inference_mode()
def propose(policy, batch, to_native, *, alternative_seeds):
    """Advance normal select_action ONCE; leave its reference queue/RNG intact.

    Return normalized/native [K,8,2], validity and the post-reference state.
    This function must be called before executing an action at a queue boundary.
    The caller executes native[0,0], NOT select_action on the same frame again.
    """
    if policy.training or len(policy._queues["action"]):
        raise ValueError("requires a frozen policy at an empty action-queue boundary")
    if len(alternative_seeds) != 4 or len(set(alternative_seeds)) != 4:
        raise ValueError("exactly four distinct, predeclared DDPM streams required")
    first = policy.select_action(batch)
    normalized = [torch.stack([first, *policy._queues["action"]], dim=1)[0]]
    if normalized[0].shape != (8, 2):
        raise ValueError("reference must preserve official eight-action chunk")
    reference_rng, reference_queues = rng_state(), copy_queues(policy)
    history = {key: torch.stack(list(queue), dim=1)
               for key, queue in policy._queues.items() if key != "action"}
    if set(history) != {"observation.state", "observation.images"}:
        raise ValueError("only observed XY and image history may condition DP")
    device = next(policy.parameters()).device.index
    for seed in alternative_seeds:
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(seed)
            normalized.append(policy.diffusion.generate_actions(history)[0])
    if not same_rng(reference_rng, rng_state()) or not same_queues(reference_queues, policy._queues):
        raise ValueError("extra proposals changed reference RNG or observation/action queues")
    normalized = torch.stack(normalized)
    # Use the exact existing per-action postprocessors, not a new inverse formula.
    native = np.stack([np.stack([to_native(action[None])[0] for action in chunk])
                       for chunk in normalized])
    finite = np.isfinite(native).all(axis=(1, 2))
    if not finite.all():
        raise ValueError("nonfinite proposal; do not silently replace it")
    valid = ((native >= 0) & (native <= 512)).all(axis=(1, 2))
    return {"normalized": normalized, "native": native, "valid": valid,
            "reference_rng": reference_rng, "reference_queues": reference_queues,
            "history": {k: v.clone() for k, v in history.items()}}
