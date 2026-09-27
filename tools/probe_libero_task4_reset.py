"""Task4 reset-only tracing; no policy, corruption, replay, or score.

Observe original reset return values without extra rendering, forwarding or
observation refresh. Repeated resets and fresh-environment controls are NOT a
reproduction of the preceding policy trajectory in the failed screen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import time
import traceback

import numpy as np

from run_pi05_libero_low_contrast import Hooks, digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rng_components(torch):
    states = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": bytes(torch.random.get_rng_state().tolist()),
              "torch_cuda": [bytes(v.tolist()) for v in torch.cuda.get_rng_state_all()]}
    return {k: hashlib.sha256(pickle.dumps(v, protocol=4)).hexdigest() for k, v in states.items()}


def leaves(value, prefix=""):
    """Copy numeric/string leaves only; never walk arbitrary runtime objects."""
    if isinstance(value, dict):
        for key in sorted(value):
            yield from leaves(value[key], f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return
        yield prefix, value.copy()
    elif isinstance(value, (np.generic, bool, int, float, str)):
        yield prefix, np.asarray(value).copy()
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from leaves(child, f"{prefix}/{i}")


def numeric_attrs(obj):
    return {k: v for k, v in vars(obj).items()
            if isinstance(v, (np.ndarray, np.generic, bool, int, float, str))}


def inventory(roots):
    return {str(p): sha(p) for root in roots for p in sorted(root.rglob("*")) if p.is_file()}


def capture(base, raw, formatted, stage, root, torch):
    """Copy only; no sim.forward/get_observations/render/controller.update calls."""
    from PIL import Image

    before = rng_components(torch)
    native = base._env.env
    data, model = native.sim.data, native.sim.model
    groups = {"raw": raw, "formatted": formatted}
    data_keys = ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "mocap_pos",
                 "mocap_quat", "qfrc_applied", "xfrc_applied", "time", "body_xpos", "body_xquat")
    model_keys = ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_rgba",
                  "site_pos", "site_quat", "cam_pos", "cam_quat", "light_pos")
    groups["sim_data"] = {k: getattr(data, k) for k in data_keys if hasattr(data, k)}
    groups["sim_model"] = {k: getattr(model, k) for k in model_keys if hasattr(model, k)}
    groups["controller"] = numeric_attrs(native.robots[0].controller)
    groups["gripper"] = numeric_attrs(native.robots[0].gripper)
    groups["env_scalars"] = numeric_attrs(native)
    groups["obs_cache"] = vars(native).get("_obs_cache", {})
    groups["observables"] = {k: numeric_attrs(v) for k, v in vars(native).get("_observables", {}).items()}
    local_rng = {}
    for owner, obj in (("base", base), ("native", native)):
        for key, value in vars(obj).items():
            if isinstance(value, np.random.Generator):
                local_rng[f"{owner}/{key}"] = digest(value.bit_generator.state)
            elif isinstance(value, np.random.RandomState):
                local_rng[f"{owner}/{key}"] = hashlib.sha256(pickle.dumps(value.get_state(), protocol=4)).hexdigest()
    copies = dict(leaves(groups))
    metadata = {}
    for key, value in copies.items():
        entry = {"dtype": str(value.dtype), "shape": list(value.shape),
                 "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
        if value.size <= 256 and "image" not in key:
            entry["value"] = value.tolist()
        metadata[key] = entry
    np.savez_compressed(root / f"{stage}.npz", **copies)
    if stage in ("after_native_reset", "after_set_init_state", "settle_10"):
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            value = raw[key]
            if value.shape != (256, 256, 3) or value.dtype != np.uint8:
                raise ValueError("Unexpected raw camera boundary")
            Image.fromarray(value).save(root / f"{stage}_{key}.png")
    after = rng_components(torch)
    if before != after:
        raise ValueError("Diagnostic capture consumed global RNG")
    row = {"stage": stage, "rng": before, "local_rng": local_rng, "leaves": metadata,
           "raw_observation_sha256": digest(raw) if raw else None,
           "formatted_observation_sha256": digest(formatted) if formatted else None}
    write_json(root / f"{stage}.json", row)
    return row


def measured_reset(env, label, out, set_seed, torch):
    from robosuite.utils.errors import RandomizationError

    root = out / label
    root.mkdir()
    base = env.envs[0].unwrapped
    wrapper, native = base._env, base._env.env
    if base.num_steps_wait != 10 or base._max_episode_steps != 280 or base.task_id != 4:
        raise ValueError("Native task4 reset parameters differ")
    hooks = Hooks()
    attempts, stages, applied = [], [], []
    originals = {"attempt": native.reset, "reset": wrapper.reset,
                 "set": wrapper.set_init_state, "step": wrapper.step}
    step_count = 0

    def snap(raw, stage, formatted=None):
        stages.append(capture(base, raw, {} if formatted is None else formatted, stage, root, torch))

    def attempt(*args, **kwargs):
        row = {"index": len(attempts), "rng_before": rng_components(torch)}
        attempts.append(row)
        write_json(root / "attempts.json", attempts)
        try:
            result = originals["attempt"](*args, **kwargs)
        except RandomizationError as error:
            row.update(status="RandomizationError", error=str(error), rng_after=rng_components(torch))
            write_json(root / "attempts.json", attempts)
            raise
        row.update(status="returned", rng_after=rng_components(torch))
        write_json(root / "attempts.json", attempts)
        return result

    def reset(*args, **kwargs):
        result = originals["reset"](*args, **kwargs)
        snap(result, "after_native_reset")
        return result

    def set_init(state):
        expected = np.asarray(base._init_states[0])
        if not np.array_equal(state, expected) or state.dtype != expected.dtype:
            raise ValueError("Actual payload is not init0")
        applied.append(hashlib.sha256(state.tobytes()).hexdigest())
        result = originals["set"](state)
        snap(result, "after_set_init_state")
        return result

    def step(action):
        nonlocal step_count
        if not np.array_equal(action, [0, 0, 0, 0, 0, 0, -1]):
            raise ValueError("Non-settling action in reset-only diagnostic")
        result = originals["step"](action)
        step_count += 1
        snap(result[0], f"settle_{step_count:02d}")
        return result

    try:
        hooks.set(native, "reset", attempt)
        hooks.set(wrapper, "reset", reset)
        hooks.set(wrapper, "set_init_state", set_init)
        hooks.set(wrapper, "step", step)
        # Exactly the original explicit pre-rollout seed boundary; construction
        # happens before this reseed, as in the paired runner.
        base.init_state_id = 0
        set_seed(1000)
        before = rng_components(torch)
        observation, _ = env.reset(seed=[1000])
        stages.append(capture(base, {}, observation, "returned_vector_observation", root, torch))
        if len(applied) != 1 or step_count != 10 or len(stages) != 13:
            raise ValueError("Unexpected reset/init/settling schedule")
        result = {"label": label, "seed": 1000, "init_index": 0, "init_payload_sha256": applied[0],
                  "rng_before_reset": before, "attempts": attempts, "stages": stages,
                  "final_raw_sha256": digest(observation), "settling_steps": step_count}
        write_json(root / "reset.json", result)
        return result
    finally:
        hooks.close()


def compare(a, b):
    differences = []
    if len(a["stages"]) != len(b["stages"]):
        raise ValueError("Different stage counts")
    for left, right in zip(a["stages"], b["stages"]):
        if left["stage"] != right["stage"]:
            raise ValueError("Mismatched stages")
        lk, rk = left["leaves"], right["leaves"]
        changed = [k for k in sorted(set(lk) | set(rk)) if lk.get(k) != rk.get(k)]
        rng_changed = [k for k in left["rng"] if left["rng"][k] != right["rng"][k]]
        differences.append({"stage": left["stage"], "changed_leaves": changed,
                            "rng_changed": rng_changed, "local_rng_equal": left["local_rng"] == right["local_rng"],
                            "raw_equal": (left["raw_observation_sha256"] == right["raw_observation_sha256"]
                                          if left["raw_observation_sha256"] is not None else None),
                            "formatted_equal": (left.get("formatted_observation_sha256") == right.get("formatted_observation_sha256")
                                                 if left.get("formatted_observation_sha256") is not None else None)})
    return {"left": a["label"], "right": b["label"],
            "init_equal": a["init_payload_sha256"] == b["init_payload_sha256"],
            "before_rng_equal": a["rng_before_reset"] == b["rng_before_reset"],
            "final_raw_equal": a["final_raw_sha256"] == b["final_raw_sha256"],
            "stages": differences}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    log = (out / "run.log").open("ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    start = time.monotonic()
    write_json(out / "status.json", {"status": "running", "pid": os.getpid()})
    opened = None
    try:
        import run_pi05_libero_b4b as b4b
        module, libero = b4b.dependencies()
        import torch
        from lerobot.envs.configs import LiberoEnv
        import libero.libero.envs.env_wrapper as native_module

        # Initialize CUDA RNG before measured intervals; no model is loaded.
        torch.cuda.get_rng_state_all()
        protocol_path = b4b.ROOT / "docs/libero-spatial-low-contrast-protocol-v1.json"
        protocol = json.loads(protocol_path.read_text())
        for path, expected in protocol["provenance"]["runtime_sources"].items():
            if sha(path) != expected:
                raise ValueError(f"Pinned runtime source drift: {path}")
        protected = [b4b.OUT, b4b.ROOT / "simulation_output/pi05_libero_low_contrast_check_v1",
                     b4b.ROOT / "simulation_output/pi05_libero_low_contrast_screen_v1"]
        before_files = inventory(protected)
        cfg = LiberoEnv(task="libero_spatial", task_ids=[4], episode_length=280,
                        init_states=True, max_parallel_tasks=1, control_mode="relative")
        write_json(out / "manifest.json", {"probe_sha256": sha(__file__),
                   "dependencies": {str(Path(m.__file__)): sha(m.__file__) for m in (b4b, module, libero, native_module)},
                   "low_contrast_runner_sha256": sha(Path(__file__).with_name("run_pi05_libero_low_contrast.py")),
                   "protocol_sha256": sha(protocol_path), "env_config": asdict(cfg),
                   "protected_files": before_files, "policy_loaded": False, "optimizer_steps": 0,
                   "intervention": "none; original reset observer hooks only",
                   "schedule": ["shared_0", "shared_1", "shared_2", "fresh_1", "fresh_2"]})
        results = []
        for group, count in (("shared", 3), ("fresh_1", 1), ("fresh_2", 1)):
            module.set_seed(1000)
            opened = module.make_env(cfg, n_envs=1, use_async_envs=False)
            env = opened["libero_spatial"][4]
            for i in range(count):
                label = f"shared_{i}" if group == "shared" else group
                result = measured_reset(env, label, out, module.set_seed, torch)
                results.append(result)
                print(f"RESET_COMPLETE {label} attempts={len(result['attempts'])} raw={result['final_raw_sha256']}", flush=True)
            module.close_envs(opened)
            opened = None
        comparisons = [compare(results[0], other) for other in results[1:]]
        if before_files != inventory(protected):
            raise ValueError("Protected previous evidence changed")
        report = {"status": "completed", "task_id": 4, "reset_count": 5, "policy_steps": 0,
                  "measured_settling_steps_total": 50, "constructor_steps_counted": False,
                  "optimizer_steps": 0, "policy_loaded": False,
                  "comparisons": comparisons, "reset_attempts": {r["label"]: r["attempts"] for r in results},
                  "protected_files_unchanged": len(before_files), "runtime_seconds": time.monotonic() - start,
                  "visual_status": "not_viewed", "score_allowed": False,
                  "limitation": "No preceding policy trajectory or terminal autoreset replay; no claim screen fixed"}
        write_json(out / "report.json", report)
        write_json(out / "status.json", {"status": "completed", "report_sha256": sha(out / "report.json")})
    except BaseException:
        write_json(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        traceback.print_exc()
        raise
    finally:
        if opened is not None:
            module.close_envs(opened)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        parser.error("Use a fresh alphanumeric attempt suffix")
    run(Path(f"/home/zsw/project_2026/simulation_output/libero_task4_reset_probe_{args.attempt}"))
