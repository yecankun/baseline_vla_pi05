"""New paired native-controller diagnostic; never rewrites historical B4b.

--stage check: two full task0 episodes. --stage screen: ten task pairs.
Uses the installed official rollout unchanged, but independently resets each
arm's policy RNG and corrupts only raw image observations before preprocessing.
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
import sys
import time
import traceback

import numpy as np

from pi05_libero_low_contrast_contract import load_protocol, schedule, summarize_pairs
from vla_benchmark_contract import VisualCorruption, validate_policy_observation
from vla_visual_noise_wrapper import EvaluationVisualObservationWrapper


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def digest(value):
    """Typed recursive observation identity, including every non-image leaf."""
    def canonical(v):
        if isinstance(v, dict):
            return {k: canonical(v[k]) for k in sorted(v)}
        if isinstance(v, (list, tuple)):
            return [canonical(x) for x in v]
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            if a.dtype.hasobject:
                raise ValueError("Object arrays cannot prove observation identity")
            return {"dtype": str(a.dtype), "shape": list(a.shape),
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(v, np.generic):
            return v.item()
        return v
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_only_images_changed(before, after):
    if set(before) != set(after) or set(before.get("pixels", {})) != {"image", "image2"}:
        raise ValueError("Raw observation schema changed")
    if set(after["pixels"]) != {"image", "image2"}:
        raise ValueError("Camera keys changed")
    if digest({k: v for k, v in before.items() if k != "pixels"}) != digest(
            {k: v for k, v in after.items() if k != "pixels"}):
        raise ValueError("Corruption modified a non-image observation")
    for value in (before, after):
        validate_policy_observation(value)
        for img in value["pixels"].values():
            if img.shape != (1, 256, 256, 3) or img.dtype != np.uint8:
                raise ValueError("Expected actual uint8 two-view BHWC256 images")


def rng_fingerprint(torch):
    states = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": bytes(torch.random.get_rng_state().tolist()),
              "torch_cuda": [bytes(s.tolist()) for s in torch.cuda.get_rng_state_all()]}
    return hashlib.sha256(pickle.dumps(states, protocol=4)).hexdigest()


class Hooks:
    def __init__(self):
        self.saved = []

    def set(self, obj, key, fn):
        self.saved.append((obj, key, key in vars(obj), vars(obj).get(key)))
        setattr(obj, key, fn)

    def close(self):
        for obj, key, owned, old in reversed(self.saved):
            if owned:
                setattr(obj, key, old)
            else:
                delattr(obj, key)
        self.saved.clear()


def episode(module, env, policy, processors, selected, condition, out, protocol):
    import torch
    from PIL import Image
    import run_pi05_libero_b4b as b4b

    out.mkdir(exist_ok=False)
    hooks = Hooks()
    base = env.envs[0].unwrapped
    applied, observations, native_actions = [], [], []
    phase = {"reset_seed": None, "policy_step": False}
    first_rng = []
    policy_inputs = []
    controller_evidence = {}
    original_preprocess = module.preprocess_observation
    original_reset = base.reset
    original_step = base.step
    original_inner_step = base._env.step
    original_set = base._env.set_init_state
    original_select = policy.select_action
    wrapper = EvaluationVisualObservationWrapper(
        env, image_paths=("pixels.image", "pixels.image2"),
        corruption=VisualCorruption("low_contrast", 2), seed=20260911,
        episode_key=f"spatial/task-{selected['task_id']}/seed-{selected['seed']}/init-0",
        verify_determinism=True)
    snapshots = {}
    expected_state = np.asarray(base._init_states[0])

    def set_state(state):
        value = array(state)
        if phase["reset_seed"] is not None:
            if not np.array_equal(value, expected_state) or value.dtype != expected_state.dtype:
                raise ValueError("Actual init payload is not the prescribed init0")
            applied.append(hashlib.sha256(value.tobytes()).hexdigest())
        return original_set(state)

    def reset(seed=None, **kwargs):
        old = phase.copy()
        phase.update(reset_seed=seed, policy_step=False)
        try:
            result = original_reset(seed=seed, **kwargs)
            if seed is not None:
                ctrl = base._env.robots[0].controller
                if (type(ctrl).__name__ != "OperationalSpaceController" or not ctrl.use_delta
                        or ctrl.impedance_mode != "fixed" or ctrl.control_dim != 6):
                    raise ValueError("Native OSC contract differs")
                expected_limits = {"input_min": [-1.] * 6, "input_max": [1.] * 6,
                                   "output_min": [-.05] * 3 + [-.5] * 3,
                                   "output_max": [.05] * 3 + [.5] * 3}
                for key, expected in expected_limits.items():
                    if not np.array_equal(getattr(ctrl, key), expected):
                        raise ValueError(f"Native controller limit differs: {key}")
                gripper = base._env.robots[0].gripper
                if type(gripper).__name__ != "PandaGripper" or gripper.dof != 1 or gripper.speed != .01:
                    raise ValueError("Native gripper contract differs")
                if base.num_steps_wait != 10:
                    raise ValueError("Settling window differs")
                controller_evidence.update(class_name=type(ctrl).__name__, use_delta=bool(ctrl.use_delta),
                                           impedance_mode=ctrl.impedance_mode, control_dim=ctrl.control_dim,
                                           limits={k: array(getattr(ctrl, k)).tolist() for k in expected_limits},
                                           gripper_class=type(gripper).__name__, gripper_dof=gripper.dof,
                                           gripper_speed=gripper.speed, settling_steps=base.num_steps_wait)
            return result
        finally:
            phase.update(old)

    def native_step(action):
        phase["policy_step"] = True
        before = len(native_actions)
        copy = array(action)
        try:
            result = original_step(action)
        finally:
            phase["policy_step"] = False
        if len(native_actions) != before + 1 or not np.array_equal(copy, native_actions[-1]):
            raise ValueError("Native action changed or policy step was not recorded exactly once")
        return result

    def inner_step(action):
        if phase["policy_step"]:
            value = array(action)
            if value.shape != (7,) or not np.isfinite(value).all():
                raise ValueError("Nonfinite or non-native7 action")
            native_actions.append(value)
        return original_inner_step(action)

    def preprocess(observation):
        step = len(observations)
        original_digest = digest(observation)
        if condition == "low_contrast":
            wrapper.frame_index = step
            supplied = wrapper._corrupt_observation(observation)
            record = wrapper.frame_records[-1]
            if not all(v["deterministic_duplicate"] and v["changed"] for v in record["views"]):
                raise ValueError("Low-contrast corruption was not deterministic/active in both views")
        else:
            supplied = observation
            record = {"frame_index": step, "condition": "clean_identity"}
        check_only_images_changed(observation, supplied)
        if digest(observation) != original_digest:
            raise ValueError("Wrapper mutated the clean observation")
        row = {"step": step, "raw_observation_sha256": original_digest,
               "supplied_observation_sha256": digest(supplied), "corruption": record}
        observations.append(row)
        with (out / "observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        for view in ("image", "image2"):
            pixels = supplied["pixels"][view][0]
            snapshots[f"last_{view}"] = pixels.copy()
            if step == 0:
                Image.fromarray(pixels).save(out / f"first_{view}.png")
        return original_preprocess(supplied)

    def select(batch):
        input_row = {"step": len(policy_inputs), "images": {}}
        for key in ("observation.images.image", "observation.images.image2"):
            value = batch[key]
            if list(value.shape) != [1, 3, 256, 256] or not torch.isfinite(value).all().item():
                raise ValueError("Actual policy image boundary differs")
            pixels = array(value)
            input_row["images"][key] = {"sha256": digest(pixels), "shape": list(pixels.shape),
                                       "spatial_std_per_channel": pixels.std(axis=(2, 3))[0].tolist()}
        if not first_rng:
            if len(policy._action_queue) != 0:
                raise ValueError("Policy action queue was not reset before first inference")
            first_rng.append(rng_fingerprint(torch))
        input_row["state_shape"] = list(batch["observation.state"].shape)
        if input_row["state_shape"] != [1, 8]:
            raise ValueError("Policy state is not native8")
        policy_inputs.append(input_row)
        with (out / "policy_inputs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(input_row) + "\n")
        return original_select(batch)

    try:
        hooks.set(base._env, "set_init_state", set_state)
        hooks.set(base, "reset", reset)
        hooks.set(base, "step", native_step)
        hooks.set(base._env, "step", inner_step)
        hooks.set(module, "preprocess_observation", preprocess)
        hooks.set(policy, "select_action", select)
        env._b4b_suite = "libero_spatial"
        b4b.select_init_state(env, [selected["seed"]])
        module.set_seed(selected["seed"])
        before_rng = rng_fingerprint(torch)
        start = time.monotonic()
        result = module.rollout(env=env, policy=policy, seeds=[selected["seed"]],
                                return_observations=False, **processors)
        elapsed = time.monotonic() - start
        actions = array(result["action"])[0]
        if not np.array_equal(actions, np.asarray(native_actions)) or len(observations) != len(actions):
            raise ValueError("Official actions, native steps and input observation counts disagree")
        if len(applied) != 1 or len(first_rng) != 1 or len(actions) not in range(1, 281):
            raise ValueError("Missing reset/RNG evidence or invalid episode length")
        if not bool(result["done"][0, -1].item()):
            raise ValueError("Incomplete official episode")
        for key, pixels in snapshots.items():
            Image.fromarray(pixels).save(out / f"{key}.png")
        np.save(out / "native_actions.npy", actions, allow_pickle=False)
        bounds = (actions < -1) | (actions > 1)
        report = {**selected, "condition": condition, "success": bool(result["success"].any().item()),
                  "steps": len(actions), "init_state_sha256": applied[0],
                  "initial_observation_sha256": observations[0]["raw_observation_sha256"],
                  "first_inference_rng_sha256": first_rng[0], "before_rollout_rng_sha256": before_rng,
                  "first_policy_images": policy_inputs[0]["images"],
                  "runtime_controller": controller_evidence,
                  "runtime_seconds": elapsed, "action_min_per_dim": actions.min(axis=0).tolist(),
                  "action_max_per_dim": actions.max(axis=0).tolist(),
                  "action_outside_unit_bounds_count_per_dim": bounds.sum(axis=0).tolist(),
                  "native_action_identity_verified": True, "non_image_fields_preserved": True,
                  "policy_images_bchw256_verified": True,
                  "visual_scope": "supplied_raw_policy_images_before_native_preprocessors_pre_action_not_terminal",
                  "visual_status": "not_viewed", "optimizer_steps": 0}
        b4b.write_json(out / "episode.json", report)
        return report
    finally:
        hooks.close()


class EvaluationFinished(Exception):
    pass


def run(out, stage, check_root=None):
    import run_pi05_libero_b4b as b4b

    out.mkdir(parents=True, exist_ok=False)
    log = (out / "run.log").open("ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    b4b.write_json(out / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
    hooks = Hooks()
    opened = None
    reports = []
    start = time.monotonic()
    try:
        path = b4b.ROOT / "docs/libero-spatial-low-contrast-protocol-v1.json"
        protocol = load_protocol(path)
        pins = protocol["provenance"]
        historical = pins["historical_protocol"]
        if b4b.sha256(b4b.ROOT / historical["path"]) != historical["sha256"]:
            raise ValueError("Historical frozen protocol drift")
        for name in ("manifest", "effective_policy_config"):
            evidence = pins["boundary_evidence"]
            if b4b.sha256(b4b.ROOT / evidence[f"{name}_path"]) != evidence[f"{name}_sha256"]:
                raise ValueError(f"Pinned boundary evidence drift: {name}")
        for name, sha in pins["source_files"].items():
            if b4b.sha256(b4b.ROOT / name) != sha:
                raise ValueError(f"Dependency source drift: {name}")
        for name, sha in pins["runtime_sources"].items():
            if b4b.sha256(name) != sha:
                raise ValueError(f"Installed source drift: {name}")
        module, _ = b4b.dependencies()
        resources = b4b.resource_checks()
        if resources["checkpoint_hashes"] != pins["checkpoint"]["sha256_by_file"]:
            raise ValueError("Checkpoint/processor hashes differ")
        if resources["versions"] != pins["versions"]:
            raise ValueError("Package versions differ")
        if resources["checkpoint_resolved_path"] != pins["checkpoint"]["resolved_path"]:
            raise ValueError("Checkpoint resolved path differs")
        old_before = {str(p.relative_to(b4b.OUT)): b4b.sha256(p) for p in b4b.OUT.rglob("*") if p.is_file()}
        metadata = {"stage": stage, "protocol_sha256": b4b.sha256(path), "runner_sha256": b4b.sha256(__file__),
                    "contract_code_sha256": b4b.sha256(Path(__file__).with_name("pi05_libero_low_contrast_contract.py")),
                    "resources": resources, "optimizer_steps": 0, "old_b4b_used_as_paired_clean": False}
        if stage == "screen":
            if check_root is None:
                raise ValueError("Screen requires the completed fixed-version check directory")
            status = json.loads((check_root / "status.json").read_text())
            if status.get("status") != "completed" or status.get("stage") != "check":
                raise ValueError("Check stage is incomplete; screen not launched")
            for filename, key in (("report.json", "report_sha256"), ("run_manifest.json", "manifest_sha256"),
                                  ("episodes.json", "episodes_sha256")):
                if b4b.sha256(check_root / filename) != status[key]:
                    raise ValueError(f"Check evidence hash changed: {filename}")
            checked = json.loads((check_root / "run_manifest.json").read_text())
            for key in ("protocol_sha256", "runner_sha256", "contract_code_sha256", "resources"):
                if checked[key] != metadata[key]:
                    raise ValueError(f"Check-to-screen version drift: {key}")
            summarize_pairs(json.loads((check_root / "episodes.json").read_text()), "check")
            metadata["prerequisite_check"] = {"path": str(check_root), "status_sha256": b4b.sha256(check_root / "status.json")}
        original_make_env, original_make_policy = module.make_env, module.make_policy

        def make_env(cfg, *args, **kwargs):
            nonlocal opened
            b4b.write_json(out / "effective_env_config.json", asdict(cfg))
            opened = original_make_env(cfg, *args, **kwargs)
            return opened

        def make_policy(*args, **kwargs):
            policy = original_make_policy(*args, **kwargs)
            actual = json.loads(json.dumps(asdict(policy.config), default=str))
            reference = json.loads((b4b.ROOT / pins["boundary_evidence"]["effective_policy_config_path"]).read_text())
            if actual != reference:
                raise ValueError("Policy config changed from boundary-verified checkpoint")
            b4b.write_json(out / "effective_policy_config.json", actual)
            return policy

        def paired(*, envs, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, **kwargs):
            tasks = schedule(stage)
            if set(envs) != {"libero_spatial"} or set(envs["libero_spatial"]) != {r["task_id"] for r in tasks}:
                raise ValueError("Unexpected effective task schedule")
            processors = dict(env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
                              preprocessor=preprocessor, postprocessor=postprocessor)
            for selected in tasks:
                for condition in ("clean", "low_contrast"):
                    prefix = f"task{selected['task_id']:02d}_{condition}"
                    report = episode(module, envs["libero_spatial"][selected["task_id"]], policy,
                                     processors, selected, condition, out / prefix, protocol)
                    reports.append(report)
                    b4b.write_json(out / "episodes.json", reports)
                    print(f"EPISODE_COMPLETE {prefix} success={report['success']} steps={report['steps']}", flush=True)
                # Fail before proceeding to another task if actual pair keys differ.
                for key in ("init_state_sha256", "initial_observation_sha256", "first_inference_rng_sha256"):
                    if reports[-1][key] != reports[-2][key]:
                        raise ValueError(f"Unpaired actual start: task {selected['task_id']} {key}")
                for key in reports[-1]["first_policy_images"]:
                    if reports[-1]["first_policy_images"][key]["sha256"] == reports[-2]["first_policy_images"][key]["sha256"]:
                        raise ValueError("Degraded images did not reach both policy input tensors")
            raise EvaluationFinished()

        hooks.set(module, "make_env", make_env)
        hooks.set(module, "make_policy", make_policy)
        hooks.set(module, "eval_policy_all", paired)
        ids = [r["task_id"] for r in schedule(stage)]
        args = [f"--policy.path={b4b.CHECKPOINT}", "--policy.n_action_steps=10", "--env.type=libero",
                "--env.task=libero_spatial", f"--env.task_ids={json.dumps(ids, separators=(',', ':'))}",
                "--env.episode_length=280", "--env.control_mode=relative", "--env.init_states=true",
                "--env.max_parallel_tasks=1", "--eval.batch_size=1", "--eval.use_async_envs=false",
                "--eval.n_episodes=1", "--seed=1000", f"--output_dir={out}"]
        metadata["official_factory_argv"] = args
        b4b.write_json(out / "run_manifest.json", metadata)
        sys.argv = [sys.argv[0], *args]
        try:
            module.main()
        except EvaluationFinished:
            pass
        else:
            raise ValueError("Did not intercept scoring with the paired runner")
        result = summarize_pairs(reports, stage)
        old_after = {str(p.relative_to(b4b.OUT)): b4b.sha256(p) for p in b4b.OUT.rglob("*") if p.is_file()}
        if old_before != old_after:
            raise ValueError("Historical B4b artifacts changed")
        result.update(runtime_seconds=round(time.monotonic() - start, 3),
                      b4b_files_unchanged=len(old_before), stage=stage, optimizer_steps=0,
                      visual_status="not_viewed", training_authorized=False,
                      historical_b4b_strict_reclassification_allowed=False)
        b4b.write_json(out / "report.json", result)
        b4b.write_json(out / "status.json", {"status": "completed", "stage": stage,
                                            "episodes": len(reports), "report_sha256": b4b.sha256(out / "report.json"),
                                            "manifest_sha256": b4b.sha256(out / "run_manifest.json"),
                                            "episodes_sha256": b4b.sha256(out / "episodes.json")})
        print(f"PAIRED_COMPLETE stage={stage} episodes={len(reports)}", flush=True)
    except BaseException:
        b4b.write_json(out / "status.json", {"status": "failed", "stage": stage,
                                            "completed_episodes": len(reports), "error": traceback.format_exc()})
        traceback.print_exc()
        raise
    finally:
        hooks.close()
        if opened is not None:
            module.close_envs(opened)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "screen"), required=True)
    parser.add_argument("--attempt", default="v1")
    parser.add_argument("--check-root", type=Path, default=Path(
        "/home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_check_v1"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        parser.error("Use a new alphanumeric attempt suffix; failed attempts are never overwritten")
    run(Path(f"/home/zsw/project_2026/simulation_output/pi05_libero_low_contrast_{args.stage}_{args.attempt}"),
        args.stage, args.check_root)
