"""Fresh-environment paired diagnostic. Frozen v1 episode mechanics are reused.

reset-check: twenty fresh resets, no policy. check: task0/task4 full pairs.
screen: user-run twenty full episodes. No training or native source patches.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

import run_pi05_libero_b4b as b4b
import run_pi05_libero_low_contrast as v1
from pi05_libero_low_contrast_v2_contract import load_protocol, schedule, summarize_pairs, validate_reset_pairs


ROOT = b4b.ROOT
DEFAULT_RESET = ROOT / "simulation_output/pi05_libero_low_contrast_v2_reset-check_v1"
DEFAULT_CHECK = ROOT / "simulation_output/pi05_libero_low_contrast_v2_check_v1"


def inventory(roots):
    return {str(p): b4b.sha256(p) for root in roots for p in sorted(root.rglob("*")) if p.is_file()}


def identity(protocol_path, parent, cfg):
    """Identical no-model provenance for reset gate, policy check and screen."""
    pins = parent["provenance"]
    for path, expected in {**pins["source_files"], **pins["runtime_sources"]}.items():
        if b4b.sha256(ROOT / path) != expected:
            raise ValueError(f"Pinned source drift: {path}")
    for field in ("manifest", "effective_policy_config"):
        e = pins["boundary_evidence"]
        if b4b.sha256(ROOT / e[f"{field}_path"]) != e[f"{field}_sha256"]:
            raise ValueError(f"Boundary evidence drift: {field}")
    if b4b.sha256(ROOT / pins["historical_protocol"]["path"]) != pins["historical_protocol"]["sha256"]:
        raise ValueError("Historical protocol drift")
    versions = {k: importlib.metadata.version(k) for k in pins["versions"]}
    if versions != pins["versions"]:
        raise ValueError("Package version drift")
    extra = Path("/home/zsw/miniconda3/envs/project2026-pi/lib/python3.10/site-packages/libero/libero/envs")
    return {"protocol_sha256": b4b.sha256(protocol_path),
            "runner_sha256": b4b.sha256(__file__),
            "contract_sha256": b4b.sha256(Path(__file__).with_name("pi05_libero_low_contrast_v2_contract.py")),
            "v1_runner_sha256": b4b.sha256(v1.__file__),
            "v1_contract_sha256": b4b.sha256(Path(v1.__file__).with_name("pi05_libero_low_contrast_contract.py")),
            "versions": versions, "parent_provenance": pins,
            "native_reset_sources": {str(extra / p): b4b.sha256(extra / p)
                                     for p in ("env_wrapper.py", "bddl_base_domain.py")},
            "canonical_env_config": asdict(cfg)}


def verify_gate(root, stage, current_identity):
    status = json.loads((root / "status.json").read_text())
    if status.get("status") != "completed" or status.get("stage") != stage:
        raise ValueError(f"Incomplete prerequisite {stage}")
    for filename, key in (("report.json", "report_sha256"), ("run_manifest.json", "manifest_sha256"),
                          ("episodes.json", "episodes_sha256")):
        if b4b.sha256(root / filename) != status.get(key):
            raise ValueError(f"Prerequisite evidence drift: {filename}")
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["identity"] != current_identity:
        raise ValueError(f"Prerequisite version drift: {stage}")
    rows = json.loads((root / "episodes.json").read_text())
    expected = validate_reset_pairs(rows) if stage == "reset-check" else summarize_pairs(rows, stage)
    report = json.loads((root / "report.json").read_text())
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Prerequisite report differs from rows: {key}")
    return rows, {"path": str(root), "status_sha256": b4b.sha256(root / "status.json")}


@contextmanager
def fresh_environment(module, factory, cfg, selected, condition, torch, records):
    """Always construct with single-task native factory; never reuse an arm."""
    one = replace(cfg, task_ids=[selected["task_id"]])
    module.set_seed(1000)
    record = {**selected, "condition": condition, "construction_index": len(records),
              "construction_seed": 1000, "pre_construction_rng_sha256": v1.rng_fingerprint(torch),
              "factory_config": asdict(one), "factory_kwargs": {"n_envs": 1, "use_async_envs": False},
              "closed": False}
    records.append(record)
    envs = factory(one, n_envs=1, use_async_envs=False)
    try:
        if set(envs) != {"libero_spatial"} or set(envs["libero_spatial"]) != {selected["task_id"]}:
            raise ValueError("Fresh factory returned wrong task schedule")
        env = envs["libero_spatial"][selected["task_id"]]
        record["post_construction_rng_sha256"] = v1.rng_fingerprint(torch)
        env._b4b_suite = "libero_spatial"
        b4b.select_init_state(env, [1000])
        yield env, record
    finally:
        module.close_envs(envs)
        record["closed"] = True


def reset_once(module, env, selected, condition, out, torch):
    from PIL import Image
    out.mkdir()
    base = env.envs[0].unwrapped
    original = base._env.set_init_state
    applied = []
    hooks = v1.Hooks()

    def set_init(state):
        expected = np.asarray(base._init_states[0])
        if not np.array_equal(state, expected) or state.dtype != expected.dtype:
            raise ValueError("Wrong actual init payload")
        applied.append(hashlib.sha256(state.tobytes()).hexdigest())
        return original(state)

    try:
        hooks.set(base._env, "set_init_state", set_init)
        module.set_seed(1000)
        before = v1.rng_fingerprint(torch)
        observation, _ = env.reset(seed=[1000])
        post = v1.rng_fingerprint(torch)
        if len(applied) != 1 or base.num_steps_wait != 10:
            raise ValueError("Reset/init schedule differs")
        v1.check_only_images_changed(observation, observation)
        for view in ("image", "image2"):
            Image.fromarray(observation["pixels"][view][0]).save(out / f"first_{view}.png")
        row = {**selected, "condition": condition, "init_state_sha256": applied[0],
               "initial_observation_sha256": v1.digest(observation),
               "post_reset_rng_sha256": post, "before_reset_rng_sha256": before,
               "measured_settling_steps": 10, "policy_loaded": False,
               "policy_steps": 0, "optimizer_steps": 0}
        b4b.write_json(out / "reset.json", row)
        return row
    finally:
        hooks.close()


@contextmanager
def guard_actual_start(env, policy, expected_reset, paired_rng, torch):
    """Abort before inference if actual reset differs from the reset-only gate."""
    hooks = v1.Hooks()
    original_reset, original_select = env.reset, policy.select_action
    inner = env.envs[0].unwrapped._env
    original_set = inner.set_init_state
    evidence = {"seeded_resets": 0, "init_payloads": 0, "first_inferences": 0}
    phase = {"seeded_reset": False}

    def set_init(state):
        if phase["seeded_reset"]:
            if hashlib.sha256(np.asarray(state).tobytes()).hexdigest() != expected_reset["init_state_sha256"]:
                raise ValueError("Actual init payload differs from reset-only gate")
            evidence["init_payloads"] += 1
        return original_set(state)

    def reset(*args, **kwargs):
        phase["seeded_reset"] = kwargs.get("seed") is not None
        try:
            result = original_reset(*args, **kwargs)
        finally:
            phase["seeded_reset"] = False
        if kwargs.get("seed") is not None:
            evidence["seeded_resets"] += 1
            if v1.digest(result[0]) != expected_reset["initial_observation_sha256"]:
                raise ValueError("Actual policy start differs from reset-only gate")
            if v1.rng_fingerprint(torch) != expected_reset["post_reset_rng_sha256"]:
                raise ValueError("Actual policy post-reset RNG differs from reset-only gate")
        return result

    def select(batch):
        if evidence["first_inferences"] == 0:
            if evidence["seeded_resets"] != 1 or evidence["init_payloads"] != 1:
                raise ValueError("Policy inference without one verified seeded reset/init payload")
            value = v1.rng_fingerprint(torch)
            if paired_rng and value != paired_rng[0]:
                raise ValueError("Actual first-inference RNG differs between arms")
            if not paired_rng:
                paired_rng.append(value)
            evidence["first_inferences"] = 1
        return original_select(batch)

    try:
        hooks.set(env, "reset", reset)
        hooks.set(inner, "set_init_state", set_init)
        hooks.set(policy, "select_action", select)
        yield evidence
    finally:
        hooks.close()


def run(out, stage, reset_root=DEFAULT_RESET, check_root=DEFAULT_CHECK):
    out.mkdir(parents=True, exist_ok=False)
    log = (out / "run.log").open("ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    b4b.write_json(out / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
    start = time.monotonic()
    rows, constructions = [], []
    hooks, bootstrap = v1.Hooks(), None
    manifest = {}
    try:
        protocol_path = ROOT / "docs/libero-spatial-low-contrast-protocol-v2.json"
        delta = load_protocol(protocol_path)
        parent = v1.load_protocol(ROOT / delta["parent_protocol"]["path"])
        module, _ = b4b.dependencies()
        import torch
        from lerobot.envs.configs import LiberoEnv
        torch.cuda.get_rng_state_all()
        cfg = LiberoEnv(task="libero_spatial", task_ids=list(range(10)), episode_length=280,
                        init_states=True, max_parallel_tasks=1, control_mode="relative")
        ident = json.loads(json.dumps(identity(protocol_path, parent, cfg), default=str))
        protected = [b4b.OUT, ROOT / "simulation_output/pi05_libero_low_contrast_check_v1",
                     ROOT / "simulation_output/pi05_libero_low_contrast_screen_v1",
                     ROOT / "simulation_output/libero_task4_reset_probe_v1"]
        before_files = inventory(protected)
        manifest = {"stage": stage, "identity": ident, "optimizer_steps": 0,
                    "protected_files": before_files, "constructions": constructions,
                    "bootstrap_environments_used_for_scoring": False}
        reset_rows = []
        if stage != "reset-check":
            reset_rows, proof = verify_gate(reset_root, "reset-check", ident)
            manifest["reset_prerequisite"] = proof
        if stage == "screen":
            _, proof = verify_gate(check_root, "check", ident)
            manifest["check_prerequisite"] = proof
            checked = json.loads((check_root / "run_manifest.json").read_text())
            if checked["reset_prerequisite"]["status_sha256"] != manifest["reset_prerequisite"]["status_sha256"]:
                raise ValueError("Check and screen refer to different actual reset gate evidence")
        factory = module.make_env
        b4b.write_json(out / "run_manifest.json", manifest)
        if stage == "reset-check":
            for selected in schedule(stage):
                for condition in ("clean", "low_contrast"):
                    with fresh_environment(module, factory, cfg, selected, condition, torch, constructions) as (env, _):
                        row = reset_once(module, env, selected, condition,
                                         out / f"task{selected['task_id']:02d}_{condition}", torch)
                    rows.append(row)
                    b4b.write_json(out / "episodes.json", rows)
                    b4b.write_json(out / "run_manifest.json", manifest)
                for key in ("init_state_sha256", "initial_observation_sha256", "post_reset_rng_sha256"):
                    if rows[-1][key] != rows[-2][key]:
                        raise ValueError(f"Fresh reset pair mismatch: task {selected['task_id']} {key}")
                print(f"RESET_PAIR_COMPLETE task={selected['task_id']}", flush=True)
            result = validate_reset_pairs(rows)
        else:
            resources = b4b.resource_checks()
            if resources["checkpoint_hashes"] != parent["provenance"]["checkpoint"]["sha256_by_file"]:
                raise ValueError("Checkpoint/processor hashes differ")
            if resources["checkpoint_resolved_path"] != parent["provenance"]["checkpoint"]["resolved_path"]:
                raise ValueError("Checkpoint resolved path differs")
            manifest["resources"] = resources
            original_policy = module.make_policy

            def make_env(env_cfg, *args, **kwargs):
                nonlocal bootstrap
                manifest["bootstrap_config"] = asdict(env_cfg)
                bootstrap = factory(env_cfg, *args, **kwargs)
                return bootstrap

            def make_policy(*args, **kwargs):
                policy = original_policy(*args, **kwargs)
                actual = json.loads(json.dumps(asdict(policy.config), default=str))
                reference = json.loads((ROOT / parent["provenance"]["boundary_evidence"]["effective_policy_config_path"]).read_text())
                if actual != reference:
                    raise ValueError("Policy config changed")
                b4b.write_json(out / "effective_policy_config.json", actual)
                return policy

            def paired(*, envs, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, **kwargs):
                nonlocal bootstrap
                if envs is not bootstrap:
                    raise ValueError("Unknown bootstrap environment")
                module.close_envs(bootstrap)
                bootstrap = None
                manifest["bootstrap_closed_before_scoring"] = True
                processors = dict(env_preprocessor=env_preprocessor, env_postprocessor=env_postprocessor,
                                  preprocessor=preprocessor, postprocessor=postprocessor)
                expected = {(r["task_id"], r["condition"]): r for r in reset_rows}
                for selected in schedule(stage):
                    paired_rng = []
                    for condition in ("clean", "low_contrast"):
                        name = f"task{selected['task_id']:02d}_{condition}"
                        with fresh_environment(module, factory, cfg, selected, condition, torch, constructions) as (env, _):
                            with guard_actual_start(env, policy, expected[(selected["task_id"], condition)], paired_rng, torch) as guard:
                                row = v1.episode(module, env, policy, processors, selected, condition, out / name, parent)
                            row["actual_start_guard"] = guard
                        rows.append(row)
                        b4b.write_json(out / name / "episode.json", row)
                        b4b.write_json(out / "episodes.json", rows)
                        b4b.write_json(out / "run_manifest.json", manifest)
                        print(f"EPISODE_COMPLETE {name} success={row['success']} steps={row['steps']}", flush=True)
                    for key in ("init_state_sha256", "initial_observation_sha256", "first_inference_rng_sha256"):
                        if rows[-1][key] != rows[-2][key]:
                            raise ValueError(f"Unpaired actual start: {selected['task_id']} {key}")
                    for key in rows[-1]["first_policy_images"]:
                        if rows[-1]["first_policy_images"][key]["sha256"] == rows[-2]["first_policy_images"][key]["sha256"]:
                            raise ValueError("Corruption did not reach actual policy images")
                raise v1.EvaluationFinished()

            hooks.set(module, "make_env", make_env)
            hooks.set(module, "make_policy", make_policy)
            hooks.set(module, "eval_policy_all", paired)
            ids = [r["task_id"] for r in schedule(stage)]
            args = [f"--policy.path={b4b.CHECKPOINT}", "--policy.n_action_steps=10", "--env.type=libero",
                    "--env.task=libero_spatial", f"--env.task_ids={json.dumps(ids, separators=(',', ':'))}",
                    "--env.episode_length=280", "--env.control_mode=relative", "--env.init_states=true",
                    "--env.max_parallel_tasks=1", "--eval.batch_size=1", "--eval.use_async_envs=false",
                    "--eval.n_episodes=1", "--seed=1000", f"--output_dir={out}"]
            manifest["official_factory_argv"] = args
            sys.argv = [sys.argv[0], *args]
            try:
                module.main()
            except v1.EvaluationFinished:
                pass
            else:
                raise ValueError("Official scoring was not intercepted")
            result = summarize_pairs(rows, stage)
        if before_files != inventory(protected):
            raise ValueError("Protected historical evidence changed")
        if len(constructions) != 2 * len(schedule(stage)) or not all(r["closed"] for r in constructions):
            raise ValueError("Incomplete fresh environment lifecycle")
        result.update(stage=stage, runtime_seconds=round(time.monotonic() - start, 3),
                      optimizer_steps=0, policy_loaded=stage != "reset-check", visual_status="not_viewed",
                      protected_files_unchanged=len(before_files),
                      measured_reset_only_settling_steps=200 if stage == "reset-check" else None,
                      constructor_and_terminal_autoreset_steps_counted=False)
        b4b.write_json(out / "run_manifest.json", manifest)
        b4b.write_json(out / "report.json", result)
        b4b.write_json(out / "status.json", {"status": "completed", "stage": stage,
                     "report_sha256": b4b.sha256(out / "report.json"),
                     "manifest_sha256": b4b.sha256(out / "run_manifest.json"),
                     "episodes_sha256": b4b.sha256(out / "episodes.json")})
    except BaseException:
        b4b.write_json(out / "run_manifest.json", manifest)
        b4b.write_json(out / "status.json", {"status": "failed", "stage": stage,
                     "completed_rows": len(rows), "error": traceback.format_exc()})
        traceback.print_exc()
        raise
    finally:
        hooks.close()
        if bootstrap is not None:
            module.close_envs(bootstrap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reset-check", "check", "screen"), required=True)
    parser.add_argument("--attempt", default="v1")
    parser.add_argument("--reset-root", type=Path, default=DEFAULT_RESET)
    parser.add_argument("--check-root", type=Path, default=DEFAULT_CHECK)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        parser.error("Use a fresh alphanumeric attempt suffix; no overwrite/resume")
    run(ROOT / f"simulation_output/pi05_libero_low_contrast_v2_{args.stage}_{args.attempt}",
        args.stage, args.reset_root, args.check_root)
