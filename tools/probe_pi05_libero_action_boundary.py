"""Short, diagnostic-only PI05/LIBERO boundary tracing; never a score runner.

Reuse the installed evaluator's factories, seed, precision and processors.
Stop after a bounded window without changing the native 280-step horizon.
Observer hooks return the original result and never supply clipped actions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np


ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.array(value, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite boundary value")
    return result


class Observer:
    """Instance-only wrappers, restored in reverse order even on failure."""

    def __init__(self, emit):
        self.emit = emit
        self.step = None
        self.restores = []

    def wrap(self, obj, name, before, after):
        original = getattr(obj, name)
        owned = name in vars(obj)
        previous = vars(obj).get(name)

        def observed(*args, **kwargs):
            step = self.step
            snapshot = before(*args, **kwargs) if step is not None else None
            result = original(*args, **kwargs)
            if step is not None:
                self.emit({"kind": name, "step": step, **after(snapshot, result)})
            return result

        setattr(obj, name, observed)
        self.restores.append((obj, name, owned, previous))

    def restore(self):
        for obj, name, owned, previous in reversed(self.restores):
            if owned:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.restores.clear()


def summarize(steps, events, controller):
    """Read-only arithmetic; expected saturation is NEVER passed to a controller."""
    if not steps or [x["step"] for x in steps] != list(range(len(steps))):
        raise ValueError("Empty or noncontiguous step trace")
    actions = np.asarray([x["environment_input_7"] for x in steps], dtype=float)
    if actions.shape != (len(steps), 7) or not np.isfinite(actions).all():
        raise ValueError("Invalid native action array")
    low, high = np.asarray(controller["input_min"]), np.asarray(controller["input_max"])
    outlow, outhigh = np.asarray(controller["output_min"]), np.asarray(controller["output_max"])
    checks = {"native_postprocessor_preserved": True, "inner_environment_input_preserved": True,
              "osc_input_preserved": True, "osc_original_output_matches_native_formula": True,
              "gripper_input_preserved": True, "gripper_original_output_matches_native_formula": True,
              "two_runtime_cameras_256": True, "two_policy_cameras_256": True,
              "state8_before_policy_preprocessor": True}
    counts = []
    for row in steps:
        s = row["step"]
        checks["native_postprocessor_preserved"] &= np.array_equal(
            row["policy_postprocessed_7"], row["environment_input_7"])
        checks["two_runtime_cameras_256"] &= row["raw_camera_shapes"] == {
            "image": [1, 256, 256, 3], "image2": [1, 256, 256, 3]}
        checks["two_policy_cameras_256"] &= row["policy_camera_shapes"] == {
            "observation.images.image": [1, 3, 256, 256],
            "observation.images.image2": [1, 3, 256, 256]}
        checks["state8_before_policy_preprocessor"] &= row["env_processed_state_shape"] == [1, 8]
        es = [e for e in events if e["step"] == s]
        inner = [e for e in es if e["kind"] == "step"]
        osc = [e for e in es if e["kind"] == "scale_action"]
        grip = [e for e in es if e["kind"] == "format_action"]
        if len(inner) != 1 or len(osc) != 1 or not grip:
            raise ValueError(f"Missing or ambiguous native calls at step {s}")
        checks["inner_environment_input_preserved"] &= np.array_equal(inner[0]["input"], actions[s])
        checks["osc_input_preserved"] &= np.array_equal(osc[0]["input"], actions[s, :6])
        expected = (np.clip(actions[s, :6], low, high) - (high + low) / 2) * (
            np.abs(outhigh - outlow) / np.abs(high - low)) + (outhigh + outlow) / 2
        checks["osc_original_output_matches_native_formula"] &= np.allclose(
            osc[0]["output"], expected, rtol=1e-6, atol=1e-8)
        for e in grip:
            checks["gripper_input_preserved"] &= np.array_equal(e["input"], actions[s, 6:])
            expected = np.clip(np.asarray(e["before"]) + np.array([-1., 1.]) *
                               controller["gripper_speed"] * np.sign(e["input"]), -1., 1.)
            checks["gripper_original_output_matches_native_formula"] &= np.allclose(
                e["output"], expected, rtol=1e-6, atol=1e-8)
        counts.append({"step": s, "osc_calls": len(osc), "gripper_calls": len(grip)})
    imgs = [e for e in events if e["kind"] == "_preprocess_images"]
    checks["internal_two_valid_224_plus_empty"] = bool(imgs) and all(
        e["shapes"] == [[1, 3, 224, 224]] * 3 and e["masks"] == [[True], [True], [False]]
        for e in imgs)
    bounds = (actions < -1) | (actions > 1)
    return {"checks": {k: bool(v) for k, v in checks.items()},
            "per_dimension": [{"name": name, "min": float(actions[:, i].min()),
                               "max": float(actions[:, i].max()),
                               "out_of_bounds_steps": int(bounds[:, i].sum())}
                              for i, name in enumerate(ACTION_NAMES)],
            "out_of_bounds_steps": int(bounds.any(axis=1).sum()),
            "arm_saturated_steps": int(((actions[:, :6] < low) | (actions[:, :6] > high)).any(axis=1).sum()),
            "native_call_counts": counts,
            "inference_image_preprocessing_calls": len(imgs)}


class ProbeFinished(Exception):
    pass


def run(out, max_steps):
    import run_pi05_libero_b4b as b4b

    out.mkdir(parents=True, exist_ok=False)
    log = (out / "run.log").open("ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    started = time.monotonic()
    b4b.write_json(out / "status.json", {"status": "running", "pid": os.getpid()})
    envs_open = None
    restore_module = []
    events, steps = [], []

    def emit(event):
        events.append(event)
        with (out / "native_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    observer = Observer(emit)
    report = {}
    try:
        module, libero = b4b.dependencies()
        import torch
        from PIL import Image

        manifest = b4b.resource_checks()
        original_run = json.loads((b4b.OUT / "run_manifest.json").read_text())
        protected_before = {str(p.relative_to(b4b.OUT)): b4b.sha256(p)
                            for p in b4b.OUT.rglob("*") if p.is_file()}
        for key in ("checkpoint_hashes", "protocol_sha256", "versions", "wrapper_sha256"):
            if manifest[key] != original_run[key]:
                raise ValueError(f"B4b provenance drift: {key}")
        manifest.update(probe_sha256=b4b.sha256(__file__), max_steps=max_steps,
                        scope="task0_seed1000_init0_short_boundary_probe_not_score",
                        differences_from_b4b=["only task0", "manual bounded window", "observation-only hooks"],
                        optimizer_steps=0, wrapper_clipping_added=False)
        original_make_env = module.make_env
        original_make_policy = module.make_policy

        def make_env(cfg, *args, **kwargs):
            nonlocal envs_open
            b4b.write_json(out / "effective_env_config.json", asdict(cfg))
            envs_open = original_make_env(cfg, *args, **kwargs)
            return envs_open

        def make_policy(*args, **kwargs):
            policy = original_make_policy(*args, **kwargs)
            b4b.write_json(out / "effective_policy_config.json", asdict(policy.config))
            reference = json.loads((b4b.OUT / "effective_policy_config.json").read_text())
            if json.loads(json.dumps(asdict(policy.config), default=str)) != reference:
                raise ValueError("Effective policy configuration differs from B4b")
            return policy

        def probe(*, envs, policy, env_preprocessor, env_postprocessor, preprocessor,
                  postprocessor, start_seed, **kwargs):
            if set(envs) != {"libero_spatial"} or set(envs["libero_spatial"]) != {0} or start_seed != 1000:
                raise ValueError("Unexpected probe schedule")
            env = envs["libero_spatial"][0]
            env._b4b_suite = "libero_spatial"
            base = env.envs[0].unwrapped
            selected = b4b.select_init_state(env, [1000])
            applied = []
            original_set = base._env.set_init_state
            set_owned = "set_init_state" in vars(base._env)
            set_previous = vars(base._env).get("set_init_state")

            def traced_set(state):
                # Initialization payload provenance must preserve original dtype.
                value = np.array(state.detach().cpu().numpy() if hasattr(state, "detach") else state, copy=True)
                applied.append({"matching_indices": [i for i, x in enumerate(base._init_states)
                                if np.array_equal(value, np.asarray(x))],
                                "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
                return original_set(state)

            base._env.set_init_state = traced_set
            try:
                policy.reset()
                observation, _ = env.reset(seed=[1000])
            finally:
                if set_owned:
                    base._env.set_init_state = set_previous
                else:
                    delattr(base._env, "set_init_state")
            if len(applied) != 1 or applied[0]["matching_indices"] != [0]:
                raise ValueError("Actual reset payload differs from init0")
            reference_states = [json.loads(line) for line in (b4b.OUT / "episode_trace.jsonl").read_text().splitlines()]
            reference_states = [e for e in reference_states if e["kind"] == "state_applied" and
                                e["task_id"] == 0 and e["seed"] == 1000 and e["init_state_index"] == 0]
            if len(reference_states) != 1 or reference_states[0]["state_sha256"] != applied[0]["sha256"]:
                raise ValueError("Actual initial payload differs from recorded B4b task0 seed1000")
            robots = base._env.robots
            if len(robots) != 1:
                raise ValueError("Expected one native robot")
            robot = robots[0]
            ctrl, gripper = robot.controller, robot.gripper
            if not ctrl.use_delta or ctrl.impedance_mode != "fixed" or ctrl.control_dim != 6 or gripper.dof != 1:
                raise ValueError("Native controller contract changed")
            controller = {k: array(getattr(ctrl, k)).tolist() for k in
                          ("input_min", "input_max", "output_min", "output_max")}
            controller.update(class_name=type(ctrl).__name__, gripper_class=type(gripper).__name__,
                              gripper_speed=float(gripper.speed), use_delta=bool(ctrl.use_delta),
                              impedance_mode=ctrl.impedance_mode)
            manifest.update(selected=selected, actual_init_state=applied, controller=controller,
                            runtime_camera_hw=[base.observation_height, base.observation_width],
                            num_steps_wait=base.num_steps_wait, native_horizon=base._max_episode_steps)
            files = {inspect.getfile(obj) for obj in (type(ctrl), type(gripper), type(robot),
                                                     type(policy), module, libero)}
            files.add(inspect.getfile(ctrl.scale_action))
            manifest["runtime_sources"] = {p: b4b.sha256(p) for p in sorted(files)}
            for p, digest in original_run["runtime_sources"].items():
                if manifest["runtime_sources"].get(p) != digest:
                    raise ValueError(f"B4b source drift: {p}")
            b4b.write_json(out / "run_manifest.json", manifest)
            observer.wrap(ctrl, "scale_action", lambda a: array(a).tolist(),
                          lambda snap, result: {"input": snap, "output": array(result).tolist()})
            observer.wrap(gripper, "format_action",
                          lambda a: {"input": array(a).tolist(), "before": array(gripper.current_action).tolist()},
                          lambda snap, result: {**snap, "output": array(result).tolist()})
            observer.wrap(base._env, "step", lambda a: array(a).tolist(),
                          lambda snap, result: {"input": snap})
            observer.wrap(policy, "_preprocess_images", lambda batch: None,
                          lambda snap, result: {"shapes": [list(x.shape) for x in result[0]],
                                                "masks": [array(x).tolist() for x in result[1]]})
            # Automatic terminal resets may add warm-up actions. Exclude them
            # from the current policy step without changing/resetting any state.
            original_reset = base.reset
            reset_owned = "reset" in vars(base)
            reset_previous = vars(base).get("reset")

            def reset_untraced(*args, **kwargs):
                old_step, observer.step = observer.step, None
                try:
                    return original_reset(*args, **kwargs)
                finally:
                    observer.step = old_step

            base.reset = reset_untraced
            frames = []
            try:
                for step in range(max_steps):
                    observer.step = step
                    row = {"step": step, "raw_camera_shapes": {
                        k: list(v.shape) for k, v in observation["pixels"].items()}}
                    if step in (0, max_steps // 2, max_steps - 1):
                        for key in ("image", "image2"):
                            name = f"step_{step:03d}_{key}.png"
                            Image.fromarray(np.asarray(observation["pixels"][key][0])).save(out / name)
                            frames.append({"step": step, "view": key, "path": name,
                                           "timing": "pre_action_raw_policy_camera"})
                    batch = module.preprocess_observation(observation)
                    batch = module.add_envs_task(env, batch)
                    batch = env_preprocessor(batch)
                    row["env_processed_state_shape"] = list(batch["observation.state"].shape)
                    batch = preprocessor(batch)
                    row["policy_camera_shapes"] = {k: list(v.shape) for k, v in batch.items()
                                                   if k.startswith("observation.images.")}
                    row["policy_batch_keys"] = sorted(batch)
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                    row["policy_selected_normalized_7"] = array(action)[0].tolist()
                    action = postprocessor(action)
                    row["policy_postprocessed_7"] = array(action)[0].tolist()
                    action = env_postprocessor({module.ACTION: action})[module.ACTION]
                    # Identical conversion and object passed to env.step as upstream.
                    native = action.to("cpu").numpy()
                    row["environment_input_7"] = array(native)[0].tolist()
                    observation, _, terminated, truncated, _ = env.step(native)
                    row["terminated"] = bool(np.any(terminated))
                    row["truncated"] = bool(np.any(truncated))
                    steps.append(row)
                    with (out / "steps.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if row["terminated"] or row["truncated"]:
                        break
            finally:
                observer.step = None
                if reset_owned:
                    base.reset = reset_previous
                else:
                    delattr(base, "reset")
            analysis = summarize(steps, events, controller)
            report.update(status="passed" if all(analysis["checks"].values()) else "failed",
                          scope=manifest["scope"], analysis=analysis, steps_executed=len(steps),
                          manual_window_limit=max_steps, native_horizon=base._max_episode_steps,
                          stop_reason="environment_done" if steps[-1]["terminated"] or steps[-1]["truncated"]
                          else "manual_diagnostic_window", visual_artifacts=frames, visual_status="not_viewed",
                          optimizer_steps=0, benchmark_score_claim_allowed=False,
                          b4b_strict_pass_reclassification_allowed=False,
                          wrapper_clipping_added=False, native_controller_changed=False)
            raise ProbeFinished()

        for name, fn in (("make_env", make_env), ("make_policy", make_policy), ("eval_policy_all", probe)):
            restore_module.append((module, name, getattr(module, name)))
            setattr(module, name, fn)
        args = [f"--policy.path={b4b.CHECKPOINT}", "--policy.n_action_steps=10",
                "--env.type=libero", "--env.task=libero_spatial", "--env.task_ids=[0]",
                "--env.episode_length=280", "--env.control_mode=relative", "--env.init_states=true",
                "--env.max_parallel_tasks=1", "--eval.batch_size=1", "--eval.use_async_envs=false",
                "--eval.n_episodes=1", "--seed=1000", f"--output_dir={out}"]
        manifest["official_factory_argv"] = args
        b4b.write_json(out / "run_manifest.json", manifest)
        sys.argv = [sys.argv[0], *args]
        try:
            module.main()
        except ProbeFinished:
            pass
        else:
            raise AssertionError("Bounded probe did not intercept official scoring")
        protected_after = {str(p.relative_to(b4b.OUT)): b4b.sha256(p)
                           for p in b4b.OUT.rglob("*") if p.is_file()}
        report["b4b_original_files_unchanged"] = protected_before == protected_after
        report["b4b_original_file_count"] = len(protected_before)
        b4b.write_json(out / "b4b_original_inventory.json", protected_before)
        if protected_before != protected_after:
            raise AssertionError("Protected B4b artifacts changed")
        report["runtime_seconds"] = round(time.monotonic() - started, 3)
        b4b.write_json(out / "report.json", report)
        if report["status"] != "passed":
            raise AssertionError("Boundary validation failed; inspect report")
        b4b.write_json(out / "status.json", {"status": "completed", "steps": len(steps),
                                            "optimizer_steps": 0, "report_sha256": b4b.sha256(out / "report.json")})
        print(f"BOUNDARY_PROBE_COMPLETE steps={len(steps)} out={out}", flush=True)
    except BaseException:
        b4b.write_json(out / "status.json", {"status": "failed", "steps": len(steps),
                                            "error": traceback.format_exc()})
        traceback.print_exc()
        raise
    finally:
        observer.restore()
        for obj, name, original in reversed(restore_module):
            setattr(obj, name, original)
        if envs_open is not None:
            module.close_envs(envs_open)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="v1")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt) or not 1 <= args.steps <= 60:
        parser.error("Use a fresh safe attempt name and 1..60 diagnostic steps")
    run(Path(f"/home/zsw/project_2026/simulation_output/pi05_libero_action_boundary_{args.attempt}"), args.steps)
