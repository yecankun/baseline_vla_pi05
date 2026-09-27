"""Frozen clean public Push-T BC/ACT training. Default: zero-update preflight.

No guidewire data or environments are imported. Long training is user-run and
requires --execute plus a matching successful preflight. Interrupted sessions
retain separate logs; resume requires an explicit committed checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

try:
    from . import pusht_bc_act_models as models
    from .prepare_pusht_bc_act_data import canonical_hash, file_hash, REPO
except ImportError:
    import pusht_bc_act_models as models
    from prepare_pusht_bc_act_data import canonical_hash, file_hash, REPO

PROTOCOL_PATH = REPO / "docs/pusht-bc-act-training-protocol-v1.json"
PREFLIGHT_ROOT = REPO / "simulation_output/pusht_bc_act_training_preflight_v1_retry2"
TRAIN_ROOT = Path("/media/zsw/SSD1T/project_2026_weights_v1/training/pusht_bc_act_v1")


def helpers():
    if __package__:
        from . import pusht_bc_act_checkpoint as checkpoint
        from . import pusht_bc_act_training_data as data
    else:
        import pusht_bc_act_checkpoint as checkpoint
        import pusht_bc_act_training_data as data
    return data, checkpoint


def protocol():
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if value["schema"] != "pusht_bc_act_controlled_training_v1":
        raise ValueError("unknown training protocol")
    return value


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def anchor_batch(indices, step, plan):
    """Stateless, equal BC/ACT anchors independent of model/dropout RNG."""
    if type(step) is not int or not 0 <= step < plan["steps"] or len(indices) == 0:
        raise ValueError("invalid sample step/pool")
    generator = torch.Generator(device="cpu").manual_seed(plan["sampler_seed"] + step)
    positions = torch.randint(len(indices), (plan["batch_size"],), generator=generator).numpy()
    return np.asarray(indices)[positions]


def json_write(path, value, exclusive=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)


class ModelRuntime:
    def __init__(self, name, stats, plan, device="cuda"):
        self.name, self.device = name, device
        self.stats = {k: {n: torch.as_tensor(v, dtype=torch.float32, device=device)
                          for n, v in values.items()} for k, values in stats.items()}
        self.pre = self.post = None
        if name == "bc":
            self.policy = models.CompactPushTBC().to(device)
            parameters = self.policy.parameters()
        elif name == "act":
            self.policy, self.pre, self.post = models.make_act_policy_and_processors(self.stats, device)
            parameters = self.policy.get_optim_params()
            if self.policy.config.optimizer_lr_backbone != plan["optimizer"][name]["lr_backbone"]:
                raise ValueError("ACT backbone learning rate drift")
        else:
            raise ValueError("unknown model")
        opt = plan["optimizer"][name]
        self.optimizer = torch.optim.AdamW(parameters, lr=opt["lr"], betas=tuple(opt["betas"]),
                                           eps=opt["eps"], weight_decay=opt["weight_decay"],
                                           foreach=False, fused=False)
        self.clip = opt["grad_clip_norm"]

    def normalize(self, value, key):
        return (value - self.stats[key]["mean"]) / (self.stats[key]["std"] + 1e-8)

    def denormalize(self, value, key):
        # Match the installed official MEAN_STD inverse (epsilon only in division).
        return value * self.stats[key]["std"] + self.stats[key]["mean"]

    def loss(self, batch):
        self.policy.train()
        obs = batch["observation"]
        if self.name == "bc":
            bc_obs = {models.IMAGE: obs[models.IMAGE], models.STATE: self.normalize(obs[models.STATE], models.STATE)}
            loss = self.policy.loss(bc_obs, self.normalize(batch["action"][:, 0], "action"))
            return loss, {"mse": float(loss.detach())}
        return self.policy(self.pre(models.act_batch(obs, action=batch["action"], action_is_pad=batch["action_is_pad"])))

    @torch.no_grad()
    def predict(self, batch):
        self.policy.eval()
        self.policy.reset()
        obs = batch["observation"]
        if self.name == "bc":
            normalized = self.policy.select_action({models.IMAGE: obs[models.IMAGE],
                models.STATE: self.normalize(obs[models.STATE], models.STATE)})
            return self.denormalize(normalized, "action")
        self.pre.reset()
        self.post.reset()
        # Official postprocessing ends on CPU. Match the metric/target device
        # explicitly; this is a value-preserving transfer, not action clipping.
        return self.post(self.policy.select_action(self.pre(models.act_batch(obs)))).to(self.device)

    def gradients(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss, components = self.loss(batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("nonfinite/non-scalar training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip, error_if_nonfinite=True)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("nonfinite or zero gradient norm")
        return {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                "components": {k: float(v) for k, v in components.items()}}


def binding_for(data, name, plan):
    dm, ckpt = helpers()
    sources = {"trainer": file_hash(__file__), "models": file_hash(models.__file__),
               "data_adapter": file_hash(dm.__file__), "checkpoint": file_hash(ckpt.__file__)}
    if name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.processor.normalize_processor import NormalizerProcessorStep
        sources["installed_act"] = file_hash(inspect.getfile(ACTPolicy))
        sources["installed_normalizer"] = file_hash(inspect.getfile(NormalizerProcessorStep))
    versions = {k: importlib.metadata.version(k) for k in
                ["torch", "torchvision", "lerobot", "numpy", "av", "pyarrow", "filelock"]}
    return {"schema": "pusht_bc_act_training_binding_v1", "model": name,
            "protocol": plan, "protocol_file_sha256": file_hash(PROTOCOL_PATH),
            "data": data.binding, "sources": sources, "package_versions": versions,
            "torch_runtime": str(torch.__version__), "cuda_device": torch.cuda.get_device_name(0)}


def evaluation(runtime, data, plan, indices=None):
    indices = data.val_indices if indices is None else np.asarray(indices)
    absolute, squared, episodes = [], [], []
    bounds = 0
    start = time.monotonic()
    # Evaluation is observational: preserve all model/dropout RNG and training mode.
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    training = runtime.policy.training
    try:
        for offset in range(0, len(indices), plan["batch_size"]):
            selected = indices[offset:offset + plan["batch_size"]]
            batch = data.batch(selected, device=runtime.device)
            prediction = runtime.predict(batch)
            if prediction.shape != (len(selected), 2) or not torch.isfinite(prediction).all():
                raise ValueError("invalid prediction")
            error = (prediction - batch["action"][:, 0]).double().cpu().numpy()
            absolute.append(np.abs(error))
            squared.append(error ** 2)
            episodes.extend(data.episode_indices[selected].tolist())
            bounds += int(((prediction < 0) | (prediction > 512)).any(dim=1).sum())
        absolute, squared = np.concatenate(absolute), np.concatenate(squared)
        episode_values = [float(absolute[np.asarray(episodes) == e].mean()) for e in sorted(set(episodes))]
        return {"frames": len(indices), "episodes": len(set(episodes)),
                "mae_xy": absolute.mean(axis=0).tolist(), "frame_mean_mae": float(absolute.mean()),
                "episode_macro_mae": float(np.mean(episode_values)),
                "rmse": float(np.sqrt(squared.mean())), "out_of_bounds_predictions": bounds,
                "runtime_seconds": time.monotonic() - start, "action_clipping": False,
                "checkpoint_selection_allowed": False}
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        runtime.policy.train(training)
        runtime.policy.reset()


def preflight(data, name, plan, output):
    dm, ckpt = helpers()
    output.mkdir(parents=True, exist_ok=False)
    report = {"stage": "preflight", "model": name, "optimizer_steps": 0,
              "training_started": False, "environment_steps": 0, "status": "running"}
    json_write(output / "status.json", report)
    try:
        seed_all(plan["seed"])
        runtime = ModelRuntime(name, data.stats, plan)
        binding = binding_for(data, name, plan)
        report.update({"binding": binding, "backward_calls": 0,
                       "data_load_diagnostics": data.load_diagnostics})
        before = ckpt.state_dict_hash(runtime.policy)
        # Include a complete real-data batch, not synthetic statistics or inputs.
        indices = anchor_batch(data.train_indices, 0, plan)
        batch = data.batch(indices, device="cuda")
        timings, traces = [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(plan["preflight"]["batches"]):
            torch.cuda.synchronize()
            started = time.monotonic()
            traces.append(runtime.gradients(batch))
            report["backward_calls"] += 1
            torch.cuda.synchronize()
            timings.append(time.monotonic() - started)
        report.update({"gradient_probes": traces, "gradient_batch_seconds": timings})
        runtime.optimizer.zero_grad(set_to_none=True)
        if ckpt.state_dict_hash(runtime.policy) != before:
            raise ValueError("zero-update gradient probe mutated policy state")
        prediction = runtime.predict(batch).detach().clone()
        checkpoint = ckpt.save_checkpoint(output, 0, runtime.policy, runtime.optimizer, binding,
                                          extra={"kind": "preflight", "optimizer_steps": 0})
        # Perturb then restore: verifies the loader actually restores model bytes.
        with torch.no_grad():
            next(runtime.policy.parameters()).add_(0.01)
        loaded = ckpt.load_checkpoint(checkpoint, runtime.policy, runtime.optimizer, binding, "cuda")
        restored = runtime.predict(batch)
        if ckpt.state_dict_hash(runtime.policy) != before or not torch.equal(prediction, restored):
            raise ValueError("checkpoint reload changed model/prediction")
        if loaded["step"] != 0 or loaded["extra"]["kind"] != "preflight":
            raise ValueError("checkpoint stage mismatch")
        report.update({"model_state_unchanged": True, "reload_prediction_exact": True,
                       "checkpoint": str(checkpoint)})
        val_probe = evaluation(runtime, data, plan, data.val_indices[:plan["batch_size"]])
        seconds = float(np.median(timings[1:]))
        report.update({"status": "passed", "binding": binding, "batch_indices": indices.tolist(),
                       "backward_calls": len(traces), "gradient_probes": traces,
                       "gradient_batch_seconds": timings, "warm_median_gradient_batch_seconds": seconds,
                       "estimated_gradient_only_training_hours": seconds * plan["steps"] / 3600,
                       "estimate_excludes": ["optimizer update", "batch construction", "validation", "checkpoint IO"],
                       "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "parameter_count": sum(p.numel() for p in runtime.policy.parameters()),
                       "model_state_unchanged": True, "model_state_sha256": before,
                       "reload_prediction_exact": True, "validation_probe": val_probe,
                       "validation_probe_is_full_validation": False,
                       "checkpoint": str(checkpoint), "full_training_or_rollout_claim_allowed": False})
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
    json_write(output / "report.json", report, exclusive=True)
    json_write(output / "status.json", {"status": report["status"], "optimizer_steps": 0, "pid": os.getpid()})
    return report


def verify_preflight(path, binding):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("status") != "passed" or report.get("stage") != "preflight"
            or report.get("optimizer_steps") != 0 or report.get("binding") != binding
            or report.get("model_state_unchanged") is not True or report.get("reload_prediction_exact") is not True):
        raise ValueError("missing, failed or drifted preflight")
    return file_hash(path)


def train(data, name, plan, run_dir, preflight_report, resume=None, stop_after=None):
    from filelock import FileLock
    dm, ckpt = helpers()
    seed_all(plan["seed"])
    runtime = ModelRuntime(name, data.stats, plan)
    binding = binding_for(data, name, plan)
    preflight_sha = verify_preflight(preflight_report, binding)
    if resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        json_write(run_dir / "run.json", {"binding": binding, "preflight_report_sha256": preflight_sha,
                   "created_at_unix": time.time(), "checkpoint_selection": plan["selection"]}, exclusive=True)
    elif not run_dir.is_dir():
        raise ValueError("resume requires existing run directory")
    with FileLock(str(run_dir / "run.lock"), timeout=0):
        manifest = json.loads((run_dir / "run.json").read_text())
        if manifest["binding"] != binding or manifest["preflight_report_sha256"] != preflight_sha:
            raise ValueError("run binding drift")
        step = 0
        if resume is not None:
            resume = Path(resume).resolve()
            if resume.parent != (run_dir / "checkpoints").resolve():
                raise ValueError("resume checkpoint must belong to this run")
            payload = ckpt.load_checkpoint(resume, runtime.policy, runtime.optimizer, binding, "cuda")
            step = payload["step"]
            if payload["extra"].get("kind") != "training" or step < 1 or step >= plan["steps"]:
                raise ValueError("preflight/final checkpoint cannot resume training")
            later = [p for p in (run_dir / "checkpoints").glob("step_*")
                     if p.is_dir() and not p.name.endswith(".partial") and int(p.name.split("_")[1]) > step]
            if later:
                raise ValueError("later committed checkpoint exists; select it explicitly")
        target = plan["steps"] if stop_after is None else stop_after
        if not step < target <= plan["steps"]:
            raise ValueError("stop-after must be after resume step and within frozen budget")
        estimated_checkpoint = sum(p.numel() * p.element_size() for p in runtime.policy.parameters()) * 3
        needed = estimated_checkpoint * (math.ceil((plan["steps"] - step) / plan["checkpoint_every"]) + 2) + 2_000_000_000
        if shutil.disk_usage(run_dir).free < needed:
            raise ValueError(f"insufficient checkpoint storage: need {needed} bytes")
        sessions = run_dir / "sessions"
        sessions.mkdir(exist_ok=True)
        session = sessions / f"{time.time_ns()}_{os.getpid()}"
        session.mkdir(exist_ok=False)
        start = time.monotonic()
        checkpoint = str(resume) if resume else None
        status = {"status": "running", "model": name, "pid": os.getpid(), "host": socket.gethostname(),
                  "step": step, "total_steps": plan["steps"], "session": str(session),
                  "resume_from": str(resume) if resume else None, "environment_steps": 0}
        json_write(run_dir / "status.json", status)
        json_write(session / "started.json", status, exclusive=True)
        torch.cuda.reset_peak_memory_stats()
        try:
            with (session / "metrics.jsonl").open("x", encoding="utf-8") as log:
                while step < target:
                    started = time.monotonic()
                    indices = anchor_batch(data.train_indices, step, plan)
                    batch = data.batch(indices, device="cuda")
                    metrics = runtime.gradients(batch)
                    runtime.optimizer.step()
                    torch.cuda.synchronize()
                    step += 1
                    metrics.update({"step": step, "update_seconds": time.monotonic() - started,
                                    "sample_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
                    if step % plan["validation_every"] == 0 or step == target:
                        metrics["validation"] = evaluation(runtime, data, plan)
                    if step % plan["checkpoint_every"] == 0 or step == target:
                        checkpoint = str(ckpt.save_checkpoint(run_dir, step, runtime.policy, runtime.optimizer, binding,
                            extra={"kind": "training", "optimizer_steps": step, "session": str(session)}))
                        metrics["checkpoint"] = checkpoint
                    log.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                    if step == 1 or step % plan["log_every"] == 0 or step == target:
                        log.flush()
                        status.update({"step": step, "latest_checkpoint": checkpoint,
                                       "session_seconds": time.monotonic() - start})
                        json_write(run_dir / "status.json", status)
                        print(f"model={name} step={step}/{plan['steps']} loss={metrics['loss']:.6g} session_seconds={status['session_seconds']:.1f}", flush=True)
            status.update({"status": "completed" if step == plan["steps"] else "stopped_at_requested_step",
                           "step": step, "latest_checkpoint": checkpoint,
                           "selected_final_checkpoint": checkpoint if step == plan["steps"] else None,
                           "session_seconds": time.monotonic() - start,
                           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "benchmark_score_claim_allowed": False})
        except BaseException as exc:
            status.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                           "step": step, "error": str(exc), "traceback": traceback.format_exc(),
                           "latest_checkpoint": checkpoint})
            raise
        finally:
            json_write(session / "report.json", status, exclusive=True)
            json_write(run_dir / "status.json", status)
        return status


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["preflight", "train"], default="preflight")
    p.add_argument("--model", choices=["bc", "act", "both"], default="both")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--preflight-dir", type=Path, default=PREFLIGHT_ROOT)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int)
    args = p.parse_args(argv)
    if args.stage == "train" and (not args.execute or args.model == "both"):
        p.error("training requires --execute and one explicit --model bc|act")
    if args.stage == "preflight" and (args.execute or args.run_dir or args.resume or args.stop_after):
        p.error("preflight has zero updates and does not accept training options")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    plan = protocol()
    dm, _ = helpers()
    print("loading pinned public data into memory; optimizer_steps=0", flush=True)
    data = dm.load_training_data()
    if args.stage == "preflight":
        for name in (["bc", "act"] if args.model == "both" else [args.model]):
            report = preflight(data, name, plan, args.preflight_dir / name)
            print(f"preflight model={name} status={report['status']} optimizer_steps=0", flush=True)
            if report["status"] != "passed":
                print(report["error"], flush=True)
                return 1
    else:
        train(data, args.model, plan, args.run_dir or TRAIN_ROOT / args.model,
              args.preflight_dir / args.model / "report.json", args.resume, args.stop_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
