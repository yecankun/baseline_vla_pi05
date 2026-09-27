"""One explicitly reauthorized fresh attempt; preserve the verified engine bytes.

The only runtime engine configuration change is the independently pinned v2
authorization identity. No training/evaluation function or numeric gate changes.
This executable calls the engine exactly once, never resumes or retries it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUTH_SHA = "05bc0012cce953d2c1d0a40020ca2de9f9b973553e21497a35bc49637cfbda66"
OLD_AUTH_SHA = "8bfe78322c44f7e4959287a9f53a920b0591c3df975b85c3b8b02d4f4f209511"
AUTH_PATH = "docs/libero-native-world-model-learning-authorization-v2.json"
ENGINE_NAME = "run_pi05_libero_world_model_learning_diagnostic"
OUT_NAME = "pi05_libero_learning_diagnostic_remote_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != expected:
        raise ValueError(f"bound retry evidence hash/path differs: {relative}")
    return path


def validate(root, authorization_sha, execute):
    if execute is not True or authorization_sha != AUTH_SHA:
        raise ValueError("explicit execution flag and hard-pinned v2 authorization SHA required")
    auth = read(checked(root, AUTH_PATH, AUTH_SHA))
    if (auth["execution_authorized"] is not True or auth["output_directory_name"] != OUT_NAME
            or auth["previous_authorization_sha256"] != OLD_AUTH_SHA):
        raise ValueError("new one-attempt authorization identity mismatch")
    evidence = {AUTH_PATH: AUTH_SHA,
        "docs/libero-native-world-model-learning-protocol-v1.json": auth["design_protocol_sha256"],
        "docs/libero-native-world-model-learning-authorization-v1.json": OLD_AUTH_SHA,
        f"tools/{ENGINE_NAME}.py": auth["repaired_engine_sha256"],
        "tools/pi05_libero_world_model_training.py": auth["training_utility_sha256"]}
    for directory_key, hashes_key in (("failed_attempt_directory", "failed_attempt_sha256"),
                                     ("failed_code_directory", "failed_code_sha256")):
        directory = root / auth[directory_key]
        if {p.name for p in directory.iterdir()} != set(auth[hashes_key]):
            raise ValueError("preserved failed attempt/snapshot membership differs")
        evidence.update({f"{auth[directory_key]}/{k}": v for k, v in auth[hashes_key].items()})
    preflight_path = f"{auth['preflight_directory']}/report.json"
    evidence[preflight_path] = auth["preflight_report_sha256"]
    for name, expected in evidence.items():
        checked(root, name, expected)
    failure = read(root / auth["failed_attempt_directory"] / "status.json")
    if failure["training_started"] is not False or failure["completed_updates"] != 0 or failure["last_attempted_step"] != 0:
        raise ValueError("fresh retry requires preserved zero-update failure")
    preflight = read(root / preflight_path)
    if (preflight["status"] != "passed" or preflight["optimizer_steps"] != 0 or preflight["backward_calls"] != 0
            or preflight["batch_count"] != 20 or preflight["all_prediction_hashes_exact"] is not True
            or preflight["all_metrics_match"] is not True):
        raise ValueError("strict repaired preflight required")
    evidence.update({f"{auth['preflight_directory']}/{k}": v for k, v in preflight["output_sha256"].items()})
    evidence.update({f"tools/{k}": v for k, v in preflight["implementation_sha256"].items()})
    for name, expected in evidence.items():
        checked(root, name, expected)
    return auth, evidence


def engine_argv(root, auth):
    return ["--feature-pack", str(root / "simulation_output/pi05_libero_feature_pair_v1"),
        "--random-reference", str(root / "simulation_output/pi05_libero_world_model_dry_run_remote_v1"),
        "--persistence-reference", str(root / "simulation_output/pi05_libero_persistence_remote_v1"),
        "--protocol", str(root / "docs/libero-native-world-model-learning-protocol-v1.json"),
        "--protocol-sha256", auth["design_protocol_sha256"],
        "--authorization", str(root / AUTH_PATH), "--authorization-sha256", AUTH_SHA,
        "--out", str(root / "simulation_output" / OUT_NAME), "--execute-authorized-diagnostic"]


def load_engine(root):
    engine = importlib.import_module(ENGINE_NAME)
    if Path(engine.__file__).resolve() != (root / "tools" / f"{ENGINE_NAME}.py").resolve():
        raise ValueError("unexpected imported engine path")
    if engine.AUTHORIZATION_SHA != OLD_AUTH_SHA:
        raise ValueError("engine authorization already reconfigured; fresh single-process invocation required")
    return engine


def execute_once(root, authorization_sha, execute_flag):
    auth, evidence = validate(root, authorization_sha, execute_flag)
    out = root / "simulation_output" / OUT_NAME
    if out.exists():
        raise FileExistsError("new attempt already exists; no overwrite/resume/retry")
    engine = load_engine(root)  # Only after the independent byte/evidence checks.
    invocation_path = root / "simulation_output" / auth["invocation_file_name"]
    own_sha = digest(__file__)
    invocation = {"schema": "libero_learning_explicit_retry_invocation_v1", "status": "running",
        "authorization_sha256": AUTH_SHA, "previous_runtime_authorization_sha256": OLD_AUTH_SHA,
        "active_runtime_authorization_sha256": AUTH_SHA, "engine_source_sha256": auth["repaired_engine_sha256"],
        "wrapper_source_sha256": own_sha, "verified_bound_evidence_sha256": evidence,
        "runtime_change": "only engine.AUTHORIZATION_SHA; no model/loss/optimizer/evaluation/step configuration change",
        "engine_argv": engine_argv(root, auth), "engine_main_calls": 0,
        "automatic_retry_allowed": False, "output_directory": str(out)}
    # A sibling invocation record prevents a second attempt even if startup fails
    # before engine.main creates its own exclusive directory. Never create out here.
    with invocation_path.open("x", encoding="utf-8") as stream:
        json.dump(invocation, stream, indent=2, allow_nan=False)
    started = time.monotonic()
    try:
        engine.AUTHORIZATION_SHA = AUTH_SHA
        invocation["engine_main_calls"] = 1
        engine.main(invocation["engine_argv"])
        report = read(out / "report.json")
        if (report["status"] != "passed" or report["optimizer_steps"] != 200 or report["backward_calls"] != 200
                or report["authorization_sha256"] != AUTH_SHA
                or report["checkpoint_metadata"]["authorization_sha256"] != AUTH_SHA):
            raise ValueError("completed engine report must bind v2 authorization and exactly200 updates")
        for name, expected in evidence.items():
            checked(root, name, expected)
        if digest(__file__) != own_sha:
            raise ValueError("wrapper source changed during execution")
        invocation.update(status="completed", report_sha256=digest(out / "report.json"))
    except BaseException as error:
        invocation.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        engine.AUTHORIZATION_SHA = OLD_AUTH_SHA
        invocation["original_authorization_pin_restored"] = engine.AUTHORIZATION_SHA == OLD_AUTH_SHA
        invocation["runtime_seconds"] = time.monotonic() - started
        invocation["engine_status"] = read(out / "status.json") if (out / "status.json").is_file() else None
        with invocation_path.open("w", encoding="utf-8") as stream:
            json.dump(invocation, stream, indent=2, allow_nan=False)
    print("EXPLICIT_FRESH_ATTEMPT_COMPLETED", invocation["report_sha256"], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--execute-authorized-diagnostic", action="store_true", required=True)
    args = parser.parse_args(argv)
    execute_once(ROOT, args.authorization_sha256, args.execute_authorized_diagnostic)


if __name__ == "__main__":
    main()
