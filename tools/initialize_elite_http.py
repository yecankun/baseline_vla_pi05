"""Bounded EC startup, default read-only; no calibration, path motion or feeder."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from restore_elite_precision_http import CameraRecording, snapshot


def startup_sample(ec):
    return {"timestamp": time.time(), "state": ec.state.value, "mode": ec.mode.value,
            "estop": ec.estop_status, "initializing": ec.get_digital_io("M473"),
            "servo": ec.servo_status, "sync": ec.sync_status,
            "brakes": ec.send_CMD("get_servo_brake_off_status"),
            "motor_positions": ec.get_motor_pos(), "motor_speed": ec.motor_speed}


def check_sample(base, current):
    if current["state"] != 0 or current["mode"] != 2 or current["estop"] != 0:
        raise RuntimeError("Startup requires STOP / REMOTE / no emergency stop")
    if (type(current["initializing"]) is not int or current["initializing"] not in (0, 1)
            or type(current["servo"]) is not bool or type(current["sync"]) is not bool
            or not isinstance(current["brakes"], list) or len(current["brakes"]) != 6
            or any(type(v) is not int or v not in (0, 1) for v in current["brakes"])):
        raise RuntimeError("Invalid startup feedback")
    before = np.asarray(base["motor_positions"], dtype=float)
    after = np.asarray(current["motor_positions"], dtype=float)
    speed = np.asarray(current["motor_speed"], dtype=float)
    if (before.shape != (6,) or after.shape != (6,) or speed.shape != (6,)
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or not np.isfinite(speed).all()):
        raise RuntimeError("Invalid motor feedback")
    # Unsynchronized computed joints/TCP are not physical coordinates. The
    # motor feedback can wrap across +/-180 degrees on synchronization.
    delta = (after-before+180) % 360-180
    if np.max(np.abs(delta)) > .2:
        raise RuntimeError("Motor changed >0.2 degrees during non-motion startup")
    return float(np.max(np.abs(delta)))


def initialize_once(ec, report, save, check_cameras, *, timeout_s=15):
    base = startup_sample(ec)
    report["before"] = base
    check_sample(base, base)
    if np.any(np.asarray(base["motor_speed"]) != 0):
        raise RuntimeError("Motors are not stationary before startup")
    report.update(samples=[], commands=[], status="initializing")
    save()
    dispatched = False

    def observe():
        check_cameras()
        current = startup_sample(ec)
        report["samples"].append(current)
        save()
        check_sample(base, current)
        return current

    def command(name, invoke, complete):
        nonlocal dispatched
        observe()
        event = {"name": name, "attempted_at": time.time(), "reply": None}
        report["commands"].append(event)
        save()
        dispatched = True  # A transport exception may occur after dispatch.
        event["reply"] = invoke()
        save()
        if event["reply"] is not True:
            raise RuntimeError(f"Startup command not acknowledged: {name}")
        deadline = time.monotonic()+timeout_s
        while time.monotonic() < deadline:
            current = observe()
            if complete(current):
                return current
            time.sleep(.05)
        raise TimeoutError(f"Startup state did not arrive after {name}")

    try:
        current = base
        if current["initializing"] == 1:
            # Manufacturer's cold-start step. This is not generic fault recovery;
            # an initial or subsequent ERROR state is rejected above.
            current = command("clearAlarm_release_startup_brakes", ec.clear_alarm,
                              lambda s: all(v == 1 for v in s["brakes"]))
        if not all(v == 1 for v in current["brakes"]):
            raise RuntimeError("Unexpected closed brake outside cold-start state")
        if not current["sync"]:
            current = command("syncMotorStatus", ec.sync, lambda s: s["sync"])
        if not current["servo"]:
            current = command("set_servo_status_1", lambda: ec.set_servo_status(1),
                              lambda s: s["servo"] and s["sync"])
        current = observe()
        if current["initializing"] != 0 or not (current["servo"] and current["sync"]):
            raise RuntimeError("Startup flags incomplete")
        report.update(status="initialized_no_calibration_or_path_motion", after=current)
        save()
    except BaseException:
        if dispatched:
            try:
                report["stop_reply"] = ec.stop()
            except BaseException as exc:
                report["stop_error"] = str(exc)
            save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elite-ip", default="192.168.5.66")
    parser.add_argument("--preview-url", default="http://192.168.5.11:8765")
    parser.add_argument("--execute-initialize", action="store_true")
    parser.add_argument("--onsite-ready", action="store_true",
                        help="current onsite clearance/personnel and camera visibility have been checked")
    args = parser.parse_args()
    if args.execute_initialize and not args.onsite_ready:
        parser.error("Current onsite conditions must be established before startup")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now().astimezone().isoformat(), "status": "starting",
              "motion_commands": 0, "feeder_packets": 0, "calibration_attempts": 0,
              "automatic_retries": False, "execute_requested": args.execute_initialize,
              "source": "https://www.elibot.com/service/articles/list/193",
              "motor_motion_guard_deg": .2,
              "guard_scope": "Sampled diagnostic bound; not a hardware stop guarantee",
              "visual_status": "not_viewed"}
    def save():
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    ec, cameras = None, None
    save()
    try:
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["socket_peer"] = list(ec.sock_cmd.getpeername())
        cameras = CameraRecording(args.preview_url, args.out)
        cameras.start()
        if args.execute_initialize:
            initialize_once(ec, report, save, cameras.check)
        else:
            report.update(status="read_only_preview", before=startup_sample(ec))
        report["controller_after"] = snapshot(ec)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if cameras:
            cameras.close()
            report.update(camera_frames=cameras.frames, camera_error=cameras.error)
        if ec:
            ec.disconnect_ETController()
        save()
        print(json.dumps({k: report.get(k) for k in ("status", "commands", "error", "stop_reply")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
