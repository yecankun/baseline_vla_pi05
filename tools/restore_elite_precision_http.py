"""One monitored EC startup calibration; no path move, feeder, homing or retry."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import urllib.request

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from run_real10_pi05_once import pose_errors


class CameraRecording:
    """Record both existing HTTP streams without acquiring USB devices."""
    def __init__(self, base, out, *, save_original_frames=False):
        self.base, self.out = base.rstrip("/"), out
        self.save_original_frames = save_original_frames
        self.error = None
        self.frames = 0
        self.last_frame = 0.
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._record, daemon=True)

    def _record(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        writer = cv2.VideoWriter(str(self.out / "cameras.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10., (1920, 540))
        try:
            if not writer.isOpened():
                raise RuntimeError("Camera video encoder unavailable")
            with (self.out / "camera_times.jsonl").open("w") as log:
                while not self.done.is_set():
                    started = time.monotonic()
                    images, stamps = [], {}
                    for role in ("side", "top"):
                        with opener.open(f"{self.base}/{role}.jpg", timeout=2) as response:
                            data = response.read()
                            stamps[role] = float(response.headers["X-Capture-Timestamp"])
                        if not 0 <= time.time() - stamps[role] <= 2:
                            raise RuntimeError("Camera image is stale")
                        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            raise RuntimeError("Invalid camera image")
                        if self.frames == 0:
                            (self.out / f"{role}_before.jpg").write_bytes(data)
                        if self.save_original_frames:
                            (self.out / f"{role}_{self.frames:04d}.jpg").write_bytes(data)
                        (self.out / f"{role}_latest.jpg").write_bytes(data)
                        images.append(cv2.resize(frame, (960, 540)))
                    combined = np.concatenate(images, axis=1)
                    cv2.putText(combined, datetime.now().strftime("%H:%M:%S.%f")[:-3], (18, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
                    writer.write(combined)
                    log.write(json.dumps({"frame": self.frames, "timestamps": stamps}) + "\n")
                    self.frames += 1
                    self.last_frame = time.monotonic()
                    self.done.wait(max(0, .1 - (time.monotonic() - started)))
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            writer.release()

    def start(self):
        self.thread.start()
        deadline = time.monotonic() + 5
        while self.frames < 3:
            if self.error or time.monotonic() > deadline:
                raise RuntimeError(self.error or "No camera recording")
            time.sleep(.05)

    def check(self):
        if self.error or time.monotonic() - self.last_frame > 2:
            raise RuntimeError(self.error or "Camera recording stopped updating")

    def close(self):
        self.done.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)


def snapshot(ec):
    return {"timestamp": time.time(), "state": ec.state.value, "mode": ec.mode.value,
            "servo": ec.servo_status, "sync": ec.sync_status, "estop": ec.estop_status,
            "precise": ec.get_servo_precise_position_status(is_block=False),
            "pose": list(ec.current_pose), "joints": list(ec.current_joint),
            "motor_positions": ec.get_motor_pos(), "encoders": ec.encoder_values}


def require_operational(sample, *, initial=False):
    if (sample["state"] not in ((0,) if initial else (0, 3)) or sample["mode"] != 2
            or sample["servo"] not in (True, 1) or sample["sync"] not in (True, 1)
            or sample["estop"] != 0 or sample["precise"] not in (0, 1)):
        raise RuntimeError("Controller state does not permit monitored startup calibration")


def check_envelope(base, current, limits=None):
    distance, angle = pose_errors(current["pose"], base["pose"])
    joints = np.asarray(current["joints"], dtype=float) - np.asarray(base["joints"], dtype=float)
    if limits is None:
        limits = {"tcp_mm": 5, "rpy_rad": .02, "joint_deg": 3}
    joint_limits = np.asarray(limits.get("joint_limits_deg", [limits["joint_deg"]]*6), dtype=float)
    if (joints.shape != (6,) or joint_limits.shape != (6,) or not np.isfinite(joints).all()
            or not np.isfinite(joint_limits).all() or np.any(joint_limits <= 0)):
        raise RuntimeError("Invalid calibration joint feedback or limits")
    exceeded = [f"J{i+1} {abs(v):.4f}>{joint_limits[i]:g} deg"
                for i, v in enumerate(joints) if abs(v) > joint_limits[i]]
    if distance > limits["tcp_mm"] or angle > limits["rpy_rad"] or exceeded:
        raise RuntimeError(f"Calibration monitoring bound exceeded: TCP {distance:.4f}/{limits['tcp_mm']:g} mm, "
                           f"RPY {angle:.5f}/{limits['rpy_rad']:.5f} rad; " + "; ".join(exceeded))
    if "tool_reach_mm" in limits:
        # Bound any rigid tool point within the onsite-verified reach. This is
        # sampled monitoring, not a collision model or a guaranteed stop radius.
        before = Rotation.from_euler("xyz", base["pose"][3:])
        after = Rotation.from_euler("xyz", current["pose"][3:])
        rotation = float((after * before.inv()).magnitude())
        swept_bound = distance + 2 * limits["tool_reach_mm"] * np.sin(rotation / 2)
        if rotation > limits["rotation_rad"] or swept_bound > limits["tool_displacement_mm"]:
            raise RuntimeError(f"Tool monitoring bound exceeded: rotation {rotation:.5f} rad, point bound {swept_bound:.3f} mm")


def restore_once(ec, report, save, check_cameras, *, timeout_s=15, limits=None, envelope_base=None):
    base = snapshot(ec)
    report["before"] = base
    require_operational(base, initial=True)
    reference = envelope_base if envelope_base is not None else base
    if envelope_base is not None:
        report["monitoring_reference"] = reference
    check_envelope(reference, base, limits)
    if base["precise"] == 1:
        report["status"] = "already_precise_no_calibration"
        save()
        return
    check_cameras()
    report.update(status="calibrating", calibration_attempts=1, samples=[])
    save()
    active = True
    try:
        reply = ec.calibrate_encoder_zero()
        report["calibration_reply"] = reply
        save()
        if reply is not True:
            raise RuntimeError(f"Calibration not acknowledged: {reply!r}")
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            check_cameras()
            current = snapshot(ec)
            report["samples"].append(current)
            save()
            require_operational(current)
            check_envelope(reference, current, limits)
            if current["precise"] == 1 and current["state"] == 0:
                report.update(status="precision_restored", after=current)
                active = False
                save()
                return
            time.sleep(.05)
        raise TimeoutError("Precision was not restored within the monitoring interval")
    finally:
        if active:
            try:
                report["stop_reply"] = ec.stop()
            except BaseException as exc:
                report["stop_error"] = str(exc)
            save()


def clear_calibration_failure_once(ec, report, save, check_cameras):
    """Explicit recovery of the onsite-identified 7000-C, never a generic clear."""
    initial = snapshot(ec)
    alarms = ec.alarm_info
    speeds = np.asarray(ec.motor_speed, dtype=float)
    report.update(recovery_before=initial, recovery_alarm_info=alarms,
                  recovery_motor_speed=speeds.tolist(), clear_alarm_attempts=0)
    save()
    if (initial["state"] != 4 or initial["mode"] != 2 or initial["precise"] != 0
            or initial["estop"] != 0 or initial["servo"] not in (True, 1)
            or initial["sync"] not in (True, 1) or not isinstance(alarms, str)
            or not (alarms.startswith("[0-7000-C]")
                    or [part for part in alarms.split(",") if part != "[]"] == ["[0-7000-C]"])
            or speeds.shape != (6,)
            or not np.isfinite(speeds).all() or np.any(speeds != 0)):
        raise RuntimeError("State/alarm differs from the reviewed calibration failure; no clearing")
    check_cameras()
    report["clear_alarm_attempts"] = 1
    save()
    reply = ec.clear_alarm()
    report["clear_alarm_reply"] = reply
    save()
    if reply is not True:
        raise RuntimeError(f"Calibration alarm clearing not acknowledged: {reply!r}")
    time.sleep(.2)
    check_cameras()
    current = snapshot(ec)
    report["recovery_after_clear"] = current
    save()
    require_operational(current, initial=True)
    check_envelope(initial, current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elite-ip", required=True)
    parser.add_argument("--preview-url", default="http://192.168.5.11:8765")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute-calibration", action="store_true")
    parser.add_argument("--onsite-clearance-60mm", action="store_true",
                        help="only after onsite confirmation of <=500 mm tool reach and >=60 mm clearance")
    parser.add_argument("--recover-confirmed-calibration-failure", action="store_true",
                        help="clear once only the reviewed 7000-C calibration failure before the one calibration attempt")
    parser.add_argument("--bounded-j5-recovery-from", type=Path,
                        help="prior software-stop report: J5 <=5 deg, all other joints <=3, original tool/TCP reference retained")
    args = parser.parse_args()
    if args.recover_confirmed_calibration_failure and not (args.execute_calibration and args.onsite_clearance_60mm):
        parser.error("failure recovery requires explicit execution and onsite clearance")
    recovery = None
    if args.bounded_j5_recovery_from:
        if not args.recover_confirmed_calibration_failure:
            parser.error("bounded J5 recovery requires confirmed calibration-failure recovery")
        recovery = json.loads(args.bounded_j5_recovery_from.read_text())
        if (recovery.get("status") != "failed" or recovery.get("stop_reply") is not True
                or recovery.get("calibration_attempts") != 1 or recovery.get("monitoring_reference") is not None
                or recovery.get("after_stop", {}).get("state") != 4
                or recovery.get("after_stop", {}).get("precise") != 0
                or not recovery.get("error", "").startswith("RuntimeError: Calibration monitoring bound exceeded:")):
            parser.error("expected a reviewed first-attempt software-stop report, not a chained retry")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"started_at": datetime.now().astimezone().isoformat(), "elite_ip": args.elite_ip,
              "status": "starting", "calibration_attempts": 0, "path_motion_commands": 0,
              "feeder_packets": 0, "monitoring_bounds": {"tcp_mm": 5, "rpy_rad": .02, "joint_deg": 3},
              "calibration_may_move_robot": True,
              "procedure_source": "https://www.elibot.com/service/articles/list/193"}
    if args.onsite_clearance_60mm:
        report["monitoring_bounds"] = {"tcp_mm": 5, "rpy_rad": float(np.deg2rad(5)), "joint_deg": 3,
                                       "rotation_rad": float(np.deg2rad(5)), "tool_reach_mm": 500,
                                       "tool_displacement_mm": 50}
        report["clearance_source"] = "operator_confirmation_required_60mm_not_a_measured_collision_model"
    if recovery:
        report["monitoring_bounds"]["joint_limits_deg"] = [3, 3, 3, 3, 5, 3]
        report["recovery_reference_report"] = str(args.bounded_j5_recovery_from.resolve())
    def save():
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    ec = None
    cameras = CameraRecording(args.preview_url, args.out)
    try:
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["socket_peer"] = list(ec.sock_cmd.getpeername())
        cameras.start()
        if recovery:
            current = snapshot(ec)
            distance, angle = pose_errors(current["pose"], recovery["after_stop"]["pose"])
            if distance > .05 or angle > .001:
                raise RuntimeError("Robot drifted since the reviewed software stop")
            check_envelope(recovery["before"], current, report["monitoring_bounds"])
        if args.recover_confirmed_calibration_failure:
            clear_calibration_failure_once(ec, report, save, cameras.check)
        if args.execute_calibration:
            restore_once(ec, report, save, cameras.check, limits=report["monitoring_bounds"],
                         envelope_base=recovery["before"] if recovery else None)
        else:
            report.update(status="read_only_preview", before=snapshot(ec))
        time.sleep(.5)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Keep recording briefly after a stop so the captured evidence includes
        # deceleration. A positive stop reply alone does not establish STOP.
        if ec is not None and report.get("stop_reply") is not None:
            try:
                time.sleep(.5)
                report["after_stop"] = snapshot(ec)
                report["after_stop_motor_speed"] = ec.motor_speed
                report["after_stop_alarms"] = ec.alarm_info
            except BaseException as exc:
                report["after_stop_query_error"] = str(exc)
        cameras.close()
        report.update(camera_frames=cameras.frames, camera_error=cameras.error)
        if ec is not None:
            ec.disconnect_ETController()
        save()
        print(json.dumps({k: report.get(k) for k in ("status", "calibration_attempts", "calibration_reply", "error", "stop_reply", "camera_frames")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
