"""Send one operator-requested feed after an independently reviewed arm-only run."""
from __future__ import annotations

import argparse
from datetime import datetime
from ipaddress import IPv4Address
from pathlib import Path
import json
import time
import urllib.request

from execute_left_model_step_http import capture
from run_real10_pi05_once import load_feeder, pose_errors, robot_ready


def validate_evidence(arm, review):
    if (arm.get("status") != "completed_arm_only_awaiting_physical_review"
            or arm.get("motion_commands_accepted") != 1 or arm.get("feeder_packets") != 0
            or review.get("arm_physical_execution_confirmed") is not True):
        raise RuntimeError("Requires a completed arm-only run and explicit physical-motion review")


def output_for_attempt(arm_dir, recovery_check, feeder_host="192.168.5.13"):
    feeder_host = str(IPv4Address(feeder_host))
    if recovery_check is None:
        return arm_dir / "feed_once", None
    # A newly requested device-recovery attempt preserves the unresolved first
    # send and has its own single-use marker. It is never an automatic retry.
    prior_path = arm_dir / "feed_once" / "report.json"
    prior = json.loads(prior_path.read_text())
    network = json.loads((recovery_check / "network.json").read_text())
    age = time.time() - datetime.fromisoformat(network["recorded_at"]).timestamp()
    if (prior.get("feeder_send_attempted") is not True
            or prior.get("feeder_ack_received") is not False
            or not prior.get("destination", "").endswith(":8888")):
        raise RuntimeError("Device recovery requires the recorded unacknowledged first attempt")
    neighbor = network.get("neighbor", "")
    if (network.get("ping_exit_code") != 0 or not 0 <= age <= 300
            or not neighbor.startswith(feeder_host + " ") or "lladdr " not in neighbor
            or any(state in neighbor for state in ("FAILED", "INCOMPLETE"))):
        raise RuntimeError("No fresh successful network check for the recovered feeder")
    return arm_dir / "feed_after_device_recovery", {
        "prior_attempt": str(prior_path.resolve()),
        "network_check": str((recovery_check / "network.json").resolve()),
        "prior_destination": prior["destination"],
        "current_destination": f"{feeder_host}:8888",
        "scope": "one newly operator-requested feed after device recovery; not a resend loop",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preview-url", default="http://192.168.5.11:8765")
    parser.add_argument("--feeder-host", default="192.168.5.13", type=lambda s: str(IPv4Address(s)))
    parser.add_argument("--device-recovery-check", type=Path,
                        help="fresh successful network.json after operator-reported device recovery; one extra attempt only")
    args = parser.parse_args()
    arm = json.loads((args.arm_dir / "report.json").read_text())
    review = json.loads((args.arm_dir / "physical_review.json").read_text())
    validate_evidence(arm, review)
    out, recovery = output_for_attempt(args.arm_dir, args.device_recovery_check, args.feeder_host)
    if not args.execute:
        print("Evidence validated; no hardware connection or packet sent")
        return
    # One fixed output location per arm run prevents an ambiguous send from
    # being repeated simply by invoking this command again.
    out.mkdir(exist_ok=False)
    report = {"arm_evidence": str(args.arm_dir.resolve()), "status": "starting",
              "robot_motion_commands": 0, "feeder_send_attempted": False,
              "feeder_packets": 0, "feeder_physical_execution_confirmed": None,
              "action_source": "operator_requested_not_policy_label",
              "action": {"command": "move", "parameters": {"action": "forward", "value": 1}},
              "destination": f"{args.feeder_host}:8888", "automatic_retries": False,
              "device_recovery": recovery}
    def save():
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.feeder_local_host = "192.168.5.11"
    args.feeder_adapter = Path("/home/zsw/PycharmProjects/real_collection/hardware/feeder_device/udp_controller.py")
    ec, feeder = None, None
    save()
    try:
        from elite import EC
        ec = EC(ip=arm["elite_ip"], auto_connect=True)
        ec.sock_cmd.settimeout(3)
        report["socket_peer"] = list(ec.sock_cmd.getpeername())
        report["robot_before"] = robot_ready(ec)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        _, stamps = capture(opener, args.preview_url, out, "before")
        if any(not 0 <= time.time()-stamp <= 2 for stamp in stamps.values()):
            raise RuntimeError("Stale cameras; no feed")
        feeder = load_feeder(args)
        pose = list(ec.current_pose)
        distance, angle = pose_errors(pose, arm["tcp_after_mm_rad"])
        if distance > .02 or angle > .001:
            raise RuntimeError("Arm drift since reviewed run; no feed")
        report.update(robot_at_send=robot_ready(ec), pose_at_send=pose,
                      feeder_send_attempted=True, feeder_packets=None, dispatch_timestamp=time.time())
        save()
        reply = feeder.feed_once(wait_response=True)
        report.update(feeder_packets=1, feeder_reply=reply.decode("utf-8", errors="replace") if reply else None,
                      feeder_ack_received=reply is not None,
                      status="one_feed_sent" if reply is not None else "one_feed_submitted_no_reply")
        save()
        time.sleep(.5)
        _, report["after_camera_timestamps"] = capture(opener, args.preview_url, out, "after")
        report["robot_after"] = robot_ready(ec)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if feeder is not None: feeder.close()
        if ec is not None: ec.disconnect_ETController()
        save()
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
