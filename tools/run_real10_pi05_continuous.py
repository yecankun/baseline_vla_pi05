"""Operator-supervised Real10 policy actions; no task-success oracle.

HOME is a separate absolute preparation move. In the default mode EXECUTE starts
at most ten fresh observation -> inference -> guarded-action cycles within 120
seconds. Ctrl-C stops the loop, and the onsite emergency stop remains the
physical backstop.
With --operator-step-mode, RESUME skips homing and each Enter triggers one fresh
cycle from the current TCP. --auto-steps removes the per-step Enter prompt;
Q/Ctrl-C stops. Per-step guards remain in force.
With --elite-path-file, HOME and EXECUTE gate a finite recorded-path Elite
fallback, while live model inference and model-driven Piper hold/feed continue.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from real10_pi05_bridge import (
    LatestColorCamera, ModelProcess, MODEL_PYTHON, SIDE_SERIAL, TOP_SERIAL,
    TRAINING_ROOT, dump_json,
)
from run_real10_pi05_once import (
    DEFAULT_HOME_POSE, MAX_DEMO_STEP_MM, action_plan, checked_joints, home_elite, load_feeder,
    move_and_feed_once, pose_errors, robot_ready, save_views,
)


def recorded_path_targets(path_file, home_pose, max_step_mm):
    """Finite XYZ polyline, interpolated so every nominal TCP leg fits the step cap."""
    points = []
    with path_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.split("#", 1)[0].strip()
            if not value:
                continue
            parts = [float(item) for item in value.replace(",", " ").split()]
            if len(parts) != 3 or not np.isfinite(parts).all():
                raise ValueError(f"{path_file}:{line_number}: expected three finite XYZ mm values")
            points.append(np.asarray(parts, dtype=float))
    if len(points) < 2 or np.linalg.norm(points[0] - np.asarray(home_pose[:3])) > .001:
        raise ValueError("Recorded path needs at least two points and must start at the selected home XYZ")
    targets = []
    for waypoint in range(1, len(points)):
        start, end = points[waypoint - 1], points[waypoint]
        distance = float(np.linalg.norm(end - start))
        if not 0 < distance <= 100:
            raise ValueError(f"Recorded path segment {waypoint} is invalid or over 100 mm")
        count = math.ceil(distance / max_step_mm)
        for substep in range(1, count + 1):
            xyz = (start + (end - start) * (substep / count)).tolist()
            targets.append({"waypoint": waypoint, "substep": substep,
                            "substeps_in_segment": count, "pose": xyz + list(home_pose[3:])})
    return targets


def bounded_actions(args, ec, feeder, cameras, worker, report, save):
    """Every iteration uses new images/TCP; neither predictions nor ACKs become history."""
    operator_steps = args.operator_step_mode
    path_mode = bool(args.elite_path_file)
    started = None  # The action-window timer begins at the first dispatch.
    last_action_finished = report["homing"]["completed_timestamp"]
    stop_reason = "recorded_path_end" if path_mode else "max_actions_reached"
    indices = (range(args.path_start_step - 1, len(args.path_targets)) if path_mode else
               itertools.count() if operator_steps else range(args.max_actions))
    for index in indices:
        if operator_steps and not args.auto_steps:
            try:
                answer = input("按 Enter 采集新图并执行下一步；输入 Q 结束 > ").strip()
            except EOFError:
                stop_reason = "operator_input_closed"
                break
            if answer:
                stop_reason = "operator_stopped_at_step_prompt"
                break
        if not (operator_steps or path_mode) and started is not None and time.monotonic() - started >= args.max_duration_s:
            stop_reason = "max_duration_reached"
            break

        frames = {view: camera.read() for view, camera in cameras.items()}
        pose_started = time.time()
        pose = [float(value) for value in ec.current_pose]
        now = time.time()
        ages = {view: now - frame[1] for view, frame in frames.items()}
        if any(age < 0 or age > 2 or frames[view][1] <= last_action_finished
               for view, age in ages.items()):
            raise RuntimeError(f"No fresh post-action dual-camera observation: {ages}")
        if now - pose_started > 2:
            raise RuntimeError("Elite pose read exceeded 2 s; no action")
        distance_at_observation, _ = pose_errors(pose, args.home_pose)
        if not (operator_steps or path_mode) and distance_at_observation > args.max_distance_from_home_mm:
            raise RuntimeError("Current TCP left the bounded demo region; no next action")

        step = {"index": index + 1, "status": "planning", "observation": {
            "task": args.task, "elite_tcp_pose_6d": pose,
            "previous_controller_state": None,
            "piper_history_validity": 0,
            "camera_host_timestamps": {view: frame[1] for view, frame in frames.items()},
            "pose_query_started": pose_started, "pose_query_finished": now,
            "distance_from_recorded_home_mm": distance_at_observation,
        }, "elite_move_attempted": False, "elite_target_reached": False,
            "piper_send_attempted": False, "piper_packets_sent": 0,
            "piper_physical_execution_confirmed": None}
        report["steps"].append(step)
        save()
        try:
            step["robot_before_inference"] = robot_ready(ec)
            begin = time.perf_counter()
            reply = worker.predict(index, {view: frame[0] for view, frame in frames.items()},
                                   pose, args.task, None)
            step.update(policy_output=reply["prediction"], state_32=reply["state_32"],
                        roundtrip_seconds=time.perf_counter() - begin)
            step["robot_before_dispatch"] = robot_ready(ec)
            current = np.asarray(ec.current_pose, dtype=float)
            drift_mm, drift_rad = pose_errors(current, pose)
            if drift_mm > .25 or drift_rad > .01:
                raise RuntimeError("TCP changed since inference; no stale action")
            plan = action_plan(reply["prediction"], current, args.max_step_mm,
                               translation_gain=1.0 if path_mode else args.translation_gain)
            if path_mode:
                target = args.path_targets[index]
                expected_base = args.home_pose if index == 0 else args.path_targets[index - 1]["pose"]
                off_path_mm, off_path_rad = pose_errors(current, expected_base)
                if off_path_mm > .5 or off_path_rad > .01:
                    raise RuntimeError("Elite TCP is off the recorded path; no next action")
                delta = np.asarray(target["pose"], dtype=float) - current
                if np.linalg.norm(delta[:3]) > args.max_step_mm + 1e-6:
                    raise RuntimeError("Recorded path target exceeds the per-step translation cap")
                plan.update(elite_action_source="recorded_path_fallback",
                            model_elite_tcp_delta_6d_not_executed=plan["raw_elite_tcp_delta_6d"],
                            recorded_path_waypoint=target["waypoint"],
                            recorded_path_substep=target["substep"],
                            recorded_path_substeps_in_segment=target["substeps_in_segment"],
                            guarded_elite_tcp_delta_6d=delta.tolist(),
                            elite_target_tcp_pose_6d=target["pose"],
                            translation_clipped=False,
                            translation_gain=1.0)
                plan.pop("demo_scaled_elite_tcp_delta_6d", None)
            else:
                plan["elite_action_source"] = "model_prediction_guarded"
            distance_from_home, _ = pose_errors(plan["elite_target_tcp_pose_6d"], args.home_pose)
            if not (operator_steps or path_mode) and distance_from_home > args.max_distance_from_home_mm:
                raise RuntimeError("Planned TCP target exceeds bounded demo region; no action")
            joints = checked_joints(ec, plan["elite_target_tcp_pose_6d"])
            step.update(plan=plan, dispatch_base_pose=current.tolist(),
                        ik_target_joints=joints, distance_from_home_mm=distance_from_home)
            before_path = args.out / f"before_{index+1:02d}.png"
            save_views(before_path, frames, f"连续演示 第{index+1}步｜动作前双视角", args.font)
            save()
            print(json.dumps({"step": index + 1, "plan": plan}, ensure_ascii=False), flush=True)
            ages_at_dispatch = {view: time.time() - frame[1] for view, frame in frames.items()}
            step["observation_age_at_dispatch_s"] = ages_at_dispatch
            if any(not 0 <= age <= 2 for age in ages_at_dispatch.values()):
                stop_reason = "stale_observation_before_dispatch"
                step["status"] = "not_dispatched"
                save()
                break
            if not (operator_steps or path_mode) and started is not None and time.monotonic() - started >= args.max_duration_s:
                stop_reason = "max_duration_reached_before_dispatch"
                step["status"] = "not_dispatched"
                save()
                break

            step["status"] = "dispatching"
            report["attempted_actions"] += 1
            if started is None:
                started = time.monotonic()
                report["action_window_started_timestamp"] = time.time()
            save()
            move_and_feed_once(ec, feeder, plan, joints, args.speed, step, save,
                               wait_feeder_ack=not path_mode)
            feed_unconfirmed = (not path_mode and step["piper_send_attempted"]
                                and step.get("piper_ack_received") is not True)
            step["status"] = ("feed_unconfirmed" if feed_unconfirmed else
                              "action_completed_feed_sent_unverified" if path_mode and step["piper_send_attempted"]
                              else "action_completed")
            step["completed_timestamp"] = time.time()
            if not feed_unconfirmed:
                report["completed_actions"] += 1
            save()
            last_action_finished = step["completed_timestamp"]

            time.sleep(.5)
            after = {view: camera.read() for view, camera in cameras.items()}
            after_path = args.out / f"after_{index+1:02d}.png"
            save_views(after_path, after,
                       f"连续演示 第{index+1}步｜动作后观测（非任务成功判断）", args.font)
            cv2.imshow("Real10 bounded execution - Ctrl-C / Q to stop", cv2.imread(str(after_path)))
            report["preview_opened"] = True
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                stop_reason = "operator_stopped_after_action"
                break

            if step["piper_send_attempted"]:
                if feed_unconfirmed:
                    stop_reason = "feed_ack_missing_no_retry"
                    break
                if args.after_feed == "stop":
                    stop_reason = "stopped_after_first_feed"
                    break
            if not (operator_steps or path_mode) and time.monotonic() - started >= args.max_duration_s:
                stop_reason = "max_duration_reached"
                break

            if operator_steps or path_mode or index + 1 < args.max_actions:
                # The interval is measured after completed execution, never as a
                # camera-FPS dispatch clock. Ctrl-C also interrupts this wait.
                remaining = args.min_interval_s - (time.time() - last_action_finished)
                until = time.monotonic() + max(0, remaining)
                while time.monotonic() < until:
                    if cv2.waitKey(50) & 0xFF in (27, ord("q"), ord("Q")):
                        stop_reason = "operator_stopped_between_actions"
                        break
                if stop_reason == "operator_stopped_between_actions":
                    break
        except BaseException as exc:
            step.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
            save()
            raise
    report["termination_reason"] = stop_reason
    report["status"] = ("stopped_feed_unconfirmed" if stop_reason == "feed_ack_missing_no_retry"
                        else "stopped_without_task_success_claim")
    save()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "real10_recorded_path_fallback_v1" if args.elite_path_file else
              "real10_operator_steps_v1" if args.operator_step_mode
              else "real10_bounded_continuous_v1", "status": "starting",
              "execute": args.execute, "task": args.task, "elite_ip": args.elite_ip,
              "side_serial": args.side_serial, "top_serial": args.top_serial,
              "operator_step_mode": args.operator_step_mode,
              "auto_steps": args.auto_steps,
              "elite_action_source": "recorded_path_fallback" if args.elite_path_file else "model_prediction_guarded",
              "elite_path_file": str(args.elite_path_file) if args.elite_path_file else None,
              "path_start_step": args.path_start_step if args.elite_path_file else None,
              "recorded_path_total_steps": len(args.path_targets) if args.elite_path_file else None,
              "max_actions": None if args.operator_step_mode or args.elite_path_file else args.max_actions,
              "max_duration_s": None if args.operator_step_mode or args.elite_path_file else args.max_duration_s,
              "min_interval_s": args.min_interval_s, "max_step_mm": args.max_step_mm,
              "elite_speed_percent": args.speed,
              "translation_gain": args.translation_gain,
              "translation_gain_scope": ("model Elite prediction recorded but not executed" if args.elite_path_file else
                                         "demo controller xyz only; raw policy output unchanged"),
              "max_distance_from_home_mm": None if args.operator_step_mode or args.elite_path_file else args.max_distance_from_home_mm,
              "after_feed": args.after_feed, "attempted_actions": 0,
              "completed_actions": 0, "steps": [],
              "task_success_state": "not_defined; operator observation only",
              "piper_history": "missing; validity=0 until an observed controller state exists",
              "feeder_queue_assumption": "user-reported: a second feed blocks until the first finishes; not independently verified",
              "feeder_ack_policy": "no_wait_no_retry" if args.elite_path_file else "wait_up_to_2_s_no_retry",
              "feeder_destination": f"{args.feeder_host}:8888",
              "feeder_source": f"{args.feeder_local_host}:37011",
              "feeder_adapter": str(args.feeder_adapter),
              "visual_status": "not_viewed",
              "homing": {"status": "pending_operator_resume" if args.operator_step_mode or args.path_start_step > 1 else "pending",
                         "target_pose": (args.path_targets[args.path_start_step - 2]["pose"]
                                         if args.path_start_step > 1 else args.home_pose),
                         "pose_source": (f"recorded path step {args.path_start_step - 1} readback"
                                         if args.path_start_step > 1 else args.home_pose_source),
                         "speed_percent": args.home_speed,
                         "max_distance_mm": args.home_max_distance_mm,
                         "max_joint_step_deg": 60.0, "timeout_s": args.home_timeout_s,
                         "position_tolerance_mm": .5, "rpy_tolerance_rad": .01,
                         "elite_move_attempted": False, "elite_target_reached": False,
                         "is_policy_action": False, "feeder_packets_allowed": 0}}
    save = lambda: dump_json(args.out / "report.json", report)
    save()
    worker, ec, feeder, cameras = None, None, None, {}
    started = time.time()
    try:
        if args.operator_step_mode or args.path_start_step > 1:
            trigger = "确认后自动逐步推理执行，Ctrl-C/Q 停止。" if args.auto_steps else "每次仅按 Enter 执行一步。"
            if args.path_start_step > 1:
                print(f"固定路径从第 {args.path_start_step} 步续跑；不归位。"
                      "确认当前 TCP 与上一目标吻合、递丝状态、后续路径净空和急停位置。", flush=True)
            else:
                print("从 Elite 当前 TCP 续跑；不归位、无总步数/时长/距起点边界。"
                      + trigger + "确认设备交接、当前姿态、导丝状态、"
                      "后续路径净空及急停位置。", flush=True)
            confirmation, prompt = "RESUME", "输入 RESUME 确认从当前位置续跑；其他输入取消 > "
        else:
            print(f"采集起点（非机械零位）：{args.home_pose}\n"
                  "归位路径无自动避障。先确认设备交接、TCP/tool、导丝状态、回程净空和急停位置。", flush=True)
            confirmation, prompt = "HOME", "输入 HOME 确认归位；其他输入取消 > "
        if input(prompt).strip() != confirmation:
            report.update(status="cancelled_before_hardware",
                          termination_reason="resume_not_confirmed" if args.operator_step_mode or args.path_start_step > 1
                          else "home_not_confirmed")
            report["homing"]["status"] = "cancelled"
            return
        from elite import EC
        ec = EC(ip=args.elite_ip, auto_connect=True)
        ec.sock_cmd.settimeout(3.0)
        report["robot_before"] = robot_ready(ec)
        if args.operator_step_mode or args.path_start_step > 1:
            current = [float(value) for value in ec.current_pose]
            if args.path_start_step > 1:
                expected = args.path_targets[args.path_start_step - 2]["pose"]
                off_path_mm, off_path_rad = pose_errors(current, expected)
                if off_path_mm > .5 or off_path_rad > .01:
                    raise RuntimeError("Current TCP does not match the previous recorded-path step; no resume")
            report["homing"].update(status="skipped_resume_current",
                                    before_pose=current, elite_after_pose=current,
                                    elite_move_attempted=False,
                                    completed_timestamp=time.time())
            save()
            print(f"续跑当前 TCP：{current}；记录采集起点：{args.home_pose}", flush=True)
        elif args.elite_path_file:
            home_elite(ec, args, report["homing"], save)
            print(f"固定路径演示：Elite 使用 {args.elite_path_file} 插值后的 {len(args.path_targets)} 步；"
                  f"每步最多 {args.max_step_mm:g} mm、速度不超过 {args.speed:g}%。"
                  "模型 Elite 输出仅显示和记录，Piper hold/feed 仍按模型意图。"
                  "无任务成功自动判定；Ctrl-C/Q 可停止，异常无重试。", flush=True)
        else:
            home_elite(ec, args, report["homing"], save)

        worker = ModelProcess(args, args.out / "worker.log")
        report["model"] = worker.ready
        for view, serial in (("side", args.side_serial), ("top", args.top_serial)):
            cameras[view] = LatestColorCamera(serial)
            cameras[view].read()
        feeder = load_feeder(args)
        if args.operator_step_mode:
            trigger = ("自动逐步采集、推理并执行" if args.auto_steps
                       else "每按一次 Enter 才采集并执行一步")
            print(f"从当前位姿续跑；{trigger}，平移最多"
                  f"{args.max_step_mm:g} mm、关节速度不超过{args.speed:g}%。"
                  "输入 Q 或 Ctrl-C 停止；无任务成功判定、无自动重试。", flush=True)
        elif args.elite_path_file:
            start_label = ("当前位置已匹配上一记录目标" if args.path_start_step > 1 else "归位已到位")
            print(f"{start_label}。将沿固定路径从第 {args.path_start_step} 步自动执行至第 {len(args.path_targets)} 步，"
                  f"每步不超过 {args.max_step_mm:g} mm、速度不超过 {args.speed:g}%。"
                  "模型 Elite 输出只作显示；Piper 意图仍来自模型，发包后不等待 ACK。", flush=True)
        else:
            print(f"归位已到位。最多{args.max_actions}步或{args.max_duration_s:g}秒；每步平移最多"
                  f"{args.max_step_mm:g}mm，速度不超过{args.speed:g}%。无任务成功判定，现场操作者"
                  "可随时 Ctrl-C；异常立即停，不自动重试。", flush=True)
        if input("输入 EXECUTE 开始执行；其他输入取消 > ").strip() != "EXECUTE":
            report.update(status="cancelled_after_homing", termination_reason="execute_not_confirmed")
            return
        pose_before_loop = [float(value) for value in ec.current_pose]
        if not args.operator_step_mode:
            expected_start = (args.path_targets[args.path_start_step - 2]["pose"]
                              if args.path_start_step > 1 else args.home_pose)
            home_distance, home_rpy_error = pose_errors(pose_before_loop, expected_start)
            if home_distance > .5 or home_rpy_error > .01:
                raise RuntimeError("Elite left the expected path start while waiting for EXECUTE; no inference/feed")
        report["execution_started_timestamp"] = time.time()
        save()
        bounded_actions(args, ec, feeder, cameras, worker, report, save)
    except BaseException as exc:
        report.update(status="operator_interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        if isinstance(exc, KeyboardInterrupt):
            print("操作者停止；检查报告和现场设备状态。", flush=True)
        else:
            raise
    finally:
        cleanup = [(f"camera:{view}", camera.close) for view, camera in cameras.items()]
        if feeder is not None:
            cleanup.append(("feeder", feeder.close))
        if ec is not None:
            cleanup.append(("Elite", ec.disconnect_ETController))
        if worker is not None:
            cleanup.append(("model", worker.close))
        if report.get("preview_opened"):
            cleanup.append(("display", cv2.destroyAllWindows))
        report["cleanup_errors"] = []
        for name, close in cleanup:
            try:
                close()
            except BaseException as exc:
                report["cleanup_errors"].append(f"{name}: {type(exc).__name__}: {exc}")
        if report["cleanup_errors"]:
            report["status"] = "failed_cleanup"
        report["elapsed_seconds"] = time.time() - started
        save()
        print(f"结束：{report['status']}；报告：{args.out / 'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("left", "right"))
    parser.add_argument("--elite-ip", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="enable onsite HOME/EXECUTE or RESUME/EXECUTE prompts")
    parser.add_argument("--operator-step-mode", action="store_true",
                        help="resume at current TCP; each Enter triggers one fresh step until operator stops")
    parser.add_argument("--auto-steps", action="store_true",
                        help="with --operator-step-mode, automatically start the next fresh step after completion")
    parser.add_argument("--elite-path-file", type=Path,
                        help="finite recorded XYZ path fallback for Elite; live model output stays visible and Piper remains model driven")
    parser.add_argument("--path-start-step", type=int, default=1,
                        help="one-based recorded-path step; >1 resumes from current TCP after RESUME/EXECUTE")
    parser.add_argument("--home-pose", nargs=6, type=float, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--home-speed", type=float, default=5.0)
    parser.add_argument("--home-max-distance-mm", type=float, default=500.0)
    parser.add_argument("--home-timeout-s", type=float, default=60.0,
                        help="Elite homing arrival timeout; onsite may choose up to 300 s")
    parser.add_argument("--max-step-mm", type=float, required=True,
                        help=f"operator-selected translation cap, at most {MAX_DEMO_STEP_MM:g} mm")
    parser.add_argument("--translation-gain", type=float, default=1.0,
                        help="demo-only multiplier for model TCP xyz before the step cap; at most 100")
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--max-actions", type=int, default=10)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--min-interval-s", type=float, default=2.0)
    parser.add_argument("--max-distance-from-home-mm", type=float, default=10.0)
    parser.add_argument("--after-feed", choices=("continue", "stop"), default="continue")
    parser.add_argument("--feeder-host", default="192.168.5.13")
    parser.add_argument("--feeder-local-host", default="192.168.5.11")
    parser.add_argument("--feeder-adapter", type=Path, default=Path(
        "/home/zsw/PycharmProjects/real_collection/hardware/feeder_device/udp_controller.py"))
    parser.add_argument("--side-serial", default=SIDE_SERIAL)
    parser.add_argument("--top-serial", default=TOP_SERIAL)
    parser.add_argument("--model-python", type=Path, default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint", type=Path, default=TRAINING_ROOT / "elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint", type=Path, default=TRAINING_ROOT / "piper/mixed_head_policy.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    if not args.execute:
        parser.error("bounded hardware execution requires explicit --execute")
    if args.auto_steps and not args.operator_step_mode:
        parser.error("--auto-steps requires --operator-step-mode")
    if args.elite_path_file and (args.operator_step_mode or args.auto_steps):
        parser.error("--elite-path-file uses its own finite automatic mode with HOME/EXECUTE")
    if not args.elite_path_file and args.path_start_step != 1:
        parser.error("--path-start-step requires --elite-path-file")
    args.home_pose_source = ("explicit --home-pose" if args.home_pose is not None else
                             "path/path1_pose.txt and path2_pose.txt first XYZ + fixed RPY")
    args.home_pose = list(DEFAULT_HOME_POSE if args.home_pose is None else args.home_pose)
    if not np.isfinite(args.home_pose).all():
        parser.error("--home-pose must contain six finite mm/rad values")
    speed_ceiling = 25.0 if args.elite_path_file else 5.0
    if not np.isfinite(args.home_speed) or not 0 < args.home_speed <= speed_ceiling:
        parser.error(f"--home-speed must be in (0, {speed_ceiling:g}] percent")
    if not np.isfinite(args.home_max_distance_mm) or not 0 < args.home_max_distance_mm <= 500:
        parser.error("--home-max-distance-mm must be in (0, 500] mm")
    if not np.isfinite(args.home_timeout_s) or not 10 <= args.home_timeout_s <= 300:
        parser.error("--home-timeout-s must be in [10, 300] seconds")
    if not np.isfinite(args.max_step_mm) or not 0 < args.max_step_mm <= MAX_DEMO_STEP_MM:
        parser.error(f"--max-step-mm must be in (0, {MAX_DEMO_STEP_MM:g}] mm")
    if args.elite_path_file:
        if args.translation_gain != 1.0:
            parser.error("--translation-gain must stay 1 in recorded path fallback mode")
        try:
            args.path_targets = recorded_path_targets(args.elite_path_file, args.home_pose, args.max_step_mm)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if not 1 <= args.path_start_step <= len(args.path_targets):
            parser.error(f"--path-start-step must be in [1, {len(args.path_targets)}]")
    else:
        args.path_targets = []
    if not np.isfinite(args.translation_gain) or not 0 < args.translation_gain <= 100:
        parser.error("--translation-gain must be in (0, 100]")
    if not np.isfinite(args.speed) or not 0 < args.speed <= speed_ceiling:
        parser.error(f"--speed must be in (0, {speed_ceiling:g}] percent")
    if not 1 <= args.max_actions <= 10 or not np.isfinite(args.max_duration_s) or not 1 <= args.max_duration_s <= 120:
        parser.error("demo hard limits are at most 10 actions and 120 seconds")
    if not np.isfinite(args.min_interval_s) or args.min_interval_s < 2:
        parser.error("--min-interval-s must be at least 2 seconds")
    if not np.isfinite(args.max_distance_from_home_mm) or not 0 < args.max_distance_from_home_mm <= 10:
        parser.error("--max-distance-from-home-mm must be in (0, 10]")
    if not (args.operator_step_mode or args.elite_path_file) and args.max_step_mm > args.max_distance_from_home_mm:
        parser.error("--max-step-mm cannot exceed --max-distance-from-home-mm")
    if args.side_serial == args.top_serial or not args.font.is_file():
        parser.error("two distinct camera serials and a Chinese font are required")
    run(args)


if __name__ == "__main__":
    main()
