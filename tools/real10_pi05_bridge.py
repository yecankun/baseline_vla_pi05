"""Real10 display-only bridge: hardware Python client -> persistent model Python.

There is deliberately no IK, motion, servo or feeder-send call in this file.
The policy keeps canonical IDs 0/1/2; ONLY the legacy preview subtracts one.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "real10_display_bridge_v1"
MODEL_PYTHON = "/home/zsw/miniconda3/envs/project2026-pi/bin/python"
TRAINING_ROOT = ROOT / "simulation_output/real10_pi05_train_v1"
SIDE_SERIAL = "250122079856"
TOP_SERIAL = "317222072584"


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def legacy_piper_command(intent_id):
    if type(intent_id) is not int or intent_id not in (0, 1, 2):
        raise ValueError("canonical piper_intent_id must be int 0/1/2")
    return intent_id - 1


def controller_preview(prediction, pose):
    delta = np.asarray(prediction["elite_tcp_delta_6d"], dtype=np.float64)
    current = np.asarray(pose, dtype=np.float64)
    if delta.shape != (6,) or current.shape != (6,) or not np.isfinite([delta, current]).all():
        raise ValueError("pose/delta must contain six finite values")
    if np.any(delta[3:] != 0):
        raise ValueError("real10 bridge expects the trained zero-rotation output")
    intent = prediction["piper_intent_id"]
    command = legacy_piper_command(intent)
    return {
        "piper_step_command": command,
        "piper_command_label": {-1: "retract", 0: "hold", 1: "feed"}[command],
        "mapping": "piper_step_command = canonical piper_intent_id - 1",
        "outside_real10_training_intents": intent == 0,
        "elite_target_tcp_pose_6d_preview": (current + delta).tolist(),
        "elite_translation_norm_mm": float(np.linalg.norm(delta[:3])),
        "units": "xyz_mm, rpy_rad; displacement, not velocity",
        "target_is_unclipped_preview_not_executable_command": True,
        "execution_enabled": False,
        "dispatch_status": "not_requested_display_only",
        "hardware_executed": False,
    }


def encode_image(image):
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected uint8 BGR image")
    # Exactly the existing real10 square INTER_AREA preprocessing; lossless PNG
    # keeps IPC small. The policy's second 224->224 resize is an identity.
    small = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", small)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(encoded).decode("ascii")


def decode_image(encoded):
    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (224, 224, 3):
        raise ValueError("wire image must decode to 224x224 BGR")
    return image


def worker_main(args):
    protocol_out = sys.stdout

    def send(value):
        print(json.dumps(value, allow_nan=False), file=protocol_out, flush=True)

    # Optional diagnostic traces go only to worker.log, never protocol stdout.
    if args.startup_trace:
        import faulthandler
        faulthandler.dump_traceback_later(30, repeat=True, file=sys.stderr)
    try:
        # Model/library prints must not corrupt the JSON-lines reply stream.
        with redirect_stdout(sys.stderr):
            from real10_pi05_policy import Real10PI05Policy
            model = Real10PI05Policy(args.elite_checkpoint, args.piper_checkpoint, device=args.device)
    finally:
        if args.startup_trace:
            faulthandler.cancel_dump_traceback_later()
    send({"kind": "ready", "version": VERSION, "pid": os.getpid(), "metadata": model.metadata})
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["request_id"]
            inputs = {
                "side_bgr": decode_image(request["side_png"]),
                "top_bgr": decode_image(request["top_png"]),
                "elite_tcp_pose_6d": request["elite_tcp_pose_6d"],
                "task": request["task"],
                "previous_controller_state": request["previous_controller_state"],
            }
            with redirect_stdout(sys.stderr):
                prediction = model.predict(**inputs)
                # Observation-only echo makes missing/history encoding auditable.
                state = model.observation(**inputs)["observation.state"].tolist()
            send({"kind": "prediction", "request_id": request_id,
                  "prediction": prediction, "state_32": state})
        except Exception as exc:
            send({"kind": "error", "request_id": request_id, "error": f"{type(exc).__name__}: {exc}"})


class ModelProcess:
    def __init__(self, args, log_path):
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                   PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.log = Path(log_path).open("w", encoding="utf-8")
        command = [str(args.model_python), "-B", "-u", str(Path(__file__).resolve()), "--worker",
                   "--elite-checkpoint", str(args.elite_checkpoint),
                   "--piper-checkpoint", str(args.piper_checkpoint), "--device", args.device]
        if getattr(args, "startup_trace", False):
            command.append("--startup-trace")
        try:
            self.process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.log,
                                            text=True, encoding="utf-8", bufsize=1)
        except Exception:
            self.log.close()
            raise
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.ready = self.receive(getattr(args, "model_load_timeout_s", 120))
            if self.ready.get("kind") != "ready" or self.ready.get("version") != VERSION:
                raise RuntimeError(f"unexpected worker startup: {self.ready}")
        except Exception:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def receive(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("model worker timeout; inspect worker.log") from exc
        if line is None:
            raise RuntimeError("model worker exited; inspect worker.log")
        reply = json.loads(line)
        if reply.get("kind") == "error":
            raise RuntimeError(reply["error"])
        return reply

    def predict(self, request_id, images, pose, task, previous):
        request = {"request_id": request_id, "task": task, "elite_tcp_pose_6d": pose,
                   "previous_controller_state": previous,
                   **{f"{view}_png": encode_image(images[view]) for view in ("side", "top")}}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        reply = self.receive(30)
        if reply.get("kind") != "prediction" or reply.get("request_id") != request_id:
            raise RuntimeError("worker response does not match this observation")
        return reply

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        self.log.close()


class LatestColorCamera:
    """Color only, with the training camera serial/profile; never starts depth."""
    def __init__(self, serial):
        import pyrealsense2 as rs
        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 15)
        self.pipeline.start(config)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_count = 0
        self.error = None
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        try:
            while not self.stop.is_set():
                color = self.pipeline.wait_for_frames(5000).get_color_frame()
                if not color:
                    raise RuntimeError("missing color frame")
                frame = (np.asanyarray(color.get_data()).copy(), time.time())
                with self.lock:
                    self.frame = frame
                    self.frame_count += 1
        except Exception as exc:
            self.error = exc

    def read(self):
        # Let RealSense auto exposure/white balance settle before policy input.
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError(f"camera {self.serial}: {self.error}")
            with self.lock:
                if self.frame is not None and self.frame_count >= 45:
                    image, timestamp = self.frame
                    return image.copy(), timestamp
            time.sleep(.01)
        raise TimeoutError(f"camera {self.serial} has no frame")

    def close(self):
        self.stop.set()
        self.pipeline.stop()
        self.thread.join(timeout=6)


def read_controller_snapshot(path, max_age):
    """Measured controller state only, NEVER predicted intent or an invented count."""
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    age = time.time() - float(value["timestamp"])
    if not 0 <= age <= max_age:
        return None
    count, busy = value["piper_step_after_command"], value["piper_busy"]
    if not isinstance(count, (int, float)) or isinstance(count, bool) or not np.isfinite(count) or count < 0:
        raise ValueError("controller count must be finite and non-negative")
    if type(busy) is not bool:
        raise ValueError("controller busy must be bool")
    return {"piper_step_after_command": count, "piper_busy": busy}


def render_preview(images, row, font_path):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), 23)
    small = ImageFont.truetype(str(font_path), 18)
    canvas = Image.new("RGB", (1280, 720), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 12), "真实10条模型桥接｜仅显示：无机械臂 / 递丝指令", font=font, fill="#86efac")
    for number, view in enumerate(("side", "top")):
        picture = Image.fromarray(cv2.cvtColor(images[view], cv2.COLOR_BGR2RGB))
        picture.thumbnail((610, 345))
        x = 22 + number * 630
        draw.text((x, 54), f"{'侧视' if view == 'side' else '俯视'} {view}", font=font, fill="white")
        canvas.paste(picture, (x + (610-picture.width)//2, 90 + (345-picture.height)//2))
    prediction, mapped = row["policy_output"], row["legacy_controller_preview"]
    dx, dy, dz = prediction["elite_tcp_delta_6d"][:3]
    direction = "左" if row["observation"]["task"] == "left" else "右"
    history = "有效（上一观测快照）" if row["state_32"][27] else "缺失，按训练编码填零 / 有效位0"
    lines = [
        f"来源：{'记录回放' if row['source'] == 'replay' else '现场只读'}  |  目标：{direction}支  |  样本：{row['sequence']}",
        f"Elite 平移 Δxyz（毫米）：[{dx:+.4f}, {dy:+.4f}, {dz:+.4f}]；旋转保持不变",
        f"Piper 原始 ID {prediction['piper_intent_id']} → 减一后 {mapped['piper_step_command']:+d} = {mapped['piper_command_label']}（未发送）",
        f"递丝历史：{history}",
        f"推理往返 {row['roundtrip_seconds']*1000:.1f} ms；位移未限幅，此预览不是可执行关节命令",
        "按 Q / Esc 退出。程序不求逆解、不移动、不递丝，不因预测 feed 增加历史次数。",
    ]
    for index, line in enumerate(lines):
        draw.text((22, 456 + index*37), line, font=small, fill="#e5e7eb")
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def run_client(args):
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    worker, cameras, ec = None, {}, None
    count, status, error = 0, "completed", None
    dump_json(args.out / "status.json", {"status": "starting", "hardware_executed": False})
    try:
        # Replay does not import or instantiate any camera or robot SDK.
        replay = json.loads(args.input_json.read_text(encoding="utf-8")) if args.source == "replay" else None
        task = replay["task"] if replay is not None else args.task
        print("正在加载常驻模型（约30秒）；本程序仅显示，绝不下发动作。", flush=True)
        worker = ModelProcess(args, args.out / "worker.log")
        if replay is None:
            from elite import EC
            cameras["side"] = LatestColorCamera(args.side_serial)
            cameras["top"] = LatestColorCamera(args.top_serial)
            try:
                ec = EC(ip=args.elite_ip, auto_connect=True)
            except SystemExit as exc:
                # This installed SDK exits the interpreter on connect failure.
                raise RuntimeError("Elite SDK connection failed; verify --elite-ip") from exc
        metadata = {
            "schema": VERSION, "source": args.source, "task": task,
            "client_python": sys.executable, "worker": worker.ready,
            "input_json": str(args.input_json) if replay is not None else None,
            "side_serial": args.side_serial if replay is None else None,
            "top_serial": args.top_serial if replay is None else None,
            "elite_ip": args.elite_ip if replay is None else None,
            "camera_profile": "1920x1080 BGR8 15fps color-only" if replay is None else None,
            "wire_images": "224x224 INTER_AREA then lossless BGR PNG; no crop/augmentation",
            "controller_state_file": str(args.controller_state_json) if args.controller_state_json else None,
            "history": "previous observed controller snapshot; missing => state[27]=0, not predicted actions",
            "mode": "display_only_no_actuation_code", "visual_status": "not_viewed",
            "hardware_executed": False,
        }
        dump_json(args.out / "metadata.json", metadata)
        previous = None
        with (args.out / "predictions.jsonl").open("x", encoding="utf-8") as log:
            while args.samples == 0 or count < args.samples:
                iteration_started = time.monotonic()
                current_controller = None
                if replay is not None:
                    images = {view: cv2.imread(str(args.input_json.parent / replay[f"{view}_image"]))
                              for view in ("side", "top")}
                    pose = replay["elite_tcp_pose_6d"]
                    previous = replay.get("previous_controller_state")
                    timing = {"source": "recorded_not_live", "timestamp": None}
                else:
                    frames = {view: camera.read() for view, camera in cameras.items()}
                    images = {view: frame[0] for view, frame in frames.items()}
                    pose_started = time.time()
                    pose = [float(value) for value in ec.current_pose]
                    now = time.time()
                    ages = {view: now - frame[1] for view, frame in frames.items()}
                    if any(age < 0 or age > args.max_observation_age for age in ages.values()) or now-pose_started > args.max_observation_age:
                        raise RuntimeError(f"stale observation: camera host ages={ages}, pose query={now-pose_started:.3f}s")
                    timing = {"source": "host_receive_times_not_exposure_sync", "timestamp": now,
                              "camera_host_timestamps": {view: frame[1] for view, frame in frames.items()},
                              "pose_query_started": pose_started, "pose_query_finished": now}
                    # Snapshot is captured before this prediction and becomes
                    # history only for the NEXT prediction, never this label.
                    current_controller = read_controller_snapshot(args.controller_state_json, args.max_observation_age)
                if task not in ("left", "right"):
                    raise ValueError("task must be left/right")
                start = time.perf_counter()
                reply = worker.predict(count, images, pose, task, previous)
                elapsed = time.perf_counter() - start
                row = {"schema": VERSION, "sequence": count, "source": args.source,
                       "observation": {"task": task, "elite_tcp_pose_6d": pose,
                                       "previous_controller_state": previous, "timing": timing},
                       "policy_output": reply["prediction"], "state_32": reply["state_32"],
                       "legacy_controller_preview": controller_preview(reply["prediction"], pose),
                       "roundtrip_seconds": elapsed, "hardware_executed": False}
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                preview = render_preview(images, row, args.font)
                if not cv2.imwrite(str(args.out / "preview.png"), preview):
                    raise RuntimeError("preview image write failed")
                print(json.dumps({"sequence": count, "elite_tcp_delta_6d": row["policy_output"]["elite_tcp_delta_6d"],
                                  "canonical_piper_intent_id": row["policy_output"]["piper_intent_id"],
                                  "legacy_piper_step_command": row["legacy_controller_preview"]["piper_step_command"],
                                  "hardware_executed": False}), flush=True)
                count += 1
                previous = current_controller
                if not args.headless:
                    cv2.imshow("Real10 bridge - DISPLAY ONLY", preview)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
                remaining = args.interval - (time.monotonic() - iteration_started)
                if remaining > 0 and (args.samples == 0 or count < args.samples):
                    time.sleep(remaining)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup = [(f"camera:{name}", camera.close) for name, camera in cameras.items()]
        if ec is not None:
            cleanup.append(("Elite read connection", ec.disconnect_ETController))
        if worker is not None:
            cleanup.append(("model subprocess", worker.close))
        if not args.headless:
            cleanup.append(("display", cv2.destroyAllWindows))
        cleanup_errors = []
        for name, close in cleanup:
            try:
                close()
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        if cleanup_errors:
            status = "failed"
        dump_json(args.out / "status.json", {"status": status, "error": error, "predictions": count,
                  "elapsed_seconds": time.time()-started, "hardware_executed": False,
                  "cleanup_errors": cleanup_errors, "visual_status": "not_viewed"})
        if cleanup_errors and error is None:
            raise RuntimeError(f"cleanup incomplete: {cleanup_errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--startup-trace", action="store_true", help="write model startup stack traces to worker.log every 30 s")
    parser.add_argument("--source", choices=("replay", "live"), default="replay")
    parser.add_argument("--input-json", type=Path, help="same observation-only JSON as real10_pi05_policy.py")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-python", type=Path, default=Path(MODEL_PYTHON))
    parser.add_argument("--elite-checkpoint", type=Path, default=TRAINING_ROOT / "elite/final_policy.pt")
    parser.add_argument("--piper-checkpoint", type=Path, default=TRAINING_ROOT / "piper/mixed_head_policy.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=("left", "right"))
    parser.add_argument("--elite-ip", help="required for live read-only pose; never guessed from old defaults")
    parser.add_argument("--side-serial", default=SIDE_SERIAL)
    parser.add_argument("--top-serial", default=TOP_SERIAL)
    parser.add_argument("--controller-state-json", type=Path, help="optional timestamped observed state; never predicted commands")
    parser.add_argument("--max-observation-age", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=.5, help="display refresh only, NOT a control cadence")
    parser.add_argument("--samples", type=int, help="replay default3; live default0 (until Q/Ctrl-C)")
    parser.add_argument("--headless", action="store_true", help="no GUI; logs and preview.png are still saved")
    default_font = "C:/Windows/Fonts/msyh.ttc" if os.name == "nt" else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    parser.add_argument("--font", type=Path, default=Path(default_font))
    args = parser.parse_args()
    if args.worker:
        worker_main(args)
        return
    if args.out is None or (args.source == "replay" and args.input_json is None):
        parser.error("--out is required; replay additionally requires --input-json")
    if args.source == "live" and (not args.elite_ip or not args.task):
        parser.error("live requires explicit --elite-ip and --task")
    if args.source == "live" and (args.input_json is not None or args.side_serial == args.top_serial):
        parser.error("live needs two distinct cameras and cannot use --input-json")
    args.samples = args.samples if args.samples is not None else (3 if args.source == "replay" else 0)
    if args.samples < 0 or not np.isfinite(args.interval) or args.interval < 0 or not np.isfinite(args.max_observation_age) or args.max_observation_age <= 0:
        parser.error("invalid samples, interval or max-observation-age")
    if not args.font.is_file():
        parser.error("Chinese display font is missing; specify --font")
    run_client(args)


if __name__ == "__main__":
    main()
