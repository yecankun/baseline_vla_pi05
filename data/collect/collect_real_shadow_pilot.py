from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPER_LABELS = {-1: "retract", 0: "hold", 1: "feed"}
CONTROL_SOURCES = (
    "auto",
    "none",
    "human",
    "senior_script",
    "collector",
    "feeder_device",
    "external_script",
    "other",
)
DEFAULT_ELITE_FIXED_RPY = [3.0661665148309702, 0.03647610536354759, 0.06842115274422467]


@dataclass
class TimedFrame:
    image: np.ndarray
    timestamp: float
    position_px: list[float] | None = None
    depth_image: np.ndarray | None = None
    depth_timestamp: float | None = None
    color_intrinsics: dict[str, Any] | None = None
    depth_intrinsics: dict[str, Any] | None = None
    depth_scale: float | None = None
    camera_serial: str | None = None


class MockCamera:
    def __init__(self, name: str, width: int = 640, height: int = 480) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.index = 0

    def read(self) -> TimedFrame:
        timestamp = time.time()
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:, :] = (55, 85, 70)
        cv2.putText(
            image,
            f"{self.name} mock {self.index}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        self.index += 1
        return TimedFrame(image=image, timestamp=timestamp, position_px=None)

    def close(self) -> None:
        return

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "mock",
            "color_profile": {"width": self.width, "height": self.height, "fps": None},
            "depth_enabled": False,
        }


class OpenCVCamera:
    def __init__(
        self,
        camera_id: int,
        name: str,
        *,
        width: int,
        height: int,
        fps: int,
        latest_frame: bool = True,
    ) -> None:
        self.name = name
        self.camera_id = int(camera_id)
        self.cap = cv2.VideoCapture(int(camera_id))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {name} OpenCV camera id {camera_id}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.actual_width = int(round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.actual_height = int(round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        print(
            f"{name} OpenCV camera {camera_id}: requested={width}x{height}@{fps}, "
            f"actual={self.actual_width}x{self.actual_height}@{self.actual_fps:.3f}"
        )
        self.latest_frame = bool(latest_frame)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.frame: TimedFrame | None = None
        if self.latest_frame:
            self.thread = threading.Thread(target=self._capture_loop, name=f"opencv-camera-{name}", daemon=True)
            self.thread.start()

    def _read_direct(self) -> TimedFrame:
        ok, frame = self.cap.read()
        timestamp = time.time()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame from {self.name} camera")
        return TimedFrame(image=frame, timestamp=timestamp, position_px=None)

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self._read_direct()
            except Exception as exc:  # pragma: no cover - hardware path
                print(f"warning: {self.name} camera background read failed: {exc}")
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame

    def read(self) -> TimedFrame:
        if not self.latest_frame:
            return self._read_direct()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self.lock:
                frame = self.frame
            if frame is not None:
                return TimedFrame(
                    image=frame.image.copy(),
                    timestamp=frame.timestamp,
                    position_px=frame.position_px,
                    depth_image=None if frame.depth_image is None else frame.depth_image.copy(),
                    depth_timestamp=frame.depth_timestamp,
                    color_intrinsics=frame.color_intrinsics,
                    depth_intrinsics=frame.depth_intrinsics,
                    depth_scale=frame.depth_scale,
                    camera_serial=frame.camera_serial,
                )
            time.sleep(0.01)
        raise RuntimeError(f"No frame arrived from {self.name} camera within 2 seconds")

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "opencv",
            "camera_id": self.camera_id,
            "color_profile": {
                "width": self.actual_width,
                "height": self.actual_height,
                "fps": self.actual_fps,
            },
            "depth_enabled": False,
        }


class RealSenseCamera:
    def __init__(
        self,
        camera_id: int,
        name: str,
        *,
        serial: str,
        color_width: int,
        color_height: int,
        color_fps: int,
        depth_width: int,
        depth_height: int,
        depth_fps: int,
        latest_frame: bool,
    ) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.name = name
        self.pipeline = rs.pipeline()
        config = rs.config()
        serial = str(serial).strip()
        if not serial:
            devices = [
                device.get_info(rs.camera_info.serial_number)
                for device in rs.context().query_devices()
                if device.get_info(rs.camera_info.name).lower() != "platform camera"
            ]
            if not 0 <= int(camera_id) < len(devices):
                raise RuntimeError(
                    f"{name} RealSense camera index {camera_id} is unavailable; found serials={devices}"
                )
            serial = devices[int(camera_id)]
        self.serial = serial
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            int(color_width),
            int(color_height),
            rs.format.bgr8,
            int(color_fps),
        )
        config.enable_stream(
            rs.stream.depth,
            int(depth_width),
            int(depth_height),
            rs.format.z16,
            int(depth_fps),
        )
        try:
            self.profile = self.pipeline.start(config)
        except RuntimeError as exc:
            available_serials: list[str] = []
            try:
                available_serials = [
                    device.get_info(rs.camera_info.serial_number)
                    for device in rs.context().query_devices()
                ]
            except Exception:
                pass
            raise RuntimeError(
                f"{name} RealSense pipeline start failed for SDK serial {serial!r}; "
                f"available SDK serials={available_serials}. Use rs.camera_info.serial_number, "
                "not the udev ID_SERIAL_SHORT value."
            ) from exc
        self.align = rs.align(rs.stream.color)
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.color_profile = self._profile_to_dict(color_profile)
        self.depth_profile = self._profile_to_dict(depth_profile)
        self.color_intrinsics = self._intrinsics_to_dict(color_profile.get_intrinsics())
        self.depth_intrinsics_native = self._intrinsics_to_dict(depth_profile.get_intrinsics())
        depth_to_color = depth_profile.get_extrinsics_to(color_profile)
        self.depth_to_color_extrinsics = {
            "rotation": [float(value) for value in depth_to_color.rotation],
            "translation_m": [float(value) for value in depth_to_color.translation],
        }
        print(
            f"{name} RealSense serial={serial}: color={self.color_profile}, "
            f"depth={self.depth_profile}, depth_scale={self.depth_scale}"
        )
        self.latest_frame = bool(latest_frame)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.frame: TimedFrame | None = None
        if self.latest_frame:
            self.thread = threading.Thread(
                target=self._capture_loop,
                name=f"realsense-camera-{name}",
                daemon=True,
            )
            self.thread.start()

    @staticmethod
    def _intrinsics_to_dict(intr: Any) -> dict[str, Any]:
        return {
            "width": getattr(intr, "width", None),
            "height": getattr(intr, "height", None),
            "fx": getattr(intr, "fx", None),
            "fy": getattr(intr, "fy", None),
            "ppx": getattr(intr, "ppx", None),
            "ppy": getattr(intr, "ppy", None),
            "model": str(getattr(intr, "model", "")),
            "coeffs": list(getattr(intr, "coeffs", []) or []),
        }

    @staticmethod
    def _profile_to_dict(profile: Any) -> dict[str, Any]:
        return {
            "width": int(profile.width()),
            "height": int(profile.height()),
            "fps": int(profile.fps()),
            "format": str(profile.format()),
        }

    @staticmethod
    def _copy_frame(frame: TimedFrame) -> TimedFrame:
        return TimedFrame(
            image=frame.image.copy(),
            timestamp=frame.timestamp,
            position_px=frame.position_px,
            depth_image=None if frame.depth_image is None else frame.depth_image.copy(),
            depth_timestamp=frame.depth_timestamp,
            color_intrinsics=frame.color_intrinsics,
            depth_intrinsics=frame.depth_intrinsics,
            depth_scale=frame.depth_scale,
            camera_serial=frame.camera_serial,
        )

    def _read_direct(self) -> TimedFrame:
        frames = self.pipeline.wait_for_frames(5000)
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"{self.name} RealSense did not return aligned color and depth frames")
        timestamp = time.time()
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        return TimedFrame(
            image=color_image,
            timestamp=timestamp,
            position_px=None,
            depth_image=depth_image,
            depth_timestamp=timestamp,
            color_intrinsics=self._intrinsics_to_dict(color_intrinsics),
            depth_intrinsics=self._intrinsics_to_dict(depth_intrinsics),
            depth_scale=self.depth_scale,
            camera_serial=self.serial,
        )

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self._read_direct()
            except Exception as exc:  # pragma: no cover - hardware path
                if not self.stop_event.is_set():
                    print(f"warning: {self.name} RealSense background read failed: {exc}")
                    time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame

    def read(self) -> TimedFrame:
        if not self.latest_frame:
            return self._read_direct()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self.lock:
                frame = self.frame
            if frame is not None:
                return self._copy_frame(frame)
            time.sleep(0.01)
        raise RuntimeError(f"No frame arrived from {self.name} RealSense within 5 seconds")

    def close(self) -> None:
        self.stop_event.set()
        self.pipeline.stop()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "realsense",
            "serial": self.serial,
            "color_profile": self.color_profile,
            "depth_profile_native": self.depth_profile,
            "depth_aligned_to": "color",
            "color_intrinsics": self.color_intrinsics,
            "depth_intrinsics_native": self.depth_intrinsics_native,
            "aligned_depth_intrinsics": self.color_intrinsics,
            "depth_to_color_extrinsics": self.depth_to_color_extrinsics,
            "depth_scale_m_per_unit": self.depth_scale,
            "depth_storage": "uint16_png_raw_units",
            "depth_enabled": True,
            "latest_frame_buffer": self.latest_frame,
        }


class DuplicateCamera:
    def __init__(self) -> None:
        self.last_source: TimedFrame | None = None

    def set_source(self, source: TimedFrame) -> None:
        self.last_source = source

    def read(self) -> TimedFrame:
        if self.last_source is None:
            raise RuntimeError("DuplicateCamera has no source frame yet")
        return TimedFrame(
            image=self.last_source.image.copy(),
            timestamp=self.last_source.timestamp,
            position_px=self.last_source.position_px,
            depth_image=None if self.last_source.depth_image is None else self.last_source.depth_image.copy(),
            depth_timestamp=self.last_source.depth_timestamp,
            color_intrinsics=self.last_source.color_intrinsics,
            depth_intrinsics=self.last_source.depth_intrinsics,
            depth_scale=self.last_source.depth_scale,
            camera_serial=self.last_source.camera_serial,
        )

    def close(self) -> None:
        return

    def metadata(self) -> dict[str, Any]:
        source_metadata = None
        if self.last_source is not None:
            source_metadata = {
                "camera_serial": self.last_source.camera_serial,
                "depth_enabled": self.last_source.depth_image is not None,
            }
        return {"source": "duplicate_side", "source_metadata": source_metadata}


def read_newer_frame(camera: Any, previous_timestamp: float | None, timeout_s: float = 5.0) -> TimedFrame:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        frame = camera.read()
        if previous_timestamp is None or frame.timestamp > previous_timestamp:
            return frame
        time.sleep(0.005)
    raise RuntimeError(
        f"Camera frame timestamp did not advance within {timeout_s:.1f} seconds; "
        f"last timestamp={previous_timestamp}"
    )


class MockPoseReader:
    def __init__(self) -> None:
        self.index = 0

    def read(self) -> tuple[list[float], float]:
        timestamp = time.time()
        pose = [float(self.index), 0.0, 320.0, 3.06617, 0.03647, 0.06842]
        self.index += 1
        return pose, timestamp

    def read_nonblocking(self) -> tuple[list[float], float, bool]:
        pose, timestamp = self.read()
        return pose, timestamp, False


class ElitePoseReader:
    def __init__(self, ip: str) -> None:
        from elite import EC

        self.ec = EC(ip=ip, auto_connect=True)
        self.lock = threading.Lock()
        self.last_pose: list[float] | None = None
        self.last_timestamp: float | None = None

    def read(self) -> tuple[list[float], float]:
        with self.lock:
            timestamp = time.time()
            pose = list(self.ec.current_pose)
        if len(pose) != 6:
            raise RuntimeError(f"Elite current_pose should have 6 values, got {len(pose)}")
        pose = [float(x) for x in pose]
        self.last_pose = pose
        self.last_timestamp = timestamp
        return pose, timestamp

    def read_nonblocking(self) -> tuple[list[float], float, bool]:
        acquired = self.lock.acquire(blocking=False)
        if acquired:
            try:
                timestamp = time.time()
                pose = list(self.ec.current_pose)
            finally:
                self.lock.release()
            if len(pose) != 6:
                raise RuntimeError(f"Elite current_pose should have 6 values, got {len(pose)}")
            pose = [float(x) for x in pose]
            self.last_pose = pose
            self.last_timestamp = timestamp
            return pose, timestamp, False
        if self.last_pose is not None and self.last_timestamp is not None:
            return list(self.last_pose), float(self.last_timestamp), True
        pose, timestamp = self.read()
        return pose, timestamp, False


def load_elite_path(path_file: Path, fixed_rpy: list[float]) -> list[list[float]]:
    poses: list[list[float]] = []
    with path_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = [float(value) for value in stripped.replace(",", " ").split()]
            if len(values) == 3:
                poses.append([*values, *fixed_rpy])
            elif len(values) == 6:
                poses.append(values)
            else:
                raise ValueError(
                    f"{path_file}:{line_no} should contain 3 xyz values or 6 xyz+rpy values, got {len(values)}"
                )
    if not poses:
        raise ValueError(f"No Elite path points found in {path_file}")
    return poses


def xyz_distance_mm(a: list[float], b: list[float]) -> float:
    return float(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5)


def print_elite_path_summary(poses: list[list[float]], start_index: int, end_index: int, max_step_mm: float) -> None:
    print(f"Elite path loaded: points={len(poses)}, selected=[{start_index}, {end_index}]")
    print(f"Elite path first selected pose: {poses[start_index]}")
    print(f"Elite path last selected pose:  {poses[end_index]}")
    if start_index < end_index:
        distances = [xyz_distance_mm(poses[i - 1], poses[i]) for i in range(start_index + 1, end_index + 1)]
        print(
            "Elite path consecutive xyz step mm: "
            f"min={min(distances):.3f}, median={sorted(distances)[len(distances)//2]:.3f}, max={max(distances):.3f}"
        )
        large = [(start_index + 1 + i, d) for i, d in enumerate(distances) if d > max_step_mm]
        if large:
            print(f"WARNING: {len(large)} Elite path jumps exceed --elite-path-max-step-mm={max_step_mm}")
            for index, distance in large[:10]:
                print(f"  jump to index {index}: {distance:.3f} mm")


class ElitePathPlayback:
    def __init__(
        self,
        args: argparse.Namespace,
        pose_reader: Any,
        piper_executor: Any | None = None,
        command_ec: Any | None = None,
        command_lock: threading.Lock | None = None,
        connection_mode: str = "shared_pose_reader",
    ) -> None:
        self.args = args
        self.piper_executor = piper_executor
        self.ec = command_ec if command_ec is not None else getattr(pose_reader, "ec", None)
        self.ec_lock = command_lock if command_lock is not None else getattr(pose_reader, "lock", threading.Lock())
        self.connection_mode = connection_mode
        self.poses = load_elite_path(Path(args.elite_path_file), [float(v) for v in args.elite_path_fixed_rpy])
        self.start_index = int(args.elite_path_start_index)
        self.end_index = len(self.poses) - 1 if int(args.elite_path_end_index) < 0 else int(args.elite_path_end_index)
        if not (0 <= self.start_index < len(self.poses)):
            raise ValueError(f"--elite-path-start-index must be within [0, {len(self.poses) - 1}]")
        if not (self.start_index <= self.end_index < len(self.poses)):
            raise ValueError(f"--elite-path-end-index must be within [{self.start_index}, {len(self.poses) - 1}]")
        print_elite_path_summary(
            self.poses,
            self.start_index,
            self.end_index,
            max_step_mm=float(args.elite_path_max_step_mm),
        )
        if self.start_index < self.end_index:
            distances = [
                xyz_distance_mm(self.poses[i - 1], self.poses[i])
                for i in range(self.start_index + 1, self.end_index + 1)
            ]
            if max(distances) > float(args.elite_path_max_step_mm) and not args.elite_path_allow_large_jump:
                raise ValueError(
                    "Elite path has a large jump. Re-check the path or pass --elite-path-allow-large-jump explicitly."
                )
        if self.ec is None and args.elite_path_execute:
            raise RuntimeError("--elite-path-execute requires --pose-source elite so the collector can reuse the Elite SDK connection")

        self.start_guard_pose: list[float] | None = None
        self.start_guard_distance_mm: float | None = None
        self.start_approach_executed = False
        self.start_approach_status = "not_requested"
        self.start_approach_requested_joints: list[float] | None = None
        self.start_approach_motion = "move_line"
        self.start_approach_command_timestamp: float | None = None
        self.start_approach_final_pose: list[float] | None = None
        self.start_approach_final_distance_mm: float | None = None
        if args.elite_path_execute:
            current_pose, _timestamp = pose_reader.read()
            self.start_guard_pose = [float(v) for v in current_pose]
            self.start_guard_distance_mm = xyz_distance_mm(self.start_guard_pose, self.poses[self.start_index])
            print(
                "Elite path start guard: "
                f"current_pose={self.start_guard_pose}, "
                f"first_selected_pose={self.poses[self.start_index]}, "
                f"xyz_distance_mm={self.start_guard_distance_mm:.3f}, "
                f"limit_mm={float(args.elite_path_max_start_distance_mm):.3f}"
            )
            if (
                self.start_guard_distance_mm > float(args.elite_path_max_start_distance_mm)
                and not args.elite_path_approach_start
            ):
                raise ValueError(
                    "Elite current pose is too far from the first selected path point: "
                    f"{self.start_guard_distance_mm:.3f} mm > "
                    f"--elite-path-max-start-distance-mm={float(args.elite_path_max_start_distance_mm):.3f}. "
                    "Use --elite-path-approach-start for an explicit unrecorded move to the selected start pose."
                )
            if args.elite_path_approach_start:
                target_pose = self.poses[self.start_index]
                self.start_approach_status = "running"
                with self.ec_lock:
                    target_joint = self.ec.get_inverse_kinematic(pose=target_pose)
                    self.start_approach_requested_joints = [float(v) for v in target_joint]
                    self.start_approach_command_timestamp = time.time()
                    move_ok = self.ec.move_line(
                        target_joint=target_joint,
                        speed=int(round(float(args.elite_path_approach_speed))),
                        speed_type=0,
                    )
                if not move_ok:
                    self.start_approach_status = "move_line_rejected"
                    raise RuntimeError("Elite move_line rejected the automatic start approach")
                deadline = time.time() + float(args.elite_path_approach_timeout_s)
                while True:
                    final_pose, _timestamp = pose_reader.read()
                    self.start_approach_final_pose = [float(v) for v in final_pose]
                    self.start_approach_final_distance_mm = xyz_distance_mm(
                        self.start_approach_final_pose,
                        target_pose,
                    )
                    if self.start_approach_final_distance_mm <= float(args.elite_path_approach_tolerance_mm):
                        break
                    if time.time() >= deadline:
                        self.start_approach_status = "timeout"
                        raise RuntimeError(
                            "Elite start approach did not reach the selected path start within "
                            f"{float(args.elite_path_approach_tolerance_mm):.3f} mm; "
                            f"final distance={self.start_approach_final_distance_mm:.3f} mm"
                        )
                    time.sleep(0.2)
                self.start_approach_executed = True
                self.start_approach_status = "complete"
                print(
                    "Elite start approach complete: "
                    f"motion={self.start_approach_motion}, "
                    f"initial_distance_mm={self.start_guard_distance_mm:.3f}, "
                    f"final_distance_mm={self.start_approach_final_distance_mm:.3f}, "
                    f"target_pose={target_pose}, joints={self.start_approach_requested_joints}"
                )

        self.lock = threading.Lock()
        self.command_submit_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.command_threads: list[threading.Thread] = []
        self.command_busy = False
        self.next_index = self.start_index + 1 if self.start_approach_executed else self.start_index
        self.state: dict[str, Any] = {
            "elite_path_file": str(args.elite_path_file),
            "elite_path_mode": args.elite_path_mode,
            "elite_path_connection_mode": self.connection_mode,
            "elite_path_index": self.start_index if self.start_approach_executed else None,
            "elite_path_status": "approach_start_complete" if self.start_approach_executed else "configured",
            "elite_path_command_busy": False,
            "elite_path_done": self.next_index > self.end_index,
            "elite_path_error": None,
            "elite_path_start_guard_pose_6d": self.start_guard_pose,
            "elite_path_start_guard_distance_mm": self.start_guard_distance_mm,
            "elite_path_max_start_distance_mm": float(args.elite_path_max_start_distance_mm),
            "elite_path_start_approach_requested": bool(args.elite_path_approach_start),
            "elite_path_start_approach_executed": self.start_approach_executed,
            "elite_path_start_approach_status": self.start_approach_status,
            "elite_path_start_approach_motion": self.start_approach_motion,
            "elite_path_start_approach_speed_mm_s": float(args.elite_path_approach_speed),
            "elite_path_start_approach_tolerance_mm": float(args.elite_path_approach_tolerance_mm),
            "elite_path_start_approach_final_pose_6d": self.start_approach_final_pose,
            "elite_path_start_approach_final_distance_mm": self.start_approach_final_distance_mm,
            "elite_requested_tcp_pose_6d": self.poses[self.start_index] if self.start_approach_executed else None,
            "elite_requested_joints": self.start_approach_requested_joints,
            "elite_path_command_timestamp": self.start_approach_command_timestamp,
            "elite_step_piper_command": None,
            "elite_step_piper_executed": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.state)

    def _set_state(self, **updates: Any) -> None:
        with self.lock:
            self.state.update(updates)

    def _run_submitted_command(self, direction: str) -> None:
        try:
            if direction == "next":
                self.command_next()
            elif direction == "previous":
                self.command_previous()
        except Exception as exc:  # pragma: no cover - hardware path
            self._set_state(elite_path_status="error", elite_path_error=str(exc))
            print(f"ERROR: Elite keyboard command failed: {exc}")
        finally:
            with self.command_submit_lock:
                self.command_busy = False
            self._set_state(elite_path_command_busy=False)

    def submit_keyboard_command(self, direction: str) -> str:
        with self.command_submit_lock:
            if self.command_busy:
                self._set_state(elite_path_command_busy=True)
                return f"busy_skip_{direction}"
            self.command_busy = True
        self._set_state(elite_path_command_busy=True, elite_path_status=f"{direction}_submitted")
        thread = threading.Thread(
            target=self._run_submitted_command,
            args=(direction,),
            name=f"elite-path-keyboard-{direction}",
            daemon=True,
        )
        thread.start()
        self.command_threads.append(thread)
        return f"{direction}_submitted"

    def command_index(self, index: int) -> None:
        pose = self.poses[index]
        joints = None
        status = "dry_run_no_ik"
        piper_executed = None
        if self.piper_executor is not None and bool(self.args.piper_feed_on_elite_path_step):
            piper_executed = self.piper_executor.submit(1)
        if self.ec is not None:
            with self.ec_lock:
                target_joint = self.ec.get_inverse_kinematic(pose=pose)
                joints = [float(v) for v in target_joint]
                if self.args.elite_path_execute:
                    self.ec.move_joint(target_joint=target_joint, speed=float(self.args.elite_path_speed))
                    status = "move_joint_sent"
                else:
                    status = "ik_ok_dry_run"
        self._set_state(
            elite_path_index=int(index),
            elite_path_status=status,
            elite_path_done=False,
            elite_path_error=None,
            elite_requested_tcp_pose_6d=[float(v) for v in pose],
            elite_requested_joints=joints,
            elite_path_command_timestamp=time.time(),
            elite_step_piper_command=1 if piper_executed is not None else None,
            elite_step_piper_executed=piper_executed,
        )
        print(
            f"Elite path {status}: index={index}/{self.end_index}, "
            f"conn={self.connection_mode}, piper={piper_executed}, pose={pose}, joints={joints}"
        )

    def command_next(self) -> None:
        if self.next_index > self.end_index:
            self._set_state(elite_path_done=True, elite_path_status="done")
            return
        index = self.next_index
        self.command_index(index)
        self.next_index += 1
        if self.next_index > self.end_index:
            self._set_state(elite_path_done=True)

    def command_previous(self) -> None:
        self.next_index = max(self.start_index, self.next_index - 2)
        self.command_next()

    def start_auto(self) -> None:
        if self.args.elite_path_mode != "auto":
            return
        self.thread = threading.Thread(target=self._auto_loop, name="elite-path-playback", daemon=True)
        self.thread.start()

    def _auto_loop(self) -> None:
        try:
            delay = max(0.0, float(self.args.elite_path_start_delay))
            if delay > 0:
                time.sleep(delay)
            while not self.stop_event.is_set() and self.next_index <= self.end_index:
                self.command_next()
                if self.next_index > self.end_index:
                    break
                period = max(0.0, float(self.args.elite_path_period))
                if self.stop_event.wait(period):
                    break
            self._set_state(elite_path_done=self.next_index > self.end_index)
        except Exception as exc:  # pragma: no cover - hardware path
            self._set_state(elite_path_status="error", elite_path_error=str(exc))
            print(f"ERROR: Elite path playback failed: {exc}")

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        for thread in list(self.command_threads):
            thread.join(timeout=2.0)


class PiperAsyncExecutor:
    def __init__(self, piper: Any, args: argparse.Namespace) -> None:
        self.piper = piper
        self.args = args
        self.lock = threading.Lock()
        self.busy = False
        self.last_status = "idle"
        self.last_command = 0
        self.last_burst_count = 1
        self.last_start_timestamp: float | None = None
        self.last_done_timestamp: float | None = None
        self.last_error: str | None = None
        self.feed_started_count = 0
        self.retract_started_count = 0
        self.threads: list[threading.Thread] = []

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "piper_busy": bool(self.busy),
                "piper_async_status": self.last_status,
                "piper_async_command": int(self.last_command),
                "piper_async_burst_count": int(self.last_burst_count),
                "piper_async_start_timestamp": self.last_start_timestamp,
                "piper_async_done_timestamp": self.last_done_timestamp,
                "piper_async_error": self.last_error,
                "piper_feed_started_count": int(self.feed_started_count),
                "piper_retract_started_count": int(self.retract_started_count),
            }

    def submit(self, command: int, count: int = 1) -> str:
        count = max(1, int(count))
        if command == 0:
            with self.lock:
                if not self.busy:
                    self.last_status = "idle"
                    self.last_command = 0
                    self.last_burst_count = 1
            return "hold"
        with self.lock:
            if self.busy:
                label = action_label(command)
                return f"busy_skip_{label}"
            self.busy = True
            self.last_status = "running"
            self.last_command = int(command)
            self.last_burst_count = int(count)
            self.last_start_timestamp = time.time()
            self.last_done_timestamp = None
            self.last_error = None
            if command > 0:
                self.feed_started_count += count
            elif command < 0:
                self.retract_started_count += count
        thread = threading.Thread(target=self._run_command, args=(int(command), int(count)), daemon=True)
        thread.start()
        self.threads.append(thread)
        label = action_label(command)
        if count == 1:
            return f"{label}_async_started"
        return f"{label}_burst_{count}_async_started"

    def _run_command(self, command: int, count: int) -> None:
        label = action_label(command)
        status = label if count == 1 else f"{label}_burst_{count}"
        error = None
        try:
            for _ in range(max(1, int(count))):
                if command > 0:
                    self.piper.step_forward(pause_time=float(self.args.piper_pause_time))
                elif command < 0:
                    self.piper.step_backward(pause_time=float(self.args.piper_pause_time))
        except Exception as exc:  # pragma: no cover - hardware path
            status = f"{status}_error"
            error = str(exc)
            print(f"ERROR: Piper async command failed: {exc}")
        with self.lock:
            self.busy = False
            self.last_status = status
            self.last_done_timestamp = time.time()
            self.last_error = error

    def close(self) -> None:
        for thread in list(self.threads):
            thread.join(timeout=2.0)


class FeederDeviceAsyncExecutor:
    def __init__(self, feeder: Any, args: argparse.Namespace) -> None:
        self.feeder = feeder
        self.args = args
        self.lock = threading.Lock()
        self.busy = False
        self.last_status = "idle"
        self.last_command = 0
        self.last_burst_count = 1
        self.last_start_timestamp: float | None = None
        self.last_done_timestamp: float | None = None
        self.last_error: str | None = None
        self.last_response: str | None = None
        self.feed_started_count = 0
        self.retract_started_count = 0
        self.threads: list[threading.Thread] = []

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "piper_busy": bool(self.busy),
                "piper_async_status": self.last_status,
                "piper_async_command": int(self.last_command),
                "piper_async_burst_count": int(self.last_burst_count),
                "piper_async_start_timestamp": self.last_start_timestamp,
                "piper_async_done_timestamp": self.last_done_timestamp,
                "piper_async_error": self.last_error,
                "piper_feed_started_count": int(self.feed_started_count),
                "piper_retract_started_count": int(self.retract_started_count),
                "feeder_transport": "udp_json",
                "feeder_udp_transport": self.args.feeder_udp_transport,
                "feeder_host": self.args.feeder_host,
                "feeder_port": int(self.args.feeder_port),
                "feeder_step_mm_nominal": None,
                "feeder_step_mm_range": None,
                "feeder_step_calibration_status": "not_applicable_event_semantics",
                "feeder_wait_response": bool(self.args.feeder_wait_response),
                "feeder_last_response": self.last_response,
            }

    def submit(self, command: int, count: int = 1) -> str:
        count = max(1, int(count))
        if command == 0:
            with self.lock:
                if not self.busy:
                    self.last_status = "idle"
                    self.last_command = 0
                    self.last_burst_count = 1
            return "hold"
        with self.lock:
            if self.busy:
                label = action_label(command)
                return f"busy_skip_{label}"
            self.busy = True
            self.last_status = "running"
            self.last_command = int(command)
            self.last_burst_count = int(count)
            self.last_start_timestamp = time.time()
            self.last_done_timestamp = None
            self.last_error = None
            self.last_response = None
            if command > 0:
                self.feed_started_count += count
            elif command < 0:
                self.retract_started_count += count
        label = action_label(command)
        transport = str(self.args.feeder_udp_transport)
        if transport == "bash_dev_udp":
            self._run_command(int(command), int(count))
            if count == 1:
                return f"{label}_feeder_{transport}_sent"
            return f"{label}_feeder_{transport}_burst_{count}_sent"
        thread = threading.Thread(target=self._run_command, args=(int(command), int(count)), daemon=True)
        thread.start()
        self.threads.append(thread)
        if count == 1:
            return f"{label}_feeder_{transport}_async_started"
        return f"{label}_feeder_{transport}_burst_{count}_async_started"

    def _run_command(self, command: int, count: int) -> None:
        label = action_label(command)
        transport = str(self.args.feeder_udp_transport)
        status = f"{label}_feeder_{transport}" if count == 1 else f"{label}_feeder_{transport}_burst_{count}"
        error = None
        response_text = None
        try:
            for _ in range(max(1, int(count))):
                if command > 0:
                    response = self.feeder.feed_once(wait_response=bool(self.args.feeder_wait_response))
                elif command < 0:
                    response = self.feeder.retract_once(wait_response=bool(self.args.feeder_wait_response))
                else:
                    response = None
                if response:
                    response_text = response.decode("utf-8", errors="replace")
                interval = max(0.0, float(self.args.feeder_command_interval_s))
                if interval > 0:
                    time.sleep(interval)
        except Exception as exc:  # pragma: no cover - hardware path
            status = f"{status}_error"
            error = str(exc)
            print(f"ERROR: feeder UDP async command failed: {exc}")
        with self.lock:
            self.busy = False
            self.last_status = status
            self.last_done_timestamp = time.time()
            self.last_error = error
            self.last_response = response_text

    def close(self) -> None:
        for thread in list(self.threads):
            thread.join(timeout=2.0)
        self.feeder.close()


def build_camera(
    source: str,
    camera_id: int,
    name: str,
    *,
    latest_frame: bool,
    width: int,
    height: int,
    fps: int,
    realsense_serial: str,
    depth_width: int,
    depth_height: int,
    depth_fps: int,
) -> Any:
    if source == "mock":
        return MockCamera(name, width=width, height=height)
    if source == "opencv":
        return OpenCVCamera(
            camera_id,
            name,
            width=width,
            height=height,
            fps=fps,
            latest_frame=latest_frame,
        )
    if source == "realsense":
        return RealSenseCamera(
            camera_id,
            name,
            serial=realsense_serial,
            color_width=width,
            color_height=height,
            color_fps=fps,
            depth_width=depth_width,
            depth_height=depth_height,
            depth_fps=depth_fps,
            latest_frame=latest_frame,
        )
    if source == "duplicate_side":
        return DuplicateCamera()
    raise ValueError(f"Unknown camera source: {source}")


def camera_metadata(camera: Any) -> dict[str, Any]:
    metadata = getattr(camera, "metadata", None)
    if callable(metadata):
        return dict(metadata())
    return {"source": type(camera).__name__}


def build_pose_reader(args: argparse.Namespace) -> Any:
    if args.pose_source == "mock":
        return MockPoseReader()
    if args.pose_source == "elite":
        return ElitePoseReader(args.elite_ip)
    raise ValueError(f"Unknown pose source: {args.pose_source}")


def build_elite_command_connection(args: argparse.Namespace) -> tuple[Any | None, threading.Lock | None, str]:
    if args.elite_path_mode == "none":
        return None, None, "none"
    if not args.elite_path_separate_connection:
        return None, None, "shared_pose_reader"
    if args.pose_source != "elite":
        raise RuntimeError("--elite-path-separate-connection requires --pose-source elite")
    from elite import EC

    ec = EC(ip=args.elite_ip, auto_connect=True)
    return ec, threading.Lock(), "separate_command_connection"


def piper_command_from_key(key: int) -> int | None:
    if key in (ord("0"), ord("h"), ord("H")):
        return 0
    if key in (ord("1"), ord("f"), ord("F"), ord("g"), ord("G"), ord("v"), ord("V")):
        return 1
    if key in (ord("r"), ord("R"), ord("b"), ord("B")):
        return -1
    return None


def piper_burst_from_key(key: int) -> int:
    if key in (ord("g"), ord("G")):
        return 2
    if key in (ord("v"), ord("V")):
        return 3
    return 1


def key_name(key: int) -> str | None:
    if key < 0:
        return None
    if 32 <= key < 127:
        return chr(key)
    if key == 27:
        return "esc"
    return str(key)


def action_label(command: int) -> str:
    return PIPER_LABELS[int(command)]


def resolve_control_source(args: argparse.Namespace, role: str) -> str:
    value = getattr(args, f"{role}_control_source")
    if value != "auto":
        return value
    if role == "piper":
        if args.enable_feeder_device_control:
            return "feeder_device"
        if args.enable_piper_control:
            return "collector"
        if args.external_piper_control:
            return "external_script"
        return "none"
    if args.elite_path_mode != "none":
        if args.elite_path_execute:
            return "collector"
        return "collector_dry_run"
    return "human"


def elite_path_key(key: int) -> str | None:
    if key in (ord("n"), ord("N"), ord("l"), ord("L")):
        return "next"
    if key in (ord("j"), ord("J")):
        return "previous"
    return None


def is_recording_start_action(key: int) -> bool:
    return elite_path_key(key) is not None or piper_command_from_key(key) is not None


def pose_delta(current_pose: list[float], next_pose: list[float], *, zero_orientation_delta: bool) -> list[float]:
    current = np.asarray(current_pose, dtype=np.float32).reshape(6)
    target = np.asarray(next_pose, dtype=np.float32).reshape(6)
    delta = target - current
    if zero_orientation_delta:
        delta[3:] = 0.0
    return delta.astype(float).tolist()


def target_pose_from_delta(current_pose: list[float], delta: list[float]) -> list[float]:
    current = np.asarray(current_pose, dtype=np.float32).reshape(6)
    delta_arr = np.asarray(delta, dtype=np.float32).reshape(6)
    target = current + delta_arr
    return target.astype(float).tolist()


def optional_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [float(x) for x in values]


def relative(path: Path, base: Path) -> str:
    return str(path.relative_to(base).as_posix())


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def record_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Failed to write image: {path}")


def record_depth_image(path: Path, depth_image: np.ndarray) -> None:
    if depth_image.dtype != np.uint16:
        raise ValueError(f"Depth image must be uint16, got {depth_image.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), depth_image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Failed to write depth image: {path}")


def preview(side: np.ndarray, top: np.ndarray, record: dict[str, Any]) -> int:
    side_small = cv2.resize(side, (480, 360), interpolation=cv2.INTER_AREA)
    top_small = cv2.resize(top, (480, 360), interpolation=cv2.INTER_AREA)
    combined = np.hstack([side_small, top_small])
    elite_status = record.get("elite_status", "")
    status = (
        f"step={record['step']} task={record['task']} "
        f"piper={record['reference_action']['piper_command_label']} "
        f"{elite_status} "
        "keys: f feed1, g feed2, v feed3, n elite next, j prev, q stop"
    )
    cv2.putText(combined, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
    if record.get("recording_waiting_for_action"):
        cv2.putText(
            combined,
            "WAITING TO RECORD: press an action key",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("real shadow pilot capture", combined)
    return cv2.waitKey(1) & 0xFF


def execute_piper_command(command: int, piper: Any, args: argparse.Namespace) -> str:
    if args.external_piper_control:
        if command > 0:
            return "external_feed"
        if command < 0:
            return "external_retract"
        return "external_hold"
    if piper is None:
        return "logged_only"
    if command > 0:
        piper.step_forward(pause_time=float(args.piper_pause_time))
        return "feed"
    if command < 0:
        piper.step_backward(pause_time=float(args.piper_pause_time))
        return "retract"
    return "hold"


def command_started_real_piper_step(command: int, executed: str) -> bool:
    if command == 0:
        return False
    return "async_started" in executed or executed.endswith("_sent") or executed in {
        "feed",
        "retract",
        "external_feed",
        "external_retract",
    }


def real_piper_command_affects_step(args: argparse.Namespace) -> bool:
    return bool(args.enable_piper_control or args.enable_feeder_device_control or args.external_piper_control)


def piper_insertion_length_mm(args: argparse.Namespace, piper_step: int) -> float | None:
    return None


def robot_command_mode(args: argparse.Namespace) -> str:
    if args.enable_feeder_device_control:
        return "feeder_device_control_enabled"
    if args.enable_piper_control:
        return "piper_control_enabled"
    if args.external_piper_control:
        return "external_piper_control"
    return "log_only"


def create_manifest(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    return {
        "mode": "real_shadow_pilot",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": args.task,
        "out_dir": str(out_dir),
        "camera_setup": {
            "side_source": args.side_source,
            "side_camera_id": args.side_camera_id,
            "side_realsense_serial": args.side_realsense_serial or None,
            "side_color_requested": {
                "width": args.side_width,
                "height": args.side_height,
                "fps": args.side_fps,
            },
            "side_depth_requested": {
                "width": args.side_depth_width,
                "height": args.side_depth_height,
                "fps": args.side_depth_fps,
            },
            "top_source": args.top_source,
            "top_camera_id": args.top_camera_id,
            "top_realsense_serial": args.top_realsense_serial or None,
            "top_color_requested": {
                "width": args.top_width,
                "height": args.top_height,
                "fps": args.top_fps,
            },
            "top_depth_requested": {
                "width": args.top_depth_width,
                "height": args.top_depth_height,
                "fps": args.top_depth_fps,
            },
            "record_depth": bool(args.record_depth),
            "top_duplicated_from_side": args.top_source == "duplicate_side",
            "opencv_latest_frame": bool(args.opencv_latest_frame),
        },
        "pose_semantics": "Elite TCP 6D pose [xyz_mm, rpy_rad]",
        "piper_semantics": "0=hold, 1=feed, -1=retract only if real retract is enabled",
        "operator_mode": "manual keyboard labels",
        "recording_start": {
            "start_on_first_action": bool(args.start_recording_on_first_action),
            "valid_action_keys": ["0", "h", "1", "f", "g", "v", "r", "b", "n", "l", "j"],
            "quit_keys_start_recording": False,
        },
        "collection_mode": args.collection_mode,
        "alignment_capture_mode": bool(args.alignment_capture_mode),
        "dataset_intent": "sim_real_alignment_calibration" if args.alignment_capture_mode else "real_shadow_pilot",
        "piper_label_source": args.piper_label_source,
        "piper_control_source": resolve_control_source(args, "piper"),
        "elite_control_source": resolve_control_source(args, "elite"),
        "feeder_device": {
            "enabled": bool(args.enable_feeder_device_control),
            "transport": "udp_json" if args.enable_feeder_device_control else None,
            "udp_transport": args.feeder_udp_transport if args.enable_feeder_device_control else None,
            "host": args.feeder_host if args.enable_feeder_device_control else None,
            "port": int(args.feeder_port) if args.enable_feeder_device_control else None,
            "step_mm_nominal": None,
            "step_mm_range": None,
            "step_calibration_status": (
                "not_applicable_event_semantics"
                if args.enable_feeder_device_control
                else None
            ),
            "timeout_s": float(args.feeder_timeout_s) if args.enable_feeder_device_control else None,
            "wait_response": bool(args.feeder_wait_response) if args.enable_feeder_device_control else None,
            "command_interval_s": float(args.feeder_command_interval_s) if args.enable_feeder_device_control else None,
            "forward_only": bool(args.feeder_forward_only) if args.enable_feeder_device_control else None,
        },
        "piper_auto_feed_period_s": args.piper_auto_feed_period,
        "piper_auto_feed_count": args.piper_auto_feed_count,
        "piper_keyboard_burst_default": args.piper_keyboard_burst,
        "piper_keyboard_burst_keys": {
            "f": 1,
            "1": 1,
            "g": 2,
            "v": 3,
        },
        "stop_after_auto_feed_count": bool(args.stop_after_auto_feed_count),
        "piper_feed_on_elite_path_step": bool(args.piper_feed_on_elite_path_step),
        "elite_path": {
            "mode": args.elite_path_mode,
            "execute": bool(args.elite_path_execute),
            "separate_connection": bool(args.elite_path_separate_connection),
            "path_file": args.elite_path_file,
            "start_index": args.elite_path_start_index,
            "end_index": args.elite_path_end_index,
            "period_s": args.elite_path_period,
            "speed": args.elite_path_speed,
            "start_delay_s": args.elite_path_start_delay,
            "max_start_distance_mm": args.elite_path_max_start_distance_mm,
            "approach_start": bool(args.elite_path_approach_start),
            "approach_motion": "move_line",
            "approach_speed_mm_s": args.elite_path_approach_speed,
            "approach_speed_type": 0,
            "approach_timeout_s": args.elite_path_approach_timeout_s,
            "approach_tolerance_mm": args.elite_path_approach_tolerance_mm,
        },
        "warmup_frames": int(args.warmup_frames),
        "session_note": args.session_note,
        "robot_command_mode": robot_command_mode(args),
        "zero_orientation_delta": bool(args.zero_orientation_delta),
        "records": "records.jsonl",
        "known_missing_fields": [
            "estimated_tip_pos_3d",
            "estimated_tip_heading_3d",
            "estimated_wall_margin",
            "route_estimator_confidence",
        ],
    }


def build_record(
    *,
    args: argparse.Namespace,
    step: int,
    side_path: Path,
    top_path: Path,
    side_depth_path: Path | None,
    top_depth_path: Path | None,
    side_frame: TimedFrame,
    top_frame: TimedFrame,
    pose: list[float],
    pose_timestamp: float,
    pose_stale: bool,
    piper_step: int,
    piper_command: int,
    piper_burst_count: int,
    piper_executed_command: str,
    piper_action_timestamp: float,
    piper_async_state: dict[str, Any] | None,
    elite_path_state: dict[str, Any] | None,
    user_event: dict[str, Any] | None,
    piper_step_after_command: int,
    out_dir: Path,
) -> dict[str, Any]:
    image_timestamp = min(side_frame.timestamp, top_frame.timestamp)
    elite_state = elite_path_state or {}
    piper_state = piper_async_state or {}
    pose_age_ms = max(0.0, (image_timestamp - pose_timestamp) * 1000.0)
    insertion_length_mm = piper_insertion_length_mm(args, piper_step)
    return {
        "timestamp": image_timestamp,
        "timestamps": {
            "side_image": side_frame.timestamp,
            "top_image": top_frame.timestamp,
            "side_depth": side_frame.depth_timestamp,
            "top_depth": top_frame.depth_timestamp,
            "elite_pose": pose_timestamp,
            "piper_action": piper_action_timestamp,
        },
        "sync": {
            "side_top_time_delta_ms": abs(side_frame.timestamp - top_frame.timestamp) * 1000.0,
            "image_pose_time_delta_ms": abs(image_timestamp - pose_timestamp) * 1000.0,
            "image_piper_time_delta_ms": abs(image_timestamp - piper_action_timestamp) * 1000.0,
            "image_action_time_delta_ms": abs(image_timestamp - piper_action_timestamp) * 1000.0,
        },
        "task": args.task,
        "step": int(step),
        "side_image": relative(side_path, out_dir),
        "top_image": relative(top_path, out_dir),
        "side_depth": relative(side_depth_path, out_dir) if side_depth_path is not None else None,
        "top_depth": relative(top_depth_path, out_dir) if top_depth_path is not None else None,
        "state": {
            "elite_tcp_pose_6d": pose,
            "piper_step": float(piper_step),
            "piper_insertion_length": insertion_length_mm,
            "piper_insertion_length_unit": "mm" if insertion_length_mm is not None else None,
            "controller_state": {
                "piper_intent": action_label(piper_command),
                "piper_executed_command": piper_executed_command,
                "piper_motion_state": "idle" if piper_command == 0 else action_label(piper_command),
                "piper_step_count": int(piper_step),
                "piper_step_after_command": int(piper_step_after_command),
                "piper_busy": piper_state.get("piper_busy", False),
                "piper_async_status": piper_state.get("piper_async_status"),
                "piper_async_command": piper_state.get("piper_async_command"),
                "piper_async_burst_count": piper_state.get("piper_async_burst_count"),
                "piper_async_start_timestamp": piper_state.get("piper_async_start_timestamp"),
                "piper_async_done_timestamp": piper_state.get("piper_async_done_timestamp"),
                "piper_async_error": piper_state.get("piper_async_error"),
                "piper_control_source": resolve_control_source(args, "piper"),
                "piper_label_source": args.piper_label_source,
                "feeder_transport": piper_state.get("feeder_transport"),
                "feeder_udp_transport": piper_state.get("feeder_udp_transport"),
                "feeder_host": piper_state.get("feeder_host"),
                "feeder_port": piper_state.get("feeder_port"),
                "feeder_step_mm_nominal": piper_state.get("feeder_step_mm_nominal"),
                "feeder_step_mm_range": piper_state.get("feeder_step_mm_range"),
                "feeder_step_calibration_status": piper_state.get("feeder_step_calibration_status"),
                "feeder_wait_response": piper_state.get("feeder_wait_response"),
                "feeder_last_response": piper_state.get("feeder_last_response"),
                "elite_control_source": resolve_control_source(args, "elite"),
                "collection_mode": args.collection_mode,
                "elite_path_file": elite_state.get("elite_path_file"),
                "elite_path_mode": elite_state.get("elite_path_mode", args.elite_path_mode),
                "elite_path_connection_mode": elite_state.get("elite_path_connection_mode"),
                "elite_path_index": elite_state.get("elite_path_index"),
                "elite_path_status": elite_state.get("elite_path_status"),
                "elite_path_done": elite_state.get("elite_path_done"),
                "elite_path_error": elite_state.get("elite_path_error"),
                "elite_path_start_guard_pose_6d": elite_state.get("elite_path_start_guard_pose_6d"),
                "elite_path_start_guard_distance_mm": elite_state.get("elite_path_start_guard_distance_mm"),
                "elite_path_max_start_distance_mm": elite_state.get("elite_path_max_start_distance_mm"),
                "elite_path_start_approach_requested": elite_state.get("elite_path_start_approach_requested"),
                "elite_path_start_approach_executed": elite_state.get("elite_path_start_approach_executed"),
                "elite_path_start_approach_status": elite_state.get("elite_path_start_approach_status"),
                "elite_path_start_approach_motion": elite_state.get("elite_path_start_approach_motion"),
                "elite_path_start_approach_speed_mm_s": elite_state.get(
                    "elite_path_start_approach_speed_mm_s"
                ),
                "elite_path_start_approach_tolerance_mm": elite_state.get(
                    "elite_path_start_approach_tolerance_mm"
                ),
                "elite_path_start_approach_final_pose_6d": elite_state.get(
                    "elite_path_start_approach_final_pose_6d"
                ),
                "elite_path_start_approach_final_distance_mm": elite_state.get(
                    "elite_path_start_approach_final_distance_mm"
                ),
                "elite_requested_tcp_pose_6d": elite_state.get("elite_requested_tcp_pose_6d"),
                "elite_requested_joints": elite_state.get("elite_requested_joints"),
                "elite_path_command_timestamp": elite_state.get("elite_path_command_timestamp"),
                "elite_step_piper_command": elite_state.get("elite_step_piper_command"),
                "elite_step_piper_executed": elite_state.get("elite_step_piper_executed"),
                "alignment_capture_mode": args.alignment_capture_mode,
                "user_event": user_event,
                "elite_executed_tcp_pose_6d": pose,
                "elite_pose_stale": bool(pose_stale),
                "elite_pose_age_ms": pose_age_ms,
                "elite_target_limited": False,
            },
            "estimated_tip_pos_3d": None,
            "estimated_tip_heading_3d": None,
            "tip_estimator_visible": False,
            "tip_estimator_confidence": 0.0,
            "estimated_contact_flag": None,
            "estimated_image_distance_px": None,
            "contact_estimator_confidence": 0.0,
            "estimated_wall_margin": None,
            "estimated_wall_margin_fraction": None,
            "route_estimator_confidence": 0.0,
            "estimated_magnet_wall_pull": None,
            "estimated_wall_side_risk": 0.0,
            "estimated_wall_normal_3d": None,
            "estimated_route_tangent_3d": None,
        },
        "reference_action": {
            "piper_step_command": int(piper_command),
            "piper_command_label": action_label(piper_command),
            "piper_burst_count": int(piper_burst_count),
            "elite_tcp_delta_6d": [0.0] * 6,
            "elite_tcp_pose_6d": pose,
        },
        "input_quality": {
            "side_visible": True,
            "top_visible": args.top_source != "duplicate_side",
            "side_depth_available": side_depth_path is not None,
            "top_depth_available": top_depth_path is not None,
            "side_camera_serial": side_frame.camera_serial,
            "top_camera_serial": top_frame.camera_serial,
            "side_depth_scale_m_per_unit": side_frame.depth_scale,
            "top_depth_scale_m_per_unit": top_frame.depth_scale,
            "tip_visible": None,
            "side_tip_pixel": optional_float_list(side_frame.position_px),
            "top_tip_pixel": optional_float_list(top_frame.position_px),
            "notes": "top duplicated from side" if args.top_source == "duplicate_side" else "",
        },
    }


def finalize_record(record: dict[str, Any], next_pose: list[float], args: argparse.Namespace) -> dict[str, Any]:
    current_pose = record["state"]["elite_tcp_pose_6d"]
    delta = pose_delta(current_pose, next_pose, zero_orientation_delta=bool(args.zero_orientation_delta))
    record["reference_action"]["elite_tcp_delta_6d"] = delta
    record["reference_action"]["elite_tcp_pose_6d"] = target_pose_from_delta(current_pose, delta)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a real-system pilot dataset for sim-to-real shadow mode. "
            "This script follows the old data_collect_mode1/2 structure but writes records.jsonl "
            "compatible with tools/real_shadow_policy_adapter.py."
        )
    )
    parser.add_argument("--out", required=True, help="Output directory, e.g. real_pilot_20260704_left_001")
    parser.add_argument("--task", choices=["left", "right"], required=True)
    parser.add_argument("--side-source", choices=["realsense", "opencv", "mock"], default="realsense")
    parser.add_argument("--side-camera-id", type=int, default=1)
    parser.add_argument("--side-realsense-serial", default="")
    parser.add_argument("--side-width", type=int, default=1920)
    parser.add_argument("--side-height", type=int, default=1080)
    parser.add_argument("--side-fps", type=int, default=15)
    parser.add_argument("--side-depth-width", type=int, default=1280)
    parser.add_argument("--side-depth-height", type=int, default=720)
    parser.add_argument("--side-depth-fps", type=int, default=15)
    parser.add_argument("--top-source", choices=["realsense", "opencv", "duplicate_side", "mock"], default="opencv")
    parser.add_argument("--top-camera-id", type=int, default=0)
    parser.add_argument("--top-realsense-serial", default="")
    parser.add_argument("--top-width", type=int, default=1920)
    parser.add_argument("--top-height", type=int, default=1080)
    parser.add_argument("--top-fps", type=int, default=15)
    parser.add_argument("--top-depth-width", type=int, default=1280)
    parser.add_argument("--top-depth-height", type=int, default=720)
    parser.add_argument("--top-depth-fps", type=int, default=15)
    parser.add_argument(
        "--record-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save aligned RealSense depth as uint16 PNG and expose depth paths in records.jsonl.",
    )
    parser.add_argument("--pose-source", choices=["elite", "mock"], default="elite")
    parser.add_argument("--elite-ip", default="192.168.137.200")
    parser.add_argument("--sample-period", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/esc")
    parser.add_argument("--initial-piper-step", type=int, default=0)
    parser.add_argument(
        "--collection-mode",
        choices=["mode1_auto_piper_manual_elite", "mode2_auto_elite_manual_piper", "fully_manual", "shadow_static", "custom"],
        default="custom",
        help="Human-readable collection intent written to manifest and records.",
    )
    parser.add_argument("--piper-control-source", choices=CONTROL_SOURCES, default="auto")
    parser.add_argument("--elite-control-source", choices=CONTROL_SOURCES, default="auto")
    parser.add_argument("--piper-label-source", choices=["keyboard", "auto_periodic"], default="keyboard")
    parser.add_argument(
        "--piper-auto-feed-period",
        type=float,
        default=0.8,
        help="Seconds between feed events when --piper-label-source auto_periodic.",
    )
    parser.add_argument(
        "--piper-auto-feed-count",
        type=int,
        default=0,
        help="Maximum auto feed events; 0 means no explicit limit.",
    )
    parser.add_argument(
        "--stop-after-auto-feed-count",
        action="store_true",
        help="Stop collection after --piper-auto-feed-count feed events when using auto_periodic labels.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Read and discard this many camera frames before recording, useful for camera auto-exposure warmup.",
    )
    parser.add_argument(
        "--opencv-latest-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For OpenCV cameras, keep a background reader and always consume the newest frame.",
    )
    parser.add_argument("--session-note", default="", help="Free-text note written into manifest.json.")
    parser.add_argument("--latched-intent", action="store_true", help="Keep last keyboard intent until changed; default is one-shot feed/retract.")
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--start-recording-on-first-action",
        action="store_true",
        help="Show live preview after setup, but write the first sample only when an action key is pressed.",
    )
    parser.add_argument("--enable-piper-control", action="store_true", help="Actually call PiperRobot step_forward/step_backward.")
    parser.add_argument(
        "--enable-feeder-device-control",
        action="store_true",
        help="Actually command the replacement UDP guidewire feeder device.",
    )
    parser.add_argument("--feeder-host", default="192.168.5.22", help="UDP feeder host.")
    parser.add_argument("--feeder-port", type=int, default=8888, help="UDP feeder port.")
    parser.add_argument(
        "--feeder-step-mm",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility argument. Fixed millimeters per feed are not valid for the curved vessel "
            "setup; providing this option is rejected."
        ),
    )
    parser.add_argument("--feeder-timeout-s", type=float, default=0.2, help="UDP feeder receive timeout.")
    parser.add_argument(
        "--feeder-udp-transport",
        choices=["python_socket", "bash_dev_udp"],
        default="python_socket",
        help="UDP send implementation. Use bash_dev_udp only on Linux when the Python socket path is not accepted onsite.",
    )
    parser.add_argument("--feeder-wait-response", action="store_true", help="Wait for a UDP response after each feeder packet.")
    parser.add_argument(
        "--feeder-command-interval-s",
        type=float,
        default=0.0,
        help="Delay between packets inside one feeder burst.",
    )
    parser.add_argument(
        "--feeder-forward-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Block keyboard retract commands when the replacement feeder device is enabled.",
    )
    parser.add_argument(
        "--external-piper-control",
        action="store_true",
        help=(
            "Do not call Piper hardware, but treat keyboard feed/retract labels as "
            "externally executed Piper commands for piper_step accounting."
        ),
    )
    parser.add_argument("--piper-pause-time", type=float, default=0.0)
    parser.add_argument(
        "--alignment-capture-mode",
        action="store_true",
        help="Mark the run as sim-real alignment/calibration data rather than expert demonstration data.",
    )
    parser.add_argument(
        "--piper-keyboard-burst",
        type=int,
        default=1,
        help="Default number of Piper feed/retract primitives for keyboard feed/retract keys unless a burst key overrides it.",
    )
    parser.add_argument(
        "--piper-feed-on-elite-path-step",
        action="store_true",
        help="When Elite path playback sends one path point, submit one asynchronous Piper feed command.",
    )
    parser.add_argument("--zero-orientation-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elite-path-mode", choices=["none", "keyboard", "auto"], default="none")
    parser.add_argument("--elite-path-file", default="", help="Elite TCP path file. Lines may be xyz_mm or xyz_mm+rpy_rad.")
    parser.add_argument("--elite-path-execute", action="store_true", help="Actually send Elite move_joint commands.")
    parser.add_argument(
        "--elite-path-separate-connection",
        action="store_true",
        help=(
            "Use a separate Elite SDK connection for path move_joint commands so the pose reader "
            "can keep polling current_pose without waiting for path commands."
        ),
    )
    parser.add_argument("--elite-path-start-index", type=int, default=0)
    parser.add_argument("--elite-path-end-index", type=int, default=-1, help="-1 means last point.")
    parser.add_argument("--elite-path-period", type=float, default=1.0)
    parser.add_argument("--elite-path-start-delay", type=float, default=1.0)
    parser.add_argument("--elite-path-speed", type=float, default=20.0)
    parser.add_argument("--elite-path-fixed-rpy", nargs=3, type=float, default=DEFAULT_ELITE_FIXED_RPY)
    parser.add_argument("--elite-path-max-step-mm", type=float, default=80.0)
    parser.add_argument(
        "--elite-path-max-start-distance-mm",
        type=float,
        default=30.0,
        help="Reject real Elite path execution when current XYZ is farther than this from the first selected point.",
    )
    parser.add_argument(
        "--elite-path-approach-start",
        action="store_true",
        help="Before recording, move Elite from its current pose to the first selected path point.",
    )
    parser.add_argument(
        "--elite-path-approach-speed",
        type=float,
        default=50.0,
        help="Elite move_line linear speed in mm/s for the unrecorded start approach (1-3000).",
    )
    parser.add_argument("--elite-path-approach-timeout-s", type=float, default=30.0)
    parser.add_argument("--elite-path-approach-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--elite-path-allow-large-jump", action="store_true")
    args = parser.parse_args()
    control_modes = [
        bool(args.enable_piper_control),
        bool(args.external_piper_control),
        bool(args.enable_feeder_device_control),
    ]
    if sum(control_modes) > 1:
        raise ValueError(
            "Use only one of --enable-piper-control, --external-piper-control, "
            "or --enable-feeder-device-control."
        )
    if args.feeder_step_mm is not None:
        raise ValueError(
            "--feeder-step-mm is disabled: feed is an event, and physical progress must remain observable/image-derived."
        )
    if args.feeder_port <= 0:
        raise ValueError("--feeder-port must be positive.")
    camera_dimensions = {
        "side_width": args.side_width,
        "side_height": args.side_height,
        "side_fps": args.side_fps,
        "side_depth_width": args.side_depth_width,
        "side_depth_height": args.side_depth_height,
        "side_depth_fps": args.side_depth_fps,
        "top_width": args.top_width,
        "top_height": args.top_height,
        "top_fps": args.top_fps,
        "top_depth_width": args.top_depth_width,
        "top_depth_height": args.top_depth_height,
        "top_depth_fps": args.top_depth_fps,
    }
    invalid_camera_dimensions = {key: value for key, value in camera_dimensions.items() if int(value) <= 0}
    if invalid_camera_dimensions:
        raise ValueError(f"Camera dimensions/FPS must be positive: {invalid_camera_dimensions}")
    if (
        args.side_source == "realsense"
        and args.top_source == "realsense"
        and args.side_realsense_serial
        and args.side_realsense_serial == args.top_realsense_serial
    ):
        raise ValueError("Side and top RealSense serials must be different.")
    if args.elite_path_max_start_distance_mm <= 0:
        raise ValueError("--elite-path-max-start-distance-mm must be positive.")
    if not 1 <= args.elite_path_approach_speed <= 3000:
        raise ValueError("--elite-path-approach-speed must be within [1, 3000] mm/s.")
    if args.elite_path_approach_timeout_s <= 0:
        raise ValueError("--elite-path-approach-timeout-s must be positive.")
    if args.elite_path_approach_tolerance_mm <= 0:
        raise ValueError("--elite-path-approach-tolerance-mm must be positive.")
    if args.feeder_udp_transport == "bash_dev_udp" and args.feeder_wait_response:
        raise ValueError("--feeder-wait-response is not supported with --feeder-udp-transport bash_dev_udp.")
    if args.elite_path_mode != "none" and not args.elite_path_file:
        raise ValueError("--elite-path-file is required when --elite-path-mode is not none.")
    if args.elite_path_execute and args.elite_path_mode == "none":
        raise ValueError("--elite-path-execute requires --elite-path-mode keyboard or auto.")
    if args.elite_path_approach_start and not args.elite_path_execute:
        raise ValueError("--elite-path-approach-start requires --elite-path-execute.")
    if args.start_recording_on_first_action:
        if not args.preview:
            raise ValueError("--start-recording-on-first-action requires --preview.")
        if args.piper_label_source != "keyboard":
            raise ValueError("--start-recording-on-first-action requires --piper-label-source keyboard.")
        if args.elite_path_mode == "auto":
            raise ValueError("--start-recording-on-first-action is incompatible with --elite-path-mode auto.")
    if args.piper_feed_on_elite_path_step:
        if not (args.enable_piper_control or args.enable_feeder_device_control):
            raise ValueError("--piper-feed-on-elite-path-step requires --enable-piper-control or --enable-feeder-device-control.")
        if args.elite_path_mode == "none":
            raise ValueError("--piper-feed-on-elite-path-step requires --elite-path-mode keyboard or auto.")

    out_dir = Path(args.out)
    frames_side = out_dir / "frames" / "side"
    frames_top = out_dir / "frames" / "top"
    frames_side_depth = out_dir / "frames" / "depth_side"
    frames_top_depth = out_dir / "frames" / "depth_top"
    logs_dir = out_dir / "logs"
    for path in (frames_side, frames_top, frames_side_depth, frames_top_depth, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(create_manifest(args, out_dir), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    records_path = out_dir / "records.jsonl"
    if records_path.exists():
        records_path.unlink()

    side_camera = build_camera(
        args.side_source,
        args.side_camera_id,
        "side",
        latest_frame=bool(args.opencv_latest_frame),
        width=int(args.side_width),
        height=int(args.side_height),
        fps=int(args.side_fps),
        realsense_serial=args.side_realsense_serial,
        depth_width=int(args.side_depth_width),
        depth_height=int(args.side_depth_height),
        depth_fps=int(args.side_depth_fps),
    )
    top_camera = build_camera(
        args.top_source,
        args.top_camera_id,
        "top",
        latest_frame=bool(args.opencv_latest_frame),
        width=int(args.top_width),
        height=int(args.top_height),
        fps=int(args.top_fps),
        realsense_serial=args.top_realsense_serial,
        depth_width=int(args.top_depth_width),
        depth_height=int(args.top_depth_height),
        depth_fps=int(args.top_depth_fps),
    )
    manifest = create_manifest(args, out_dir)
    manifest["camera_setup"]["side_runtime"] = camera_metadata(side_camera)
    manifest["camera_setup"]["top_runtime"] = camera_metadata(top_camera)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pose_reader = build_pose_reader(args)

    previous_side_warmup_timestamp: float | None = None
    previous_top_warmup_timestamp: float | None = None
    for warmup_idx in range(max(0, int(args.warmup_frames))):
        side_frame = read_newer_frame(side_camera, previous_side_warmup_timestamp)
        previous_side_warmup_timestamp = side_frame.timestamp
        if isinstance(top_camera, DuplicateCamera):
            top_camera.set_source(side_frame)
            top_frame = top_camera.read()
        else:
            top_frame = read_newer_frame(top_camera, previous_top_warmup_timestamp)
        previous_top_warmup_timestamp = top_frame.timestamp
        if (warmup_idx + 1) % 10 == 0 or warmup_idx + 1 == int(args.warmup_frames):
            print(f"warmup frame {warmup_idx + 1}/{int(args.warmup_frames)}")

    piper = None
    piper_executor: Any | None = None
    if args.enable_piper_control:
        from utils.robot.utils_piper import PiperRobot

        piper = PiperRobot()
        piper_executor = PiperAsyncExecutor(piper, args)
    elif args.enable_feeder_device_control:
        from hardware.feeder_device.udp_controller import UdpFeederConfig, UdpFeederDevice

        feeder_config = UdpFeederConfig(
            host=args.feeder_host,
            port=int(args.feeder_port),
            timeout_s=float(args.feeder_timeout_s),
            transport=args.feeder_udp_transport,
        )
        feeder = UdpFeederDevice(feeder_config)
        feeder.connect()
        piper_executor = FeederDeviceAsyncExecutor(feeder, args)
    elite_command_ec, elite_command_lock, elite_connection_mode = build_elite_command_connection(args)
    elite_playback = (
        ElitePathPlayback(
            args,
            pose_reader,
            piper_executor,
            command_ec=elite_command_ec,
            command_lock=elite_command_lock,
            connection_mode=elite_connection_mode,
        )
        if args.elite_path_mode != "none"
        else None
    )

    piper_step = int(args.initial_piper_step)
    current_command = 0
    observed_piper_feed_count = 0
    observed_piper_retract_count = 0
    auto_feed_count = 0
    auto_next_feed_time = time.time() + float(args.piper_auto_feed_period)
    pending_record: dict[str, Any] | None = None
    step = 0
    stopped_by_user = False
    recording_wait_started_at = time.time()
    recording_started = not bool(args.start_recording_on_first_action)
    recording_start_timestamp = recording_wait_started_at if recording_started else None
    recording_start_key: str | None = None
    recording_start_wait_s = 0.0 if recording_started else None
    piper_command_counts: Counter[str] = Counter()
    piper_executed_counts: Counter[str] = Counter()

    piper_log_path = logs_dir / "piper_state.csv"
    pose_log_path = logs_dir / "elite_tcp_pose.csv"
    command_log_path = logs_dir / "controller_commands.csv"
    with (
        piper_log_path.open("w", newline="", encoding="utf-8") as piper_log,
        pose_log_path.open("w", newline="", encoding="utf-8") as pose_log,
        command_log_path.open("w", newline="", encoding="utf-8") as command_log,
    ):
        piper_writer = csv.writer(piper_log)
        pose_writer = csv.writer(pose_log)
        command_writer = csv.writer(command_log)
        piper_writer.writerow(["step", "timestamp", "piper_step", "piper_command"])
        pose_writer.writerow(["step", "timestamp", "x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad"])
        command_writer.writerow(
            [
                "step",
                "timestamp",
                "user_key",
                "user_event_type",
                "piper_requested",
                "piper_burst_count",
                "piper_executed",
                "elite_path_index",
                "elite_path_status",
                "elite_submit_status",
                "elite_step_piper_executed",
                "elite_requested_tcp_pose_6d",
                "elite_requested_joints",
            ]
        )

        try:
            if args.enable_feeder_device_control and args.feeder_forward_only:
                print("Controls: 0/h hold, 1/f feed, n/l Elite next, j Elite previous, q/esc stop; feeder retract blocked")
            else:
                print("Controls: 0/h hold, 1/f feed, r/b retract, n/l Elite next, j Elite previous, q/esc stop")
            if elite_playback is not None:
                print(
                    f"Elite path mode={args.elite_path_mode}, execute={args.elite_path_execute}, "
                    f"file={args.elite_path_file}"
                )
                elite_playback.start_auto()
            while True:
                loop_start = time.time()
                if args.max_frames > 0 and step >= args.max_frames:
                    break
                if (
                    args.stop_after_auto_feed_count
                    and args.piper_label_source == "auto_periodic"
                    and args.piper_auto_feed_count > 0
                    and auto_feed_count >= int(args.piper_auto_feed_count)
                ):
                    break

                side_frame = side_camera.read()
                if isinstance(top_camera, DuplicateCamera):
                    top_camera.set_source(side_frame)
                top_frame = top_camera.read()
                if hasattr(pose_reader, "read_nonblocking"):
                    pose, pose_timestamp, pose_stale = pose_reader.read_nonblocking()
                else:
                    pose, pose_timestamp = pose_reader.read()
                    pose_stale = False

                key = -1
                elite_snapshot = elite_playback.snapshot() if elite_playback is not None else {}
                elite_status = ""
                if elite_snapshot:
                    elite_status = (
                        f"elite_idx={elite_snapshot.get('elite_path_index')} "
                        f"elite={elite_snapshot.get('elite_path_status')} "
                        f"busy={elite_snapshot.get('elite_path_command_busy')}"
                    )
                provisional_record = {
                    "step": step,
                    "task": args.task,
                    "reference_action": {"piper_command_label": action_label(current_command)},
                    "elite_status": elite_status,
                    "recording_waiting_for_action": not recording_started,
                }
                if args.preview:
                    key = preview(side_frame.image, top_frame.image, provisional_record)
                if key in (27, ord("q"), ord("Q")):
                    stopped_by_user = True
                    break
                starts_recording_this_loop = False
                if not recording_started:
                    if not is_recording_start_action(key):
                        elapsed = time.time() - loop_start
                        if elapsed < float(args.sample_period):
                            time.sleep(float(args.sample_period) - elapsed)
                        continue
                    recording_started = True
                    starts_recording_this_loop = True
                    recording_start_timestamp = time.time()
                    recording_start_key = key_name(key)
                    recording_start_wait_s = recording_start_timestamp - recording_wait_started_at
                    print(
                        "Recording started by first action: "
                        f"key={recording_start_key}, wait_s={recording_start_wait_s:.3f}"
                    )
                user_event: dict[str, Any] = {
                    "key": key_name(key),
                    "type": None,
                    "recording_started": starts_recording_this_loop,
                    "piper_command": 0,
                    "piper_burst_count": 1,
                    "elite_command": None,
                    "elite_submit_status": None,
                }
                elite_key = elite_path_key(key)
                if elite_playback is not None and args.elite_path_mode == "keyboard" and elite_key is not None:
                    submit_status = elite_playback.submit_keyboard_command(elite_key)
                    print(f"Elite keyboard command {submit_status}")
                    user_event["type"] = "elite_path_keyboard"
                    user_event["elite_command"] = elite_key
                    user_event["elite_submit_status"] = submit_status
                    elite_snapshot = elite_playback.snapshot()

                piper_burst_count = 1
                if args.piper_label_source == "auto_periodic":
                    now = time.time()
                    count_allows_feed = args.piper_auto_feed_count <= 0 or auto_feed_count < int(args.piper_auto_feed_count)
                    if count_allows_feed and now >= auto_next_feed_time:
                        current_command = 1
                        piper_burst_count = 1
                        auto_feed_count += 1
                        auto_next_feed_time += float(args.piper_auto_feed_period)
                    else:
                        current_command = 0
                else:
                    keyed_command = piper_command_from_key(key)
                    if keyed_command is not None:
                        if args.enable_feeder_device_control and args.feeder_forward_only and keyed_command < 0:
                            print("Feeder retract key ignored: --feeder-forward-only is enabled")
                            keyed_command = 0
                            user_event["type"] = "feeder_retract_blocked_forward_only"
                        current_command = keyed_command
                        piper_burst_count = piper_burst_from_key(key)
                        if piper_burst_count == 1 and keyed_command != 0:
                            piper_burst_count = max(1, int(args.piper_keyboard_burst))
                        user_event["type"] = (
                            "piper_keyboard" if user_event["type"] is None else f"{user_event['type']}+piper_keyboard"
                        )
                        user_event["piper_command"] = int(current_command)
                        user_event["piper_burst_count"] = int(piper_burst_count)
                    elif not args.latched_intent:
                        current_command = 0

                piper_step_before = int(piper_step)
                if args.piper_feed_on_elite_path_step and piper_executor is not None:
                    piper_async_state = piper_executor.snapshot()
                    feed_count = int(piper_async_state.get("piper_feed_started_count") or 0)
                    retract_count = int(piper_async_state.get("piper_retract_started_count") or 0)
                    feed_delta = max(0, feed_count - observed_piper_feed_count)
                    retract_delta = max(0, retract_count - observed_piper_retract_count)
                    observed_piper_feed_count = feed_count
                    observed_piper_retract_count = retract_count
                    if feed_delta > 0:
                        current_command = 1
                        executed = "feed_async_started_by_elite_path"
                        piper_step += feed_delta
                    elif retract_delta > 0:
                        current_command = -1
                        executed = "retract_async_started_by_elite_path"
                        piper_step = max(0, piper_step - retract_delta)
                    else:
                        current_command = 0
                        executed = "hold"
                elif piper_executor is not None:
                    executed = piper_executor.submit(current_command, count=piper_burst_count)
                    piper_async_state = piper_executor.snapshot()
                else:
                    executed = execute_piper_command(current_command, piper, args)
                    piper_async_state = {}
                piper_action_timestamp = time.time()
                command_affects_real_step = real_piper_command_affects_step(args)
                if (
                    not args.piper_feed_on_elite_path_step
                    and command_affects_real_step
                    and current_command > 0
                    and command_started_real_piper_step(current_command, executed)
                ):
                    piper_step += max(1, int(piper_burst_count))
                elif (
                    not args.piper_feed_on_elite_path_step
                    and command_affects_real_step
                    and current_command < 0
                    and command_started_real_piper_step(current_command, executed)
                ):
                    piper_step = max(0, piper_step - max(1, int(piper_burst_count)))
                piper_step_for_record = int(piper_step) if args.piper_feed_on_elite_path_step else piper_step_before
                piper_command_counts[action_label(current_command)] += max(1, int(piper_burst_count)) if current_command != 0 else 1
                piper_executed_counts[str(executed)] += 1

                side_path = frames_side / f"{step:06d}.png"
                top_path = frames_top / f"{step:06d}.png"
                record_image(side_path, side_frame.image)
                record_image(top_path, top_frame.image)
                side_depth_path = None
                top_depth_path = None
                if args.record_depth and side_frame.depth_image is not None:
                    side_depth_path = frames_side_depth / f"{step:06d}.png"
                    record_depth_image(side_depth_path, side_frame.depth_image)
                if args.record_depth and top_frame.depth_image is not None:
                    top_depth_path = frames_top_depth / f"{step:06d}.png"
                    record_depth_image(top_depth_path, top_frame.depth_image)

                record = build_record(
                    args=args,
                    step=step,
                    side_path=side_path,
                    top_path=top_path,
                    side_depth_path=side_depth_path,
                    top_depth_path=top_depth_path,
                    side_frame=side_frame,
                    top_frame=top_frame,
                    pose=pose,
                    pose_timestamp=pose_timestamp,
                    pose_stale=pose_stale,
                    piper_step=piper_step_for_record,
                    piper_command=current_command,
                    piper_burst_count=piper_burst_count,
                    piper_executed_command=executed,
                    piper_action_timestamp=piper_action_timestamp,
                    piper_async_state=piper_async_state,
                    elite_path_state=elite_snapshot,
                    user_event=user_event if user_event["type"] is not None else None,
                    piper_step_after_command=int(piper_step),
                    out_dir=out_dir,
                )
                if pending_record is not None:
                    write_jsonl_row(records_path, finalize_record(pending_record, pose, args))
                pending_record = record

                piper_writer.writerow([step, time.time(), piper_step_for_record, current_command])
                pose_writer.writerow([step, pose_timestamp, *pose])
                command_writer.writerow(
                    [
                        step,
                        time.time(),
                        key_name(key),
                        user_event.get("type"),
                        action_label(current_command),
                        int(piper_burst_count),
                        executed,
                        elite_snapshot.get("elite_path_index"),
                        elite_snapshot.get("elite_path_status"),
                        user_event.get("elite_submit_status"),
                        elite_snapshot.get("elite_step_piper_executed"),
                        json.dumps(elite_snapshot.get("elite_requested_tcp_pose_6d"), ensure_ascii=False),
                        json.dumps(elite_snapshot.get("elite_requested_joints"), ensure_ascii=False),
                    ]
                )

                print(
                    f"step={step} task={args.task} piper={action_label(current_command)} "
                    f"piper_step={piper_step_for_record} pose_xyz={pose[:3]} "
                    f"elite_idx={elite_snapshot.get('elite_path_index') if elite_snapshot else None}"
                )
                step += 1
                elapsed = time.time() - loop_start
                if elapsed < float(args.sample_period):
                    time.sleep(float(args.sample_period) - elapsed)
        finally:
            if elite_playback is not None:
                elite_playback.stop()
            if pending_record is not None:
                write_jsonl_row(records_path, finalize_record(pending_record, pending_record["state"]["elite_tcp_pose_6d"], args))
            side_camera.close()
            top_camera.close()
            if args.preview:
                cv2.destroyAllWindows()
            if piper_executor is not None:
                piper_executor.close()
            if piper is not None:
                try:
                    piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
                except Exception as exc:  # pragma: no cover - hardware cleanup best effort
                    print(f"warning: failed to reset Piper gripper: {exc}")

    elite_final_snapshot = elite_playback.snapshot() if elite_playback is not None else {}
    summary = {
        "out": str(out_dir),
        "records": sum(1 for _ in records_path.open("r", encoding="utf-8")) if records_path.exists() else 0,
        "stopped_by_user": stopped_by_user,
        "records_jsonl": str(records_path),
        "final_piper_step": int(piper_step),
        "estimated_feeder_insertion_mm": piper_insertion_length_mm(args, int(piper_step)),
        "auto_feed_count": int(auto_feed_count),
        "piper_command_counts": dict(piper_command_counts),
        "piper_executed_counts": dict(piper_executed_counts),
        "robot_command_mode": create_manifest(args, out_dir)["robot_command_mode"],
        "piper_label_source": args.piper_label_source,
        "collection_mode": args.collection_mode,
        "recording_started": recording_started,
        "recording_start_timestamp": recording_start_timestamp,
        "recording_start_key": recording_start_key,
        "recording_start_wait_s": recording_start_wait_s,
        "elite_path_start_guard_pose_6d": elite_final_snapshot.get("elite_path_start_guard_pose_6d"),
        "elite_path_start_guard_distance_mm": elite_final_snapshot.get("elite_path_start_guard_distance_mm"),
        "elite_path_start_approach_requested": elite_final_snapshot.get("elite_path_start_approach_requested"),
        "elite_path_start_approach_executed": elite_final_snapshot.get("elite_path_start_approach_executed"),
        "elite_path_start_approach_status": elite_final_snapshot.get("elite_path_start_approach_status"),
        "elite_path_start_approach_motion": elite_final_snapshot.get("elite_path_start_approach_motion"),
        "elite_path_start_approach_speed_mm_s": elite_final_snapshot.get(
            "elite_path_start_approach_speed_mm_s"
        ),
        "elite_path_start_approach_tolerance_mm": elite_final_snapshot.get(
            "elite_path_start_approach_tolerance_mm"
        ),
        "elite_path_start_approach_final_pose_6d": elite_final_snapshot.get(
            "elite_path_start_approach_final_pose_6d"
        ),
        "elite_path_start_approach_final_distance_mm": elite_final_snapshot.get(
            "elite_path_start_approach_final_distance_mm"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
