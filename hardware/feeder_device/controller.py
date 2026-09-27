"""Minimal serial adapter for the replacement guidewire feeder device.

This module intentionally keeps a small surface area. The raw UI/firmware files
live under ``reference_code/feeder_device_raw``; this adapter only wraps the
observed serial protocol so collection code can later depend on a stable API.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable


try:
    import serial
except ImportError:  # pragma: no cover - exercised only on machines without pyserial
    serial = None


STM_FRAME_TAIL = bytes([0x0D, 0x0A])

# Current firmware evidence in USER/main.c maps 0x00 to Guidewire Forward and
# 0x03 to Guidewire Backward. The raw Python/readme also mention 0xFF as a
# backward command, so the probe tool exposes that legacy byte separately.
CMD_GUIDEWIRE_FORWARD = 0x00
CMD_GUIDEWIRE_BACKWARD = 0x03
CMD_GUIDEWIRE_BACKWARD_LEGACY = 0xFF

# These are present in the raw Python UI/readme, but the inspected current
# firmware does not contain matching 0x0F/0xF0 branches. Treat them as unverified.
CMD_LOCK_RAW_UI = 0x0F
CMD_UNLOCK_RAW_UI = 0xF0

ROTATE_CW_HEADER = [0x7B, 0x01, 0x02, 0x01, 0x20]
ROTATE_CCW_HEADER = [0x7B, 0x01, 0x02, 0x00, 0x20]


@dataclass(frozen=True)
class SerialPortConfig:
    port: str
    baudrate: int
    timeout_s: float = 0.1


def require_pyserial() -> None:
    if serial is None:
        raise RuntimeError("pyserial is required for feeder hardware control: pip install pyserial")


def build_stm_command(command_byte: int) -> bytes:
    if not 0 <= int(command_byte) <= 0xFF:
        raise ValueError(f"command byte out of range: {command_byte!r}")
    return bytes([int(command_byte)]) + STM_FRAME_TAIL


def build_rotation_frame(header: Iterable[int], angle_degrees: int) -> bytes:
    """Build the 11-byte rotation-motor frame copied from the raw UI logic."""
    angle_times_10 = int(angle_degrees) * 10
    payload = list(header) + [
        (angle_times_10 >> 8) & 0xFF,
        angle_times_10 & 0xFF,
        0x00,
        0x64,
    ]
    bcc = 0
    for value in payload:
        bcc ^= value
    return bytes(payload + [bcc, 0x7D])


class FeederDevice:
    """Small blocking controller for manual/probe use.

    Parameters are explicit so Windows COM ports and Linux ``/dev/ttyUSB*`` or
    ``/dev/ttyACM*`` paths can be supplied by the operator at runtime.
    """

    def __init__(
        self,
        stm: SerialPortConfig,
        motor: SerialPortConfig | None = None,
    ) -> None:
        self.stm_config = stm
        self.motor_config = motor
        self.stm = None
        self.motor = None

    def connect(self) -> None:
        require_pyserial()
        self.stm = serial.Serial(
            port=self.stm_config.port,
            baudrate=self.stm_config.baudrate,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=self.stm_config.timeout_s,
        )
        if self.motor_config is not None:
            self.motor = serial.Serial(
                port=self.motor_config.port,
                baudrate=self.motor_config.baudrate,
                bytesize=serial.EIGHTBITS,
                stopbits=serial.STOPBITS_ONE,
                parity=serial.PARITY_NONE,
                timeout=self.motor_config.timeout_s,
            )

    def close(self) -> None:
        for port in (self.motor, self.stm):
            if port is not None and port.is_open:
                port.close()

    def __enter__(self) -> "FeederDevice":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def send_stm_byte(self, command_byte: int, read_window_s: float = 0.2) -> bytes:
        if self.stm is None or not self.stm.is_open:
            raise RuntimeError("STM serial port is not open")
        frame = build_stm_command(command_byte)
        self.stm.write(frame)
        self.stm.flush()
        return self.read_stm_available(read_window_s)

    def read_stm_available(self, read_window_s: float = 0.2) -> bytes:
        if self.stm is None or not self.stm.is_open:
            raise RuntimeError("STM serial port is not open")
        deadline = time.time() + max(0.0, read_window_s)
        chunks: list[bytes] = []
        while time.time() < deadline:
            waiting = self.stm.in_waiting
            if waiting:
                chunks.append(self.stm.read(waiting))
            time.sleep(0.01)
        return b"".join(chunks)

    def feed_once(self, read_window_s: float = 0.2) -> bytes:
        return self.send_stm_byte(CMD_GUIDEWIRE_FORWARD, read_window_s=read_window_s)

    def retract_once(self, read_window_s: float = 0.2, legacy: bool = False) -> bytes:
        command = CMD_GUIDEWIRE_BACKWARD_LEGACY if legacy else CMD_GUIDEWIRE_BACKWARD
        return self.send_stm_byte(command, read_window_s=read_window_s)

    def send_rotation(self, clockwise: bool, angle_degrees: int) -> None:
        if self.motor is None or not self.motor.is_open:
            raise RuntimeError("rotation motor serial port is not open")
        header = ROTATE_CW_HEADER if clockwise else ROTATE_CCW_HEADER
        self.motor.write(build_rotation_frame(header, angle_degrees))
        self.motor.flush()
