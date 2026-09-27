"""UDP adapter for the replacement guidewire feeder device."""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import subprocess
from typing import Any, Literal


FeederAction = Literal["forward", "backward", "turn_left", "turn_right"]


@dataclass(frozen=True)
class UdpFeederConfig:
    host: str = "192.168.5.22"
    port: int = 8888
    timeout_s: float = 0.2
    encoding: str = "utf-8"
    transport: Literal["python_socket", "bash_dev_udp"] = "python_socket"


def build_udp_move_payload(action: FeederAction, value: float = 1) -> bytes:
    if action not in {"forward", "backward", "turn_left", "turn_right"}:
        raise ValueError(f"unsupported feeder action: {action!r}")
    payload: dict[str, Any] = {
        "command": "move",
        "parameters": {
            "action": action,
            "value": value,
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


class UdpFeederDevice:
    """Small UDP controller for the networked guidewire feeder.

    The device-side protocol is currently:

    ``{"command":"move","parameters":{"action":"forward","value":1}}``

    sent over UDP to ``192.168.5.22:8888`` by default. Each ``forward`` or
    ``backward`` packet executes one feeder step. ``value`` is currently used
    only by ``turn_left`` and ``turn_right``.
    """

    def __init__(self, config: UdpFeederConfig | None = None) -> None:
        self.config = config or UdpFeederConfig()
        self.socket: socket.socket | None = None

    def connect(self) -> None:
        if self.config.transport == "bash_dev_udp":
            return
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.config.timeout_s)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def __enter__(self) -> "UdpFeederDevice":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def send_move(self, action: FeederAction, value: float = 1, wait_response: bool = False) -> bytes | None:
        payload = build_udp_move_payload(action, value=value)
        if self.config.transport == "bash_dev_udp":
            if wait_response:
                raise ValueError("bash_dev_udp transport does not support response reads")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'printf "%s" "$3" > "/dev/udp/$1/$2"',
                    "bash",
                    self.config.host,
                    str(self.config.port),
                    payload.decode(self.config.encoding),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.decode(self.config.encoding, errors="replace").strip()
                raise RuntimeError(f"bash /dev/udp send failed: {detail or result.returncode}")
            return None
        if self.socket is None:
            raise RuntimeError("UDP feeder socket is not open")
        self.socket.sendto(payload, (self.config.host, self.config.port))
        if not wait_response:
            return None
        try:
            response, _addr = self.socket.recvfrom(4096)
            return response
        except socket.timeout:
            return None

    def feed_once(self, wait_response: bool = False) -> bytes | None:
        return self.send_move("forward", value=1, wait_response=wait_response)

    def retract_once(self, wait_response: bool = False) -> bytes | None:
        return self.send_move("backward", value=1, wait_response=wait_response)

    def turn_left(self, degrees: float, wait_response: bool = False) -> bytes | None:
        return self.send_move("turn_left", value=degrees, wait_response=wait_response)

    def turn_right(self, degrees: float, wait_response: bool = False) -> bytes | None:
        return self.send_move("turn_right", value=degrees, wait_response=wait_response)
