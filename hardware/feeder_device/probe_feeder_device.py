"""Manual probe CLI for the replacement guidewire feeder serial protocol.

The script requires ``--execute`` before it writes to hardware. Without it, the
command prints the bytes that would be sent.
"""

from __future__ import annotations

import argparse
import sys
import time

from .controller import (
    CMD_GUIDEWIRE_BACKWARD,
    CMD_GUIDEWIRE_BACKWARD_LEGACY,
    CMD_GUIDEWIRE_FORWARD,
    CMD_LOCK_RAW_UI,
    CMD_UNLOCK_RAW_UI,
    FeederDevice,
    SerialPortConfig,
    build_rotation_frame,
    build_stm_command,
    ROTATE_CCW_HEADER,
    ROTATE_CW_HEADER,
)


COMMAND_BYTES = {
    "feed": CMD_GUIDEWIRE_FORWARD,
    "retract": CMD_GUIDEWIRE_BACKWARD,
    "retract-legacy": CMD_GUIDEWIRE_BACKWARD_LEGACY,
    "lock-raw-ui": CMD_LOCK_RAW_UI,
    "unlock-raw-ui": CMD_UNLOCK_RAW_UI,
}


def format_bytes(data: bytes) -> str:
    return " ".join(f"0x{value:02X}" for value in data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stm-port", required=True, help="STM32 serial port, e.g. COM7 or /dev/ttyUSB0")
    parser.add_argument("--stm-baud", type=int, default=9600)
    parser.add_argument("--motor-port", help="Optional rotation motor serial port, e.g. COM4")
    parser.add_argument("--motor-baud", type=int, default=115200)
    parser.add_argument("--timeout-s", type=float, default=0.1)
    parser.add_argument(
        "--command",
        required=True,
        choices=[*COMMAND_BYTES.keys(), "raw", "rotate-cw", "rotate-ccw"],
    )
    parser.add_argument("--raw-byte", type=lambda s: int(s, 0), help="Byte for --command raw, e.g. 0x00")
    parser.add_argument("--angle-degrees", type=int, default=1080)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--read-window-s", type=float, default=0.3)
    parser.add_argument("--execute", action="store_true", help="Actually send bytes to hardware")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    if args.command == "raw":
        if args.raw_byte is None:
            raise SystemExit("--raw-byte is required for --command raw")
        stm_frame = build_stm_command(args.raw_byte)
    elif args.command in COMMAND_BYTES:
        stm_frame = build_stm_command(COMMAND_BYTES[args.command])
    else:
        header = ROTATE_CW_HEADER if args.command == "rotate-cw" else ROTATE_CCW_HEADER
        motor_frame = build_rotation_frame(header, args.angle_degrees)
        print(f"motor frame: {format_bytes(motor_frame)}")
        if not args.execute:
            print("dry-run only; add --execute to write to hardware")
            return 0
        if not args.motor_port:
            raise SystemExit("--motor-port is required for rotation commands")
        with FeederDevice(
            stm=SerialPortConfig(args.stm_port, args.stm_baud, args.timeout_s),
            motor=SerialPortConfig(args.motor_port, args.motor_baud, args.timeout_s),
        ) as device:
            for idx in range(args.count):
                device.send_rotation(clockwise=args.command == "rotate-cw", angle_degrees=args.angle_degrees)
                print(f"sent {idx + 1}/{args.count}")
                time.sleep(args.interval_s)
        return 0

    print(f"stm frame: {format_bytes(stm_frame)}")
    if not args.execute:
        print("dry-run only; add --execute to write to hardware")
        return 0

    with FeederDevice(stm=SerialPortConfig(args.stm_port, args.stm_baud, args.timeout_s)) as device:
        command_byte = stm_frame[0]
        for idx in range(args.count):
            response = device.send_stm_byte(command_byte, read_window_s=args.read_window_s)
            print(f"sent {idx + 1}/{args.count}: {format_bytes(stm_frame)}")
            if response:
                print(f"rx: {response!r}")
            time.sleep(args.interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
