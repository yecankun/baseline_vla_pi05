"""Manual probe CLI for the UDP guidewire feeder protocol.

The script requires ``--execute`` before it sends UDP packets. Without it, the
command prints the JSON payload that would be sent.
"""

from __future__ import annotations

import argparse
import time

from .udp_controller import UdpFeederConfig, UdpFeederDevice, build_udp_move_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.5.13")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        "--action",
        required=True,
        choices=["forward", "backward", "turn_left", "turn_right"],
    )
    parser.add_argument("--value", type=float, default=1)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=0.2)
    parser.add_argument("--wait-response", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send UDP packets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    payload = build_udp_move_payload(args.action, value=args.value)
    print(f"target: {args.host}:{args.port}/udp")
    print(f"payload: {payload.decode('utf-8')}")
    if not args.execute:
        print("dry-run only; add --execute to send UDP packets")
        return 0

    config = UdpFeederConfig(host=args.host, port=args.port, timeout_s=args.timeout_s)
    with UdpFeederDevice(config) as device:
        for idx in range(args.count):
            response = device.send_move(args.action, value=args.value, wait_response=args.wait_response)
            print(f"sent {idx + 1}/{args.count}")
            if response:
                print(f"rx: {response!r}")
            time.sleep(args.interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
