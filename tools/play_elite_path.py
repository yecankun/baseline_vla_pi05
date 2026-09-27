from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any


DEFAULT_FIXED_RPY = [3.0661665148309702, 0.03647610536354759, 0.06842115274422467]


def load_path(path_file: Path, fixed_rpy: list[float]) -> list[list[float]]:
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
        raise ValueError(f"No path points found in {path_file}")
    return poses


def position_distance_mm(a: list[float], b: list[float]) -> float:
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)) ** 0.5


def print_path_summary(poses: list[list[float]], *, start_index: int, end_index: int, max_step_mm: float) -> None:
    print(f"loaded path points: {len(poses)}")
    print(f"selected range: [{start_index}, {end_index}]")
    print(f"first selected pose: {poses[start_index]}")
    print(f"last selected pose:  {poses[end_index]}")
    if start_index < end_index:
        distances = [position_distance_mm(poses[i - 1], poses[i]) for i in range(start_index + 1, end_index + 1)]
        print(
            "consecutive xyz step mm: "
            f"min={min(distances):.3f}, median={sorted(distances)[len(distances)//2]:.3f}, max={max(distances):.3f}"
        )
        large = [(start_index + 1 + i, d) for i, d in enumerate(distances) if d > max_step_mm]
        if large:
            print(f"WARNING: {len(large)} path jumps exceed --max-step-mm={max_step_mm}")
            for index, distance in large[:10]:
                print(f"  jump to index {index}: {distance:.3f} mm")


class ElitePathPlayer:
    def __init__(self, ip: str, *, execute: bool, check_ik: bool, speed: float) -> None:
        self.execute = bool(execute)
        self.check_ik = bool(check_ik or execute)
        self.speed = float(speed)
        self.ec: Any | None = None
        if self.check_ik:
            from elite import EC

            self.ec = EC(ip=ip, auto_connect=True)

    def current_pose(self) -> list[float] | None:
        if self.ec is None:
            return None
        pose = list(self.ec.current_pose)
        return [float(v) for v in pose]

    def move_or_check(self, pose: list[float]) -> tuple[str, list[float] | None]:
        if self.ec is None:
            return "dry_run_no_ik", None
        target_joint = self.ec.get_inverse_kinematic(pose=pose)
        target_joint_list = [float(v) for v in target_joint]
        if not self.execute:
            return "ik_ok_dry_run", target_joint_list
        self.ec.move_joint(target_joint=target_joint, speed=self.speed)
        return "move_joint_sent", target_joint_list


def write_log_row(writer: csv.writer, index: int, pose: list[float], status: str, joints: list[float] | None) -> None:
    writer.writerow(
        [
            time.time(),
            index,
            status,
            *pose,
            *(joints if joints is not None else [None] * 6),
        ]
    )


def run_auto(
    player: ElitePathPlayer,
    poses: list[list[float]],
    *,
    start_index: int,
    end_index: int,
    pause: float,
    writer: csv.writer,
) -> None:
    for index in range(start_index, end_index + 1):
        pose = poses[index]
        print(f"[{index}/{end_index}] target pose: {pose}")
        status, joints = player.move_or_check(pose)
        print(f"  {status}; joints={joints}")
        write_log_row(writer, index, pose, status, joints)
        if pause > 0 and index < end_index:
            time.sleep(pause)


def run_step(
    player: ElitePathPlayer,
    poses: list[list[float]],
    *,
    start_index: int,
    end_index: int,
    pause: float,
    writer: csv.writer,
) -> None:
    index = start_index
    while start_index <= index <= end_index:
        pose = poses[index]
        current = player.current_pose()
        if current is not None:
            print(f"current pose: {current}")
            print(f"distance to target xyz: {position_distance_mm(current, pose):.3f} mm")
        print(f"[{index}/{end_index}] target pose: {pose}")
        try:
            command = input("Enter/n=send next, b=back, g <index>=jump, q=quit > ").strip()
        except EOFError:
            print("stdin closed; exiting step mode")
            break
        if command.lower() in ("q", "quit", "exit"):
            break
        if command.lower() in ("b", "back"):
            index = max(start_index, index - 1)
            continue
        if command.lower().startswith("g "):
            try:
                target_index = int(command.split(maxsplit=1)[1])
            except ValueError:
                print("invalid index")
                continue
            if not (start_index <= target_index <= end_index):
                print(f"index must be within [{start_index}, {end_index}]")
                continue
            index = target_index
            continue

        status, joints = player.move_or_check(pose)
        print(f"  {status}; joints={joints}")
        write_log_row(writer, index, pose, status, joints)
        if pause > 0:
            time.sleep(pause)
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Play or dry-run an Elite TCP path file. Path lines may be xyz_mm or xyz_mm+rpy_rad. "
            "The script does not move the real robot unless --execute is passed."
        )
    )
    parser.add_argument("--path-file", required=True, type=Path)
    parser.add_argument("--elite-ip", default="192.168.5.66")
    parser.add_argument("--mode", choices=["summary", "step", "auto"], default="summary")
    parser.add_argument("--execute", action="store_true", help="Actually send move_joint commands to Elite.")
    parser.add_argument("--check-ik", action="store_true", help="Connect to Elite and compute IK without moving.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="-1 means last point.")
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--fixed-rpy", nargs=3, type=float, default=DEFAULT_FIXED_RPY)
    parser.add_argument("--max-step-mm", type=float, default=80.0)
    parser.add_argument("--allow-large-jump", action="store_true")
    parser.add_argument("--log", type=Path, default=Path("elite_path_playback_log.csv"))
    args = parser.parse_args()

    poses = load_path(args.path_file, [float(v) for v in args.fixed_rpy])
    end_index = len(poses) - 1 if int(args.end_index) < 0 else int(args.end_index)
    start_index = int(args.start_index)
    if not (0 <= start_index < len(poses)):
        raise ValueError(f"--start-index must be within [0, {len(poses) - 1}]")
    if not (start_index <= end_index < len(poses)):
        raise ValueError(f"--end-index must be within [{start_index}, {len(poses) - 1}]")

    print_path_summary(poses, start_index=start_index, end_index=end_index, max_step_mm=float(args.max_step_mm))
    if start_index < end_index:
        distances = [position_distance_mm(poses[i - 1], poses[i]) for i in range(start_index + 1, end_index + 1)]
        if max(distances) > float(args.max_step_mm) and not args.allow_large_jump:
            raise SystemExit("Refusing to continue because the selected path has a large jump. Use --allow-large-jump to override.")

    if args.mode == "summary":
        if args.check_ik or args.execute:
            player = ElitePathPlayer(args.elite_ip, execute=False, check_ik=True, speed=float(args.speed))
            print(f"current Elite pose: {player.current_pose()}")
        print("summary complete; no motion was sent")
        return

    if not args.execute:
        print("DRY RUN: no robot motion will be sent. Add --execute only after visually confirming the path is safe.")
    else:
        print("EXECUTE ENABLED: this will command the real Elite robot.")
        try:
            confirmation = input("Type EXECUTE to continue > ").strip()
        except EOFError:
            confirmation = ""
        if confirmation != "EXECUTE":
            raise SystemExit("Execution cancelled.")

    player = ElitePathPlayer(args.elite_ip, execute=bool(args.execute), check_ik=True, speed=float(args.speed))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "path_index",
                "status",
                "x_mm",
                "y_mm",
                "z_mm",
                "rx_rad",
                "ry_rad",
                "rz_rad",
                "j1",
                "j2",
                "j3",
                "j4",
                "j5",
                "j6",
            ]
        )
        if args.mode == "auto":
            run_auto(
                player,
                poses,
                start_index=start_index,
                end_index=end_index,
                pause=float(args.pause),
                writer=writer,
            )
        else:
            run_step(
                player,
                poses,
                start_index=start_index,
                end_index=end_index,
                pause=float(args.pause),
                writer=writer,
            )
    print(f"wrote log: {args.log}")


if __name__ == "__main__":
    main()
