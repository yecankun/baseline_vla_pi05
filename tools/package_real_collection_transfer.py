from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REQUIRED_FILES = [
    "data/collect/collect_real_shadow_pilot.py",
    "tools/play_elite_path.py",
    "tools/validate_real_shadow_pilot.py",
    "docs/real-collection-lab-runbook.md",
    "docs/real-data-collection-checklist.md",
    "utils/camera/utils_camera.py",
    "utils/camera/utils_cap.py",
    "utils/camera/hsv_locate.py",
    "utils/camera/get_colors.py",
    "utils/robot/utils_piper.py",
    "utils/robot/utils_elirobot.py",
]

INIT_FILES = [
    "data/__init__.py",
    "data/collect/__init__.py",
    "tools/__init__.py",
    "utils/__init__.py",
    "utils/camera/__init__.py",
    "utils/robot/__init__.py",
]


README = """# Real Collection Transfer Package

Copy this package into the root of the project checkout on the robot-control
computer, preserving the directory structure.

Minimum Python packages expected on the robot-control computer:

```text
opencv-python
numpy
pyrealsense2
piper_sdk
elite
```

`elite` is imported as:

```python
from elite import EC
```

If `elite` is not installed as a package on that computer, copy/install the
same Elite SDK used by the senior code before running real collection.

Start with the static smoke in:

```text
docs/real-collection-lab-runbook.md
```

Do not run commands with `--enable-piper-control` until camera, Elite pose,
write path, and validation all pass.
"""


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable real collection transfer package.")
    parser.add_argument("--out", default="simulation_output/real_collection_transfer_pkg_20260705")
    parser.add_argument("--zip", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for rel in REQUIRED_FILES:
        src = root / rel
        if src.exists():
            copy_file(src, out_dir / rel)
            copied.append(rel)
        else:
            missing.append(rel)

    for rel in INIT_FILES:
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (root / rel).exists():
            copy_file(root / rel, dst)
        elif not dst.exists():
            dst.write_text("", encoding="utf-8")
        copied.append(rel)

    (out_dir / "README_REAL_COLLECTION_TRANSFER.md").write_text(README, encoding="utf-8")
    copied.append("README_REAL_COLLECTION_TRANSFER.md")

    if missing:
        (out_dir / "MISSING_FILES.txt").write_text("\n".join(missing) + "\n", encoding="utf-8")

    zip_path = None
    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(out_dir))

    print(f"wrote package: {out_dir}")
    if zip_path:
        print(f"wrote zip: {zip_path}")
    print("copied:")
    for rel in copied:
        print(f"  {rel}")
    if missing:
        print("missing:")
        for rel in missing:
            print(f"  {rel}")


if __name__ == "__main__":
    main()
