from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TEXT_SUFFIXES = (
    ".urdf",
    ".xacro",
    ".xml",
    ".launch",
    ".md",
    ".txt",
    ".csv",
    ".yaml",
    ".yml",
    ".rviz",
    ".cfg",
    ".json",
)


@dataclass(frozen=True)
class AssetPackage:
    name: str
    source: Path
    destination: Path
    aliases: tuple[str, ...] = ()
    default_variants: tuple[str, ...] = ()


PACKAGES = (
    AssetPackage(
        name="piper_description",
        source=Path("robot_assets/piper_ros/src/piper_description"),
        destination=Path("piper_description"),
        aliases=("gripper_description",),
        default_variants=(
            "urdf/piper_no_gripper_description.urdf",
            "urdf/piper_description.urdf",
            "urdf/piper_description_with_camera.urdf",
            "urdf/piper_description_v100.urdf",
        ),
    ),
    AssetPackage(
        name="elite_description",
        source=Path("robot_assets/elite_ros/src/elite_description"),
        destination=Path("elite_description"),
        aliases=(),
        default_variants=("urdf/ec66_description.urdf", "urdf/ec63_description.urdf", "urdf/ec612_description.urdf"),
    ),
)


def is_text_file(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in TEXT_SUFFIXES)


def rewrite_package_uris(text: str, package_names: list[str], file_path: Path, package_root: Path) -> str:
    relative_root = os.path.relpath(package_root, file_path.parent).replace("\\", "/")
    if relative_root == ".":
        relative_root = ""
    prefix = f"{relative_root}/" if relative_root else ""
    for package_name in package_names:
        text = text.replace(f"package://{package_name}/", prefix)
    return text


def copy_package(package: AssetPackage, root_out: Path) -> dict:
    src = package.source.resolve()
    dst = (root_out / package.destination).resolve()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    package_names = [package.name, *package.aliases]
    touched = 0
    for file_path in dst.rglob("*"):
        if file_path.is_file() and is_text_file(file_path):
            original = file_path.read_text(encoding="utf-8", errors="ignore")
            rewritten = rewrite_package_uris(original, package_names, file_path, dst)
            if rewritten != original:
                touched += 1
                file_path.write_text(rewritten, encoding="utf-8")

    rel_dst = dst.relative_to(root_out).as_posix()
    return {
        "name": package.name,
        "source": src.as_posix(),
        "destination": rel_dst,
        "aliases": list(package.aliases),
        "default_variants": list(package.default_variants),
        "rewritten_files": touched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize ROS robot assets into a self-contained folder layout.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", default="robot_assets/standardized")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": repo_root.as_posix(),
        "output_root": out_root.as_posix(),
        "packages": [],
    }

    for package in PACKAGES:
        manifest["packages"].append(copy_package(package, out_root))

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = [
        "# Standardized Robot Assets",
        "",
        "This folder is a self-contained export of the two ROS robot packages.",
        "",
        "Packages:",
    ]
    for pkg in manifest["packages"]:
        readme.append(f"- `{pkg['name']}` -> `{pkg['destination']}`")
        if pkg["aliases"]:
            readme.append(f"  - aliases: {', '.join(pkg['aliases'])}")
        if pkg["default_variants"]:
            readme.append(f"  - variants: {', '.join(pkg['default_variants'])}")
    readme.append("")
    readme.append("The exporter rewrites `package://...` mesh references to relative paths inside each copied package.")
    (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
