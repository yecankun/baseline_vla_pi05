from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import shutil
from pathlib import Path
from typing import Any


def safe_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<unavailable>"


def class_methods(cls: type[Any]) -> dict[str, str]:
    names = (
        "__init__",
        "forward",
        "predict_action_chunk",
        "select_action",
        "reset",
        "embed_prefix",
        "embed_suffix",
        "sample_actions",
    )
    return {
        name: safe_signature(getattr(cls, name))
        for name in names
        if hasattr(cls, name)
    }


def source_class_names(module: Any) -> list[str]:
    return sorted(
        name
        for name, value in vars(module).items()
        if inspect.isclass(value) and getattr(value, "__module__", None) == module.__name__
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the installed LeRobot PI05 implementation before adding a project mixed action head. "
            "This is a read-only interface probe and does not instantiate model weights."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-out", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.policies.pi05 import modeling_pi05
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    source_path = Path(inspect.getfile(PI05Policy)).resolve()
    source_bytes = source_path.read_bytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.source_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, args.source_out)

    classes: dict[str, Any] = {}
    for name in source_class_names(modeling_pi05):
        value = getattr(modeling_pi05, name)
        classes[name] = {
            "methods": class_methods(value),
            "bases": [base.__name__ for base in value.__bases__],
        }

    try:
        version = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"

    report = {
        "scope": "installed LeRobot PI05 interface inspection only",
        "lerobot_version": version,
        "module": modeling_pi05.__name__,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "copied_source": str(args.source_out),
        "pi05_policy_signature": safe_signature(PI05Policy),
        "pi05_config_signature": safe_signature(PI05Config),
        "classes": classes,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
