from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing sim manifest to a senior-like single-camera compatibility manifest. "
            "The selected camera is duplicated as side/top so the existing dual-camera model can be used "
            "while the visual input semantics match branchs more closely."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", choices=["side", "top"], default="side")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = []
    missing = 0
    for sample in manifest.get("samples", []):
        images = sample.get("images", {})
        chosen = images.get(args.camera)
        if not chosen:
            missing += 1
            continue
        converted = copy.deepcopy(sample)
        converted["images"] = {"side": chosen, "top": chosen}
        converted.setdefault("source", {})
        if isinstance(converted["source"], dict):
            converted["source"].update(
                {
                    "single_camera_compat": True,
                    "single_camera_source": args.camera,
                    "original_manifest": str(manifest_path),
                    "note": (
                        "Selected sim camera duplicated as side/top for branchs-style senior-like "
                        "input-schema alignment diagnostics."
                    ),
                }
            )
        samples.append(converted)
        if args.max_samples and args.max_samples > 0 and len(samples) >= int(args.max_samples):
            break

    out_manifest = copy.deepcopy(manifest)
    out_manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_manifest["mode"] = "single_camera_seniorlike_compat"
    out_manifest["source_manifest"] = str(manifest_path)
    out_manifest["single_camera_source"] = str(args.camera)
    out_manifest["image_semantics"] = (
        f"sim {args.camera} camera duplicated as both side/top for branchs-style single-camera compatibility"
    )
    out_manifest["observation_schema_recommended"] = "senior_piper_real_like"
    out_manifest["elite_action_representation_recommended"] = "tcp_delta"
    out_manifest["piper_head_recommended"] = "step_classification"
    out_manifest["samples"] = samples
    out_manifest["counts"] = {
        "samples": len(samples),
        "missing_selected_camera": missing,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out_manifest["counts"], indent=2, ensure_ascii=False))
    print(f"wrote single-camera senior-like manifest to {out_path}")


if __name__ == "__main__":
    main()
