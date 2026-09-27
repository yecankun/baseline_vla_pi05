from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ASSETS = {
    "vessel_mesh": "utils/interface/model/0422.stl",
    "route_config": "simulation/routes/vessel_0422_wire_route_v1.json",
    "camera_config": "simulation/camera_configs/mujoco_camera_top_manual_v1.json",
    "scene_config": "simulation_output/robot_scene_mvp/scene_config.json",
}


ISAAC_BUILD_STAGE_SCRIPT = r'''from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _points_from_rows(rows: Any) -> list[list[float]]:
    points: list[list[float]] = []
    if not isinstance(rows, list):
        return points
    for row in rows:
        if isinstance(row, dict):
            point = row.get("point")
        else:
            point = row
        if isinstance(point, list) and len(point) >= 3:
            points.append([float(point[0]), float(point[1]), float(point[2])])
    return points


def _route_wire_points(route: dict[str, Any], sample: dict[str, Any]) -> list[list[float]]:
    task = str(sample.get("task") or "")
    points: list[list[float]] = []

    outlet = route.get(f"{task}_wire_visual_piper_exit_point") or route.get("wire_visual_piper_exit_point")
    if isinstance(outlet, dict) and isinstance(outlet.get("point"), list):
        points.append([float(x) for x in outlet["point"][:3]])

    entry = route.get(f"{task}_wire_visual_entry_point") or route.get("wire_visual_entry_point")
    if isinstance(entry, dict) and isinstance(entry.get("point"), list):
        points.append([float(x) for x in entry["point"][:3]])

    task_route_points = _points_from_rows(route.get(f"{task}_wire_visual_route_points"))
    points.extend(task_route_points if task_route_points else _points_from_rows(route.get("wire_visual_route_points")))

    task_dynamic = route.get(f"{task}_wire_visual_dynamic_route_points")
    points.extend(_points_from_rows(task_dynamic if task_dynamic is not None else route.get("wire_visual_dynamic_route_points")))

    sim_state = sample.get("sim_state") if isinstance(sample.get("sim_state"), dict) else {}
    tip = sim_state.get("tip_pos") or sim_state.get("estimated_tip_pos_3d")
    if isinstance(tip, list) and len(tip) >= 3:
        points.append([float(tip[0]), float(tip[1]), float(tip[2])])

    deduped: list[list[float]] = []
    for point in points:
        if not deduped or sum((point[i] - deduped[-1][i]) ** 2 for i in range(3)) > 1e-10:
            deduped.append(point)
    return deduped


def _bounds_center_radius(route: dict[str, Any]) -> tuple[list[float], float]:
    bounds = route.get("_bounds")
    if isinstance(bounds, list) and len(bounds) == 2:
        lo = [float(x) for x in bounds[0][:3]]
        hi = [float(x) for x in bounds[1][:3]]
        center = [(lo[i] + hi[i]) * 0.5 for i in range(3)]
        radius = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3))) * 0.5
        return center, max(radius, 0.25)
    return [0.0, 0.0, 0.0], 0.5


def _camera_pose(camera_cfg: dict[str, Any], route: dict[str, Any], camera_name: str) -> tuple[list[float], list[float]]:
    params = camera_cfg.get(camera_name) if isinstance(camera_cfg.get(camera_name), dict) else {}
    center, radius = _bounds_center_radius(route)
    offset = params.get("lookat_offset", [0.0, 0.0, 0.0])
    target = [center[i] + float(offset[i]) for i in range(3)]
    azim = math.radians(float(params.get("azimuth", 0.0)))
    elev = math.radians(float(params.get("elevation", -15.0)))
    dist = max(radius * float(params.get("distance_scale", 1.0)), 0.3)
    eye = [
        target[0] + dist * math.cos(elev) * math.sin(azim),
        target[1] - dist * math.cos(elev) * math.cos(azim),
        target[2] + dist * math.sin(elev),
    ]
    return eye, target


async def _convert_asset(src: Path, dst: Path) -> bool:
    import omni.kit.asset_converter

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    converter = omni.kit.asset_converter.get_instance()
    task = converter.create_converter_task(str(src), str(dst))
    return bool(await task.wait_until_finished())


def _make_material(stage: Any, path: str, color: tuple[float, float, float], roughness: float = 0.45) -> Any:
    from pxr import UsdShade, Sdf

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _make_stage(
    stage_path: Path,
    sample: dict[str, Any],
    route: dict[str, Any],
    camera_cfg: dict[str, Any],
    scene_cfg: dict[str, Any],
    vessel_usd: Path,
    camera_name: str,
) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    vessel_mat = _make_material(stage, "/World/Materials/VesselGlass", (0.45, 0.62, 0.58), roughness=0.18)
    wire_mat = _make_material(stage, "/World/Materials/WireBlack", (0.015, 0.014, 0.012), roughness=0.55)
    tip_mat = _make_material(stage, "/World/Materials/WireTipRed", (0.78, 0.04, 0.02), roughness=0.42)
    table_mat = _make_material(stage, "/World/Materials/GreenBackground", (0.18, 0.34, 0.25), roughness=0.75)

    table = UsdGeom.Cube.Define(stage, "/World/GreenTable")
    table.AddScaleOp().Set(Gf.Vec3f(1.4, 1.4, 0.01))
    table.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.9, -0.13))
    _bind(table.GetPrim(), table_mat)

    vessel = UsdGeom.Xform.Define(stage, "/World/Vessel")
    vessel.GetPrim().GetReferences().AddReference(str(vessel_usd))
    scale = float(scene_cfg.get("vessel_scale", 0.077))
    vessel.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    _bind(vessel.GetPrim(), vessel_mat)

    points = _route_wire_points(route, sample)
    if len(points) >= 2:
        curve = UsdGeom.BasisCurves.Define(stage, "/World/GuidewireTail")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        curve.CreateWidthsAttr([0.0016])
        _bind(curve.GetPrim(), wire_mat)

        tip = UsdGeom.Sphere.Define(stage, "/World/GuidewireTip")
        tip.CreateRadiusAttr(0.004)
        tip.AddTranslateOp().Set(Gf.Vec3f(*points[-1]))
        _bind(tip.GetPrim(), tip_mat)

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(550.0)
    light.CreateAngleAttr(3.0)
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(80.0)
    dome.CreateColorAttr(Gf.Vec3f(0.72, 0.80, 0.76))

    eye, target = _camera_pose(camera_cfg, route, camera_name)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 100.0))
    camera.CreateFocalLengthAttr(35.0)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0))
    camera.AddTransformOp().Set(view.GetInverse())
    stage.GetRootLayer().defaultPrim = "World"
    stage.GetRootLayer().Save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build USD stages for the Project 2026 Isaac renderer-only spike package.")
    parser.add_argument("--package", default="package_manifest.json")
    parser.add_argument("--out", default="isaac_stages")
    parser.add_argument("--camera", default=None, help="Override package camera, e.g. side or top.")
    parser.add_argument("--frame-id", default=None, help="Build only one frame id from samples.jsonl.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit generated frames; 0 means all selected samples.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    app = SimulationApp({"headless": bool(args.headless)})
    try:
        package_path = Path(args.package).resolve()
        package_dir = package_path.parent
        package = _load_json(package_path)
        assets = package["packaged_assets"]
        route = _load_json(package_dir / assets["route_config"])
        camera_cfg = _load_json(package_dir / assets["camera_config"])
        scene_cfg = _load_json(package_dir / assets["scene_config"])
        samples = _load_jsonl(package_dir / package["main_samples_jsonl"])
        if args.frame_id:
            samples = [row for row in samples if row.get("frame_id") == args.frame_id]
        if args.max_frames > 0:
            samples = samples[: args.max_frames]
        if not samples:
            raise RuntimeError("No samples selected.")

        vessel_src = (package_dir / assets["vessel_mesh"]).resolve()
        vessel_usd = (package_dir / "isaac_assets" / "vessel_0422.usd").resolve()
        ok = asyncio.get_event_loop().run_until_complete(_convert_asset(vessel_src, vessel_usd))
        if not ok:
            raise RuntimeError(f"Failed to convert vessel mesh: {vessel_src}")

        out_dir = (package_dir / args.out).resolve()
        camera_name = args.camera or str(package.get("camera", "side"))
        written = []
        for sample in samples:
            frame_id = str(sample["frame_id"])
            stage_path = out_dir / f"{frame_id}.usd"
            _make_stage(stage_path, sample, route, camera_cfg, scene_cfg, vessel_usd, camera_name)
            written.append(str(stage_path))
        print(json.dumps({"written": written, "camera": camera_name}, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
'''


STATE_KEYS = [
    "elite_tcp_pose_6d",
    "robot_state",
    "controller_state",
    "tip_pos",
    "heading",
    "path_progress",
    "piper_step",
    "piper_insertion_length",
    "magnetic_pose",
    "elirobot_pose",
    "estimated_tip_pos_3d",
    "estimated_tip_heading_3d",
    "estimated_wall_margin",
    "estimated_route_progress",
    "estimated_wall_normal_3d",
    "estimated_route_tangent_3d",
]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a samples list")
    return samples


def _group_by_task(samples: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, sample in enumerate(samples):
        task = str(sample.get("task") or sample.get("branch") or "unknown")
        grouped.setdefault(task, []).append((index, sample))
    return grouped


def _pick_evenly(items: list[tuple[int, dict[str, Any]]], count: int) -> list[tuple[int, dict[str, Any]]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    return [items[int(round(pos))] for pos in np.linspace(0, len(items) - 1, count)]


def _image_path(sample: dict[str, Any], camera: str) -> str | None:
    images = sample.get("images")
    if not isinstance(images, dict):
        return None
    value = images.get(camera) or images.get("side") or images.get("top")
    return str(value) if value else None


def _resolve(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _compact_state(sample: dict[str, Any]) -> dict[str, Any]:
    state = sample.get("state") if isinstance(sample.get("state"), dict) else {}
    result: dict[str, Any] = {}
    for key in STATE_KEYS:
        if key in state:
            result[key] = state[key]
    return result


def _copy_image(src: Path, dst: Path, base_dir: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(base_dir).as_posix())


def _copy_asset(src_text: str, root: Path, out_dir: Path, subdir: str) -> str:
    src = _resolve(src_text, root)
    if not src.exists():
        raise FileNotFoundError(src)
    dst = out_dir / "assets" / subdir / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.relative_to(out_dir).as_posix())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a small renderer-only Isaac Sim spike package. The package preserves the "
            "current MuJoCo route/control semantics and selected reference images, so Isaac can "
            "be tested as a photorealistic rendering layer without migrating physics/control."
        )
    )
    parser.add_argument("--sim-manifest", required=True)
    parser.add_argument("--real-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera", default="side")
    parser.add_argument("--samples-per-task", type=int, default=5)
    parser.add_argument("--copy-reference-images", action="store_true")
    parser.add_argument("--no-copy-assets", action="store_true", help="Keep asset paths as project-relative pointers instead of copying them into the package.")
    parser.add_argument("--vessel-mesh", default=DEFAULT_ASSETS["vessel_mesh"])
    parser.add_argument("--route-config", default=DEFAULT_ASSETS["route_config"])
    parser.add_argument("--camera-config", default=DEFAULT_ASSETS["camera_config"])
    parser.add_argument("--scene-config", default=DEFAULT_ASSETS["scene_config"])
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_samples = _load_manifest(Path(args.sim_manifest))
    real_samples = _load_manifest(Path(args.real_manifest))
    sim_groups = _group_by_task(sim_samples)
    real_groups = _group_by_task(real_samples)
    tasks = sorted(set(sim_groups) & set(real_groups))
    if not tasks:
        raise ValueError("No shared tasks between sim and real manifests")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in tasks:
        sim_picks = _pick_evenly(sim_groups[task], args.samples_per_task)
        real_picks = _pick_evenly(real_groups[task], args.samples_per_task)
        for pair_index, ((sim_index, sim_sample), (real_index, real_sample)) in enumerate(
            zip(sim_picks, real_picks)
        ):
            sim_image = _image_path(sim_sample, args.camera)
            real_image = _image_path(real_sample, args.camera)
            if not sim_image or not real_image:
                missing.append(f"{task}/{pair_index}: missing image path")
                continue
            sim_abs = _resolve(sim_image, root)
            real_abs = _resolve(real_image, root)
            if not sim_abs.exists() or not real_abs.exists():
                missing.append(f"{task}/{pair_index}: missing image file sim={sim_abs} real={real_abs}")
                continue

            frame_id = f"{task}_{pair_index:03d}"
            row: dict[str, Any] = {
                "frame_id": frame_id,
                "task": task,
                "camera": args.camera,
                "sim_sample_index": sim_index,
                "real_sample_index": real_index,
                "sim_reference_image": sim_image,
                "real_reference_image": real_image,
                "sim_state": _compact_state(sim_sample),
                "sim_action": sim_sample.get("action", {}),
                "real_state": _compact_state(real_sample),
                "real_reference_action": real_sample.get("action") or real_sample.get("reference_action", {}),
                "render_target": {
                    "isaac_image": f"isaac_renders/{frame_id}.png",
                    "comparison_note": "Render this state in Isaac, then compare against sim_reference_image and real_reference_image.",
                },
            }
            if args.copy_reference_images:
                row["copied_sim_reference_image"] = _copy_image(
                    sim_abs,
                    out_dir / "reference_images" / "sim" / f"{frame_id}{sim_abs.suffix}",
                    out_dir,
                )
                row["copied_real_reference_image"] = _copy_image(
                    real_abs,
                    out_dir / "reference_images" / "real" / f"{frame_id}{real_abs.suffix}",
                    out_dir,
                )
            rows.append(row)

    samples_path = out_dir / "samples.jsonl"
    _write_jsonl(samples_path, rows)

    assets = {
        "vessel_mesh": args.vessel_mesh,
        "route_config": args.route_config,
        "camera_config": args.camera_config,
        "scene_config": args.scene_config,
    }
    if args.no_copy_assets:
        packaged_assets = dict(assets)
    else:
        packaged_assets = {
            "vessel_mesh": _copy_asset(args.vessel_mesh, root, out_dir, "mesh"),
            "route_config": _copy_asset(args.route_config, root, out_dir, "config"),
            "camera_config": _copy_asset(args.camera_config, root, out_dir, "config"),
            "scene_config": _copy_asset(args.scene_config, root, out_dir, "config"),
        }
    (out_dir / "isaac_build_stage.py").write_text(ISAAC_BUILD_STAGE_SCRIPT, encoding="utf-8")

    package = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Isaac Sim renderer-only feasibility spike",
        "sim_manifest": args.sim_manifest,
        "real_manifest": args.real_manifest,
        "camera": args.camera,
        "tasks": tasks,
        "samples": len(rows),
        "missing": missing,
        "assets": assets,
        "packaged_assets": packaged_assets,
        "main_samples_jsonl": "samples.jsonl",
        "isaac_build_script": "isaac_build_stage.py",
        "migration_boundary": {
            "keep_in_mujoco_currently": [
                "guidewire route/progress semantics",
                "fixed S-bend visual prefix plus dynamic route-to-tip logic",
                "Piper/Elite action schema and controller semantics",
                "training/evaluation/manifest pipeline",
            ],
            "test_in_isaac": [
                "camera/background/material/lighting realism",
                "robot/fixture visual appearance",
                "branchs-like crop, perspective, and occlusion",
            ],
        },
    }
    (out_dir / "package_manifest.json").write_text(
        json.dumps(package, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    readme = f"""# Isaac Renderer-Only Spike Package

This package is for testing Isaac Sim as a rendering layer only. It intentionally
does not migrate MuJoCo guidewire physics, control, data collection, or training.

## Boundary

Keep in the current MuJoCo/project pipeline:

- guidewire route semantics and the tuned S-bend/prefix/dynamic route behavior;
- Piper/Elite action schema and controller logs;
- manifest/training/evaluation diagnostics.

Use Isaac Sim only to test:

- branchs-like camera crop and perspective;
- green lab background / fixture / robot visual appearance;
- lighting, material, and occlusion realism.

## Files

- `package_manifest.json`: package metadata and asset/config pointers.
- `samples.jsonl`: selected frame states and sim/real reference images.
- `assets/`: copied vessel STL plus route/camera/scene configs, unless
  `--no-copy-assets` was used.
- `isaac_build_stage.py`: run this on a machine with Isaac Sim installed to
  convert the vessel mesh to USD and build per-frame USD stages.
- `reference_images/`: copied references if `--copy-reference-images` was used.

## Run On The Isaac Host

Copy this whole directory to the Isaac host, then run from inside the package:

```powershell
<ISAAC_SIM_ROOT>\python.bat isaac_build_stage.py --package package_manifest.json --headless
```

On Linux:

```bash
<ISAAC_SIM_ROOT>/python.sh isaac_build_stage.py --package package_manifest.json --headless
```

This writes one USD file per selected frame under `isaac_stages/`. Open those
USD files in Isaac Sim first for visual inspection. Rendering automation should
only be added after the USD scene loads correctly on your Isaac version.

## Acceptance Criteria

Render the selected states in Isaac and run the same visual-domain audit against
`branchs`. Continue only if Isaac images are visibly and statistically closer to
real images than the current MuJoCo renderings.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "out": str(out_dir),
                "tasks": tasks,
                "samples": len(rows),
                "missing": missing,
                "samples_jsonl": str(samples_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
