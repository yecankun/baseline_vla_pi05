import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, Mesh3DPathExpert, Mesh3DRecoveryExpert, MeshGuidewire3DEnv


def make_tube(points, radius=0.018, sections=10):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return None
    cylinders = []
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length < 1e-5:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        transform = trimesh.geometry.align_vectors([0, 0, 1], segment / length)
        transform[:3, 3] = 0.5 * (start + end)
        cyl.apply_transform(transform)
        cylinders.append(cyl)
    if not cylinders:
        return None
    return trimesh.util.concatenate(cylinders)


def make_sphere(center, radius, color):
    sphere = trimesh.creation.uv_sphere(radius=radius, count=[16, 16])
    sphere.apply_translation(center)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.6,
        baseColorFactor=color,
    )
    return pyrender.Mesh.from_trimesh(sphere, material=material, smooth=True)


def make_arrow(start, direction, length=0.28, color=[1.0, 0.05, 0.05, 1.0]):
    direction = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm
    start = np.asarray(start, dtype=np.float32)
    shaft_len = length * 0.72
    shaft = trimesh.creation.cylinder(radius=0.018, height=shaft_len, sections=12)
    shaft_transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    shaft_transform[:3, 3] = start + direction * shaft_len * 0.5
    shaft.apply_transform(shaft_transform)

    cone = trimesh.creation.cone(radius=0.052, height=length * 0.28, sections=16)
    cone_transform = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cone_transform[:3, 3] = start + direction * (shaft_len + length * 0.14)
    cone.apply_transform(cone_transform)
    mesh = trimesh.util.concatenate([shaft, cone])
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.45,
        baseColorFactor=color,
    )
    return pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)


def camera_pose_for_mesh(bounds, yaw_deg=-38.0, pitch_deg=-20.0, distance_scale=1.05):
    center = bounds.mean(axis=0)
    span = float(np.max(bounds[1] - bounds[0]))
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    direction = np.array([
        np.cos(pitch) * np.sin(yaw),
        -np.cos(pitch) * np.cos(yaw),
        np.sin(pitch),
    ])
    eye = center + direction * span * distance_scale
    up = np.array([0.0, 0.0, 1.0])
    forward = center - eye
    forward /= max(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-6)
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


class PyrenderRolloutRenderer:
    def __init__(
        self,
        env: MeshGuidewire3DEnv,
        width=960,
        height=720,
        yaw=-38.0,
        pitch=-20.0,
        distance_scale=1.05,
        vessel_alpha=0.42,
        vessel_color=(0.48, 0.58, 0.72),
    ):
        self.env = env
        self.width = width
        self.height = height
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.camera_pose = camera_pose_for_mesh(env.bounds, yaw_deg=yaw, pitch_deg=pitch, distance_scale=distance_scale)
        self.vessel_alpha = vessel_alpha
        self.vessel_color = vessel_color

    def close(self):
        self.renderer.delete()

    def render(self):
        scene = pyrender.Scene(bg_color=[248, 248, 248, 255], ambient_light=[0.45, 0.45, 0.45])

        vessel_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.86,
            alphaMode="BLEND",
            baseColorFactor=[self.vessel_color[0], self.vessel_color[1], self.vessel_color[2], self.vessel_alpha],
        )
        vessel_mesh = pyrender.Mesh.from_trimesh(self.env.mesh, material=vessel_material, smooth=False)
        scene.add(vessel_mesh)

        shell_material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.95,
            alphaMode="BLEND",
            baseColorFactor=[0.25, 0.32, 0.42, min(self.vessel_alpha * 0.45, 0.28)],
        )
        shell = self.env.mesh.copy()
        shell.apply_scale(1.003)
        scene.add(pyrender.Mesh.from_trimesh(shell, material=shell_material, smooth=False))

        path_tube = make_tube(self.env.paths[self.env.task], radius=0.01, sections=8)
        if path_tube is not None:
            path_material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.55,
                baseColorFactor=[0.15, 0.72, 0.25, 1.0],
            )
            scene.add(pyrender.Mesh.from_trimesh(path_tube, material=path_material, smooth=True))

        wire_tube = make_tube(np.asarray(self.env.guidewire_points, dtype=np.float32), radius=0.017, sections=10)
        if wire_tube is not None:
            wire_material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                roughnessFactor=0.35,
                baseColorFactor=[0.95, 0.42, 0.08, 1.0],
            )
            scene.add(pyrender.Mesh.from_trimesh(wire_tube, material=wire_material, smooth=True))

        scene.add(make_sphere(self.env.tip, 0.055, [0.95, 0.02, 0.02, 1.0]))
        scene.add(make_sphere(self.env.targets[self.env.task], 0.07, [0.05, 0.75, 0.12, 1.0]))
        tactile = self.env._tactile(self.env.tip)
        if tactile["contact_flag"]:
            normal = np.asarray(tactile["contact_normal"], dtype=np.float32)
            arrow = make_arrow(self.env.tip, normal, length=0.22 + 0.22 * tactile["contact_strength"])
            if arrow is not None:
                scene.add(arrow)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)
        scene.add(camera, pose=self.camera_pose)
        light_pose = self.camera_pose.copy()
        light_pose[:3, 3] += np.array([0.0, 0.0, 1.5])
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=light_pose)

        color, _depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        frame = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
        text = f"{self.env.task}  step {self.env.step_count}  contact {tactile['contact_direction']} {tactile['contact_strength']:.2f}"
        cv2.putText(frame, text, (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (45, 45, 45), 2, cv2.LINE_AA)
        return frame


def main():
    parser = argparse.ArgumentParser(description="Render a polished 3D guidewire rollout with pyrender.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--route-config", default="")
    parser.add_argument("--task", choices=["left", "right"], default="left")
    parser.add_argument("--scenario", choices=["normal", "noisy", "contact_recovery"], default="normal")
    parser.add_argument("--contact-side", choices=["auto", "left", "right", "upper", "lower"], default="auto")
    parser.add_argument("--out", default="simulation_output/mesh_3d_visual_pyrender/left.mp4")
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--contact-snapshot", default="")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--yaw", type=float, default=-38.0)
    parser.add_argument("--pitch", type=float, default=-20.0)
    parser.add_argument("--distance-scale", type=float, default=1.05)
    parser.add_argument("--vessel-alpha", type=float, default=0.42)
    parser.add_argument("--vessel-color", choices=["blue", "gray", "red"], default="blue")
    parser.add_argument("--frame-stride", type=int, default=3)
    args = parser.parse_args()

    env = MeshGuidewire3DEnv(Mesh3DEnvConfig(mesh_path=args.mesh, route_config_path=args.route_config), seed=args.seed)
    if args.scenario == "normal":
        expert = Mesh3DPathExpert(rng_seed=args.seed, steer_noise=0.0)
    elif args.scenario == "noisy":
        expert = Mesh3DPathExpert(rng_seed=args.seed, steer_noise=0.035)
    else:
        expert = Mesh3DRecoveryExpert(rng_seed=args.seed, side=args.contact_side)
    _obs, info = env.reset(options={"task": args.task})
    vessel_colors = {
        "blue": (0.48, 0.58, 0.72),
        "gray": (0.55, 0.57, 0.60),
        "red": (0.72, 0.42, 0.42),
    }
    renderer = PyrenderRolloutRenderer(
        env,
        width=args.width,
        height=args.height,
        yaw=args.yaw,
        pitch=args.pitch,
        distance_scale=args.distance_scale,
        vessel_alpha=args.vessel_alpha,
        vessel_color=vessel_colors[args.vessel_color],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Could not open video writer for {out_path}")

    final_frame = None
    best_contact_frame = None
    best_contact_strength = -1.0
    try:
        while True:
            frame = renderer.render()
            writer.write(frame)
            final_frame = frame
            tactile = env._tactile(env.tip)
            if tactile["contact_strength"] > best_contact_strength:
                best_contact_strength = tactile["contact_strength"]
                best_contact_frame = frame.copy()
            terminated = False
            truncated = False
            for _ in range(max(1, args.frame_stride)):
                action = expert.act(env)
                _obs, _reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if terminated or truncated:
                frame = renderer.render()
                for _ in range(args.fps):
                    writer.write(frame)
                final_frame = frame
                break
    finally:
        writer.release()
        renderer.close()

    if args.snapshot:
        snap_path = Path(args.snapshot)
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snap_path), final_frame)
    if args.contact_snapshot and best_contact_frame is not None:
        contact_path = Path(args.contact_snapshot)
        contact_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(contact_path), best_contact_frame)

    obs_dict = info["obs_dict"]
    print(f"saved {out_path} task={args.task} success={obs_dict['success']} failure={obs_dict['failure_reason']}")


if __name__ == "__main__":
    main()
