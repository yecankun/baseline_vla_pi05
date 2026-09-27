# Guidewire Simulation

This folder contains task-level guidewire simulation environments for early VLA
data format, policy, and visualization testing. They do not model true flexible
guidewire physics yet.

## What It Provides

- A simple Y-shaped 2D vessel environment.
- A projected-STL 2D vessel environment.
- An STL-derived 3D vessel environment.
- Guidewire tip position, heading, insertion length, and polyline history.
- Tactile/contact events generated from distance to the vessel wall.
- `gymnasium.Env` style `reset()` and `step(action)` APIs for STL environments.
- Rule-based expert for automatic imitation data generation.
- RGB frame rendering, rollout videos, and JSONL episode export.

## Action

```text
{
  "steer": [dx, dy],
  "advance": 0 or 1,
  "retreat": 0 or 1
}
```

## Observation

Each observation includes the language instruction, tip state, target state,
distance-to-wall, contact direction, contact strength, and success/failure flags.

## Collect Example Data

```bash
python -m simulation.collect_mvp1 --episodes 4 --out simulation_output/mvp1
```

With the projected STL vessel environment:

```bash
python -m simulation.collect_mesh --episodes 4 --out simulation_output/mesh
```

With the STL-derived 3D environment:

```bash
python -m simulation.collect_mesh_3d --episodes 4 --out simulation_output/mesh_3d
```

Export and use a route template with explicit entry/waypoints/targets:

```bash
python -m simulation.export_route_template --out simulation/routes/route_template.json
python -m simulation.collect_mesh_3d --episodes 4 --route-config simulation/routes/route_template.json --out simulation_output/mesh_3d_configured
```

Pick route points interactively on the STL mesh:

```bash
python -m simulation.pick_route_points_open3d --out simulation/routes/picked_points.json --route-out simulation/routes/picked_route.json
```

The picker samples the STL into a blue point cloud because point picking is more
reliable than mesh-surface picking in Open3D. Suggested picking order is
`entry -> shared waypoints -> left_target -> right_target`.

Use `--sample-count 400000` if the picked points feel too sparse, or
`--show-mesh` if you also want the original mesh shown as context.
By default, the route config uses a local centerized point estimated from nearby
sampled surface points, while `picked_points.json` keeps both the original
surface point and the centerized route point. Use `--no-center` only if you want
the route to use raw surface points.

Collect contact-rich recovery trajectories:

```bash
python -m simulation.collect_mesh_3d --episodes 4 --scenario contact_recovery --out simulation_output/mesh_3d_contact
```

Render an automatic 3D rollout video:

```bash
python -m simulation.visualize_mesh_3d --task left --out simulation_output/mesh_3d_visual/left.mp4 --snapshot simulation_output/mesh_3d_visual/left.png
```

Render a smoother mesh-based 3D rollout with transparent vessel and tube-shaped
guidewire:

```bash
python -m simulation.render_mesh_3d_pyrender --task left --out simulation_output/mesh_3d_visual_pyrender/left.mp4 --snapshot simulation_output/mesh_3d_visual_pyrender/left.png
```

Render a contact-recovery rollout and save the strongest-contact frame:

```bash
python -m simulation.render_mesh_3d_pyrender --task left --scenario contact_recovery --out simulation_output/mesh_3d_visual_pyrender/left_contact.mp4 --snapshot simulation_output/mesh_3d_visual_pyrender/left_contact_final.png --contact-snapshot simulation_output/mesh_3d_visual_pyrender/left_contact_peak.png
```

Build a first VLA-style dataset manifest:

```bash
python -m simulation.build_vla_dataset --episodes-per-combo 2 --image-mode camera_pair --out simulation_output/vla_dataset_v1
```

Build a dual-arm style dataset manifest:

```bash
python -m simulation.collect_dual_arm_3d --episodes-per-combo 2 --out simulation_output/dual_arm_dataset_v1
```

For stronger recovery data, prefer:

```bash
python -m simulation.collect_dual_arm_3d --episodes-per-combo 2 --scenarios contact_recovery noisy --start-progress-mode mixed --out simulation_output/dual_arm_dataset_aug
```

Train a simple dual-arm baseline:

```bash
python -m simulation.train_dual_arm_baseline --manifest simulation_output/dual_arm_dataset_v1/manifest.json --out simulation_output/baseline_dual_arm_camera_state
```

`camera_pair` saves two camera-like views per step:

```text
frames/side/000000.png
frames/top/000000.png
```

Train the simple dual-camera + state baseline:

```bash
python -m simulation.train_baseline --manifest simulation_output/vla_dataset_v1/manifest.json --out simulation_output/baseline_dual_camera_state
```

Evaluate it in closed-loop simulation:

```bash
python -m simulation.eval_baseline_rollout --checkpoint simulation_output/baseline_dual_camera_state/best_model.pt --route-config simulation/routes/vessel_0422_route_v1.json --video --out simulation_output/baseline_rollout
```

Each episode contains:

```text
meta.json
states.jsonl
actions.jsonl
frames/000000.png
```

## Scope

MVP1 is designed to make the task loop alive before Isaac Sim, ROS, STL mesh
collision, or flexible guidewire dynamics are introduced.

`mesh_guidewire_env.py` is the next step: it loads the existing STL vessel,
projects it to a 2D mask, exposes a `gymnasium.Env` interface, and generates
contact/tactile signals from distance to the projected vessel wall. It is still
task-level simulation, not flexible guidewire physics.

`mesh_guidewire_3d_env.py` lifts the STL-projected paths back to 3D, exposes a
3D action/state space, estimates local vessel radius from nearby mesh vertices,
and renders XY/XZ/YZ views for debugging.
