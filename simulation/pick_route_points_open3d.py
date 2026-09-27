import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def load_mesh(path: str):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {path}")
    mesh.compute_vertex_normals()
    return mesh


def make_pickable_point_cloud(mesh, sample_count: int, point_size_hint: float = 0.0):
    pcd = mesh.sample_points_uniformly(number_of_points=sample_count)
    pcd.paint_uniform_color([0.05, 0.35, 0.95])
    return pcd


def make_route_template(points, existing_route=None, as_wire_visual_via: bool = False):
    route = existing_route or {}
    route_points = [p["route_point"] for p in points]
    if as_wire_visual_via:
        if not route_points:
            return route
        route["wire_visual_via_points"] = [
            {"frame": "route_raw", "point": point}
            for point in route_points
        ]
        route["_wire_visual_via_note"] = (
            "Visual-only guidewire tail via points. The MuJoCo renderer projects these raw route-frame "
            "points onto the active registered route, then draws Piper TCP -> first via -> route -> red tip."
        )
        return route
    route.setdefault("entry", route_points[0] if route_points else [0.0, 0.0, 0.0])
    route.setdefault("shared_waypoints", route_points[1:-2] if len(route_points) >= 4 else [])
    if len(points) >= 2:
        route.setdefault("left_target", route_points[-2])
        route.setdefault("right_target", route_points[-1])
    else:
        route.setdefault("left_target", [0.0, 0.0, 0.0])
        route.setdefault("right_target", [0.0, 0.0, 0.0])
    route.setdefault("left_waypoints", [])
    route.setdefault("right_waypoints", [])
    route["_picked_point_order_note"] = (
        "Default mapping assumes picked order: entry, shared waypoints..., left_target, right_target. "
        "Edit the JSON if your intended ordering is different."
    )
    return route


def centerize_point(point, all_points, tree, k: int):
    k = min(k, len(all_points))
    _dist, idx = tree.query(point, k=k)
    local = all_points[np.atleast_1d(idx)]
    return np.median(local, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Pick route points on the STL mesh with Open3D.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--out", default="simulation/routes/picked_points.json")
    parser.add_argument("--route-out", default="simulation/routes/picked_route.json")
    parser.add_argument("--existing-route", default="")
    parser.add_argument("--sample-count", type=int, default=200000)
    parser.add_argument("--center-k", type=int, default=600)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--show-mesh", action="store_true")
    parser.add_argument(
        "--as-wire-visual-via",
        action="store_true",
        help="Write picked points as visual-only wire_visual_via_points instead of entry/target route points.",
    )
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    pcd = make_pickable_point_cloud(mesh, args.sample_count)
    points = np.asarray(pcd.points)
    center_tree = cKDTree(points)

    if args.show_mesh:
        mesh.paint_uniform_color([0.75, 0.82, 0.9])

    print("\nOpen3D point picking controls:")
    print("  Shift + left click : pick a point")
    print("  Shift + right click: undo last pick")
    print("  Q or close window  : finish picking\n")
    print("Suggested picking order:")
    if args.as_wire_visual_via:
        print("  visual via points from Piper TCP outlet into the vessel lumen, in tail-to-tip order\n")
    else:
        print("  entry -> shared waypoints along the full vessel -> left_target -> right_target\n")
    print(f"Sampling {args.sample_count} points from the STL for easier picking.")
    print("Tip: zoom in before picking. Pick the blue point cloud, not empty space.\n")
    if args.no_center:
        print("Route config will use picked surface points directly (--no-center enabled).\n")
    else:
        print(f"Route config will use local centerized points estimated from {args.center_k} neighbors.\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick guidewire route points", width=1280, height=900)
    if args.show_mesh:
        vis.add_geometry(mesh)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 4.0
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.run()
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    picked = []
    for order, idx in enumerate(picked_indices):
        surface_point = points[idx].astype(float)
        center_point = surface_point if args.no_center else centerize_point(surface_point, points, center_tree, args.center_k)
        picked.append(
            {
                "order": order,
                "point_index": int(idx),
                "surface_point": surface_point.tolist(),
                "center_point": center_point.astype(float).tolist(),
                "route_point": center_point.astype(float).tolist(),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"mesh": args.mesh, "picked_points": picked}, indent=2), encoding="utf-8")
    print(f"Saved picked points to {out_path}")

    existing_route = None
    if args.existing_route:
        existing_route = json.loads(Path(args.existing_route).read_text(encoding="utf-8"))
    if args.as_wire_visual_via and not picked:
        print("No visual via points picked; leaving route config unchanged.")
        return
    route = make_route_template(picked, existing_route=existing_route, as_wire_visual_via=args.as_wire_visual_via)
    route_path = Path(args.route_out)
    route_path.parent.mkdir(parents=True, exist_ok=True)
    route_path.write_text(json.dumps(route, indent=2), encoding="utf-8")
    print(f"Saved route draft to {route_path}")

    if picked:
        print("\nPicked coordinates:")
        for item in picked:
            print(f"  {item['order']:02d}: route={item['route_point']} surface={item['surface_point']}")
    else:
        print("No points picked.")


if __name__ == "__main__":
    main()
