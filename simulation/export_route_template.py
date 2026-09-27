import argparse
import json
from pathlib import Path

from simulation.mesh_guidewire_3d_env import Mesh3DEnvConfig, MeshGuidewire3DEnv


def main():
    parser = argparse.ArgumentParser(description="Export a route config template from current automatic 3D route points.")
    parser.add_argument("--mesh", default="utils/interface/model/0422.stl")
    parser.add_argument("--out", default="simulation/routes/route_template.json")
    args = parser.parse_args()

    env = MeshGuidewire3DEnv(Mesh3DEnvConfig(mesh_path=args.mesh))
    data = {
        "entry": env.entry.astype(float).tolist(),
        "shared_waypoints": [],
        "left_waypoints": [],
        "right_waypoints": [],
        "left_target": env.targets["left"].astype(float).tolist(),
        "right_target": env.targets["right"].astype(float).tolist(),
        "_notes": [
            "Edit entry/waypoints/targets to force the guidewire through the desired full vessel route.",
            "Each route is planned as entry -> shared_waypoints -> side_waypoints -> side_target.",
            "Points are snapped to the nearest STL mesh vertex before surface-graph path planning.",
        ],
        "_bounds": env.bounds.astype(float).tolist(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"saved route template to {out_path}")


if __name__ == "__main__":
    main()

