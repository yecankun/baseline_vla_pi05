# Standardized Robot Assets

This folder is a self-contained export of the two ROS robot packages.

Packages:
- `piper_description` -> `piper_description`
  - aliases: gripper_description
  - variants: urdf/piper_no_gripper_description.urdf, urdf/piper_description.urdf, urdf/piper_description_with_camera.urdf, urdf/piper_description_v100.urdf
- `elite_description` -> `elite_description`
  - variants: urdf/ec66_description.urdf, urdf/ec63_description.urdf, urdf/ec612_description.urdf

The exporter rewrites `package://...` mesh references to relative paths inside each copied package.