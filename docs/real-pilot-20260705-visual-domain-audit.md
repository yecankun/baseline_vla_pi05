# Real Pilot 20260705 Visual-Domain Audit

## Purpose

This audit compares the current accepted MuJoCo branchs-like render candidate
against the failed/stuck real pilot images collected on 2026-07-05.

Reference inputs:

```text
sim:
  simulation_output/formal_tip_line_branchs_like_v1_full/manifest.json

real:
  collected_data/real_pilot_20260705_left_mode2_linked_elite_piper_001
  collected_data/real_pilot_20260705_right_mode2_linked_elite_piper_fast_001
```

The real pilot runs are not successful training data because the guidewire
stuck in the physical vessel and the old capture shows large image-pose/action
timestamp deltas. They are still useful as real-camera visual references.

## Outputs

```text
docs/_sim_real_visual_domain_audit_real_pilot_20260705_side
docs/_sim_real_visual_domain_audit_real_pilot_20260705_top
```

Each output directory contains:

```text
contact_sheet.png
report.md
summary.json
```

## Side Camera Result

```text
sample pairs: 20
sim luminance_mean: 118.525
real luminance_mean: 104.246
sim-real luminance delta: +14.278

sim luminance_std: 23.247
real luminance_std: 54.828
sim-real luminance-std delta: -31.581

sim saturation_mean: 0.408
real saturation_mean: 0.264
sim-real saturation delta: +0.144

sim dark_ratio: 0.001
real dark_ratio: 0.268
sim-real dark-ratio delta: -0.267

sim edge_density: 0.027
real edge_density: 0.052
sim-real edge-density delta: -0.025
```

Interpretation:

```text
The side view is closer than the top view in average luminance, but the real
side camera has much stronger contrast, many more dark regions, and denser
edges. The current sim side render remains cleaner and more uniformly lit than
the real pilot camera.
```

## Top Camera Result

```text
sample pairs: 20
sim luminance_mean: 155.367
real luminance_mean: 97.882
sim-real luminance delta: +57.486

sim luminance_std: 26.582
real luminance_std: 56.283
sim-real luminance-std delta: -29.701

sim saturation_mean: 0.344
real saturation_mean: 0.331
sim-real saturation delta: +0.013

sim dark_ratio: 0.000
real dark_ratio: 0.305
sim-real dark-ratio delta: -0.305

sim edge_density: 0.034
real edge_density: 0.062
sim-real edge-density delta: -0.028
```

Interpretation:

```text
The top camera is the larger mismatch. The current sim top view is far brighter
and cleaner, while the real top images contain substantial dark regions,
stronger contrast, and denser edges. This is consistent with real camera
exposure, glass-vessel/reflection effects, background/fixture clutter, and
slight blur/crop differences not yet captured by the MuJoCo renderer.
```

## Current Conclusion

The branchs-like render preset remains useful, but the 2026-07-05 real pilot
images show that it is not visually aligned enough to be considered solved.
The next local simulator-realism work should focus on rendering/camera effects,
not another small-BC rollout-tuning loop.

Recommended local render targets:

1. Make the top camera darker and less uniformly lit.
2. Add stronger dark-background / fixture / occlusion structure.
3. Add mild camera blur and exposure variation.
4. Add glass-vessel-like reflection/highlight artifacts if practical.
5. Preserve the accepted thin black guidewire tail and wider red head.
6. Keep S-bend route geometry fixed unless visual review finds a new route
   issue.

Before accepting any new render preset, rerun:

```powershell
.\.venv\Scripts\python.exe tools\audit_sim_real_visual_domain.py `
  --sim-manifest simulation_output\<new_sim_dataset>\manifest.json `
  --real-manifest collected_data\real_pilot_20260705_left_mode2_linked_elite_piper_001 collected_data\real_pilot_20260705_right_mode2_linked_elite_piper_fast_001 `
  --real-kind records `
  --out docs\_sim_real_visual_domain_audit_<new_name>_side `
  --camera side `
  --samples-per-task 10

.\.venv\Scripts\python.exe tools\audit_sim_real_visual_domain.py `
  --sim-manifest simulation_output\<new_sim_dataset>\manifest.json `
  --real-manifest collected_data\real_pilot_20260705_left_mode2_linked_elite_piper_001 collected_data\real_pilot_20260705_right_mode2_linked_elite_piper_fast_001 `
  --real-kind records `
  --out docs\_sim_real_visual_domain_audit_<new_name>_top `
  --camera top `
  --samples-per-task 10
```
