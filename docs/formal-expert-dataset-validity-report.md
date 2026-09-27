# Formal Expert Dataset Validity Report

Last updated: 2026-06-23

## Scope

This report evaluates the current formal route-plan expert dataset as a
synthetic data-generation candidate, independent of behavior-cloning rollout
quality.

Primary artifacts:

```text
simulation_output/formal_route_plan_tip_dataset_step020_ahead010_v1
simulation_output/formal_route_plan_tip_dataset_step020_ahead010_stress_wide_v1
```

Related audit output:

```text
docs/_formal_expert_dataset_validity_audit
docs/_formal_expert_dataset_stress_wide_validity_audit
docs/_route_plan_anchor_phase_diag
```

Highest-priority criterion:

```text
simulation realism and real-system correspondence > BC rollout success
```

## Executive Judgment

The current formal route-plan expert dataset passes the present expert-data
quality gate and should be treated as the best current formal synthetic-data
reference.

The broader stress-wide expert-only run also passes the same gate. It widened
the start range to about `0.50-0.82`, accepted `80/80` episodes, reached
`env_success` on every episode, had zero rejected attempts, zero contact, no
stall/slow samples, positive wall clearance, complete side/top image references,
and millimeter-scale `tip_to_magnetic`.

It should not be treated as proof that the whole simulator is already
sim-to-real solved. The expert/control mechanism is plausible only under the
assumption that the real system can provide a registered vessel route, a
calibrated Elite robot-to-vessel transform, and a Piper/route schedule or
progress proxy. Those assumptions still need real-system validation.

BC rollout failures should not be used as the main reason to keep tuning the
expert. The latest diagnostics show that the small BC policy introduces a
closed-loop phase mismatch between scheduled route progress and actual
tip/Piper progress. That is a model/control-closure limitation, not direct
evidence that the expert dataset is invalid.

The first formal expert vs real alignment report has also been generated:

```text
docs/_formal_expert_real_alignment/formal_expert_real_alignment.md
```

It supports the current expert datasets on basic motion scale when compared by
saved samples: formal expert `elirobot_pose` step-p95 median is about
`5.56-5.68 mm/saved sample`, close to the real `branchs` Elite/magnetic-arm
pose step-p95 median of `5.07 mm/frame`. This is not a complete time-base
match: normalized per-env-step simulated Elite/magnetic motion is only about
`1.11-1.14 mm/env step`, so calibration and sampling cadence still need
explicit treatment.

The clearest remaining sim-to-real semantic gap is Piper. Senior's real Piper
inference path in `intervention/bc_piper/inference.py` uses binary labels:
`0` means stop/hold, and `1` calls `piper.step_forward(...)` to advance one
Piper step. The current formal simulator instead emits continuous
`piper_feed` with positive feed and negative retract commands. Its saved-sample
Piper transition fraction is about `0.94`, while real binary labels have median
transition fraction about `0.040`. Treat the current formal expert data as
clean for route-plan geometry/contact checks, but not yet as real-action-aligned
Piper imitation data. This does not mean retract should be removed from the
research direction: retract is likely a useful real control capability, but it
needs an explicit real Piper command path and corresponding labels rather than
being inherited implicitly from the simulator.

Implementation note: new collection outputs now keep `piper_feed` for continuous
MuJoCo execution and additionally write `piper_step_command`
(`-1=retract`, `0=hold`, `1=feed`) plus `piper_command_label`. This is a
compatibility step toward a real-implementable signed-step Piper interface; the
existing formal datasets must be recollected before they contain these fields.

## Dataset Quality Evidence

Dataset-level checks:

```text
episodes: 40
left/right: 20/20
success: 40/40
env_success: 40/40
rejected attempts: 0
samples: 1360
image references: 2720/2720 present
mode: tip_guided_wire
guidance_mode: physical
action schema: piper_feed_elite_joint
expert_mode: route_plan
route_plan_step: 0.20
elite_ahead: 0.010
```

Stress-wide expert-only check:

```text
artifact: simulation_output/formal_route_plan_tip_dataset_step020_ahead010_stress_wide_v1
audit: docs/_formal_expert_dataset_stress_wide_validity_audit
status: formal_valid
episodes: 80
left/right: 40/40
success: 80/80
env_success: 80/80
rejected attempts: 0
samples: 2661
image references: 5322/5322 present
start_fraction min/median/max: 0.5068 / 0.6384 / 0.8181
max_contact_strength max: 0.0
contact_p95 max: 0.0
min_segment_distance_to_wall min: 1.386 mm
stall_fraction max: 0.0
slow_fraction max: 0.0
tip_to_magnetic sample median/p95/max: 3.747 / 9.655 / 15.729 mm
sampled expert Elite joint step L-inf median/p95/max: 0.0117 / 0.0157 / 0.0170
```

Episode quality:

```text
left steps median/p95/max: 164.5 / 201.55 / 212
right steps median/p95/max: 164.5 / 205.20 / 209

left min_tip_distance_to_wall min/median: 1.386 / 1.471 mm
right min_tip_distance_to_wall min/median: 2.122 / 2.200 mm

max_contact_strength max: 0.0 left, 0.0 right
contact_p95 max: 0.0 left, 0.0 right
stall_fraction max: 0.0
slow_fraction max: 0.0
```

Magnet/tip alignment:

```text
left sample tip_to_magnetic median/p95/max: 3.40 / 6.39 / 7.17 mm
right sample tip_to_magnetic median/p95/max: 4.14 / 9.97 / 13.25 mm

left episode tip_to_magnetic_p95 median/max: 5.50 / 6.42 mm
right episode tip_to_magnetic_p95 median/max: 8.57 / 11.08 mm
```

Action quality:

```text
Piper feed values: mostly 0.7013 with scheduled -1.0 retract phases
Piper negative fraction: about 46% on both branches

left Elite joint step L-inf median/p95/max: 0.0123 / 0.0150 / 0.0156
right Elite joint step L-inf median/p95/max: 0.0118 / 0.0163 / 0.0170
```

Interpretation: the dataset is internally clean. It has no contact, positive
wall clearance, complete image references, smooth Elite labels, and low
magnet/tip separation. It is a suitable formal expert-data reference for the
current tip-centric route-plan abstraction.

## Formal-Validity Evidence

The current audit tool classifies the dataset as:

```text
status: formal_valid
formal_allowed: true
```

Embedded manifest reason:

```text
Route-plan expert avoids exact tip position and exact wall/contact feedback. It
uses a pre-registered vessel route, scheduled progress, Piper insertion state,
and Elite IK as a formal-validity probe.
```

Mechanism audit:

| Mechanism | Current Use | Real-System Plausibility | Judgment |
| --- | --- | --- | --- |
| Real vessel route | Registered route plan from vessel geometry | Plausible if pre-op geometry and registration are available | Acceptable assumption |
| Scheduled route progress | Open-loop route progress schedule | Plausible if tied to Piper feed/robot execution, but needs validation | Needs real validation |
| Piper insertion state | Feed/retract schedule and insertion window | Plausible through robot/actuator state | Acceptable assumption |
| Elite IK | Places tool near planned route point | Plausible with calibrated Elite-to-vessel transform | Acceptable assumption |
| Exact simulated tip feedback | Not used by route-plan expert | Not required | Pass |
| Exact wall/contact feedback | Not used by route-plan expert | Not required | Pass |
| Rollout helper anchors | Not part of dataset collection | Diagnostic only | Keep out of formal data |

## BC Rollout Interpretation

The BC line should now be frozen as a diagnostic baseline, not treated as the
main optimization target.

Observed BC behavior:

```text
absolute BC: numerical success, but sustained contact and large tip_to_magnetic
Elite-delta BC: improved open-loop labels, failed closed-loop without anchor
routeanchor0010: restores success and smooths Elite commands, but contact and
                 centimeter-scale tip_to_magnetic remain
```

The route-plan anchor phase diagnostic explains why:

```text
expert dataset plan_minus_actual median/p95:
  left  -2.567 / -0.428
  right -2.467 / -0.428

anchored BC rollout plan_minus_actual median/p95:
  left   4.808 / 11.640
  right  3.354 / 10.987

anchored BC rollout corr(plan_lead, contact):
  left  0.820
  right 0.866
```

In expert data, scheduled plan progress is slightly behind actual tip progress
at sampled states. In BC rollout, scheduled plan progress runs ahead of actual
tip progress, and contact rises with that positive lead. This is a closed-loop
policy/control phase problem.

Conclusion: do not keep shrinking anchor windows, changing BC losses, or
rendering rollout videos as the default next step. Those changes tune the small
model around a clean expert, rather than validating the simulator as a data
source.

## Remaining Risks

The current dataset is clean, but the following risks remain before treating
simulation as a large-scale replacement for real collection:

- Tip-centric dynamics are a simplification. They are plausible because the real
  guidewire is relatively hard and the head contact is the main observed safety
  event, but they still need visual and task-level real validation.
- The route-plan expert assumes a known route and calibrated robot-vessel
  geometry. This needs to be explicitly matched to the real lab workflow.
- Piper semantics remain mismatched: senior real data uses binary stop/advance
  labels (`0` stop/hold, `1` step_forward), while simulation uses continuous
  normalized feed/retract `piper_feed` with negative retract phases. The
  desired real interface may need a third retract action or signed step command,
  not a forced downgrade to the senior two-action interface.
- Camera appearance, lighting, background, and image domain are not yet
  validated against real side/top camera data.
- The current dataset covers 40 accepted episodes from start fractions
  `~0.58-0.70`; the stress-wide dataset has expanded start coverage to about
  `~0.50-0.82`, but route perturbations, scene variation, camera/domain
  variation, and Piper command semantics are still not validated against the
  real workflow.

## Go / No-Go

```text
Go for using this dataset as the current formal expert-data reference: yes.
Go for scaling expert-only data checks before more BC tuning: yes.
Go for declaring the simulator fully real-aligned: no.
Go for continuing BC rollout tuning as the main path: no.
```

## Recommended Next Step

Do not continue BC optimization as the next default task. Instead:

1. Treat the stress-wide formal route-plan dataset as the current expert-data
   stress-test reference.
2. Implement or test a real-implementable Piper action path before treating
   formal data as real-system-ready imitation data. At minimum it must preserve
   the senior semantics `0=stop/hold`, `1=advance one Piper step`; if retract is
   needed, define it explicitly as a third action or signed step rather than an
   undocumented continuous simulator-only value.
3. Refine time-base and calibration assumptions for comparing simulated
   Elite/magnetic-arm motion against real `branchs` pose samples.
4. Expand coverage only through real-plausible perturbations such as calibrated
   route/scene offsets, camera/domain variation, or Piper schedule variants.
5. Only return to model training after the expert-data and sim-to-real checks
   remain clean under these broader conditions.
