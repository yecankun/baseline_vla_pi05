# VLA Training-Data Readiness Audit

Date: 2026-07-17

## Scope

This is a read-only algorithm-track audit of the experimental simulation data
currently used by the PI05/LeRobot baseline. It does not modify collectors,
labels, simulator experts, real-data procedures, or
`docs/data-track-handoff.md`.

Audited export:

```text
root: simulation_output/lerobot_project2026_frontup_step024_v1
repo-id: project2026/guidewire-openpi-compat-frontup-step024-v1
dataset size: 4936 records / 40 episodes
training slice: first 4096 records
split: 3480 train / 616 validation, episode-level, seed 123
```

Primary evidence:

- `simulation_output/pi05_training_data_audit_frontup_step024_v1.json`
- `simulation_output/pi05_training_data_audit_frontup_step024_v1.png`
- `simulation_output/openpi_tactile_signal_audit_frontup_step024_esttip_reggeom_v1.json`
- `docs/_formal_tip_line_frontup_step024_esttip_reggeom_dataset_v1_audit/observation_provenance_audit.md`
- `docs/algorithm-track-handoff.md`

## Decision

Yes, the current data can materially contribute to the weak training result.
There is no evidence of file corruption or a broken exported action schema,
but the dataset is not yet a strong basis for deciding whether PI05, state
conditioning, or tactile conditioning is effective.

Use this export for interface closure, checkpoint loading, short training
smokes, and controlled diagnostics. Do not use it as the primary evidence for
algorithm ranking, tactile benefit, three-class Piper control, or real-system
readiness.

The next full state-conditioned PI05 ablation is paused until the data and
evaluation gates below are addressed. A shape/gradient smoke may still be run
after implementation, but it must not be interpreted as a performance result.

## What Passed

- All audited state and action values are finite.
- `elite_tcp_delta_6d` exactly matches `action_32[0:6]`.
- `piper_intent_id` exactly matches the compatibility one-hot slice.
- `action_32[9:32]` is zero padding as specified.
- The validation episodes are separate from the training episodes.
- No validation image pair had an exact hash match in the training split.

These checks rule out basic serialization, schema, and exact image-duplicate
failure as the main explanation.

## Material Risks

### 1. Strong temporal redundancy

Elite translation lag-one correlations are `0.9194`, `0.8940`, and `0.8732`.
Consecutive Elite actions are exactly equal in `32.7%` of within-episode frame
pairs. A previous-action baseline reaches translation MAE `0.2487`, compared
with `0.7395` for the low-resolution image ridge baseline and `0.7846` for the
state ridge baseline.

Piper labels have median and P95 run length `10`, and only `9.6%` of adjacent
frames change class. Copying the previous Piper label reaches `90.5%` held-out
accuracy, compared with `52.9%` for the majority class, `64.1%` for state
ridge, and `56.7%` for image ridge.

The nominal frame count therefore overstates independent control diversity.
Frame-averaged metrics are dominated by continuing an existing primitive and
underweight the transition frames where a policy decision is required.

### 2. Degenerate and narrow action support

Elite rotation dimensions `3:6` are zero in every audited record. The earlier
six-dimensional PI05 aggregate improvement was consequently dominated by
learning these constant targets, while translation remained weak. Elite
translation dimension 1 is never negative, and each translation dimension is
zero in about one third of the audited records.

Piper contains only hold and feed (`1950/2146` in the 4096-record slice) and no
retract examples. This is compatible with the current minimum real interface,
but it cannot support a three-class control claim. The action coverage is a
narrow local behavior slice, not a general guidewire-control distribution.

### 3. No positive tactile/contact target

Across all 4936 exported records, `estimated_contact_flag` is always zero.
The varying confidence and image-distance fields have only weak linear
association with Piper intent and Elite motion. This export can verify that a
tactile field is wired through the model, but it cannot test whether contact
conditioning improves decisions around contact.

### 4. Teacher-student observation gap

The formal simulator expert uses estimated tip and registered-route geometry.
Those quantities are stored with estimator provenance, but they are not
explicitly represented in the current `state_32` layout. The student must
recover the relevant geometry from the images and the smaller state surface.

LeRobot `0.4.4` PI05 currently ignores `observation.state` in the diffusion
path, so the actual Elite policy receives only images and language. This is a
confirmed algorithm-interface limitation layered on top of the data issue; it
is not valid to attribute the full failure to either data or model alone.

### 5. Limited visual and split diversity

There are no exact image duplicates, but adjacent-frame dual-view pixel MAE is
only `0.00146` at the median and `0.0000124` at P05 on a `[0,1]` scale. The
contact sheet shows fixed cameras and backgrounds, small guidewire/contact
regions, and mostly gradual pose changes. The median validation-to-nearest-
training image-signature distance is `0.00570`, so the split is visually close
even without exact leakage.

The 4096-record cap also truncates episode 33 to 29 frames and excludes the
remaining full-dataset episodes. Training is left-task heavy (`2220/1260`),
while validation is right-task heavy (`249/367`). Both tasks are present, but
future comparisons should split whole episodes without a mid-episode cap.

## Failure Attribution

| Candidate cause | Current judgment |
| --- | --- |
| Corrupt export or mismatched action fields | Unlikely; schema checks pass |
| Temporal redundancy and primitive persistence | Strong evidence |
| Narrow or degenerate target coverage | Strong evidence |
| Missing useful contact examples | Confirmed |
| Teacher information not explicit at the student interface | Material risk |
| PI05 diffusion ignoring state | Confirmed algorithm limitation |
| PI05 architecture itself is unsuitable | Not established |

The current negative result is therefore a combined data/interface result,
not a clean PI05 architecture result.

## Readiness Gates

Before another full PI05 comparison, require:

1. A whole-episode train/validation split with both tasks represented and no
   partial episode created by `max_records`.
2. Separate metrics for Piper transition frames versus steady primitive frames,
   plus per-translation-dimension Elite metrics. Do not use six-dimensional
   aggregate MAE as the primary Elite score while rotation targets are constant.
3. An explicit cross-track decision on teacher-student observability: either
   expose accepted estimator-derived equivalents at the policy interface or
   define the experiment as image-only geometry recovery.
4. Positive and negative estimator-derived contact examples before claiming a
   tactile-conditioning ablation. Otherwise declare tactile unavailable for
   this dataset.
5. A temporal baseline alongside PI05. Report previous-action/previous-intent
   performance and evaluate transition windows separately so persistence is
   not mistaken for policy quality.

## Recommended Next Step

Keep model implementation paused and first add a no-relabel, no-collector-change
evaluation view over the current export:

- whole-episode sampling instead of the 4096 mid-episode cap;
- steady versus Piper-transition strata;
- frame-strided results to reduce adjacent-frame redundancy;
- zero/mean/previous-action/state/image baselines on the same records.

This is the cheapest controlled test of whether temporal redundancy is hiding
the learnable signal. In parallel, the data track can decide whether a revised
dataset candidate should expose accepted estimator fields and include
contact-rich examples. Only after those results should the algorithm track run
a full state-conditioned PI05 ablation.
