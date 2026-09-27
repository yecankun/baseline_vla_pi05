# Frozen Gated Adapter: Precision and History Dependence Audit

Predeclared 2026-09-21, before inference. No optimization, architecture change,
checkpoint selection, sensor/label change, or hardware execution.

## Frozen inputs and comparisons

Load `real10_pi05_gated_action_v1/gated_action_adapter.pt` at its fixed step100,
with the same original PI05/Piper pair. Freeze every parameter. Reuse the exact
30 training-panel contexts and per-context inference noise from that run.
Reproduce saved original and true-history predictions before interpreting any
counterfactual. Do not treat this panel as held-out: the base saw all10 episodes,
and each episode contributes an enriched 2 hold + 1 feed diagnostic sample set.

Keep the current image pair, current state, instruction, slot timestamps, valid
masks, noise and weights fixed. Modify **only past history content**, jointly
moving/replacing its visual features and normalized state:

1. `true_history`: the original [t-2, t-1, t] observations.
2. `repeat_current`: replace past content with copies of current content;
   retain original past-slot times and masks to isolate history content.
3. `swap_past`: swap the two past contents, keeping current content and all slot
   times unchanged. This is a deterministic order counterfactual, not new data.

All30 chosen contexts have two valid past slots. These interventions are offline
diagnostics, may be out of distribution, and cannot establish causal benefit or
success rates. No choice is deployed or selected using its MAE.

## Numerical measurement on the actual production hook

Wrap the existing state-residual hook transparently, capturing the original state
token s, requested float32 residual r, cast BF16 residual q, and actual updated
state token. Return the original hook output unchanged. Verify the captured
result exactly equals its native BF16 addition; preserve the original masks.

Report separately:

- r -> q cast relative L2 error and nonzero-coordinate fractions;
- effective residual e = float32(updated state) - float32(s);
- coordinate-change / swallowed-nonzero fractions, requested/effective RMS,
  effective/requested L2 ratio, relative L2 error, and directional cosine.

An RMS ratio is **not** a retained-information percentage: rounding can suppress
many coordinates while amplifying the ones that change. No FP32 policy variant
or gate rescaling is run. Save captured vectors so interpretation does not need
another model load.

## History-dependence measurements and decision boundary

For each counterfactual, report past-input changes, shared-token/residual relative
L2 changes, effective-token changes after rounding, active xyz action equality,
absolute action changes in mm and normalized units, and per-episode changes.
Report xyz MAE and fixed-time flow loss only as supporting train-panel diagnostics.
Piper probabilities must remain identical across all conditions.

Nonzero dependence is not useful dependence. Insensitivity can reflect weak past
variation, near-permutation-invariant encoding, quantization, or a small gate;
the layers measured here help distinguish those possibilities without claiming
an untested cause. Do not retrain, increase the gate, tune precision, or add world
loss after seeing the results. Any next implementation needs its own fixed scope.

Outputs are new read-only audit JSON / compact numerical arrays and Chinese
figures. A failed attempt stays preserved; since this is a short inference job,
recover by a new explicit output directory after verifying no old process is
running, never by overwriting a completed report or any source checkpoint.
