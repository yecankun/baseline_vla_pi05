# Real10 Joint Temporal Prefix: Prospective Short Comparison

Locked before execution, 2026-09-21. Algorithm-only engineering experiment.
This protocol does not authorize hardware execution or replace deployment weights.

## Question and scope

Does next-record latent prediction provide a usable training signal to the shared
temporal prefix, beyond the same prefix trained with action loss alone?
Only the existing temporal-prefix and world-prediction modules are trainable.
Original PI05 and the Piper state_only head remain frozen. No LoRA, new contact
input, image augmentation, ROI change, response labels, or simulation mixing.

Source: the complete `real10_pi05_compat_v1` index (2719 transitions, 10 episodes).
The existing base checkpoint was trained on all ten episodes. Every number below
is a **training diagnostic**, not held-out generalization or a success rate.
Future images are detached targets, never current policy inputs. Histories stay
within their source episode. Terminal images remain targets; no zero-action
terminal samples are synthesized. Elite conditioning is measured pose delta;
Piper conditioning is requested intent, not confirmed physical wire motion.

## Predeclared calibration

Select exactly three distinct contexts from each episode, all at step >= 2:
hold at floor(0.25*(n_hold-1)), hold at floor(0.75*(n_hold-1)), and feed at
floor(0.50*(n_feed-1)), in source-step order. Save all 30 sample IDs before fitting.
This deliberately feed-enriched panel is not an estimate of dataset prevalence.
It is reused for training diagnostics, not model selection.

Create a fresh adapter with seed 123; do not use the previously updated diagnostic
adapter. For each view, normalize frozen image features with the existing fixed
LayerNorm and compute the panel's mean squared next-record feature increment.
Freeze `s_view = max(mean_increment_MSE, 1e-6)` and use
`L_future_scaled = mean_view(MSE_view / s_view)`.
This changes loss units only, not the predicted target or network structure.

At that initialization, measure per-context L2 gradients over all shared temporal
parameters. Use a fixed flow time of 0.5 and seeded Gaussian noise per context.
Freeze `lambda = 0.1 / median(||grad L_future_scaled|| / ||grad L_action||)`.
The target 0.1 is fixed in advance. It refers only to the median **initial**
shared-gradient ratio on this panel/time, not every future update or native flow
time. Record raw/scaled/weighted ratios and gradient cosine similarity. No metric
search or post-result weight adjustment. Nonfinite or zero gradients stop the run.

## Matched optimization

- Primary comparison: identical temporal adapter with action-only vs action plus
  the calibrated future loss. A frozen original PI05 and the initialized adapter
  are reference conditions, not substitutes for the matched action-only arm.
- Both arms start from identical values in separate parameter storage. Same
  complete-index permutation, batch size 1, first 100 optimizer updates, seed 123.
  This is a short optimization budget, not an episode-truncated dataset.
- Same Gaussian flow noise and native PI05 sampled time at every paired step.
  The native frozen policy remains in eval mode; new modules have dropout 0.
- AdamW, learning rate 1e-4, weight decay 0. Shared temporal gradients and world
  gradients are clipped **separately** at L2 1.0, so extra world-head gradients
  cannot globally shrink the policy update. Same temporal clipping in both arms.
- Action-only has no world objective or world updates; unused world parameters
  are held fixed. This additional compute is reported, not hidden as equal FLOPs.
- Save complete paired checkpoints every 25 updates and compare the fixed final
  update 100. No best-epoch or diagnostic-panel checkpoint selection. Recovery
  resumes both arms from the same completed paired update, with deterministic
  per-step draws; never overwrite existing completed results.

## Reporting and next decision

On the same 30 training-panel contexts, use identical seeded inference noise for
original PI05, initialized adapter, action-only, and joint. Report denormalized
Elite xyz MAE (mm), episode macro MAE, left/right and per-episode paired changes;
fixed-time action loss; raw next-record feature MSE by view and its ratio to
persistence; frozen Piper output invariance. Report realized training-order
coverage, calibration range, gradient norms, runtime and peak GPU memory.

Use actual saved reports for figures. No significance test or success-rate claim
from these 30 correlated training samples. One short seed cannot establish stable
benefit. Improving a largely static latent target alone is not evidence of useful
action effects, contact, or causal dynamics.

Interpretation is bounded: finite matched optimization establishes a runnable
experiment; a favorable training-panel difference is only a reason to pursue an
independent evaluation, not to deploy. An unfavorable difference is reported
without trying more seeds/weights against this panel. Generalization needs new
independent trajectories or a clean re-training split from an uncontaminated base.
The existing onsite checkpoints and shared action interface remain unchanged.
