"""Pure saved-log arithmetic for the fixed visual-normalization diagnostic.

No filesystem, NumPy, torch, fitting, forward pass or optimizer is imported.
Sampled-batch training logs cannot locate saturation or establish its cause.
"""
from __future__ import annotations

import math
import re
import statistics


SEEDS = (20260912, 20260913, 20260914)
ARMS = ("observed_action", "normalized_zero_action")
VARIANTS = ("baseline", "train_visual_normalized")
STEP_RANGES = ((1, 20), (21, 50), (51, 100), (101, 150), (151, 200))
EPISODE_WINDOWS = {"1458": 124, "1476": 119, "1530": 120, "1566": 137}
PARAMETER_SHAPES = {
    "view_projections.0.weight": [128, 2048], "view_projections.0.bias": [128],
    "view_projections.1.weight": [128, 2048], "view_projections.1.bias": [128],
    "history_projection.weight": [128, 274], "history_projection.bias": [128],
    "context_gru.weight_ih_l0": [384, 128], "context_gru.weight_hh_l0": [384, 128],
    "context_gru.bias_ih_l0": [384], "context_gru.bias_hh_l0": [384],
    "task_embedding.weight": [1, 128], "context_task_projection.weight": [128, 256],
    "context_task_projection.bias": [128], "action_projection.weight": [128, 7], "action_projection.bias": [128],
    "action_gru.weight_ih_l0": [384, 128], "action_gru.weight_hh_l0": [384, 128],
    "action_gru.bias_ih_l0": [384], "action_gru.bias_hh_l0": [384],
    "visual_residual_head.weight": [4096, 128], "visual_residual_head.bias": [4096],
    "state_residual_head.weight": [8, 128], "state_residual_head.bias": [8],
}
LOGGED = ("loss", "visual_loss", "state_loss", "global_grad_norm_before_clip", "global_grad_norm_after_clip")
DETAIL_KEYS = frozenset({"step", "loss", "visual_loss", "state_loss", "visual_sum", "state_sum", "visual_count", "state_count",
    "global_grad_norm_before_clip", "global_grad_norm_after_clip", "parameters_with_grad", "parameter_changed",
    "parameter_change_check", "arm", "action_projection_weight_exact_zero_grad", "per_parameter_gradients_after_clip",
    "individual_zero_gradients_allowed", "input_target_bytes_versions_unchanged"})
INPUT_KEYS = {"history_visual_latent", "history_visual_valid", "history_state", "history_state_valid", "candidate_actions", "task_instruction"}
TARGET_KEYS = {"future_visual_latent", "future_visual_valid", "state_delta", "state_target_valid"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
             f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _integer(value, expected, name):
    _require(type(value) is int and value == expected, f"{name} differs")


def _close(left, right, name, *, logged=False):
    _require(math.isfinite(left) and math.isfinite(right)
             and math.isclose(left, right, rel_tol=2e-6 if logged else 1e-10, abs_tol=1e-9 if logged else 1e-12),
             f"{name} reconstruction differs")


def _range(values):
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)}


def _fingerprints(value, keys, name):
    _require(type(value) is dict and set(value) == keys, f"{name} keys differ")
    for key, digest in value.items():
        _require(type(digest) is str and bool(digest), f"{name}.{key} must be text")
        if key != "task_instruction":
            _require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"{name}.{key} SHA differs")


def _training_row(row, expected):
    _require(type(row) is dict and set(row) == {"seed", "arm", "variant", "step", "window_indices", "input_sha256", "target_sha256", "details"},
             "training trace row keys differ")
    seed, arm, variant, step = expected
    _integer(row["seed"], seed, "trace seed/order")
    _integer(row["step"], step, "trace step/order")
    _require(row["arm"] == arm and row["variant"] == variant, "trace arm/variant/order differs")
    indices = row["window_indices"]
    _require(type(indices) is list and len(indices) == 16
             and all(type(v) is int and 0 <= v < 1086 for v in indices), "fixed16 train window draws required")
    _fingerprints(row["input_sha256"], INPUT_KEYS, "input fingerprints")
    _fingerprints(row["target_sha256"], TARGET_KEYS, "target fingerprints")
    d = row["details"]
    _require(type(d) is dict and set(d) == DETAIL_KEYS, "original native training details keys differ")
    _integer(d["step"], step, "native step")
    _integer(d["parameters_with_grad"], 23, "native parameter gradient count")
    _integer(d["visual_count"], 16 * 3 * 2 * 2048, "training visual support")
    _integer(d["state_count"], 16 * 3 * 8, "training state support")
    _require(d["arm"] == arm and d["parameter_changed"] is True and d["individual_zero_gradients_allowed"] is True
             and d["input_target_bytes_versions_unchanged"] is True
             and d["parameter_change_check"] == "per-parameter float64 sum and squared sum; runner separately verifies final full SHA256",
             "original native training guards differ")
    for key in (*LOGGED, "visual_sum", "state_sum"):
        _number(d[key], key, positive=key.startswith("global_"))
    _close(d["visual_loss"], d["visual_sum"] / d["visual_count"], "logged visual loss", logged=True)
    _close(d["state_loss"], d["state_sum"] / d["state_count"], "logged state loss", logged=True)
    _close(d["loss"], d["visual_loss"] + .25 * d["state_loss"], "logged total loss", logged=True)
    _require(d["global_grad_norm_after_clip"] <= 1 + 1e-6
             and d["global_grad_norm_after_clip"] <= d["global_grad_norm_before_clip"] * (1 + 1e-6), "fixed clipping norm differs")
    gradients = d["per_parameter_gradients_after_clip"]
    _require(type(gradients) is list and len(gradients) == 23, "all23 native gradients required")
    for gradient, (name, shape) in zip(gradients, PARAMETER_SHAPES.items()):
        _require(type(gradient) is dict and set(gradient) == {"name", "shape", "l2_norm", "finite", "exact_zero"}
                 and gradient["name"] == name and type(gradient["shape"]) is list and gradient["shape"] == shape
                 and all(type(v) is int for v in gradient["shape"]) and gradient["finite"] is True,
                 "native gradient name/order/shape/finite guard differs")
        norm = _number(gradient["l2_norm"], "after-clip parameter norm")
        _require(type(gradient["exact_zero"]) is bool and gradient["exact_zero"] == (norm == 0), "gradient exact-zero evidence differs")
    reconstructed = math.sqrt(math.fsum(g["l2_norm"] ** 2 for g in gradients))
    _close(reconstructed, d["global_grad_norm_after_clip"], "global norm from23 after-clip parameter norms")
    zero = gradients[list(PARAMETER_SHAPES).index("action_projection.weight")]["exact_zero"]
    _require(type(d["action_projection_weight_exact_zero_grad"]) is bool
             and d["action_projection_weight_exact_zero_grad"] == zero
             and (arm != "normalized_zero_action" or zero), "normalized-zero arm weight gradient contract differs")
    return d


def summarize_training_trace(rows):
    """Summarize existing update-before-current-batch logs; never a new curve."""
    _require(type(rows) is list and len(rows) == 2400, "exactly2400 fixed training rows required")
    runs, paired = {}, {}
    cursor = 0
    for seed in SEEDS:
        runs[str(seed)] = {}
        for arm in ARMS:
            runs[str(seed)][arm] = {}
            for variant in VARIANTS:
                values = []
                for step in range(1, 201):
                    row = rows[cursor]
                    cursor += 1
                    values.append(_training_row(row, (seed, arm, variant, step)))
                    common = {key: row[key] for key in ("window_indices", "input_sha256", "target_sha256")}
                    pair = (seed, arm, step)
                    if variant == "baseline":
                        paired[pair] = common
                    else:
                        _require(common == paired[pair], "variants must reuse exact same draws, raw inputs and targets")
                    if arm == ARMS[1]:
                        other = paired[(seed, ARMS[0], step)]
                        _require(common["window_indices"] == other["window_indices"] and common["target_sha256"] == other["target_sha256"],
                                 "action arms must share the same draw list and targets")
                        _require(all(common["input_sha256"][key] == other["input_sha256"][key]
                                     for key in INPUT_KEYS - {"candidate_actions"}), "action arms changed non-action inputs")
                intervals = []
                for first, last in STEP_RANGES:
                    part = values[first - 1:last]
                    gradients = {name: {"shape": list(shape), "count": len(part),
                        **_range([d["per_parameter_gradients_after_clip"][i]["l2_norm"] for d in part]),
                        "exact_zero_steps": sum(d["per_parameter_gradients_after_clip"][i]["exact_zero"] for d in part)}
                        for i, (name, shape) in enumerate(PARAMETER_SHAPES.items())}
                    intervals.append(dict(first_step=first, last_step=last, step_count=len(part),
                        logged={key: _range([d[key] for d in part]) for key in LOGGED},
                        after_clip_parameter_gradients=gradients))
                runs[str(seed)][arm][variant] = {"total_steps": 200, "intervals": intervals}
    return dict(schema="libero_normalization_training_trace_diagnosis_v1", rows=cursor, seeds=list(SEEDS),
        arms=list(ARMS), variants=list(VARIANTS), step_ranges=[list(v) for v in STEP_RANGES], runs=runs,
        semantics=dict(logged_losses="pre-update current sampled training batch; changing batch, not fixed train/validation learning curve",
            logged_gradients="same pre-update backward gradients, parameter norms after global clipping",
            optimizer_step_deltas_available=False, instantaneous_tanh_available=False,
            saturation_time_identifiable=False, small_gradient_proves_saturation_or_cause=False,
            unequal_interval_lengths_preserved=True, new_forward_calls=0, new_optimizer_steps=0))


def _stat(value, count, name):
    _require(type(value) is dict and set(value) == {"count", "absolute_error_sum", "squared_error_sum", "mae", "rmse"},
             f"{name} original metric keys differ")
    _integer(value["count"], count, name + " count")
    for key in ("absolute_error_sum", "squared_error_sum", "mae", "rmse"):
        _number(value[key], name + " " + key)
    _close(value["mae"], value["absolute_error_sum"] / count, name + " MAE")
    _close(value["rmse"], math.sqrt(value["squared_error_sum"] / count), name + " RMSE")
    return {"mae": value["mae"], "rmse": value["rmse"]}


def _sum_ties(parent, children, name):
    for key in ("count", "absolute_error_sum", "squared_error_sum"):
        _close(parent[key], math.fsum(c[key] for c in children), name + " " + key)


def _score(score, windows):
    _require(type(score) is dict and score.get("schema") == "pi05_libero_native_world_model_metrics_v1", "saved native metric schema differs")
    _integer(score["window_count"], windows, "complete saved episode windows")
    _integer(score["horizon"], 3, "native horizon")
    _integer(score["visual_dim"], 2048, "native visual dimension")
    visual, state = score["visual"], score["state_normalized"]
    _require(type(visual) is dict and set(visual) == {"aggregate", "per_view", "per_horizon", "per_horizon_view"}
             and type(visual["per_view"]) is list and len(visual["per_view"]) == 2
             and type(visual["per_horizon"]) is list and len(visual["per_horizon"]) == 3
             and type(visual["per_horizon_view"]) is list and len(visual["per_horizon_view"]) == 3
             and all(type(v) is list and len(v) == 2 for v in visual["per_horizon_view"]), "saved visual stratification differs")
    vout = {"aggregate": _stat(visual["aggregate"], windows * 3 * 2 * 2048, "visual aggregate"),
        "per_view": [_stat(v, windows * 3 * 2048, "visual view") for v in visual["per_view"]],
        "per_horizon": [_stat(v, windows * 2 * 2048, "visual horizon") for v in visual["per_horizon"]],
        "per_horizon_view": [[_stat(v, windows * 2048, "visual horizon/view") for v in horizon] for horizon in visual["per_horizon_view"]]}
    _sum_ties(visual["aggregate"], visual["per_view"], "visual aggregate/views")
    _sum_ties(visual["aggregate"], visual["per_horizon"], "visual aggregate/horizons")
    for h in range(3):
        _sum_ties(visual["per_horizon"][h], visual["per_horizon_view"][h], "visual horizon/views")
    for v in range(2):
        _sum_ties(visual["per_view"][v], [r[v] for r in visual["per_horizon_view"]], "visual view/horizons")
    _require(type(state) is dict and type(state.get("per_horizon")) is list and len(state["per_horizon"]) == 3,
             "saved normalized-state horizon decomposition differs")
    sout = {"aggregate": _stat(state["aggregate"], windows * 3 * 8, "state aggregate"),
            "per_horizon": [_stat(v, windows * 8, "state horizon") for v in state["per_horizon"]]}
    _sum_ties(state["aggregate"], state["per_horizon"], "state aggregate/horizons")
    objective = score["masked_objective"]
    _require(type(objective) is dict and set(objective) == {"visual_sum", "state_sum", "visual_count", "state_count", "visual_loss", "state_loss", "total"},
             "saved objective decomposition differs")
    for key, count in (("visual", windows * 3 * 2 * 2048), ("state", windows * 3 * 8)):
        _integer(objective[key + "_count"], count, "objective support")
        _number(objective[key + "_sum"], "objective sum")
        _number(objective[key + "_loss"], "objective component")
        _close(objective[key + "_loss"], objective[key + "_sum"] / count, "objective component mean")
    _number(objective["total"], "objective total")
    _close(objective["total"], objective["visual_loss"] + .25 * objective["state_loss"], "saved weighted objective")
    return dict(objective=dict(visual_smooth_l1=objective["visual_loss"], state_smooth_l1=objective["state_loss"],
        weighted_state_smooth_l1=.25 * objective["state_loss"], total=objective["total"]), visual=vout, state_normalized=sout)


def _combine(values, operation):
    """Apply arithmetic to identical numeric trees; input metadata never enters."""
    first = values[0]
    if type(first) is dict:
        _require(all(type(v) is dict and set(v) == set(first) for v in values), "summary tree keys differ")
        return {key: _combine([v[key] for v in values], operation) for key in first}
    if type(first) is list:
        _require(all(type(v) is list and len(v) == len(first) for v in values), "summary tree lengths differ")
        return [_combine([v[i] for v in values], operation) for i in range(len(first))]
    result = operation(values)
    _require(math.isfinite(result), "summary arithmetic overflow")
    return result


def _paired(baseline, normalized):
    delta = _combine([baseline, normalized], lambda v: v[1] - v[0])
    o = delta["objective"]
    _close(o["weighted_state_smooth_l1"], .25 * o["state_smooth_l1"], "paired state objective contribution")
    _close(o["total"], o["visual_smooth_l1"] + o["weighted_state_smooth_l1"], "paired total objective delta")
    return {"baseline": baseline, "train_visual_normalized": normalized, "delta": delta}


def summarize_saved_errors(report):
    """Read saved final200 CLEAN per-episode scores, not raw predictions.

    MAE/RMSE strata and aggregate SmoothL1 objectives are different quantities.
    SmoothL1 per-view/horizon cannot be recovered from saved MAE/RMSE alone.
    """
    _require(type(report) is dict and report.get("schema") == "libero_visual_normalization_result_v1"
             and report.get("status") == "completed_fixed_budget_normalization_comparison", "complete normalization result required")
    _integer(report["optimizer_steps"], 2400, "completed training budget")
    source = report["runs"]
    _require(type(source) is dict and set(source) == set(map(str, SEEDS)), "all3 saved seeds required")
    runs = {}
    for seed in SEEDS:
        key = str(seed)
        _require(type(source[key]) is dict and set(source[key]) == set(ARMS), "both saved action arms required")
        runs[key] = {}
        for arm in ARMS:
            variants = source[key][arm]
            _require(type(variants) is dict and set(variants) == set(VARIANTS), "both saved normalization variants required")
            scores = {}
            for variant in VARIANTS:
                final = variants[variant]["final"]
                _integer(final["windows"], 500, "complete clean validation windows")
                episodes = final["conditions"]["clean"]["per_episode"]
                _require(type(episodes) is dict and set(episodes) == set(EPISODE_WINDOWS), "exact4 saved validation episodes required")
                scores[variant] = {eid: _score(episodes[eid], count) for eid, count in EPISODE_WINDOWS.items()}
            paired = {eid: _paired(scores[VARIANTS[0]][eid], scores[VARIANTS[1]][eid]) for eid in EPISODE_WINDOWS}
            macro = {v: _combine(list(scores[v].values()), statistics.mean) for v in VARIANTS}
            runs[key][arm] = {"per_episode": paired, "episode_macro": _paired(macro[VARIANTS[0]], macro[VARIANTS[1]])}
    arms = {}
    for arm in ARMS:
        means = {v: _combine([runs[str(s)][arm]["episode_macro"][v] for s in SEEDS], statistics.mean) for v in VARIANTS}
        arms[arm] = dict(role="primary" if arm == ARMS[0] else "separate_zero_action_comparison",
            three_seed_mean=_paired(means[VARIANTS[0]], means[VARIANTS[1]]),
            paired_delta_population_std=_combine([runs[str(s)][arm]["episode_macro"]["delta"] for s in SEEDS], statistics.pstdev))
    return dict(schema="libero_normalization_saved_error_diagnosis_v1", runs=runs, arms=arms,
        semantics=dict(phase="final200", condition="clean", objective_weights={"visual": 1.0, "state": .25},
            difference="train_visual_normalized minus paired fresh baseline; absolute signed differences",
            aggregation="equal-episode means within each seed, then equal-seed means; macro RMSE is mean of episode RMSEs",
            visual_strata="MAE/RMSE only; view indices0/1 and horizon list indices0/1/2 (future steps1/2/3)",
            smooth_l1_stratification_available=False, smooth_l1_not_derivable_from_mae_rmse=True,
            validation_is_reused_development_set=True, causal_or_saturation_timing_claim=False,
            new_forward_calls=0, new_optimizer_steps=0))
