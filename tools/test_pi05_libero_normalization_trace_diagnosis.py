"""Synthetic saved-trace/report arithmetic only; no model, data or training."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import pi05_libero_normalization_trace_diagnosis as diagnosis


def fake_training_rows():
    rows = []
    for seed in diagnosis.SEEDS:
        for arm in diagnosis.ARMS:
            for variant in diagnosis.VARIANTS:
                for step in range(1, 201):
                    grads = []
                    for index, (name, shape) in enumerate(diagnosis.PARAMETER_SHAPES.items()):
                        value = .001 * (index + 1) / 23 * (1 + step / 200)
                        if arm == "normalized_zero_action" and name == "action_projection.weight":
                            value = 0.0
                        grads.append(dict(name=name, shape=list(shape), l2_norm=value, finite=True, exact_zero=value == 0))
                    norm = math.sqrt(math.fsum(v["l2_norm"] ** 2 for v in grads))
                    visual, state = .01 + step / 100000, .04 + step / 200000
                    if variant != "baseline":
                        visual += .001
                        state -= .004
                    details = dict(step=step, loss=visual + .25 * state, visual_loss=visual, state_loss=state,
                        visual_sum=visual * 196608, state_sum=state * 384, visual_count=196608, state_count=384,
                        global_grad_norm_before_clip=norm, global_grad_norm_after_clip=norm, parameters_with_grad=23,
                        parameter_changed=True, parameter_change_check="per-parameter float64 sum and squared sum; runner separately verifies final full SHA256",
                        arm=arm, action_projection_weight_exact_zero_grad=arm == "normalized_zero_action",
                        per_parameter_gradients_after_clip=grads, individual_zero_gradients_allowed=True,
                        input_target_bytes_versions_unchanged=True)
                    rows.append(dict(seed=seed, arm=arm, variant=variant, step=step, window_indices=[(step + i) % 1086 for i in range(16)],
                        input_sha256={key: ('["synthetic"]' if key == "task_instruction" else "a" * 64) for key in diagnosis.INPUT_KEYS},
                        target_sha256={key: "b" * 64 for key in diagnosis.TARGET_KEYS}, details=details))
    return rows


def metric(count, mae, variance=.002):
    return dict(count=count, absolute_error_sum=count * mae, squared_error_sum=count * (mae * mae + variance),
                mae=mae, rmse=math.sqrt(mae * mae + variance))


def aggregate(rows):
    count = sum(v["count"] for v in rows)
    absolute = math.fsum(v["absolute_error_sum"] for v in rows)
    square = math.fsum(v["squared_error_sum"] for v in rows)
    return dict(count=count, absolute_error_sum=absolute, squared_error_sum=square, mae=absolute / count, rmse=math.sqrt(square / count))


def fake_score(windows, seed_offset, variant_offset, episode_offset):
    cells = [[metric(windows * 2048, .07 + seed_offset + episode_offset + h * .005 + v * .002 + variant_offset)
              for v in range(2)] for h in range(3)]
    horizons = [aggregate(row) for row in cells]
    views = [aggregate([row[v] for row in cells]) for v in range(2)]
    visual = aggregate(horizons)
    states = [metric(windows * 8, .05 + seed_offset + episode_offset + h * .002 - 2 * variant_offset) for h in range(3)]
    state = aggregate(states)
    vloss = .5 * visual["squared_error_sum"] / visual["count"]
    sloss = .5 * state["squared_error_sum"] / state["count"]
    return dict(schema="pi05_libero_native_world_model_metrics_v1", window_count=windows, horizon=3, visual_dim=2048,
        visual=dict(aggregate=visual, per_view=views, per_horizon=horizons, per_horizon_view=cells),
        state_normalized=dict(aggregate=state, per_horizon=states),
        masked_objective=dict(visual_count=visual["count"], state_count=state["count"],
            visual_sum=vloss * visual["count"], state_sum=sloss * state["count"], visual_loss=vloss, state_loss=sloss, total=vloss + .25 * sloss))


def fake_report():
    runs = {}
    for si, seed in enumerate(diagnosis.SEEDS):
        runs[str(seed)] = {}
        for ai, arm in enumerate(diagnosis.ARMS):
            runs[str(seed)][arm] = {}
            for variant in diagnosis.VARIANTS:
                offset = 0 if variant == "baseline" else .001 * (si + 1)
                episodes = {eid: fake_score(count, si * .001 + ai * .002, offset, ei * .003)
                            for ei, (eid, count) in enumerate(diagnosis.EPISODE_WINDOWS.items())}
                runs[str(seed)][arm][variant] = {"final": {"windows": 500, "conditions": {"clean": {"per_episode": episodes}}}}
    return dict(schema="libero_visual_normalization_result_v1", status="completed_fixed_budget_normalization_comparison",
                optimizer_steps=2400, runs=runs)


class SavedTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fake_training_rows()
        cls.report = fake_report()

    def test_fixed_complete_ordered_training_ranges_and_means(self):
        summary = diagnosis.summarize_training_trace(self.rows)
        self.assertEqual(summary["rows"], 2400)
        self.assertEqual(summary["step_ranges"], [[1, 20], [21, 50], [51, 100], [101, 150], [151, 200]])
        intervals = summary["runs"]["20260912"]["observed_action"]["baseline"]["intervals"]
        self.assertEqual([row["step_count"] for row in intervals], [20, 30, 50, 50, 50])
        for row in intervals:
            source = self.rows[row["first_step"] - 1:row["last_step"]]
            self.assertEqual(row["logged"]["loss"]["mean"], statistics.mean(v["details"]["loss"] for v in source))
            self.assertEqual(len(row["after_clip_parameter_gradients"]), 23)
        json.dumps(summary, allow_nan=False)

    def test_normalized_zero_weight_grad_is_zero_without_invalidating_run(self):
        result = diagnosis.summarize_training_trace(self.rows)
        for seed in diagnosis.SEEDS:
            for variant in diagnosis.VARIANTS:
                for row in result["runs"][str(seed)]["normalized_zero_action"][variant]["intervals"]:
                    item = row["after_clip_parameter_gradients"]["action_projection.weight"]
                    self.assertEqual(item["exact_zero_steps"], row["step_count"])
                    self.assertEqual(item["min"], 0)
                    self.assertEqual(item["max"], 0)
                    self.assertEqual(item["mean"], 0)

    def test_training_rejects_missing_extra_duplicate_reordered_rows(self):
        for rows in (self.rows[:-1], self.rows + self.rows[:1], tuple(self.rows)):
            with self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)
        for first, second in ((0, 1), (0, 200), (0, 800)):
            rows = list(self.rows)
            rows[first], rows[second] = rows[second], rows[first]
            with self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)
        rows = list(self.rows)
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            diagnosis.summarize_training_trace(rows)

    def test_training_rejects_gradient_names_shapes_count_and_finite_flags(self):
        for change in ("name", "order", "shape", "count", "nonfinite", "finite_flag", "zero_flag"):
            rows = list(self.rows)
            rows[0] = deepcopy(rows[0])
            d = rows[0]["details"]
            g = d["per_parameter_gradients_after_clip"]
            if change == "name": g[0]["name"] = "visual_mean"
            elif change == "order": g[0], g[1] = g[1], g[0]
            elif change == "shape": g[0]["shape"] = [4, 2048]
            elif change == "count": d["parameters_with_grad"] = 22
            elif change == "nonfinite": g[0]["l2_norm"] = float("nan")
            elif change == "finite_flag": g[0]["finite"] = False
            else: g[0]["exact_zero"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)

    def test_training_rejects_unknown_fields_and_invalid_scalar_types(self):
        for key, value in (("seed", True), ("step", 1.0), ("variant", "unknown"), ("oracle", 0)):
            rows = list(self.rows)
            rows[0] = {**rows[0], key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)
        for key, value in (("loss", True), ("loss", float("inf")), ("visual_sum", -1), ("global_grad_norm_before_clip", 0)):
            rows = list(self.rows)
            rows[0] = deepcopy(rows[0])
            rows[0]["details"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)

    def test_training_checks_logged_loss_and_global_norm_reconstructions(self):
        for key in ("loss", "visual_loss", "state_loss", "global_grad_norm_after_clip"):
            rows = list(self.rows)
            rows[0] = deepcopy(rows[0])
            rows[0]["details"][key] *= 1.2
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)

    def test_training_shared_draw_and_target_and_variant_input_checks(self):
        for key in ("window_indices", "input_sha256", "target_sha256"):
            rows = list(self.rows)
            rows[200] = deepcopy(rows[200])
            if key == "window_indices": rows[200][key][0] += 1
            else: rows[200][key][next(k for k in rows[200][key] if k != "task_instruction")] = "f" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnosis.summarize_training_trace(rows)

    def test_training_cross_arm_only_candidate_action_fingerprint_may_change(self):
        rows = list(self.rows)
        for index in range(400, 800):
            rows[index] = {**rows[index], "input_sha256": {**rows[index]["input_sha256"], "candidate_actions": "c" * 64}}
        diagnosis.summarize_training_trace(rows)
        rows[400] = {**rows[400], "input_sha256": {**rows[400]["input_sha256"], "history_state": "d" * 64}}
        with self.assertRaisesRegex(ValueError, "non-action"):
            diagnosis.summarize_training_trace(rows)

    def test_saved_clean_error_strata_and_objective_delta_reconstruction(self):
        result = diagnosis.summarize_saved_errors(self.report)
        self.assertEqual(result["arms"]["observed_action"]["role"], "primary")
        self.assertEqual(result["arms"]["normalized_zero_action"]["role"], "separate_zero_action_comparison")
        row = result["runs"]["20260912"]["observed_action"]["per_episode"]["1458"]
        self.assertEqual(len(row["baseline"]["visual"]["per_horizon_view"]), 3)
        self.assertEqual(len(row["baseline"]["visual"]["per_view"]), 2)
        o = row["delta"]["objective"]
        self.assertAlmostEqual(o["total"], o["visual_smooth_l1"] + .25 * o["state_smooth_l1"], places=15)
        self.assertAlmostEqual(o["weighted_state_smooth_l1"], .25 * o["state_smooth_l1"], places=15)
        self.assertNotIn("smooth_l1", row["baseline"]["visual"]["per_view"][0])
        json.dumps(result, allow_nan=False)

    def test_saved_episode_then_seed_equal_means_not_window_weighted(self):
        result = diagnosis.summarize_saved_errors(self.report)
        rows = result["runs"]["20260912"]["observed_action"]["per_episode"]
        expected = statistics.mean(v["baseline"]["visual"]["aggregate"]["mae"] for v in rows.values())
        actual = result["runs"]["20260912"]["observed_action"]["episode_macro"]["baseline"]["visual"]["aggregate"]["mae"]
        self.assertEqual(actual, expected)
        weighted = sum(rows[e]["baseline"]["visual"]["aggregate"]["mae"] * n for e, n in diagnosis.EPISODE_WINDOWS.items()) / 500
        self.assertNotEqual(actual, weighted)
        seed_deltas = [result["runs"][str(s)]["observed_action"]["episode_macro"]["delta"]["visual"]["aggregate"]["mae"] for s in diagnosis.SEEDS]
        actual_std = result["arms"]["observed_action"]["paired_delta_population_std"]["visual"]["aggregate"]["mae"]
        self.assertEqual(actual_std, statistics.pstdev(seed_deltas))

    def test_saved_rejects_wrong_source_coverage(self):
        for kind in ("schema", "status", "budget", "seed", "arm", "variant", "episode", "windows"):
            report = deepcopy(self.report)
            variants = report["runs"]["20260912"]["observed_action"]
            clean = variants["baseline"]["final"]
            if kind == "schema": report["schema"] = "different"
            elif kind == "status": report["status"] = "partial"
            elif kind == "budget": report["optimizer_steps"] = 2399
            elif kind == "seed": report["runs"].pop("20260914")
            elif kind == "arm": report["runs"]["20260912"].pop("normalized_zero_action")
            elif kind == "variant": variants.pop("train_visual_normalized")
            elif kind == "episode": clean["conditions"]["clean"]["per_episode"].pop("1458")
            else: clean["windows"] = 499
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                diagnosis.summarize_saved_errors(report)

    def test_saved_rejects_objective_mae_rmse_and_stratification_corruption(self):
        for kind in ("objective", "weight", "mae", "rmse", "stratum", "count", "nonfinite"):
            report = deepcopy(self.report)
            score = report["runs"]["20260912"]["observed_action"]["baseline"]["final"]["conditions"]["clean"]["per_episode"]["1458"]
            if kind == "objective": score["masked_objective"]["visual_loss"] += .01
            elif kind == "weight": score["masked_objective"]["total"] += .01
            elif kind == "mae": score["visual"]["aggregate"]["mae"] += .01
            elif kind == "rmse": score["state_normalized"]["aggregate"]["rmse"] += .01
            elif kind == "stratum":
                value = score["visual"]["per_horizon_view"][0][0]
                value["absolute_error_sum"] *= 2
                value["mae"] *= 2
            elif kind == "count": score["visual"]["per_view"][0]["count"] -= 1
            else: score["masked_objective"]["state_loss"] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                diagnosis.summarize_saved_errors(report)

    def test_saved_does_not_read_intervention_or_initial_scores(self):
        report = deepcopy(self.report)
        for seed in report["runs"].values():
            for arm in seed.values():
                for variant in arm.values():
                    variant["initial"] = "not in approved saved-clean analysis"
                    variant["final"]["conditions"]["fixed_train_anchor"] = "not inspected"
        self.assertEqual(diagnosis.summarize_saved_errors(report), diagnosis.summarize_saved_errors(self.report))

    def test_pure_owned_summary_no_file_io_and_no_causal_timing_claim(self):
        before_rows, before_report = deepcopy(self.rows), deepcopy(self.report)
        with patch("builtins.open", side_effect=AssertionError("disk read forbidden")):
            trace = diagnosis.summarize_training_trace(self.rows)
            errors = diagnosis.summarize_saved_errors(self.report)
        self.assertEqual(self.rows, before_rows)
        self.assertEqual(self.report, before_report)
        self.assertFalse(trace["semantics"]["saturation_time_identifiable"])
        self.assertFalse(trace["semantics"]["instantaneous_tanh_available"])
        self.assertFalse(trace["semantics"]["optimizer_step_deltas_available"])
        self.assertFalse(errors["semantics"]["smooth_l1_stratification_available"])
        trace["runs"]["20260912"]["observed_action"]["baseline"]["intervals"][0]["after_clip_parameter_gradients"]["view_projections.0.weight"]["shape"][0] = -1
        self.assertEqual(diagnosis.PARAMETER_SHAPES["view_projections.0.weight"], [128, 2048])


if __name__ == "__main__":
    unittest.main()
