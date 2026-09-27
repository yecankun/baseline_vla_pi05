"""Synthetic runner checks; no public-data model loading or training."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_pi05_libero_visual_normalization as runner
from test_pi05_libero_action_study_step0 import TinyDataset, REGISTRY
from test_pi05_libero_visual_normalization import synthetic_pack


class NormalizationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        pack, split = synthetic_pack()
        self.norm = runner.fit_visual_statistics(pack, split)
        self.config = runner.LiberoWorldModelConfig(hidden_dim=4)
        self.dataset = TinyDataset()
        for index, item in enumerate(self.dataset.items):
            for frame in range(4):
                item["inputs"]["history_visual_latent"][frame].fill(index + frame + 1.)
        self.inputs = runner.collate_window_inputs([self.dataset[i]["inputs"] for i in range(2)])
        self.seed, self.arm = runner.base.SEEDS[0], runner.base.ARMS[0]
        self.anchor = torch.full((2, 2048), 2.5)
        self.models = {v: runner.build_model(self.seed, REGISTRY, v, self.norm, self.config) for v in runner.VARIANTS}

    def references(self, model):
        groups, refs = {}, {}
        for i, w in enumerate(self.dataset.windows):
            groups.setdefault(w.episode_index, []).append(i)
        for ids in groups.values():
            raw = runner.collate_window_inputs([self.dataset[i]["inputs"] for i in ids])
            inputs = runner.action_ablation_inputs(raw, self.arm)
            targets = runner.collate_window_targets([self.dataset[i]["targets"] for i in ids])
            with torch.inference_mode():
                pred = model(**inputs)
            refs[(self.seed, self.arm, tuple(ids))] = dict(input_sha256=runner.base.fingerprints(inputs),
                target_sha256=runner.base.fingerprints(targets), prediction_sha256=runner.base.fingerprints(pred))
        return refs

    def evaluate(self, variant, *, phase="final200", refs=None, trace=None):
        return runner.evaluate(self.models[variant], self.dataset, {"state_std": [1.] * 8}, seed=self.seed,
            arm=self.arm, variant=variant, phase=phase, anchor=self.anchor,
            references=refs or self.references(self.models["baseline"]), trace=trace)

    def test_initial_named_parameters_match_but_buffers_and_forward_differ(self):
        a, b = self.models.values()
        self.assertEqual(runner.named_parameter_sha(a), runner.named_parameter_sha(b))
        self.assertNotEqual(runner.base.parameter_hash(a), runner.base.parameter_hash(b))
        with torch.inference_mode():
            self.assertNotEqual(runner.base.fingerprints(a(**self.inputs)), runner.base.fingerprints(b(**self.inputs)))

    def test_capture_uses_actual_normalized_projection_input_and_raw_skip(self):
        model = self.models[runner.VARIANTS[1]]
        count = []
        h = model.register_forward_hook(lambda *args: count.append(1))
        before = runner.base.parameter_hash(model)
        try:
            captured, views = runner.capture_views(model, self.inputs)
        finally:
            h.remove()
        self.assertEqual(len(count), 1)
        self.assertEqual(runner.base.parameter_hash(model), before)
        for view in (0, 1):
            expected = (self.inputs["history_visual_latent"][:, :, view] - model.visual_mean[view]) / model.visual_scale[view]
            self.assertTrue(torch.equal(views[str(view)]["projected_input"], expected))
        self.assertTrue(torch.equal(captured["predictions"]["pred_future_visual_latent"],
                                   self.inputs["history_visual_latent"][:, -1, None, None] + captured["residual"]))

    def test_capture_restores_hooks_on_exception(self):
        model = self.models["baseline"]
        before = runner._hook_state(model)
        with patch.object(model.history_projection, "forward", side_effect=ValueError("synthetic failure")):
            with self.assertRaises(ValueError):
                runner.capture_views(model, self.inputs)
        self.assertEqual(runner._hook_state(model), before)

    def test_evaluation_complete_unequal_episodes_and_three_conditions(self):
        trace = io.StringIO()
        result = self.evaluate(runner.VARIANTS[1], trace=trace)
        self.assertEqual(result["windows"], 4)
        self.assertEqual(result["model_forwards"], 6)
        self.assertEqual(set(result["conditions"]), set(runner.CONDITIONS))
        rows = [json.loads(x) for x in trace.getvalue().splitlines()]
        self.assertEqual(len(rows), 6)
        for condition in runner.CONDITIONS:
            summary = result["conditions"][condition]
            self.assertEqual(set(summary["per_episode"]), {"11", "22"})
            self.assertEqual(summary["episode_macro"], runner.base.macro_metrics(summary["per_episode"]))
        self.assertEqual(result["conditions"]["clean"]["drift"]["micro"]["state_output_delta"]["exact_changed_count"], 0)
        self.assertEqual(result["conditions"]["repeat_current"]["drift"]["micro"]["anchor_skip_delta"]["exact_changed_count"], 0)

    def test_baseline_fails_closed_if_old_prediction_differs(self):
        refs = self.references(self.models["baseline"])
        next(iter(refs.values()))["prediction_sha256"]["pred_state_delta"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.evaluate("baseline", refs=refs)

    def test_input_and_target_reference_tampering_rejected(self):
        for key in ("input_sha256", "target_sha256"):
            refs = self.references(self.models["baseline"])
            item = next(iter(refs.values()))[key]
            item[next(iter(item))] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate(runner.VARIANTS[1], refs=refs)

    def test_step0_has_only_clean_condition(self):
        result = self.evaluate("baseline", phase="step0")
        self.assertEqual(set(result["conditions"]), {"clean"})
        self.assertEqual(result["model_forwards"], 2)

    def test_execution_flags_fail_before_preflight_or_output(self):
        with patch.object(runner, "validate", side_effect=AssertionError("must not run preflight")):
            with self.assertRaises(ValueError): runner.smoke(self.root, "a" * 64, False)
            with self.assertRaises(ValueError): runner.train(self.root, "a" * 64, None, True)
            with self.assertRaises(ValueError): runner.main(["--stage", "preflight", "--plan-sha256", "a" * 64, "--execute"])

    def test_claim_never_overwrites(self):
        out = runner.claim(self.root, "out", {"test": True})
        with self.assertRaises(ValueError): runner.claim(self.root, "out", {})
        self.assertTrue((out / "started.json").exists())

    def binding(self, model, variant):
        stats_bytes = (json.dumps(self.norm, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        return dict(seed=self.seed, arm=self.arm, variant=variant, step=200, plan_sha256="a" * 64,
                    smoke_report_sha256="b" * 64, normalization_sha256=runner.hashlib.sha256(stats_bytes).hexdigest(),
                    normalization_applied=variant != "baseline", draw_sha256="c" * 64,
                    initial_named_parameter_sha256="d" * 64,
                    final_state_sha256=runner.base.parameter_hash(model), final_named_parameter_sha256=runner.named_parameter_sha(model))

    def test_new_checkpoint_roundtrip_preserves_normalization_and_raw_predictions(self):
        variant = runner.VARIANTS[1]
        model = self.models[variant]
        # Exercise serialization only; real optimizer counters covered by reused native tests.
        with patch.object(runner.native, "_optimizer") as guard:
            loaded, record = runner.save_reload(self.root / "new.pt", model, object(), registry=REGISTRY,
                visual_stats=self.norm, binding=self.binding(model, variant))
        self.assertEqual(guard.call_args.args[2], 200)
        self.assertEqual(runner.base.parameter_hash(loaded), runner.base.parameter_hash(model))
        self.assertFalse(record["optimizer_state_saved"] or record["resumable"])
        with torch.inference_mode():
            self.assertEqual(runner.base.fingerprints(loaded(**self.inputs)), runner.base.fingerprints(model(**self.inputs)))

    def test_checkpoint_wrong_bound_hash_and_overwrite_rejected(self):
        model = self.models["baseline"]
        binding = self.binding(model, "baseline")
        binding["final_state_sha256"] = "0" * 64
        with patch.object(runner.native, "_optimizer"):
            with self.assertRaises(ValueError):
                runner.save_reload(self.root / "bad.pt", model, object(), registry=REGISTRY,
                    visual_stats=self.norm, binding=binding)
            runner.save_reload(self.root / "bad.pt", model, object(), registry=REGISTRY,
                visual_stats=self.norm, binding=self.binding(model, "baseline"))
            with self.assertRaises(ValueError):
                runner.save_reload(self.root / "bad.pt", model, object(), registry=REGISTRY,
                    visual_stats=self.norm, binding=self.binding(model, "baseline"))

    def test_checkpoint_rejects_variant_seed_stats_and_buffer_confusion(self):
        model = self.models[runner.VARIANTS[1]]
        for key, value in (("variant", "baseline"), ("seed", 42), ("normalization_sha256", "0" * 64),
                           ("normalization_applied", False), ("arm", "unknown"), ("step", 201)):
            binding = self.binding(model, runner.VARIANTS[1]); binding[key] = value
            with self.subTest(key=key), patch.object(runner.native, "_optimizer"), self.assertRaises(ValueError):
                runner.save_reload(self.root / "invalid.pt", model, object(), registry=REGISTRY,
                    visual_stats=self.norm, binding=binding)
        altered = deepcopy(self.norm); altered["mean"][0][0] += 1.
        binding = self.binding(model, runner.VARIANTS[1])
        binding["normalization_sha256"] = runner.hashlib.sha256((json.dumps(altered, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
        with patch.object(runner.native, "_optimizer"), self.assertRaisesRegex(ValueError, "buffers"):
            runner.save_reload(self.root / "wrong-buffer.pt", model, object(), registry=REGISTRY,
                visual_stats=altered, binding=binding)

    def test_decision_does_not_promote_desaturation_or_state_only_gain(self):
        runs = {}
        for seed in runner.base.SEEDS:
            runs[str(seed)] = {}
            for arm in runner.base.ARMS:
                runs[str(seed)][arm] = {}
                for variant in runner.VARIANTS:
                    normalized = variant != "baseline"
                    metric = dict(normalized_state_mae=.8 if normalized else 1., visual_mae=1.2 if normalized else 1.,
                                  masked_objective=.9 if normalized else 1.)
                    runs[str(seed)][arm][variant] = {"final": dict(conditions={"clean": {"episode_macro": metric}},
                        activation={"episode_macro": {v: {"saturated_fraction": .1 if normalized else 1.} for v in ("0", "1")}})}
        result = runner.summarize(runs)
        self.assertTrue(result["primary_desaturation_all_seed_view"])
        self.assertFalse(result["primary_fixed_validation_prediction_gain"])
        for seed in runner.base.SEEDS:
            runs[str(seed)][runner.base.ARMS[0]][runner.VARIANTS[1]]["final"]["conditions"]["clean"]["episode_macro"]["visual_mae"] = .9
        self.assertTrue(runner.summarize(runs)["primary_fixed_validation_prediction_gain"])
        runs[str(runner.base.SEEDS[0])][runner.base.ARMS[0]][runner.VARIANTS[1]]["final"]["conditions"]["clean"]["episode_macro"]["masked_objective"] = 1.1
        self.assertFalse(runner.summarize(runs)["primary_fixed_validation_prediction_gain"])


if __name__ == "__main__":
    unittest.main()
