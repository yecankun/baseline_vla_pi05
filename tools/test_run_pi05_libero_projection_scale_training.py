"""Synthetic-only checks of matched training orchestration, not real training."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_pi05_libero_projection_scale_training as runner
from test_pi05_libero_action_study_step0 import TinyDataset, REGISTRY, forbid_training
from test_pi05_libero_visual_normalization import synthetic_pack


class ScaleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = TinyDataset()
        for i,item in enumerate(self.dataset.items): item["inputs"]["history_visual_latent"].fill(i+1.)
        pack, split = synthetic_pack()
        self.norm = runner.previous.fit_visual_statistics(pack, split)
        self.seed, self.arm = runner.base.SEEDS[0], runner.base.ARMS[0]
        cfg = runner.previous.LiberoWorldModelConfig(hidden_dim=4)
        self.models = {v: runner.gate.build_model(self.seed, REGISTRY, v, self.norm, cfg) for v in runner.VARIANTS}
        self.stats = {"state_std": [1.]*8}

    def references(self):
        refs = {}
        for eid,ids in ((11,[0]), (22,[1,2,3])):
            raw = runner.previous.collate_window_inputs([self.dataset[i]["inputs"] for i in ids])
            targets = runner.previous.collate_window_targets([self.dataset[i]["targets"] for i in ids])
            inputs = runner.previous.action_ablation_inputs(raw, self.arm)
            captured, _ = runner.gate.capture_views(self.models[runner.VARIANTS[0]], inputs)
            score = runner.base.Totals(self.stats); score.update(captured["predictions"], targets)
            for phase in ("step0", "final200"):
                refs[(self.seed,self.arm,phase,tuple(ids))] = dict(episode_index=eid,
                    input_sha256=runner.base.fingerprints(inputs), target_sha256=runner.base.fingerprints(targets),
                    prediction_sha256=runner.base.fingerprints(captured["predictions"]), score=score.summary())
        return refs

    def evaluate(self, variant, refs=None, trace=None):
        return runner.evaluate(self.models[variant], self.dataset, self.stats, seed=self.seed, arm=self.arm,
            variant=variant, phase="final200", references=refs or self.references(), trace=trace)

    def test_clean_only_full_coverage_macro_horizons_and_no_training(self):
        trace = io.StringIO()
        with forbid_training(): result = self.evaluate(runner.VARIANTS[0], trace=trace)
        self.assertEqual((result["windows"],result["model_forwards"]), (4,2))
        self.assertEqual(len(trace.getvalue().splitlines()),2)
        self.assertEqual(set(result["per_episode"]), {"11","22"})
        for field,label in (("visual","visual"),("state_normalized","normalized_state")):
            for h in range(3):
                for metric in ("mae","rmse"):
                    want = sum(r[field]["per_horizon"][h][metric] for r in result["per_episode"].values())/2
                    self.assertEqual(result["episode_macro"][f"{label}_h{h+1}_{metric}"],want)
        self.assertNotEqual(result["episode_macro"]["normalized_state_h1_rmse"], result["micro"]["state_normalized"]["per_horizon"][0]["rmse"])

    def test_scaled_effective_rms_and_state_immutability(self):
        variant = runner.VARIANTS[1]
        before = runner.base.parameter_hash(self.models[variant])
        result = self.evaluate(variant)
        self.assertEqual(before,runner.base.parameter_hash(self.models[variant]))
        for episode in result["activation"]["vector_rms"].values():
            for view in episode.values():
                self.assertLessEqual(view["pre"]["max"],1+1e-6)
                self.assertGreater(view["raw_affine"]["max"],view["pre"]["max"])

    def test_reference_prediction_or_score_tamper_fails(self):
        for field in ("prediction_sha256","score"):
            refs = self.references(); row = refs[(self.seed,self.arm,"final200",(0,))]
            row[field] = {}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,"replay"):
                self.evaluate(runner.VARIANTS[0],refs)

    def test_reference_inputs_targets_episode_tamper_both_variants(self):
        for field in ("input_sha256","target_sha256","episode_index"):
            for variant in runner.VARIANTS:
                refs = self.references(); refs[(self.seed,self.arm,"final200",(0,))][field] = None
                with self.subTest(field=field,variant=variant), self.assertRaisesRegex(ValueError,"fixed evaluation"):
                    self.evaluate(variant,refs)

    def trace_row(self):
        return dict(seed=self.seed,arm=self.arm,step=1,window_indices=[1,2],input_sha256={"a":"b"},target_sha256={"c":"d"},details={"loss":.2})

    def test_trace_reference_exact_and_variant_only_metadata(self):
        row=self.trace_row(); stream=io.StringIO()
        trace=runner.PairedTrace(stream,runner.VARIANTS[0],{(self.seed,self.arm,1):row})
        trace.write(json.dumps(row)); trace.flush()
        self.assertEqual(trace.steps,1)
        self.assertEqual(json.loads(stream.getvalue()),dict(row,variant=runner.VARIANTS[0]))
        with self.assertRaisesRegex(ValueError,"sequence"): trace.write(json.dumps(row))

    def test_trace_reference_details_and_shared_draws_fail_closed(self):
        for variant in runner.VARIANTS:
            for field in ("details","window_indices","input_sha256","target_sha256"):
                row=self.trace_row(); bad=deepcopy(row); bad[field]={}
                trace=runner.PairedTrace(io.StringIO(),variant,{(self.seed,self.arm,1):row})
                if variant==runner.VARIANTS[1] and field=="details":
                    trace.write(json.dumps(bad)); self.assertEqual(trace.steps,1)
                else:
                    with self.subTest(variant=variant,field=field),self.assertRaises(ValueError): trace.write(json.dumps(bad))

    def summary_fixture(self):
        result=self.evaluate(runner.VARIANTS[0])
        runs={str(s):{a:{v:dict(final=deepcopy(result)) for v in runner.VARIANTS} for a in runner.base.ARMS} for s in runner.base.SEEDS}
        old=dict(runs={str(s):{a:dict(baseline=dict(final=dict(conditions=dict(clean=dict(per_episode=deepcopy(result["per_episode"]))))))
            for a in runner.base.ARMS} for s in runner.base.SEEDS})
        for r in runs.values():
            for a in r.values():
                for key in a[runner.VARIANTS[1]]["final"]["episode_macro"]:
                    a[runner.VARIANTS[1]]["final"]["episode_macro"][key]*=.9
        return runs,old

    def test_summary_success_and_separate_raw_reference(self):
        runs,old=self.summary_fixture(); summary=runner.summarize(runs,old)
        self.assertTrue(summary["primary_fixed_validation_prediction_gain"])
        self.assertFalse(summary["automatic_checkpoint_promotion"])
        self.assertEqual(set(summary["primary_mean_MAE_guards"]),{"visual_mae","normalized_state_mae",*[f"{name}_h{h}_mae" for name in ("visual","normalized_state") for h in (1,2,3)]})
        self.assertIn("changes_vs_saved_raw_baseline",summary)

    def test_one_bad_horizon_fails_even_if_aggregate_improves(self):
        runs,old=self.summary_fixture()
        for r in runs.values(): r[self.arm][runner.VARIANTS[1]]["final"]["episode_macro"]["visual_h1_mae"]*=2
        summary=runner.summarize(runs,old)
        self.assertFalse(summary["primary_fixed_validation_prediction_gain"])
        self.assertFalse(summary["primary_mean_MAE_guards"]["visual_h1_mae"])

    def test_one_seed_equal_objective_fails(self):
        runs,old=self.summary_fixture(); a=runs[str(self.seed)][self.arm]
        a[runner.VARIANTS[1]]["final"]["episode_macro"]["masked_objective"]=a[runner.VARIANTS[0]]["final"]["episode_macro"]["masked_objective"]
        self.assertFalse(runner.summarize(runs,old)["primary_fixed_validation_prediction_gain"])

    def test_secondary_only_improvement_does_not_pass(self):
        runs,old=self.summary_fixture()
        for r in runs.values(): r[self.arm][runner.VARIANTS[1]]=deepcopy(r[self.arm][runner.VARIANTS[0]])
        self.assertFalse(runner.summarize(runs,old)["primary_fixed_validation_prediction_gain"])

    def test_reference_maps_duplicate_and_incomplete_fail(self):
        path=self.root/runner.previous.OUT; path.mkdir(parents=True)
        row=dict(variant="train_visual_normalized",condition="clean",seed=self.seed,arm=self.arm,phase="step0",window_indices=[1])
        runner.base.write_json(path/"unused.json",{})
        with (path/"training_trace.jsonl").open("x") as stream: pass
        for rows in ([row],[row,row]):
            with (path/"evaluation_trace.jsonl").open("w") as stream:
                for r in rows: stream.write(json.dumps(r)+"\n")
            with self.assertRaises(ValueError): runner.reference_maps(self.root)

    def test_explicit_execution_gate_and_preflight_no_training(self):
        with patch.object(runner,"validate",side_effect=AssertionError("no preflight")):
            with self.assertRaises(ValueError): runner.train(self.root,"a"*64,False)
            with self.assertRaises(ValueError): runner.main(["--stage","preflight","--plan-sha256","a"*64,"--execute"])
        with patch.object(runner,"validate",return_value=({"status":"ok","input_sha256":{}},None,None,None)), patch.object(runner,"train",side_effect=AssertionError("no training")):
            runner.main(["--stage","preflight","--plan-sha256","a"*64])

    def test_output_claim_never_overwrites(self):
        runner.previous.claim(self.root,"new",{})
        with self.assertRaises(ValueError): runner.previous.claim(self.root,"new",{})


if __name__ == "__main__": unittest.main()
