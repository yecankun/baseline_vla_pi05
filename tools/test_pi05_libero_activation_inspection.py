"""Synthetic CPU regression only; no real checkpoint/cache/extraction/training."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import pi05_libero_activation_inspection as helper
import run_pi05_libero_activation_inspection as runner
from test_pi05_libero_visual_dependence import fixture
from test_pi05_libero_action_study_step0 import TinyDataset


class ActivationInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actual_capture_exact_predictions_post_and_owned_tensors(self):
        model,inputs,_=fixture(); before=runner.base.parameter_hash(model)
        with torch.inference_mode(): expected=model(**inputs)
        record=helper.capture_activations(model,inputs)
        self.assertEqual(runner.base.fingerprints(expected),runner.base.fingerprints(record["predictions"]))
        t=record["tensors"]
        self.assertEqual(len(t),17)
        for name,(pre,post) in runner.TANH_PAIRS.items():
            actual=t[post][:,0] if name=="context_task" else t[post]
            self.assertTrue(helper._same_tensor(t[pre].tanh(),actual))
        self.assertTrue(all(not torch.is_inference(v) and not v.requires_grad for v in t.values()))
        self.assertEqual(before,runner.base.parameter_hash(model))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        raw=inputs["history_state"].clone(); t["history_concat"].zero_()
        self.assertTrue(torch.equal(raw,inputs["history_state"]))

    def test_exception_and_existing_hook_cleanup(self):
        model,inputs,_=fixture()
        existing=model.history_projection.register_forward_hook(lambda *_:None)
        hooks=helper._hook_state(model)
        with patch.object(model.action_gru,"forward",side_effect=RuntimeError("synthetic")), self.assertRaisesRegex(RuntimeError,"synthetic"):
            helper.capture_activations(model,inputs)
        self.assertEqual(hooks,helper._hook_state(model)); existing.remove()

    def test_duplicate_layer_and_mutated_output_rejected(self):
        for mode in ("duplicate","output"):
            model,inputs,_=fixture(); original=model.forward
            def bad(**kwargs):
                result=original(**kwargs)
                if mode=="duplicate": model.view_projections[0](inputs["history_visual_latent"][:,:,0])
                else: result["pred_state_delta"]+=1
                return result
            with patch.object(model,"forward",bad),self.assertRaises(ValueError): helper.capture_activations(model,inputs)
            self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_frozen_masks_and_inference_guards(self):
        for mode in ("train","grad","mask","inference"):
            model,inputs,_=fixture()
            if mode=="train": model.train()
            if mode=="grad": model.requires_grad_(True)
            if mode=="mask": inputs["history_visual_valid"][0,0,0]=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                if mode=="inference":
                    with torch.inference_mode(): helper.capture_activations(model,inputs)
                else: helper.capture_activations(model,inputs)

    def test_no_backward_optimizer_rng_change(self):
        model,inputs,_=fixture(); rng=torch.get_rng_state().clone()
        with patch("torch.optim.AdamW",side_effect=AssertionError("optimizer")),patch.object(torch.Tensor,"backward",side_effect=AssertionError("backward")):
            helper.capture_activations(model,inputs)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_zero_action_projection_includes_bias(self):
        model,inputs,_=fixture(); inputs["candidate_actions"].zero_()
        t=helper.capture_activations(model,inputs)["tensors"]
        self.assertTrue(torch.equal(t["action_pre"],model.action_projection.bias.expand_as(t["action_pre"])))
        self.assertTrue(torch.equal(t["action_post"],model.action_projection.bias.tanh().expand_as(t["action_post"])))

    def test_moment_quantities_coordinate_variation_and_split(self):
        x=torch.tensor([[1.,3.],[5.,7.]],dtype=torch.float64)
        one,two=helper.MomentTotals(2),helper.MomentTotals(2)
        one.update(x)
        for row in x: two.update(row[None])
        a,b=one.summary(),two.summary(); a.pop("updates"); b.pop("updates")
        self.assertEqual(a,b); self.assertEqual(a["count"],4)
        self.assertEqual(a["sum"],16);self.assertEqual(a["sum_sq"],84)
        self.assertEqual(a["centered_sum_sq"],20);self.assertEqual(a["coordinate_std_rms"],2)
        constant=helper.MomentTotals(2);constant.update(torch.tensor([[1.,5.],[1.,5.]]))
        c=constant.summary();self.assertEqual(c["constant_coordinate_count"],2)
        self.assertGreater(c["std"],0);self.assertEqual(c["coordinate_std_rms"],0)

    def test_stable_centered_variance(self):
        x=torch.tensor([[1e12+1,1e12+3],[1e12+5,1e12+7]],dtype=torch.float64)
        m=helper.MomentTotals();m.update(x)
        self.assertAlmostEqual(m.summary()["std"],np.std(x.numpy()),places=8)

    def test_invalid_update_atomicity(self):
        m=helper.MomentTotals(2);m.update(torch.ones(1,2));before=m.summary()
        for x in (torch.ones(2),torch.ones(2,3),torch.full((1,2),float("nan")),torch.full((1,2),1e308,dtype=torch.float64)):
            with self.assertRaises(ValueError):m.update(x)
            self.assertEqual(m.summary(),before)

    def test_tanh_counts_and_slope_proxy(self):
        pre=torch.tensor([[0.,3.,10.],[-3.,-10.,0.]])
        post=pre.tanh(); t=helper.TanhTotals();t.update(pre,post);s=t.summary()
        self.assertEqual(s["count"],6);self.assertEqual(s["post_abs_exact_1_count"],2)
        self.assertEqual(s["post_abs_ge"]["0.99"]["count"],4)
        self.assertEqual(s["post_abs_ge"]["0.999"]["count"],2)
        self.assertAlmostEqual(s["sum_slope"],float((1-post.double().square()).sum()))
        before=t.summary()
        with self.assertRaises(ValueError): t.update(pre,post+.1)
        self.assertEqual(t.summary(),before)

    def test_affine_parts_complete_and_cancellation(self):
        layer=torch.nn.Linear(4,2).eval().requires_grad_(False)
        with torch.no_grad():layer.weight.fill_(1);layer.bias.fill_(.5)
        x=torch.tensor([[1.,2.,-1.,-2.],[2.,3.,-2.,-3.]])
        parts=runner.affine_parts(layer,x,{"a":(0,2),"b":(2,4)},layer(x))
        self.assertEqual(parts["max_abs_float64_reconstruction_error"],0)
        self.assertTrue(torch.equal(sum(parts["parts"].values()),layer(x).double()))
        self.assertGreater(float(parts["parts"]["a"].abs().mean()),float(layer(x).abs().mean()))
        for groups in ({"a":(0,3)},{"a":(0,3),"b":(2,4)},{"bias":(0,4)}):
            with self.assertRaises(ValueError):runner.affine_parts(layer,x,groups,layer(x))

    def make_refs(self,model,dataset,phase,arm):
        grouped={}
        for i,w in enumerate(dataset.windows):grouped.setdefault(w.episode_index,[]).append(i)
        refs={}
        for selected in grouped.values():
            raw=runner.collate_window_inputs([dataset[i]["inputs"] for i in selected])
            inputs=runner.action_ablation_inputs(raw,arm)
            target=runner.collate_window_targets([dataset[i]["targets"] for i in selected])
            pred=helper.capture_activations(model,inputs)["predictions"]
            a,b,c=map(runner.base.fingerprints,(inputs,target,pred))
            refs[tuple(selected)]=dict(arm_input_sha256={arm:a},common_target_sha256=b,predictions_sha256={arm:c}) if phase=="step0" else dict(input_sha256=a,target_sha256=b,prediction_sha256=c)
        return refs

    def test_runner_both_phases_arms_exact_coverage_macros(self):
        model,_,_=fixture();dataset=TinyDataset()
        for phase in runner.PHASES:
            for arm in runner.base.ARMS:
                refs=self.make_refs(model,dataset,phase,arm);trace=io.StringIO()
                r=runner.inspect_model(model,dataset,phase=phase,seed=20260912,arm=arm,references=refs,trace=trace)
                self.assertEqual((r["windows"],r["batches"]),(4,2))
                self.assertEqual(len(trace.getvalue().splitlines()),2)
                for name,val in r["micro"]["moments"].items():
                    parts=[e["moments"][name] for e in r["per_episode"].values()]
                    self.assertEqual(val["count"],sum(p["count"] for p in parts))
                    self.assertAlmostEqual(val["sum"],sum(p["sum"] for p in parts),places=7)
                self.assertEqual(r["episode_macro"],runner.macros(r["per_episode"]))

    def test_runner_reference_failure_before_or_after_forward(self):
        model,_,_=fixture();dataset=TinyDataset();arm=runner.base.ARMS[0]
        original=self.make_refs(model,dataset,"final200",arm)
        for kind in ("input_sha256","target_sha256","prediction_sha256","missing"):
            refs=deepcopy(original)
            if kind=="missing":del refs[(0,)]
            else:refs[(0,)][kind]={"bad":"0"*64}
            trace=io.StringIO()
            with self.assertRaises(ValueError):runner.inspect_model(model,dataset,phase="final200",seed=20260912,arm=arm,references=refs,trace=trace)
            self.assertEqual(trace.getvalue(),"")

    def test_cache_scales_unique_partition_rows_and_no_mutation(self):
        from types import SimpleNamespace
        values=np.arange(4*2*2048,dtype=np.float32).reshape(4,2,2048)/1000
        pack=SimpleNamespace(indices={1:np.array([0,1]),2:np.array([2,3])},arrays={"visual_latent":values,"visual_valid":np.ones((4,2),bool)})
        split={"train_episode_indices":[1],"validation_episode_indices":[2]};before=values.copy()
        with patch.object(runner,"PARTITION_ROWS",{"train":2,"validation":2}):
            r=runner.cache_scales(pack,split)
            self.assertEqual(r["train"]["unique_rows"],2)
            split["train_episode_indices"]=[1,1]
            with self.assertRaises(ValueError):runner.cache_scales(pack,split)
        np.testing.assert_array_equal(values,before)

    def test_cli_execution_and_exclusive_output(self):
        for stage,execute in (("preflight",True),("inspect",False)):
            args=["--stage",stage,"--plan-sha256","0"*64]+(["--execute"] if execute else [])
            with patch.object(runner,"run") as run,patch.object(runner,"validate") as val,self.assertRaises(ValueError):runner.main(args)
            run.assert_not_called();val.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/runner.OUT).mkdir(parents=True)
            with self.assertRaises(ValueError):runner.run(root,"0"*64,True)


if __name__=="__main__":unittest.main()
