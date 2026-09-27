"""Read-only native activation capture and descriptive moments, no training."""
from __future__ import annotations

import math
import numpy as np
import torch

from pi05_libero_visual_dependence import _require, _same_tensor, _same_inputs, _input_copy, _model_state, _hook_state


def capture_activations(model, inputs):
    snapshot = _input_copy(inputs)
    before, hooks = _model_state(model), _hook_state(model)
    saved, handles = {}, []

    def save(key, value):
        _require(key not in saved, "every inspected layer must run exactly once")
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                 and value.device.type == "cpu" and bool(torch.isfinite(value).all()), "finite native activation required")
        saved[key] = value.detach().clone()

    def linear(name):
        def hook(_module, args, output):
            save(name + "_in", args[0]); save(name, output)
            return None
        return hook

    def gru(name):
        def hook(_module, args, output):
            save(name + "_in", args[0]); save(name + "_seq", output[0]); save(name + "_hidden", output[1])
            if name == "action_gru":
                save(name + "_initial", args[1])
            return None
        return hook

    layers = {"view0": model.view_projections[0], "view1": model.view_projections[1],
              "history": model.history_projection, "task": model.task_embedding,
              "context_task": model.context_task_projection, "action": model.action_projection,
              "visual_head": model.visual_residual_head, "state_head": model.state_residual_head}
    try:
        for name, layer in layers.items():
            if name == "task":
                def task_hook(_module, _args, output):
                    save("task", output)
                    return None
                handles.append(layer.register_forward_hook(task_hook))
            else:
                handles.append(layer.register_forward_hook(linear(name)))
        for name in ("context_gru", "action_gru"):
            handles.append(getattr(model, name).register_forward_hook(gru(name)))
        with torch.inference_mode():
            predictions = model(**inputs)
        # Ordinary owned clones, outside inference mode.
        saved = {k: v.clone() for k, v in saved.items()}
        b, h = inputs["history_state"].shape[0], model.config.hidden_dim
        tensors = dict(view0_pre=saved["view0"], view1_pre=saved["view1"],
            view0_post=saved["history_in"][..., :h].clone(), view1_post=saved["history_in"][..., h:2*h].clone(),
            history_concat=saved["history_in"], history_pre=saved["history"], history_post=saved["context_gru_in"],
            context_hidden=saved["context_gru_hidden"][-1].clone(), task_embedding=saved["task"],
            context_task_concat=saved["context_task_in"], context_task_pre=saved["context_task"],
            initial_post=saved["action_gru_initial"].transpose(0,1).clone(),
            action_pre=saved["action"], action_post=saved["action_gru_in"],
            future=saved["visual_head_in"], visual_residual=saved["visual_head"].reshape(b,1,3,2,2048).clone(),
            state_residual=saved["state_head"])
        expected_shapes = {"view0_pre":(b,4,h), "view1_pre":(b,4,h), "view0_post":(b,4,h), "view1_post":(b,4,h),
            "history_concat":(b,4,2*h+18), "history_pre":(b,4,h), "history_post":(b,4,h),
            "context_hidden":(b,h), "task_embedding":(b,h), "context_task_concat":(b,2*h),
            "context_task_pre":(b,h), "initial_post":(b,1,h), "action_pre":(b,3,h), "action_post":(b,3,h),
            "future":(b,1,3,h), "visual_residual":(b,1,3,2,2048), "state_residual":(b,1,3,8)}
        _require(all(tuple(tensors[k].shape) == shape for k,shape in expected_shapes.items()), "actual activation shape differs")
        for pre, post in (("view0_pre","view0_post"),("view1_pre","view1_post"),("history_pre","history_post"),
                          ("context_task_pre","initial_post"),("action_pre","action_post")):
            actual = tensors[post][:,0] if post == "initial_post" else tensors[post]
            _require(_same_tensor(torch.tanh(tensors[pre]), actual), "actual downstream tanh post differs")
        history_expected = torch.cat([tensors["view0_post"], tensors["view1_post"], inputs["history_visual_valid"].float(),
                                      inputs["history_state"], inputs["history_state_valid"].float()], dim=-1)
        _require(_same_tensor(history_expected,tensors["history_concat"]), "history fusion input differs")
        _require(_same_tensor(torch.cat([tensors["context_hidden"],tensors["task_embedding"]],-1),tensors["context_task_concat"]), "context/task fusion input differs")
        _require(_same_tensor(saved["view0_in"],inputs["history_visual_latent"][:,:,0])
                 and _same_tensor(saved["view1_in"],inputs["history_visual_latent"][:,:,1])
                 and _same_tensor(saved["action_in"],inputs["candidate_actions"][:,0]), "projection input differs")
        _require(_same_tensor(saved["state_head_in"],tensors["future"])
                 and _same_tensor(saved["action_gru_seq"].reshape(b,1,3,h),tensors["future"]), "future/head inputs differ")
        _require(set(predictions) == {"pred_state_delta","pred_future_visual_latent"}
                 and _same_tensor(predictions["pred_state_delta"],tensors["state_residual"])
                 and _same_tensor(predictions["pred_future_visual_latent"],inputs["history_visual_latent"][:,-1,None,None]+tensors["visual_residual"]), "original head/skip reconstruction differs")
        return dict(predictions={k:v.detach().clone() for k,v in predictions.items()}, tensors=tensors)
    finally:
        for handle in handles:
            handle.remove()
        _require(_hook_state(model)==hooks, "inspection changed hook registries")
        _require(_same_inputs(inputs,snapshot), "inspection mutated inputs")
        _require(_model_state(model)==before, "inspection changed frozen model")


def _array(tensor):
    _require(isinstance(tensor,torch.Tensor) and tensor.layout==torch.strided and tensor.device.type=="cpu"
             and tensor.dtype in (torch.float32,torch.float64) and tensor.ndim>=1 and tensor.numel()>0
             and not tensor.requires_grad and not torch.is_inference(tensor)
             and bool(torch.isfinite(tensor).all()), "finite ordinary detached CPU float tensor required")
    return tensor.detach().double().numpy()


class MomentTotals:
    def __init__(self,feature_width=None):
        _require(feature_width is None or (type(feature_width) is int and feature_width>0), "positive feature width required")
        self.width=feature_width
        self.n=self.windows=self.updates=self.positions=0
        self.sum=self.sq=self.absolute=self.mean=self.m2=0.
        self.low=float("inf"); self.high=-float("inf")
        self.fm=None if feature_width is None else np.zeros(feature_width,np.float64)
        self.f2=None if feature_width is None else np.zeros(feature_width,np.float64)

    def update(self,tensor):
        value=_array(tensor)
        _require(self.width is None or (value.ndim>=2 and value.shape[-1]==self.width),"feature width requires matching last axis and at least two dimensions")
        with np.errstate(over="ignore",invalid="ignore"):
            energy=float(np.square(value).sum()); magnitude=float(np.abs(value).sum())
        _require(math.isfinite(self.sq+energy) and math.isfinite(self.absolute+magnitude),
                 "moment sums would overflow; accumulator unchanged")
        for row in value:
            flat=row.reshape(-1); count=flat.size; mean=float(flat.mean()); m2=float(np.square(flat-mean).sum())
            difference=mean-self.mean; new=self.n+count
            self.m2+=m2+difference*(difference*(self.n/new))*count
            self.mean+=difference*count/new; self.n=new
            self.sum+=float(flat.sum()); self.sq+=float(np.square(flat).sum()); self.absolute+=float(np.abs(flat).sum())
            self.low=min(self.low,float(flat.min())); self.high=max(self.high,float(flat.max()))
            if self.width is not None:
                for feature in row.reshape(-1,self.width):
                    self.positions+=1; d=feature-self.fm; self.fm+=d/self.positions; self.f2+=d*(feature-self.fm)
        self.windows+=value.shape[0]; self.updates+=1

    def summary(self):
        n=self.n
        result=dict(windows=self.windows,updates=self.updates,count=n,sum=self.sum,sum_sq=self.sq,sum_abs=self.absolute,
            min=self.low if n else None,max=self.high if n else None,mean=self.sum/n if n else None,
            mean_abs=self.absolute/n if n else None,rms=math.sqrt(self.sq/n) if n else None,
            std=math.sqrt(max(0,self.m2/n)) if n else None,centered_sum_sq=self.m2,feature_width=self.width)
        if self.width is not None:
            std=np.sqrt(np.maximum(0,self.f2/max(1,self.positions)))
            result.update(feature_position_count=self.positions,
                coordinate_mean_rms=float(np.sqrt(np.mean(self.fm*self.fm))) if n else None,
                coordinate_std_rms=float(np.sqrt(np.mean(std*std))) if n else None,
                coordinate_std_min=float(std.min()) if n else None,coordinate_std_max=float(std.max()) if n else None,
                constant_coordinate_count=int(np.count_nonzero(self.f2==0)))
        return result


class TanhTotals:
    def __init__(self):
        self.windows=self.updates=self.n=self.exact=0
        self.slope=0.; self.low=float("inf"); self.high=-float("inf")
        self.abs_counts={"0.99":0,"0.999":0}; self.slope_counts={"0.01":0,"0.001":0}

    def update(self,pre,post):
        _array(pre); values=_array(post)
        _require(pre.dtype==post.dtype==torch.float32 and pre.shape==post.shape
                 and _same_tensor(torch.tanh(pre),post), "actual float32 tanh correspondence required")
        for row in values:
            absolute=np.abs(row); slope=1-row*row
            _require(bool((slope>=0).all()),"invalid tanh output")
            self.n+=row.size; self.slope+=float(slope.sum()); self.exact+=int(np.count_nonzero(absolute==1))
            self.low=min(self.low,float(slope.min())); self.high=max(self.high,float(slope.max()))
            for k in self.abs_counts: self.abs_counts[k]+=int(np.count_nonzero(absolute>=float(k)))
            for k in self.slope_counts: self.slope_counts[k]+=int(np.count_nonzero(slope<=float(k)))
        self.windows+=values.shape[0]; self.updates+=1

    def summary(self):
        n=self.n
        return dict(windows=self.windows,updates=self.updates,count=n,sum_slope=self.slope,
            mean_slope=self.slope/n if n else None,min_slope=self.low if n else None,max_slope=self.high if n else None,
            post_abs_exact_1_count=self.exact,post_abs_exact_1_fraction=self.exact/n if n else None,
            post_abs_ge={k:dict(count=v,fraction=v/n if n else None) for k,v in self.abs_counts.items()},
            local_slope_le={k:dict(count=v,fraction=v/n if n else None) for k,v in self.slope_counts.items()},
            threshold_spec=dict(post_abs_ge=[.99,.999],local_slope_le=[.01,.001],
                slope="1-actual_float32_post_cast_to_float64_squared; local proxy, not backward gradients or GRU gates"))
