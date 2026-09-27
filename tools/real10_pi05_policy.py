"""Real10 observation -> action inference, with no robot/camera connections.

Images are uint8 BGR arrays (OpenCV convention). Pose is xyz_mm/rpy_rad.
Pass only the PREVIOUS controller snapshot; None at episode reset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from eval_pi05_mixed_head_open_loop import load_mixed_head_checkpoint
from export_openpi_compat_to_lerobot import task_text
from pi05_state_conditioning import enable_state_conditioning
from prepare_real_pi05_pack import ADAPTER_VERSION, encode_real_state, resize_real_bgr
from probe_pi05_lerobot_adapter import build_config, import_pi05, set_seed


class Real10PI05Policy:
    def __init__(self, elite_checkpoint: Path, piper_checkpoint: Path, *, device="cuda", seed=123):
        set_seed(seed)
        started = time.perf_counter()
        checkpoint = torch.load(elite_checkpoint, map_location="cpu", mmap=True, weights_only=True)
        contract = checkpoint["split_details"]
        if contract.get("adapter_version") != ADAPTER_VERSION or checkpoint.get("active_action_dims") != 3:
            raise ValueError("expected the real10 translation-only Elite checkpoint")
        if not checkpoint.get("state_conditioning", {}).get("enabled"):
            raise ValueError("the real10 Elite checkpoint must include its trained state token")
        if checkpoint.get("checkpoint_selection") != "final_step":
            raise ValueError("expected the fixed final Elite checkpoint")
        args = argparse.Namespace(**checkpoint["args"])
        args.device = device
        args.gradient_checkpointing = False
        self.stats = {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in fields.items()}
                      for key, fields in checkpoint["normalization_stats"].items()}
        pi05 = import_pi05()
        self.config = build_config(args, pi05, 32, 32)
        self.preprocessor, _ = pi05["make_pi05_pre_post_processors"](self.config, self.stats)
        self.policy = pi05["PI05Policy"](self.config).to(device)
        enable_state_conditioning(self.policy, 32)
        self.policy.load_state_dict(checkpoint["model_state"], strict=True)
        prefix_dim = int(self.policy.model.paligemma_with_expert.paligemma.config.text_config.hidden_size)
        self.head, piper = load_mixed_head_checkpoint(piper_checkpoint, prefix_dim=prefix_dim, state_dim=32, device=device)
        if self.head.mode != "state_only" or piper.get("checkpoint_selection") != "final_step":
            raise ValueError("expected the final real10 state_only Piper head")
        if piper["split_details"].get("adapter_version") != ADAPTER_VERSION:
            raise ValueError("Piper input layout differs from real10")
        if piper["normalization_stats"] != checkpoint["normalization_stats"]:
            raise ValueError("Elite/Piper normalization mismatch")
        expected_base = Path(piper["base_policy_checkpoint"])
        if not expected_base.is_absolute():
            expected_base = Path(__file__).resolve().parents[1] / expected_base
        if expected_base.resolve() != elite_checkpoint.resolve():
            raise ValueError("Piper checkpoint references a different Elite policy; preserve or explicitly migrate the pair")
        self.policy.requires_grad_(False).eval()
        self.head.requires_grad_(False).eval()
        self.action_mean = self.stats["action"]["mean"].to(device)
        self.action_std = self.stats["action"]["std"].to(device)
        self.metadata = {
            "adapter_version": ADAPTER_VERSION, "elite_checkpoint": str(elite_checkpoint),
            "piper_checkpoint": str(piper_checkpoint), "elite_step": checkpoint["checkpoint_step"],
            "piper_step": piper["checkpoint_step"], "checkpoint_selection": "final_step",
            "state_conditioning": checkpoint["state_conditioning"], "state_sampling_context_fix": True,
            "normalization_equal": True, "weights_loaded_strictly": True,
            "num_inference_steps": self.config.num_inference_steps, "device": device,
            "load_seconds": time.perf_counter() - started, "train_only": True,
        }
        del checkpoint, piper

    @staticmethod
    def observation(*, side_bgr, top_bgr, elite_tcp_pose_6d, task, previous_controller_state=None):
        state, _ = encode_real_state(elite_tcp_pose_6d, task, previous_controller_state)
        observation = {"observation.state": torch.from_numpy(state), "task": task_text({"task": task})}
        for view, image in (("side", side_bgr), ("top", top_bgr)):
            if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{view} must be a uint8 BGR image")
            rgb = cv2.cvtColor(resize_real_bgr(image), cv2.COLOR_BGR2RGB)
            observation[f"observation.images.{view}"] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        return observation

    @torch.inference_mode()
    def predict(self, *, side_bgr, top_bgr, elite_tcp_pose_6d, task, previous_controller_state=None):
        started = time.perf_counter()
        observation = self.observation(side_bgr=side_bgr, top_bgr=top_bgr,
                                       elite_tcp_pose_6d=elite_tcp_pose_6d, task=task,
                                       previous_controller_state=previous_controller_state)
        processed = self.preprocessor(observation)  # deliberately contains NO action/label
        logits = self.head(None, None, processed["observation.state"])
        self.policy.reset()
        normalized = self.policy.predict_action_chunk(processed)
        translation = (normalized * self.action_std + self.action_mean)[0, 0, :3]
        probabilities = torch.softmax(logits.float(), dim=-1)[0]
        if not torch.isfinite(translation).all() or not torch.isfinite(probabilities).all():
            raise ValueError("non-finite policy output; no action packet emitted")
        intent = int(probabilities.argmax().item())
        return {
            "elite_tcp_delta_6d": translation.float().cpu().tolist() + [0.0, 0.0, 0.0],
            "piper_intent_id": intent,
            "diagnostics": {"piper_probabilities_retract_hold_feed": probabilities.cpu().tolist(),
                            "outside_training_intents": intent not in (1, 2),
                            "inference_seconds": time.perf_counter() - started,
                            "rotation_policy": "fixed zero; rotation was not trained",
                            "units": "xyz_mm, rpy_rad; displacement, not velocity",
                            "hardware_executed": False},
        }

    @torch.inference_mode()
    def piper_probabilities(self, state_rows, tasks):
        """State-only head fit diagnostic; no image or target enters the head."""
        batch = {"observation.state": torch.as_tensor(state_rows, dtype=torch.float32),
                 "task": [task_text({"task": task}) for task in tasks]}
        processed = self.preprocessor(batch)
        return self.head(None, None, processed["observation.state"]).softmax(-1).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elite-checkpoint", type=Path, default=Path("simulation_output/real10_pi05_train_v1/elite/final_policy.pt"))
    parser.add_argument("--piper-checkpoint", type=Path, default=Path("simulation_output/real10_pi05_train_v1/piper/mixed_head_policy.pt"))
    parser.add_argument("--input-json", type=Path, required=True,
                        help="side_image/top_image paths, task, elite_tcp_pose_6d, previous_controller_state")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    request = json.loads(args.input_json.read_text(encoding="utf-8"))
    model = Real10PI05Policy(args.elite_checkpoint, args.piper_checkpoint, device=args.device)
    images = {view: cv2.imread(str(args.input_json.parent / request[f"{view}_image"])) for view in ("side", "top")}
    result = model.predict(side_bgr=images["side"], top_bgr=images["top"],
                           elite_tcp_pose_6d=request["elite_tcp_pose_6d"], task=request["task"],
                           previous_controller_state=request.get("previous_controller_state"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
