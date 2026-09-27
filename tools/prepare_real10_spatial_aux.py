"""Prepare sparse tip supervision and a zero-update spatial-response preflight.

No train stage, pseudo-labels, tracking, new image crop, PI05 or hardware calls.
The renderer shows HUMAN TARGETS, never untrained model predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from review_real10_wire_correspondence import CASES, VIEWS, CorrespondenceStore
from run_real10_response_baseline import folds_for, image_path


SCHEMA = "real10_spatial_tip_aux_preparation_v1"
ROI = {"side": [0, 540, 1440, 1040], "top": [480, 560, 1920, 1080]}
PROTOCOL = Path("docs/algorithm-real10-spatial-aux-protocol-20260922.md")


def point_uv(xy, view):
    x0, y0, x1, y1 = ROI[view]
    if not (x0 <= xy[0] < x1 and y0 <= xy[1] < y1):
        raise ValueError("human point outside the unchanged ROI; do not clamp or move it")
    # Existing cv2.INTER_AREA resize uses a half-pixel pixel-center convention.
    return [(xy[0]-x0+.5)/(x1-x0), (xy[1]-y0+.5)/(y1-y0)]


def build(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve existing output; choose a fresh --out")
    rows = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    folds = read_json(args.reference / "folds.json")
    latest = {r["window_id"]: r for r in read_jsonl(args.pack / "annotations_joint_v2.jsonl")}
    if len(rows) != 45 or folds != folds_for(rows) or latest != {
            r["window_id"]: r["human_annotation"] for r in rows}:
        raise ValueError("the frozen 45 revised windows / ten episode folds changed")
    if read_json(args.image_pack / "protocol.json")["roi_xyxy_original"] != ROI:
        raise ValueError("reuse the existing fixed scene ROI")
    store = CorrespondenceStore(args.points, Path("."))
    if tuple(c["ui_index"] for c in store.cases.values()) != CASES:
        raise ValueError("only the existing six human correspondence cases are in scope")
    points_before = {p: (args.points / p).read_bytes() for p in
                     ("manifest.json", "annotations_v2.jsonl", "vessel_reference.jsonl")}
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    by_ui = {r["ui_index"]: r for r in rows}
    point_lookup, audits = {}, []
    for case_id, case in store.cases.items():
        source = by_ui[case["ui_index"]]
        expected = [source["anchor_frame_id"], *source["future_frame_ids"]]
        if source["source_episode"] != case["source_episode"] or source["task"] != case["task"]:
            raise ValueError("correspondence case identity mismatch")
        annotation = store.annotations[case_id]
        for view in VIEWS:
            sequence = [f'{case["source_episode"]}/{Path(f["path"]).stem}' for f in case["frames"][view]]
            if sequence != expected:
                raise ValueError("point frames do not match the original whole window")
            av = annotation["views"][view]
            for index in case["keyframe_indices"]:
                point = av["frames"][str(index)]["wire"]
                is_tip = av["feature_type"] == "tip" and point["status"] == "visible"
                uv = point_uv(point["xy"], view) if is_tip else None
                entry = {"window_id": source["window_id"], "ui_index": case["ui_index"],
                         "frame_id": sequence[index], "frame_index": index, "view": view,
                         "source_episode": case["source_episode"], "feature_type": av["feature_type"],
                         "feature_description": av["feature_description"], "wire": point,
                         "correspondence_confirmed": av["correspondence_confirmed"],
                         "annotation_revision": annotation["revision"], "tip_target_valid": is_tip,
                         "tip_xy_uv": uv, "same_point_motion_label_created": False}
                point_lookup[source["window_id"], index, view] = entry
                audits.append(entry)
    inputs, targets, images = [], [], set()
    for i, row in enumerate(rows):
        frames = [row["anchor_frame_id"], *row["future_frame_ids"]]
        sequence, xy, valid = [], [], []
        times = [observations[f]["observation_available_at_s"] for f in frames]
        for j, frame in enumerate(frames):
            paths = [image_path(args.image_pack, frame, v).as_posix() for v in VIEWS]
            images.update(paths)
            sequence.append({"images": dict(zip(VIEWS, paths))})
            found = [point_lookup.get((row["window_id"], j, v)) for v in VIEWS]
            xy.append([p["tip_xy_uv"] if p else None for p in found])
            valid.append([p["tip_target_valid"] if p else False for p in found])
        inputs.append({"sample_index": i, "window_id": row["window_id"],
                       "model_input": {"sequence": sequence, "dt_s": np.diff(times).tolist(),
                                       "task_right": float(row["task"] == "right")}})
        targets.append({"sample_index": i, "window_id": row["window_id"],
                        "response_label": row["human_annotation"]["joint_motion_response"],
                        "response_y": int(row["human_annotation"]["joint_motion_response"] == "advance"),
                        "tip_xy_uv": xy, "tip_valid": valid})
    if any(not Path(p).is_file() for p in images):
        raise FileNotFoundError("missing existing ROI224 image; no automatic image upload/crop")
    counts = [sum(sum(v) for v in t["tip_valid"]) for t in targets]
    fold_coverage = [{"heldout_episode": f["heldout_episode"],
                      "train_tip_points": sum(counts[i] for i in f["train_indices"]),
                      "test_tip_points": sum(counts[i] for i in f["test_indices"]),
                      "train_point_windows": sum(counts[i] > 0 for i in f["train_indices"]),
                      "test_point_windows": sum(counts[i] > 0 for i in f["test_indices"])} for f in folds]
    if any((args.points / p).read_bytes() != before for p, before in points_before.items()):
        raise ValueError("human point source changed during preparation")
    flags = {"training_executed": False, "optimizer_steps": 0, "policy_input_allowed": False,
             "formal_data_allowed": False, "deployable": False, "real_system_validated": False}
    protocol = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), **flags,
                "reference": args.reference.as_posix(), "image_pack": args.image_pack.as_posix(),
                "points": args.points.as_posix(), "roi_xyxy_original": ROI,
                "arms": {"spatial_window": 0.0, "spatial_window_tip_aux": 0.1},
                "encoder": "ImageNet ResNet18 frozen through layer1: 64x56x56; no new weights",
                "location": "shared spatial attention; 8 appearance + 2 UV coordinates per view",
                "target": "known tip only; one-cell Gaussian coordinate encoding, NOT measured uncertainty",
                "loss": "window balanced BCE + lambda * masked per-window-mean normalized heatmap CE",
                "model_input_whitelist": ["decoded ROI224 pixels", "task_right", "observed dt_s"],
                "target_only": ["response_y", "tip_xy_uv", "tip_valid"],
                "audit_only": ["point_audit.jsonl", "human notes/status/identity", "vessel_reference_snapshot.json"],
                "recognition_not_prediction": "future window frames allowed only for offline post-action recognition",
                "no_new_labels": "unknown is unsupervised, material points are not tip labels; no frame response labels",
                "reference_scope": "four-point vessel reference remains six-case diagnostic-only; NOT input or loss",
                "evaluation": "same reused development folds; whole-episode point isolation; no independent test",
                "training_runner_implemented": False, "next_training_requires_separate_execution": True}
    report = {"schema": SCHEMA, "status": "prepared", **flags, "windows": len(rows), "episodes": len(folds),
              "unique_image_references": len(images), "images_decoded_in_build": 0,
              "tip_points": sum(counts), "tip_supervised_windows": sum(c > 0 for c in counts),
              "tip_point_episodes": len({r["source_episode"] for r in audits if r["tip_target_valid"]}),
              "point_status_counts": dict(Counter(r["wire"]["status"] for r in audits)),
              "material_points_excluded_from_tip_loss": sum(r["feature_type"] == "distinct_shaft_landmark"
                                                            and r["wire"]["status"] == "visible" for r in audits),
              "fold_coverage": fold_coverage, "sources_unchanged": True,
              "whole_episode_folds_unchanged": True, "elapsed_s": time.perf_counter()-started}
    args.out.mkdir(parents=True)
    for name, data in (("model_inputs", inputs), ("supervision", targets), ("point_audit", audits)):
        write_jsonl(args.out / f"{name}.jsonl", data)
    for name, data in (("protocol", protocol), ("report", report), ("folds", folds),
                       ("vessel_reference_snapshot", store.reference)):
        write_json(args.out / f"{name}.json", data)
    for filename in ("annotation_snapshot.jsonl",):
        shutil.copyfile(args.reference / filename, args.out / filename)
    shutil.copyfile(PROTOCOL, args.out / "design_protocol.md")
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    shutil.copyfile(Path(__file__).with_name("real10_spatial_response.py"), args.out / "model_snapshot.py")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def preflight(args):
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18
    from real10_spatial_response import ARMS, SpatialResponse, point_distribution, response_losses, spatial_grid

    if (args.out / "preflight.json").exists():
        raise FileExistsError("preflight already recorded; do not silently repeat")
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = read_jsonl(args.out / "annotation_snapshot.jsonl")
    all_inputs = read_jsonl(args.out / "model_inputs.jsonl")
    all_targets = read_jsonl(args.out / "supervision.jsonl")
    chosen = [i for i, r in enumerate(source) if r["ui_index"] in CASES]
    inputs = [all_inputs[i]["model_input"] for i in chosen]
    targets = [all_targets[i] for i in chosen]
    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"cached weights required; no download: {checkpoint}")
    cnn = resnet18(weights=None)
    cnn.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    encoder = torch.nn.Sequential(cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1)
    encoder.requires_grad_(False).eval().to(args.device)
    paths = sorted({p for r in inputs for f in r["sequence"] for p in f["images"].values()})
    cache = {}
    mean = torch.tensor([.485, .456, .406], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=args.device).view(1, 3, 1, 1)
    with torch.no_grad():
        for start in range(0, len(paths), 16):
            batch = paths[start:start+16]
            pixels = []
            for path in batch:
                with Image.open(path) as im:
                    if im.size != (224, 224):
                        raise ValueError("use unchanged ROI224 images")
                    pixels.append(torch.from_numpy(np.array(im.convert("RGB"), copy=True)).permute(2, 0, 1))
            x = torch.stack(pixels).to(device=args.device, dtype=torch.float32) / 255
            features = encoder((x-mean)/std)
            cache.update({p: f for p, f in zip(batch, features)})
    n, t = len(inputs), max(len(r["sequence"]) for r in inputs)
    maps = torch.zeros(n, t, 2, 64, 56, 56, device=args.device)
    dt = torch.zeros(n, t-1, device=args.device)
    valid = torch.zeros(n, t-1, dtype=torch.bool, device=args.device)
    xy = torch.zeros(n, t, 2, 2, device=args.device)
    tip = torch.zeros(n, t, 2, dtype=torch.bool, device=args.device)
    for i, (r, target) in enumerate(zip(inputs, targets)):
        length = len(r["sequence"])
        dt[i, :length-1] = torch.tensor(r["dt_s"], device=args.device)
        valid[i, :length-1] = True
        for j, frame in enumerate(r["sequence"]):
            for v, view in enumerate(VIEWS):
                maps[i, j, v] = cache[frame["images"][view]]
                if target["tip_valid"][j][v]:
                    xy[i, j, v] = torch.tensor(target["tip_xy_uv"][j][v], device=args.device)
                    tip[i, j, v] = True
    task = torch.tensor([r["task_right"] for r in inputs], device=args.device)
    y = torch.tensor([r["response_y"] for r in targets], dtype=torch.float32, device=args.device)
    heldout = next(r["source_episode"] for r in source if r["ui_index"] == 65)
    fold = next(f for f in read_json(args.out / "folds.json") if f["heldout_episode"] == heldout)
    train = torch.tensor([i in fold["train_indices"] for i in chosen], device=args.device)
    train_y = [all_targets[i]["response_y"] for i in fold["train_indices"]]
    pos_weight = torch.tensor(train_y.count(0)/train_y.count(1), device=args.device)
    model = SpatialResponse().to(args.device)
    other = SpatialResponse().to(args.device)
    other.load_state_dict(model.state_dict())
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = model(maps, dt, task, valid)
    with torch.no_grad():
        matched = other(maps, dt, task, valid)
    torch.testing.assert_close(out["window_logit"], matched["window_logit"], rtol=0, atol=0)
    losses = {arm: response_losses(out, y, xy, tip, train, pos_weight, arm) for arm in ARMS}
    aux = losses["spatial_window_tip_aux"]
    aux_gradient = torch.autograd.grad(aux["localization"], out["heatmap_logits"], retain_graph=True)[0]
    invalid_max = float(aux_gradient[~aux["active_tip_mask"]].abs().max())
    assert invalid_max == 0 and torch.isfinite(aux_gradient).all()
    aux["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert not any(p.requires_grad or p.grad is not None for p in encoder.parameters())
    distribution = point_distribution(xy, tip)
    expected_xy = distribution.flatten(-2) @ spatial_grid(xy.device, xy.dtype).reshape(-1, 2)
    max_uv_error = float((expected_xy[tip]-xy[tip]).abs().max())
    c4, c77 = [source[i]["ui_index"] for i in chosen].index(4), [source[i]["ui_index"] for i in chosen].index(77)
    assert not tip[c4].any() and not tip[c77, 0, 0] and tip[c77, len(inputs[c77]["sequence"])-1, 0]
    report = {"schema": SCHEMA, "status": "preflight_completed", "training_executed": False,
              "optimizer_steps": 0, "checkpoint_saved": False, "policy_input_allowed": False,
              "examples": list(CASES), "real_images_encoded": len(paths), "feature_shape": list(maps.shape),
              "trainable_parameters_each_arm": sum(p.numel() for p in model.parameters()),
              "same_initialization_forward_max_diff": float((out["window_logit"]-matched["window_logit"]).detach().abs().max()),
              "heldout_episode_excluded_from_loss": heldout, "tip_targets_in_examples": int(tip.sum()),
              "tip_targets_in_training_loss": int(aux["active_tip_mask"].sum()),
              "invalid_or_heldout_aux_gradient_max": invalid_max, "parameters_unchanged_after_backward": True,
              "encoder_frozen_eval": True, "all_model_gradients_finite": True,
              "material_points_excluded": True, "missing_anchor_not_filled_from_later_point": True,
              "coordinate_encoding_max_uv_mean_error": max_uv_error,
              "untrained_loss_diagnostic_only": {a: {k: float(v[k].detach()) for k in ("total", "response", "localization")}
                                                 for a, v in losses.items()},
              "performance_metrics": None, "elapsed_s": time.perf_counter()-started}
    write_json(args.out / "preflight.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def render(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from PIL import Image

    font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "svg.fonttype": "path"})
    audits = read_jsonl(args.out / "point_audit.jsonl")
    report = read_json(args.out / "report.json")
    source = {r["ui_index"]: r for r in read_jsonl(args.out / "annotation_snapshot.jsonl")}
    inputs = {r["window_id"]: r["model_input"] for r in read_jsonl(args.out / "model_inputs.jsonl")}
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    selection = [(4, "top"), (65, "top"), (77, "side")]
    data = {"real_or_mock": "real human annotations; deterministic coordinate mapping, no predictions",
            "cases": selection, "point_counts": {k: report[k] for k in
                ("tip_points", "tip_supervised_windows", "material_points_excluded_from_tip_loss")}}
    write_json(figures / "supervision_coverage_data.json", data)
    (figures / "data-manifest.md").write_text(
        "| Figure | Data | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n"
        "| 空间辅助监督覆盖 | supervision_coverage_data.json + ../point_audit.jsonl | real/derived | "
        "latest human point annotations and existing ROI224 images | tools/prepare_real10_spatial_aux.py --stage render | "
        "spatial_supervision_zh.png / .svg |\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(10, 7.4))
    colors = {"tip": "#009988", "material": "#EE7733"}
    for column, (ui, view) in enumerate(selection):
        row = source[ui]
        sequence = inputs[row["window_id"]]["sequence"]
        for ri, index in enumerate((0, len(sequence)-1)):
            ax = axes[ri, column]
            a = next(p for p in audits if p["ui_index"] == ui and p["view"] == view and p["frame_index"] == index)
            with Image.open(sequence[index]["images"][view]) as im:
                ax.imshow(np.asarray(im.convert("RGB")), origin="upper")
            if a["wire"]["xy"] is not None:
                uv = point_uv(a["wire"]["xy"], view)
                px = np.array(uv)*224-.5
                color = colors["tip" if a["tip_target_valid"] else "material"]
                ax.scatter(*px, s=95, facecolors="none", edgecolors=color, linewidths=1.8)
                label = "尖端：有定位监督" if a["tip_target_valid"] else "材料点：不作尖端监督"
            else:
                label = "原状态未标：不产生负标签"
            ax.set_title(f"#{ui} {view.upper()} · {'首帧' if ri == 0 else '末帧'}\n{label}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("空间辅助监督：人工点只进入损失，不进入模型输入", fontsize=14, y=.99)
    fig.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=colors["tip"], label="已标尖端"),
                        Line2D([0], [0], marker="o", linestyle="", color=colors["material"], label="已标材料点（排除）")],
               loc="lower center", bbox_to_anchor=(.5, .086), ncol=2, frameon=False)
    fig.text(.5, .058, f"45窗／10轨迹保持；{report['tip_points']}个尖端点覆盖{report['tip_supervised_windows']}窗；4个材料点不进入尖端损失。",
             ha="center", fontsize=10)
    fig.text(.5, .028, "#77首帧缺失保持缺失，末帧可单独监督。上述圆圈是人工标注，不是模型预测。", ha="center", fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.17, hspace=.32, wspace=.13)
    for extension in ("png", "svg"):
        fig.savefig(figures / f"spatial_supervision_zh.{extension}", dpi=450)
    plt.close(fig)
    (args.out / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>空间辅助监督准备</title>'
        '<body style="font:18px system-ui;max-width:1100px;margin:24px auto">'
        '<h1>空间辅助监督准备 · 未训练</h1><p>只有人工点目标和零优化步接口验证；没有识别准确率或控制收益。</p>'
        '<img style="width:100%" src="figures/spatial_supervision_zh.png"></body></html>', encoding="utf-8")
    print(figures / "spatial_supervision_zh.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("build", "preflight", "render"), required=True)
    p.add_argument("--reference", type=Path, default=Path("simulation_output/real10_window_response_mil_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--points", type=Path, default=Path("simulation_output/real10_wire_correspondence_v1"))
    p.add_argument("--image-pack", type=Path, default=Path("simulation_output/real10_response_roi_images_v1"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_spatial_aux_v1"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    {"build": build, "preflight": preflight, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
