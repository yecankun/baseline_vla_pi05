"""Weak window-label response encoding; no tracking, policy or hardware changes.

Frozen ROI ResNet18 features, matched mean/logmeanexp pooling, whole-episode OOF.
Only pooled window logits receive labels; per-transition logits are NOT labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import time

import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_baseline import (
    LABELS, SEED, embed_images, fit_ridge, folds_for, image_path, metrics, predict_ridge,
)


SCHEMA = "real10_window_response_mil_v1"
ARMS = ("mean_pool", "logmeanexp_pool")
NAMES = {"mean_pool": "平均汇聚", "logmeanexp_pool": "突变偏重汇聚",
         "controller_ridge": "控制器参照", "always_advance": "总预测推进", "train_majority": "训练折多数类"}
LABEL_ZH = {"stationary": "无明显推进", "advance": "推进"}
PARAMETERS = {"steps": 200, "lr": .001, "weight_decay": .01, "gradient_clip": 1.,
              "projection_width": 8, "hidden_width": 16, "standardized_clip": 5.,
              "logmeanexp_temperature": 1., "threshold": .5, "seed": SEED}


def make_model():
    import torch
    from torch import nn

    class WindowResponse(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(512, 8), nn.GELU())
            self.score = nn.Sequential(nn.Linear(50, 16), nn.GELU(), nn.Linear(16, 1))

        def forward(self, features, dt, task, mask, pooling):
            z = self.project(features).flatten(2)  # N,T,2*8; shared across cameras
            current, previous = z[:, 1:], z[:, :-1]
            anchor = z[:, :1].expand_as(current)
            context = torch.stack((task[:, None].expand_as(dt), dt), dim=-1)
            logits = self.score(torch.cat((anchor, current-anchor, current-previous, context), dim=-1)).squeeze(-1)
            return pool_logits(logits, mask, pooling), logits

    return WindowResponse()


def pool_logits(logits, mask, pooling):
    import torch
    count = mask.sum(1).clamp_min(1)
    if pooling == ARMS[0]:
        return logits.masked_fill(~mask, 0).sum(1)/count
    if pooling == ARMS[1]:
        return torch.logsumexp(logits.masked_fill(~mask, -torch.inf), dim=1)-torch.log(count)
    raise ValueError(pooling)


def check():
    import torch
    torch.manual_seed(SEED)
    scores = torch.tensor([[.7, .7, 900.], [-.3, -.3, -.3]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    for arm in ARMS:
        value = pool_logits(scores, mask, arm)
        torch.testing.assert_close(value, torch.tensor([.7, -.3]))
        padded = torch.cat((scores, torch.full((2, 2), -999.)), dim=1)
        extra = torch.cat((mask, torch.zeros((2, 2), dtype=torch.bool)), dim=1)
        torch.testing.assert_close(value, pool_logits(padded, extra, arm))
        value.sum().backward(retain_graph=True)
    assert torch.isfinite(scores.grad).all() and scores.grad[0, 2] == 0
    model = make_model()
    x = torch.randn(3, 7, 2, 512)
    output, local = model(x, torch.ones(3, 6), torch.zeros(3), torch.ones(3, 6, dtype=torch.bool), ARMS[0])
    output.sum().backward()
    assert output.shape == (3,) and local.shape == (3, 6)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    return {"padding_and_constant_pooling": True, "finite_gradient": True,
            "parameters": sum(p.numel() for p in model.parameters()), "real_accuracy_claim": False}


def prepare(args):
    if args.out.exists():
        raise FileExistsError("preserve existing/partial output; use a fresh --out")
    rows = read_jsonl(args.reference / "annotation_snapshot.jsonl")
    folds = read_json(args.reference / "folds.json")
    current = {r["window_id"]: r for r in read_jsonl(args.pack / "annotations_joint_v2.jsonl")}
    frozen = {r["window_id"]: r["human_annotation"] for r in rows}
    if len(rows) != 45 or len(folds) != 10 or folds != folds_for(rows) or frozen != current:
        raise ValueError("the 45 revised human windows / original episode folds changed")
    original = read_jsonl(args.baseline / "annotation_snapshot.jsonl")
    roi_rows = read_jsonl(args.image_pack / "annotation_snapshot.jsonl")
    for other in (original, roi_rows):
        if len(other) != len(rows) or any(any(a[k] != b[k] for k in (
                "window_id", "source_episode", "task", "anchor_frame_id", "future_frame_ids")) for a, b in zip(rows, other)):
            raise ValueError("cached controller / ROI input window identity changed")
    roi_protocol = read_json(args.image_pack / "protocol.json")
    expected_roi = {"side": [0, 540, 1440, 1040], "top": [480, 560, 1920, 1080]}
    if roi_protocol["roi_xyxy_original"] != expected_roi or not (args.image_pack / "crop_completed.json").exists():
        raise ValueError("expected completed unchanged scene ROI pack")
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    shutil.copyfile(Path("docs/algorithm-real10-window-response-protocol-20260922.md"), args.out / "design_protocol.md")
    protocol = {"schema": SCHEMA, "frozen_before_features_utc": datetime.now(timezone.utc).isoformat(),
                "parameters": PARAMETERS, "arms": list(ARMS), "samples": len(rows), "episodes": len(folds),
                "label_counts": dict(Counter(r["human_annotation"]["joint_motion_response"] for r in rows)),
                "reference": str(args.reference), "image_pack": str(args.image_pack), "roi_xyxy_original": expected_roi,
                "input": "only frozen image sequence, task_right and relative observation availability dt",
                "supervision": "human full-window advance/stationary; no transition labels or tracked pseudo-targets",
                "future_images": "allowed ONLY for this offline post-action recognition, never pre-action prediction",
                "normalization": "training-fold valid frame slots only, per camera, clip5; same transform for both arms",
                "fit": "4937 parameters; balanced BCE, final 200 full-batch updates; matched fold initialization; no tuning",
                "references": "same-current-label controller ridge is contextual only, different input modality; majority and always-advance",
                "diagnostic": "repeat anchor image features through clip with original dt/task, frozen model; OOD sensitivity only",
                "forbidden": ["human point/visibility/evidence metadata", "tracking/flow predictions", "contact/wall/tip/route truth", "event IDs"],
                "policy_input_allowed": False, "formal_data_allowed": False, "deployable": False,
                "PI05_changed": False, "real_system_validated": False, "independent_test": False}
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "folds.json", folds)
    write_jsonl(args.out / "annotation_snapshot.jsonl", rows)
    start = time.perf_counter()
    embeddings, encoder = embed_images(rows, args.image_pack, args.device, 64)
    observations = {r["frame_id"]: r for r in read_jsonl(args.pack / "observations.jsonl")}
    lengths = np.array([1+len(r["future_frame_ids"]) for r in rows])
    features = np.zeros((len(rows), int(lengths.max()), 2, 512), dtype=np.float32)
    dt = np.zeros((len(rows), int(lengths.max())-1), dtype=np.float32)
    ids = []
    for i, row in enumerate(rows):
        frames = [row["anchor_frame_id"], *row["future_frame_ids"]]
        ids.append(frames)
        for j, frame in enumerate(frames):
            features[i, j] = np.stack([embeddings[frame, view] for view in ("side", "top")])
        dt[i, :lengths[i]-1] = np.diff([observations[f]["observation_available_at_s"] for f in frames])
        if np.any(dt[i, :lengths[i]-1] <= 0):
            raise ValueError("nonpositive observation dt")
    with np.load(args.baseline / "features.npz", allow_pickle=False) as cache:
        controller = cache["controller_only"].copy()
    y = np.array([LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])
    task = np.array([r["task"] == "right" for r in rows], dtype=np.float32)
    np.savez_compressed(args.out / "features.npz", features=features, dt=dt, lengths=lengths,
                        task=task, y=y, controller=controller)
    write_json(args.out / "prepared.json", {"schema": SCHEMA, "encoder": encoder, "frame_ids": ids,
               "lengths": lengths.tolist(), "feature_shape": list(features.shape),
               "seconds": time.perf_counter()-start, "frames_interpolated": False})


def normalize(features, dt, lengths, train):
    valid_frames = np.arange(features.shape[1])[None, :] < lengths[:, None]
    values = features[train][valid_frames[train]]
    mean, std = values.mean(0), values.std(0)
    std[std < 1e-6] = 1.
    transitions = np.arange(dt.shape[1])[None, :] < lengths[:, None]-1
    actual_dt = dt[train][transitions[train]]
    dt_mean, dt_std = float(actual_dt.mean()), float(actual_dt.std())
    if dt_std < 1e-6:
        dt_std = 1.
    transformed = np.clip((features-mean)/std, -5, 5)
    normalized_dt = np.clip((dt-dt_mean)/dt_std, -5, 5)
    return transformed, normalized_dt, transitions, {"mean": mean, "std": std, "dt_mean": dt_mean, "dt_std": dt_std}


def grouped(y, score, rows):
    pred = (score >= .5).astype(np.int64)
    result = {"pooled": metrics(y, pred)}
    for key in ("source_episode", "task"):
        result[key] = {}
        for value in sorted({r[key] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[key] == value]
            result[key][value] = metrics(y[idx], pred[idx])
    result["episode_macro_accuracy"] = float(np.mean([m["accuracy"] for m in result["source_episode"].values()]))
    return result


def train(args, *, base_seed=SEED):
    import torch
    from torch.nn import functional as F
    if (args.out / "fold_models").exists() or (args.out / "report.json").exists():
        raise FileExistsError("preserve completed/partial fit; choose new --out, not a new checkpoint")
    protocol = read_json(args.out / "protocol.json")
    parameters = {**PARAMETERS, "seed": int(base_seed)}
    if protocol["parameters"] != parameters or protocol["schema"] != SCHEMA:
        raise ValueError("frozen protocol differs from entrypoint")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    folds = read_json(args.out / "folds.json")
    read_json(args.out / "prepared.json")
    with np.load(args.out / "features.npz", allow_pickle=False) as data:
        cache = {k: data[k] for k in data.files}
    x, dt, lengths, y = (cache[k] for k in ("features", "dt", "lengths", "y"))
    expected_y = np.array([LABELS.index(r["human_annotation"]["joint_motion_response"]) for r in rows])
    if not np.array_equal(y, expected_y) or folds != folds_for(rows) or not np.isfinite(x).all():
        raise ValueError("cached row/label/fold identity or finite image features failed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    scores = {arm: np.full(len(y), np.nan) for arm in (*ARMS, "controller_ridge", "train_majority", "always_advance")}
    scores["always_advance"][:] = 1.
    repeated = {arm: np.full(len(y), np.nan) for arm in ARMS}
    evidence = {arm: [None]*len(y) for arm in ARMS}
    logs, coverage = [], np.zeros(len(y), dtype=int)
    for number, fold in enumerate(folds):
        fold_start = time.perf_counter()
        tr, te = np.array(fold["train_indices"]), np.array(fold["test_indices"])
        coverage[te] += 1
        normalized, normalized_dt, mask, norm = normalize(x, dt, lengths, tr)
        tx, td, tm, task = [torch.as_tensor(a, device=args.device) for a in
                           (normalized, normalized_dt, mask, cache["task"])]
        train_idx = torch.as_tensor(tr, device=args.device)
        test_idx = torch.as_tensor(te, device=args.device)
        targets = torch.as_tensor(y[tr], dtype=torch.float32, device=args.device)
        counts = np.bincount(y[tr], minlength=2)
        weights = torch.as_tensor(len(tr)/(2.*counts[y[tr]]), dtype=torch.float32, device=args.device)
        initial = None
        for arm in ARMS:
            torch.manual_seed(base_seed+number)
            model = make_model().to(args.device)
            init = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if initial is None:
                initial = init
            elif not all(torch.equal(initial[k], v) for k, v in init.items()):
                raise ValueError("paired model initialization mismatch")
            optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMETERS["lr"], weight_decay=PARAMETERS["weight_decay"])
            model.train()
            for step in range(PARAMETERS["steps"]):
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(tx[train_idx], td[train_idx], task[train_idx], tm[train_idx], arm)
                loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none")*weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), PARAMETERS["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                training_logits, _ = model(tx[train_idx], td[train_idx], task[train_idx], tm[train_idx], arm)
                training_score = torch.sigmoid(training_logits).cpu().numpy()
                expected, _ = model(tx[test_idx], td[test_idx], task[test_idx], tm[test_idx], arm)
            destination = args.out / "fold_models" / arm / f"fold_{number:02d}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema": SCHEMA, "pooling": arm, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "normalization": {k: torch.as_tensor(v) for k, v in norm.items()},
                        "heldout_episode": fold["heldout_episode"], "parameters": parameters,
                        "train_class_counts": counts.tolist(), "deployable": False}, destination)
            saved = torch.load(destination, map_location=args.device, weights_only=True)
            restored = make_model().to(args.device).eval()
            restored.load_state_dict(saved["state_dict"], strict=True)
            with torch.inference_mode():
                output, local = restored(tx[test_idx], td[test_idx], task[test_idx], tm[test_idx], arm)
                replay_error = float((output-expected).abs().max().item())
                if replay_error > 1e-6:
                    raise ValueError("saved model replay mismatch")
                static = tx[test_idx, :1].expand(-1, tx.shape[1], -1, -1)
                static_output, _ = restored(static, td[test_idx], task[test_idx], tm[test_idx], arm)
                scores[arm][te] = torch.sigmoid(output).cpu().numpy()
                repeated[arm][te] = torch.sigmoid(static_output).cpu().numpy()
                local_values = local.cpu().numpy()
                for k, i in enumerate(te):
                    evidence[arm][i] = local_values[k, :lengths[i]-1].tolist()
            logs.append({"fold": number, "heldout_episode": fold["heldout_episode"], "arm": arm,
                         "train_count": len(tr), "test_count": len(te), "seed": base_seed+number,
                         "parameters": sum(p.numel() for p in model.parameters()), "updates": PARAMETERS["steps"],
                         "final_train_loss": float(loss.detach().item()),
                         "train_metrics": metrics(y[tr], (training_score >= .5).astype(int)),
                         "same_pair_initialization": True, "replay_max_logit_error": replay_error})
        reference = fit_ridge(cache["controller"][tr], y[tr], alpha=1., block_sizes=[40], standardized_clip=5.)
        scores["controller_ridge"][te] = predict_ridge(reference, cache["controller"][te])
        reference_dir = args.out / "fold_models/controller_ridge"
        reference_dir.mkdir(exist_ok=True)
        np.savez_compressed(reference_dir / f"fold_{number:02d}.npz", **reference)
        scores["train_majority"][te] = int(counts[1] >= counts[0])
        print(json.dumps({"fold": number+1, "of": len(folds), "seconds": time.perf_counter()-fold_start}), flush=True)
    if not np.all(coverage == 1) or any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError("incomplete/nonfinite OOF predictions")
    summaries = {arm: grouped(y, values, rows) for arm, values in scores.items()}
    pair = {key: 100*(summaries[ARMS[1]]["pooled"][key]-summaries[ARMS[0]]["pooled"][key])
            for key in ("accuracy", "balanced_accuracy")}
    correct = [(scores[arm] >= .5) == y for arm in ARMS]
    improvements = [int(rows[i]["ui_index"]) for i in np.flatnonzero(~correct[0] & correct[1])]
    regressions = [int(rows[i]["ui_index"]) for i in np.flatnonzero(correct[0] & ~correct[1])]
    episode_change = {e: summaries[ARMS[1]]["source_episode"][e]["accuracy"]-summaries[ARMS[0]]["source_episode"][e]["accuracy"]
                      for e in summaries[ARMS[0]]["source_episode"]}
    ablation = {arm: {"metrics": grouped(y, repeated[arm], rows),
                     "changed_predictions": int(np.sum((scores[arm] >= .5) != (repeated[arm] >= .5))),
                     "mean_absolute_score_change": float(np.abs(scores[arm]-repeated[arm]).mean())} for arm in ARMS}
    predictions = []
    for i, row in enumerate(rows):
        item = {"sample_index": i, "window_id": row["window_id"], "ui_index": row["ui_index"],
                "episode": row["source_episode"], "task": row["task"], "label": LABELS[y[i]], "methods": {}}
        for arm in scores:
            prediction = {"score": float(scores[arm][i]), "prediction": LABELS[int(scores[arm][i] >= .5)]}
            if arm in ARMS:
                prediction.update(repeated_anchor_score=float(repeated[arm][i]), transition_logits=evidence[arm][i],
                                  largest_logit_target_frame_index=int(np.argmax(evidence[arm][i]))+1)
            item["methods"][arm] = prediction
        predictions.append(item)
    result = {"schema": SCHEMA, "status": "completed", "methods": summaries,
              "paired_logmeanexp_minus_mean_pp": pair, "improved_ui_indices": improvements, "worsened_ui_indices": regressions,
              "episode_accuracy_change": episode_change,
              "episode_win_tie_loss": [sum(v > 0 for v in episode_change.values()), sum(v == 0 for v in episode_change.values()), sum(v < 0 for v in episode_change.values())],
              "repeat_anchor_diagnostic": ablation,
              "mean_training_accuracy_by_arm": {arm: float(np.mean([r["train_metrics"]["accuracy"] for r in logs if r["arm"] == arm])) for arm in ARMS},
              "fit_and_evaluate_seconds": time.perf_counter()-started, "oof_once_per_window": True,
              "total_neural_updates": len(folds)*len(ARMS)*PARAMETERS["steps"], "trainable_parameters_per_model": 4937,
              "device": args.device, "torch_version": torch.__version__, "visual_status": "not_viewed",
              "policy_input_allowed": False, "deployable": False, "formal_data_allowed": False, "real_system_validated": False}
    write_jsonl(args.out / "oof_predictions.jsonl", predictions)
    write_jsonl(args.out / "fold_log.jsonl", logs)
    write_json(args.out / "report.json", result)
    print(json.dumps({"status": "completed", "seconds": result["fit_and_evaluate_seconds"],
                      "methods": {a: s["pooled"] for a, s in summaries.items()}}, ensure_ascii=False), flush=True)


def render(args):
    from PIL import Image, ImageDraw, ImageFont
    report = read_json(args.out / "report.json")
    rows = read_jsonl(args.out / "annotation_snapshot.jsonl")
    predictions = read_jsonl(args.out / "oof_predictions.jsonl")
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)
    title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
    canvas = Image.new("RGB", (1420, 940), "#f4f6f9")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 22), "人工窗口监督：平均汇聚 vs 突变偏重汇聚", font=title, fill="#173049")
    draw.text((30, 74), "45个人工窗口 / 10条轨迹留一；同输入、同初始化、4937可训参数、每折200步；非独立测试", font=font, fill="#34495d")
    for x, heading in ((30, "方法"), (390, "准确率"), (590, "平衡准确率"),
                       (820, "轨迹宏准确率"), (1080, "混淆矩阵[无推进, 推进]")):
        draw.text((x, 120), heading, font=font, fill="#173049")
    lines = ["# 人工窗口监督响应编码器", "", "45窗/10折开发诊断，仅动作后识别，不是动作前世界模型或接触估计。", "",
             "| 方法 | 准确率 | BA | 轨迹宏准确率 | 混淆矩阵 |", "|---|---:|---:|---:|---|"]
    for j, arm in enumerate((*ARMS, "controller_ridge", "always_advance", "train_majority")):
        m = report["methods"][arm]
        pooled = m["pooled"]
        matrix = pooled["confusion_matrix_true_rows_predicted_columns"]
        y = 178+j*65
        for x, value in ((30, NAMES[arm]), (390, f"{pooled['accuracy']:.2%}"), (590, f"{pooled['balanced_accuracy']:.2%}"),
                         (820, f"{m['episode_macro_accuracy']:.2%}"), (1080, str(matrix))):
            draw.text((x, y), value, font=font, fill="#173049")
        lines.append(f"| {NAMES[arm]} | {pooled['accuracy']:.2%} | {pooled['balanced_accuracy']:.2%} | {m['episode_macro_accuracy']:.2%} | {matrix} |")
    delta = report["paired_logmeanexp_minus_mean_pp"]
    details = [f"突变汇聚−平均汇聚：准确率 {delta['accuracy']:+.2f} pp；BA {delta['balanced_accuracy']:+.2f} pp",
               f"改对窗口 {report['improved_ui_indices']}；改错窗口 {report['worsened_ui_indices']}；轨迹胜/平/负 {report['episode_win_tie_loss']}"]
    for arm in ARMS:
        ab = report["repeat_anchor_diagnostic"][arm]
        details.append(f"{NAMES[arm]}：平均训练准确率 {report['mean_training_accuracy_by_arm'][arm]:.2%}；重复首帧后BA {ab['metrics']['pooled']['balanced_accuracy']:.2%}；{ab['changed_predictions']}窗改判")
    details += ["重复首帧为分布外敏感性检查；帧对logit不是推进时刻、导丝轨迹或校准概率。",
                "控制器参照使用实测状态，输入不同；不能将差异归因于模型架构。",
                "未改标签/PI05/控制器；不把未来画面送入动作前决策；结果不直接用于部署。"]
    for j, text in enumerate(details):
        draw.text((30, 535+51*j), text, font=small, fill="#34495d")
    canvas.save(args.out / "comparison_zh.png")
    lines += ["", *details, ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    by_ui = {r["ui_index"]: (r, p) for r, p in zip(rows, predictions)}
    hard = Image.new("RGB", (1500, 1080), "#f4f6f9")
    hd = ImageDraw.Draw(hard)
    hd.text((25, 16), "固定六个复核窗：原标签不变 / 模型输出不是重新标注", font=title, fill="#173049")
    for j, ui in enumerate((4, 39, 62, 65, 77, 100)):
        r, p = by_ui[ui]
        x, y = 25+(j % 3)*490, 80+(j // 3)*490
        hd.text((x, y), f"#{ui}  人工：{LABEL_ZH[p['label']]}", font=font, fill="#173049")
        for a, arm in enumerate(ARMS):
            result = p["methods"][arm]
            hd.text((x, y+34+28*a), f"{NAMES[arm]}：{LABEL_ZH[result['prediction']]} / {result['score']:.3f}", font=small, fill="#34495d")
        for v, view in enumerate(("side", "top")):
            for f, frame in enumerate((r["anchor_frame_id"], r["future_frame_ids"][-1])):
                path = image_path(args.image_pack, frame, view)
                with Image.open(path) as im:
                    tile = Image.new("RGB", (200, 160), "#f4f6f9")
                    tile.paste(im.convert("RGB").resize((160, 160)), (20, 0))
                px, py = x+f*225, y+116+v*187
                hd.text((px, py-24), f"{view} / {'首帧' if f == 0 else '末帧'}", font=small, fill="#34495d")
                hard.paste(tile, (px, py))
    hd.text((25, 1040), "已有固定ROI缩略图；不用于重判人工标签。两组只改变窗口汇聚，均可看到整个响应窗口。", font=small, fill="#34495d")
    hard.save(args.out / "hardcases_zh.png")
    table = ['<table><tr><th>窗口</th><th>任务/人工</th><th>平均汇聚</th><th>突变汇聚</th></tr>']
    for p in predictions:
        table.append(f'<tr><td>{p["ui_index"]}</td><td>{html.escape(p["task"])} / {LABEL_ZH[p["label"]]}</td>' + ''.join(
            f'<td>{LABEL_ZH[p["methods"][a]["prediction"]]} ({p["methods"][a]["score"]:.3f})</td>' for a in ARMS) + '</tr>')
    table.append('</table>')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>窗口响应编码器</title><style>body{font:18px system-ui;max-width:1500px;margin:24px auto;background:#f4f6f9}img{width:100%}td,th{padding:8px;border:1px solid #ccd}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
    (args.out / "index.html").write_text(page+'<img src="comparison_zh.png"><img src="hardcases_zh.png">'+''.join(table), encoding="utf-8")
    print(json.dumps({"rendered": ["comparison_zh.png", "hardcases_zh.png", "index.html"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("check", "prepare", "train", "all", "render"), required=True)
    p.add_argument("--reference", type=Path, default=Path("simulation_output/real10_pre_action_label_retest_v1"))
    p.add_argument("--pack", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--baseline", type=Path, default=Path("simulation_output/real10_response_baseline_v1"))
    p.add_argument("--image-pack", type=Path, default=Path("simulation_output/real10_response_roi_images_v1"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_window_response_mil_v1"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = p.parse_args()
    if args.stage == "check":
        print(json.dumps(check()))
    elif args.stage == "render":
        render(args)
    else:
        if args.stage in ("prepare", "all"):
            prepare(args)
        if args.stage in ("train", "all"):
            train(args)


if __name__ == "__main__":
    main()
