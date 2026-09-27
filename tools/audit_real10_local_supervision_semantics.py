"""Read-only pixel-support audit of frozen Real10 local-motion supervision.

No fit, mask adoption, label update, policy input or hardware call. Original RGB
is read on Windows; synchronize this final entrypoint before local extraction.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import html
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from prepare_real10_event_windows import read_json, read_jsonl, write_json, write_jsonl
from run_real10_response_local_motion import crop_original, pair_motion, PARAMETERS, VIEWS, GRID


SCHEMA = "real10_local_supervision_semantic_audit_v1"
HUMAN_WINDOWS = (4, 39, 62, 65, 77, 100)
# Existing review callouts, NOT apparatus segmentation or feature masks.
CALLOUTS = {39: ("side", [0, 0, 215, 92]), 65: ("top", [205, 25, 330, 115])}
HEAT_MAX_PX_S = 2.0  # display saturation only; all numeric values remain unclipped


def select_cases(args):
    obs = read_jsonl(args.targets / "observations.jsonl")
    future = read_jsonl(args.targets / "response_targets.jsonl")
    groups = read_jsonl(args.targets / "sample_groups.jsonl")
    indices = read_json(args.targets / "preview_indices.json")
    cases = []
    for i in indices:
        episode = groups[i]["source_episode"]
        frames = [obs[i]["model_observation"]["history"][-1], *future[i]["future_observations"]]
        refs = {v: [{"path": (Path(episode) / "frames" / v / Path(f["images"][v]["path"]).name).as_posix(),
                     "time_s": f["images"][v]["relative_time_s"]} for f in frames] for v in VIEWS}
        cases.append({"id": f"macro_{i + 1:03d}", "source_kind": "elite_macro",
            "sample_index": i, "source_episode": episode, "task": groups[i]["task"],
            "human_response": None, "human_notes": "未人工标注，不移植旧窗口标签", "frames": refs})
    rows = read_jsonl(args.labels / "annotation_snapshot.jsonl")
    observations = {r["frame_id"]: r for r in read_jsonl(args.windows / "observations.jsonl")}
    for number in HUMAN_WINDOWS:
        row = next(r for r in rows if r["ui_index"] == number)
        frames = [row["anchor_frame_id"], *row["future_frame_ids"]]
        refs = {v: [{"path": observations[f]["images"][v]["path"],
                    "time_s": observations[f]["images"][v]["recorded_timestamp_s"]} for f in frames] for v in VIEWS}
        cases.append({"id": f"human_{number:03d}", "source_kind": "human_window", "ui_index": number,
            "source_episode": row["source_episode"], "task": row["task"],
            "human_response": row["human_annotation"]["joint_motion_response"],
            "human_notes": row["human_annotation"]["notes"], "frames": refs})
    return cases


def cells(shape):
    h, w = shape
    return [np.s_[r*h//GRID[0]:(r+1)*h//GRID[0], c*w//GRID[1]:(c+1)*w//GRID[1]]
            for r in range(GRID[0]) for c in range(GRID[1])]


def list_or_null(a):
    return [float(x) if np.isfinite(x) else None for x in a]


def marked(rgb, interior, visual, dt, heat=False, box=None):
    result = rgb.copy()
    support = visual["support"]
    speed = np.linalg.norm(visual["residual_px"], axis=-1) / dt
    alpha = np.clip(speed / HEAT_MAX_PX_S, 0, 1) * .85 if heat else np.full(support.shape, .60)
    alpha *= support
    color = np.array([245, 55, 45] if heat else [0, 220, 240])
    result = (result * (1-alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)
    contours, _ = cv2.findContours(interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (245, 198, 30), 1)
    if box is not None:
        x0, y0, x1, y1 = box
        cv2.rectangle(result, (x0, y0), (x1-1, y1-1), (255, 90, 215), 2)
    return result


def render_case(out, case, previews, summaries, font_path):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 18)
    title = ImageFont.truetype(str(font_path), 28)
    response = {None: "未标注", "advance": "推进", "stationary": "未推进"}[case["human_response"]]
    name = f"人工窗口 #{case['ui_index']}" if case["source_kind"] == "human_window" else f"Elite宏请求 #{case['sample_index']+1}"
    canvas = Image.new("RGB", (1500, 1510), "#f6f8fb")
    d = ImageDraw.Draw(canvas)
    d.text((20, 12), f"{name}｜人工响应：{response}｜真实参与统计的像素", font=title, fill="#203247")
    d.text((20, 55), "青色=暗线支持；黄色=旧固定掩膜边界；红色深浅=该帧对的残余速度。", font=font, fill="#203247")
    d.text((20, 87), "全程未改掩膜。没有导丝身份跟踪、尖端定位或沿血管投影；中间帧对不代表整窗。", font=small, fill="#405568")
    for j, view in enumerate(VIEWS):
        p, summary = previews[view], summaries[view]
        y = 125 + j*650
        d.text((20, y), f"{view}｜整窗均格速度 {summary['mean_valid_cell_speed_px_s']:.4f} ROI px/s；"
               f"{summary['pair_count']}个相邻帧对；中间帧对 {p['from_name']} → {p['to_name']}", font=font, fill="#203247")
        panels = [("整窗起点 RGB", p["first"]), ("整窗终点 RGB", p["last"]),
                  ("中间帧对：实际暗线支持像素", p["support"]),
                  (f"同一帧对：残余速度（颜色封顶 {HEAT_MAX_PX_S:g} px/s）", p["heat"])]
        for k, (label, array) in enumerate(panels):
            x, py = 20 + (k % 2)*740, y+36+(k//2)*300
            d.text((x, py), label, font=small, fill="#203247")
            canvas.paste(Image.fromarray(array), (x, py+25))
    d.text((20, 1450), "紫框仅为既有器械邻近审查区，框内也可能含导丝/血管；不能当作器械分割或污染率。", font=small, fill="#405568")
    canvas.save(out / f"{case['id']}.png", dpi=(450, 450))


def run(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("preserve previous audit; use a fresh --out after any incomplete run")
    cases = select_cases(args)
    if read_json(args.targets / "protocol.json")["parameters"] != PARAMETERS:
        raise ValueError("frozen extractor parameters changed")
    args.out.mkdir(parents=True)
    shutil.copyfile(__file__, args.out / "entrypoint_snapshot.py")
    protocol = {"schema": SCHEMA, "targets": args.targets.as_posix(), "labels_snapshot": args.labels.as_posix(),
        "selection": "same ten median-per-episode macro previews, plus six preselected human-reviewed semantic cases",
        "human_windows": list(HUMAN_WINDOWS), "parameters": PARAMETERS,
        "pair_display_selection": "middle adjacent pair by index, never by flow strength or prediction error",
        "callouts_audit_only": {str(k): {"view": v, "xyxy": b} for k, (v, b) in CALLOUTS.items()},
        "callout_definition": "old manually chosen nearby-apparatus review boxes; may include wire and vessel; NOT object segmentation",
        "heat_saturation_px_s_display_only": HEAT_MAX_PX_S,
        "numeric_summary": "duration-weighted per-cell speed, then equal mean across cells valid throughout the window",
        "scope": "supervision semantics audit, not independent validation or classifier evaluation",
        "training_executed": False, "labels_modified": False, "masks_modified": False,
        "proposed_vessel_envelope_applied": False, "policy_input_allowed": False, "formal_data_allowed": False}
    write_json(args.out / "protocol.json", protocol)
    write_json(args.out / "selection.json", cases)
    with np.load(args.targets / "preview_arrays.npz", allow_pickle=False) as z:
        masks = {v: (z[f"interior_{v}"].copy(), z[f"rim_{v}"].copy()) for v in VIEWS}
    with np.load(args.targets / "local_targets.npz", allow_pickle=False) as z:
        frozen = z["targets"].copy().reshape(-1, 2, 12, 3)

    @lru_cache(maxsize=32)
    def rgb(path, view):
        return cv2.cvtColor(crop_original(cv2.imread(str(args.raw_root / path)), view), cv2.COLOR_BGR2RGB)

    summaries, pair_rows, matches, sections = [], [], [], []
    pixel_checks, total_pairs, box_checks = [], 0, []
    for case in cases:
        by_view, previews = {}, {}
        for vi, view in enumerate(VIEWS):
            refs, interior, rim = case["frames"][view], *masks[view]
            numbers, durations, box_parts, counts, reference_ok = [], [], [], [], []
            selected_pair = (len(refs)-2)//2
            callout = CALLOUTS.get(case.get("ui_index"))
            box = callout[1] if callout and callout[0] == view else None
            box_mask = np.zeros(interior.shape, bool)
            if box:
                x0, y0, x1, y1 = box
                box_mask[y0:y1, x0:x1] = True
            for pi, (a, b) in enumerate(zip(refs, refs[1:])):
                dt = b["time_s"]-a["time_s"]
                before, after = rgb(a["path"], view), rgb(b["path"], view)
                matrix, quality, visual = pair_motion(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(after, cv2.COLOR_RGB2GRAY), interior, rim, dt)
                numbers.append(matrix[:, :3])
                durations.append(dt)
                reference_ok.append(quality["valid"])
                total_pairs += 1
                pair_rows.append({"case": case["id"], "view": view, "from": a["path"], "to": b["path"], **quality})
                if visual is None:
                    raise ValueError("selected pair lacks a valid numeric reference; do not display as zero motion")
                support = visual["support"]
                speed = np.linalg.norm(visual["residual_px"], axis=-1) / dt
                inside, outside, support_counts = [], [], []
                for ci, region in enumerate(cells(interior.shape)):
                    active = support[region]
                    n = int(active.sum())
                    support_counts.append(n)
                    if np.isfinite(matrix[ci, 2]):
                        # Float32 arithmetic differs slightly from mean(displacement)/dt.
                        reconstructed = speed[region][active].mean()
                        pixel_checks.append(abs(float(reconstructed)-matrix[ci, 2]))
                        inside.append(float(speed[region][active & box_mask[region]].sum())/n)
                        outside.append(float(speed[region][active & ~box_mask[region]].sum())/n)
                    else:
                        inside.append(np.nan)
                        outside.append(np.nan)
                box_parts.append((inside, outside))
                counts.append(support_counts)
                if pi == selected_pair:
                    previews[view] = {"first": rgb(refs[0]["path"], view), "last": rgb(refs[-1]["path"], view),
                        "support": marked(before, interior, visual, dt, box=box),
                        "heat": marked(before, interior, visual, dt, heat=True, box=box),
                        "from_name": Path(a["path"]).stem, "to_name": Path(b["path"]).stem}
            values, dt = np.asarray(numbers), np.asarray(durations)
            valid = np.isfinite(values).all(axis=(0, 2))
            mean = np.sum(np.nan_to_num(values)*dt[:, None, None], axis=0)/dt.sum()
            mean[~valid] = np.nan
            summary = {"pair_count": len(dt), "valid_reference_pairs": sum(reference_ok),
                "observed_span_s": float(dt.sum()), "full_valid_cells": int(valid.sum()),
                "cell_speed_px_s": list_or_null(mean[:, 2]),
                "mean_valid_cell_speed_px_s": float(np.nanmean(mean[:, 2])),
                "mean_support_pixels_by_cell": np.mean(counts, axis=0).tolist(),
                "display_pair_index": selected_pair}
            if box:
                decomposition = np.sum(np.asarray(box_parts)*dt[:, None, None], axis=0)/dt.sum()
                denominator = float(mean[valid, 2].sum())
                error = float(np.max(np.abs(decomposition[:, valid].sum(axis=0)-mean[valid, 2])))
                box_checks.append(error)
                summary["callout_diagnostic"] = {"box_xyxy": box,
                    "share_of_summed_valid_cell_speed": float(decomposition[0, valid].sum()/denominator),
                    "decomposition_max_error_px_s": error,
                    "inside_cell_contribution_px_s": list_or_null(decomposition[0]),
                    "outside_cell_contribution_px_s": list_or_null(decomposition[1]),
                    "not_an_apparatus_contamination_fraction": True}
            if case["source_kind"] == "elite_macro":
                expected = frozen[case["sample_index"], vi]
                np.testing.assert_allclose(mean, expected, rtol=0, atol=1e-10, equal_nan=True)
                matches.append({"case": case["id"], "view": view, "max_abs_error": float(np.nanmax(np.abs(mean-expected)))})
            by_view[view] = summary
        summaries.append({"case": case["id"], "source_kind": case["source_kind"],
            "human_response": case["human_response"], "human_notes": case["human_notes"], "by_view": by_view})
        render_case(args.out, case, previews, by_view, args.font)
        sections.append(f'<section><h2>{html.escape(case["id"])}</h2><p>{html.escape(case["human_notes"])}</p>'
                        f'<a href="{case["id"]}.png"><img src="{case["id"]}.png"></a></section>')
        print(f"audit {case['id']} ({time.perf_counter()-started:.1f}s)", flush=True)
    if max(pixel_checks) > 2e-5 or max(box_checks) > 2e-5:
        raise ValueError("pixel-support reconstruction disagrees with frozen statistics")
    write_jsonl(args.out / "case_metrics.jsonl", summaries)
    write_jsonl(args.out / "pair_quality.jsonl", pair_rows)
    report = {"schema": SCHEMA, "cases": len(cases), "macro_cases": len(matches)//2,
        "human_cases": len(HUMAN_WINDOWS), "frame_pair_instances": total_pairs,
        "valid_reference_pairs": sum(r["valid"] for r in pair_rows),
        "macro_saved_target_matches": matches, "max_pixel_stat_reconstruction_error_px_s": max(pixel_checks),
        "max_callout_decomposition_error_px_s": max(box_checks),
        "human_cases_are_not_labels_for_macro_cases": True, "semantic_wire_truth_available": False,
        "wire_purity_or_recall_computable": False, "training_executed": False, "hardware_executed": False,
        "labels_modified": False, "masks_modified": False, "policy_input_allowed": False,
        "formal_data_allowed": False, "real_system_validated": False, "visual_status": "not_viewed",
        "run_seconds": time.perf_counter()-started}
    write_json(args.out / "report.json", report)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>局部监督语义审计</title>'
    page += '<style>body{font:18px sans-serif;max-width:1500px;margin:24px auto;color:#203247}img{width:100%}section{margin:28px 0;border-top:1px solid #ccd6df}</style>'
    page += '<h1>局部监督究竟统计了什么？</h1><p>只读审计：固定10个宏请求代表窗＋6个人审语义案例。未训练、未改标签或掩膜。青色为实际暗线支持，不是导丝分割；速度不是沿管推进。</p>'
    page += ''.join(sections) + '</html>'
    (args.out / "index.html").write_text(page, encoding="utf-8")
    (args.out / "data-manifest.md").write_text(
        "# 真实数据清单\n\n原图：collected_data；精确路径/时间见selection.json。"
        "固定175窗监督来源与45窗修订快照见protocol.json；本次只抽查16窗，不迁移标签。\n\n"
        "PNG是原图ROI＋原pair_motion返回的真实支持/残余速度；热图只显示中间相邻帧对，"
        "颜色封顶2 ROI px/s不改变数值。case_metrics为整窗统计。没有伪造曲线、语义分割或尖端坐标。\n",
        encoding="utf-8")
    print(report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", type=Path, default=Path("simulation_output/real10_elite_local_response_targets_v1"))
    p.add_argument("--labels", type=Path, default=Path("simulation_output/real10_pre_action_label_retest_v1"))
    p.add_argument("--windows", type=Path, default=Path("simulation_output/real10_event_windows_v1"))
    p.add_argument("--raw-root", type=Path, default=Path("collected_data"))
    p.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    p.add_argument("--out", type=Path, default=Path("simulation_output/real10_local_supervision_semantic_audit_v1"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
