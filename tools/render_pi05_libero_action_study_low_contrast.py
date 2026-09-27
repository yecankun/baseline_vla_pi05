"""Chinese contact sheets of hash-bound clean/degraded validation previews.

Presentation only: no model, source dataset edits, new corruption, or scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(features, report_sha, output, font_path):
    report_path = features / "report.json"
    if sha(report_path) != report_sha:
        raise ValueError("feature report SHA differs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "passed_validation_only_frozen_clean_replay_and_low_contrast":
        raise ValueError("completed paired feature report required")
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(str(font_path), 20)
    title_font = ImageFont.truetype(str(font_path), 29)
    episodes = ((1530, 126), (1476, 125), (1458, 130), (1566, 143))
    columns = (("clean", False, "首帧 · 干净"), ("low_contrast", False, "首帧 · 低对比度"),
               ("clean", True, "末帧 · 干净"), ("low_contrast", True, "末帧 · 低对比度"))
    verified, results = {}, []
    for kind, label in (("source", "原始图像"), ("processed", "原生预处理后图像")):
        sheet = Image.new("RGB", (1216, 2370), "#f4f6f8")
        draw = ImageDraw.Draw(sheet)
        draw.text((24, 16), f"固定验证集：干净 / 低对比度配对 · {label}", fill="#172c43", font=title_font)
        draw.text((24, 57), "强度 2，系数 0.45；只改变历史图像输入，预测目标保持干净。", fill="#31475c", font=font)
        for col, (_, _, caption) in enumerate(columns):
            draw.text((146+266*col, 99), caption, fill="#172c43", font=font)
        for eindex, (eid, count) in enumerate(episodes):
            for view in range(2):
                y = 136 + (2*eindex+view)*274
                draw.text((12, y+90), f"轨迹 {eid}", fill="#172c43", font=font)
                draw.text((12, y+122), f"视角 {view+1}", fill="#31475c", font=font)
                for col, (condition, last, _) in enumerate(columns):
                    frame = count-1 if last else 0
                    name = f"{condition}/episode{eid}_frame{frame:04d}_view{view}_{kind}.png"
                    path = features / name
                    digest = report["output_sha256"][name]
                    if sha(path) != digest:
                        raise ValueError(f"preview SHA differs: {name}")
                    verified[name] = digest
                    with Image.open(path) as image:
                        image.load()
                        expected = (256, 256) if kind == "source" else (224, 224)
                        if image.mode != "RGB" or image.size != expected:
                            raise ValueError("unexpected native preview shape/mode")
                        # Preserve original pixels and aspect; center224 within256.
                        sheet.paste(image, (138+266*col+(256-image.width)//2, y+(256-image.height)//2))
        draw.text((24, 2338), "仅为受控可见度扰动；不代表真实血管氧化，也不构成用户视觉验收。", fill="#31475c", font=font)
        path = output / f"{kind}_paired_review.png"
        sheet.save(path)
        results.append({"path": path.name, "sha256": sha(path), "size": list(sheet.size)})
    evidence = {"schema": "libero_low_contrast_visual_review_v1", "feature_report_sha256": report_sha,
                "font": str(font_path), "previews_verified": verified, "outputs": results,
                "visual_status": "not_viewed", "simulation_steps": 0, "model_forward_calls": 0}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--feature-report-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--font", required=True, type=Path)
    args = parser.parse_args()
    result = render(args.features_dir, args.feature_report_sha256, args.out, args.font)
    print(json.dumps({"images": len(result["outputs"]), "verified_previews": len(result["previews_verified"])}))


if __name__ == "__main__":
    main()
