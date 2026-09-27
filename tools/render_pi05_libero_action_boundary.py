"""Render the short probe's recorded observations and boundary counts only."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / "report.json").read_text())
    if report["status"] != "passed" or report["benchmark_score_claim_allowed"]:
        raise ValueError("Expected passed diagnostic-only probe")
    font = ImageFont.truetype(str(args.font), 18)
    title = ImageFont.truetype(str(args.font), 25)
    canvas = Image.new("RGB", (872, 1060), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "PI0.5 / LIBERO：短窗口动作边界检查", font=title, fill="#18334e")
    draw.text((24, 58), f"task 0 · seed 1000 · init 0 · {report['steps_executed']} 步 · 非成功率评测", font=font, fill="#40566c")
    entries = report["visual_artifacts"]
    positions = sorted(set(e["step"] for e in entries))
    for e in entries:
        col = positions.index(e["step"])
        row = ["image", "image2"].index(e["view"])
        x, y = 24 + 282 * col, 96 + 294 * row
        with Image.open(args.root / e["path"]) as source:
            if source.size != (256, 256):
                raise ValueError("Unexpected recorded camera size")
            canvas.paste(source.convert("RGB"), (x, y + 27))
        label = "主视角" if row == 0 else "腕部视角"
        draw.text((x, y), f"第 {e['step']} 步前 · {label}", font=font, fill="#18334e")
    y = 700
    draw.text((24, y), "环境输入逐维范围与超限次数（原始值，未额外裁剪）", font=font, fill="#18334e")
    y += 33
    draw.text((30, y), "动作维度", font=font, fill="#40566c")
    draw.text((255, y), "最小值", font=font, fill="#40566c")
    draw.text((465, y), "最大值", font=font, fill="#40566c")
    draw.text((670, y), "超出 [-1,1]", font=font, fill="#40566c")
    labels = ["平移 X", "平移 Y", "平移 Z", "旋转 X", "旋转 Y", "旋转 Z", "夹爪"]
    for label, value in zip(labels, report["analysis"]["per_dimension"], strict=True):
        y += 29
        color = "#a43828" if value["out_of_bounds_steps"] else "#18334e"
        for x, text in ((30, label), (255, f"{value['min']:.6f}"), (465, f"{value['max']:.6f}"),
                        (705, str(value["out_of_bounds_steps"]))):
            draw.text((x, y), text, font=font, fill=color)
    draw.text((24, 997), "仅为当前短窗口机制证据；不回填 B4b 逐维统计，不作失败归因。", font=font, fill="#40566c")
    draw.text((24, 1025), "图像为动作前的双路原始观测；不代表终态，尚未获用户视觉验收。", font=font, fill="#40566c")
    destination = args.root / "boundary_summary_zh.png"
    canvas.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
