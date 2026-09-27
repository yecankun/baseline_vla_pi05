"""Chinese image-only evidence sheet, not a robustness result plot."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.root / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Only render a completed paired stage")
    font = ImageFont.truetype(str(args.font), 18)
    title = ImageFont.truetype(str(args.font), 23)
    canvas = Image.new("RGB", (850, 770), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "配对低对比度输入检查：task 0 / init 0", font=title, fill="#18334e")
    draw.text((24, 55), "同一初始观测，只改变两路图像；状态与任务不变", font=font, fill="#40566c")
    for col, condition in enumerate(("clean", "low_contrast")):
        x = 125 + col * 355
        label = "原始输入（clean）" if col == 0 else "低对比度（系数 0.45）"
        draw.text((x, 97), label, font=font, fill="#18334e")
        for row, view in enumerate(("image", "image2")):
            y = 136 + row * 288
            if col == 0:
                draw.text((24, y + 115), "主视角" if row == 0 else "腕部视角", font=font, fill="#18334e")
            with Image.open(args.root / f"task00_{condition}" / f"first_{view}.png") as frame:
                if frame.size != (256, 256):
                    raise ValueError("Unexpected recorded dimensions")
                canvas.paste(frame.convert("RGB"), (x, y))
    draw.text((24, 713), "图像为首个动作前实际提供给预处理器的原始像素，保留原方向。", font=font, fill="#40566c")
    draw.text((24, 741), "仅用于输入检查，不是完整鲁棒性结论，也不代表真实血管氧化分布。", font=font, fill="#40566c")
    path = args.root / "paired_input_preview_zh.png"
    canvas.save(path)
    print(path)


if __name__ == "__main__":
    main()
