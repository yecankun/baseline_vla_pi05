from __future__ import annotations

import re
from pathlib import Path


def label_png_folder(img_dir: str, label_file: str, overwrite: bool = True) -> None:
    """
    交互式图片标注：
    - img_dir: 图片文件夹路径，内部图片命名为 1.png, 2.png, ... x.png
    - label_file: 标签文件路径（如 /xxx/piper.txt）
    - overwrite: True=覆盖写入(默认)；False=追加写入

    标注方式：
    - 若安装了 opencv-python：显示窗口后直接按键 0 或 1 写入；ESC/q 退出
    - 否则：用 matplotlib 显示图，同时在终端输入 0/1（需回车）；q 退出
    """
    img_dir_path = Path(img_dir)
    if not img_dir_path.is_dir():
        raise FileNotFoundError(f"图片文件夹不存在：{img_dir}")

    # 只收集形如 "数字.png" 的文件，并按数字排序
    pairs = []
    for p in img_dir_path.iterdir():
        if p.is_file() and p.suffix.lower() == ".png":
            m = re.fullmatch(r"(\d+)\.png", p.name, flags=re.IGNORECASE)
            if m:
                pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    img_paths = [p for _, p in pairs]

    print(f"图片数量：{len(img_paths)}")
    if not img_paths:
        print("未找到符合命名规则的图片（例如 1.png, 2.png ...）。")
        return

    label_path = Path(label_file)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"

    # 尝试使用 OpenCV（最符合“按键输入”的体验）
    try:
        import cv2  # type: ignore
        use_cv = True
    except Exception:
        use_cv = False

    with open(label_path, mode, encoding="utf-8") as f:
        if use_cv:
            import cv2  # type: ignore

            window_name = "Labeler (press 0/1, ESC/q to quit)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            total = len(img_paths)
            for i, p in enumerate(img_paths, start=1):
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[{i}/{total}] 读取失败，跳过：{p.name}")
                    continue

                cv2.imshow(window_name, img)
                print(f"[{i}/{total}] {p.name}  -> 按 0/1 标注（ESC 或 q 退出）")

                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key in (ord("0"), ord("1")):
                        f.write(chr(key) + "\n")
                        f.flush()
                        break
                    if key in (27, ord("q"), ord("Q")):  # ESC or q
                        cv2.destroyAllWindows()
                        print("已退出，当前已写入的标签保留在文件中。")
                        return

            cv2.destroyAllWindows()
            print(f"完成：已写入 {total} 行标签到 {label_path}")
        else:
            # 降级：matplotlib + input()
            from PIL import Image
            import matplotlib.pyplot as plt

            total = len(img_paths)
            for i, p in enumerate(img_paths, start=1):
                try:
                    img = Image.open(p)
                except Exception as e:
                    print(f"[{i}/{total}] 读取失败，跳过：{p.name} ({e})")
                    continue

                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"[{i}/{total}] {p.name} (input 0/1 in terminal, q to quit)")
                plt.show(block=False)
                plt.pause(0.001)

                while True:
                    s = input(f"[{i}/{total}] 请输入 0 或 1（q 退出）：").strip()
                    if s in ("0", "1"):
                        f.write(s + "\n")
                        f.flush()
                        break
                    if s.lower() == "q":
                        plt.close(fig)
                        print("已退出，当前已写入的标签保留在文件中。")
                        return
                    print("无效输入，请输入 0 或 1。")

                plt.close(fig)

            print(f"完成：已写入 {total} 行标签到 {label_path}")

if __name__ == "__main__":
    PATH_INDEX = 14
    img_dir = f"D:/datasource/piper_data/branch1/image1/path{PATH_INDEX}"
    label_file = f"D:/datasource/piper_data/branch1/piper/label{PATH_INDEX}.txt"
    label_png_folder(img_dir, label_file)

    #[137, 173, 138, 140, 191, 223, 140, 80, 206, 190, 0, 0, 0, 169]
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 201, 233, 138, ]