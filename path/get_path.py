import numpy as np
from scipy.interpolate import splprep, splev

from utils.robot.utils_elirobot import get_raw_path_points

RUN_INDEX = 2
RAW_RUN_PATH = f"/home/xwj/桌面/project_2026/path/path{RUN_INDEX}_pose1.txt"
raw_data = get_raw_path_points(run_path=RAW_RUN_PATH)

# 提取前3列作为三维坐标
points = np.array(raw_data)[:, :3]  # shape: (7, 3)

# 使用 splprep 进行参数样条插值（s=0 表示插值，曲线通过所有点）
# quiet=1 避免警告（如果点太接近可能有警告，但这里没问题）
tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0, k=3)  # k=3 表示三次样条

# 生成20个均匀参数值（从0到1）
u_new = np.linspace(0, 1, 20)

# 计算插值点
interpolated_points = np.column_stack(splev(u_new, tck))

# 打印结果
print("插值得到的20个三维点：")
for i, pt in enumerate(interpolated_points):
    print(f"P{i:02d}: [{pt[0]:.6f}, {pt[1]:.6f}, {pt[2]:.6f}]")

# 可选：保存为文件
np.savetxt(f"path{RUN_INDEX}_pose.txt", interpolated_points, fmt="%.6f")