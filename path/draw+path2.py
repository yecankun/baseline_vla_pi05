from utils.robot.utils_elirobot import get_path_points, get_right_path_points
import os
import ast
import numpy as np
import matplotlib

# use non-interactive backend if no DISPLAY
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _parse_point(p):
    """Normalize a single point into a list of floats [x,y,z].
    Accepts list/tuple/np.ndarray or string like "[x,y,z]" or "x,y,z".
    If only x,y provided, z is set to 0. Returns None if cannot parse.
    """
    if p is None:
        return None
    # already sequence
    if isinstance(p, (list, tuple, np.ndarray)):
        try:
            vals = [float(x) for x in p]
        except Exception:
            return None
    elif isinstance(p, str):
        s = p.strip()
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                vals = [float(x) for x in val]
            else:
                # if literal_eval returns a single number
                vals = [float(val)]
        except Exception:
            # fallback: remove brackets/parens and split by common separators
            s2 = s.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ')
            s2 = s2.replace(',', ' ').replace(';', ' ')
            parts = s2.split()
            try:
                vals = [float(x) for x in parts]
            except Exception:
                return None
    else:
        return None

    if len(vals) == 0:
        return None
    if len(vals) == 1:
        # single value -> treat as x, set y=z=0
        vals = [vals[0], 0.0, 0.0]
    elif len(vals) == 2:
        vals = [vals[0], vals[1], 0.0]
    else:
        # take first three
        vals = vals[:3]
    return vals


def _parse_points_list(points_list):
    """Parse a list of points using _parse_point."""
    pts_parsed = []
    for p in points_list:
        parsed = _parse_point(p)
        if parsed is None:
            continue
        pts_parsed.append(parsed)
    return pts_parsed


def draw_two_trajectories(points1, points2):
    """Draw two 3D trajectories on the same plot with different colors.

    - points1: iterable of points for trajectory 1
    - points2: iterable of points for trajectory 2
    - Each point can be a (x,y,z) sequence, or a string representation like "[x,y,z]" or "x,y,z".
    - Projects to 3D and draws scatter + connecting line in the given order for both trajectories.
    - If no DISPLAY, saves to 'path_3d_plot.png' next to this script.
    """
    pts1_parsed = _parse_points_list(points1)
    pts2_parsed = _parse_points_list(points2)

    if len(pts1_parsed) == 0:
        print('No valid points for trajectory 1 to draw.')
        return
    if len(pts2_parsed) == 0:
        print('No valid points for trajectory 2 to draw.')
        return

    # Convert to numpy arrays
    pts1 = np.array(pts1_parsed, dtype=float)
    pts2 = np.array(pts2_parsed, dtype=float)

    # Ensure 3 columns
    if pts1.shape[1] < 3:
        padded = np.zeros((pts1.shape[0], 3), dtype=float)
        padded[:, :pts1.shape[1]] = pts1
        pts1 = padded
    if pts2.shape[1] < 3:
        padded = np.zeros((pts2.shape[0], 3), dtype=float)
        padded[:, :pts2.shape[1]] = pts2
        pts2 = padded

    x1, y1, z1 = pts1[:, 0], pts1[:, 1], pts1[:, 2]
    x2, y2, z2 = pts2[:, 0], pts2[:, 1], pts2[:, 2]

    print("轨迹1 X坐标:", x1)
    print("轨迹1 Y坐标:", y1)
    print("轨迹1 Z坐标:", z1)
    print("轨迹2 X坐标:", x2)
    print("轨迹2 Y坐标:", y2)
    print("轨迹2 Z坐标:", z2)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一个轨迹
    ax.scatter(x1, y1, z1, color='red', s=60, depthshade=True)
    ax.plot(x1, y1, z1, color='red', linewidth=2, alpha=0.7, label='左分支路径')

    # 绘制第二个轨迹
    ax.scatter(x2, y2, z2, color='blue', s=60, depthshade=True)
    ax.plot(x2, y2, z2, color='blue', linewidth=2, alpha=0.7, label='右分支路径')

    # 为两个轨迹的起点和终点添加特殊标记
    if len(x1) > 1:
        # 轨迹1起点用绿色标记
        ax.scatter([x1[0]], [y1[0]], [z1[0]], color='green', s=100, depthshade=False,
                   label='起点', edgecolors='black', linewidth=0.5)
        # 轨迹1终点用紫色标记
        ax.scatter([x1[-1]], [y1[-1]], [z1[-1]], color='purple', s=100, depthshade=False,
                   label='左分支终点', edgecolors='black', linewidth=0.5)

    if len(x2) > 1:
        # 轨迹2起点用橙色标记
        # ax.scatter([x2[0]], [y2[0]], [z2[0]], color='orange', s=100, depthshade=False,
        #            label='轨迹2-起点', edgecolors='black', linewidth=0.5)
        # 轨迹2终点用青色标记
        ax.scatter([x2[-1]], [y2[-1]], [z2[-1]], color='cyan', s=100, depthshade=False,
                   label='右分支终点', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # call 3D-only methods via getattr to avoid static analyzer warnings
    fn_set_zlabel = getattr(ax, 'set_zlabel', None)
    if callable(fn_set_zlabel):
        fn_set_zlabel('Z')
    ax.set_title('示教轨迹')

    # 添加图例
    ax.legend()

    # 计算整体范围以设置合适的坐标轴比例
    all_x = np.concatenate([x1, x2])
    all_y = np.concatenate([y1, y2])
    all_z = np.concatenate([z1, z2])

    try:
        max_range = np.array([all_x.max() - all_x.min(), all_y.max() - all_y.min(), all_z.max() - all_z.min()]).max()
        mid_x = (all_x.max() + all_x.min()) * 0.5
        mid_y = (all_y.max() + all_y.min()) * 0.5
        mid_z = (all_z.max() + all_z.min()) * 0.5
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        fn_set_zlim = getattr(ax, 'set_zlim', None)
        if callable(fn_set_zlim):
            fn_set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    except Exception:
        pass

    if os.environ.get('DISPLAY'):
        plt.show()
    else:
        out = os.path.join(os.path.dirname(__file__), 'path_3d_plot.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved 3D plot to {out}')


if __name__ == '__main__':
    points1 = get_path_points()
    points2 = get_right_path_points()

    # make sure they're lists
    try:
        points1 = list(points1)
    except Exception:
        points1 = [points1]

    try:
        points2 = list(points2)
    except Exception:
        points2 = [points2]

    # keep behavior similar to original (use first 20 if many)
    if len(points1) >= 20:
        pts1_to_draw = points1[:20]
    else:
        pts1_to_draw = points1

    if len(points2) >= 20:
        pts2_to_draw = points2[:20]
    else:
        pts2_to_draw = points2

    draw_two_trajectories(pts1_to_draw, pts2_to_draw)
