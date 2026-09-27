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


def draw(points):
    """Draw a 3D scatter + line plot from an iterable of points.

    - points: iterable of points where each point can be a (x,y,z) sequence,
      or a string representation like "[x,y,z]" or "x,y,z".
    - Projects to 3D and draws scatter + connecting line in the given order.
    - If no DISPLAY, saves to 'path_3d_plot.png' next to this script.
    """
    pts_parsed = []
    for p in points:
        parsed = _parse_point(p)
        if parsed is None:
            continue
        pts_parsed.append(parsed)

    if len(pts_parsed) == 0:
        print('No valid points to draw.')
        return

    pts = np.array(pts_parsed, dtype=float)
    if pts.shape[1] < 3:
        # ensure 3 columns
        padded = np.zeros((pts.shape[0], 3), dtype=float)
        padded[:, :pts.shape[1]] = pts
        pts = padded

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    print(x, y, z)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点，使用较大的点以便清晰可见
    sc = ax.scatter(x, y, z, color='red', s=60, depthshade=True, label='路径点')

    # 连接所有点形成连续路径
    ax.plot(x, y, z, color='blue', linewidth=2, alpha=0.7, label='路径线')

    # 如果点数较多，还可以添加起点和终点的特殊标记
    if len(x) > 1:
        # 起点用绿色标记
        ax.scatter([x[0]], [y[0]], [z[0]], color='green', s=100, depthshade=False,
                   label='起点', edgecolors='black', linewidth=0.5)
        # 终点用紫色标记
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color='purple', s=100, depthshade=False,
                   label='终点', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # call 3D-only methods via getattr to avoid static analyzer warnings
    fn_set_zlabel = getattr(ax, 'set_zlabel', None)
    if callable(fn_set_zlabel):
        fn_set_zlabel('Z')
    ax.set_title('左分支-示教轨迹')

    # 添加图例
    ax.legend()

    # 设置相等的坐标轴比例以获得更好的视觉效果
    try:
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
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
    points = get_path_points()
    points2 =get_path_points()
    # make sure it's a list
    try:
        points = list(points)
    except Exception:
        points = [points]

    # keep behavior similar to original (use first 20 if many)
    if len(points) >= 20:
        pts_to_draw = points[:20]
    else:
        pts_to_draw = points

    draw(pts_to_draw)
