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

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # scatter with a single uniform color (do not color by z)
    sc = ax.scatter(x, y, z, color='tab:blue', s=40, depthshade=True)
    # connect points in order with a contrasting color
    ax.plot(x, y, z, color='orange', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # call 3D-only methods via getattr to avoid static analyzer warnings
    fn_set_zlabel = getattr(ax, 'set_zlabel', None)
    if callable(fn_set_zlabel):
        fn_set_zlabel('Z')
    ax.set_title('3D Path')

    # set equal aspect ratio for 3D plot (approximate)
    try:
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        fn_set_zlim = getattr(ax, 'set_zlim', None)
        if callable(fn_set_zlim):
            fn_set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    except Exception:
        pass

    # no colorbar (removed as requested)

    if os.environ.get('DISPLAY'):
        plt.show()
    else:
        out = os.path.join(os.path.dirname(__file__), 'path_3d_plot.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved 3D plot to {out}')


if __name__ == '__main__':
    points = get_right_path_points()
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
