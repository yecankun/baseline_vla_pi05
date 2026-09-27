from elite import EC
from utils.robot.utils_elirobot import get_raw_path_points, get_path_points, get_raw_right_path_points, \
    get_right_path_points

RUN_INDEX = 2
RAW_RUN_PATH = f"/home/xwj/桌面/project_2026/path/path{RUN_INDEX}_pose1.txt"
def reset():
    ec = EC(ip="192.168.5.66", auto_connect=True)
    points = get_right_path_points()
    target_joint = ec.get_inverse_kinematic(pose=points[0])
    ec.move_joint(target_joint=target_joint, speed=20)

if __name__ == "__main__":
    reset()