from elite import EC
import time
RUN_INDEX = 1
RUN_PATH = f"/home/xwj/桌面/project_2026/path/path{RUN_INDEX}_pose.txt"
last_3_dimension = [3.0661665148309702, 0.03647610536354759, 0.06842115274422467]
def get_path_points(run_path=RUN_PATH):
    with open(run_path, "r") as f:
        points = [[float(val) for val in line.split()] for line in f if line.strip()]
    return [[point[0], point[1], point[2], last_3_dimension[0], last_3_dimension[1], last_3_dimension[2]] for point in points]

RAW_RUN_PATH = f"/home/xwj/桌面/project_2026/path/path{RUN_INDEX}_pose1.txt"
def get_raw_path_points(run_path=RAW_RUN_PATH):
    with open(run_path, "r") as f:
        points = [[float(val) for val in line.split()] for line in f if line.strip()]
    return points

RIGHT_PATH = "/home/xwj/桌面/project_2026/path/path2_pose.txt"
RAW_RIGHT_PATH = "/home/xwj/桌面/project_2026/path/path2_pose1.txt"
def get_right_path_points(run_path=RIGHT_PATH):
    with open(run_path, "r") as f:
        points = [[float(val) for val in line.split()] for line in f if line.strip()]
    return [[point[0], point[1], point[2], last_3_dimension[0], last_3_dimension[1], last_3_dimension[2]] for point in points]

def get_raw_right_path_points(run_path=RAW_RIGHT_PATH):
    with open(run_path, "r") as f:
        points = [[float(val) for val in line.split()] for line in f if line.strip()]
    return points

if __name__ == '__main__':
    points = get_path_points()
    #points = get_raw_path_points(run_path=RAW_RUN_PATH)
    ec = EC(ip="192.168.137.200", auto_connect=True)
    elirobot_index = 0
    pre_time = time.time()
    while elirobot_index < len(points):
        target_pose = points[elirobot_index]
        target = ec.get_inverse_kinematic(pose=target_pose)
        ec.move_joint(target_joint=target, speed=20)
        print(f'Elirobot moved to point {elirobot_index + 1}/{len(points)}')
        elirobot_index += 1
        print('time: ',time.time()-pre_time)
        pre_time = time.time()