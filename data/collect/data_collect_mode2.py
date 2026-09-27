# 采集数据模式2： 大机械臂匀速运动，人工控制小机械臂跟踪导丝
import threading
import os
import time

import cv2
from elite import EC
import keyboard

from utils.camera.utils_camera import Camera
from utils.robot.utils_elirobot import get_path_points
from utils.robot.utils_piper import PiperRobot

BRANCH_INDEX = 2
PATH_INDEX = 8
RUN_INDEX = 2
T = 0.2
PIPER_LEN = 13

stop_event = threading.Event()
# save_path1 为固定的摄像头的图片存储路径
root_path = '/home/xwj/data'
save_path1 = os.path.join(root_path + f'/branch{BRANCH_INDEX}/image1/path{PATH_INDEX}')
# save_path2 为机械臂上的摄像头
#save_path2 = os.path.join(root_path + f'/branch1/image2/path{PATH_INDEX}')
# 存储关节角轨迹的文件路径
joint_path = root_path + f'/branch{BRANCH_INDEX}/path'
pose_file_name = joint_path + f'/pose{PATH_INDEX}.txt'
camera1_pos_dir = root_path + f'/branch{BRANCH_INDEX}/camera1'
# 存储侧视导丝头位置的文件路径: x,y, piper_step
camera1_pos_file = camera1_pos_dir + f'/pos{PATH_INDEX}.txt'
# 机械臂运行的点位
RUN_PATH = f"/home/xwj/桌面/project_2026/path/path{RUN_INDEX}_pose.txt"

points = get_path_points(run_path=RUN_PATH)

ec = EC(ip="192.168.137.200", auto_connect=True)
piper_current_step = 0
saved_count = 1
RECORD = True

def write_in(file_name: str, data):
    with open(file_name, encoding="utf-8", mode="a") as file:
        file.write(str(data) + '\n')

def on_press_callback(event):
    global piper_current_step
    piper = PiperRobot()
    if event.name == 'esc':
        piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
        print('Data collection stopped by user.')
        stop_event.set()
    elif event.name == 'a':
        piper_current_step -= 1
        print(f'piper moving to previous point {piper_current_step}/{PIPER_LEN}')
        piper.step_backward()
    elif event.name == 'd':
        piper_current_step += 1
        print(f'piper moving to next point {piper_current_step}/{PIPER_LEN}')
        piper.step_forward()
    if piper_current_step == PIPER_LEN:
        piper.piper.GripperCtrl(20 * 1000, 1000, 0x01, 0)
        stop_event.set()

def keyboard_listener():
    keyboard.on_press(on_press_callback)
    keyboard.wait('esc')

def elirobot_run():
    elirobot_index = 1
    time.sleep(1)
    while elirobot_index < len(points):
        if stop_event.is_set():
            break
        target_pose = points[elirobot_index]
        target = ec.get_inverse_kinematic(pose=target_pose)
        ec.move_joint(target_joint=target, speed=20)
        print(f'Elirobot moved to point {elirobot_index + 1}/{len(points)}')
        elirobot_index += 1
        time.sleep(1)


def record_main():
    global saved_count
    camera0 = Camera(1)
    camera0.get_image_depth()
    print('camera 侧视 initialized')
    time.sleep(1)
    intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
    pre_time = time.time()
    while not stop_event.is_set():
        pos, result_contours = camera0.predict_pos()
        pose = ec.current_pose
        intr, depth_intrin, color_image, depth_image, depth_frame = camera0.get_image_depth()
        if RECORD:
            cv2.imwrite(os.path.join((save_path1), "{}.png".format(saved_count)), color_image)
            write_in(pose_file_name, pose)
            write_in(camera1_pos_file, f"{camera0.position[0]} {camera0.position[1]} {piper_current_step}")
            saved_count += 1
        time.sleep(T)
        print(f'waste time: {time.time() - pre_time}') #0.15s
        pre_time = time.time()
    print('Recording thread stopped.')

def write_data(camera):
    global saved_count
    pose = ec.current_pose
    pos1, _ = camera.predict_pos()
    intr, depth_intrin, color_image, depth_image, depth_frame = camera.get_image_depth()
    cv2.imwrite(os.path.join(save_path1, "{}.png".format(saved_count)), color_image)
    write_in(pose_file_name, pose)
    write_in(camera1_pos_file, f"{camera.position[0]} {camera.position[1]} {piper_current_step}")
    saved_count += 1

if __name__ == "__main__":
    if RECORD:
        dir_list = [save_path1, joint_path, camera1_pos_dir]
        for dir_path in dir_list:
            if not os.path.exists(dir_path):
                print(f'文件夹{dir_path}不存在，创建文件夹')
                os.makedirs(dir_path)
        file_list = [pose_file_name, camera1_pos_file]
        for file_name in file_list:
            print(f'文件{file_name}不存在，创建文件')
            with open(file_name, 'w') as f:
                pass

    t0 = threading.Thread(target=keyboard_listener)
    t1 = threading.Thread(target=record_main)
    t2 = threading.Thread(target=elirobot_run)
    t0.start()
    t1.start()
    t2.start()
    t0.join()
    t1.join()
    t2.join()
    print('Data collection mode 2 completed.')
"""
sudo PYTHONPATH=/home/xwj/桌面/project_2026:/home/xwj/anaconda3/envs/mygenesis/lib/python3.11/site-packages \
     /home/xwj/anaconda3/envs/mygenesis/bin/python3.11 \
     data/collect/data_collect_mode2.py
"""

#
#branch1_old  0-8 mode 1, 9-15 mode2 无副样本， 16 - 22 有负样本
#branch2 path 2，3有问题