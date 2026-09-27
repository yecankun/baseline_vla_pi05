import threading
import os
import time

import cv2
from elite import EC
import keyboard

from utils.robot.utils_elirobot import get_path_points, get_raw_path_points, get_raw_right_path_points, \
    get_right_path_points

stop_event = threading.Event()
points = get_right_path_points()
ec = EC(ip="192.168.137.200", auto_connect=True)
elirobot_index = 0
def on_press_callback(event):
    global elirobot_index
    if event.name == 'esc':
        print('Data collection stopped by user.')
        stop_event.set()
    elif event.name == 'a':
        elirobot_index = max(0, elirobot_index - 1)
        target = points[elirobot_index]
        print(f'Elirobot moving to previous point {elirobot_index + 1}/{len(points)}: {target}')
        ec.move_joint(target_joint=ec.get_inverse_kinematic(pose=target), speed=20)
    elif event.name == 'd':
        elirobot_index = min(len(points) - 1, elirobot_index + 1)
        target = points[elirobot_index]
        print(f'Elirobot moving to next point {elirobot_index + 1}/{len(points)}: {target}')
        ec.move_joint(target_joint=ec.get_inverse_kinematic(pose=target), speed=20)


def keyboard_listener():
    keyboard.on_press(on_press_callback)
    keyboard.wait('esc')

if __name__ == '__main__':
    #ec.move_joint(target_joint=ec.get_inverse_kinematic(pose=points[0]), speed=20)
    keyboard_listener()
    """
    sudo PYTHONPATH=/home/xwj/桌面/project_2026:/home/xwj/anaconda3/envs/mygenesis/lib/python3.11/site-packages \
         /home/xwj/anaconda3/envs/mygenesis/bin/python3.11 \
         path/path_test.py
    """